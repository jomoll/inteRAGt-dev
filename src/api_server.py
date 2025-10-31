from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

from src.agent_gemini import GeminiChatAgent, GeminiConfig
from src.agent_tools import LabValuesTool, ReportsRAGTool

logger = logging.getLogger(__name__)


class AgentRunStore:
    def __init__(self, max_runs: int = 50) -> None:
        self._max_runs = max_runs
        self._runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def _prune_locked(self) -> None:
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)

    def start_run(self, run_id: str) -> None:
        with self._lock:
            self._runs[run_id] = {
                "events": [],
                "created_at": time.time(),
            }
            self._prune_locked()

    def add_event(self, run_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            if run_id in self._runs:
                self._runs[run_id]["events"].append(event)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            return {
                "run_id": run_id,
                "events": list(run["events"]),
                "created_at": run["created_at"],
            }

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"run_id": run_id, "created_at": data["created_at"], "event_count": len(data["events"])}
                for run_id, data in reversed(self._runs.items())
            ]


run_store = AgentRunStore()


def _normalise_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": event.get("type", "unknown"),
        "timestamp": float(event.get("timestamp") or time.time()),
        "payload": event.get("payload", {}),
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    patient_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None


class ChatReply(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    run_id: Optional[str] = None
    events: Optional[List[Dict[str, Any]]] = None


class PatientSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    details: Optional[str] = None


PATIENTS: List[PatientSummary] = [
    PatientSummary(
        id="0001005847",
        name="Max Mustermann",
        description="Demo-Fall aus der lokalen Datenbank",
        details="Diagnosis: Multiples Myelom, IgA kappa\n" \
        "Initial diagnosis: 12/2017\n" \
        "Paraprotein at diagnosis: n.a.\n" \
        "Cytogenetics/FISH: del(13q14), del(3#IGH), del(17p)\n" \
        "R-ISS or R2-ISS: III\n" \
        "Initial SLiM CRAB: Calcium: n.a., Creatinine: 1.0 mg/dL, Hb: 14.2 g/dL, CT: n.a., BM-Infiltration: n.a., FLC-Ratio: 19.5" 
    )
]


def build_patient_context(patient_id: Optional[str]) -> str:
    if not patient_id:
        return ""
    patient = next((entry for entry in PATIENTS if entry.id == patient_id), None)
    if not patient:
        return ""
    lines = [
        f"Patienten-ID: {patient.id}",
        f"Name: {patient.name}" if patient.name else "",
        f"Kurzbeschreibung: {patient.description}" if patient.description else "",
        f"Details: {patient.details}" if patient.details else "",
    ]
    return "\n".join([line for line in lines if line])


def create_agent() -> GeminiChatAgent:
    api_key = os.getenv("GEMINI_API_KEY")
    model  = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not api_key:
        print("Set GEMINI_API_KEY to run live Gemini integration test.")
    config = GeminiConfig(api_key, model=model)
    tools = [ReportsRAGTool(), LabValuesTool()]
    return GeminiChatAgent(config=config, tools=tools)


def create_app() -> FastAPI:
    fastapi_app = FastAPI(title="Clinical RAG Assistant API", version="0.1.0")

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://10.184.8.240:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/patients", response_model=List[PatientSummary])
    async def list_patients() -> List[PatientSummary]:
        return PATIENTS

    @fastapi_app.post("/chat", response_model=ChatReply)
    async def chat(request: ChatRequest) -> ChatReply:
        logger.info("Received chat request for patient %s", request.patient_id)
        run_id = str(uuid.uuid4())
        run_store.start_run(run_id)
        events: List[Dict[str, Any]] = []

        def capture(event: Dict[str, Any]) -> None:
            record = _normalise_event(event)
            events.append(record)
            run_store.add_event(run_id, record)

        agent = create_agent()
        patient_context = build_patient_context(request.patient_id)
        try:
            reply = agent.answer_with_rag(
                request.question,
                patient_context=patient_context,
                event_handler=capture,
            )
            metadata = reply.additional_kwargs or {}
            return ChatReply(
                content=reply.content or "",
                metadata=metadata,
                run_id=run_id,
                events=events,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Agent call failed: %s", exc)
            error_event = {
                "type": "run_failed",
                "payload": {"message": str(exc)},
                "timestamp": time.time(),
            }
            capture(error_event)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @fastapi_app.post("/chat/stream")
    async def chat_stream(request: Request, chat_request: ChatRequest) -> StreamingResponse:
        logger.info("Streaming chat request for patient %s", chat_request.patient_id)
        run_id = str(uuid.uuid4())
        run_store.start_run(run_id)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        patient_context = build_patient_context(chat_request.patient_id)

        def capture(event: Dict[str, Any]) -> None:
            record = _normalise_event(event)
            run_store.add_event(run_id, record)
            loop.call_soon_threadsafe(queue.put_nowait, record)

        def worker() -> None:
            agent = create_agent()
            try:
                agent.answer_with_rag(
                    chat_request.question,
                    patient_context=patient_context,
                    event_handler=capture,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Agent call failed during stream: %s", exc)
                capture(
                    {
                        "type": "run_failed",
                        "payload": {"message": str(exc)},
                        "timestamp": time.time(),
                    }
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, worker)

        async def event_generator() -> AsyncIterator[str]:
            try:
                while True:
                    if await request.is_disconnected():
                        logger.info("Client disconnected from stream for run %s", run_id)
                        break
                    event = await queue.get()
                    if event is None:
                        break
                    payload = {"run_id": run_id, **event}
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
            finally:
                # Drain remaining events to avoid pending tasks
                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                logger.info("Stream for run %s completed", run_id)

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    @fastapi_app.get("/monitor/runs/{run_id}")
    async def get_run(run_id: str) -> Dict[str, Any]:
        run = run_store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @fastapi_app.get("/monitor/runs")
    async def list_runs() -> Dict[str, Any]:
        return {"runs": run_store.list_runs()}

    return fastapi_app


app = create_app()
