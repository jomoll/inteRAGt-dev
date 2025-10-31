"""Integration checks for the lightweight agents talking to live endpoints."""

from __future__ import annotations

import json
import logging
import re
import os
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
from dotenv import load_dotenv
from llama_index.core.base.llms.types import MessageRole

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("src.agent_gemini").setLevel(logging.DEBUG)
logging.getLogger("src.agent_tgi").setLevel(logging.DEBUG)

try:  # pragma: no cover - optional heavy dependencies
    from src.agent_tgi import TGIChatAgent, TGIConfig
    from src.agent_gemini import GeminiChatAgent, GeminiConfig
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Integration tests require optional agent dependencies: {exc}",
        allow_module_level=True,
    )

from src.agent_tools import load_default_tools

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


@pytest.fixture(scope="module")
def default_tools():
    return load_default_tools()


def _rag_scenarios() -> list[Dict[str, Any]]:
    return [
        {
            "label": "retrieve_reports",
            "question": "Does the patient meet the CAR-T eligibility criteria?",
            "patient_context": "",
            "kwargs": {},
            "min_context": 1,
        }
    ]


def _print_chat_history(agent, label: str) -> None:
    history = getattr(agent, "history", None)
    if history is None:
        history = getattr(agent, "_chat_history", [])
    print(f"[Integration:{label}] Conversation transcript:")
    for idx, message in enumerate(history, start=1):
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        additional = getattr(message, "additional_kwargs", None)
        if role == MessageRole.TOOL:
            display = _summarise_tool_message(content, additional)
            print(f"  [{idx}] role={role} content={display!r}")
        else:
            print(f"  [{idx}] role={role} content={content!r}")


def _summarise_tool_message(content: str | None, additional: dict | None) -> str:
    summary = ""
    try:
        parsed = json.loads(content or "{}")
    except Exception:
        parsed = {}
    response_text = ""
    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            response_text = response
        elif isinstance(response, dict):
            text = response.get("response")
            if isinstance(text, str):
                response_text = text
        context_nodes = parsed.get("context_nodes")
        if isinstance(context_nodes, list):
            summary += f" [{len(context_nodes)} nodes]"

    response_text = re.sub(r"\s+", " ", response_text).strip()
    if response_text:
        summary = response_text[:120] + ("..." if len(response_text) > 120 else "") + summary
    elif content:
        summary = re.sub(r"\s+", " ", content).strip()
        summary = summary[:120] + ("..." if len(summary) > 120 else "")

    if additional and isinstance(additional, dict):
        name = additional.get("name")
        args = additional.get("arguments")
        snippet = f"{name or 'tool'}"
        if isinstance(args, dict):
            filters = {k: v for k, v in args.items() if v not in (None, "") and k != "top_k"}
            if filters:
                snippet += f" {filters}"
        summary = f"{snippet}: {summary}"

    return summary


@pytest.mark.integration
def test_tgi_agent_live(default_tools):
    endpoint = os.getenv("TGI_ENDPOINT")
    if not endpoint:
        pytest.skip("Set TGI_ENDPOINT to run live TGI integration test.")

    api_key = os.getenv("TGI_API_KEY")
    model_name = os.getenv("TGI_MODEL", "casperhansen/llama-3.3-70b-instruct-awq")

    debug_httpx = os.getenv("TGI_DEBUG_HTTPX", "0") == "1"

    http_client = None
    http_client_created = False
    if debug_httpx:
        httpx_logger = logging.getLogger("httpx")
        if not any(isinstance(handler, logging.StreamHandler) for handler in httpx_logger.handlers):
            httpx_logger.addHandler(logging.StreamHandler())
        httpx_logger.setLevel(logging.DEBUG)

        limits = httpx.Limits(
            max_connections=int(os.getenv("TGI_HTTPX_MAX_CONNECTIONS", "3")),
            max_keepalive_connections=int(os.getenv("TGI_HTTPX_MAX_KEEPALIVE", "1")),
        )
        timeout = httpx.Timeout(
            connect=float(os.getenv("TGI_HTTPX_TIMEOUT_CONNECT", "10.0")),
            read=float(os.getenv("TGI_HTTPX_TIMEOUT_READ", "120.0")),
            write=float(os.getenv("TGI_HTTPX_TIMEOUT_WRITE", "10.0")),
            pool=float(os.getenv("TGI_HTTPX_TIMEOUT_POOL", "10.0")),
        )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        http_client = httpx.Client(
            base_url=endpoint,
            headers=headers,
            timeout=timeout,
            http2=os.getenv("TGI_HTTPX_HTTP2", "0") == "1",
            limits=limits,
        )
        http_client_created = True

    config = TGIConfig(
        endpoint_url=endpoint,
        api_key=api_key,
        model=model_name,
    )
    agent = TGIChatAgent(
        config=config,
        tools=default_tools,
        http_client=http_client,
    )

    try:
        for scenario in _rag_scenarios():
            agent.reset()
            reply = agent.answer_with_rag(
                scenario["question"],
                patient_context=scenario["patient_context"],
                **scenario["kwargs"],
            )
            context_nodes = reply.additional_kwargs.get("context_nodes", [])
            assert len(context_nodes) >= scenario["min_context"], "Expected context nodes from RAG tool"
            assert reply.content and reply.content.strip(), "Answer should not be empty"
            _print_chat_history(agent, f"TGI[{scenario['label']}]")
    finally:
        if http_client_created and http_client is not None:
            http_client.close()


@pytest.mark.integration
def test_gemini_agent_live(default_tools):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Set GEMINI_API_KEY to run live Gemini integration test.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    config = GeminiConfig(api_key=api_key, model=model_name)
    agent = GeminiChatAgent(config=config, tools=default_tools)

    for scenario in _rag_scenarios():
        agent.reset()
        reply = agent.answer_with_rag(
            scenario["question"],
            patient_context=scenario["patient_context"],
            **scenario["kwargs"],
        )
        context_nodes = reply.additional_kwargs.get("context_nodes", [])
        assert len(context_nodes) >= scenario["min_context"], "Expected context nodes from RAG tool"
        assert reply.content and reply.content.strip(), "Answer should not be empty"
        _print_chat_history(agent, f"Gemini[{scenario['label']}]")
