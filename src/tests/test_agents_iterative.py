import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip(
    "llama_index.llms.openai_like",
    reason="iterative agent tests require the OpenAILike client",
    exc_type=ImportError,
)

from llama_index.core.base.llms.types import ChatMessage, MessageRole  # noqa: E402

from src.agent_tgi import TGIChatAgent, TGIConfig  # noqa: E402
from src.toolkit import BaseTool, ToolOutput, ToolMetadata  # noqa: E402


class DummyTool(BaseTool):
    """Returns predefined context nodes per query."""

    def __init__(self, responses: dict[str, list[dict]]):
        self._metadata = ToolMetadata(
            name="retrieve_reports",
            description="Dummy RAG tool for tests.",
        )
        self.responses = responses
        self.calls: list[str] = []
        self.arguments: list[dict] = []

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return self._metadata

    def __call__(self, *, query: str, **kwargs) -> ToolOutput:  # type: ignore[override]
        self.calls.append(query)
        self.arguments.append({"query": query, **kwargs})
        nodes = self.responses.get(query, [])
        payload = {
            "response": f"Results for {query}",
            "context_nodes": nodes,
        }
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(payload),
            raw_input={"kwargs": {"query": query, **kwargs}},
            raw_output=payload,
        )


class DummyLLM:
    """Placeholder OpenAI-like client; the tests patch agent methods instead."""

    def __init__(self, *_, **__):
        self.responses: list = []

    def chat(self, *_, **__):
        if not self.responses:
            raise AssertionError("No dummy responses queued for DummyLLM.chat")
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def patch_agent_dependencies(monkeypatch):
    from src import agent_tgi

    monkeypatch.setattr(agent_tgi, "OpenAILike", DummyLLM)


def _stub_compose_answer(agent: TGIChatAgent) -> None:
    def _compose(
        self,
        question,
        patient_context,
        context_nodes,
        *,
        report_type=None,
        report_date=None,
        max_context=5,
    ):
        return f"{question} :: {len(context_nodes)} snippet(s)"

    agent._compose_answer = MethodType(_compose, agent)


def make_tool_call(query: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="retrieve_reports",
            arguments=json.dumps({"query": query}),
        ),
    )


def make_response(tool_calls: list[SimpleNamespace], content: str = "") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(message=message, tool_calls=tool_calls)


def test_iterative_rag_adds_follow_up_query():
    responses = {
        "initial query": [],
        "follow-up query": [
            {
                "section_id": "sec-1",
                "section_name": "Summary",
                "snippet": "Important clinical details.",
            }
        ],
    }
    tool = DummyTool(responses)

    agent = TGIChatAgent(
        config=TGIConfig(endpoint_url="http://dummy"),
        tools=[tool],
    )

    _stub_compose_answer(agent)

    agent._llm.responses = [
        make_response([make_tool_call("initial query", "call-1")], content=""),
        make_response([make_tool_call("follow-up query", "call-2")], content=""),
        make_response([], content="Bereit."),
    ]

    reply = agent.answer_with_rag("How is the patient?", patient_context="demo")

    assert tool.calls == ["initial query", "follow-up query"]
    assert reply.additional_kwargs["subqueries"] == ["initial query", "follow-up query"]
    assert reply.additional_kwargs["context_nodes"][0]["section_id"] == "sec-1"
    assert reply.content.endswith("1 snippet(s)")


def test_iterative_rag_respects_round_limit():
    responses = {
        "initial query": [],
        "extra-1": [],
        "extra-2": [],
    }
    tool = DummyTool(responses)

    agent = TGIChatAgent(
        config=TGIConfig(endpoint_url="http://dummy"),
        tools=[tool],
    )

    _stub_compose_answer(agent)

    agent._llm.responses = [
        make_response([make_tool_call("initial query", "call-1")], content=""),
        make_response([], content="Fertig."),
    ]

    reply = agent.answer_with_rag(
        "Provide update",
        patient_context="",
        report_type="Arztbrief",
        report_date="2020",
        top_k=3,
    )

    assert tool.calls == ["initial query"]
    tool_messages = [msg for msg in agent._chat_history if msg.role == MessageRole.TOOL]
    assert tool_messages, "Expected tool messages in chat history"
    first_tool_args = tool_messages[0].additional_kwargs.get("arguments", {})
    assert first_tool_args.get("report_type") == "Arztbrief"
    assert first_tool_args.get("report_date") == "2020"
    assert first_tool_args.get("top_k") == 3
    assert reply.additional_kwargs["subqueries"] == ["initial query"]
