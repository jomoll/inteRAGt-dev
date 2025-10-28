import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

try:  # pragma: no cover - optional dependencies
    from src.agent_tgi import TGIChatAgent, TGIConfig
    from src.agent_gemini import GeminiChatAgent, GeminiConfig
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Agent tests require optional dependencies: {exc}", allow_module_level=True)

from src.agent_tools import WordCountTool, SummationTool, EchoTool


def _fake_tgi_response(content: str | None, tool_calls: list[dict] | None):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                }
            }
        ]
    }


def test_tgi_agent_executes_tool_and_returns_final_message():
    tool = WordCountTool()
    config = TGIConfig(endpoint_url="http://dummy")
    agent = TGIChatAgent(config=config, tools=[tool])

    first = _fake_tgi_response(
        content=None,
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "arguments": json.dumps({"text": "hi there"}),
                },
            }
        ],
    )
    second = _fake_tgi_response(content="final reply", tool_calls=None)

    with patch("src.agent_tgi.requests.post") as mock_post:
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: first),
            MagicMock(status_code=200, json=lambda: second),
        ]

        reply = agent.chat("hello")

    assert reply.content == "final reply"
    first_message = agent._chat_history[0]  # type: ignore[attr-defined]
    assert first_message.role == MessageRole.USER
    assert first_message.content == "hello"
    # Tool output should have been appended for the second model call
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    assert len(tool_messages) == 1
    assert json.loads(tool_messages[0].content)["word_count"] == 2


def test_tgi_agent_handles_tool_error():
    class FailingTool(EchoTool):
        def __call__(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    tool = FailingTool()
    config = TGIConfig(endpoint_url="http://dummy")
    agent = TGIChatAgent(config=config, tools=[tool])

    first = _fake_tgi_response(
        content=None,
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "arguments": "{}",
                },
            }
        ],
    )
    second = _fake_tgi_response(content="done", tool_calls=None)

    with patch("src.agent_tgi.requests.post") as mock_post:
        mock_post.side_effect = [
            MagicMock(status_code=200, json=lambda: first),
            MagicMock(status_code=200, json=lambda: second),
        ]
        reply = agent.chat("hello")

    assert reply.content == "done"
    # Tool should have been attempted once; error response is captured as ToolOutput.
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    assert tool_messages
    assert "Tool error" in tool_messages[0].content


def _make_gemini_response(text: str, tool_call: SimpleNamespace | None):
    parts = []
    if tool_call is not None:
        parts.append(SimpleNamespace(function_call=tool_call.function_call))
    if text:
        parts.append(SimpleNamespace(text=text))
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[candidate])


@patch("src.agent_gemini.configure")
def test_gemini_agent_executes_tool(mock_configure):
    tool = SummationTool()
    config = GeminiConfig(api_key="dummy", model="gemini-test")

    mock_model = MagicMock()
    tool_call = SimpleNamespace(
        function_call=SimpleNamespace(
            name=tool.metadata.name, args_json=json.dumps({"numbers": [1, 2, 3]})
        )
    )

    # First call requests the tool; second call returns final text.
    mock_model.generate_content.side_effect = [
        _make_gemini_response("", tool_call),
        _make_gemini_response("completed", None),
    ]

    with patch(
        "src.agent_gemini.GenerativeModel",
        return_value=mock_model,
    ):
        agent = GeminiChatAgent(config=config, tools=[tool])
        reply = agent.chat("summarize patient")

    assert reply.content == "completed"
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    assert tool_messages
    assert json.loads(tool_messages[0].content)["sum"] == 6
    assert mock_configure.called
