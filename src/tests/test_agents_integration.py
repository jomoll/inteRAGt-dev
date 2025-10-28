"""Tests for integration of TGI and Gemini chat agents with live endpoints.
Run: 
python -m pytest -s src/tests/test_agents_integration.py"""
import os
from pathlib import Path

import pytest

from src.agent_tgi import TGIChatAgent, TGIConfig
from src.agent_gemini import GeminiChatAgent, GeminiConfig
from src.agent_tools import load_default_tools
from llama_index.core.base.llms.types import MessageRole
from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)


@pytest.fixture(scope="module")
def default_tools():
    return load_default_tools()


@pytest.mark.integration
def test_tgi_agent_live(default_tools):
    endpoint = os.getenv("TGI_ENDPOINT")
    if not endpoint:
        pytest.skip("Set TGI_ENDPOINT to run live TGI integration test.")

    config = TGIConfig(
        endpoint_url=endpoint,
        api_key=os.getenv("TGI_API_KEY"),
        model=os.getenv("TGI_MODEL"),
    )
    agent = TGIChatAgent(config=config, tools=default_tools)

    prompt = (
        "Use the word_count tool to count the words in the phrase 'open source ai'. "
        "After calling the tool, report the count."
    )

    print("\n[Integration:TGI] Prompt:", prompt)
    reply = agent.chat(prompt)

    print("[Integration:TGI] Final reply:", reply.content)
    assert reply.content is not None
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    print("[Integration:TGI] Tool messages:")
    for msg in tool_messages:
        print("  -", msg.content)
    successful = [msg for msg in tool_messages if "Tool error" not in msg.content]
    assert successful, "Expected at least one successful tool call."  # noqa: PT018


@pytest.mark.integration
def test_gemini_agent_live(default_tools):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("Set GEMINI_API_KEY to run live Gemini integration test.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    config = GeminiConfig(api_key=api_key, model=model_name)
    agent = GeminiChatAgent(config=config, tools=default_tools)

    prompt = (
        "Call the sum_numbers tool with the numbers [1, 2, 3] and then explain the result."
    )

    print("\n[Integration:Gemini] Prompt:", prompt)
    reply = agent.chat(prompt)

    print("[Integration:Gemini] Final reply:", reply.content)
    assert reply.content is not None
    tool_messages = [m for m in agent._chat_history if m.role == MessageRole.TOOL]  # type: ignore[attr-defined]
    print("[Integration:Gemini] Tool messages:")
    for msg in tool_messages:
        print("  -", msg.content)
    successful = [msg for msg in tool_messages if "Tool error" not in msg.content]
    assert successful, "Expected at least one successful tool call."  # noqa: PT018
