"""Integration tests for agents backed by live endpoints.

Run:
python -m pytest -s src/tests/test_agents_integration.py
"""

import json
import logging
import os
import time
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, MessageRole

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("src.agent_gemini").setLevel(logging.DEBUG)
logging.getLogger("src.agent_tgi").setLevel(logging.DEBUG)

try:  # pragma: no cover - optional heavy dependencies
    from src.agent_tgi import TGIChatAgent, TGIConfig
    from src.agent_gemini import GeminiChatAgent, GeminiConfig
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Integration tests require optional agent dependencies: {exc}", allow_module_level=True)

from src.agent_tools import load_default_tools

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=False)

@pytest.fixture(scope="module")
def default_tools():
    return load_default_tools()

def _tool_scenarios():
    word_text = "open source ai"
    word_args = {"input": word_text}
    sum_args = {"numbers": [1, 2, 3]}
    echo_payload = "hello integration"
    echo_args = {"input": echo_payload}

    def _word_validator(expected: int):
        def _validator(data: dict) -> bool:
            value = data.get("word_count")
            try:
                return int(float(value)) == expected
            except (TypeError, ValueError):
                return False

        return _validator

    def _sum_validator(expected_sum: float):
        def _validator(data: dict) -> bool:
            value = data.get("sum")
            try:
                return float(value) == float(expected_sum)
            except (TypeError, ValueError):
                return False

        return _validator

    def _echo_validator(expected_payload: dict):
        def _validator(data: dict) -> bool:
            # Gemini sometimes echoes using `input` rather than `payload`
            candidate = dict(data)
            if "input" in candidate and "payload" not in candidate:
                candidate["payload"] = candidate["input"]
            return all(str(candidate.get(key)) == str(val) for key, val in expected_payload.items())

        return _validator

    return [
        {
            "label": "word_count",
            "tool_name": "word_count",
            "arguments": word_args,
            "prompt": (
                "You must call the word_count tool exactly once using the arguments "
                f"{json.dumps(word_args)} (the tool expects an 'input' argument). "
                "After the tool returns, reply with only the number of words."
            ),
            "validator": _word_validator(len(word_text.split())),
        },
        {
            "label": "sum_numbers",
            "tool_name": "sum_numbers",
            "arguments": sum_args,
            "prompt": (
                "You must call the sum_numbers tool exactly once using the arguments "
                f"{json.dumps(sum_args)}. After the tool returns, reply with only the sum."
            ),
            "validator": _sum_validator(sum(sum_args["numbers"])),
        },
        {
            "label": "echo_tool",
            "tool_name": "echo_tool",
            "arguments": echo_args,
            "prompt": (
                "You must call the echo_tool exactly once using the arguments "
                f"{json.dumps(echo_args)} (the tool expects an 'input' argument). "
                "After the tool returns, repeat the payload verbatim."
            ),
            "validator": _echo_validator({"payload": echo_payload}),
        },
    ]


def _extract_response_payload(message) -> dict | None:
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict):
        payload = additional.get("response")
        if isinstance(payload, dict):
            return payload

    content = getattr(message, "content", None)
    if content is None:
        return None
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return None


def _print_chat_history(agent, label: str) -> None:
    history = getattr(agent, "history", None)
    if history is None:
        history = getattr(agent, "_chat_history", [])
    print(f"[Integration:{label}] Conversation transcript:")
    for idx, message in enumerate(history, start=1):
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        additional = getattr(message, "additional_kwargs", None)
        print(f"  [{idx}] role={role} content={content!r} additional={additional}")


@pytest.mark.integration
def test_tgi_agent_live(default_tools):
    endpoint = os.getenv("TGI_ENDPOINT")
    if not endpoint:
        pytest.skip("Set TGI_ENDPOINT to run live TGI integration test.")

    api_key = os.getenv("TGI_API_KEY")
    model_name = os.getenv("TGI_MODEL", "casperhansen/llama-3.3-70b-instruct-awq")

    debug_events = os.getenv("TGI_DEBUG_EVENTS", "0") == "1"
    debug_httpx = os.getenv("TGI_DEBUG_HTTPX", "0") == "1"

    event_handler = None
    if debug_events:
        def _log_event(event: object) -> None:
            print(f"[workflow event] {event}")

        event_handler = _log_event

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

    config = TGIConfig(endpoint_url=endpoint, api_key=api_key, model=model_name)
    agent = TGIChatAgent(
        config=config,
        tools=default_tools,
        event_handler=event_handler,
        http_client=http_client,
    )

    max_retries = int(os.getenv("TGI_MAX_RETRIES", "100"))
    retry_delay = float(os.getenv("TGI_RETRY_DELAY_SECONDS", "1"))
    max_iterations = int(os.getenv("TGI_MAX_ITERATIONS", "40"))
    scenarios = _tool_scenarios()

    def run_prompt(prompt_text: str) -> ChatMessage:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return agent.chat(prompt_text, max_iterations=max_iterations)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response else None
                agent.reset()
                print(
                    f"[Integration:TGI] Attempt {attempt}/{max_retries} failed with HTTP {status}: {exc!s}"
                )
                if status == 404:
                    pytest.skip(
                        f"TGI endpoint {endpoint} returned 404 Not Found. "
                        "Verify the server exposes an OpenAI-compatible /chat/completions route."
                    )
                if status in {401, 403}:
                    pytest.skip(
                        f"TGI endpoint {endpoint} rejected the request (status {status}). "
                        "Check API key or auth configuration."
                    )
                if attempt < max_retries:
                    time.sleep(retry_delay)
            except httpx.RequestError as exc:
                last_error = exc
                agent.reset()
                print(
                    f"[Integration:TGI] Attempt {attempt}/{max_retries} failed: {exc!s}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
            except Exception as exc:  # noqa: BLE001 - surface unexpected issues
                last_error = exc
                agent.reset()
                print(
                    f"[Integration:TGI] Attempt {attempt}/{max_retries} raised unexpected error: {exc!s}"
                )
                if attempt < max_retries:
                    time.sleep(retry_delay)
        pytest.skip(
            f"TGI endpoint unavailable after {max_retries} attempts: {last_error}"
        )

    try:
        for scenario in scenarios:
            agent.reset()
            print(f"\n[Integration:TGI][{scenario['label']}] Prompt: {scenario['prompt']}")
            reply = run_prompt(scenario["prompt"])
            final_content = getattr(reply, "content", None)
            print(f"[Integration:TGI][{scenario['label']}] Final reply: {final_content}")

            tool_messages = [
                msg
                for msg in agent._chat_history
                if msg.role == MessageRole.TOOL  # type: ignore[attr-defined]
            ]
            print("[Integration:TGI] Tool messages:")
            for msg in tool_messages:
                print("  -", msg.content)
            _print_chat_history(agent, f"TGI[{scenario['label']}]")

            matching_messages = [
                msg
                for msg in tool_messages
                if msg.additional_kwargs.get("name") == scenario["tool_name"]
            ]
            assert matching_messages, (
                f"Expected at least one tool call for {scenario['tool_name']} but none were recorded."
            )

            successful_payloads = []
            for msg in matching_messages:
                content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
                if "Tool error" in content_str:
                    continue
                payload = _extract_response_payload(msg)
                if payload is None:
                    continue
                if scenario["validator"](payload):
                    successful_payloads.append(payload)

            assert (
                successful_payloads
            ), f"Tool {scenario['tool_name']} did not produce the expected result."
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

    scenarios = _tool_scenarios()

    for scenario in scenarios:
        agent.reset()
        print(f"\n[Integration:Gemini][{scenario['label']}] Prompt: {scenario['prompt']}")
        reply = agent.chat(scenario["prompt"])

        print(f"[Integration:Gemini][{scenario['label']}] Final reply: {getattr(reply, 'content', None)}")
        tool_messages = [
            msg
            for msg in agent._chat_history
            if msg.role == MessageRole.TOOL  # type: ignore[attr-defined]
        ]
        print("[Integration:Gemini] Tool messages:")
        for msg in tool_messages:
            print("  -", msg.content)
        _print_chat_history(agent, f"Gemini[{scenario['label']}]")

        matching_messages = []
        for idx, msg in enumerate(tool_messages, start=1):
            additional = getattr(msg, "additional_kwargs", {})
            print(
                f"[Integration:Gemini][{scenario['label']}] tool_message[{idx}] additional_kwargs=",
                additional,
            )
            if additional.get("name") == scenario["tool_name"]:
                matching_messages.append(msg)
        assert matching_messages, (
            f"Expected at least one tool call for {scenario['tool_name']} but none were recorded."
        )

        successful_payloads = []
        for msg in matching_messages:
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(
                f"[Integration:Gemini][{scenario['label']}] matching message content=",
                content_str,
            )
            if "Tool error" in content_str:
                print(
                    f"[Integration:Gemini][{scenario['label']}] skipping due to tool error"
                )
                continue
            payload = _extract_response_payload(msg)
            print(
                f"[Integration:Gemini][{scenario['label']}] extracted payload=",
                payload,
            )
            if payload is None:
                continue
            if scenario["validator"](payload):
                successful_payloads.append(payload)
            else:
                print(
                    f"[Integration:Gemini][{scenario['label']}] payload did not pass validator"
                )

        assert (
            successful_payloads
        ), f"Tool {scenario['tool_name']} did not produce the expected result."
