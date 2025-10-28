"""
Minimal chat agent skeleton for Google Gemini models using the `google-generativeai`
SDK. The class mirrors the high-level structure of `MedGeminiAgent`, but leaves the
details of RAG integration and tool implementations to the caller.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from google.generativeai import GenerativeModel, configure  # type: ignore
except Exception:  # pragma: no cover
    GenerativeModel = None  # type: ignore

    def configure(*args, **kwargs):  # type: ignore
        raise ImportError(
            "google-generativeai is required for GeminiChatAgent; install the package to enable it."
        )

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from .toolkit import BaseTool, ToolOutput

logger = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    """Gemini client configuration."""

    api_key: str
    model: str = "gemini-2.0-flash"
    safety_settings: Optional[Dict[str, Any]] = None
    generation_config: Optional[Dict[str, Any]] = None


class GeminiChatAgent:
    """
    Thin wrapper around a Gemini model that understands OpenAI-style tool metadata.
    The agent keeps track of the running conversation and executes tools selected by
    the model.
    """

    def __init__(
        self,
        *,
        config: GeminiConfig,
        tools: Optional[Iterable[BaseTool]] = None,
    ) -> None:
        configure(api_key=config.api_key)
        self.config = config
        if GenerativeModel is None:  # pragma: no cover - dependency missing
            raise ImportError(
                "google-generativeai is required for GeminiChatAgent; install the package to enable it."
            )

        self.model = GenerativeModel(
            model_name=config.model,
            safety_settings=config.safety_settings,
            generation_config=config.generation_config,
        )
        self.tools: List[BaseTool] = list(tools or [])
        self._chat_history: List[ChatMessage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        *,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> ChatMessage:
        """
        Send a user instruction to Gemini, execute any requested tools, and return
        the final assistant message.
        """
        self._append_user_message(message)
        while True:
            model_response = self._invoke_model(tool_choice)
            assistant_message = self._wrap_assistant_message(model_response)
            self._chat_history.append(assistant_message)

            tool_calls = self._extract_tool_calls(model_response)
            if not tool_calls:
                return assistant_message

            logger.debug("Gemini requested tool calls: %s", tool_calls)
            self._run_tools(tool_calls)

    def reset(self) -> None:
        """Clear accumulated chat history."""
        self._chat_history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _append_user_message(self, message: str) -> None:
        self._chat_history.append(ChatMessage(role=MessageRole.USER, content=message))

    def _invoke_model(self, tool_choice: str | Dict[str, Any]):
        """
        Call Gemini's `generate_content` with the accumulated history and tool
        descriptors translated into Gemini's format.
        """
        history = [self._to_gemini_content(msg) for msg in self._chat_history]
        tool_specs = self._convert_tools_to_gemini()
        return self.model.generate_content(
            history,
            tools=tool_specs if tool_specs else None,
            tool_config={"tool_choice": tool_choice} if tool_choice != "auto" else None,
        )

    def _wrap_assistant_message(self, response) -> ChatMessage:
        """Convert Gemini's top-candidate output into a ChatMessage."""
        text = ""
        if getattr(response, "candidates", None):
            parts = getattr(response.candidates[0].content, "parts", [])
            for part in parts:
                candidate_text = getattr(part, "text", None)
                if candidate_text:
                    text = candidate_text
                    break
        return ChatMessage(role=MessageRole.ASSISTANT, content=text, additional_kwargs={"raw": response})

    def _run_tools(self, tool_calls: Any) -> None:
        """Execute each tool requested by Gemini and append tool outputs to history."""
        for tool_call in tool_calls:
            function_call = getattr(tool_call, "function_call", tool_call)
            if function_call is None:
                continue

            name = getattr(function_call, "name", None)
            if not name:
                continue

            args_json = getattr(function_call, "args_json", None)
            arguments = json.loads(args_json) if args_json else getattr(function_call, "args", {}) or {}
            logger.info("Running tool %s with args %s", name, arguments)

            tool = self._get_tool(name)
            output = self._safe_tool_call(tool, arguments)

            if isinstance(output.raw_output, dict):
                response_payload = output.raw_output
            else:
                try:
                    response_payload = json.loads(output.content)
                except Exception:  # pragma: no cover - fallback when not json
                    response_payload = {"content": output.content}

            tool_message = ChatMessage(
                role=MessageRole.TOOL,
                content=str(output),
                additional_kwargs={"name": name, "response": response_payload},
            )
            self._chat_history.append(tool_message)

    def _extract_tool_calls(self, response: Any) -> List[Any]:
        calls: List[Any] = []
        if not getattr(response, "candidates", None):
            return calls

        for candidate in response.candidates:
            parts = getattr(candidate, "content", None)
            parts = getattr(parts, "parts", []) if parts else []
            for part in parts:
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    calls.append(function_call)
        return calls

    def _convert_tools_to_gemini(self) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tool metadata into the Gemini `Tool` schema. This is a
        pared down version of the logic used in `med_agent.py`.
        """
        gemini_tools: List[Dict[str, Any]] = []
        for tool in self.tools:
            fn = tool.metadata.to_openai_tool()["function"]
            gemini_tools.append(
                {
                    "function_declarations": [
                        {
                            "name": fn["name"],
                            "description": fn.get("description", ""),
                            "parameters": self._sanitize_schema(fn.get("parameters", {})),
                        }
                    ]
                }
            )
        return gemini_tools

    def _sanitize_schema(self, schema: Any) -> Any:
        """Remove unsupported keys from JSON schema definitions."""
        if isinstance(schema, dict):
            return {k: self._sanitize_schema(v) for k, v in schema.items() if k != "title"}
        if isinstance(schema, list):
            return [self._sanitize_schema(item) for item in schema]
        return schema

    def _to_gemini_content(self, message: ChatMessage) -> Dict[str, Any]:
        """Translate ChatMessage objects into Gemini Content blocks."""
        if message.role == MessageRole.USER:
            return {"role": "user", "parts": [{"text": message.content or ""}]}

        if message.role == MessageRole.TOOL:
            name = message.additional_kwargs.get("name")
            payload = message.additional_kwargs.get("response", {})
            return {
                "role": "model",
                "parts": [
                    {
                        "function_response": {
                            "name": name,
                            "response": payload,
                        }
                    }
                ],
            }

        return {"role": "model", "parts": [{"text": message.content or ""}]}

    def _get_tool(self, name: str) -> BaseTool:
        for tool in self.tools:
            if tool.metadata.name == name:
                return tool
        raise ValueError(f"Tool '{name}' is not registered.")

    def _safe_tool_call(self, tool: BaseTool, arguments: Dict[str, Any]) -> ToolOutput:
        try:
            return tool(**arguments)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Tool %s failed: %s", tool.metadata.name, exc)
            return ToolOutput(
                tool_name=tool.metadata.name,
                content=f"Tool error: {exc}",
                raw_input={"kwargs": arguments},
                raw_output=None,
                is_error=True,
            )
