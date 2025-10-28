"""
Minimal chat agent skeleton for a Text Generation Inference (TGI) endpoint.

The goal is to demonstrate how to wrap a local TGI server that has been launched
with the `tool_choice` Guidance program so that it accepts OpenAI-style tool
specifications and returns tool-calling events. Fill in the TODOs with your
project-specific logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from .toolkit import BaseTool, ToolOutput

logger = logging.getLogger(__name__)


@dataclass
class TGIConfig:
    """Connection settings for the TGI endpoint."""

    endpoint_url: str = "http://localhost:8080/v1/chat/completions"
    timeout: float = 120.0
    api_key: Optional[str] = None  # Set if your gateway enforces auth
    model: Optional[str] = None  # Optional: pass-through for multi-model routers


class TGIChatAgent:
    """
    Lightweight controller that speaks to a TGI server using the OpenAI-compatible
    tool-calling contract. This class intentionally mirrors the public surface of
    `MedGeminiAgent` so you can swap implementations easily.
    """

    def __init__(
        self,
        *,
        config: TGIConfig,
        tools: Optional[Iterable[BaseTool]] = None,
    ) -> None:
        self.config = config
        self.tools: List[BaseTool] = list(tools or [])
        self._chat_history: List[ChatMessage] = []

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def chat(
        self,
        message: str,
        *,
        tool_choice: str | Dict[str, Any] = "auto",
    ) -> ChatMessage:
        """
        Send a message to the TGI endpoint and orchestrate any required tool calls.
        Returns the final assistant message.
        """
        self._append_user_message(message)

        while True:
            payload = self._build_payload(tool_choice)
            logger.debug("Dispatching payload to TGI: %s", payload)

            response = self._invoke_tgi(payload)
            assistant_message = self._extract_message(response)

            self._chat_history.append(assistant_message)
            tool_calls = response.get("tool_calls") or assistant_message.additional_kwargs.get(
                "tool_calls"
            )

            if not tool_calls:
                return assistant_message

            logger.debug("Handling tool calls returned by the model: %s", tool_calls)
            self._run_tools(tool_calls)

    def reset(self) -> None:
        """Clear the conversation state."""
        self._chat_history.clear()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _append_user_message(self, message: str) -> None:
        self._chat_history.append(ChatMessage(role=MessageRole.USER, content=message))

    def _build_payload(self, tool_choice: str | Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct the OpenAI-style payload expected by TGI's `tool_choice` guidance.
        Override this method to expose additional knobs (temperature, stop, etc.).
        """
        payload: Dict[str, Any] = {
            "messages": [m.model_dump() for m in self._chat_history],
            "tools": [tool.metadata.to_openai_tool() for tool in self.tools],
            "tool_choice": tool_choice,
        }
        if self.config.model:
            payload["model"] = self.config.model
        return payload

    def _invoke_tgi(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = requests.post(
            self.config.endpoint_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    def _extract_message(self, response: Dict[str, Any]) -> ChatMessage:
        """
        Convert the OpenAI-style message dict returned by TGI into a ChatMessage.
        """
        content = response.get("content")
        additional_kwargs = {k: v for k, v in response.items() if k != "content"}
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            additional_kwargs=additional_kwargs,
        )

    def _run_tools(self, tool_calls: List[Dict[str, Any]]) -> None:
        """
        Execute each tool selected by the model, append the results to the history,
        and loop back so the model can reason over the tool outputs.
        """
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"] or "{}")
            logger.info("Invoking tool %s with arguments %s", tool_name, arguments)

            tool = self._get_tool(tool_name)
            tool_output = self._safe_tool_call(tool, arguments)

            tool_message = ChatMessage(
                role=MessageRole.TOOL,
                content=str(tool_output),
                additional_kwargs={
                    "name": tool_name,
                    "tool_call_id": tool_call.get("id"),
                },
            )
            self._chat_history.append(tool_message)

    def _get_tool(self, name: str) -> BaseTool:
        for tool in self.tools:
            if tool.metadata.name == name:
                return tool
        raise ValueError(f"Tool with name '{name}' not registered.")

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
