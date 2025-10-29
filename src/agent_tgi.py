"""
Modernized chat agent that wraps a local OpenAI-compatible endpoint using
llama-index's AgentWorkflow and OpenAILike client.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike

from .toolkit import BaseTool, ToolOutput

logger = logging.getLogger(__name__)


@dataclass
class TGIConfig:
    """Connection and model settings for the OpenAI-compatible endpoint."""

    endpoint_url: str = "http://localhost:8080/v1/chat/completions"
    timeout: float = 120.0
    api_key: Optional[str] = None
    model: Optional[str] = None
    system_prompt: str = (
        "You are a helpful tool-using assistant. Examine the available tools, decide "
        "when each is needed, invoke them with the required arguments, then compose "
        "your final reply using the tool outputs."
    )
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def api_base(self) -> str:
        """Return the base URL expected by OpenAILike."""
        url = self.endpoint_url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        return url

    def model_name(self) -> str:
        return self.model or "casperhansen/llama-3.3-70b-instruct-awq"


class TGIChatAgent:
    """Thin wrapper around AgentWorkflow for local TGI-style deployments."""

    def __init__(
        self,
        *,
        config: TGIConfig,
        tools: Optional[Iterable[BaseTool]] = None,
        event_handler: Optional[Callable[[Any], None]] = None,
        http_client: Optional[Any] = None,
        async_http_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.tools: List[BaseTool] = list(tools or [])
        self._chat_history: List[ChatMessage] = []
        self._event_handler = event_handler
        self._http_client = http_client
        self._async_http_client = async_http_client

        self._llm = self._create_llm()
        self._workflow = self._create_workflow(self.tools)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        message: str,
        *,
        tool_choice: str | Dict[str, Any] = "auto",
        **run_kwargs: Any,
    ) -> ChatMessage:
        """Send a message through the workflow and return the assistant reply."""
        # AgentWorkflow handles tool routing internally; tool_choice is currently
        # unused but kept for API compatibility.
        del tool_choice

        workflow_result = self._execute_workflow(message, **run_kwargs)

        user_message = ChatMessage(role=MessageRole.USER, content=message)
        self._chat_history.append(user_message)

        tool_messages = self._materialize_tool_messages(workflow_result.tool_calls)
        self._chat_history.extend(tool_messages)

        assistant_message = self._ensure_chat_message(workflow_result.response)
        self._chat_history.append(assistant_message)
        return assistant_message

    def reset(self) -> None:
        """Clear any stored conversation state."""
        self._chat_history.clear()

    @property
    def history(self) -> Sequence[ChatMessage]:  # pragma: no cover - convenience
        return tuple(self._chat_history)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _create_llm(self) -> OpenAILike:
        llm_kwargs = dict(self.config.llm_kwargs)
        llm_kwargs.setdefault("timeout", self.config.timeout)
        llm_kwargs.setdefault("is_chat_model", True)
        llm_kwargs.setdefault("is_function_calling_model", True)

        return OpenAILike(
            model=self.config.model_name(),
            api_base=self.config.api_base(),
            api_key=self.config.api_key,
             http_client=self._http_client,
             async_http_client=self._async_http_client,
            **llm_kwargs,
        )

    def _create_workflow(self, tools: Iterable[BaseTool]) -> AgentWorkflow:
        kwargs: Dict[str, Any] = dict(
            llm=self._llm,
            system_prompt=self.config.system_prompt,
        )
        if self._event_handler is not None:
            kwargs["event_handler"] = self._event_handler

        return AgentWorkflow.from_tools_or_functions(list(tools), **kwargs)

    def _execute_workflow(self, message: str, **run_kwargs: Any):
        history = list(self._chat_history)

        async def _run() -> Any:
            return await self._workflow.run(
                user_msg=message,
                chat_history=history,
                **run_kwargs,
            )

        return _run_sync(_run())

    def _materialize_tool_messages(self, tool_calls: Any) -> List[ChatMessage]:
        messages: List[ChatMessage] = []
        for call in tool_calls or []:
            tool_output = getattr(call, "tool_output", None)
            content: str
            if tool_output is None:
                content = ""
            else:
                content = getattr(tool_output, "content", str(tool_output))

            additional_kwargs = {
                "name": getattr(call, "tool_name", None),
            }
            tool_call_id = getattr(call, "tool_id", None)
            if tool_call_id is not None:
                additional_kwargs["tool_call_id"] = tool_call_id

            logger.debug(
                "Tool call: name=%s kwargs=%s output=%s",
                additional_kwargs.get("name"),
                getattr(call, "tool_kwargs", None),
                getattr(tool_output, "raw_output", content if tool_output else None),
            )

            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=content,
                    additional_kwargs=additional_kwargs,
                )
            )
        return messages

    @staticmethod
    def _ensure_chat_message(response: Any) -> ChatMessage:
        if isinstance(response, ChatMessage):
            return response
        return ChatMessage(role=MessageRole.ASSISTANT, content=str(response or ""))


def _run_sync(awaitable):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    if loop.is_running():
        raise RuntimeError(
            "TGIChatAgent.chat() cannot be called while an event loop is running in the same thread."
        )

    return loop.run_until_complete(awaitable)
