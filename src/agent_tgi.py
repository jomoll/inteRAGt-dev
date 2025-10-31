"""OpenAI-compatible (TGI) agent that delegates orchestration to the DSPy base class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike

from .agent_base import DSPyAgentBase
from .toolkit import BaseTool


@dataclass
class TGIConfig:
    """Connection and model settings for the OpenAI-compatible endpoint."""

    endpoint_url: str = "http://localhost:8080/v1/chat/completions"
    timeout: float = 120.0
    api_key: Optional[str] = None
    model: Optional[str] = None
    system_prompt: str = (
        "Du bist eine wissbegierige klinische Assistenz. Plane die Arbeitsschritte, nutze verfügbare Werkzeuge "
        "gezielt und begründe Antworten mit Quellen."
    )
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    def api_base(self) -> str:
        url = self.endpoint_url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        return url

    def model_name(self) -> str:
        return self.model or "casperhansen/llama-3.3-70b-instruct-awq"


class TGIChatAgent(DSPyAgentBase):
    """DSPy-enabled agent for OpenAI-compatible (TGI) deployments."""

    def __init__(
        self,
        *,
        config: TGIConfig,
        tools: Optional[Iterable[BaseTool]] = None,
        http_client: Any = None,
        async_http_client: Any = None,
        max_tool_rounds: int = 6,
    ) -> None:
        self.config = config
        self._http_client = http_client
        self._async_http_client = async_http_client
        self._system_prompt = config.system_prompt
        self._llm = self._create_llm()
        super().__init__(tools=tools, max_tool_rounds=max_tool_rounds)

    # ------------------------------------------------------------------
    # DSPyAgentBase bridge
    # ------------------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        response = self._llm.chat(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return self._extract_text_from_chat(response)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def chat(self, message: str, **kwargs: Any) -> ChatMessage:
        """Single-turn chat helper without RAG tooling."""
        del kwargs
        self.reset()
        self._append_user_message(message)
        reply_text = self._complete(message)
        reply = ChatMessage(role=MessageRole.ASSISTANT, content=reply_text)
        self._history.append(reply)
        return reply

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_llm(self) -> OpenAILike:
        llm_kwargs = dict(self.config.llm_kwargs)
        llm_kwargs.setdefault("timeout", self.config.timeout)
        llm_kwargs.setdefault("is_chat_model", True)
        llm_kwargs.setdefault("is_function_calling_model", False)

        return OpenAILike(
            model=self.config.model_name(),
            api_base=self.config.api_base(),
            api_key=self.config.api_key,
            http_client=self._http_client,
            async_http_client=self._async_http_client,
            **llm_kwargs,
        )

    @staticmethod
    def _extract_text_from_chat(response: Any) -> str:
        if hasattr(response, "message") and getattr(response.message, "content", None):
            return response.message.content
        if hasattr(response, "content"):
            return getattr(response, "content") or ""
        return str(response or "")
