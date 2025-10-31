"""
Gemini-backed clinical agent that follows the DSPy planning → tool → summary flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

try:  # pragma: no cover - optional dependency
    from google.generativeai import GenerativeModel, configure  # type: ignore
except Exception:  # pragma: no cover
    GenerativeModel = None  # type: ignore

    def configure(*args, **kwargs):  # type: ignore
        raise ImportError("google-generativeai is required for GeminiChatAgent.")

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from .agent_base import DSPyAgentBase
from .toolkit import BaseTool


@dataclass
class GeminiConfig:
    """Gemini client configuration."""

    api_key: str
    model: str
    safety_settings: Optional[dict[str, Any]] = None
    generation_config: Optional[dict[str, Any]] = None
    system_prompt: Optional[str] = (
        "Du bist eine klinische Assistenz, die strukturierte Pläne erstellt, Werkzeuge gezielt einsetzt "
        "und Antworten mit Quellenangaben liefert. Arbeite auf Deutsch."
    )


class GeminiChatAgent(DSPyAgentBase):
    """Gemini-based implementation of the DSPy agent scaffold."""

    def __init__(
        self,
        *,
        config: GeminiConfig,
        tools: Optional[Iterable[BaseTool]] = None,
        max_tool_rounds: int = 6,
    ) -> None:
        if GenerativeModel is None:  # pragma: no cover - dependency missing
            raise ImportError("google-generativeai is required for GeminiChatAgent.")

        configure(api_key=config.api_key)
        self.config = config
        self.model = GenerativeModel(
            model_name=config.model,
            safety_settings=config.safety_settings,
            generation_config=config.generation_config,
            system_instruction=config.system_prompt,
        )
        super().__init__(tools=tools, max_tool_rounds=max_tool_rounds)

    # ------------------------------------------------------------------
    # DSPyAgentBase bridge
    # ------------------------------------------------------------------
    def _complete(self, prompt: str) -> str:
        response = self.model.generate_content([prompt])
        return self._extract_text_from_response(response)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def chat(self, message: str, **kwargs: Any) -> ChatMessage:
        """Lightweight single-turn chat without RAG orchestration."""
        del kwargs
        self.reset()
        self._append_user_message(message)
        reply_text = self._complete(message)
        reply = ChatMessage(role=MessageRole.ASSISTANT, content=reply_text)
        self._history.append(reply)
        return reply

    # ------------------------------------------------------------------
    # Gemini helpers
    # ------------------------------------------------------------------
    def _extract_text_from_response(self, response: Any) -> str:
        if not getattr(response, "candidates", None):
            return ""
        for candidate in response.candidates:
            parts = getattr(candidate, "content", None)
            parts = getattr(parts, "parts", []) if parts else []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    return text
        return ""
