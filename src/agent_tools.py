"""
Lightweight dummy tools for the local agents.

These mirror the structure of the production tools in
`LLM_RAG_Agent/RAGent/DSPY/agent_tools_dummy.py`, but keep the implementation
simple so they are safe to use in tests or demos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .toolkit import BaseTool, ToolOutput, ToolMetadata


__all__ = [
    "EchoTool",
    "WordCountTool",
    "SummationTool",
    "load_default_tools",
]


class EchoTool(BaseTool):
    """Return the supplied payload verbatim."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="echo_tool",
            description="Return the provided payload as JSON.",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(self, **kwargs) -> ToolOutput:
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(kwargs),
            raw_input={"kwargs": kwargs},
            raw_output=kwargs,
        )


class WordCountTool(BaseTool):
    """Count words in supplied text, tolerating varied argument formats."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="word_count",
            description="Count how many words appear in the provided text.",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(
        self,
        text: str | None = None,
        input: Any | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        candidate = self._coerce_text(text, input, kwargs)
        words = candidate.split()
        result = {"word_count": len(words)}
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result),
            raw_input={"kwargs": {"text": candidate}},
            raw_output=result,
        )

    def _coerce_text(self, text, input_value, extra_kwargs) -> str:
        if text:
            return text
        for payload in (input_value, extra_kwargs.get("input")):
            if payload:
                if isinstance(payload, str):
                    try:
                        parsed = json.loads(payload)
                        if isinstance(parsed, dict) and "text" in parsed:
                            return parsed["text"]
                        return parsed if isinstance(parsed, str) else payload
                    except json.JSONDecodeError:
                        return payload
                if isinstance(payload, dict) and "text" in payload:
                    return payload["text"]
        raise ValueError("missing a required argument: 'text'")



@dataclass
class SummationInput:
    numbers: Iterable[float]


class SummationTool(BaseTool):
    """Compute the sum of numeric values."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="sum_numbers",
            description="Return the sum of numbers passed in the 'numbers' list.",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(  # type: ignore[override]
        self,
        numbers: Iterable[float] | None = None,
        input: Any | None = None,
        **kwargs: Any,
    ) -> ToolOutput:
        """Accept arguments in multiple formats to accommodate different model outputs."""
        extracted_numbers = self._coerce_numbers(numbers, input, kwargs)
        total = sum(extracted_numbers)
        result = {"sum": total}
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result),
            raw_input={"kwargs": {"numbers": extracted_numbers}},
            raw_output=result,
        )

    def _coerce_numbers(
        self,
        numbers: Iterable[float] | None,
        input_value: Any,
        extra_kwargs: Dict[str, Any],
    ) -> List[float]:
        if numbers is None:
            candidate = self._parse_dynamic_payload(input_value)
            if candidate is None:
                candidate = self._parse_dynamic_payload(extra_kwargs)
            numbers = candidate

        if numbers is None:
            raise ValueError("missing a required argument: 'numbers'")

        coerced: List[float] = []
        for value in numbers:
            try:
                coerced.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unable to convert '{value}' to a float.") from exc
        return coerced

    def _parse_dynamic_payload(self, payload: Any) -> Iterable[float] | None:
        if payload is None:
            return None

        if isinstance(payload, (list, tuple)):
            return payload

        if isinstance(payload, dict):
            if "numbers" in payload and isinstance(payload["numbers"], (list, tuple)):
                return payload["numbers"]
            # handle {"input": {...}}
            if "input" in payload:
                return self._parse_dynamic_payload(payload["input"])
            # search nested dicts for first list
            for value in payload.values():
                candidate = self._parse_dynamic_payload(value)
                if candidate is not None:
                    return candidate
            return None

        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            return self._parse_dynamic_payload(parsed)

        return None


def load_default_tools() -> List[BaseTool]:
    """Return a small bundle of dummy tools."""
    return [EchoTool(), WordCountTool(), SummationTool()]
