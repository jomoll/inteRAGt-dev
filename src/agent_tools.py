"""
Lightweight dummy tools for the local agents.

These mirror the structure of the production tools in
`LLM_RAG_Agent/RAGent/DSPY/agent_tools_dummy.py`, but keep the implementation
simple so they are safe to use in tests or demos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List

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
    """Count the number of words in the provided text."""

    def __init__(self) -> None:
        self._metadata = ToolMetadata(
            name="word_count",
            description="Count how many words appear in the 'text' field.",
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(self, text: str) -> ToolOutput:  # type: ignore[override]
        words = text.split()
        result = {"word_count": len(words)}
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result),
            raw_input={"kwargs": {"text": text}},
            raw_output=result,
        )


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

    def __call__(self, numbers: Iterable[float]) -> ToolOutput:  # type: ignore[override]
        total = sum(numbers)
        result = {"sum": total}
        return ToolOutput(
            tool_name=self._metadata.name,
            content=json.dumps(result),
            raw_input={"kwargs": {"numbers": list(numbers)}},
            raw_output=result,
        )


def load_default_tools() -> List[BaseTool]:
    """Return a small bundle of dummy tools."""
    return [EchoTool(), WordCountTool(), SummationTool()]
