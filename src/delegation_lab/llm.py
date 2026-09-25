"""Provider-neutral boundary: normalized proposals enter, never trusted facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolProposal:
    name: str
    arguments: dict[str, Any]
    proposal_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 1
    wall_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelOutput:
    text: str = ""
    tool_calls: tuple[ToolProposal, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    raw: Any = None


class LLMAdapter(Protocol):
    """The only interface a generic agent or candidate miner needs."""

    model_id: str
    model_revision: str

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSchema] = ()) -> ModelOutput: ...


class MockLLMAdapter:
    """Deterministic test adapter proving no core component depends on Qwen."""

    model_id = "mock/deterministic"
    model_revision = "v1"

    def __init__(self, outputs: Sequence[ModelOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[tuple[tuple[ChatMessage, ...], tuple[ToolSchema, ...]]] = []

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSchema] = ()) -> ModelOutput:
        self.calls.append((tuple(messages), tuple(tools)))
        if not self._outputs:
            return ModelOutput(text="I need to defer.", usage=ModelUsage(model_calls=1))
        return self._outputs.pop(0)
