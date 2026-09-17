"""Request and work-item types used by the scheduler simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestState(Enum):
    QUEUED = auto()
    PREFILL = auto()
    DECODING = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass
class Request:
    request_id: str
    prompt: tuple[str, ...]
    max_new_tokens: int
    state: RequestState = RequestState.QUEUED
    block_ids: list[int] = field(default_factory=list)
    generated: list[str] = field(default_factory=list)
    in_flight: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.state in {RequestState.COMPLETED, RequestState.CANCELLED}


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    phase: str
    block_id: int
    payload: tuple[str, ...]


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token: str


def next_token(request_id: str, position: int, context: tuple[str, ...]) -> str:
    """A deterministic stand-in for a model forward pass."""

    if context:
        signature = f"{len(context)}:{context[0]}:{context[-1]}"
    else:
        signature = "0:<empty>:<empty>"
    return f"{request_id}.{position}<-{signature}"
