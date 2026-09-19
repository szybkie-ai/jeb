from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ScoreItem:
    prompt_ids: list[int]
    label_ids: list[int]
    chat: dict | None = None  # {"system", "user", "images"}: score through the chat endpoint (images need it)


@dataclass
class ScoreResult:
    logprobs: list[float]  # aligned with ScoreItem.label_ids; log-softmax over the full vocabulary
    prompt_tokens: int
    missing: int = 0  # labels the engine did not report (filled with a floor)


class EngineError(RuntimeError):
    pass


class Engine(Protocol):
    name: str

    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]: ...

    async def aclose(self) -> None: ...
