"""Deterministic offline engine for tests and for exercising the pipeline without a model."""

from __future__ import annotations

import hashlib
import math

from jeb.engines.base import ScoreItem, ScoreResult


class FakeEngine:
    name = "fake"

    def __init__(self, scripted: dict[tuple[int, ...], list[float]] | None = None, in_set_mass: float = 0.9, seed: int = 0) -> None:
        """`scripted` maps a label-id tuple to fixed label probabilities (they need not sum to 1)."""
        self.scripted = scripted or {}
        self.in_set_mass = in_set_mass
        self.seed = seed
        self.calls: list[list[ScoreItem]] = []

    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]:
        self.calls.append(items)
        out = []
        for it in items:
            key = tuple(it.label_ids)
            if key in self.scripted:
                probs = self.scripted[key]
            else:
                h = hashlib.sha256((str(it.prompt_ids[-40:]) + str(it.label_ids) + str(self.seed)).encode()).digest()
                raw = [1.0 + h[i % len(h)] / 64.0 for i in range(len(it.label_ids))]
                probs = raw
            z = sum(probs)
            lps = [math.log(p / z * self.in_set_mass) for p in probs]
            out.append(ScoreResult(logprobs=lps, prompt_tokens=len(it.prompt_ids)))
        return out

    async def aclose(self) -> None:
        return None
