from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from jeb.engines import FakeEngine
from jeb.prompt import LETTERS, NO, YES, Labeler
from jeb.server import Settings, create_app


class DummyTokenizer:
    """Single ids for labels (mirroring Qwen3.5: A..Z = 32..57, Yes/No), pseudo-ids for everything else."""

    LABELS = {**{c: 32 + i for i, c in enumerate(LETTERS)}, YES: 9175, NO: 2665}

    def encode(self, text: str) -> list[int]:
        if text in self.LABELS:
            return [self.LABELS[text]]
        return [1000 + int.from_bytes(hashlib.md5(w.encode()).digest()[:2], "big") for w in text.split()]


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def labeler(tokenizer: DummyTokenizer) -> Labeler:
    return Labeler(tokenizer)


@pytest.fixture
def settings() -> Settings:
    return Settings(engine="fake", tokenizer="dummy", model_id="jeb-test", api_keys=[])


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(settings: Settings, fake_engine: FakeEngine, tokenizer: DummyTokenizer):
    app = create_app(settings, engine=fake_engine, tokenizer=tokenizer)
    with TestClient(app) as c:
        yield c


TICKET_REQUEST = {
    "state": "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
    "model": "jeb-test",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {"billing": "Payment issues", "technical": "Bugs or integration problems", "sales": None},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated the customer appears",
            "criteria": ["Calm", "Frustrated but civil", "Very angry"],
        },
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?", "criteria": {"true": "Time-sensitive", "false": "No rush"}},
    },
}
