"""Wire compatibility: the upstream `typesafe-sdk` client must be able to talk to JEB unchanged."""

import socket
import threading
import time

import pytest
import uvicorn

from jeb.engines import FakeEngine
from jeb.server import Settings, create_app
from tests.conftest import DummyTokenizer

typesafe_sdk = pytest.importorskip("typesafe_sdk")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    app = create_app(Settings(engine="fake", tokenizer="dummy", model_id="jeb-test", api_keys=["secret"]), engine=FakeEngine(), tokenizer=DummyTokenizer())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=5)


def test_typesafe_sdk_round_trip(base_url):
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    with TypeSafeClient(api_key="secret", base_url=base_url, model="jeb-test") as client:
        resp = client.system_one(
            state={"ticket": "My card was charged twice.", "policy": "Duplicate charges are refunded."},
            questions={
                "refund": Noul(instructions="Does `ticket` ask for a refund?"),
                "team": Choice(instructions="Which team?", criteria={"billing": "money", "technical": "bugs"}),
                "anger": Score(instructions="How angry?", criteria=["calm", "annoyed", "furious"]),
            },
        )
        assert resp.model == "jeb-test"
        assert 0 <= resp.nouls["refund"].noul <= 1
        assert resp.choices["team"].choice in {"billing", "technical"}
        assert resp.scores["anger"].legend == {0: "calm", 1: "annoyed", 2: "furious"}
        assert resp.usage.input_tokens and resp.usage.output_tokens == 3
        assert client.models.list().models[0].name == "jeb-test"


def test_typesafe_sdk_auth_error(base_url):
    from typesafe_sdk import Noul, TypeSafeAPIError, TypeSafeClient

    with TypeSafeClient(api_key="wrong", base_url=base_url, model="jeb-test") as client:
        with pytest.raises(TypeSafeAPIError):
            client.system_one(state="x", questions={"q": Noul(instructions="?")})
