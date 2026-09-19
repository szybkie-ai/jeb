import copy

from fastapi.testclient import TestClient

from jeb.engines import FakeEngine
from jeb.server import Settings, create_app
from tests.conftest import TICKET_REQUEST, DummyTokenizer


def test_systemone_shape(client):
    r = client.post("/v1/systemone", json=TICKET_REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "jeb-test" and set(body) == {"model", "answers", "usage"}  # no debug by default
    a = body["answers"]
    assert a["department"]["type"] == "choice" and a["department"]["choice"] in a["department"]["probabilities"]
    assert abs(sum(a["department"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["department"]["confidence"] <= 1
    assert a["frustration"]["type"] == "score" and set(a["frustration"]["legend"]) == {"0", "1", "2"} and 0 <= a["frustration"]["score"] <= 2
    assert a["is_urgent"]["type"] == "noul" and 0 <= a["is_urgent"]["noul"] <= 1
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 3


def test_usage_counts_prefix_once(client):
    one = copy.deepcopy(TICKET_REQUEST)
    one["questions"] = {"is_urgent": one["questions"]["is_urgent"]}
    a = client.post("/v1/systemone", json=one).json()["usage"]["input_tokens"]
    b = client.post("/v1/systemone", json=TICKET_REQUEST).json()["usage"]["input_tokens"]
    assert a < b < 3 * a  # three questions cost less than three separate prompts


def test_debug_and_options(client, fake_engine):
    req = {**TICKET_REQUEST, "options": {"permutations": 3, "debug": True, "calibration": "raw"}}
    body = client.post("/v1/systemone", json=req).json()
    assert body["debug"]["_engine"]["prompts_sent"] == 3 + 3 + 1  # choice x3 rotations, score x3 (normal, reversed, shifted), noul x1
    assert 0 < body["debug"]["department"]["in_set_mass"] <= 1
    assert len(fake_engine.calls[-1]) == 7


def test_model_aliases_and_unknown_model(client):
    for m in ("jeb-latest", "openjev-latest", "jev-latest"):
        assert client.post("/v1/systemone", json={**TICKET_REQUEST, "model": m}).status_code == 200
    r = client.post("/v1/systemone", json={**TICKET_REQUEST, "model": "gpt-9"})
    assert r.status_code == 422 and r.json()["detail"]["error_type"] == "invalid_request_error"


def test_validation_errors(client):
    bad = copy.deepcopy(TICKET_REQUEST)
    bad["questions"]["department"]["criteria"] = {"only": "one option"}
    assert client.post("/v1/systemone", json=bad).status_code == 422
    bad = copy.deepcopy(TICKET_REQUEST)
    del bad["questions"]["is_urgent"]["instructions"]
    assert client.post("/v1/systemone", json=bad).status_code == 422
    bad = copy.deepcopy(TICKET_REQUEST)
    bad["questions"]["frustration"]["criteria"] = ["one level"]
    assert client.post("/v1/systemone", json=bad).status_code == 422
    assert client.post("/v1/systemone", json={**TICKET_REQUEST, "questions": {}}).status_code == 422


def test_api_keys():
    s = Settings(engine="fake", tokenizer="dummy", model_id="jeb-test", api_keys=["k1"])
    with TestClient(create_app(s, engine=FakeEngine(), tokenizer=DummyTokenizer())) as c:
        r = c.post("/v1/systemone", json=TICKET_REQUEST)
        assert r.status_code == 403 and r.json()["detail"]["error_type"] == "authentication_error"
        assert c.post("/v1/systemone", json=TICKET_REQUEST, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.post("/v1/systemone", json=TICKET_REQUEST, headers={"Authorization": "Bearer k1"}).status_code == 200
        assert c.get("/healthz").status_code == 200  # health is unauthenticated


def test_models_health_metrics(client):
    m = client.get("/v1/models").json()
    assert m["models"][0]["name"] == "jeb-test" and set(m["models"][0]) == {"name", "description", "release_date"}
    assert client.get("/healthz").json()["ok"] is True
    client.post("/v1/systemone", json=TICKET_REQUEST)
    assert "jeb_questions_total" in client.get("/metrics").text
