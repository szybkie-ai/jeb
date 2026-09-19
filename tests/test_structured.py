import pytest

from jeb.structured import SchemaError, assemble, compile_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "refund_requested": {"type": "boolean", "description": "Does the customer ask for money back?"},
        "team": {"type": "string", "enum": ["billing", "technical", "sales"], "x-descriptions": {"billing": "Payments, refunds"}},
        "order_id": {"type": "string", "x-candidates": ["A-104", "A-107"], "description": "Which order id does the message refer to?"},
        "anger": {"type": "integer", "minimum": 1, "maximum": 3, "x-levels": ["calm", "annoyed", "furious"]},
        "topics": {"type": "array", "items": {"enum": ["payment", "shipping", "bug"]}, "x-threshold": 0.6},
        "meta": {"type": "object", "properties": {"urgent": {"type": "boolean"}}},
    },
}


def test_compile_covers_the_subset():
    questions, plan = compile_schema(SCHEMA)
    assert questions["refund_requested"]["type"] == "noul"
    assert questions["team"]["criteria"] == {"billing": "Payments, refunds", "technical": None, "sales": None}
    assert questions["order_id"]["type"] == "choice" and list(questions["order_id"]["criteria"]) == ["A-104", "A-107"]
    assert questions["anger"] == {"type": "score", "instructions": "What is `anger`, from 1 to 3?", "criteria": ["calm", "annoyed", "furious"]}
    assert {k for k in questions if k.startswith("topics[")} == {"topics[payment]", "topics[shipping]", "topics[bug]"}
    assert questions["meta.urgent"]["type"] == "noul"
    assert plan["anger"] == {"kind": "integer", "minimum": 1} and plan["topics"]["threshold"] == 0.6


def test_assemble_shapes_the_value():
    _, plan = compile_schema(SCHEMA)
    answers = {
        "refund_requested": {"noul": 0.9}, "team": {"choice": "billing"}, "order_id": {"choice": "A-107"}, "anger": {"score": 1.4},
        "topics[payment]": {"noul": 0.7}, "topics[shipping]": {"noul": 0.2}, "topics[bug]": {"noul": 0.61}, "meta.urgent": {"noul": 0.3},
    }
    assert assemble(plan, answers) == {"refund_requested": True, "team": "billing", "order_id": "A-107", "anger": 2, "topics": ["payment", "bug"], "meta": {"urgent": False}}


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "array"},
        {"type": "object", "properties": {"x": {"type": "string"}}},  # free-form string: would need generation
        {"type": "object", "properties": {"x": {"type": "integer", "minimum": 0, "maximum": 40}}},
        {"type": "object", "properties": {"x": {"type": "string", "enum": ["only"]}}},
        {"type": "object", "properties": {"x": {"type": "number"}}},
    ],
)
def test_unsupported_schemas_are_rejected(bad):
    with pytest.raises(SchemaError):
        compile_schema(bad)


def test_structured_endpoint(client):
    r = client.post("/v1/structured", json={"state": "I was charged twice for order A-104, please refund it now!", "model": "jeb-test", "schema": SCHEMA})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"model", "value", "fields", "usage"}
    v = body["value"]
    assert isinstance(v["refund_requested"], bool) and v["team"] in {"billing", "technical", "sales"} and v["order_id"] in {"A-104", "A-107"}
    assert v["anger"] in {1, 2, 3} and set(v["topics"]) <= {"payment", "shipping", "bug"} and isinstance(v["meta"]["urgent"], bool)
    assert body["fields"]["team"]["type"] == "choice" and "probabilities" in body["fields"]["team"]
    assert body["usage"]["output_tokens"] == 8  # 4 single questions + 3 multi-label nouls + 1 nested
    bad = client.post("/v1/structured", json={"state": "x", "model": "jeb-test", "schema": {"type": "object", "properties": {"name": {"type": "string"}}}})
    assert bad.status_code == 422 and "schema" in bad.json()["detail"]["message"]
