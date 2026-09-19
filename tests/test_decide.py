import json
import math

from jeb.decide import Calibration, Scored, build_answers, choice_confidence, score_confidence, softmax
from jeb.prompt import Plan, build_plans
from jeb.schema import SystemOneRequest
from tests.conftest import TICKET_REQUEST


def lp(*probs: float, mass: float = 1.0) -> list[float]:
    z = sum(probs)
    return [math.log(p / z * mass) for p in probs]


def test_softmax_renormalises_restricted_logprobs():
    probs = softmax(lp(0.6, 0.3, 0.1, mass=0.5))
    assert [round(p, 3) for p in probs] == [0.6, 0.3, 0.1]


def test_confidence_formulas():
    assert choice_confidence([1 / 3] * 3) == 0.0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert round(choice_confidence([0.85, 0.08, 0.07]), 3) == 0.775
    assert score_confidence([0.0, 1.0, 0.0]) == 1.0
    assert round(score_confidence([1 / 3] * 3), 6) == 0.0  # uniform -> 0 (MAD from mode equals uniform MAD)
    assert 0 < score_confidence([0.05, 0.3, 0.65]) < 1


def test_build_answers_averages_permutations_over_option_keys(labeler):
    req = SystemOneRequest.model_validate(TICKET_REQUEST)
    plans, _ = build_plans(req, labeler, permutations=3)
    scored = []
    for p in plans:
        if p.kind == "choice":
            # The engine always likes label "A" (position bias): after rotation the mass spreads over all options.
            scored.append(Scored(p, lp(0.8, 0.1, 0.1)))
        elif p.kind == "score":
            # Reversed presentation flips the levels; a model that always says "A" gives level 0 then level 2.
            scored.append(Scored(p, lp(0.7, 0.2, 0.1)))
        else:
            scored.append(Scored(p, lp(0.9, 0.1, mass=0.8)))
    answers, dbg = build_answers(req, scored, cal=None, debug=True)
    dept = answers["department"]
    assert dept.type == "choice" and set(dept.probabilities) == {"billing", "technical", "sales"}
    assert all(abs(v - 1 / 3) < 1e-3 for v in dept.probabilities.values())  # bias averaged away
    assert dept.confidence < 0.01
    frus = answers["frustration"]
    assert frus.legend == {"0": "Calm", "1": "Frustrated but civil", "2": "Very angry"}
    assert abs(sum(frus.probabilities.values()) - 1) < 1e-3 and 0 <= frus.score <= 2
    assert answers["is_urgent"].noul == 0.9
    assert dbg["is_urgent"]["in_set_mass"] == 0.8 and len(dbg["department"]["permutations"]) == 3
    assert dbg["department"]["prompts"][0].startswith("<|im_start|>system")


def test_calibration_temperature_and_prior(tmp_path):
    cal = Calibration(temperature={"choice": 2.0}, log_prior={"choice": {"A": 1.0}})
    logits = cal.apply("choice", ["A", "B"], [-0.5, -1.5])
    assert logits == [(-0.5 - 1.0) / 2.0, -1.5 / 2.0]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperature": {"noul": 1.5}, "log_prior": {"noul": {"Yes": 0.2}}}))
    loaded = Calibration.load(path)
    assert loaded.temperature == {"noul": 1.5, "choice": 1.0, "score": 1.0} and loaded.log_prior == {"noul": {"Yes": 0.2}}
    assert Calibration.load(None).source == "identity"


def test_noul_calibration_changes_probability():
    plan = Plan(qid="q", kind="noul", prompt_ids=[1], prefix_len=0, label_ids=[9175, 2665], targets=["yes", "no"])
    req = SystemOneRequest.model_validate({"state": "s", "model": "m", "questions": {"q": {"type": "noul", "instructions": "?"}}})
    raw, _ = build_answers(req, [Scored(plan, lp(0.9, 0.1))], cal=None)
    hot, _ = build_answers(req, [Scored(plan, lp(0.9, 0.1))], cal=Calibration(temperature={"noul": 3.0}))
    assert raw["q"].noul == 0.9 and 0.5 < hot["q"].noul < 0.9
