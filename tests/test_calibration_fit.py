"""Fitting on synthetic records with a known position bias and known over-confidence."""

import math
import random

from jeb.calibration_fit import Record, evaluate, fit, fit_priors, fit_temperature, predict, variants
from jeb.decide import Calibration

OPTIONS = ["world", "sports", "business", "sci_tech"]
LETTERS = ["A", "B", "C", "D"]


def synth_choice(n: int, position_bias: float, sharpness: float, seed: int = 0) -> list[Record]:
    """A model that is right with logit `sharpness` on the gold option, plus a bias toward label A, shown in 4 rotations."""
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        gold = rng.randrange(4)
        content = [sharpness if i == gold else rng.gauss(0, 0.7) for i in range(4)]
        perms = []
        for r in range(4):
            order = [(i + r) % 4 for i in range(4)]  # option order[i] sits at label position i
            logits = [content[order[i]] + (position_bias if i == 0 else 0.0) for i in range(4)]
            m = max(logits)
            z = sum(math.exp(x - m) for x in logits)
            logprobs = [x - m - math.log(z) for x in logits]
            perms.append({"labels": LETTERS, "targets": [OPTIONS[j] for j in order], "logprobs": logprobs})
        records.append(Record(kind="choice", gold=OPTIONS[gold], permutations=perms))
    return records


def test_priors_recover_position_bias():
    records = synth_choice(300, position_bias=1.5, sharpness=2.0)
    priors = fit_priors(records, "choice")
    assert abs(priors["A"] - 1.5 * 0.75) < 0.2  # centred: A gets +1.5, the mean over positions is 1.5/4
    assert all(abs(priors[k] - (-1.5 / 4)) < 0.2 for k in "BCD")


def test_temperature_softens_overconfident_model():
    records = synth_choice(300, position_bias=0.0, sharpness=6.0)  # very peaked but only ~right by content
    base = Calibration()
    t = fit_temperature(records, "choice", base)
    raw = evaluate(records, None)
    cal = evaluate(records, Calibration(temperature={"choice": t}))
    assert cal["nll"] <= raw["nll"] + 1e-9
    assert cal["accuracy"] == raw["accuracy"]  # temperature never changes the argmax


def test_fit_and_variants_table():
    records = synth_choice(200, position_bias=1.0, sharpness=1.5)
    cal = fit(records, permutations=4, meta={"note": "synthetic"})
    assert set(cal.log_prior["choice"]) == set(LETTERS) and cal.temperature["choice"] > 0
    assert cal.permutations == 4 and cal.meta["n"] == {"choice": 200}
    table = variants(records, cal)
    assert table["+ temperature"]["nll"] <= table["raw (1 presentation)"]["nll"]
    assert table["averaged presentations"]["accuracy"] >= table["raw (1 presentation)"]["accuracy"] - 0.05
    loaded = Calibration.load(_write(cal))
    assert loaded.temperature["choice"] == cal.temperature["choice"] and loaded.permutations == 4


def test_predict_uses_only_requested_presentations():
    r = synth_choice(1, position_bias=3.0, sharpness=0.5)[0]
    one = predict(r, None, presentations=1)
    four = predict(r, None)
    # With a huge A-bias the single presentation is dominated by whichever option sat at A; averaging spreads it.
    assert max(one.values()) > max(four.values())


def test_noul_records_get_temperature_only():
    rng = random.Random(1)
    recs = []
    for _ in range(100):
        gold = rng.random() < 0.5
        p = 0.99 if gold else 0.03  # over-confident yes/no model, right 90% of the time
        if rng.random() < 0.1:
            p = 1 - p
        recs.append(Record(kind="noul", gold="yes" if gold else "no", permutations=[{"labels": ["Yes", "No"], "targets": ["yes", "no"], "logprobs": [math.log(p), math.log(1 - p)]}]))
    cal = fit(recs)
    assert cal.log_prior["noul"] == {} and cal.temperature["noul"] > 1.0


def _write(cal: Calibration) -> str:
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.write(cal.to_json())
    f.close()
    return f.name
