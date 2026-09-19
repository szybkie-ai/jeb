"""Fit a `Calibration` from recorded evaluations (no model training involved).

Input records are what `jeb eval` writes: one line per item with the raw label
log-probabilities of every presentation (permutation) and the gold target:

    {"kind": "choice", "gold": "sports",
     "permutations": [{"labels": ["A","B","C","D"], "targets": ["world","sports","business","sci_tech"],
                       "logprobs": [-3.1, -0.05, -4.0, -5.2]}, ...]}

Two corrections, in this order:

1. **Position priors** (unsupervised). Under cyclic rotations every option visits every label
   position equally often, so the per-position mean log-probability isolates the model's
   preference for a *position* (e.g. "A") from its preference for content. We subtract that
   mean (centred) from the logits. Nouls have no rotations; their yes/no bias is fitted with
   the temperature instead.
2. **Temperature** (supervised). One scalar per primitive minimising the NLL of the gold
   target under the permutation-averaged distribution; > 1 softens an over-confident model.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from jeb.decide import Calibration, combine, confidence, distribution


@dataclass
class Record:
    kind: str
    gold: str
    permutations: list[dict[str, Any]]  # labels, targets, logprobs
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Record:
        return cls(kind=d["kind"], gold=str(d["gold"]), permutations=d["permutations"], extra={k: v for k, v in d.items() if k not in ("kind", "gold", "permutations")})


def predict(r: Record, cal: Calibration | None, presentations: int | None = None) -> dict[str, float]:
    """Permutation-averaged (optionally calibrated) target distribution for one record."""
    pres = r.permutations if presentations is None else r.permutations[:presentations]
    return combine([distribution(r.kind, p["labels"], p["targets"], p["logprobs"], cal) for p in pres])


# --------------------------------------------------------------------------- fitting


def fit_priors(records: list[Record], kind: str) -> dict[str, float]:
    """Centred per-position mean log-probability; empty when the records carry no rotations."""
    rows = [r for r in records if r.kind == kind]
    if kind == "noul" or not rows or all(len(r.permutations) < 2 for r in rows):
        return {}
    sums: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for p in r.permutations:
            for lab, lp in zip(p["labels"], p["logprobs"]):
                sums[lab].append(lp)
    means = {lab: statistics.fmean(v) for lab, v in sums.items()}
    center = statistics.fmean(means.values())
    return {lab: round(m - center, 5) for lab, m in means.items()}


def nll(records: list[Record], cal: Calibration | None) -> float:
    return -statistics.fmean(math.log(max(predict(r, cal)[r.gold], 1e-12)) for r in records)


def fit_temperature(records: list[Record], kind: str, base: Calibration, lo: float = 0.2, hi: float = 10.0) -> float:
    """Golden-section search of the NLL over log-temperature."""
    rows = [r for r in records if r.kind == kind]
    if not rows:
        return 1.0

    def loss(log_t: float) -> float:
        cal = Calibration(temperature={**base.temperature, kind: math.exp(log_t)}, log_prior=base.log_prior)
        return nll(rows, cal)

    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
    return round(math.exp((a + b) / 2), 4)


def fit(records: list[Record], permutations: int | None = None, meta: dict[str, Any] | None = None) -> Calibration:
    kinds = sorted({r.kind for r in records})
    cal = Calibration(source="fit", permutations=permutations, meta=meta or {})
    for k in kinds:
        cal.log_prior[k] = fit_priors(records, k)
    for k in kinds:
        cal.temperature[k] = fit_temperature(records, k, cal)
    cal.meta["n"] = {k: sum(1 for r in records if r.kind == k) for k in kinds}
    return cal


# --------------------------------------------------------------------------- evaluation


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    """Expected calibration error over (predicted-class probability, correct) pairs, plus the reliability table."""
    total, table = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [(p, c) for p, c in pairs if (lo < p <= hi) or (b == 0 and p == 0)]
        if not in_bin:
            continue
        conf = statistics.fmean(p for p, _ in in_bin)
        acc = statistics.fmean(c for _, c in in_bin)
        total += len(in_bin) / len(pairs) * abs(acc - conf)
        table.append({"bin": f"({lo:.1f}, {hi:.1f}]", "n": len(in_bin), "mean_p": round(conf, 3), "accuracy": round(acc, 3)})
    return total, table


def evaluate(records: list[Record], cal: Calibration | None, presentations: int | None = None, thresholds=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95)) -> dict[str, Any]:
    """Accuracy, ECE, Brier, NLL and the coverage-vs-confidence curve; MAE of the expected level for scores."""
    pairs, brier, logs, confs, correct, abs_err = [], [], [], [], [], []
    for r in records:
        probs = predict(r, cal, presentations)
        pred = max(probs, key=probs.get)
        ok = pred == r.gold
        pairs.append((probs[pred], ok))
        brier.append((1 - probs[r.gold]) ** 2)
        logs.append(math.log(max(probs[r.gold], 1e-12)))
        confs.append(confidence(r.kind, probs))
        correct.append(ok)
        if r.kind == "score":
            expected = sum(int(k) * p for k, p in probs.items())
            abs_err.append(abs(expected - int(r.gold)))
    e, table = ece(pairs)
    out: dict[str, Any] = {
        "n": len(records),
        "accuracy": round(statistics.fmean(correct), 4),
        "ece": round(e, 4),
        "brier": round(statistics.fmean(brier), 4),
        "nll": round(-statistics.fmean(logs), 4),
        "reliability": table,
        "coverage": [
            {
                "confidence>=": t,
                "coverage": round(sum(1 for c in confs if c >= t) / len(confs), 3),
                "accuracy": (round(statistics.fmean(ok for c, ok in zip(confs, correct) if c >= t), 3) if any(c >= t for c in confs) else None),
            }
            for t in thresholds
        ],
    }
    if abs_err:
        out["score_mae"] = round(statistics.fmean(abs_err), 4)
    return out


def variants(records: list[Record], cal: Calibration) -> dict[str, dict[str, Any]]:
    """The ablation table: raw single presentation → averaged → +priors → +temperature."""
    kinds = {r.kind for r in records}
    priors_only = Calibration(log_prior=cal.log_prior)
    return {
        "raw (1 presentation)": evaluate(records, None, presentations=1),
        "averaged presentations": evaluate(records, None),
        "+ position priors": evaluate(records, priors_only),
        "+ temperature": evaluate(records, cal),
        **({"temperature only": evaluate(records, Calibration(temperature=cal.temperature))} if kinds - {"noul"} else {}),
    }
