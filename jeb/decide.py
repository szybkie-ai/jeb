"""From label log-probabilities to typed answers.

The engine returns, for each plan, the log-probability of every label token at the answer
position (log-softmax over the full vocabulary). Renormalising over the labels is the
restricted softmax; the mass the labels had before renormalising ("in-set mass") tells us
whether the model followed the format at all.

Calibration is applied on the label logits: a temperature per primitive and an optional
per-position log-prior (letter/position bias), fitted offline (`jeb.calibration_fit`) and
shipped with a model manifest as `calibration.json`.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jeb.prompt import LETTERS, NO, YES, Plan
from jeb.schema import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, ScoreQuestion, SystemOneRequest


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def position_labels(kind: str, n: int) -> list[str]:
    """The names calibration priors are keyed by: letters for choice, positions for score, Yes/No for noul."""
    if kind == "noul":
        return [YES, NO]
    if kind == "choice":
        return list(LETTERS[:n])
    return [f"pos{i}" for i in range(n)]


@dataclass
class Calibration:
    """Per-primitive temperature and per-position log-prior corrections (identity by default)."""

    temperature: dict[str, float] = field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})
    log_prior: dict[str, dict[str, float]] = field(default_factory=dict)  # kind -> position label -> log prior to subtract
    permutations: int | None = None  # recommended option orderings per question, if the fit used them
    source: str = "identity"
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> Calibration:
        if not path:
            return cls()
        d = json.loads(Path(path).read_text())
        return cls(
            temperature={**cls().temperature, **d.get("temperature", {})},
            log_prior=d.get("log_prior", {}),
            permutations=d.get("permutations"),
            source=str(path),
            meta=d.get("meta", {}),
        )

    def to_json(self) -> str:
        return json.dumps(
            {"temperature": self.temperature, "log_prior": self.log_prior, "permutations": self.permutations, "meta": self.meta},
            indent=2,
        )

    def apply(self, kind: str, labels: list[str], logprobs: list[float]) -> list[float]:
        t = self.temperature.get(kind, 1.0)
        prior = self.log_prior.get(kind, {})
        return [(lp - prior.get(lab, 0.0)) / t for lab, lp in zip(labels, logprobs)]


# --------------------------------------------------------------------------- confidence


def choice_confidence(probs: list[float]) -> float:
    """Peak probability rescaled so uniform -> 0 and certain -> 1."""
    n = len(probs)
    if n == 1:
        return 1.0
    u = 1.0 / n
    return max(0.0, (max(probs) - u) / (1.0 - u))


def score_confidence(probs: list[float]) -> float:
    """Concentration around the modal level: 1 - MAD-from-mode / MAD of the uniform distribution."""
    n = len(probs)
    if n == 1:
        return 1.0
    mode = max(range(n), key=probs.__getitem__)
    mad = sum(p * abs(i - mode) for i, p in enumerate(probs))
    center = (n - 1) / 2
    uniform_mad = sum(abs(i - center) for i, _ in enumerate(probs)) / n
    return max(0.0, 1.0 - mad / uniform_mad)


def confidence(kind: str, probs_by_target: dict[str, float]) -> float:
    if kind == "noul":
        return abs(2 * probs_by_target["yes"] - 1)
    if kind == "choice":
        return choice_confidence(list(probs_by_target.values()))
    return score_confidence([probs_by_target[str(i)] for i in range(len(probs_by_target))])


# --------------------------------------------------------------------------- combination


def distribution(kind: str, labels: list[str], targets: list[str], logprobs: list[float], cal: Calibration | None) -> dict[str, float]:
    """Target -> probability for one presentation, after optional calibration on the label logits."""
    logits = cal.apply(kind, labels, logprobs) if cal else logprobs
    return dict(zip(targets, softmax(logits)))


def combine(dists: list[dict[str, float]]) -> dict[str, float]:
    """Average presentations over targets (option keys / level indices / yes-no) and renormalise."""
    keys = list(dists[0])
    avg = {k: sum(d[k] for d in dists) / len(dists) for k in keys}
    z = sum(avg.values())
    return {k: v / z for k, v in avg.items()}


@dataclass
class Scored:
    plan: Plan
    logprobs: list[float]  # aligned with plan.label_ids


def build_answers(req: SystemOneRequest, scored: list[Scored], cal: Calibration | None, debug: bool = False) -> tuple[dict[str, Answer], dict[str, Any]]:
    by_q: dict[str, list[Scored]] = defaultdict(list)
    for s in scored:
        by_q[s.plan.qid].append(s)

    answers: dict[str, Answer] = {}
    dbg: dict[str, Any] = {}
    for qid, q in req.questions.items():
        group = by_q[qid]
        kind = group[0].plan.kind
        pres = [(position_labels(kind, len(s.plan.label_ids)), s) for s in group]
        dists = [distribution(kind, labels, s.plan.targets, s.logprobs, cal) for labels, s in pres]
        masses = [sum(math.exp(lp) for lp in s.logprobs) for s in group]
        avg = combine(dists)

        if kind == "noul":
            answers[qid] = NoulAnswer(noul=round(avg["yes"], 4))
        elif kind == "choice":
            best = max(avg, key=avg.get)
            answers[qid] = ChoiceAnswer(
                choice=best,
                probabilities={k: round(v, 4) for k, v in avg.items()},
                confidence=round(choice_confidence(list(avg.values())), 4),
            )
        else:
            assert isinstance(q, ScoreQuestion)
            n = len(q.criteria)
            ordered = [avg[str(i)] for i in range(n)]
            answers[qid] = ScoreAnswer(
                score=round(sum(i * p for i, p in enumerate(ordered)), 4),
                legend={str(i): lvl for i, lvl in enumerate(q.criteria)},
                probabilities={str(i): round(p, 4) for i, p in enumerate(ordered)},
                confidence=round(score_confidence(ordered), 4),
            )
        if debug:
            dbg[qid] = {
                "kind": kind,
                "in_set_mass": round(sum(masses) / len(masses), 4),
                "permutations": [
                    {"labels": labels, "targets": s.plan.targets, "logprobs": [round(x, 5) for x in s.logprobs], "probabilities": [round(p, 4) for p in d.values()]}
                    for (labels, s), d in zip(pres, dists)
                ],
                "prompts": [s.plan.text for s in group],
            }
    return answers, dbg
