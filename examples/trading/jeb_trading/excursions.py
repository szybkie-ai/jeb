"""Price-movement labels and the strategy derived from them (the private trading model's objective).

Instead of asking which action to take, the model is asked what the price will do over the next `HORIZON_MIN` minutes,
in units of the 1-hour ATR: which 2-ATR move comes first, how far the highest and lowest prices get, and where the
price ends up. The labels are the realised values, one-hot over ordered levels, so across many similar states the model
learns P(level | state): a real conditional distribution, not a softmax over one path. Side, stop, target and whether to
trade at all are then computed in code from the predicted distributions (`Strategy`), and can be tuned without retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HORIZON_MIN = 480  # 8 hours: intraday swing, what the 15-minute and 4-hour panels can speak to
ATR_MINUTES = 60  # the unit: 1-hour ATR over 14 bars (~14 hours); the 15-minute ATR collapses at night and inflates every level
FIRST_MOVE_ATR = 2.0  # at the 8-hour horizon 1 ATR is reached almost always within minutes; 2 ATR splits up/down/neither about 40/40/20
EXCURSION_EDGES = [0.5, 1.0, 2.0, 3.0, 5.0]  # ATR units, 6 levels
CLOSE_EDGES = [-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]  # signed ATR units, 7 levels
EXCURSION_LEVELS = ["less than 0.5 ATR", "0.5 to 1 ATR", "1 to 2 ATR", "2 to 3 ATR", "3 to 5 ATR", "more than 5 ATR"]
CLOSE_LEVELS = ["more than 3 ATR lower", "1 to 3 ATR lower", "0.3 to 1 ATR lower", "within 0.3 ATR of the current price",
                "0.3 to 1 ATR higher", "1 to 3 ATR higher", "more than 3 ATR higher"]
FIRST_MOVE = {"up_first": "Price rises 2 ATR above the current level before it falls 2 ATR below it",
              "down_first": "Price falls 2 ATR below the current level before it rises 2 ATR above it",
              "neither": "Neither a 2 ATR rise nor a 2 ATR fall happens within the horizon"}
QIDS = ("first_move", "up_move", "down_move", "close_move")
# representative value per level when no data-derived table is given (data/level_values.json from level_values.py)
DEFAULT_VALUES = {"up_move": [0.25, 0.75, 1.5, 2.5, 4.0, 6.5], "down_move": [0.25, 0.75, 1.5, 2.5, 4.0, 6.5],
                  "close_move": [-4.0, -2.0, -0.65, 0.0, 0.65, 2.0, 4.0]}


def level_of(x: float, edges: list[float]) -> int:
    return int(sum(x >= e for e in edges))


def outlook_state(state: dict[str, Any], atr: float, digits: int, horizon: int = HORIZON_MIN) -> dict[str, Any]:
    """The state the questions refer to: the horizon and what 1 ATR is in price units (same at labelling and inference)."""
    return {**state, "outlook": {"horizon": f"the next {horizon // 60} hours", "one_atr": f"{atr:.{digits}f} price units (1-hour ATR, 14 bars)"}}


def forecast_state(env, atr: float, every: int, horizon: int = HORIZON_MIN, why: str | None = None) -> dict[str, Any]:
    """The market-only state of a scheduled forecast: no account, no position, no trading rules. The forecaster is asked
    every `every` minutes whatever the account is doing; positions, cooldowns and triggers belong to the strategy layer.
    The recent moves are also given in ATR units, so a burst is visible on the same scale as the questions."""
    from jeb_trading.state import compile_state
    base = compile_state(env, why or f"scheduled forecast, asked every {every} minutes", with_charts=False)
    market = dict(base["market"])
    close = float(env.closes[env.i])
    if atr > 0:
        market["move_atr"] = {k: round(v * close / 100.0 / atr, 2) for k, v in market["move_pct"].items() if v is not None}
    state = {"instrument": base["rules"]["instrument"], "time": base["time"], "why_now": base["trigger"], "market": market}
    return outlook_state(state, atr, env.spec["digits"], horizon)


def excursion_questions(state: dict[str, Any], horizon: int = HORIZON_MIN) -> dict[str, Any]:
    h = f"{horizon // 60} hours"
    return {
        "first_move": {"type": "choice", "instructions": f"Over {h}, which happens first: a rise of 2 ATR above the current price, or a fall of 2 ATR below it? (1 ATR is given in `outlook`.)", "criteria": dict(FIRST_MOVE)},
        "up_move": {"type": "score", "instructions": f"How far above the current price will the highest price of the next {h} be, in ATR units?", "criteria": list(EXCURSION_LEVELS)},
        "down_move": {"type": "score", "instructions": f"How far below the current price will the lowest price of the next {h} be, in ATR units?", "criteria": list(EXCURSION_LEVELS)},
        "close_move": {"type": "score", "instructions": f"Where will the price be at the end of the next {h}, relative to the current price, in ATR units?", "criteria": list(CLOSE_LEVELS)},
    }


def label_point(env, i: int, atr: float, horizon: int = HORIZON_MIN) -> dict[str, Any] | None:
    """Realised movement after bar i (mid prices): highest/lowest excursion, close, and which 1-ATR move came first."""
    end = i + horizon
    if not atr > 0 or end >= len(env.closes):
        return None
    p0 = float(env.closes[i])
    hi, lo = env.highs[i + 1:end + 1], env.lows[i + 1:end + 1]
    up = (float(hi.max()) - p0) / atr
    down = (p0 - float(lo.min())) / atr
    close = (float(env.closes[end]) - p0) / atr
    up_hit = np.flatnonzero(hi >= p0 + FIRST_MOVE_ATR * atr)
    dn_hit = np.flatnonzero(lo <= p0 - FIRST_MOVE_ATR * atr)
    fu = int(up_hit[0]) if len(up_hit) else None
    fd = int(dn_hit[0]) if len(dn_hit) else None
    if fu is None and fd is None:
        first = "neither"
    elif fd is None or (fu is not None and fu < fd):
        first = "up_first"
    elif fu is None or fd < fu:
        first = "down_first"
    else:  # both in the same one-minute bar: the bar's direction decides
        opens = getattr(env, "opens", None)
        if opens is None:
            opens = env.opens = env.bars["open"].to_numpy()
        j = i + 1 + fu
        first = "up_first" if env.closes[j] >= opens[j] else "down_first"
    return {"up": round(up, 3), "down": round(down, 3), "close": round(close, 3), "first_move": first,
            "up_level": level_of(up, EXCURSION_EDGES), "down_level": level_of(down, EXCURSION_EDGES), "close_level": level_of(close, CLOSE_EDGES)}


def label_targets(lab: dict[str, Any]) -> dict[str, tuple[dict[str, float], list[str]]]:
    """qid -> (one-hot target over the question's options, canonical option order)."""
    lv = lambda k, n: {str(j): float(j == k) for j in range(n)}  # noqa: E731
    return {"first_move": ({k: float(k == lab["first_move"]) for k in FIRST_MOVE}, list(FIRST_MOVE)),
            "up_move": (lv(lab["up_level"], len(EXCURSION_LEVELS)), [str(j) for j in range(len(EXCURSION_LEVELS))]),
            "down_move": (lv(lab["down_level"], len(EXCURSION_LEVELS)), [str(j) for j in range(len(EXCURSION_LEVELS))]),
            "close_move": (lv(lab["close_level"], len(CLOSE_LEVELS)), [str(j) for j in range(len(CLOSE_LEVELS))])}


def load_values(path: str | Path | None) -> dict[str, list[float]]:
    if path and Path(path).exists():
        return {**DEFAULT_VALUES, **json.loads(Path(path).read_text())}
    return dict(DEFAULT_VALUES)


def estimates(answers: dict[str, Any], values: dict[str, list[float]]) -> dict[str, Any]:
    """What the answers imply: direction probabilities, expected excursions and close, tail probabilities at the level edges."""
    fm = answers["first_move"]["probabilities"]

    def dist(qid: str) -> list[float]:
        pr = answers[qid]["probabilities"]
        p = [float(pr.get(str(k), 0.0)) for k in range(len(values[qid]))]
        z = sum(p) or 1.0
        return [x / z for x in p]

    up, down, close = dist("up_move"), dist("down_move"), dist("close_move")
    ev = lambda p, v: float(sum(pi * vi for pi, vi in zip(p, v)))  # noqa: E731
    tail = lambda p, edges: {e: float(sum(p[k] for k in range(len(p)) if k >= level_of(e, edges))) for e in edges}  # noqa: E731
    return {"p_up_first": float(fm.get("up_first", 0.0)), "p_down_first": float(fm.get("down_first", 0.0)), "p_neither": float(fm.get("neither", 0.0)),
            "e_up": ev(up, values["up_move"]), "e_down": ev(down, values["down_move"]), "e_close": ev(close, values["close_move"]),
            "p_up_ge": tail(up, EXCURSION_EDGES), "p_down_ge": tail(down, EXCURSION_EDGES)}


@dataclass
class Strategy:
    """Side, stop and target from the predicted movement; trade only when the implied expected R clears `theta`."""
    delta: float = 0.15  # required margin between P(up first) and P(down first)
    theta: float = 0.2  # minimum expected R of the chosen stop/target
    max_adverse_p: float = 0.35  # the stop is the tightest one the adverse excursion clears with at most this probability
    min_target_p: float = 0.45  # the target is the farthest one the favourable excursion reaches with at least this probability
    stops: tuple[float, ...] = (0.5, 1.0, 2.0)
    targets: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
    risk_pct: float = 1.0
    horizon: int = HORIZON_MIN
    tuned: dict[str, Any] = field(default_factory=dict)

    def decide(self, est: dict[str, Any]) -> dict[str, Any]:
        d = est["p_up_first"] - est["p_down_first"]
        if d >= self.delta and est["e_close"] > 0:
            side = "long"
        elif -d >= self.delta and est["e_close"] < 0:
            side = "short"
        else:
            return {"action": "stay_flat", "why": f"direction margin {d:+.2f}, expected close {est['e_close']:+.2f} ATR"}
        fav, adv = (est["p_up_ge"], est["p_down_ge"]) if side == "long" else (est["p_down_ge"], est["p_up_ge"])
        stop = next((s for s in self.stops if adv[s] <= self.max_adverse_p), self.stops[-1])
        target = max((t for t in self.targets if fav[t] >= self.min_target_p and t > stop), default=None)
        if target is None:
            return {"action": "stay_flat", "why": f"{side}: no target beyond a {stop} ATR stop with P>={self.min_target_p}"}
        # P(target before stop): the target is reached and the stop is not, or both are reached and the favourable move came
        # first, with the first-move answer as the ordering prior. A heuristic on marginals; the gym measures the truth.
        p_fav_first = est["p_up_first"] if side == "long" else est["p_down_first"]
        p_adv_first = est["p_down_first"] if side == "long" else est["p_up_first"]
        order = p_fav_first / max(p_fav_first + p_adv_first, 1e-6)
        p_win = fav[target] * (1.0 - adv[stop]) + fav[target] * adv[stop] * order
        exp_r = p_win * (target / stop) - (1.0 - p_win)
        if exp_r < self.theta:
            return {"action": "stay_flat", "why": f"{side}: expected {exp_r:+.2f}R below {self.theta}R"}
        return {"action": "buy" if side == "long" else "sell", "side": side, "stop_atr": stop, "target_atr": target, "target_r": round(target / stop, 2),
                "p_win": round(p_win, 3), "expected_r": round(exp_r, 3), "direction_margin": round(d, 3)}


def apply_excursion(env, state: dict[str, Any], answers: dict[str, Any] | None, strategy: Strategy, values: dict[str, list[float]], atr: float) -> dict[str, Any]:
    """Flat: open per the strategy. In a position: close at the horizon, or when a fresh forecast turns against the
    position by the entry margin; otherwise hold to the stop or the target."""
    pos = env.account.position
    est = estimates(answers, values) if answers else None
    summary = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in est.items() if not isinstance(v, dict)} if est else None
    if pos is None:
        if est is None:
            return {"action": "stay_flat", "why": "no forecast", "result": "stayed flat"}
        dec = strategy.decide(est)
        dec["estimates"] = summary
        if dec["action"] in ("buy", "sell"):
            dec["result"] = env.open(dec["side"], strategy.risk_pct, dec["stop_atr"], dec["target_r"], atr, reason=state.get("why_now", ""))
        else:
            dec["result"] = "stayed flat"
        return dec
    held = int((env.now - pos.opened).total_seconds() // 60)
    if held >= strategy.horizon:
        return {"action": "close", "why": f"horizon reached ({held} min)", "result": env.close("horizon reached"), "estimates": summary}
    if est is not None:
        against = (est["p_down_first"] - est["p_up_first"]) if pos.side == "long" else (est["p_up_first"] - est["p_down_first"])
        close_against = est["e_close"] < 0 if pos.side == "long" else est["e_close"] > 0
        if against >= strategy.delta and close_against:
            return {"action": "close", "why": f"forecast turned against the {pos.side} (margin {against:+.2f})", "result": env.close("forecast reversed"), "estimates": summary}
    return {"action": "hold", "result": "held", "estimates": summary}
