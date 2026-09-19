"""Hindsight labels for the trading gym: at every trigger, replay each entry option through the real price path with the
same frictions and turn the realised results into *soft* targets (softmax of R over a temperature), plus a clarity margin
so noisy decision points can be filtered out. Output rows use the training pipeline's teacher format.

    .venv/bin/python hindsight.py --symbol XAUUSD --bars data/XAUUSD_2019_2022_1m.parquet --start 2019-02-01 --end 2022-12-31 \
        --out data/hindsight_XAUUSD_2019_2022.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from jeb_trading.env import TradingEnv
from jeb_trading.excursions import ATR_MINUTES, HORIZON_MIN, excursion_questions, forecast_state, label_point, label_targets, outlook_state
from jeb_trading.questions import build_questions
from jeb_trading.state import compile_state

STOPS = {"tight": 0.5, "normal": 1.0, "wide": 2.0}
TARGETS = {"1R": 1.0, "2R": 2.0, "3R": 3.0, "trail": None}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def simulate(env: TradingEnv, i: int, side: str, stop_mult: float, target_r: float | None, atr: float, horizon: int = 3 * 1440) -> float:
    """Realised result in R of one entry at bar i, managed by the rule (stop/target or trailing stop), costs included."""
    spec = env.spec
    slip = 0.8 * env.slippage_pips * spec["pip"]  # E|N(0, s)| ~ 0.8 s
    j = min(len(env.closes) - 1, i + horizon)
    hi, lo = env.highs[i + 1:j + 1], env.lows[i + 1:j + 1]
    bid, ask = env.bids[i + 1:j + 1], env.asks[i + 1:j + 1]
    half = env.spread[i + 1:j + 1] / 2
    if len(hi) < 10:
        return float("nan")
    dist = max(stop_mult * atr, 3 * spec["pip"])
    entry = env.asks[i] + slip if side == "long" else env.bids[i] - slip
    if side == "long":
        stop_hit = (lo - half) <= (entry - dist)
        if target_r is None:
            run_max = np.maximum.accumulate(hi)
            trail = np.concatenate([[entry - dist], run_max[:-1] - dist])  # stop for bar t uses highs before t
            trail = np.maximum(trail, entry - dist)
            stop_hit = (lo - half) <= trail
            stop_px = trail
        else:
            stop_px = np.full(len(hi), entry - dist)
            tgt_hit = (hi - half) >= (entry + target_r * dist)
        pnl_per_unit_exit = lambda px: px - entry  # noqa: E731
        end_px = bid[-1] - slip
    else:
        stop_hit = (hi + half) >= (entry + dist)
        if target_r is None:
            run_min = np.minimum.accumulate(lo)
            trail = np.concatenate([[entry + dist], run_min[:-1] + dist])
            trail = np.minimum(trail, entry + dist)
            stop_hit = (hi + half) >= trail
            stop_px = trail
        else:
            stop_px = np.full(len(hi), entry + dist)
            tgt_hit = (lo + half) <= (entry - target_r * dist)
        pnl_per_unit_exit = lambda px: entry - px  # noqa: E731
        end_px = ask[-1] + slip
    t_stop = int(np.argmax(stop_hit)) if stop_hit.any() else None
    t_tgt = None
    if target_r is not None and tgt_hit.any():
        t_tgt = int(np.argmax(tgt_hit))
    if t_stop is not None and (t_tgt is None or t_stop <= t_tgt):  # stop first when both in the same bar
        px = stop_px[t_stop] - slip if side == "long" else stop_px[t_stop] + slip
    elif t_tgt is not None:
        px = entry + target_r * dist if side == "long" else entry - target_r * dist
    else:
        px = end_px
    r = pnl_per_unit_exit(px) / dist
    r -= spec["commission_per_lot"] / (dist * spec["lot_units"])  # commission in R terms (independent of size)
    return float(r)


def soft(scores: dict[str, float], tau: float) -> dict[str, float]:
    m = max(scores.values())
    w = {k: math.exp((v - m) / tau) for k, v in scores.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def presentation(dist: dict[str, float], option_order: list[str]) -> dict:
    """One 'teacher presentation' in the pipeline's format: labels in the question's canonical option order."""
    labels = list(LETTERS[: len(option_order)])
    return {"labels": labels, "targets": option_order, "logprobs": [math.log(max(dist.get(o, 0.0), 1e-6)) for o in option_order]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD"); ap.add_argument("--bars", required=True)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.75, help="softmax temperature in R units")
    ap.add_argument("--trigger", type=float, default=0.15); ap.add_argument("--trigger-minutes", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels", choices=["excursions", "actions"], default="excursions", help="excursions: what the price does next (jeb_trading.excursions); actions: softmax over the realised R of each action")
    ap.add_argument("--horizon", type=int, default=HORIZON_MIN, help="excursion horizon in minutes")
    ap.add_argument("--every", type=int, default=0, help="excursions: a scheduled forecast every N minutes instead of move triggers")
    ap.add_argument("--trigger-atr", type=float, default=0.5, help="excursions: a decision point is every bar whose move over --trigger-minutes is at least this many hourly ATRs (no spacing, no cooldown)")
    a = ap.parse_args()
    bars = pd.read_parquet(a.bars)
    env = TradingEnv(a.symbol, bars, a.start, a.end, trigger_pct=a.trigger, trigger_minutes=a.trigger_minutes)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    n_points = n_rows = 0
    margins = []
    import collections
    firsts = collections.Counter(); levels = {k: collections.Counter() for k in ("up", "down", "close")}
    def hourly_atr_series(n: int = 14) -> np.ndarray:
        """Per-bar hourly ATR (14 completed 60-minute blocks, no lookahead), for the vectorised trigger test."""
        k = len(env.closes) // 60
        h = env.highs[:k * 60].reshape(k, 60).max(1); l = env.lows[:k * 60].reshape(k, 60).min(1); c = env.closes[:k * 60].reshape(k, 60)[:, -1]
        prev = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
        atr_block = pd.Series(tr).rolling(n, min_periods=n).mean().shift(1).to_numpy()  # value known at the start of each block
        per_bar = np.repeat(atr_block, 60)
        return np.concatenate([per_bar, np.full(len(env.closes) - len(per_bar), per_bar[-1] if len(per_bar) else np.nan)])

    def decision_points():
        """(trigger text, bar index): every bar after a significant move (no spacing, no cooldown), or a clock with --every."""
        if a.labels == "excursions":
            end_i = int(env.bars.index.searchsorted(pd.Timestamp(a.end)))
            if a.every > 0:
                i = env.i + a.every
                while i < end_i:
                    env.i = i
                    yield f"scheduled forecast, asked every {a.every} minutes", i
                    i += a.every
                return
            atr_h = hourly_atr_series()
            m = a.trigger_minutes
            move = np.full(len(env.closes), np.nan); move[m:] = env.closes[m:] - env.closes[:-m]
            hit = np.flatnonzero((np.abs(move) >= a.trigger_atr * atr_h) & (np.arange(len(move)) >= env.i) & (np.arange(len(move)) < end_i))
            for i in hit:
                env.i = int(i)
                yield f"{env.symbol} moved {move[i] / atr_h[i]:+.2f} hourly ATR in the last {m} minutes", int(i)
            return
        while True:
            trig = env.next_decision()
            if trig is None:
                return
            yield trig, env.i

    with out.open("w") as f:
        for trig, i in decision_points():
            if a.limit and n_points >= a.limit:
                break
            atr = env.atr(ATR_MINUTES) if a.labels == "excursions" else env.atr(15)
            if not (atr > 0):
                continue
            if a.labels == "excursions":
                lab = label_point(env, i, atr, a.horizon)
                if lab is None:
                    continue
                state = forecast_state(env, atr, a.every, a.horizon, why=trig)
                questions = excursion_questions(state, a.horizon)
                for qid, (dist, order) in label_targets(lab).items():
                    row = {"state": state, "question": questions[qid], "kind": questions[qid]["type"], "qid": qid, "episode": f"{a.symbol}-{env.now:%Y-%m}", "tic": int(i),
                           "hindsight": {"labels": "excursions", "horizon_min": a.horizon, **lab}, "teacher": "hindsight", "teacher_presentations": [presentation(dist, order)], "teacher_in_set_mass": 1.0}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n"); n_rows += 1
                n_points += 1; firsts[lab["first_move"]] += 1; levels["up"][lab["up_level"]] += 1; levels["down"][lab["down_level"]] += 1; levels["close"][lab["close_level"]] += 1
                if n_points % 500 == 0:
                    print(f"{n_points} decision points ({env.now:%Y-%m-%d}), first move {dict(firsts)}", flush=True)
                continue
            R = {(side, s, t): simulate(env, i, side, sm, tr, atr) for side in ("long", "short") for s, sm in STOPS.items() for t, tr in TARGETS.items()}
            if any(math.isnan(v) for v in R.values()):
                continue
            act = {"buy": float(np.mean([v for (sd, _, _), v in R.items() if sd == "long"])), "sell": float(np.mean([v for (sd, _, _), v in R.items() if sd == "short"])), "stay_flat": 0.0}
            best = max(act, key=act.get)
            ranked = sorted(act.values(), reverse=True)
            margin = ranked[0] - ranked[1]
            side = "long" if best == "buy" else "short" if best == "sell" else ("long" if act["buy"] >= act["sell"] else "short")
            stop_scores = {s: float(np.mean([R[(side, s, t)] for t in TARGETS])) for s in STOPS}
            target_scores = {t: float(np.mean([R[(side, s, t)] for s in STOPS])) for t in TARGETS}
            conv_level = min(4.0, max(0.0, margin / 0.5))  # 0 = no edge ... 4 = very strong, one level per 0.5R of clarity
            conv = {str(k): math.exp(-((k - conv_level) ** 2) / 0.5) for k in range(5)}
            z = sum(conv.values()); conv = {k: v / z for k, v in conv.items()}
            state = compile_state(env, trig, with_charts=False)
            questions = build_questions(state)
            rows = {
                "action": (questions["action"], soft(act, a.tau), list(questions["action"]["criteria"])),
                "stop": (questions["stop"], soft(stop_scores, a.tau), list(questions["stop"]["criteria"])),
                "target": (questions["target"], soft(target_scores, a.tau), list(questions["target"]["criteria"])),
                "conviction": (questions["conviction"], conv, [str(k) for k in range(5)]),
            }
            for qid, (q, dist, order) in rows.items():
                row = {"state": state, "question": q, "kind": q["type"], "qid": qid, "episode": f"{a.symbol}-{env.now:%Y-%m}", "tic": int(i),
                       "hindsight": {"action_R": act, "margin": round(margin, 3), "stop_R": stop_scores, "target_R": target_scores},
                       "teacher": "hindsight", "teacher_presentations": [presentation(dist, order)], "teacher_in_set_mass": 1.0}
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); n_rows += 1
            n_points += 1; margins.append(margin)
            if n_points % 500 == 0:
                print(f"{n_points} decision points ({env.now:%Y-%m-%d}), median margin {np.median(margins):.2f}R, clear (>=1R): {np.mean(np.array(margins) >= 1):.0%}", flush=True)
    if a.labels == "excursions":
        print(f"{n_points} decision points, {n_rows} rows -> {out}; first move {dict(firsts)}; up levels {dict(sorted(levels['up'].items()))}; down levels {dict(sorted(levels['down'].items()))}; close levels {dict(sorted(levels['close'].items()))}")
        return
    print(f"{n_points} decision points, {n_rows} rows -> {out}; margin median {np.median(margins):.2f}R, >=1R: {np.mean(np.array(margins) >= 1):.0%}, >=0.5R: {np.mean(np.array(margins) >= 0.5):.0%}")


if __name__ == "__main__":
    main()
