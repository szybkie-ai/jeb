"""Run a policy through the trading gym.

    .venv/bin/python run.py --policy momentum --symbol XAUUSD --start 2024-01-02 --end 2024-03-31
    .venv/bin/python run.py --url http://127.0.0.1:8020 --images --symbol XAUUSD --start 2024-01-02 --end 2024-01-31 --log out/stu-xau/decisions.jsonl
    .venv/bin/python run.py --url http://127.0.0.1:8022 --model teacher --images --collect ...     # teacher play = labelled data
    TYPESAFE_API_KEY=... .venv/bin/python run.py --url https://api.typesafe.ai --model jev-latest --plain ...   # comparison only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from jeb_trading.charts import charts
from jeb_trading.env import TradingEnv
from jeb_trading.policies import MomentumPolicy, SystemOnePolicy
from jeb_trading.questions import apply_answers, build_questions
from jeb_trading.state import compile_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--bars", default=None, help="1m bars parquet (default data/<symbol>_1m.parquet)")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--policy", default="jeb", choices=["jeb", "momentum", "excursion"], help="excursion: the model predicts the price movement, code derives side/stop/target (jeb_trading.excursions)")
    ap.add_argument("--theta", type=float, default=0.2, help="excursion: minimum expected R to trade"); ap.add_argument("--delta", type=float, default=0.15, help="excursion: required P(up first) - P(down first) margin")
    ap.add_argument("--horizon", type=int, default=480, help="excursion: minutes; positions are closed at the horizon"); ap.add_argument("--level-values", default="data/level_values.json"); ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=60, help="excursion: ask the forecaster every N minutes, flat or not (no move trigger, no cooldown)")
    ap.add_argument("--url", default="http://127.0.0.1:8020"); ap.add_argument("--key", default=os.environ.get("JEB_API_KEY") or os.environ.get("TYPESAFE_API_KEY"))
    ap.add_argument("--model", default="jeb-latest"); ap.add_argument("--permutations", type=int, default=2)
    ap.add_argument("--images", action="store_true", help="attach the three charts to every request")
    ap.add_argument("--collect", action="store_true", help="store raw per-presentation logprobs (teacher play as training data)")
    ap.add_argument("--plain", action="store_true", help="bare System One requests (the original API)")
    ap.add_argument("--trigger", type=float, default=0.15, help="entry trigger: |move| over --trigger-minutes in percent")
    ap.add_argument("--trigger-minutes", type=int, default=15)
    ap.add_argument("--manage-minutes", type=int, default=30)
    ap.add_argument("--max-decisions", type=int, default=100000)
    ap.add_argument("--log", default=None, help="decisions.jsonl (charts saved next to it)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bars = pd.read_parquet(a.bars or f"data/{a.symbol}_1m.parquet")
    if a.policy == "excursion":
        env = TradingEnv(a.symbol, bars, a.start, a.end, trigger_pct=0.0, trigger_minutes=a.every, manage_minutes=a.every, cooldown_minutes=0, seed=a.seed)
    else:
        env = TradingEnv(a.symbol, bars, a.start, a.end, trigger_pct=a.trigger, trigger_minutes=a.trigger_minutes, manage_minutes=a.manage_minutes, seed=a.seed)
    policy = MomentumPolicy() if a.policy == "momentum" else SystemOnePolicy(a.url, a.key, a.model, a.permutations, collect=a.collect, images=a.images, plain=a.plain)
    if a.policy == "excursion":
        from jeb_trading.excursions import ATR_MINUTES, Strategy, apply_excursion, excursion_questions, forecast_state, load_values
        strategy = Strategy(delta=a.delta, theta=a.theta, risk_pct=a.risk, horizon=a.horizon); values = load_values(a.level_values); policy.name = "excursion"
    log_path = Path(a.log) if a.log else None
    logf = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True); (log_path.parent / "charts").mkdir(exist_ok=True)
        logf = log_path.open("a")
    t_start = time.perf_counter()
    while env.decisions < a.max_decisions:
        trigger = env.next_decision()
        if trigger is None:
            break
        state = compile_state(env, trigger, with_charts=a.images or bool(log_path))
        pos = env.account.position
        imgs = charts(env.bars, env.i, {"entry": pos.entry, "stop": pos.stop, "target": pos.target} if pos else None, env.spec["digits"]) if (a.images or log_path) else None
        if a.policy == "excursion":
            atr = env.atr(ATR_MINUTES)
            if trigger.startswith("open position is moving against"):
                questions, answers = {}, None  # off-clock review: the strategy holds to the stop; forecasts happen on the clock only
            else:
                state = forecast_state(env, atr, a.every, a.horizon); questions = excursion_questions(state, a.horizon)
                answers = policy.decide(state, questions, imgs)
            decision = apply_excursion(env, state, answers, strategy, values, atr)
        else:
            questions = build_questions(state)
            answers = policy.decide(state, questions, imgs)
            decision = apply_answers(env, state, answers)
        env.decisions += 1
        if logf:
            paths = {}
            for k, png in (imgs or {}).items():
                rel = f"charts/{env.decisions:06d}_{k}.png"; (log_path.parent / rel).write_bytes(png); paths[k] = rel
            if imgs:
                from jeb_trading.charts import composite as _comp; rel = f"charts/{env.decisions:06d}_all.png"; (log_path.parent / rel).write_bytes(_comp(imgs)); paths["all"] = rel
            logf.write(json.dumps({"n": env.decisions, "time": str(env.now), "trigger": trigger, "state": state, "questions": questions, "answers": answers, "decision": decision,
                                   "images": paths, "policy": policy.name, "teacher": getattr(policy, "model_id", None), "debug": getattr(policy, "last_debug", None)}, ensure_ascii=False) + "\n"); logf.flush()
        if env.decisions % 25 == 0:
            print(f"[{env.now}] {env.decisions} decisions, {len(env.account.trades)} trades, equity {env.equity():.0f}, last: {decision.get('result')}", flush=True)
    if env.account.position is not None:
        env.close("end of run")
    s = env.summary()
    s["wall_seconds"] = round(time.perf_counter() - t_start, 1)
    lat = getattr(policy, "latencies", [])
    if lat:
        s["latency_p50_ms"] = round(sorted(lat)[len(lat) // 2] * 1000)
    print(json.dumps(s, default=str))
    if log_path:
        eq = pd.DataFrame(env.equity_curve, columns=["time", "equity"]).set_index("time")
        eq.to_csv(log_path.parent / "equity.csv")
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3), dpi=110); ax.plot(eq.index, eq["equity"]); ax.set_title(f"{a.symbol} {a.start}..{a.end} {policy.name}"); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(log_path.parent / "equity.png"); plt.close(fig)
        (log_path.parent / "trades.json").write_text(json.dumps([t.__dict__ for t in env.account.trades], default=str, indent=1))


if __name__ == "__main__":
    main()
