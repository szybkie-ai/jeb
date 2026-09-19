"""The state a trading model sees: instrument, time, quotes and recent moves, volatility, the account, the position, costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jeb_trading.env import TradingEnv


def session(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 7 <= h < 12: return "London morning"
    if 12 <= h < 16: return "London/New York overlap"
    if 16 <= h < 21: return "New York afternoon"
    return "Asia / off-hours"


def compile_state(env: TradingEnv, trigger: str, with_charts: bool) -> dict[str, Any]:
    i, spec = env.i, env.spec
    pip, d = spec["pip"], spec["digits"]
    bid, ask = env.bids[i], env.asks[i]
    spread_pips = (ask - bid) / pip
    close = env.closes[i]
    atr15, atr4h = env.atr(15), env.atr(240)
    day_start = max(0, i - env.now.hour * 60 - env.now.minute)
    hi_day, lo_day = float(np.max(env.highs[day_start:i + 1])), float(np.min(env.lows[day_start:i + 1]))
    j20 = max(0, i - 20 * 1440)
    hi20, lo20 = float(np.max(env.highs[j20:i + 1])), float(np.min(env.lows[j20:i + 1]))
    prev_close = env.closes[max(0, day_start - 1)]
    fmt = lambda v: f"{v:.{d}f}"  # noqa: E731
    acct, pos = env.account, env.account.position
    state: dict[str, Any] = {
        "rules": {
            "instrument": f"{env.symbol}, 1 pip = {pip}, 1 lot = {spec['lot_units']} units, quoted in {spec['quote']}",
            "costs": f"spread as quoted, commission {spec['commission_per_lot']} {spec['quote']} per lot round trip, slippage about {env.slippage_pips} pips on market orders",
            "execution": "market orders only; stops and targets are executed on one-minute highs/lows; decisions are asked on triggers and on position reviews, the account runs on its own in between",
            "risk_policy": "position size comes from the chosen risk percentage and the stop distance; one position at a time",
        },
        "time": {"weekday": env.now.strftime("%A"), "hour_utc": env.now.strftime("%H:%M"), "session": session(env.now)},
        "trigger": trigger,
        "market": {
            "bid": fmt(bid), "ask": fmt(ask), "spread_pips": round(spread_pips, 1),
            "move_pct": {"5m": round(env.move_pct(5), 3), "15m": round(env.move_pct(15), 3), "1h": round(env.move_pct(60), 3), "4h": round(env.move_pct(240), 3), "1d": round(env.move_pct(1440), 3), "5d": round(env.move_pct(5 * 1440), 3)},
            "atr_pips": {"15m": round(atr15 / pip, 1) if atr15 == atr15 else None, "4h": round(atr4h / pip, 1) if atr4h == atr4h else None},
            "today": {"high": fmt(hi_day), "low": fmt(lo_day), "position_in_range": round((close - lo_day) / (hi_day - lo_day), 2) if hi_day > lo_day else None, "vs_previous_close_pct": round((close / prev_close - 1) * 100, 3)},
            "last_20_days": {"high": fmt(hi20), "low": fmt(lo20), "position_in_range": round((close - lo20) / (hi20 - lo20), 2) if hi20 > lo20 else None},
        },
        "account": {"currency": acct.currency, "balance": round(acct.balance, 2), "equity": round(env.equity(), 2), "leverage": f"{acct.leverage:.0f}:1",
                    "margin_used": round(env.margin_used(), 2), "free_margin": round(env.equity() - env.margin_used(), 2),
                    "max_drawdown_so_far_pct": round(acct.max_drawdown * 100, 2), "closed_trades": len(acct.trades),
                    "last_trades": [f"{t.side} {t.lots:.2f} lots, held {int((t.closed - t.opened).total_seconds() // 60)} min, closed {int((env.now - t.closed).total_seconds() // 3600)} h ago: {t.pnl:+.0f} ({t.r_multiple:+.1f}R, {t.reason})" for t in acct.trades[-5:]] or "none yet"},
    }
    if pos is not None:
        px = bid if pos.side == "long" else ask
        pnl = env.unrealised()
        state["position"] = {
            "side": pos.side, "lots": round(pos.lots, 2), "entry": fmt(pos.entry),
            "held_minutes": int((env.now - pos.opened).total_seconds() // 60), "current_price": fmt(px),
            "unrealised": f"{pnl:+.2f} {acct.currency} ({(pnl / pos.risk_money if pos.risk_money else 0):+.2f}R)",
            "stop": fmt(pos.stop), "stop_distance_pips": round(abs(px - pos.stop) / pip, 1), "target": fmt(pos.target) if pos.target else "none",
            "max_favourable_pips": round(pos.mfe / pip, 1), "max_adverse_pips": round(pos.mae / pip, 1), "entry_reason": pos.entry_reason,
        }
    else:
        state["position"] = "flat"
    if with_charts:
        state["charts"] = "the attached image has four chart panels: top-left 1-minute (last 2 hours), top-right 15-minute (last 24 hours), bottom-left 4-hour (last 20 days), bottom-right daily (last 6 months); candles with EMA20 (yellow), EMA50 (blue), EMA200 (purple); the x axes are time before now; white/red/green lines are the entry/stop/target of the open position"
    return state
