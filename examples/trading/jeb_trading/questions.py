"""The fan-out per decision point, and how the answers become orders."""

from __future__ import annotations

from typing import Any

from jeb_trading.env import TradingEnv

ACTIONS = {"buy": "Open a long position now", "sell": "Open a short position now", "stay_flat": "No trade: the move is not worth following"}
STOPS = {"tight": "about 0.5 ATR(15m) away: quick invalidation, small risk per pip", "normal": "about 1 ATR(15m) away", "wide": "about 2 ATR(15m) away: room to breathe"}
TARGETS = {"1R": "take profit at one times the stop distance", "2R": "take profit at two times the stop distance", "3R": "take profit at three times the stop distance", "trail": "no fixed target: trail the stop and let it run"}
RISKS = {"quarter": "risk 0.25% of the balance", "half": "risk 0.5% of the balance", "one": "risk 1% of the balance"}
MANAGE = {"hold": "Keep the position as it is", "close": "Close the position now at market", "breakeven": "Move the stop to the entry price (lock in no loss)", "trail": "Trail the stop: keep the original stop distance behind the current price"}
REGIMES = {"trending_up": "Higher highs and higher lows on the higher timeframes", "trending_down": "Lower highs and lower lows", "ranging": "Sideways between clear levels", "volatile": "Sharp swings both ways, likely news-driven"}


def build_questions(state: dict[str, Any]) -> dict[str, Any]:
    q: dict[str, Any] = {
        "regime": {"type": "choice", "instructions": "Looking at the 4-hour and 15-minute picture, which regime is the market in?", "criteria": REGIMES},
        "conviction": {"type": "score", "instructions": "How clear is the opportunity right now (either direction)?", "criteria": ["no edge", "weak", "moderate", "strong", "very strong"]},
    }
    if state["position"] == "flat":
        q["action"] = {"type": "choice", "instructions": "Given the trigger, the recent moves, the volatility and the charts, what should the account do now? Stay flat unless the setup is clear; spread and slippage cost real money.", "criteria": ACTIONS}
        q["stop"] = {"type": "choice", "instructions": "If a position is opened, how far should the initial stop be?", "criteria": STOPS}
        q["target"] = {"type": "choice", "instructions": "If a position is opened, how should it be taken off?", "criteria": TARGETS}
        q["risk"] = {"type": "choice", "instructions": "If a position is opened, how much of the balance should be at risk?", "criteria": RISKS}
    else:
        q["manage"] = {"type": "choice", "instructions": "There is an open position. Given its unrealised result, its excursions, the stop distance and what the charts show now, what should be done with it?", "criteria": MANAGE}
        q["exit_soon"] = {"type": "noul", "instructions": "Is this position likely to hit its stop before its target?", "criteria": {"true": "Likely to lose", "false": "Likely to work out"}}
    return q


def apply_answers(env: TradingEnv, state: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """Turn the answers into orders; return the decision record."""
    dec: dict[str, Any] = {"regime": answers["regime"]["choice"], "conviction": round(answers["conviction"]["score"], 2)}
    if state["position"] == "flat":
        action = answers["action"]["choice"]
        dec.update({"action": action, "stop": answers["stop"]["choice"], "target": answers["target"]["choice"], "risk": answers["risk"]["choice"]})
        if action in ("buy", "sell"):
            atr = env.atr(15)
            mult = {"tight": 0.5, "normal": 1.0, "wide": 2.0}[dec["stop"]]
            tr = {"1R": 1.0, "2R": 2.0, "3R": 3.0, "trail": None}[dec["target"]]
            risk = {"quarter": 0.25, "half": 0.5, "one": 1.0}[dec["risk"]]
            dec["result"] = env.open("long" if action == "buy" else "short", risk, mult, tr, atr, reason=state["trigger"])
        else:
            dec["result"] = "stayed flat"
    else:
        m = answers["manage"]["choice"]
        dec.update({"manage": m, "exit_soon": round(answers["exit_soon"]["noul"], 2)})
        if m == "close":
            dec["result"] = env.close("closed by decision")
        elif m in ("breakeven", "trail"):
            dec["result"] = env.move_stop(m)
        else:
            dec["result"] = "held"
    dec["confidence"] = {k: v.get("confidence", abs(2 * v["noul"] - 1) if "noul" in v else None) for k, v in answers.items()}
    return dec
