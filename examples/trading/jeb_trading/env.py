"""A virtual trading account driven by one-minute bars, with the frictions of a real one.

Decision points are events, not every bar: when flat, a *trigger* (a move of at least `trigger_pct` over
`trigger_minutes`, i.e. something happened) opens a decision; with a position open, the agent is asked again every
`manage_minutes` and whenever the trade moves against it by a stop fraction. Between decisions the account runs on
its own: stops and targets are executed on the one-minute high/low (stop first when both are hit in the same bar),
spread is paid on entry and exit, slippage is added on fills, commission per lot round trip, overnight swap ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SPECS = {
    "EURUSD": {"pip": 0.0001, "lot_units": 100_000, "quote": "USD", "commission_per_lot": 7.0, "min_lot": 0.01, "digits": 5},
    "GBPUSD": {"pip": 0.0001, "lot_units": 100_000, "quote": "USD", "commission_per_lot": 7.0, "min_lot": 0.01, "digits": 5},
    "XAUUSD": {"pip": 0.1, "lot_units": 100, "quote": "USD", "commission_per_lot": 7.0, "min_lot": 0.01, "digits": 2},
    "USATECHIDXUSD": {"pip": 1.0, "lot_units": 1, "quote": "USD", "commission_per_lot": 0.0, "min_lot": 0.1, "digits": 2},
}


@dataclass
class Position:
    side: str  # long | short
    lots: float
    entry: float
    opened: pd.Timestamp
    stop: float
    target: float | None
    risk_money: float
    mfe: float = 0.0  # max favourable excursion, price units
    mae: float = 0.0  # max adverse excursion
    entry_reason: str = ""


@dataclass
class Trade:
    side: str
    lots: float
    entry: float
    exit: float
    opened: pd.Timestamp
    closed: pd.Timestamp
    pnl: float
    reason: str
    r_multiple: float


@dataclass
class Account:
    balance: float = 10_000.0
    leverage: float = 30.0
    currency: str = "USD"
    trades: list[Trade] = field(default_factory=list)
    position: Position | None = None
    peak: float = 10_000.0
    max_drawdown: float = 0.0


class TradingEnv:
    def __init__(self, symbol: str, bars_1m: pd.DataFrame, start: str, end: str, trigger_pct: float = 0.15, trigger_minutes: int = 15,
                 manage_minutes: int = 30, slippage_pips: float = 0.3, balance: float = 10_000.0, seed: int = 0, cooldown_minutes: int = 30) -> None:
        self.symbol, self.spec = symbol, SPECS[symbol]
        self.bars = bars_1m.loc[(bars_1m.index >= pd.Timestamp(start) - pd.Timedelta(days=400)) & (bars_1m.index <= pd.Timestamp(end))]
        self.start = pd.Timestamp(start)
        self.trigger_pct, self.trigger_minutes, self.manage_minutes, self.cooldown_minutes = trigger_pct, trigger_minutes, manage_minutes, cooldown_minutes
        self.slippage_pips = slippage_pips
        self.rng = np.random.default_rng(seed)
        self.account = Account(balance=balance, peak=balance)
        self.i = int(self.bars.index.searchsorted(self.start))
        self.last_decision_i = -10**9
        self.last_close_i = -10**9
        self.closes = self.bars["close"].to_numpy()
        self.highs, self.lows = self.bars["high"].to_numpy(), self.bars["low"].to_numpy()
        self.bids, self.asks = self.bars["bid_close"].to_numpy(), self.bars["ask_close"].to_numpy()
        self.spread = self.bars["spread_mean"].to_numpy()
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.decisions = 0

    # ------------------------------------------------------------------ helpers
    @property
    def now(self) -> pd.Timestamp:
        return self.bars.index[self.i]

    @property
    def done(self) -> bool:
        return self.i >= len(self.bars) - 1

    def pip_value_per_lot(self) -> float:
        return self.spec["pip"] * self.spec["lot_units"]  # quote currency per pip per lot (USD-quoted instruments)

    def unrealised(self, price: float | None = None) -> float:
        p = self.account.position
        if p is None:
            return 0.0
        px = price if price is not None else (self.bids[self.i] if p.side == "long" else self.asks[self.i])
        diff = (px - p.entry) if p.side == "long" else (p.entry - px)
        return diff * self.spec["lot_units"] * p.lots

    def equity(self) -> float:
        return self.account.balance + self.unrealised()

    def margin_used(self) -> float:
        p = self.account.position
        return 0.0 if p is None else p.lots * self.spec["lot_units"] * self.closes[self.i] / self.account.leverage

    def move_pct(self, minutes: int) -> float:
        j = max(0, self.i - minutes)
        return (self.closes[self.i] / self.closes[j] - 1.0) * 100.0

    def atr(self, tf_minutes: int, n: int = 14) -> float:
        """ATR over the last n bars of `tf_minutes` minutes, in price units."""
        need = tf_minutes * (n + 1)
        j = max(0, self.i - need)
        h, l, c = self.highs[j:self.i + 1], self.lows[j:self.i + 1], self.closes[j:self.i + 1]
        if len(c) < tf_minutes * 2:
            return float("nan")
        k = len(c) // tf_minutes
        h = h[len(c) - k * tf_minutes:].reshape(k, tf_minutes).max(1)
        l = l[len(c) - k * tf_minutes:].reshape(k, tf_minutes).min(1)
        c = c[len(c) - k * tf_minutes:].reshape(k, tf_minutes)[:, -1]
        prev = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum(h - l, np.maximum(abs(h - prev), abs(l - prev)))
        return float(tr[-n:].mean())

    # ------------------------------------------------------------------ stepping
    def next_decision(self) -> str | None:
        """Advance the clock to the next decision point; return the trigger description, or None when out of data."""
        while not self.done:
            self.i += 1
            self._run_bar()
            if self.account.position is None:
                if self.i - self.last_close_i < self.cooldown_minutes or self.i - self.last_decision_i < self.trigger_minutes:
                    continue
                mv = self.move_pct(self.trigger_minutes)
                if abs(mv) >= self.trigger_pct and self.spread[self.i] < 5 * np.nanmedian(self.spread[max(0, self.i - 1440):self.i + 1]):
                    self.last_decision_i = self.i
                    return f"{self.symbol} moved {mv:+.2f}% in the last {self.trigger_minutes} minutes"
            else:
                p = self.account.position
                adverse = ((p.entry - self.bids[self.i]) if p.side == "long" else (self.asks[self.i] - p.entry))
                stop_dist = abs(p.entry - p.stop)
                if self.i - self.last_decision_i >= self.manage_minutes or (adverse > 0.6 * stop_dist and self.i - self.last_decision_i >= 5):
                    self.last_decision_i = self.i
                    return "open position review" if adverse <= 0.6 * stop_dist else "open position is moving against you"
        return None

    def _run_bar(self) -> None:
        """Execute stops/targets on this bar, update excursions, record equity."""
        p = self.account.position
        if p is not None:
            hi, lo = self.highs[self.i], self.lows[self.i]
            half_spread = self.spread[self.i] / 2
            if p.side == "long":
                p.mfe = max(p.mfe, hi - p.entry); p.mae = max(p.mae, p.entry - lo)
                if lo - half_spread <= p.stop:
                    self._close(p.stop - self._slip(), "stop")
                elif p.target is not None and hi - half_spread >= p.target:
                    self._close(p.target, "target")
            else:
                p.mfe = max(p.mfe, p.entry - lo); p.mae = max(p.mae, hi - p.entry)
                if hi + half_spread >= p.stop:
                    self._close(p.stop + self._slip(), "stop")
                elif p.target is not None and lo + half_spread <= p.target:
                    self._close(p.target, "target")
        eq = self.equity()
        self.account.peak = max(self.account.peak, eq)
        self.account.max_drawdown = max(self.account.max_drawdown, (self.account.peak - eq) / self.account.peak)
        if self.i % 60 == 0:
            self.equity_curve.append((self.now, eq))

    def _slip(self) -> float:
        return float(abs(self.rng.normal(0, self.slippage_pips)) * self.spec["pip"])

    # ------------------------------------------------------------------ orders
    def open(self, side: str, risk_pct: float, stop_atr_mult: float, target_r: float | None, atr: float, reason: str = "") -> str:
        if self.account.position is not None:
            return "already in a position"
        if not (atr > 0):
            return "no volatility estimate"
        price = (self.asks[self.i] + self._slip()) if side == "long" else (self.bids[self.i] - self._slip())
        stop_dist = max(stop_atr_mult * atr, 3 * self.spec["pip"])
        risk_money = self.account.balance * risk_pct / 100.0
        lots = risk_money / (stop_dist * self.spec["lot_units"])
        lots = max(self.spec["min_lot"], math.floor(lots / self.spec["min_lot"]) * self.spec["min_lot"])
        margin = lots * self.spec["lot_units"] * price / self.account.leverage
        if margin > 0.8 * self.equity():
            lots = max(self.spec["min_lot"], math.floor(0.8 * self.equity() * self.account.leverage / (self.spec["lot_units"] * price) / self.spec["min_lot"]) * self.spec["min_lot"])
        stop = price - stop_dist if side == "long" else price + stop_dist
        target = None if target_r is None else (price + target_r * stop_dist if side == "long" else price - target_r * stop_dist)
        self.account.balance -= self.spec["commission_per_lot"] * lots
        self.account.position = Position(side, lots, price, self.now, stop, target, risk_money=lots * stop_dist * self.spec["lot_units"], entry_reason=reason)
        return f"opened {side} {lots:.2f} lots at {price:.{self.spec['digits']}f}, stop {stop:.{self.spec['digits']}f}, target {'none' if target is None else f'{target:.{self.spec['digits']}f}'}"

    def close(self, reason: str = "closed by decision") -> str:
        p = self.account.position
        if p is None:
            return "no position"
        px = (self.bids[self.i] - self._slip()) if p.side == "long" else (self.asks[self.i] + self._slip())
        return self._close(px, reason)

    def _close(self, px: float, reason: str) -> str:
        p = self.account.position
        assert p is not None
        diff = (px - p.entry) if p.side == "long" else (p.entry - px)
        pnl = diff * self.spec["lot_units"] * p.lots
        self.account.balance += pnl
        r = pnl / p.risk_money if p.risk_money else 0.0
        self.account.trades.append(Trade(p.side, p.lots, p.entry, px, p.opened, self.now, pnl, reason, r))
        self.account.position = None
        self.last_close_i = self.i
        return f"closed {p.side} at {px:.{self.spec['digits']}f} ({reason}): {pnl:+.2f} {self.account.currency}, {r:+.2f}R"

    def move_stop(self, where: str) -> str:
        p = self.account.position
        if p is None:
            return "no position"
        if where == "breakeven":
            new = p.entry + (self.spread[self.i] if p.side == "long" else -self.spread[self.i])
        elif where == "trail":
            dist = abs(p.entry - p.stop)
            new = (self.bids[self.i] - dist) if p.side == "long" else (self.asks[self.i] + dist)
        else:
            return "unknown stop move"
        better = (new > p.stop) if p.side == "long" else (new < p.stop)
        if better:
            p.stop = new
            return f"stop moved to {new:.{self.spec['digits']}f} ({where})"
        return "stop unchanged (would loosen it)"

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict[str, Any]:
        t = self.account.trades
        wins = [x for x in t if x.pnl > 0]
        gross_win = sum(x.pnl for x in wins); gross_loss = -sum(x.pnl for x in t if x.pnl <= 0)
        return {"symbol": self.symbol, "from": str(self.start.date()), "to": str(self.now.date()), "decisions": self.decisions, "trades": len(t),
                "win_rate": round(len(wins) / len(t), 3) if t else None, "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
                "net_pnl": round(self.account.balance - 10_000.0 + self.unrealised(), 2), "return_pct": round((self.equity() / 10_000.0 - 1) * 100, 2),
                "max_drawdown_pct": round(self.account.max_drawdown * 100, 2), "avg_r": round(float(np.mean([x.r_multiple for x in t])), 2) if t else None,
                "avg_hold_minutes": round(float(np.mean([(x.closed - x.opened).total_seconds() / 60 for x in t])), 1) if t else None,
                "exits": dict(pd.Series([x.reason for x in t]).value_counts()) if t else {}}
