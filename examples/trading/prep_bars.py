"""Tick day-parquets (the backtesting repo's Dukascopy cache) -> one-minute bars with spread, per symbol and range.

    .venv/bin/python prep_bars.py --symbol XAUUSD --start 2023-01-01 --end 2024-12-31
    -> data/XAUUSD_1m.parquet   (columns: open high low close (mid), bid_close ask_close, spread_mean spread_max, ticks)

15-minute and 4-hour bars are resampled from these at run time.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

CACHE = Path.home() / "git/backtesting/data/cache/ticks"


def day_bars(path: Path) -> pd.DataFrame:
    t = pd.read_parquet(path)
    mid = (t["bid"] + t["ask"]) / 2
    spread = t["ask"] - t["bid"]
    o = mid.resample("1min")
    bars = pd.DataFrame({"open": o.first(), "high": o.max(), "low": o.min(), "close": o.last(),
                         "bid_close": t["bid"].resample("1min").last(), "ask_close": t["ask"].resample("1min").last(),
                         "spread_mean": spread.resample("1min").mean(), "spread_max": spread.resample("1min").max(),
                         "ticks": mid.resample("1min").count()})
    return bars.dropna(subset=["open"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    frames, d, missing = [], start, 0
    while d <= end:
        p = CACHE / a.symbol / str(d.year) / f"{d:%Y%m%d}.parquet"
        if p.exists():
            frames.append(day_bars(p))
        elif d.weekday() < 5:
            missing += 1
        d += timedelta(days=1)
    if not frames:
        raise SystemExit("no tick days in range")
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    out = Path(a.out or f"data/{a.symbol}_1m.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out)
    print(f"{a.symbol}: {len(frames)} days, {len(bars)} one-minute bars, {missing} weekdays without ticks -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    print(bars.tail(2))


if __name__ == "__main__":
    main()
