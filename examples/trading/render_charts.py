"""Render the four-panel composite for every decision point of a hindsight file (multiprocessing), so the rows can carry an
image for image-conditioned training. Rebuilds the same environment, so the bar index in each row lands on the same bar.

    .venv/bin/python render_charts.py --rows data/hindsight_XAUUSD_2019_2022.jsonl --symbol XAUUSD --bars data/XAUUSD_2019_2022_1m.parquet \
        --start 2019-02-01 --end 2022-12-31 --out data/trading_charts/XAUUSD_2019_2022
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from jeb_trading.charts import charts, composite
from jeb_trading.env import TradingEnv

_bars = None


def _init(bars_path: str, symbol: str, start: str, end: str) -> None:
    global _bars
    b = pd.read_parquet(bars_path)
    env = TradingEnv(symbol, b, start, end)
    _bars = env.bars


def _one(job: tuple[int, str, int]) -> str:
    tic, out, digits = job
    p = Path(out)
    if not p.exists():
        p.write_bytes(composite(charts(_bars, tic, None, digits)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--symbol", required=True); ap.add_argument("--bars", required=True)
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tics = sorted({json.loads(l)["tic"] for l in open(a.rows)})
    digits = 2 if a.symbol in ("XAUUSD", "USATECHIDXUSD") else 5
    jobs = [(t, str(out / f"{t}.png"), digits) for t in tics]
    print(f"{len(jobs)} decision points to render with {a.procs} processes", flush=True)
    with Pool(a.procs, initializer=_init, initargs=(a.bars, a.symbol, a.start, a.end)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_one, jobs, chunksize=16), 1):
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
