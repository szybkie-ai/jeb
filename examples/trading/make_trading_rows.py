"""Hindsight rows -> the training task's raw file: attach the rendered chart (relative path under data/raw) and the same
"charts" note the state carries at inference, keep the clear decision points, drop rows whose image is missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NOTE = "the attached image has four chart panels: top-left 1-minute (last 2 hours), top-right 15-minute (last 24 hours), bottom-left 4-hour (last 20 days), bottom-right daily (last 6 months); candles with EMA20 (yellow), EMA50 (blue), EMA200 (purple); the x axes are time before now; white/red/green lines are the entry/stop/target of the open position"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--charts", required=True, help="dir with <tic>.png")
    ap.add_argument("--rel", required=True, help="the charts dir as seen from the task's data/raw (e.g. trading_charts/XAUUSD_2019_2022)")
    ap.add_argument("--out", required=True); ap.add_argument("--min-margin", type=float, default=0.5, help="action labels only: drop points whose best action leads by less")
    ap.add_argument("--points", type=int, default=0, help="keep a random sample of this many decision points (all their rows)"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    charts = Path(a.charts); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped_margin = dropped_image = 0
    keep_tics = None
    if a.points:
        import random
        tics = sorted({json.loads(l)["tic"] for l in open(a.rows)})
        keep_tics = set(random.Random(a.seed).sample(tics, min(a.points, len(tics))))
    with out.open("w") as f:
        for l in open(a.rows):
            r = json.loads(l)
            if keep_tics is not None and r["tic"] not in keep_tics:
                continue
            if "margin" in r["hindsight"] and r["qid"] != "conviction" and r["hindsight"]["margin"] < a.min_margin:
                dropped_margin += 1; continue
            if not (charts / f"{r['tic']}.png").exists():
                dropped_image += 1; continue
            r["image"] = f"{a.rel}/{r['tic']}.png"
            r["state"] = dict(r["state"], charts=NOTE)
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); kept += 1
    print(f"{kept} rows -> {out} (dropped {dropped_margin} noisy, {dropped_image} without a chart)")


if __name__ == "__main__":
    main()
