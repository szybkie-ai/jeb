"""Representative value per level for the excursion questions, from the labels: the mean realised excursion / close of
the rows that landed in each level. Written to data/level_values.json; run.py's excursion policy reads it for expectations.

    .venv/bin/python level_values.py data/hindsight_ex_*.jsonl --out data/level_values.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jeb_trading.excursions import CLOSE_LEVELS, DEFAULT_VALUES, EXCURSION_LEVELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+"); ap.add_argument("--out", default="data/level_values.json")
    a = ap.parse_args()
    sums: dict[str, dict[int, list[float]]] = {"up_move": defaultdict(list), "down_move": defaultdict(list), "close_move": defaultdict(list)}
    for path in a.rows:
        for line in open(path):
            r = json.loads(line)
            if r["qid"] != "first_move":  # one row per point is enough; every row carries the whole label
                continue
            h = r["hindsight"]
            sums["up_move"][h["up_level"]].append(h["up"]); sums["down_move"][h["down_level"]].append(h["down"]); sums["close_move"][h["close_level"]].append(h["close"])
    out = {}
    for qid, n in (("up_move", len(EXCURSION_LEVELS)), ("down_move", len(EXCURSION_LEVELS)), ("close_move", len(CLOSE_LEVELS))):
        out[qid] = [round(sum(v) / len(v), 3) if (v := sums[qid].get(k)) else DEFAULT_VALUES[qid][k] for k in range(n)]
        counts = [len(sums[qid].get(k, [])) for k in range(n)]
        print(f"{qid:10s} values {out[qid]}  counts {counts}")
    Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)


if __name__ == "__main__":
    main()
