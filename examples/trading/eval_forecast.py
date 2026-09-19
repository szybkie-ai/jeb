"""Held-out cross-entropy of a System One server on the price-movement questions: the cleaner signal test next to the
equity curve. One request per decision point (the four questions, the chart attached with --images), scored against the
realised level; baselines: uniform (log K) and the empirical marginal of the evaluated points.

    .venv/bin/python eval_forecast.py --rows ../../../openjev-train/data/raw/trading_ex.holdout.jsonl --url http://127.0.0.1:8024 --model jeb-latest --images --points 500 --out out/forecast-r1.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import httpx


def gold_of(row: dict) -> str:
    p = row["teacher_presentations"][0]
    return str(p["targets"][max(range(len(p["logprobs"])), key=lambda i: p["logprobs"][i])])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--image-root", default=str(Path.home() / "git/openjev-train/data/raw"))
    ap.add_argument("--url", default="http://127.0.0.1:8024"); ap.add_argument("--model", default="jeb-latest"); ap.add_argument("--key", default="local")
    ap.add_argument("--images", action="store_true"); ap.add_argument("--permutations", type=int, default=2); ap.add_argument("--points", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_tic: dict[tuple[str, int], dict] = defaultdict(dict)
    for line in open(a.rows):
        r = json.loads(line)
        by_tic[(r["episode"].split("-")[0], r["tic"])][r["qid"]] = r
    keys = sorted(by_tic)
    if a.points:
        keys = sorted(random.Random(a.seed).sample(keys, min(a.points, len(keys))))
    sem = asyncio.Semaphore(a.concurrency)
    results: list[dict] = []

    async def one(client: httpx.AsyncClient, key: tuple[str, int]) -> None:
        rows = by_tic[key]
        first = next(iter(rows.values()))
        body = {"state": first["state"], "model": a.model, "questions": {q: r["question"] for q, r in rows.items()}, "options": {"permutations": a.permutations}}
        if a.images and first.get("image"):
            png = Path(a.image_root, first["image"]).read_bytes()
            body["images"] = ["data:image/png;base64," + base64.b64encode(png).decode()]
        async with sem:
            for attempt in range(6):
                try:
                    resp = await client.post("/v1/systemone", json=body)
                    if resp.status_code < 500:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(3 * (attempt + 1))
            resp.raise_for_status()
        ans = resp.json()["answers"]
        rec = {"symbol": key[0], "tic": key[1]}
        for q, r in rows.items():
            pr = ans[q]["probabilities"]; g = gold_of(r)
            rec[q] = {"gold": g, "p_gold": float(pr.get(g, 0.0)), "pred": max(pr, key=pr.get), "k": len(r["question"]["criteria"]), "probs": pr}
        results.append(rec)
        if len(results) % 50 == 0:
            print(f"{len(results)}/{len(keys)} points", flush=True)

    async with httpx.AsyncClient(base_url=a.url, headers={"authorization": f"Bearer {a.key}"}, timeout=180) as client:
        await asyncio.gather(*(one(client, k) for k in keys))

    report: dict = {"model": a.model, "images": a.images, "points": len(results), "questions": {}}
    for q in ("first_move", "up_move", "down_move", "close_move"):
        recs = [r[q] for r in results if q in r]
        if not recs:
            continue
        ce = statistics.fmean(-math.log(max(x["p_gold"], 1e-6)) for x in recs)
        uniform = statistics.fmean(math.log(x["k"]) for x in recs)
        counts = defaultdict(int)
        for x in recs:
            counts[x["gold"]] += 1
        marginal = -sum(c / len(recs) * math.log(c / len(recs)) for c in counts.values())  # CE of always predicting the empirical marginal
        acc = statistics.fmean(x["pred"] == x["gold"] for x in recs)
        conf = [(max(x["probs"].values()), x["pred"] == x["gold"]) for x in recs]
        bins = defaultdict(list)
        for c, ok in conf:
            bins[min(9, int(c * 10))].append((c, ok))
        ece = sum(abs(statistics.fmean(o for _, o in v) - statistics.fmean(c for c, _ in v)) * len(v) for v in bins.values()) / len(conf)
        report["questions"][q] = {"n": len(recs), "cross_entropy": round(ce, 4), "uniform": round(uniform, 4), "marginal": round(marginal, 4), "gain_vs_uniform": round(uniform - ce, 4),
                                  "gain_vs_marginal": round(marginal - ce, 4), "accuracy": round(acc, 4), "ece": round(ece, 4),
                                  "by_symbol": {s: round(statistics.fmean(-math.log(max(r[q]["p_gold"], 1e-6)) for r in results if r["symbol"] == s and q in r), 4) for s in sorted({r["symbol"] for r in results})}}
        print(f"{q:11s} n={len(recs):5d} CE {ce:.3f} | uniform {uniform:.3f} marginal {marginal:.3f} | gain vs marginal {marginal - ce:+.3f} | acc {acc:.3f} ece {ece:.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({**report, "records": results}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    asyncio.run(main())
