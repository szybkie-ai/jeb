"""Run an evaluation set through a server and record raw label log-probabilities per item."""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from jeb.evals.datasets import Item


async def _one(client: httpx.AsyncClient, item: Item, model: str, permutations: int, sem: asyncio.Semaphore) -> dict[str, Any]:
    body = {
        "state": item.state,
        "model": model,
        "questions": {"q": item.question},
        "options": {"debug": True, "calibration": "raw", "permutations": permutations},
    }
    async with sem:
        t0 = time.perf_counter()
        r = await client.post("/v1/systemone", json=body)
        dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    dbg = data["debug"]["q"]
    ans = data["answers"]["q"]
    return {
        "kind": item.kind,
        "gold": item.gold,
        "permutations": [{"labels": p["labels"], "targets": p["targets"], "logprobs": p["logprobs"]} for p in dbg["permutations"]],
        "in_set_mass": dbg["in_set_mass"],
        "answer": ans,
        "latency_s": round(dt, 4),
        "input_tokens": data["usage"]["input_tokens"],
        "meta": item.meta,
    }


async def run(items: list[Item], url: str, key: str | None, model: str, permutations: int, concurrency: int, out: Path) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    sem = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=120) as client:
        rows = await asyncio.gather(*(_one(client, it, model, permutations, sem) for it in items))
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    lat = sorted(r["latency_s"] for r in rows)
    return {
        "n": len(rows),
        "latency_p50": lat[len(lat) // 2],
        "latency_p90": lat[int(len(lat) * 0.9)],
        "in_set_mass_mean": round(statistics.fmean(r["in_set_mass"] for r in rows), 4),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "out": str(out),
    }
