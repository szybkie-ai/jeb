"""Who decides: a scripted momentum baseline, or a System One server (JEB or the original API)."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx


from jeb_trading.charts import composite  # noqa: E402


class MomentumPolicy:
    """Follow the trigger's direction with a 1-ATR stop and a 2R target, half a percent at risk; breakeven after +1R."""

    name = "momentum"

    def decide(self, state: dict[str, Any], questions: dict[str, Any], images: dict[str, bytes] | None) -> dict[str, Any]:
        a: dict[str, Any] = {"regime": {"type": "choice", "choice": "ranging", "probabilities": {}, "confidence": 0.0},
                             "conviction": {"type": "score", "score": 2.0, "legend": {}, "probabilities": {}, "confidence": 0.0}}
        if state["position"] == "flat":
            mv = state["market"]["move_pct"]["15m"]
            a["action"] = {"type": "choice", "choice": "buy" if mv > 0 else "sell", "probabilities": {}, "confidence": 0.0}
            a["stop"] = {"type": "choice", "choice": "normal", "probabilities": {}, "confidence": 0.0}
            a["target"] = {"type": "choice", "choice": "2R", "probabilities": {}, "confidence": 0.0}
            a["risk"] = {"type": "choice", "choice": "half", "probabilities": {}, "confidence": 0.0}
        else:
            r = float(state["position"]["unrealised"].split("(")[1].split("R")[0])
            a["manage"] = {"type": "choice", "choice": "breakeven" if r >= 1.0 else "hold", "probabilities": {}, "confidence": 0.0}
            a["exit_soon"] = {"type": "noul", "noul": 0.5}
        return a


class SystemOnePolicy:
    def __init__(self, url: str, key: str | None, model: str, permutations: int = 2, collect: bool = False, images: bool = False, plain: bool = False) -> None:
        self.client = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=600)
        self.model, self.permutations, self.collect, self.images, self.plain = model, permutations, collect, images and not plain, plain
        self.name = "system-one"
        self.latencies: list[float] = []
        self.last_debug: dict | None = None
        self.model_id: str | None = None

    def decide(self, state: dict[str, Any], questions: dict[str, Any], images: dict[str, bytes] | None) -> dict[str, Any]:
        body: dict[str, Any] = {"state": state, "model": self.model, "questions": questions}
        if not self.plain:
            opts: dict[str, Any] = {"permutations": self.permutations}
            if self.collect:
                opts.update({"debug": True, "calibration": "raw"})
            body["options"] = opts
        if self.images and images:
            # One image per request (the engines are started with one image per prompt): the three charts stacked.
            body["images"] = ["data:image/png;base64," + base64.b64encode(composite(images)).decode()]
        t0 = time.perf_counter()
        r = None
        for attempt in range(8):
            try:
                r = self.client.post("/v1/systemone", json=body)
            except httpx.HTTPError as e:
                print(f"[policy] {e.__class__.__name__}, retry {attempt + 1}/8", flush=True); time.sleep(5 * (attempt + 1)); continue
            if r.status_code >= 500:
                print(f"[policy] HTTP {r.status_code}, retry {attempt + 1}/8", flush=True); time.sleep(5 * (attempt + 1)); continue
            break
        self.latencies.append(time.perf_counter() - t0)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"System One HTTP {r.status_code if r is not None else 'unreachable'}: {r.text[:300] if r is not None else ''}")
        data = r.json()
        self.model_id = data.get("model")
        self.last_debug = {qid: {"presentations": [{"labels": p["labels"], "targets": p["targets"], "logprobs": p["logprobs"]} for p in d["permutations"]], "in_set_mass": d["in_set_mass"]} for qid, d in data["debug"].items() if not qid.startswith("_")} if (self.collect and data.get("debug")) else None
        return data["answers"]
