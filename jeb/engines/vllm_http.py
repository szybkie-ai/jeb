"""vLLM over its OpenAI-compatible HTTP API (stock vLLM, no modifications).

One `/v1/completions` call per distinct label set, with every prompt of that set batched in
`prompt` (a list of token-id lists), `max_tokens=1`, and `logprob_token_ids` so the engine
reports exactly our label tokens. Missing labels (an engine without that field, or one that
caps logprobs) are filled with a floor and counted, so callers can see it happening.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from typing import Any

import httpx

from jeb.engines.base import EngineError, ScoreItem, ScoreResult

FLOOR_LOGPROB = -30.0


class VllmHttpEngine:
    name = "vllm-http"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout: float = 60.0,
        max_batch: int = 64,
        top_logprobs: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_batch = max_batch
        self.top_logprobs = top_logprobs
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]:
        groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        results: list[ScoreResult | None] = [None] * len(items)
        chat_idx = [i for i, it in enumerate(items) if it.chat is not None]
        for i, it in enumerate(items):
            if it.chat is None:
                groups[tuple(it.label_ids)].append(i)
        if chat_idx:
            sem = asyncio.Semaphore(self.max_batch)

            async def one(i: int) -> None:
                async with sem:
                    results[i] = await self._score_chat(items[i])

            await asyncio.gather(*(one(i) for i in chat_idx))

        async def run(label_ids: tuple[int, ...], idxs: list[int]) -> None:
            for start in range(0, len(idxs), self.max_batch):
                chunk = idxs[start : start + self.max_batch]
                body = {
                    "model": self.model,
                    "prompt": [items[i].prompt_ids for i in chunk],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": self.top_logprobs,
                    "logprob_token_ids": list(label_ids),
                    "return_tokens_as_token_ids": True,
                }
                data = await self._post(body)
                choices = sorted(data.get("choices", []), key=lambda c: c.get("index", 0))
                if len(choices) != len(chunk):
                    raise EngineError(f"engine returned {len(choices)} choices for {len(chunk)} prompts")
                for i, ch in zip(chunk, choices):
                    results[i] = self._parse(ch, list(label_ids), len(items[i].prompt_ids))

        await asyncio.gather(*(run(k, v) for k, v in groups.items()))
        return [r for r in results if r is not None]

    async def _score_chat(self, item: ScoreItem) -> ScoreResult:
        """One prompt through /v1/chat/completions (the only endpoint that takes images), same label-logprob trick.
        The chat template with thinking disabled renders exactly the token-id prompt of the text path."""
        c = item.chat or {}
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in c.get("images") or []]
        content.append({"type": "text", "text": c["user"]})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": c["system"]}, {"role": "user", "content": content}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "logprob_token_ids": list(item.label_ids),
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = await self._post(body, path="/v1/chat/completions")
        choices = data.get("choices") or []
        if not choices:
            raise EngineError("engine returned no choices for a chat prompt")
        lp = choices[0].get("logprobs") or {}
        first = (lp.get("content") or [None])[0] or {}
        by_id: dict[int, float] = {}
        for entry in [first, *(first.get("top_logprobs") or [])]:
            t = entry.get("token") if isinstance(entry, dict) else None
            if isinstance(t, str) and t.startswith("token_id:") and entry.get("logprob") is not None:
                by_id.setdefault(int(t.split(":", 1)[1]), float(entry["logprob"]))
        missing = sum(1 for i in item.label_ids if i not in by_id)
        floor = min([*by_id.values(), FLOOR_LOGPROB]) - 1.0 if by_id else FLOOR_LOGPROB
        prompt_tokens = int((data.get("usage") or {}).get("prompt_tokens") or len(item.prompt_ids))
        return ScoreResult(logprobs=[by_id.get(i, floor) for i in item.label_ids], prompt_tokens=prompt_tokens, missing=missing)

    async def chat_prompt_tokens(self, system: str, text: str, images: list[str]) -> int:
        """How many prompt tokens the engine counts for a chat turn (images included): used to learn image token costs."""
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in images]
        content.append({"type": "text", "text": text})
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "max_tokens": 1, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        data = await self._post(body, path="/v1/chat/completions")
        return int((data.get("usage") or {}).get("prompt_tokens") or 0)

    async def _post(self, body: dict[str, Any], path: str = "/v1/completions") -> dict[str, Any]:
        try:
            r = await self._client.post(path, json=body)
        except httpx.HTTPError as e:
            raise EngineError(f"engine unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"engine HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    @staticmethod
    def _parse(choice: dict[str, Any], label_ids: list[int], prompt_tokens: int) -> ScoreResult:
        lp = choice.get("logprobs") or {}
        top = (lp.get("top_logprobs") or [None])[0] or {}
        # With return_tokens_as_token_ids the keys look like "token_id:32"; be lenient about the format.
        by_id: dict[int, float] = {}
        for k, v in top.items():
            if isinstance(k, str) and k.startswith("token_id:"):
                by_id[int(k.split(":", 1)[1])] = float(v)
            elif isinstance(k, int):
                by_id[k] = float(v)
        # The sampled token is reported separately; include it too.
        toks, tlps = lp.get("tokens") or [], lp.get("token_logprobs") or []
        for t, v in zip(toks, tlps):
            if isinstance(t, str) and t.startswith("token_id:") and v is not None:
                by_id.setdefault(int(t.split(":", 1)[1]), float(v))
        missing = sum(1 for i in label_ids if i not in by_id)
        floor = min([*by_id.values(), FLOOR_LOGPROB]) - 1.0 if by_id else FLOOR_LOGPROB
        return ScoreResult(logprobs=[by_id.get(i, floor) for i in label_ids], prompt_tokens=prompt_tokens, missing=missing)

    async def models(self) -> list[str]:
        data = await self._post_get("/v1/models")
        return [m["id"] for m in data.get("data", [])]

    async def _post_get(self, path: str) -> dict[str, Any]:
        r = await self._client.get(path)
        if r.status_code != 200:
            raise EngineError(f"engine HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()


def in_set_mass(logprobs: list[float]) -> float:
    return sum(math.exp(x) for x in logprobs)
