"""llama.cpp's `llama-server` (GGUF builds, Ollama's engine) through its native `/completion` API.

Text prompts go as token-id lists with `n_predict=1` and `n_probs`, which returns the log-probabilities of the top
tokens by id; labels outside the top list get a floor and are counted as missing, so a small `n_probs` shows up in the
answer's `missing` field rather than silently. Images go through `/v1/chat/completions` (the server's multimodal
projector must be loaded with `--mmproj`); its OpenAI-style logprobs carry token strings, which are matched back to the
label ids through the tokenizer-supplied strings in `ScoreItem.chat["label_strings"]` when present.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from jeb.engines.base import EngineError, ScoreItem, ScoreResult

FLOOR_LOGPROB = -30.0


class LlamaServerEngine:
    name = "llama-server"

    def __init__(self, base_url: str, model: str = "", api_key: str = "local", timeout: float = 120.0, max_batch: int = 8, n_probs: int = 40) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_batch = max_batch  # llama-server handles concurrency through its slots (-np); keep the client modest
        self.n_probs = n_probs
        self._client = httpx.AsyncClient(base_url=self.base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=httpx.Timeout(timeout, connect=5.0))

    async def score(self, items: list[ScoreItem]) -> list[ScoreResult]:
        sem = asyncio.Semaphore(self.max_batch)
        results: list[ScoreResult | None] = [None] * len(items)

        async def one(i: int) -> None:
            async with sem:
                results[i] = await (self._score_chat(items[i]) if items[i].chat is not None else self._score_ids(items[i]))

        await asyncio.gather(*(one(i) for i in range(len(items))))
        return [r for r in results if r is not None]

    async def _score_ids(self, item: ScoreItem) -> ScoreResult:
        body = {"prompt": item.prompt_ids, "n_predict": 1, "n_probs": self.n_probs, "temperature": 0.0, "cache_prompt": True}
        data = await self._post(body, "/completion")
        cp = data.get("completion_probabilities") or []
        first = cp[0] if cp else {}
        by_id: dict[int, float] = {}
        for entry in [first, *(first.get("top_logprobs") or [])]:
            if isinstance(entry, dict) and entry.get("id") is not None and entry.get("logprob") is not None:
                by_id.setdefault(int(entry["id"]), float(entry["logprob"]))
        return self._result(by_id, item.label_ids, int(data.get("tokens_evaluated") or len(item.prompt_ids)))

    async def _score_chat(self, item: ScoreItem) -> ScoreResult:
        c = item.chat or {}
        content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in c.get("images") or []]
        content.append({"type": "text", "text": c["user"]})
        body = {"model": self.model or "default", "messages": [{"role": "system", "content": c["system"]}, {"role": "user", "content": content}],
                "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": min(self.n_probs, 20), "chat_template_kwargs": {"enable_thinking": False}}
        data = await self._post(body, "/v1/chat/completions")
        choices = data.get("choices") or []
        if not choices:
            raise EngineError("engine returned no choices for a chat prompt")
        first = ((choices[0].get("logprobs") or {}).get("content") or [None])[0] or {}
        strings = c.get("label_strings") or {}
        by_string = {entry["token"]: float(entry["logprob"]) for entry in [first, *(first.get("top_logprobs") or [])] if isinstance(entry, dict) and entry.get("token") is not None and entry.get("logprob") is not None}
        by_id = {lid: by_string[s] for lid, s in strings.items() if s in by_string}
        return self._result(by_id, item.label_ids, int((data.get("usage") or {}).get("prompt_tokens") or len(item.prompt_ids)))

    @staticmethod
    def _result(by_id: dict[int, float], label_ids: list[int], prompt_tokens: int) -> ScoreResult:
        missing = sum(1 for i in label_ids if i not in by_id)
        floor = min([*by_id.values(), FLOOR_LOGPROB]) - 1.0 if by_id else FLOOR_LOGPROB
        return ScoreResult(logprobs=[by_id.get(i, floor) for i in label_ids], prompt_tokens=prompt_tokens, missing=missing)

    async def _post(self, body: dict[str, Any], path: str) -> dict[str, Any]:
        try:
            r = await self._client.post(path, json=body)
        except httpx.HTTPError as e:
            raise EngineError(f"engine unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"engine HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()
