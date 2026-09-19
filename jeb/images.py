"""Images in the shared prefix (the `images` extension).

An image costs the engine a fixed number of tokens that depends on the model's vision processor and the image size
(a 320x240 frame is ~284 tokens on Qwen3.5-4B, ~300 on the teacher). The prefix cache works in blocks, so the shared
prefix must be padded to a block multiple *including* those tokens; we learn the count per (width, height) once by
probing the engine (one extra one-token request), then pad the text after the state with a 1:1 pad token.
"""

from __future__ import annotations

import base64
import struct
from typing import Any

PROBE_SYSTEM = "You answer with one word."
PROBE_TEXT = "Say yes."


def image_size(url: str) -> tuple[int, int] | None:
    """(width, height) from a data: URL holding a PNG or JPEG, else None."""
    if not url.startswith("data:"):
        return None
    try:
        head = url.split(",", 1)[1]
        raw = base64.b64decode(head[:4096] + "=" * (-len(head[:4096]) % 4), validate=False)
    except Exception:
        return None
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        w, h = struct.unpack(">II", raw[16:24])
        return int(w), int(h)
    if raw[:2] == b"\xff\xd8":  # JPEG: walk the markers to a SOF
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker, seglen = raw[i + 1], struct.unpack(">H", raw[i + 2 : i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
                return int(w), int(h)
            i += 2 + seglen
    return None


class ImageTokenCounter:
    """Tokens per image, learned per (width, height) from the engine; unknown sizes cost one probe."""

    def __init__(self) -> None:
        self.by_size: dict[tuple[int, int], int] = {}
        self._text_only: int | None = None

    async def tokens_for(self, engine: Any, url: str) -> int:
        size = image_size(url)
        if size is not None and size in self.by_size:
            return self.by_size[size]
        probe = getattr(engine, "chat_prompt_tokens", None)
        if probe is None:
            return 0
        if self._text_only is None:
            self._text_only = await probe(PROBE_SYSTEM, PROBE_TEXT, [])
        n = max(0, await probe(PROBE_SYSTEM, PROBE_TEXT, [url]) - self._text_only)
        if size is not None:
            self.by_size[size] = n
        return n
