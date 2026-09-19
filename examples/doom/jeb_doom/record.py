"""Recording: game frames with a judgments panel (the Choice/Noul distributions per tick) composed into an animated GIF
plus the per-tick JSON. ffmpeg-free on purpose; convert the PNG frames to MP4 elsewhere if wanted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 300
GREEN = (60, 220, 100)
DIM = (40, 90, 60)
BG = (8, 14, 10)
TXT = (200, 230, 200)


def _font(size: int):
    for name in ("Menlo.ttc", "DejaVuSansMono.ttf", "Courier New.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(judgment: dict, height: int) -> Image.Image:
    img = Image.new("RGB", (PANEL_W, height), BG)
    d = ImageDraw.Draw(img)
    f, fs = _font(11), _font(10)
    y = 6
    dec = judgment.get("decision", {})
    d.text((6, y), f"tic {judgment.get('tic', 0)}  goal: {dec.get('goal', '')}", fill=TXT, font=f)
    y += 14
    if dec.get("subject"):
        d.text((6, y), f"subject: {dec['subject']}", fill=TXT, font=fs)
        y += 13
    for qid, a in (judgment.get("answers") or {}).items():
        if y > height - 30:
            break
        d.text((6, y), qid.upper(), fill=GREEN, font=fs)
        y += 12
        if a.get("type") == "noul":
            p = a["noul"]
            d.rectangle([80, y + 2, 80 + int(200 * p), y + 9], fill=GREEN)
            d.text((6, y), f"yes {p:.2f}", fill=TXT, font=fs)
            y += 13
        else:
            probs = a.get("probabilities", {})
            best = a.get("choice") or (max(probs, key=probs.get) if probs else "")
            for k, p in sorted(probs.items(), key=lambda kv: -kv[1])[:5]:
                d.text((6, y), k[:12], fill=TXT if k == best else DIM, font=fs)
                d.rectangle([100, y + 2, 100 + int(180 * p), y + 9], fill=GREEN if k == best else DIM)
                y += 12
            d.text((6, y), f"conf {a.get('confidence', 0):.2f}", fill=DIM, font=fs)
            y += 13
        y += 3
    return img


def write_recording(out_dir: Path, frames: list[np.ndarray], judgments: list[dict], decision_tics: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judgments.jsonl").write_text("\n".join(json.dumps(j) for j in judgments) + "\n")
    composed = []
    for fr, j in zip(frames, judgments):
        game = Image.fromarray(fr)
        game = game.resize((game.width * 2, game.height * 2), Image.NEAREST)
        p = panel(j, game.height)
        canvas = Image.new("RGB", (game.width + PANEL_W, game.height), BG)
        canvas.paste(game, (0, 0))
        canvas.paste(p, (game.width, 0))
        composed.append(canvas)
    if composed:
        duration_ms = int(1000 * decision_tics / 35)
        composed[0].save(out_dir / "episode.gif", save_all=True, append_images=composed[1:], duration=duration_ms, loop=0, optimize=False)
        composed[len(composed) // 2].save(out_dir / "sample_frame.png")
    return out_dir / "episode.gif"
