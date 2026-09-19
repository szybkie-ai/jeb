"""Four chart panels per decision (1m, 15m, 4h, 1d) as one 2x2 image: candles, EMAs on the higher timeframes, the open
position's entry/stop/target. No dates anywhere: axes are relative ("-6h"), so a model cannot recall what the market did
on a known day; the weekday and time of day live in the state."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

# frame: (resample rule, bars shown, minutes per bar, EMAs drawn, title)
FRAMES = {
    "1m": ("1min", 120, 1, (20,), "1-minute · last 2 hours"),
    "15m": ("15min", 96, 15, (20, 50, 200), "15-minute · last 24 hours"),
    "4h": ("4h", 120, 240, (20, 50, 200), "4-hour · last 20 days"),
    "1d": ("1D", 120, 1440, (20, 50, 200), "daily · last 6 months"),
}
STYLE = {"bg": "#111118", "fg": "#d8d8e4", "up": "#2fd07a", "down": "#e0514f", "grid": "#26262f", "ema20": "#f2c14e", "ema50": "#5aa9ff", "ema200": "#c56cf0"}
HISTORY_BARS = 220  # extra bars before the window so an EMA200 is settled


def resample(bars_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = pd.DataFrame({"open": bars_1m["open"].resample(rule).first(), "high": bars_1m["high"].resample(rule).max(),
                       "low": bars_1m["low"].resample(rule).min(), "close": bars_1m["close"].resample(rule).last()})
    return df.dropna(subset=["open"])


def window(bars_1m: pd.DataFrame, i: int, frame: str) -> pd.DataFrame:
    rule, n, minutes, emas, _ = FRAMES[frame]
    j = max(0, i - (n + HISTORY_BARS) * minutes)
    sub = bars_1m.iloc[j:i + 1]
    df = (sub[["open", "high", "low", "close"]] if frame == "1m" else resample(sub, rule)).copy()
    for k in emas:
        df[f"ema{k}"] = df["close"].ewm(span=k, adjust=False).mean()
    return df.iloc[-n:]


def _offset_label(bars_ago: int, minutes: int) -> str:
    m = bars_ago * minutes
    if m == 0:
        return "now"
    if m < 120:
        return f"-{m}m"
    if m < 48 * 60:
        return f"-{m // 60}h"
    return f"-{m // 1440}d"


def render(df: pd.DataFrame, frame: str, position: dict | None, digits: int, size=(5.6, 3.2), dpi=100) -> bytes:
    _, n, minutes, emas, title = FRAMES[frame]
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    fig.patch.set_facecolor(STYLE["bg"]); ax.set_facecolor(STYLE["bg"])
    x = np.arange(len(df))
    up = df["close"].to_numpy() >= df["open"].to_numpy()
    colors = np.where(up, STYLE["up"], STYLE["down"])
    ax.vlines(x, df["low"], df["high"], color=colors, linewidth=0.8)
    ax.bar(x, (df["close"] - df["open"]).abs().to_numpy() + 1e-12, bottom=np.minimum(df["open"], df["close"]).to_numpy(), color=colors, width=0.7, linewidth=0)
    for k in emas:
        ax.plot(x, df[f"ema{k}"], color=STYLE[f"ema{k}"], linewidth=1.0, label=f"EMA{k}")
    if position:
        for key, col, ls in (("entry", "#ffffff", "-"), ("stop", STYLE["down"], "--"), ("target", STYLE["up"], "--")):
            v = position.get(key)
            if v is not None and df["low"].min() * 0.98 < v < df["high"].max() * 1.02:
                ax.axhline(v, color=col, linestyle=ls, linewidth=0.9)
    ax.set_xlim(-1, len(df))
    ticks = np.linspace(0, len(df) - 1, 5).astype(int)
    ax.set_xticks(ticks); ax.set_xticklabels([_offset_label(len(df) - 1 - t, minutes) for t in ticks], fontsize=7, color=STYLE["fg"])
    ax.tick_params(axis="y", labelsize=7, colors=STYLE["fg"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f"%.{digits}f"))
    ax.grid(color=STYLE["grid"], linewidth=0.5)
    for s in ax.spines.values():
        s.set_color(STYLE["grid"])
    ax.set_title(f"{title}   last {df['close'].iloc[-1]:.{digits}f}", fontsize=8, color=STYLE["fg"], loc="left")
    if emas:
        ax.legend(loc="upper left", fontsize=6, frameon=False, labelcolor=STYLE["fg"])
    fig.tight_layout(pad=0.6)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=STYLE["bg"]); plt.close(fig)
    return buf.getvalue()


def charts(bars_1m: pd.DataFrame, i: int, position: dict | None, digits: int) -> dict[str, bytes]:
    return {f: render(window(bars_1m, i, f), f, position, digits) for f in FRAMES}


def composite(images: dict[str, bytes]) -> bytes:
    """The four panels as one image: 1m top-left, 15m top-right, 4h bottom-left, 1d bottom-right."""
    ims = {k: Image.open(io.BytesIO(v)).convert("RGB") for k, v in images.items()}
    w, h = ims["1m"].width, ims["1m"].height
    out = Image.new("RGB", (2 * w, 2 * h), (17, 17, 24))
    for k, (cx, cy) in {"1m": (0, 0), "15m": (w, 0), "4h": (0, h), "1d": (w, h)}.items():
        out.paste(ims[k], (cx, cy))
    buf = io.BytesIO(); out.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
