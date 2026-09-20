"""The loop: every `decision_tics` ticks compile the state, ask JEB everything at once, act until the next decision.

Runs the game in synchronous PLAYER mode: the engine waits for each decision, so the recording plays at the game's
tick rate whatever the model's latency; wall-clock latency is reported separately (that is the real-time claim).
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import base64
import io

import httpx
import numpy as np
import vizdoom as vzd

from jeb_doom.control import BUTTONS, IDX, actions_for
from jeb_doom.questions import apply_answers, build_questions
from jeb_doom.state import GAME_VARIABLES, Memory, Tracker, compile_state


def make_game(scenario: str, visible: bool = False, seed: int | None = None, doom_map: str | None = None, skill: int = 3, wad: str | None = None) -> vzd.DoomGame:
    """A scenario config (`deathmatch`, `deadly_corridor`, ...) or, with `doom_map`, a full level of the bundled freedoom2.wad
    (MAP01..MAP32) or of `wad` (e.g. a doom2.wad you own) at the given skill (1 easy .. 5 nightmare)."""
    g = vzd.DoomGame()
    if doom_map:
        if wad:
            g.set_doom_game_path(wad)
        g.set_doom_map(doom_map)
        g.set_doom_skill(skill)
    else:
        g.load_config(vzd.scenarios_path + f"/{scenario}.cfg")
    g.set_window_visible(visible)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_objects_info_enabled(True)
    g.set_labels_buffer_enabled(True)
    g.set_sectors_info_enabled(True)
    g.set_available_game_variables(GAME_VARIABLES)
    g.set_available_buttons(BUTTONS)
    g.set_episode_timeout(0)
    g.set_render_hud(True)
    if seed is not None:
        g.set_seed(seed)
    g.init()
    return g


def calibrate_turn_sign(g: vzd.DoomGame) -> float:
    """+1 if a positive TURN_LEFT_RIGHT_DELTA increases the angle (turns left), else -1."""
    g.new_episode()
    a0 = g.get_game_variable(vzd.GameVariable.ANGLE)
    act = [0.0] * len(BUTTONS)
    act[IDX[vzd.Button.TURN_LEFT_RIGHT_DELTA]] = 10.0
    g.make_action(act, 1)
    a1 = g.get_game_variable(vzd.GameVariable.ANGLE)
    d = (a1 - a0 + 180) % 360 - 180
    return 1.0 if d > 0 else -1.0


class JebPolicy:
    def __init__(self, url: str, key: str | None = None, model: str = "jeb-latest", permutations: int = 1, collect: bool = False, image: bool = False, plain: bool = False) -> None:
        """`collect` asks for raw per-presentation logprobs so a teacher's play doubles as labelled training data.
        `image` attaches the current frame (the player's view) to every request, for vision-language models.
        `plain` sends the bare System One request (no JEB extensions): for the original API."""
        self.client = httpx.Client(base_url=url, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=300)
        self.model, self.permutations, self.collect, self.image, self.plain = model, permutations, collect, image and not plain, plain
        self.latencies: list[float] = []
        self.tokens = 0
        self.last_debug: dict | None = None
        self.model_id: str | None = None

    def decide(self, state: dict[str, Any], raw: dict[str, Any], mem: Memory, questions: dict | None = None, frame_png: bytes | None = None) -> tuple[dict, dict]:
        questions = questions or build_questions(state, raw, mem)
        opts = {"permutations": self.permutations}
        if self.collect:
            opts.update({"debug": True, "calibration": "raw"})
        body = {"state": state, "model": self.model, "questions": questions}
        if not self.plain:
            body["options"] = opts
        if self.image and frame_png is not None:
            body["images"] = ["data:image/png;base64," + base64.b64encode(frame_png).decode()]
        t0 = time.perf_counter()
        r = None
        for attempt in range(8):
            # A server restart or an engine reload must not kill an hours-long run: back off and retry on transport
            # errors and 5xx (529 is the server's "engine unavailable").
            try:
                r = self.client.post("/v1/systemone", json=body)
            except httpx.HTTPError as e:
                print(f"[policy] request failed ({e.__class__.__name__}), retry {attempt + 1}/8 in {5 * (attempt + 1)} s", flush=True)
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 500:
                print(f"[policy] HTTP {r.status_code}, retry {attempt + 1}/8 in {5 * (attempt + 1)} s", flush=True)
                time.sleep(5 * (attempt + 1))
                continue
            break
        self.latencies.append(time.perf_counter() - t0)
        if r is None or r.status_code != 200:
            raise RuntimeError(f"JEB HTTP {r.status_code if r is not None else 'unreachable'}: {r.text[:400] if r is not None else ''}\nquestions: {json.dumps(questions)[:600]}")
        data = r.json()
        self.tokens += int((data.get("usage") or {}).get("input_tokens") or 0)
        self.model_id = data.get("model")
        self.last_debug = {qid: {"presentations": [{"labels": p["labels"], "targets": p["targets"], "logprobs": p["logprobs"]} for p in d["permutations"]], "in_set_mass": d["in_set_mass"]} for qid, d in data["debug"].items() if not qid.startswith("_")} if (self.collect and not self.plain and data.get("debug")) else None
        return apply_answers(data["answers"], raw, mem), data["answers"]


class ScriptedPolicy:
    """Baseline: face the nearest enemy and fire; walk toward it; no dodging."""

    latencies: list[float] = []
    tokens = 0

    def decide(self, state, raw, mem, questions=None, frame_png=None):
        e = raw["enemies"][0] if raw["enemies"] else None
        mem.goal, mem.subject = ("kill_enemies", e["id"]) if e else ("scout", None)
        return {"goal": mem.goal, "subject": mem.subject, "subject_obj": e, "move": "close_in" if e else "explore_ahead", "dodge": "carry_on", "fire": bool(e and e["distance"] < 900), "walk_item": None, "confidence": {}}, {}


class RandomPolicy:
    latencies: list[float] = []
    tokens = 0

    def decide(self, state, raw, mem, questions=None, frame_png=None):
        rng = random.Random()
        return {"goal": "scout", "subject": None, "subject_obj": None, "move": rng.choice(["close_in", "back_off", "strafe_left", "strafe_right", "hold_ground"]), "dodge": "carry_on", "fire": rng.random() < 0.5, "walk_item": None, "confidence": {}}, {}


DIRECTOR_POOL = ["Zombieman", "Zombieman", "ShotgunGuy", "ShotgunGuy", "DoomImp", "DoomImp", "Demon", "ChaingunGuy", "Cacodemon", "LostSoul"]


def encode_frame(screen_buffer) -> bytes:
    """The engine's RGB24 frame as PNG bytes (about 30 KB at 320x240)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.array(screen_buffer)).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def run_episode(g: vzd.DoomGame, policy, decision_tics: int = 4, max_decisions: int = 600, orders: str = "", record_dir: Path | None = None, sign: float = 1.0, log_path: Path | None = None, episode_id: str = "0", director_every: int = 0, wake: bool = False, director_seed: int = 0) -> dict[str, Any]:
    """`wake`: fire one shot at the start so the level's monsters are alerted. `director_every` > 0: when no enemy has been
    visible for that many decisions, summon one monster ahead through the engine console (a fallback against dead air,
    not a spawn fountain)."""
    g.new_episode()
    tracker, mem = Tracker(), Memory(orders=orders)
    rng = random.Random(director_seed)
    spawned = 0
    last_visible, last_summon, summon_pending = 0, -10**9, None
    if wake:
        shot = [0.0] * len(BUTTONS)
        shot[IDX[vzd.Button.ATTACK]] = 1
        g.make_action(shot, 2)
        g.make_action([0.0] * len(BUTTONS), 6)
    frames, judgments = [], []
    n = 0
    t_start = time.perf_counter()
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = log_path.open("a") if log_path else None  # append: one log holds every episode of a run
    while not g.is_episode_finished() and n < max_decisions:
        state, raw = compile_state(g, tracker, mem, decision_tics)
        frame_png = encode_frame(g.get_state().screen_buffer)
        frame_rel = None
        if logf:
            frame_rel = f"frames/{episode_id}-{n:04d}.png"
            (log_path.parent / "frames").mkdir(exist_ok=True)
            (log_path.parent / frame_rel).write_bytes(frame_png)
        if getattr(policy, "image", False):
            state["screen"] = "the attached image is the player's current view (320x240); the strip at the bottom is the HUD (ammo, health, arms, armor)"
        if any(e.get("visible") for e in raw["enemies"]):
            last_visible = n
            summon_pending = None
        elif summon_pending is not None and n - summon_pending > 12:
            # The last summon never came into view (dropped behind something): stop feeding that spot for a long while.
            summon_pending, last_summon = None, n + 300
        if director_every and n - last_visible >= director_every and n - last_summon >= director_every and raw["walls"].get("ahead", 0) >= 128:
            # Fallback only: the level's own monsters are the fight; when nothing has been in view for `director_every`
            # decisions, summon one monster ahead (into open space: `summon` ignores walls and doors) to end the dead air.
            g.send_game_command(f"summon {rng.choice(DIRECTOR_POOL)}")
            spawned += 1
            last_summon = n
            summon_pending = n
        questions = build_questions(state, raw, mem)
        decision, answers = policy.decide(state, raw, mem, questions, frame_png)
        if record_dir is not None:
            frames.append(np.array(g.get_state().screen_buffer))
            judgments.append({"tic": g.get_episode_time(), "latency_ms": round(1000 * policy.latencies[-1], 1) if getattr(policy, "latencies", None) else None, "decision": {k: v for k, v in decision.items() if k not in ("subject_obj", "walk_item")}, "answers": answers})
        if logf:
            logf.write(json.dumps({"episode": episode_id, "map": g.get_doom_map() if hasattr(g, "get_doom_map") else None, "tic": g.get_episode_time(), "frame": frame_rel, "state": state, "questions": questions, "answers": answers, "decision": {k: v for k, v in decision.items() if k not in ("subject_obj", "walk_item")},
                                   "teacher": getattr(policy, "model_id", None) if getattr(policy, "collect", False) else None, "debug": getattr(policy, "last_debug", None)}) + "\n")
        for act, tics in actions_for(decision, raw, decision_tics):
            act = list(act)
            act[IDX[vzd.Button.TURN_LEFT_RIGHT_DELTA]] *= sign
            g.make_action(act, tics)
            if g.is_episode_finished():
                break
        n += 1
    wall = time.perf_counter() - t_start
    kills = g.get_game_variable(vzd.GameVariable.KILLCOUNT)
    dead = g.is_player_dead()
    tics = g.get_episode_time()
    if logf:
        logf.close()
    result = {
        "decisions": n, "explored_cells": len(mem.visited), "game_tics": int(tics), "game_seconds": round(tics / 35, 1), "wall_seconds": round(wall, 1),
        "kills": int(kills), "died": bool(dead), "health": g.get_game_variable(vzd.GameVariable.HEALTH), "spawned": spawned,
        "kills_per_game_minute": round(kills / max(1e-9, tics / 35 / 60), 2),
        "latency_p50_ms": round(1000 * statistics.median(policy.latencies), 1) if policy.latencies else None,
        "latency_p90_ms": round(1000 * sorted(policy.latencies)[int(0.9 * len(policy.latencies))], 1) if policy.latencies else None,
        "input_tokens": policy.tokens,
    }
    if record_dir is not None:
        from jeb_doom.record import write_recording

        result["recording"] = str(write_recording(record_dir, frames, judgments, decision_tics))
    return result
