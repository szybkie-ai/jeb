"""JEB plays Atari: a game the model never trained on, as a forgetting check next to Doom.

    uv run python atari_eval.py --game pong --policy scripted --episodes 3 --decisions 400
    uv run python atari_eval.py --game pong --policy jeb --url http://127.0.0.1:8020 --model jeb-4b --image --episodes 3

Every decision: the RAM-decoded state (positions from the AtariARI annotations) and, with --image, the frame upscaled 3x,
one request with one or two typed questions, code turns the answer into the joystick. Policies: random, scripted (follow
the ball / flee the nearest ghost), jeb. Prints a JSON summary (mean score, episode length) and writes it to --out.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import httpx
import numpy as np
from PIL import Image

gym.register_envs(ale_py)

GAMES = {
    "pong": {"env": "ALE/Pong-v5", "moves": {"up": "RIGHT", "down": "LEFT", "stay": "NOOP"}, "fire": False},
    "breakout": {"env": "ALE/Breakout-v5", "moves": {"left": "LEFT", "right": "RIGHT", "stay": "NOOP"}, "fire": True},
    "mspacman": {"env": "ALE/MsPacman-v5", "moves": {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}, "fire": False},
}


def decode(game: str, ram: np.ndarray, prev: dict | None) -> dict:
    """The game's objects from RAM (AtariARI address annotations), plus code-computed relations the judgment needs."""
    r = [int(x) for x in ram]
    if game == "pong":
        s = {"player_paddle_y": r[51], "opponent_paddle_y": r[50], "ball_x": r[49], "ball_y": r[54], "player_score": r[14], "opponent_score": r[13]}
        s["ball_relative_to_paddle"] = "above" if s["ball_y"] < s["player_paddle_y"] - 4 else ("below" if s["ball_y"] > s["player_paddle_y"] + 4 else "level")
        s["ball_vertical_offset"] = s["ball_y"] - s["player_paddle_y"]
        s["ball_in_play"] = s["ball_x"] > 0
    elif game == "breakout":
        s = {"paddle_x": r[72], "ball_x": r[99], "ball_y": r[101], "blocks_hit": r[77], "score": r[84], "lives": r[57]}
        s["ball_relative_to_paddle"] = "left" if s["ball_x"] < s["paddle_x"] - 3 else ("right" if s["ball_x"] > s["paddle_x"] + 3 else "level")
        s["ball_horizontal_offset"] = s["ball_x"] - s["paddle_x"]
        s["ball_in_play"] = s["ball_y"] > 0
    else:
        px, py = r[10], r[16]
        ghosts = [{"id": f"ghost_{i + 1}", "x": r[6 + i], "y": r[12 + i]} for i in range(4)]
        for g in ghosts:
            g["dx"], g["dy"] = g["x"] - px, g["y"] - py
            g["distance"] = abs(g["dx"]) + abs(g["dy"])
        ghosts.sort(key=lambda g: g["distance"])
        s = {"player_x": px, "player_y": py, "ghosts": ghosts, "nearest_ghost": ghosts[0]["id"], "nearest_ghost_distance": ghosts[0]["distance"],
             "fruit_x": r[11], "fruit_y": r[17], "dots_eaten": r[119], "lives": r[123]}
    if prev:
        for k in ("ball_x", "ball_y"):
            if k in s and k in prev:
                s[k.replace("_", "_velocity_")] = s[k] - prev[k]
    return s


def frame_data_url(obs: np.ndarray, scale: int = 3) -> str:
    im = Image.fromarray(obs).resize((obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def questions(game: str, state: dict) -> dict:
    moves = GAMES[game]["moves"]
    if game == "pong":
        q = {"move": {"type": "choice", "instructions": "Move the player's paddle so that it meets the ball.", "criteria": {"up": "Move the paddle up", "down": "Move the paddle down", "stay": "Keep the paddle where it is"}}}
    elif game == "breakout":
        q = {"move": {"type": "choice", "instructions": "Move the paddle so that it is under the ball when the ball comes down.", "criteria": {"left": "Move the paddle left", "right": "Move the paddle right", "stay": "Keep the paddle where it is"}}}
        if not state.get("ball_in_play", True):
            q["fire"] = {"type": "noul", "instructions": "The ball is not in play. Launch it now?", "criteria": {"true": "Launch the ball", "false": "Wait"}}
    else:
        q = {"move": {"type": "choice", "instructions": "Choose the direction for Ms. Pac-Man: eat dots and keep away from the ghosts.", "criteria": {k: f"Move {k}" for k in moves}}}
    return q


class JebPolicy:
    def __init__(self, url: str, model: str, key: str = "local", image: bool = False, permutations: int = 1) -> None:
        self.url, self.model, self.image, self.permutations = url.rstrip("/"), model, image, permutations
        self.client = httpx.Client(timeout=60, headers={"authorization": f"Bearer {key}"})
        self.latencies: list[float] = []

    def __call__(self, game: str, state: dict, obs: np.ndarray) -> dict:
        body = {"model": self.model, "state": {"game": game, **state}, "questions": questions(game, state), "options": {"permutations": self.permutations}}
        if self.image:
            body["images"] = [frame_data_url(obs)]
        for attempt in range(4):
            t0 = time.perf_counter()
            resp = self.client.post(self.url + "/v1/systemone", json=body)
            if resp.status_code < 500:
                break
            time.sleep(1.5 * (attempt + 1))
        resp.raise_for_status()
        self.latencies.append(time.perf_counter() - t0)
        ans = resp.json()["answers"]
        out = {"move": ans["move"]["choice"]}
        if "fire" in ans:
            out["fire"] = float(ans["fire"]["noul"]) >= 0.5
        return out


def scripted(game: str, state: dict, obs: np.ndarray) -> dict:
    if game == "pong":
        d = state["ball_vertical_offset"]
        return {"move": "up" if d < -3 else ("down" if d > 3 else "stay")}
    if game == "breakout":
        # aim where the ball will cross the paddle's height (walls at x ~57 and ~199), not where it is now
        x, y, vx, vy = state["ball_x"], state["ball_y"], state.get("ball_velocity_x", 0), state.get("ball_velocity_y", 0)
        if vy > 0:
            x = x + vx * (190 - y) / vy
            while x < 57 or x > 199:
                x = 114 - x if x < 57 else 398 - x
        d = x - state["paddle_x"]
        return {"move": "left" if d < -4 else ("right" if d > 4 else "stay"), "fire": True}
    g = state["ghosts"][0]
    if g["distance"] < 40:
        return {"move": ("left" if g["dx"] > 0 else "right") if abs(g["dx"]) >= abs(g["dy"]) else ("up" if g["dy"] > 0 else "down")}
    return {"move": random.choice(list(GAMES[game]["moves"]))}


def run_episode(game: str, policy, seed: int, decisions: int, hold: int, record: Path | None) -> dict:
    spec = GAMES[game]
    env = gym.make(spec["env"], obs_type="rgb", frameskip=4, repeat_action_probability=0.0, full_action_space=False)
    meanings = env.unwrapped.get_action_meanings()
    obs, _ = env.reset(seed=seed)
    total, prev, frames, n = 0.0, None, [], 0
    for n in range(1, decisions + 1):
        state = decode(game, env.unwrapped.ale.getRAM(), prev); prev = state
        d = policy(game, state, obs)
        name = spec["moves"][d["move"]]
        if d.get("fire") and spec["fire"] and not state.get("ball_in_play", True):
            name = "FIRE" if name + "FIRE" not in meanings else name + "FIRE"  # launch only when the ball is out; otherwise the move wins
        action = meanings.index(name) if name in meanings else 0
        done = False
        for _ in range(hold):
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            if record and len(frames) < 1500:
                frames.append(Image.fromarray(obs))
            if term or trunc:
                done = True
                break
        if done:
            break
    env.close()
    if record and frames:
        record.mkdir(parents=True, exist_ok=True)
        frames[0].save(record / f"{game}-{seed}.gif", save_all=True, append_images=frames[1::2], duration=60, loop=0)
    return {"seed": seed, "score": total, "decisions": n, "finished": done}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=list(GAMES), default="pong"); ap.add_argument("--policy", choices=["random", "scripted", "jeb"], default="scripted")
    ap.add_argument("--url", default="http://127.0.0.1:8020"); ap.add_argument("--model", default="jeb-4b"); ap.add_argument("--key", default="local")
    ap.add_argument("--image", action="store_true"); ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=3); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--hold", type=int, default=2, help="env steps (x4 frames) per decision"); ap.add_argument("--record", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.policy == "random":
        policy = lambda game, state, obs: {"move": random.choice(list(GAMES[game]["moves"])), "fire": True}  # noqa: E731
    elif a.policy == "scripted":
        policy = scripted
    else:
        policy = JebPolicy(a.url, a.model, a.key, image=a.image, permutations=a.permutations)
    random.seed(a.seed)
    eps = [run_episode(a.game, policy, a.seed + i, a.decisions, a.hold, Path(a.record) if a.record else None) for i in range(a.episodes)]
    scores = [e["score"] for e in eps]
    summary = {"game": a.game, "policy": a.policy, "model": a.model if a.policy == "jeb" else None, "image": a.image, "episodes": eps,
               "mean_score": statistics.fmean(scores), "mean_decisions": statistics.fmean(e["decisions"] for e in eps),
               "latency_p50_s": (sorted(policy.latencies)[len(policy.latencies) // 2] if isinstance(policy, JebPolicy) and policy.latencies else None)}
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
