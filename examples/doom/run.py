"""JEB plays Doom.

    uv run python run.py --url http://127.0.0.1:8020 --scenario deathmatch --decisions 300 --record out/ep1
    uv run python run.py --policy scripted --scenario deathmatch --decisions 300      # baseline, no model
    uv run python run.py --url ... --orders "Do not fire, simply dodge"
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

from jeb_doom.agent import JebPolicy, RandomPolicy, ScriptedPolicy, calibrate_turn_sign, make_game, run_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8020")
    ap.add_argument("--key", default=os.environ.get("JEB_API_KEY") or os.environ.get("TYPESAFE_API_KEY"), help="API key (or JEB_API_KEY / TYPESAFE_API_KEY in the environment)")
    ap.add_argument("--model", default="jeb-latest")
    ap.add_argument("--policy", choices=["jeb", "scripted", "random"], default="jeb")
    ap.add_argument("--scenario", default="deathmatch", help="a ViZDoom scenario config; ignored when --map is given")
    ap.add_argument("--map", default=None, help="full level(s) instead of a scenario: MAP01 or a comma list cycled per episode (freedoom2 unless --wad)")
    ap.add_argument("--wad", default=None, help="IWAD path (default: the bundled freedoom2.wad)")
    ap.add_argument("--skill", type=int, default=3)
    ap.add_argument("--decisions", type=int, default=300)
    ap.add_argument("--decision-tics", type=int, default=4)
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orders", default="")
    ap.add_argument("--record", default=None, help="directory for episode.gif + judgments.jsonl")
    ap.add_argument("--log", default=None, help="per-tick state/answers JSONL")
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--collect", action="store_true", help="store raw per-presentation logprobs in the tick log (teacher play as training data)")
    ap.add_argument("--director", type=int, default=0, help="summon one monster when no enemy has been visible for N decisions (0 = off)")
    ap.add_argument("--wake", action="store_true", help="fire one shot at episode start to alert the level's monsters")
    ap.add_argument("--plain", action="store_true", help="bare System One requests, no JEB extensions (for the original API: --url https://api.typesafe.ai --model jev-latest)")
    ap.add_argument("--image", action="store_true", help="attach the current frame to every request (vision-language model behind the server)")
    a = ap.parse_args()

    maps = [m.strip() for m in a.map.split(",")] if a.map else [None]
    g = make_game(a.scenario, visible=a.visible, seed=a.seed, doom_map=maps[0], skill=a.skill, wad=a.wad)
    sign = calibrate_turn_sign(g)
    policy = {"jeb": lambda: JebPolicy(a.url, a.key, a.model, a.permutations, collect=a.collect, image=a.image, plain=a.plain), "scripted": ScriptedPolicy, "random": RandomPolicy}[a.policy]()
    results = []
    for ep in range(a.episodes):
        if maps[ep % len(maps)]:
            g.set_doom_map(maps[ep % len(maps)])
        rec = Path(a.record) / f"ep{ep}" if a.record and a.episodes > 1 else (Path(a.record) if a.record else None)
        r = run_episode(g, policy, a.decision_tics, a.decisions, a.orders, rec, sign, Path(a.log) if a.log else None, episode_id=f"{a.seed}-{ep}-{maps[ep % len(maps)] or a.scenario}", director_every=a.director, wake=a.wake, director_seed=a.seed + ep)
        print(json.dumps({"episode": ep, "policy": a.policy, "turn_sign": sign, **r}))
        results.append(r)
    g.close()
    if len(results) > 1:
        print(json.dumps({"mean_kills_per_game_minute": round(sum(r["kills_per_game_minute"] for r in results) / len(results), 2), "deaths": sum(r["died"] for r in results)}))


if __name__ == "__main__":
    main()
