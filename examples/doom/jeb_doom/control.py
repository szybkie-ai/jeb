"""Turn a decision into ViZDoom button states. Aiming is a proportional turn toward the subject's bearing (applied on the
first tick of a decision), movement/dodge/trigger persist for the whole decision window. Weapon choice is code-owned."""

from __future__ import annotations

import math

import vizdoom as vzd

from jeb_doom.state import CELL, ray_distance, rel_bearing

BUTTONS = [
    vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD, vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
    vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.SELECT_WEAPON2, vzd.Button.SELECT_WEAPON3, vzd.Button.SELECT_WEAPON4,
    vzd.Button.SELECT_WEAPON5, vzd.Button.SELECT_WEAPON6, vzd.Button.USE,
]
IDX = {b: i for i, b in enumerate(BUTTONS)}
MAX_TURN_PER_TIC = 45.0  # degrees; the delta button is applied once per decision window
EXPLORE_HEADINGS = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 135.0, -135.0, 180.0)


def explore_heading(raw: dict) -> tuple[float, float | None]:
    """Code-owned exploration: the most open heading, weighted toward map cells not visited yet and toward carrying
    straight on (so the walk does not jitter); a closed door counts as an opening. Returns (relative turn in degrees,
    distance to a door on that heading if it is close enough to use)."""
    px, py, pa = raw["player"]
    lines, visited = raw.get("lines") or [], raw.get("visited") or set()
    best, best_turn, best_door = -1.0, 0.0, None
    for rel in EXPLORE_HEADINGS:
        dist, is_door = ray_distance(lines, px, py, pa + rel, 800.0, raw.get("dead_doors"))
        a = math.radians(pa + rel)
        novelty = 1.0
        for probe in (160.0, 320.0):
            if probe > dist:
                break
            if (int((px + probe * math.cos(a)) // CELL), int((py + probe * math.sin(a)) // CELL)) in visited:
                novelty *= 0.6
        straight = 1.0 if abs(rel) <= 30 else 0.85 if abs(rel) <= 90 else 0.65
        score = min(dist + (250.0 if is_door else 0.0), 800.0) * novelty * straight
        if score > best:
            best, best_turn, best_door = score, rel, (dist if is_door and dist < 100 else None)
    return best_turn, best_door


def best_weapon(vars_: dict[str, float]) -> int | None:
    """The strongest owned weapon with ammo, as a SELECT_WEAPONn slot; None to keep the current one."""
    prefs = [(6, "AMMO5", "WEAPON6"), (5, "AMMO4", "WEAPON5"), (4, "AMMO2", "WEAPON4"), (3, "AMMO3", "WEAPON3"), (2, "AMMO2", None)]
    for slot, ammo, owned in prefs:
        if (owned is None or vars_.get(owned, 0) > 0) and vars_.get(ammo, 0) > 0:
            return slot
    return None


def actions_for(decision: dict, raw: dict, decision_tics: int) -> list[tuple[list[float], int]]:
    """A list of (button vector, tics) pairs covering one decision window."""
    px, py, pa = raw["player"]
    base = [0.0] * len(BUTTONS)
    target = decision.get("walk_item") if decision.get("walk_item") and decision["goal"] != "kill_enemies" else decision.get("subject_obj")
    turn = 0.0
    if target is not None:
        o = target["_obj"]
        turn = rel_bearing(px, py, pa, o.position_x, o.position_y)
    elif decision["move"] == "turn_around":
        turn = 180.0
    move, dodge = decision["move"], decision["dodge"]
    walls = raw.get("walls", {})
    # Closing in on something we cannot see and cannot walk straight to (a wall, a ledge, a barrier in the way) would
    # mean pushing against geometry: explore instead and let the map bring us round.
    if move in ("close_in", "walk_to_item") and target is not None and not target.get("visible", False):
        o = target["_obj"]
        dist = math.hypot(o.position_x - px, o.position_y - py)
        reach, _ = ray_distance(raw.get("lines") or [], px, py, pa + turn, dist)
        if not target.get("line_of_sight", True) or reach < dist - 40:
            move, turn = "explore_ahead", 0.0
    if raw.get("stuck", 0) >= 5 and move in ("close_in", "walk_to_item", "explore_ahead"):
        # Pinned for five decisions against something no ray saw: turn well away and let exploration re-plan.
        move, turn = "explore_ahead", 150.0
        if raw.get("mem") is not None:
            raw["mem"].stuck = 0
        at_door, door = False, None
    elif move == "explore_ahead":
        # Exploring: code picks the heading (open, unvisited, doors welcome); the model only decided *that* we explore.
        turn, door = explore_heading(raw)
        at_door = door is not None
    else:
        # A closed door ahead is not a wall: walk into it and press USE (Doom opens doors on use), no steering away.
        door = raw.get("door_ahead")
        at_door = door is not None and door < 100 and move in ("close_in", "walk_to_item")
        if at_door:
            turn = 0.0
        # Code-owned obstacle avoidance: about to walk into a wall -> steer toward the more open side instead of pushing.
        if not at_door and move in ("close_in", "walk_to_item") and walls.get("ahead", 999) < 120 and abs(turn) < 30:
            turn = 45.0 if walls.get("left", 0) >= walls.get("right", 0) else -45.0
    if at_door and raw.get("door_tries") is not None:
        # Remember which door we are pushing; one that stays shut after a dozen presses is a wall from now on.
        a = math.radians(pa + turn)
        key = (int((px + door * math.cos(a)) // CELL), int((py + door * math.sin(a)) // CELL))
        raw["door_tries"][key] = raw["door_tries"].get(key, 0) + 1
        if raw["door_tries"][key] > 12 and raw.get("dead_doors") is not None:
            raw["dead_doors"].add(key)
    if dodge == "dodge_left":
        base[IDX[vzd.Button.MOVE_LEFT]] = 1
    elif dodge == "dodge_right":
        base[IDX[vzd.Button.MOVE_RIGHT]] = 1
    elif dodge == "dodge_back":
        base[IDX[vzd.Button.MOVE_BACKWARD]] = 1
    elif move in ("close_in", "walk_to_item", "explore_ahead"):
        base[IDX[vzd.Button.MOVE_FORWARD]] = 1
    elif move == "back_off":
        base[IDX[vzd.Button.MOVE_BACKWARD]] = 1
    elif move == "strafe_left":
        base[IDX[vzd.Button.MOVE_LEFT]] = 1
    elif move == "strafe_right":
        base[IDX[vzd.Button.MOVE_RIGHT]] = 1
    if decision["fire"] and raw["vars"].get("SELECTED_WEAPON_AMMO", 0) > 0:
        base[IDX[vzd.Button.ATTACK]] = 1
    slot = best_weapon(raw["vars"])
    if slot is not None and int(raw["vars"].get("SELECTED_WEAPON", 0)) != slot and raw["vars"].get("SELECTED_WEAPON_AMMO", 0) <= 0:
        base[IDX[getattr(vzd.Button, f"SELECT_WEAPON{slot}")]] = 1
    if raw.get("mem") is not None:
        raw["mem"].last_forward = bool(base[IDX[vzd.Button.MOVE_FORWARD]]) and abs(turn) < 60
    # ViZDoom's turn delta is in degrees per tic and positive turns LEFT (counter-clockwise), like the angle convention.
    steps: list[tuple[list[float], int]] = []
    remaining = decision_tics
    t = turn
    while remaining > 0:
        a = list(base)
        if at_door and not steps:
            a[IDX[vzd.Button.USE]] = 1  # USE is edge-triggered in Doom: press it on the first tic of the window only
        if abs(t) > 1.0:
            step = max(-MAX_TURN_PER_TIC, min(MAX_TURN_PER_TIC, t))
            a[IDX[vzd.Button.TURN_LEFT_RIGHT_DELTA]] = step
            t -= step
            steps.append((a, 1))
            remaining -= 1
        else:
            steps.append((a, remaining))
            remaining = 0
    return steps
