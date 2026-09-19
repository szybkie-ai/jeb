"""Compile ViZDoom's engine state into the JSON a System One model can judge.

Code does the geometry (bearings, distances, wall ray-casts, projectile closest approach, stable letter ids);
the model only judges. Nothing here is Doom-specific beyond the object-name tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import vizdoom as vzd

ENEMIES = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "Cacodemon", "LostSoul", "HellKnight",
    "BaronOfHell", "Revenant", "Mancubus", "Arachnotron", "PainElemental", "Archvile", "Cyberdemon", "SpiderMastermind",
    "WolfensteinSS", "MarineChainsawVzd", "MarineBFG", "MarineShotgun", "MarineChaingun", "MarineRocket", "MarinePlasma",
}
ITEMS = {
    "Stimpack": ("health", "+10 health"), "Medikit": ("health", "+25 health"), "HealthBonus": ("health", "+1 health"), "Soulsphere": ("health", "+100 health"),
    "ArmorBonus": ("armor", "+1 armor"), "GreenArmor": ("armor", "100 armor"), "BlueArmor": ("armor", "200 armor"),
    "Clip": ("ammo", "bullets"), "ClipBox": ("ammo", "box of bullets"), "Shells": ("ammo", "shells"), "ShellBox": ("ammo", "box of shells"),
    "RocketAmmo": ("ammo", "rocket"), "RocketBox": ("ammo", "box of rockets"), "Cell": ("ammo", "cells"), "CellPack": ("ammo", "cell pack"),
    "Shotgun": ("weapon", "shotgun"), "SuperShotgun": ("weapon", "super shotgun"), "Chaingun": ("weapon", "chaingun"),
    "RocketLauncher": ("weapon", "rocket launcher"), "PlasmaRifle": ("weapon", "plasma rifle"), "BFG9000": ("weapon", "BFG"), "Chainsaw": ("weapon", "chainsaw"),
}
PROJECTILES = {"Rocket", "PlasmaBall", "DoomImpBall", "CacodemonBall", "BaronBall", "ArachnotronPlasma", "RevenantTracer", "FatShot", "BFGBall"}
WEAPON_NAMES = {0: "fist", 1: "chainsaw or fist", 2: "pistol", 3: "shotgun", 4: "chaingun", 5: "rocket launcher", 6: "plasma rifle", 7: "BFG"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WEAPON_RANGE = {"fist": 64, "chainsaw or fist": 64, "pistol": 900, "shotgun": 500, "chaingun": 1100, "rocket launcher": 1400, "plasma rifle": 1200, "BFG": 1400}

GAME_VARIABLES = [
    vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.SELECTED_WEAPON, vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.POSITION_Z, vzd.GameVariable.ANGLE, vzd.GameVariable.VELOCITY_X, vzd.GameVariable.VELOCITY_Y,
    vzd.GameVariable.KILLCOUNT, vzd.GameVariable.DAMAGE_TAKEN, vzd.GameVariable.HITCOUNT, vzd.GameVariable.AMMO2, vzd.GameVariable.AMMO3,
    vzd.GameVariable.AMMO4, vzd.GameVariable.AMMO5, vzd.GameVariable.AMMO6, vzd.GameVariable.WEAPON3, vzd.GameVariable.WEAPON4,
    vzd.GameVariable.WEAPON5, vzd.GameVariable.WEAPON6,
]


def rel_bearing(px: float, py: float, pangle: float, x: float, y: float) -> float:
    """Degrees from the player's facing to (x, y): 0 = dead ahead, positive = to the left (Doom angles grow counter-clockwise)."""
    a = math.degrees(math.atan2(y - py, x - px)) - pangle
    return (a + 180) % 360 - 180


def side(bearing: float) -> str:
    b = abs(bearing)
    where = "ahead" if b < 15 else "ahead-left" if bearing > 0 and b < 60 else "ahead-right" if b < 60 else "left" if bearing > 0 and b < 120 else "right" if b < 120 else "behind"
    return where


@dataclass
class Tracker:
    """Stable letter ids per object id, so 'cacodemon A' stays A across ticks."""

    ids: dict[int, str] = field(default_factory=dict)
    next_letter: dict[str, int] = field(default_factory=dict)

    def letter(self, kind: str, obj_id: int) -> str:
        if obj_id not in self.ids:
            n = self.next_letter.get(kind, 0)
            self.ids[obj_id] = LETTERS[n % 26] + ("" if n < 26 else str(n // 26))
            self.next_letter[kind] = n + 1
        return self.ids[obj_id]


@dataclass
class Memory:
    last_damage: float = 0.0
    last_kills: float = 0.0
    last_hits: float = 0.0
    last_health: float = 100.0
    damage_log: list = field(default_factory=list)  # (tic, damage) pairs for the last-second trend
    goal: str = "kill_enemies"
    subject: str | None = None
    orders: str = ""
    visited: set = field(default_factory=set)  # coarse map cells the player has stood in (exploration memory)
    door_tries: dict = field(default_factory=dict)  # hit cell -> USE presses at that door
    dead_doors: set = field(default_factory=set)  # doors that stayed shut after several USE presses: walls from now on
    bumps: list = field(default_factory=list)  # invisible barriers learned by walking into them (synthetic wall segments)
    last_pos: tuple | None = None
    last_forward: bool = False
    stuck: int = 0  # consecutive decisions spent pushing forward without moving
    faced_unseen: dict = field(default_factory=dict)  # enemy id -> decisions faced (in 2D line of sight) without appearing on screen


PLAYER_HEIGHT = 56.0
MAX_ENEMIES, MAX_ITEMS = 8, 8  # listed per tick (in view first, then line of sight, then nearest)
STEP_HEIGHT = 24.0


def blocking_lines(sectors, pz: float | None = None) -> list[tuple[float, float, float, float, bool, bool, bool]]:
    """Map lines that matter, as (x1, y1, x2, y2, blocks_walking, blocks_sight, is_door). ViZDoom flags only one-sided
    and explicitly blocking lines; a two-sided line is impassable when the gap between the higher floor and the lower
    ceiling is under the player's height (a shut door, a low ceiling) or when the far floor is more than a step above the
    near one (a ledge: dropping down is fine, climbing is not; `pz` = the player's z picks the near side), and it blocks
    sight only when that gap is nothing. A shut door is a closed sector on one side (ceiling on the floor): walking into
    it and pressing USE may open it."""
    seg: dict[tuple, dict] = {}
    for sec in sectors or []:
        for l in sec.lines:
            k1 = (round(l.x1), round(l.y1), round(l.x2), round(l.y2))
            key = min(k1, (k1[2], k1[3], k1[0], k1[1]))
            e = seg.setdefault(key, {"blocking": False, "sectors": []})
            e["blocking"] = e["blocking"] or bool(l.is_blocking)
            fc = (float(sec.floor_height), float(sec.ceiling_height))
            if fc not in e["sectors"]:
                e["sectors"].append(fc)
    out = []
    for (x1, y1, x2, y2), e in seg.items():
        secs = e["sectors"]
        if e["blocking"] or len(secs) < 2:
            out.append((x1, y1, x2, y2, True, True, False))
            continue
        (f1, c1), (f2, c2) = secs[0], secs[1]
        gap = min(c1, c2) - max(f1, f2)
        closed = gap < PLAYER_HEIGHT
        door = closed and (c1 <= f1 + 1 or c2 <= f2 + 1)
        if pz is None:
            climb = abs(f1 - f2)
        else:
            near, far = (f1, f2) if abs(f1 - pz) <= abs(f2 - pz) else (f2, f1)
            climb = far - near
        walk = closed or climb > STEP_HEIGHT
        sight = gap <= 0
        if walk or sight:
            out.append((x1, y1, x2, y2, walk, sight, door))
    return out


OBSTACLE_SKIP = ("Dead", "Gibbed", "Gibs", "ColonGibs", "SmallBloodPool", "Blood", "BulletPuff", "TeleportFog", "ItemFog",
                 "BrainStem", "Candlestick", "DoomPlayer", "Puff", "Smoke", "Spark")


def obstacle_segments(objects, px: float, py: float, within: float = 700.0, radius: float = 20.0) -> list[tuple[float, float, float, float, bool, bool, bool]]:
    """Solid decorations (barrels, columns, trees, hanging bodies...) as small squares that block walking but not sight.
    ViZDoom's object list does not say what is solid, so everything that is not an enemy, an item, a projectile, the
    player or an obviously non-solid effect/corpse counts; routing around a corpse costs nothing."""
    segs = []
    for o in objects or []:
        n = o.name
        if n in ENEMIES or n in ITEMS or n in PROJECTILES or n.startswith(OBSTACLE_SKIP):
            continue
        if math.hypot(o.position_x - px, o.position_y - py) > within:
            continue
        x, y, r = o.position_x, o.position_y, radius
        c = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        for k in range(4):
            (x1, y1), (x2, y2) = c[k], c[(k + 1) % 4]
            segs.append((x1, y1, x2, y2, True, False, False))
    return segs


CELL = 96.0  # map units per exploration cell


def cell(x: float, y: float) -> tuple[int, int]:
    return int(x // CELL), int(y // CELL)


def _lines(geometry) -> list[tuple[float, float, float, float, bool, bool, bool]]:
    """Accept either the precomputed line list or ViZDoom's sectors."""
    if geometry and not isinstance(geometry[0], tuple):
        return blocking_lines(geometry)
    return geometry or []


def ray_distance(lines, px: float, py: float, angle_deg: float, max_dist: float = 600.0, dead_doors: set | None = None) -> tuple[float, bool]:
    """Distance from (px, py) along `angle_deg` to the first line that blocks walking (max_dist if none) and whether it is
    a door. Doors that did not open when used (`dead_doors`, by hit cell) count as walls."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    best, best_door = max_dist, False
    for x1, y1, x2, y2, walk, _sight, is_door in _lines(lines):
        if not walk:
            continue
        # ray p + t d, segment a + u (b - a)
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x1 - px) * ey - (y1 - py) * ex) / den
        u = ((x1 - px) * dy - (y1 - py) * dx) / den
        if t > 0 and 0 <= u <= 1 and t < best:
            best, best_door = t, is_door
    if best_door and dead_doors and cell(px + best * dx, py + best * dy) in dead_doors:
        best_door = False
    return best, best_door


def wall_distances(geometry, px: float, py: float, pangle: float, max_dist: float = 600.0, dead_doors: set | None = None) -> tuple[dict[str, str], dict[str, float], dict[str, bool]]:
    """Ray-cast at left / ahead / right: (descriptions, distances, hit-is-a-door)."""
    out, num, door = {}, {}, {}
    lines = _lines(geometry)
    for name, rel in (("left", 60.0), ("ahead", 0.0), ("right", -60.0)):
        best, is_door = ray_distance(lines, px, py, pangle + rel, max_dist, dead_doors)
        out[name] = "open" if best >= max_dist else (f"closed door in {best:.0f} units" if is_door else f"wall in {best:.0f} units")
        num[name] = best
        door[name] = is_door and best < max_dist
    return out, num, door


def line_of_sight(geometry, px: float, py: float, x: float, y: float) -> bool:
    """No blocking line (walls, closed doors) between the player and (x, y)."""
    dx, dy = x - px, y - py
    for x1, y1, x2, y2, _walk, sight, _door in _lines(geometry):
        if not sight:
            continue
        ex, ey = x2 - x1, y2 - y1
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x1 - px) * ey - (y1 - py) * ex) / den
        u = ((x1 - px) * dy - (y1 - py) * dx) / den
        if 0 < t < 1 and 0 <= u <= 1:
            return False
    return True


def compile_state(game: vzd.DoomGame, tracker: Tracker, mem: Memory, decision_tics: int) -> dict[str, Any]:
    s = game.get_state()
    v = dict(zip([str(g).split(".")[-1] for g in GAME_VARIABLES], [float(x) for x in s.game_variables]))
    px, py, pa = v["POSITION_X"], v["POSITION_Y"], v["ANGLE"]
    vx, vy = v["VELOCITY_X"], v["VELOCITY_Y"]
    speed = math.hypot(vx, vy) * 35
    vdir = rel_bearing(0, 0, pa, vx, vy) if speed > 1 else 0.0
    visible = {l.object_id for l in (s.labels or []) if l.width >= 3 and l.height >= 3}  # a 1-2 px sliver at the view's edge is not 'visible'
    lines = blocking_lines(s.sectors, v.get("POSITION_Z")) + obstacle_segments(s.objects, px, py)
    # Barriers the geometry does not show (block-everything lines, unknown solids): pushed forward for a whole window and
    # did not move -> remember a wall right in front of where we stood, unless a monster was the obstacle.
    # (The engine can report a velocity while the player is pinned against a wall, so only the position counts.)
    if mem.last_pos is not None and mem.last_forward and math.hypot(px - mem.last_pos[0], py - mem.last_pos[1]) < 4:
        lx, ly, la = mem.last_pos
        nx, ny = math.cos(math.radians(la)), math.sin(math.radians(la))
        blocked_by_monster = any(o.name in ENEMIES and math.hypot(o.position_x - lx, o.position_y - ly) < 72 and abs(rel_bearing(lx, ly, la, o.position_x, o.position_y)) < 50 for o in s.objects or [])
        mem.stuck += 1
        if not blocked_by_monster:
            # A learned wall blocks sight as well: monsters behind it are not fightable from here.
            cx, cy = lx + 28 * nx, ly + 28 * ny
            mem.bumps.append((cx - 48 * ny, cy + 48 * nx, cx + 48 * ny, cy - 48 * nx, True, True, False))
            mem.bumps = mem.bumps[-60:]
    else:
        mem.stuck = 0
    mem.last_pos = (px, py, pa)
    lines = lines + mem.bumps
    mem.visited.add(cell(px, py))
    enemies, items, projectiles = [], [], []
    for o in s.objects or []:
        d = math.hypot(o.position_x - px, o.position_y - py)
        b = rel_bearing(px, py, pa, o.position_x, o.position_y)
        if o.name in ENEMIES:
            closing = (o.velocity_x * (px - o.position_x) + o.velocity_y * (py - o.position_y)) > 0
            enemies.append({"id": f"{o.name.lower()} {tracker.letter('enemy', o.id)}", "kind": o.name, "distance": round(d), "bearing": f"{b:+.0f}° ({side(b)})",
                            "visible": o.id in visible, "line_of_sight": line_of_sight(lines, px, py, o.position_x, o.position_y) if d < 1500 else False,
                            "moving": "toward the player" if closing and math.hypot(o.velocity_x, o.velocity_y) > 0.5 else "not toward the player", "_obj": o})
        elif o.name in ITEMS:
            cat, desc = ITEMS[o.name]
            items.append({"id": f"{desc} {tracker.letter(cat, o.id)}", "category": cat, "distance": round(d), "bearing": f"{b:+.0f}° ({side(b)})", "visible": o.id in visible, "_obj": o})
        elif o.name in PROJECTILES:
            sp = math.hypot(o.velocity_x, o.velocity_y)
            if sp > 0.1:
                rx, ry = px - o.position_x, py - o.position_y
                t = max(0.0, (rx * o.velocity_x + ry * o.velocity_y) / (sp * sp))
                cx, cy = o.position_x + o.velocity_x * t - px, o.position_y + o.velocity_y * t - py
                miss = math.hypot(cx, cy)
                projectiles.append({"kind": o.name, "distance": round(d), "bearing": f"{b:+.0f}° ({side(b)})", "closest_approach": f"{miss:.0f} units in {t:.0f} tics" + (" (will hit if the player stays)" if miss < 40 else ""),
                                    "passes_on": "left" if rel_bearing(px, py, pa, px + cx, py + cy) > 0 else "right"})
    # What the model gets is capped; a crowded fight must not push an on-screen monster out in favour of sleeping ones
    # behind walls: rank by visible, then line of sight, then distance, and say how many were left out.
    enemies.sort(key=lambda e: (not e["visible"], not e["line_of_sight"], e["distance"]))
    items.sort(key=lambda i: (not i["visible"], i["distance"]))
    enemies_omitted, items_omitted = max(0, len(enemies) - MAX_ENEMIES), max(0, len(items) - MAX_ITEMS)
    # The renderer is the ground truth: an enemy the player has faced for three decisions without it appearing on screen
    # is behind something the 2D line-of-sight ray missed (a ledge, a lift, a window frame). Mark it occluded so it stops
    # being offered as a target and the model explores instead of walking into the wall in front of it.
    for e in enemies:
        b = abs(float(e["bearing"].split("°")[0]))
        if e["visible"]:
            mem.faced_unseen.pop(e["id"], None)
        elif e["line_of_sight"] and b < 35:
            mem.faced_unseen[e["id"]] = mem.faced_unseen.get(e["id"], 0) + 1
        elif b >= 35:
            mem.faced_unseen.pop(e["id"], None)
        if mem.faced_unseen.get(e["id"], 0) >= 3 and not e["visible"]:
            e["line_of_sight"] = False
            e["occluded"] = "faced it but it never came into view: something is in the way"
    took_damage = v["DAMAGE_TAKEN"] > mem.last_damage
    killed = v["KILLCOUNT"] > mem.last_kills
    hit = v["HITCOUNT"] > mem.last_hits
    walls, walls_num, walls_door = wall_distances(lines, px, py, pa, dead_doors=mem.dead_doors)
    # What the gun is pointing at. Doom auto-aims vertically, so a screen-pixel reticle test fails whenever the enemy
    # stands on a ledge or in a pit (MAP02 has both): use the aim cone instead — any visible enemy whose angular width
    # covers the facing direction (plus a small tolerance) is what a shot would hit.
    def in_aim_cone(e):
        half_width = math.degrees(math.atan2(24.0, max(e["distance"], 1))) + 2.5
        return e["visible"] and abs(float(e["bearing"].split("°")[0])) <= half_width
    aimed_at = [e["id"] for e in sorted(enemies, key=lambda e: e["distance"]) if in_aim_cone(e)]
    aimed = [l.object_name for l in (s.labels or []) if l.object_name != "DoomPlayer" and abs(l.x + l.width / 2 - s.screen_buffer.shape[1] / 2) < 6]
    tic = int(s.tic)
    dmg_now = v["DAMAGE_TAKEN"] - mem.last_damage
    if dmg_now > 0:
        mem.damage_log.append((tic, dmg_now))
    mem.damage_log = [(t_, d_) for t_, d_ in mem.damage_log if tic - t_ <= 35]
    damage_last_second = sum(d_ for _, d_ in mem.damage_log)
    pressed = speed < 25 and walls_num.get("ahead", 999) < 48
    weapon_name = WEAPON_NAMES.get(int(v["SELECTED_WEAPON"]), "")
    reach = WEAPON_RANGE.get(weapon_name, 900)
    nearest = enemies[0] if enemies else None
    attackers = [e for e in enemies if e["distance"] < 450 and e["moving"] == "toward the player"]
    threat = {
        "under_attack": bool(took_damage or attackers),
        "nearest_enemy": f"{nearest['id']} at {nearest['distance']} units, {nearest['bearing']}, {'visible' if nearest['visible'] else ('line of sight but not on screen' if nearest['line_of_sight'] else 'not visible, no line of sight')}" if nearest else "none",
        "enemies_in_view_or_line_of_sight": sum(1 for e in enemies if e["visible"] or e["line_of_sight"]),
        "enemies_within_500_units": sum(1 for e in enemies if e["distance"] < 500),
        "enemies_behind_or_beside": sum(1 for e in enemies if abs(float(e["bearing"].split("°")[0])) > 60),
        "advice": "An enemy that is visible or has line_of_sight within ~500 units is the immediate threat: face it and fight. Enemies that are not visible and have no line_of_sight are asleep behind walls: they cannot be fought or reached from here, so ignore them and explore ahead; they wake up when they see or hear the player. A closed door opens when the player walks into it.",
        "damage_taken_last_second": int(damage_last_second),
    }
    for e in enemies:
        e["in_range_of_current_weapon"] = e["distance"] <= reach
    mem.last_damage, mem.last_kills, mem.last_hits, mem.last_health = v["DAMAGE_TAKEN"], v["KILLCOUNT"], v["HITCOUNT"], v["HEALTH"]
    weapon = int(v["SELECTED_WEAPON"])
    owned = [WEAPON_NAMES[w] for w in (3, 4, 5, 6) if v.get(f"WEAPON{w}", 0) > 0]
    state = {
        "rules": {
            "one_game_tick": "1/35 second",
            "control_rate": f"decisions update every {decision_tics} ticks (~{decision_tics / 35:.2f} s); controls persist between decisions",
            "controls": "aiming (turning toward a bearing), movement and the trigger operate simultaneously",
            "projectile_prediction": "closest approach assumes the player stays at the current position; moving changes it",
        },
        "player": {
            "health": f"{v['HEALTH']:.0f}/100", "armor": f"{v['ARMOR']:.0f}/200", "weapon": WEAPON_NAMES.get(weapon, str(weapon)), "ammo_in_weapon": int(v["SELECTED_WEAPON_AMMO"]),
            "owned_weapons": ["fist", "pistol"] + owned, "ammo": {"bullets": int(v["AMMO2"]), "shells": int(v["AMMO3"]), "rockets": int(v["AMMO4"]), "cells": int(v["AMMO5"])},
            "velocity": {"speed": f"{speed:.0f} map units/second", "direction": f"{vdir:+.0f}° relative to facing"} if speed > 1 else "standing still",
            "just_took_damage": took_damage, "just_scored_a_hit": hit, "just_killed": killed,
            "weapon_effective_range": f"{reach} units",
            "aimed_at": aimed_at or ("nothing" if not aimed else f"{aimed[0].lower()} (not an enemy)"),
            "pressed_against_wall": pressed,
        },
        "threat": threat,
        "walls": walls,
        "enemies": [{k: e[k] for k in e if k != "_obj"} for e in enemies[:MAX_ENEMIES]] or "none known",
        **({"enemies_not_listed": f"{enemies_omitted} more (farther, not in view)"} if enemies_omitted else {}),
        "projectiles_in_flight": projectiles[:4] or "none",
        "items": [{k: i[k] for k in i if k != "_obj"} for i in items[:MAX_ITEMS]] or "none known",
        **({"items_not_listed": f"{items_omitted} more (farther)"} if items_omitted else {}),
        "current_priority": {"goal": mem.goal, "subject": mem.subject},
    }
    if mem.orders:
        state["standing_orders"] = mem.orders
    return state, {"enemies": enemies, "items": items, "player": (px, py, pa), "vars": v, "walls": walls_num, "door_ahead": walls_num["ahead"] if walls_door.get("ahead") else None, "lines": lines, "visited": mem.visited, "door_tries": mem.door_tries, "dead_doors": mem.dead_doors, "mem": mem, "stuck": mem.stuck}
