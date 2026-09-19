"""The per-tick fan-out: every judgment the controller might need, asked together in one request.

Second-hop questions (movement, dodge, trigger) are conditioned on the *previous* tick's goal and subject, so
one request per tick is enough; controls persist between decisions, so a one-tick-old priority is fine.
"""

from __future__ import annotations

from typing import Any

GOALS = {
    "kill_enemies": "Attack: an enemy that is visible or has line of sight is the immediate problem",
    "restore_health": "Health is low: get a health item first",
    "add_armor": "Pick up armor while it is safe",
    "stock_ammo": "Ammo is low: get ammunition",
    "upgrade_weapon": "A better weapon is nearby and worth grabbing",
    "scout": "Nothing in view: move on through the level to find enemies or items",
}
MOVES = {
    "close_in": "Move toward the current subject (only useful with line_of_sight)",
    "back_off": "Move away from the current subject, keeping it in view",
    "hold_ground": "Stay put and shoot: only when a visible enemy is coming to you or is lined up in the aim cone",
    "strafe_left": "Sidestep left while facing the subject",
    "strafe_right": "Sidestep right while facing the subject",
    "walk_to_item": "Walk to the chosen item",
    "explore_ahead": "Walk ahead through open space to find enemies or items (the default when nothing is visible)",
    "turn_around": "Turn around to look behind",
}
DODGES = {
    "carry_on": "No incoming threat needs a dodge right now",
    "dodge_left": "Sidestep left to avoid a projectile or charge",
    "dodge_right": "Sidestep right to avoid a projectile or charge",
    "dodge_back": "Step back to avoid a melee attacker or a blast",
}


def build_questions(state: dict[str, Any], raw: dict[str, Any], mem) -> dict[str, Any]:
    prio = f"The player's current top priority is {mem.goal}" + (f", focusing on {mem.subject}" if mem.subject else "") + "."
    q: dict[str, Any] = {
        "goal": {"type": "choice", "instructions": "Considering `threat`, `player`, `enemies` and `items`, what is the player's highest-priority goal right now? Enemies nearby (see `threat`) come first unless health is critical.", "criteria": GOALS},
    }
    # The dodge question is only asked when something can hit us: a projectile in flight, a melee enemy closing in,
    # or damage just taken. The untrained 4B otherwise dodges on most ticks for no reason (the teacher and Jev never do);
    # skipping the question keeps the raw model honest and saves a prompt.
    pr = state.get("projectiles_in_flight")
    melee = any(e["visible"] and e["distance"] < 120 and e.get("moving") == "toward the player" for e in raw["enemies"])
    if (isinstance(pr, list) and pr) or melee or state["threat"].get("under_attack"):
        q["dodge"] = {"type": "choice", "instructions": f"{prio} Considering `projectiles_in_flight` and enemies that are close and moving toward the player, what does this exact moment call for?", "criteria": DODGES}
    q.update({
        "trigger": {"type": "noul", "instructions": "Should the player hold the trigger down right now? Fire when `player.aimed_at` lists an enemy that is `in_range_of_current_weapon` (the game auto-aims vertically, so height differences do not matter, and a rough aim is enough). Hold fire when nothing is in the aim cone or ammo is 0.",
                    "criteria": {"true": "Fire", "false": "Hold fire"}},
    })
    enemies = raw["enemies"][:8]
    # Only enemies the player can see or has line of sight to are fightable now; the rest are asleep behind walls and
    # only clutter the choice (asking about them made the model turn toward walls instead of exploring).
    engageable = [e for e in enemies if e["visible"] or e["line_of_sight"]]
    behind = [e for e in engageable if abs(float(e["bearing"].split("°")[0])) > 110 and e["distance"] < 700]
    if engageable:
        q["target_enemy"] = {"type": "choice", "instructions": "Which enemy should the player focus on? Prefer the nearest visible one and anything moving toward the player.",
                             "criteria": {e["id"]: f"{e['kind']}, {e['distance']} units, {e['bearing']}, {'visible' if e['visible'] else 'line of sight but not on screen'}" for e in engageable} | {"none": "No enemy is worth focusing on right now"}}
        moves = {k: v for k, v in MOVES.items() if k != "turn_around" or behind}
        q["movement"] = {"type": "choice", "instructions": f"{prio} Given `walls`, `player.pressed_against_wall`, the enemies' visibility/line_of_sight/in_range flags and `threat.damage_taken_last_second`, how should the player move right now?", "criteria": moves}
    else:
        # Nothing to fight: standing still is never right in Doom unless something is hurting us, so the menu is
        # explore / go for an item (/ hold only while under attack). Turning around is offered when an enemy is behind.
        keys = ("explore_ahead", "walk_to_item") + (("hold_ground",) if state["threat"]["under_attack"] else ()) + (("turn_around",) if behind else ())
        q["movement"] = {"type": "choice", "instructions": f"{prio} No enemy is in view or in line of sight. Given `walls` and `items`, how should the player move right now? Exploring ahead is the default.",
                         "criteria": {k: MOVES[k] for k in keys}}
    items = raw["items"][:8]
    if items:
        q["target_item"] = {"type": "choice", "instructions": "If the player goes for an item, which one is worth it right now (need, distance, safety)?",
                            "criteria": {i["id"]: f"{i['category']}, {i['distance']} units, {i['bearing']}" for i in items} | {"none": "No item is worth it right now"}}
    return q


def apply_answers(answers: dict[str, Any], raw: dict[str, Any], mem) -> dict[str, Any]:
    """Read the fan-out; update the memory's priority; return the decision the controller executes."""
    goal = answers["goal"]["choice"]
    subject_obj = None
    subject = None
    enemy_ids = {e["id"]: e for e in raw["enemies"]}
    item_ids = {i["id"]: i for i in raw["items"]}
    te = answers.get("target_enemy", {}).get("choice")
    ti = answers.get("target_item", {}).get("choice")
    if goal == "kill_enemies" and te in enemy_ids:
        subject, subject_obj = te, enemy_ids[te]
    elif goal in ("restore_health", "add_armor", "stock_ammo", "upgrade_weapon") and ti in item_ids:
        subject, subject_obj = ti, item_ids[ti]
    elif te in enemy_ids:
        subject, subject_obj = te, enemy_ids[te]
    mem.goal, mem.subject = goal, subject
    move = answers["movement"]["choice"]
    dodge = answers["dodge"]["choice"] if "dodge" in answers else "carry_on"
    fire = answers["trigger"]["noul"] > 0.5
    return {"goal": goal, "subject": subject, "subject_obj": subject_obj, "move": move, "dodge": dodge, "fire": fire,
            "walk_item": item_ids.get(ti) if move == "walk_to_item" or goal in ("restore_health", "add_armor", "stock_ammo", "upgrade_weapon") else None,
            "confidence": {k: v.get("confidence", abs(2 * v["noul"] - 1) if "noul" in v else None) for k, v in answers.items()}}
