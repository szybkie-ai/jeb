# JEB plays Doom

A System One agent for Doom: every few game ticks the engine state is compiled into JSON, one request with
six typed questions goes to an JEB server, and code turns the answers into controls. The model never
generates text and never sees pixels (yet); code owns geometry, memory and policy.

```bash
uv sync                                                        # vizdoom (bundles freedoom), httpx, numpy, pillow
uv run python run.py --policy scripted --decisions 300         # baseline: face the nearest enemy and shoot
uv run python run.py --url http://127.0.0.1:8020 --decisions 300 --record out/ep1 \
    --orders "Fight aggressively: when an enemy is visible and roughly ahead, keep firing at it and close in; never back off while health is above 50."
```

`--record` writes `episode.gif` (game view + a judgments panel), `judgments.jsonl` and, with `--log`,
the full per-tick state/answers. `--orders` is free text appended to the state — the same "standing
orders" lever as TypeSafe's demo.

Real levels instead of the arena: `--map MAP01` (a comma list cycles maps per episode) plays the bundled
freedoom2 levels at `--skill 1..5` (`--wad doom2.wad` for the original). Doom's monsters are dormant until
they notice you, so a bare level is static; `--wake` fires one shot at the start to alert the level and
`--director N` is a fallback against dead air, not a spawn fountain: when no enemy has been in view for N decisions
it summons one monster ahead through the engine console (only into open space: `summon` ignores walls and doors). The
level's own monsters are the fight; the first shot and being seen wake them up. Exploration is code-owned: when the
model chooses `explore_ahead`, the controller picks the most open heading weighted toward map cells it has not visited
and toward carrying straight on, treats a shut door as an opening (walks into it and presses USE), and gives up on a
door that stays shut. Map lines are classified from the sectors buffer (ViZDoom flags neither doors nor ledges): a
two-sided line blocks walking when the gap between floors and ceilings is under the player's height or the floors differ
by more than a step, and blocks sight only when the gap is nothing.

Usable freedoom2 maps with this ViZDoom build: MAP01, MAP02, MAP06, MAP07, MAP10, MAP11. MAP03/04/08/09/12 crash the
engine at load as soon as the sectors buffer is enabled (needed for the ray-cast); MAP05 starts behind a door that USE
does not open. Run one map per process — switching maps inside a process crashes too. `--collect` stores the model's raw
per-presentation logprobs in the tick log, which makes a teacher's play directly usable as labelled
training data.

Every decision's frame is saved next to the tick log (`frames/<episode>-<n>.png`, referenced by the row's `frame`
field), so a run doubles as image-labelled data. `--image` attaches that frame to every request (the server's `images`
extension, for a vision-language model behind it): the model then sees the player's view *and* the compiled state.

## What the model sees

`jeb_doom/state.py` compiles ViZDoom's object list, labels (visibility), sectors and variables into:

- `rules` (tick length, decision cadence, that controls persist), `player` (health, armor, weapon and its
  effective range, ammo, velocity, `aimed_at` — the enemy under the crosshair from the labels buffer —,
  `pressed_against_wall`, just-took-damage / hit / killed flags), `walls` (ray-cast left/ahead/right),
- `threat` (under attack?, nearest enemy, enemies within 500 units, enemies behind, damage taken in the
  last second — computed in code so the judgment is easy); every enemy carries `in_range_of_current_weapon`,
- `enemies` and `items` with stable letter ids ("shotgunguy A", "+25 health K"), distance, bearing and
  visibility; `projectiles_in_flight` with a code-computed closest approach,
- `current_priority` (last tick's goal and subject) and `standing_orders`.

`player.aimed_at` is the aim cone, not a pixel test: every visible enemy whose angular width covers the facing
direction (Doom auto-aims vertically, so ledges and pits do not matter). The menus are gated by the state: only
enemies in view or in line of sight are offered as targets; with nothing engageable the movement menu is explore /
walk to an item (hold only while under attack), so the model never camps at a wall.

## What it asks (one request per decision)

| question | type | options |
|---|---|---|
| goal | choice | kill_enemies, restore_health, add_armor, stock_ammo, upgrade_weapon, scout |
| target_enemy | choice | the known enemies + none |
| target_item | choice | the nearest items + none |
| movement | choice | close_in, back_off, hold_ground, strafe_left/right, walk_to_item (+ explore/turn around when alone) |
| dodge | choice | carry_on, dodge_left, dodge_right, dodge_back |
| trigger | noul | hold the trigger down? |

Second-hop questions are conditioned on the *previous* tick's priority, so it stays one request per tick.
`control.py` turns the subject's bearing into a proportional turn, the movement/dodge into buttons, adds
wall avoidance, and picks the weapon in code.

## First numbers (raw Qwen3.5-4B FP8, no fine-tuning, GPU shared with a 176B teacher, 150 decisions)

| policy (deathmatch, 3 episodes, seed 1, ≤300 decisions) | kills / game-minute | survival (game s) | fire when enemy visible ahead |
|---|---|---|---|
| scripted baseline | 4.2 mean (3 / 0 / 0 kills) | 14.3 / 3.9 / 9.0, all died | always |
| JEB, no orders (single run) | 0.0 | died | 8% (backs off from everything) |
| JEB + aggressive standing orders | **13.2 mean** (8 / 6 / 1 kills) | 27.5 / 21.3 / 11.6, all died | 100%, closes in |

The scenario spawns monsters continuously, so every run ends in death; the agent lives 2–3× longer and
kills 3× more than the scripted baseline, with the *untrained* student.

Wall-clock was ~0.8 s per decision in these runs (2k-token state, contended GPU): the game runs in
synchronous mode, so it plays in slow motion and records at the true tick rate. Real-time needs ≤150 ms
per decision — a free GPU, a leaner state and the fine-tuned student.
