import random, time
import vizdoom as vzd
g = vzd.DoomGame()
g.load_config(vzd.scenarios_path + "/deathmatch.cfg")
g.set_window_visible(False)
g.set_mode(vzd.Mode.PLAYER)
g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
g.set_objects_info_enabled(True)
g.set_labels_buffer_enabled(True)
g.set_sectors_info_enabled(True)
g.set_available_game_variables([vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.SELECTED_WEAPON, vzd.GameVariable.SELECTED_WEAPON_AMMO,
                                vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.POSITION_Z, vzd.GameVariable.ANGLE,
                                vzd.GameVariable.VELOCITY_X, vzd.GameVariable.VELOCITY_Y, vzd.GameVariable.KILLCOUNT, vzd.GameVariable.DAMAGE_TAKEN, vzd.GameVariable.HITCOUNT])
print("buttons:", [str(b) for b in g.get_available_buttons()])
g.init()
g.new_episode()
t0 = time.perf_counter(); n = 0
names = set(); labels = set()
for i in range(120):
    s = g.get_state()
    if s is None: break
    if i % 40 == 0:
        print(f"tic {s.tic}: vars={list(s.game_variables)}")
        objs = s.objects or []
        print(f"  objects: {len(objs)}; e.g.", [(o.name, round(o.position_x), round(o.position_y), round(o.angle)) for o in objs[:6]])
        print(f"  labels (visible): {[(l.object_name, l.x, l.y, l.width, l.height) for l in (s.labels or [])[:5]]}")
        print(f"  screen: {s.screen_buffer.shape}")
        o = objs[0]; print("  object fields:", [a for a in dir(o) if not a.startswith('_')])
        sec = s.sectors or []; print(f"  sectors: {len(sec)}")
    names |= {o.name for o in (s.objects or [])}; labels |= {l.object_name for l in (s.labels or [])}
    act = [0] * len(g.get_available_buttons()); act[random.randrange(len(act))] = 1
    g.make_action(act, 4); n += 1
print(f"{n} decisions in {time.perf_counter()-t0:.2f}s ({n/(time.perf_counter()-t0):.0f} decisions/s headless)")
print("object names seen:", sorted(names)[:40])
print("label names seen:", sorted(labels)[:20])
g.close()
