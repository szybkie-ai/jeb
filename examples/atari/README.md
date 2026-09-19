# JEB plays Atari (forgetting check)

A second game the model never trained on. `ale-py` bundles the ROMs; the state is decoded from RAM (AtariARI addresses),
`--image` adds the frame upscaled 3x, and one request per decision carries one or two typed questions (`move`, and
`fire` in Breakout when the ball is out). Code maps the answer to the joystick and holds it for `--hold` env steps.

```bash
uv sync
uv run python atari_eval.py --game pong --policy scripted --episodes 3            # baselines: scripted, random
uv run python atari_eval.py --game mspacman --policy jeb --url http://127.0.0.1:8020 --model jeb-4b --image --episodes 3 --out out/mspacman-raw.json
```

Games: `pong` (score = points for minus against), `breakout` (bricks; use `--hold 1`, the ball outruns a two-step hold), `mspacman` (dots, ghosts). The forgetting check
compares the raw student with the trained one under the same harness: a collapse relative to raw is forgetting; the
scripted and random policies bound the range.
