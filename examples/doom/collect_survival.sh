#!/bin/zsh
# Teacher play under standing orders, frames on: round-2 Doom data (set doom_s). Two order styles so the student learns both
# and the eval can steer: balanced (kill, but survive) on MAP02 and MAP01, survival-first on MAP07.
#   ./collect_survival.sh <url> <model>
set -e
url=$1; model=$2
balanced="Kill the monsters you meet, but stay alive: keep health above 50, pick up health and armor whenever they are near, back off and dodge when badly hurt or outnumbered, and never chase a kill at low health."
survival="Stay alive first: keep health above 50, pick up health and armor whenever they are near, back off and dodge when under fire, fight only what you can beat from a safe distance, and never chase a kill at low health."
# wait for the engine behind the server, not just the server: a real one-question request must succeed
until curl -s -m 60 $url/v1/systemone -H "Content-Type: application/json" -d '{"state":"probe","model":"'$model'","questions":{"q":{"type":"noul","instructions":"Is this a probe?"}}}' | grep -q '"noul"'; do sleep 20; done
for spec in "MAP02 balanced" "MAP01 balanced" "MAP07 survival"; do
  set -A p ${=spec}; m=$p[1]; style=$p[2]; orders=${(P)style}
  d=out/surv-$m-$style; rm -rf $d
  .venv/bin/python run.py --url $url --model $model --map $m --skill 3 --decisions 800 --director 60 --wake --seed 101 --collect --image --orders "$orders" --log $d/ticks.jsonl > $d.log 2>&1 || true
  echo "surv $m $style: $(grep kills $d.log | tail -1 | cut -c1-240)"
done
echo COLLECT DONE
