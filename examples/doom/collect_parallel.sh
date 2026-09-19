#!/bin/zsh
# Round-2 Doom data, three teacher episodes in parallel (the engine batches them): two text-only balanced episodes and one
# frame-attached survival episode at one presentation (the image path costs ~8 s per decision at two presentations).
set -e
url=$1; model=$2
balanced="Kill the monsters you meet, but stay alive: keep health above 50, pick up health and armor whenever they are near, back off and dodge when badly hurt or outnumbered, and never chase a kill at low health."
survival="Stay alive first: keep health above 50, pick up health and armor whenever they are near, back off and dodge when under fire, fight only what you can beat from a safe distance, and never chase a kill at low health."
run() { d=out/surv-$1-$2; rm -rf $d; .venv/bin/python run.py --url $url --model $model --map $1 --skill 3 --decisions $3 --director 60 --wake --seed 101 --collect --orders "$4" --log $d/ticks.jsonl ${=5} > $d.log 2>&1 || true; echo "surv $1 $2: $(grep kills $d.log | tail -1 | cut -c1-240)"; }
run MAP02 balanced 800 "$balanced" "" &
run MAP01 balanced 800 "$balanced" "" &
run MAP07 survival 300 "$survival" "--image --permutations 1" &
wait
echo COLLECT DONE
