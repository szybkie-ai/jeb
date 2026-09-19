#!/usr/bin/env zsh
# Raw teacher play (no orders, text-only, one presentation) over several maps and seeds, six episodes in parallel per
# wave (the teacher batches; MEASURED 2026-09-19: 3 parallel = 6 s/decision p50). Same game settings as eval_long.sh
# (skill 3, director 60, wake, 1,200 decisions). Output: out/tch-<map>-s<seed>{,.log}; survivors picked by
# openjev-train/tools/make_doom_t.sh.   Usage: ./collect_teacher2.sh <url> <model> "<maps>" "<seeds>" [skill]
set -e
url=$1; model=$2; maps=(${=3}); seeds=(${=4}); skill=${5:-3}
until curl -s -m 60 $url/v1/systemone -H "Content-Type: application/json" -d '{"state":"probe","model":"'$model'","questions":{"q":{"type":"noul","instructions":"Is this a probe?"}}}' | grep -q '"noul"'; do sleep 20; done
run() { d=out/tch-$1-s$2; rm -rf $d; .venv/bin/python run.py --url $url --model $model --map $1 --skill $skill --decisions 1200 --director 60 --wake --seed $2 --collect --permutations 1 --log $d/ticks.jsonl > $d.log 2>&1 || true; echo "tch $1 seed $2: $(grep '"died"' $d.log | tail -1 | cut -c1-200)"; }
todo=()
for s in $seeds; do for m in $maps; do todo+=("$m $s"); done; done
i=0
while [ $i -lt ${#todo[@]} ]; do
  for j in "${todo[@]:$i:6}"; do run ${=j} & done
  wait
  i=$((i + 6))
done
echo COLLECT2 DONE
