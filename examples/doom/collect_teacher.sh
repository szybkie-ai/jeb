#!/bin/zsh
# Raw teacher play, no standing orders, text-only, one presentation, 1200 decisions: round-3 Doom data (only episodes the
# teacher survives are used). Three episodes in parallel per seed; the engine batches them.
#   ./collect_teacher.sh <url> <model> [seeds...]
set -e
url=$1; model=$2; shift 2
seeds=(${@:-201 202})
until curl -s -m 60 $url/v1/systemone -H "Content-Type: application/json" -d '{"state":"probe","model":"'$model'","questions":{"q":{"type":"noul","instructions":"Is this a probe?"}}}' | grep -q '"noul"'; do sleep 20; done
run() { d=out/tch-$1-s$2; rm -rf $d; .venv/bin/python run.py --url $url --model $model --map $1 --skill 3 --decisions 1200 --director 60 --wake --seed $2 --collect --permutations 1 --log $d/ticks.jsonl > $d.log 2>&1 || true; echo "tch $1 seed $2: $(grep kills $d.log | tail -1 | cut -c1-240)"; }
for s in $seeds; do
  run MAP02 $s & run MAP01 $s & run MAP07 $s &
  wait
done
echo COLLECT DONE
