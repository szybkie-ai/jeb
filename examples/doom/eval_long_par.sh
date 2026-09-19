#!/bin/zsh
# Long-horizon Doom evaluation, all six episodes in parallel (the engines batch): 1,200 decisions on MAP01 and MAP02,
# seeds 91-93, same settings as eval_long.sh.   ./eval_long_par.sh <tag> <url> <model> [extra run.py flags...]
set -e
tag=$1; url=$2; model=$3; shift 3
for m in MAP02 MAP01; do for s in 91 92 93; do
  d=out/long-$tag-$m-s$s; rm -rf $d
  ( .venv/bin/python run.py --url $url --model $model --map $m --skill 3 --decisions 1200 --director 60 --wake --seed $s --log $d/ticks.jsonl "$@" > $d.log 2>&1 || true
    echo "$tag $m seed $s: $(grep '"died"' $d.log | tail -1 | cut -c1-220)" ) &
done; done
wait
echo "EVAL $tag DONE"
