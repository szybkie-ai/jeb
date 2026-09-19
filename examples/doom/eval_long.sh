#!/bin/zsh
# Long-horizon Doom evaluation: 1200 decisions (~2.3 game-minutes) on MAP01 and MAP02, seeds 91-93, one player at a time.
#   ./eval_long.sh <tag> <url> <model> [extra run.py flags...]      e.g. ./eval_long.sh trained http://127.0.0.1:8024 jeb-latest --image
set -e
tag=$1; url=$2; model=$3; shift 3
for m in MAP02 MAP01; do for s in 91 92 93; do
  d=out/long-$tag-$m-s$s; rm -rf $d
  .venv/bin/python run.py --url $url --model $model --map $m --skill 3 --decisions 1200 --director 60 --wake --seed $s --log $d/ticks.jsonl "$@" > $d.log 2>&1 || true
  echo "$tag $m seed $s: $(grep kills $d.log | tail -1 | cut -c1-200)"
done; done
