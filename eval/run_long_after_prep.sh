#!/bin/sh
# Waits for long-video prep AND the short-300 run to finish, then runs the
# long-video eval: videoctx + naive on ~30 questions over 30-60 min videos.
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

until grep -q "^DONE:" eval/results/prep_long.log 2>/dev/null; do sleep 60; done
echo "long prep finished; waiting for short300"
until grep -q "SHORT300 COMPLETE" eval/results/short300.log 2>/dev/null; do sleep 60; done
echo "starting long eval"

$PY eval/run.py eval/data/videomme_long.jsonl --methods videoctx --out eval/results/long_videoctx.json --skip-existing
$PY eval/run.py eval/data/videomme_long.jsonl --methods naive    --out eval/results/long_naive.json    --skip-existing
echo "LONG EVAL COMPLETE"
