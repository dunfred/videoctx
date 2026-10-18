#!/bin/sh
# Free-tier completion sweep:
#   1. Medium subset (10 videos, 4-15 min): download, index, caption
#   2. Medium: videoctx + naive (Claude)
#   3. Flash-Lite native column: short pilot 30, medium 30, long 30
#      (long attempts may fail on the lite tier's 250k TPM cap; failures are
#       recorded as data, each costs one of 500 daily requests)
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

$PY eval/prepare_videomme.py --duration medium --videos 10 --out eval/data
$PY eval/prep_indexes.py eval/data/videomme_medium.jsonl

$PY eval/run.py eval/data/videomme_medium.jsonl --methods videoctx --out eval/results/medium_videoctx.json --skip-existing
$PY eval/run.py eval/data/videomme_medium.jsonl --methods naive    --out eval/results/medium_naive.json    --skip-existing

$PY eval/run.py eval/data/videomme_short_pilot30.jsonl --methods gemini-lite --out eval/results/short30_gemini_lite.json --skip-existing
$PY eval/run.py eval/data/videomme_medium.jsonl        --methods gemini-lite --out eval/results/medium_gemini_lite.json  --skip-existing
$PY eval/run.py eval/data/videomme_long.jsonl          --methods gemini-lite --out eval/results/long_gemini_lite.json    --skip-existing

echo "MEDIUM AND LITE COMPLETE"
