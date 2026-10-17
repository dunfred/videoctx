#!/bin/sh
# Full short-subset run: 100 videos / ~300 questions, dev/test split by video.
# videoctx on everything; naive baseline on the test half only.
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

cp -n eval/data/videomme_short.jsonl eval/data/videomme_short_pilot30.jsonl || true
$PY eval/prepare_videomme.py --duration short --videos 100 --out eval/data
$PY eval/split_dev_test.py eval/data/videomme_short.jsonl
$PY eval/prep_indexes.py eval/data/videomme_short.jsonl
$PY eval/run.py eval/data/videomme_short_dev.jsonl  --methods videoctx --out eval/results/short300_dev_videoctx.json  --skip-existing
$PY eval/run.py eval/data/videomme_short_test.jsonl --methods videoctx --out eval/results/short300_test_videoctx.json --skip-existing
$PY eval/run.py eval/data/videomme_short_test.jsonl --methods naive    --out eval/results/short300_test_naive.json    --skip-existing
echo "SHORT300 COMPLETE"
