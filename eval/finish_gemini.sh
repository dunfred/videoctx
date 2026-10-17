#!/bin/sh
# Gemini free-tier quota frees up on a ROLLING 24h window (not a daily reset),
# so drip-complete: sweep the remaining errored questions every 45 min; each
# sweep fills however many request slots have freed. Converges within ~12h.
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python
OUT=eval/results/pilot_short_gemini_partial.json

while :; do
  n_err=$($PY -c "import json; print(sum('error' in r for r in json.load(open('$OUT'))))")
  if [ "$n_err" -eq 0 ]; then break; fi
  echo "sweep: $n_err questions remaining"
  $PY eval/run.py eval/data/videomme_short_pilot30.jsonl --methods gemini \
    --out "$OUT" --skip-existing || true
  n_err=$($PY -c "import json; print(sum('error' in r for r in json.load(open('$OUT'))))")
  [ "$n_err" -eq 0 ] && break
  sleep 2700
done
echo "GEMINI SHORT FINAL"
