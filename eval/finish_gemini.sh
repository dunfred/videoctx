#!/bin/sh
# Gemini free-tier quota frees up on a ROLLING 24h window (not a daily reset),
# so drip-complete: sweep the remaining errored questions every 45 min; each
# sweep fills however many request slots have freed. Converges within ~12h.
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python
OUT=eval/results/pilot_short_gemini_partial.json

done_count() {
  $PY -c "import json; print(sum('error' not in r for r in json.load(open('$OUT'))))"
}

while :; do
  n_ok=$(done_count)
  if [ "$n_ok" -ge 30 ]; then break; fi
  echo "sweep: $n_ok/30 complete"
  $PY eval/run.py eval/data/videomme_short_pilot30.jsonl --methods gemini \
    --out "$OUT" --skip-existing || true
  n_ok=$(done_count)
  [ "$n_ok" -ge 30 ] && break
  sleep 2700
done
echo "GEMINI SHORT FINAL"
