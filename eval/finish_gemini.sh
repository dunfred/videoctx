#!/bin/sh
# Waits for the Gemini free-tier daily quota to reset, then completes the
# remaining short-subset questions (resume via --skip-existing).
set -e
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
PY=.venv/bin/python

probe() {
  $PY - <<'EOF'
import os, sys
from google import genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("AISTUDIO_API_KEY"))
try:
    client.models.generate_content(model="gemini-3-flash-preview", contents="ok")
    sys.exit(0)
except Exception as e:
    sys.exit(1)
EOF
}

until probe; do
  echo "quota still exhausted; sleeping 1h"
  sleep 3600
done
echo "quota available; completing gemini run"
$PY eval/run.py eval/data/videomme_short_pilot30.jsonl --methods gemini \
  --out eval/results/pilot_short_gemini_partial.json --skip-existing
echo "GEMINI SHORT COMPLETE"
