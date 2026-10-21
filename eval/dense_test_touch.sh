#!/bin/bash
# v0.5 one-touch: dense-enrich medium and long TEST indexes (criterion: any
# shot >= 45s; an index property, not a performance pick), then single runs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
set -a; source .env; set +a

retry() {
  local n=0
  until "$@"; do
    n=$((n+1)); [ $n -ge 5 ] && { echo "RETRY EXHAUSTED: $*"; exit 1; }
    echo "attempt $n failed, sleeping 60s: $*"; sleep 60
  done
}

for q in eval/data/videomme_medium.jsonl eval/data/videomme_long.jsonl; do
  for idx in $($PY -c "
import json
from pathlib import Path
vids = sorted({json.loads(l)['video'] for l in open('$q') if l.strip()})
for v in vids:
    p = Path(v)
    print(p.parent / (p.stem + '.vctx'))
"); do
    retry .venv/bin/videoctx caption --dense "$idx"
  done
done
echo "TEST ENRICHMENT DONE"

retry env VIDEOCTX_NO_EMBED=1 $PY eval/run.py eval/data/videomme_medium.jsonl \
  --methods videoctx --skip-existing --out eval/results/FINAL_v05_medium30.json
retry env VIDEOCTX_NO_EMBED=1 $PY eval/run.py eval/data/videomme_long.jsonl \
  --methods videoctx --skip-existing --out eval/results/FINAL_v05_long30.json
echo "V05 TOUCH DONE"
