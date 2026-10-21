#!/bin/bash
# Dense-caption A/B: enrich dev2 + long-dev indexes with dense keyframe
# captions, then run both dev sets with dense lines included. Controls are
# the committed v0.4 receipts (same code, same indexes pre-enrichment,
# equivalent to VIDEOCTX_NO_DENSE=1).
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

for q in eval/data/videomme_medium_dev2.jsonl eval/data/videomme_long_dev.jsonl; do
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

echo "DENSE ENRICHMENT DONE"
retry env VIDEOCTX_NO_EMBED=1 $PY eval/run.py eval/data/videomme_medium_dev2.jsonl \
  --methods videoctx --out eval/results/meddev2_dense.json
retry env VIDEOCTX_NO_EMBED=1 $PY eval/run.py eval/data/videomme_long_dev.jsonl \
  --methods videoctx --out eval/results/longdev_dense.json
echo "DENSE AB DONE"
