#!/bin/bash
# Expand the medium dev split with fresh pool videos (never touches frozen
# splits: videomme_medium.jsonl and videomme_medium_dev.jsonl are excluded
# and left unmodified). Full enrichment chain, then a baseline vctx run so
# the new questions have receipts from day one.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
set -a; source .env; set +a

$PY eval/prepare_videomme.py --duration medium --videos 15 \
  --exclude eval/data/videomme_medium.jsonl \
  --exclude eval/data/videomme_medium_dev.jsonl \
  --qfile-name videomme_medium_dev2.jsonl

# Every step below is idempotent (indexes, captions, OCR, embeddings skip
# existing work), so transient network failures are retried, not fatal.
retry() {
  local n=0
  until "$@"; do
    n=$((n+1))
    [ $n -ge 5 ] && { echo "RETRY EXHAUSTED: $*"; exit 1; }
    echo "attempt $n failed, sleeping 60s: $*"
    sleep 60
  done
}

retry $PY eval/prep_indexes.py eval/data/videomme_medium_dev2.jsonl

for idx in $($PY -c "
import json
from pathlib import Path
vids = sorted({json.loads(l)['video'] for l in open('eval/data/videomme_medium_dev2.jsonl') if l.strip()})
for v in vids:
    p = Path(v)
    print(p.parent / (p.stem + '.vctx'))
"); do
  retry .venv/bin/videoctx ocr "$idx"
  retry .venv/bin/videoctx embed "$idx"
done

echo "ENRICHMENT DONE, running baseline (one-touch, embeddings off)"
retry env VIDEOCTX_NO_EMBED=1 $PY eval/run.py eval/data/videomme_medium_dev2.jsonl \
  --methods videoctx --skip-existing \
  --out eval/results/meddev2_base.json

echo "MEDIUM DEV2 CHAIN DONE"
