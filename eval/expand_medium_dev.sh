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

$PY eval/prep_indexes.py eval/data/videomme_medium_dev2.jsonl

for idx in $($PY -c "
import json
from pathlib import Path
vids = sorted({json.loads(l)['video'] for l in open('eval/data/videomme_medium_dev2.jsonl') if l.strip()})
for v in vids:
    p = Path(v)
    print(p.parent / (p.stem + '.vctx'))
"); do
  .venv/bin/videoctx ocr "$idx"
  .venv/bin/videoctx embed "$idx"
done

echo "ENRICHMENT DONE, running baseline (one-touch, embeddings off)"
VIDEOCTX_NO_EMBED=1 $PY eval/run.py eval/data/videomme_medium_dev2.jsonl \
  --methods videoctx --skip-existing \
  --out eval/results/meddev2_base.json

echo "MEDIUM DEV2 CHAIN DONE"
