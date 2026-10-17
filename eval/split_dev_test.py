"""Split a questions jsonl into dev/test halves BY VIDEO (no leakage:
all questions about one video land on the same side).

Usage: python eval/split_dev_test.py eval/data/videomme_short.jsonl
Writes <stem>_dev.jsonl and <stem>_test.jsonl next to the input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
videos = sorted({r["video"] for r in rows})
dev_videos = set(videos[0::2])

dev = [r for r in rows if r["video"] in dev_videos]
test = [r for r in rows if r["video"] not in dev_videos]
for name, subset in (("dev", dev), ("test", test)):
    out = src.with_name(f"{src.stem}_{name}.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in subset) + "\n")
    print(f"{out.name}: {len(subset)} questions, {len({r['video'] for r in subset})} videos")
