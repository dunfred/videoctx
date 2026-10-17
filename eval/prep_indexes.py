"""Pre-build and caption indexes for every video referenced in a questions
file, so eval runs pay query cost only.

Usage:
    python eval/prep_indexes.py eval/data/videomme_long.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from videoctx.captions import caption_shots  # noqa: E402
from videoctx.indexer import build_index, load_index  # noqa: E402

questions_file = Path(sys.argv[1])
videos = sorted({json.loads(l)["video"] for l in questions_file.read_text().splitlines() if l.strip()})
print(f"{len(videos)} videos to prepare", flush=True)

total_caption_usd = 0.0
for i, v in enumerate(videos, 1):
    video_path = Path(v)
    index_dir = video_path.parent / f"{video_path.stem}.vctx"
    t0 = time.monotonic()
    if not (index_dir / "index.json").exists():
        print(f"[{i}/{len(videos)}] indexing {video_path.name}", flush=True)
        build_index(video_path, out_dir=index_dir, progress=lambda *_: None)
    ctx = load_index(index_dir)
    if any(s.caption is None for s in ctx.shots):
        meter = caption_shots(index_dir, ctx)
        total_caption_usd += meter.usd
        print(f"[{i}/{len(videos)}] {len(ctx.shots)} shots, captions ${meter.usd:.3f}, "
              f"{time.monotonic()-t0:.0f}s", flush=True)
    else:
        print(f"[{i}/{len(videos)}] already prepared ({len(ctx.shots)} shots)", flush=True)

print(f"DONE: all indexes ready, caption spend ${total_caption_usd:.3f}", flush=True)
