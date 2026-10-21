"""Offline smoke test: exercises everything except the live API call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from videoctx.indexer import load_index
from videoctx.query import GET_FRAMES_TOOL, _frame_blocks, select_frames
from videoctx.textview import text_view

index_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "indexes/demo.vctx")
ctx = load_index(index_dir)

# Text view renders and includes every transcript segment.
tv = text_view(ctx)
assert all(seg.text[:20] in tv for seg in ctx.transcript.segments), "transcript missing from text view"
assert f"{len(ctx.shots)} shots" in tv

# The header must survive OCR rendering (a shadowed variable once destroyed
# it on every OCR-enriched index).
ctx.shots[0].ocr = "SMOKE OCR LINE"
tv_ocr = text_view(ctx)
assert tv_ocr.startswith("VIDEO:"), "header lost when OCR present"
assert "SMOKE OCR LINE" in tv_ocr
ctx.shots[0].ocr = None

# Frame selection: in-range, subsampled, and nearest-fallback.
frames = select_frames(ctx, 8.0, 16.0, max_frames=4)
assert frames and all(8.0 <= f.t_s <= 16.0 for f in frames), "in-range selection failed"
one = select_frames(ctx, 0.0, ctx.video.duration_s, max_frames=1)
assert len(one) == 1, "subsampling failed"
nearest = select_frames(ctx, 999.0, 1000.0, max_frames=4)
assert len(nearest) == 1, "nearest-fallback failed"

# Tool-result blocks are valid image blocks with real JPEG bytes.
blocks = _frame_blocks(index_dir, frames)
images = [b for b in blocks if b["type"] == "image"]
assert len(images) == len(frames)
import base64
assert base64.standard_b64decode(images[0]["source"]["data"])[:2] == b"\xff\xd8", "not a JPEG"

# Tool schema sanity.
assert GET_FRAMES_TOOL["input_schema"]["required"] == ["start_s", "end_s"]

print(f"smoke OK: {len(ctx.shots)} shots, {sum(len(s.keyframes) for s in ctx.shots)} keyframes, "
      f"{len(ctx.transcript.segments)} transcript segments, "
      f"text view {len(tv)} chars (~{len(tv)//4} tokens)")
