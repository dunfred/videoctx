from __future__ import annotations

from pathlib import Path

from .schema import VideoContext

MIN_CONFIDENCE = 0.8
MAX_CHARS_PER_SHOT = 400


def ocr_shots(index_dir: Path, ctx: VideoContext, max_shots: int = 150,
              progress=lambda *_: None) -> int:
    """Run local OCR over each shot's middle keyframe and store the detected
    on-screen text in the index. Free, offline, one-time; makes verbatim
    screen text part of the cheap text layer.

    Idempotent per index: shots with ocr already set (including the empty
    string, which means "ran, nothing found") are skipped. Returns the number
    of shots processed."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    # Per-INDEX cap, like captions: bounded text views are an index invariant.
    budget = max_shots - sum(1 for s in ctx.shots if s.ocr is not None)
    todo = [s for s in ctx.shots if s.ocr is None and s.keyframes]
    if budget <= 0:
        return 0
    if len(todo) > budget:
        step = (len(todo) - 1) / (budget - 1) if budget > 1 else 0
        todo = [todo[round(i * step)] for i in range(budget)]
    done = 0
    for shot in todo:
        kf = shot.keyframes[len(shot.keyframes) // 2]
        result, _ = engine(str(index_dir / kf.file))
        lines = []
        seen = set()
        for _box, text, conf in result or []:
            text = text.strip()
            key = text.lower()
            if conf >= MIN_CONFIDENCE and len(text) >= 3 and key not in seen:
                seen.add(key)
                lines.append(text)
        joined = " | ".join(lines)
        shot.ocr = joined[:MAX_CHARS_PER_SHOT]
        done += 1
        if shot.ocr:
            progress(f"  shot {shot.id}: {shot.ocr[:70]}")
    (index_dir / "index.json").write_text(ctx.model_dump_json(indent=2))
    return done
