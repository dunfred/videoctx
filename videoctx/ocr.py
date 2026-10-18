from __future__ import annotations

from pathlib import Path

from .schema import VideoContext

MIN_CONFIDENCE = 0.8
MAX_CHARS_PER_SHOT = 400


def ocr_shots(index_dir: Path, ctx: VideoContext, progress=lambda *_: None) -> int:
    """Run local OCR over each shot's middle keyframe and store the detected
    on-screen text in the index. Free, offline, one-time; makes verbatim
    screen text part of the cheap text layer.

    Idempotent per index: shots with ocr already set (including the empty
    string, which means "ran, nothing found") are skipped. Returns the number
    of shots processed."""
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    done = 0
    for shot in ctx.shots:
        if shot.ocr is not None or not shot.keyframes:
            continue
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
