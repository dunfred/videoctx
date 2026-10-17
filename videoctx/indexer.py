from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .frames import extract_frame
from .probe import probe
from .schema import VideoContext
from .shots import detect_shots, keyframe_times
from .transcribe import transcribe


def build_index(
    video_path: Path,
    out_dir: Path | None = None,
    max_edge: int = 768,
    whisper_model: str = "small",
    skip_transcript: bool = False,
    progress=print,
) -> Path:
    """Index a video into a .vctx directory. Returns the index path."""
    video_path = video_path.resolve()
    if out_dir is None:
        out_dir = video_path.with_suffix("").with_name(video_path.stem + ".vctx")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    progress(f"probing {video_path.name}")
    info = probe(video_path)

    progress(f"detecting shots ({info.duration_s:.0f}s video)")
    shots = detect_shots(video_path, info.duration_s)
    progress(f"  {len(shots)} shots")

    progress("extracting keyframes")
    n_frames = 0
    for shot in shots:
        for t in keyframe_times(shot):
            out_path = frames_dir / f"shot{shot.id:04d}_t{t:08.2f}.jpg"
            shot.keyframes.append(extract_frame(video_path, t, out_path, max_edge=max_edge))
            n_frames += 1
    progress(f"  {n_frames} keyframes")

    if skip_transcript or not info.has_audio:
        from .schema import Transcript
        transcript = Transcript()
        progress("skipping transcript" if skip_transcript else "no audio stream")
    else:
        progress(f"transcribing (faster-whisper/{whisper_model})")
        transcript = transcribe(video_path, model_size=whisper_model)
        progress(f"  {len(transcript.segments)} segments, language={transcript.language}")

    ctx = VideoContext(video=info, shots=shots, transcript=transcript)
    (out_dir / "index.json").write_text(ctx.model_dump_json(indent=2))

    manifest = {
        "format": "vctx",
        "format_version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": {"name": "videoctx", "version": __version__},
        "params": {
            "max_edge": max_edge,
            "whisper_model": whisper_model,
            "shot_detector": "scenedetect/AdaptiveDetector",
        },
        "source": {
            "filename": video_path.name,
            "path": str(video_path),  # hint only; consumers must tolerate it moving
            "sha256": _sha256(video_path),
            "size_bytes": video_path.stat().st_size,
        },
        "indexing_wall_s": round(time.monotonic() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    progress(f"index written to {out_dir} ({manifest['indexing_wall_s']}s)")
    return out_dir


def load_index(index_dir: Path) -> VideoContext:
    return VideoContext.model_validate_json((Path(index_dir) / "index.json").read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
