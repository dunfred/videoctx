from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .schema import VideoInfo


def probe(video_path: Path) -> VideoInfo:
    """Read duration, dimensions, fps, and audio presence via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-show_entries", "stream=codec_type,width,height,avg_frame_rate",
            "-of", "json", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    duration = float(data["format"]["duration"])
    video_stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    num, _, den = video_stream.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0

    return VideoInfo(
        duration_s=duration,
        width=video_stream["width"],
        height=video_stream["height"],
        fps=round(fps, 3),
        has_audio=has_audio,
    )
