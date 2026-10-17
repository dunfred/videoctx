from __future__ import annotations

import subprocess
from pathlib import Path

from .schema import Keyframe


def extract_frame(
    video_path: Path,
    t_s: float,
    out_path: Path,
    max_edge: int = 768,
    quality: int = 4,
) -> Keyframe:
    """Extract one frame at t_s, scaled so the long edge is <= max_edge."""
    scale = (
        f"scale='if(gt(iw,ih),min(iw,{max_edge}),-2)'"
        f":'if(gt(iw,ih),-2,min(ih,{max_edge}))'"
    )
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{t_s:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-vf", scale, "-q:v", str(quality),
            "-strict", "unofficial",  # ffmpeg 9 mjpeg rejects non-full-range YUV otherwise
            str(out_path),
        ],
        check=True,
    )
    width, height = _image_size(out_path)
    return Keyframe(t_s=round(t_s, 3), file=f"frames/{out_path.name}", width=width, height=height)


def _image_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    w, h = out.stdout.strip().split("x")
    return int(w), int(h)
