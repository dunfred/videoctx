from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .schema import Transcript, TranscriptSegment


def _decode_audio(video_path: Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode the audio track to 16 kHz mono float32 via ffmpeg. Avoids
    faster-whisper's PyAV decode path (version-conflicts with scenedetect)."""
    out = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-f", "f32le", "-",
        ],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32)


def transcribe(video_path: Path, model_size: str = "small") -> Transcript:
    """Transcribe with faster-whisper. Returns empty transcript on no speech."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    audio = _decode_audio(video_path)
    segments_iter, info = model.transcribe(audio, vad_filter=True)

    segments = [
        TranscriptSegment(start_s=round(s.start, 2), end_s=round(s.end, 2), text=s.text.strip())
        for s in segments_iter
        if s.text.strip()
    ]
    return Transcript(
        language=info.language if segments else None,
        model=f"faster-whisper/{model_size}",
        segments=segments,
    )
