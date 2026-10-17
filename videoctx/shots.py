from __future__ import annotations

from pathlib import Path

from .schema import Shot

# Long shots get extra keyframes; see keyframe_times().
MAX_SHOT_GAP_S = 20.0


def detect_shots(video_path: Path, duration_s: float, min_shot_s: float = 1.0) -> list[Shot]:
    """Shot boundaries via PySceneDetect's AdaptiveDetector.

    Falls back to fixed 30-second chunks when detection finds nothing
    (static videos, slides, screen recordings).
    """
    from scenedetect import AdaptiveDetector, detect

    scene_list = detect(str(video_path), AdaptiveDetector(min_scene_len=int(min_shot_s * 15)))
    spans: list[tuple[float, float]] = [
        (start.get_seconds(), end.get_seconds()) for start, end in scene_list
    ]

    if not spans:
        spans = _fixed_chunks(duration_s, chunk_s=30.0)

    # Ensure full coverage to the end of the video.
    if spans and spans[-1][1] < duration_s - 0.5:
        spans.append((spans[-1][1], duration_s))

    return [Shot(id=i, start_s=round(s, 3), end_s=round(e, 3)) for i, (s, e) in enumerate(spans)]


def _fixed_chunks(duration_s: float, chunk_s: float) -> list[tuple[float, float]]:
    spans = []
    t = 0.0
    while t < duration_s:
        spans.append((t, min(t + chunk_s, duration_s)))
        t += chunk_s
    return spans


def keyframe_times(shot: Shot) -> list[float]:
    """Timestamps to extract for a shot: its midpoint, plus one frame every
    MAX_SHOT_GAP_S for long shots."""
    length = shot.end_s - shot.start_s
    if length <= MAX_SHOT_GAP_S:
        return [shot.start_s + length / 2]
    times = []
    t = shot.start_s + MAX_SHOT_GAP_S / 2
    while t < shot.end_s:
        times.append(t)
        t += MAX_SHOT_GAP_S
    return times
