from __future__ import annotations

from .schema import VideoContext


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def text_view(ctx: VideoContext) -> str:
    """Compact plain-text rendering of an index: shot structure interleaved
    with transcript segments by timestamp. This is the cheap path an agent
    reads before deciding whether it needs frames."""
    lines = [
        f"VIDEO: {ctx.video.duration_s:.0f}s, {ctx.video.width}x{ctx.video.height}, "
        f"{len(ctx.shots)} shots, "
        f"transcript: {'yes (' + str(ctx.transcript.language) + ')' if ctx.transcript.segments else 'none'}",
        "",
    ]
    # Two event streams merged by time: visual anchors (captioned shots) and
    # speech blocks (transcript segments grouped into ~25s windows). Shots
    # that carry neither a caption nor speech collapse into gap lines. This
    # keeps hour-long views in the 15-25k token range instead of 50k+.
    SPEECH_WINDOW_S = 25.0

    events: list[tuple[float, str]] = []

    captioned = [s for s in ctx.shots if s.caption]
    for shot in captioned:
        events.append((shot.start_s, f"[{_ts(shot.start_s)}] {shot.caption}"))

    # Silence+uncaptioned gaps (so the model knows the span is accounted for).
    cap_or_speech: list[tuple[float, float]] = [(s.start_s, s.end_s) for s in captioned]
    cap_or_speech += [(seg.start_s, seg.end_s) for seg in ctx.transcript.segments]
    gap_run: list = []
    for shot in ctx.shots:
        covered = any(a < shot.end_s and b > shot.start_s for a, b in cap_or_speech)
        if covered:
            if len(gap_run) >= 2:
                events.append((
                    gap_run[0].start_s,
                    f"[{_ts(gap_run[0].start_s)}-{_ts(gap_run[-1].end_s)}] "
                    f"({len(gap_run)} uncaptioned silent shots)",
                ))
            gap_run = []
        else:
            gap_run.append(shot)
    if len(gap_run) >= 2:
        events.append((
            gap_run[0].start_s,
            f"[{_ts(gap_run[0].start_s)}-{_ts(gap_run[-1].end_s)}] "
            f"({len(gap_run)} uncaptioned silent shots)",
        ))

    # Speech grouped into windows.
    block: list = []
    for seg in ctx.transcript.segments:
        if block and seg.end_s - block[0].start_s > SPEECH_WINDOW_S:
            events.append((block[0].start_s,
                           f"  {_ts(block[0].start_s)} \"{' '.join(s.text for s in block)}\""))
            block = []
        block.append(seg)
    if block:
        events.append((block[0].start_s,
                       f"  {_ts(block[0].start_s)} \"{' '.join(s.text for s in block)}\""))

    events.sort(key=lambda e: e[0])
    lines.extend(text for _, text in events)
    return "\n".join(lines)
