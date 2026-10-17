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
    segs = list(ctx.transcript.segments)
    si = 0
    for shot in ctx.shots:
        header = f"[shot {shot.id} | {_ts(shot.start_s)}-{_ts(shot.end_s)}]"
        if shot.caption:
            header += f" {shot.caption}"
        lines.append(header)
        while si < len(segs) and segs[si].start_s < shot.end_s:
            seg = segs[si]
            lines.append(f"  {_ts(seg.start_s)} \"{seg.text}\"")
            si += 1
    # Any trailing segments past the last shot boundary.
    for seg in segs[si:]:
        lines.append(f"  {_ts(seg.start_s)} \"{seg.text}\"")
    return "\n".join(lines)
