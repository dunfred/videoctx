"""videoctx MCP server.

Gives any MCP client (Claude Code, Cursor, ...) the ability to read videos the
way it reads images: index once, then work from the cheap text layer and pull
frames only when needed. The calling agent's own model does the reasoning —
the core tools below require no API key and make no model calls.

Install into Claude Code:
    claude mcp add videoctx -- /path/to/videoctx/.venv/bin/videoctx-mcp

Intended agent loop:
    1. video_overview(path)   -> shot structure + captions + transcript (cheap)
    2. survey_video(path)     -> one thumbnail per shot, when unsure where to look
    3. get_frames(path, t0, t1) -> full-res frames from the exact moment
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

from .indexer import build_index, load_index
from .schema import VideoContext
from .textview import text_view

server = MCPServer(
    "videoctx",
    instructions=(
        "Tools for reading video files. Call video_overview first — its text "
        "layer (shots, captions, transcript) answers most questions without "
        "any frames. Fetch frames only for visual details the text does not "
        "cover, on the narrowest time range possible. Evidence rule: answer "
        "visual questions only from frames you have actually seen; if fetched "
        "frames do not show the answer, fetch a different range or higher "
        "density instead of guessing."
    ),
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _resolve(path: str) -> tuple[Path, VideoContext]:
    """Accept a video file or a .vctx index dir; auto-index videos on first use."""
    p = Path(path).expanduser().resolve()
    if p.is_dir() and (p / "index.json").exists():
        return p, load_index(p)
    if p.suffix.lower() in VIDEO_SUFFIXES and p.exists():
        index_dir = p.parent / f"{p.stem}.vctx"
        if not (index_dir / "index.json").exists():
            build_index(p, out_dir=index_dir, progress=lambda *_: None)
        return index_dir, load_index(index_dir)
    raise ValueError(f"not a video file or .vctx index: {path}")


def _source_video(index_dir: Path) -> Path | None:
    import json

    manifest = index_dir / "manifest.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())["source"]
        for candidate in (Path(source.get("path", "")), index_dir.parent / source["filename"]):
            if candidate.name and candidate.exists():
                return candidate
    return None


@server.tool()
def index_video(path: str) -> str:
    """Index a video file for querying (shots, keyframes, speech transcript).
    Runs locally; takes roughly 10-20% of the video's duration. Idempotent —
    returns immediately if the video is already indexed. The other tools
    auto-index too, so calling this first is only needed to control when the
    wait happens."""
    index_dir, ctx = _resolve(path)
    return (
        f"indexed: {index_dir}\n"
        f"{ctx.video.duration_s:.0f}s, {len(ctx.shots)} shots, "
        f"{len(ctx.transcript.segments)} transcript segments "
        f"(language={ctx.transcript.language})"
    )


@server.tool()
def video_overview(path: str) -> str:
    """The cheap text layer of a video: shot list with timestamps and visual
    captions (when present), interleaved with the speech transcript. Read this
    before fetching any frames — it answers most questions by itself."""
    _, ctx = _resolve(path)
    return text_view(ctx)


@server.tool()
def survey_video(path: str) -> list:
    """One small thumbnail per shot across the whole video (subsampled to at
    most 24), each labeled with its timestamp. Cheap visual sweep for locating
    the relevant moment before zooming in with get_frames."""
    from .query import MAX_SURVEY_THUMBS, _thumb

    index_dir, ctx = _resolve(path)
    shots = [s for s in ctx.shots if s.keyframes]
    if len(shots) > MAX_SURVEY_THUMBS:
        step = (len(shots) - 1) / (MAX_SURVEY_THUMBS - 1)
        shots = [shots[round(i * step)] for i in range(MAX_SURVEY_THUMBS)]
    out: list = []
    for shot in shots:
        kf = shot.keyframes[len(shot.keyframes) // 2]
        out.append(f"shot {shot.id} @ {kf.t_s:.1f}s:")
        out.append(Image(path=_thumb(index_dir, kf)))
    return out or ["no keyframes available"]


@server.tool()
def get_frames(path: str, start_s: float, end_s: float, max_frames: int = 4) -> list:
    """Full-resolution frames evenly spaced across [start_s, end_s]. Frames
    are expensive — use the narrowest range the question allows, and raise
    max_frames (up to 8) only when a sparse sample missed the moment."""
    from .query import extract_on_demand, select_frames

    index_dir, ctx = _resolve(path)
    max_frames = max(1, min(int(max_frames), 8))
    video = _source_video(index_dir)
    if video is not None:
        frames = extract_on_demand(index_dir, video, ctx, start_s, end_s, max_frames)
    else:
        frames = select_frames(ctx, start_s, end_s, max_frames)
    out: list = []
    for kf in frames:
        out.append(f"frame @ {kf.t_s:.1f}s:")
        out.append(Image(path=index_dir / kf.file))
    return out or ["no frames available in that range"]


@server.tool()
def transcript_search(path: str, query: str, context_s: float = 15.0) -> str:
    """Case-insensitive search over the transcript. Returns matching segments
    with surrounding context and timestamps — use the timestamps to target
    get_frames."""
    _, ctx = _resolve(path)
    q = query.lower()
    hits = []
    segs = ctx.transcript.segments
    for i, seg in enumerate(segs):
        if q in seg.text.lower():
            around = [s for s in segs if abs(s.start_s - seg.start_s) <= context_s]
            block = " ".join(s.text for s in around)
            hits.append(f"[{seg.start_s:.1f}s] ...{block}...")
    if not hits:
        return f"no transcript matches for {query!r}"
    return "\n".join(hits[:10])


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
