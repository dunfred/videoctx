from __future__ import annotations

import base64
from pathlib import Path

import anthropic

from .pricing import CostMeter
from .schema import Keyframe, VideoContext
from .textview import text_view

DEFAULT_MODEL = "claude-opus-4-8"
MAX_TOOL_TURNS = 6

GET_FRAMES_TOOL = {
    "name": "get_frames",
    "description": (
        "Fetch keyframes from the video within a time range. Call this when the "
        "text index is not enough to answer and you need to see the frames. Use "
        "the narrowest time range the question allows; frames are expensive."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start_s": {"type": "number", "description": "Range start, in seconds"},
            "end_s": {"type": "number", "description": "Range end, in seconds"},
            "max_frames": {
                "type": "integer",
                "description": "Maximum frames to return (default 4, max 8)",
            },
            "detail": {
                "type": "string",
                "enum": ["standard", "high"],
                "description": (
                    "Use 'high' (2x resolution, ~4x token cost) when the "
                    "question hinges on small text, colors, patterns, or "
                    "fine object details; pair it with a narrow time range."
                ),
            },
        },
        "required": ["start_s", "end_s"],
    },
}

SURVEY_TOOL = {
    "name": "survey",
    "description": (
        "Get a low-resolution thumbnail of every shot in the video, in order, "
        "each labeled with its timestamp. Cheap (~50 tokens per shot). Use this "
        "to visually locate the relevant moment before calling get_frames, "
        "whenever the text index does not tell you where to look."
    ),
    "input_schema": {"type": "object", "properties": {}},
}

SYSTEM = (
    "You answer questions about a video using its index: a shot list with "
    "timestamps and visual captions, interleaved with the speech transcript "
    "and 'screen text' lines (verbatim on-screen text captured by OCR; treat "
    "it as reliable evidence for questions about text shown on screen).\n"
    "Strategy:\n"
    "1. Answer from the text index alone when it is sufficient.\n"
    "2. CAPTIONS ARE LOSSY one-line summaries. Never answer questions about "
    "counts, colors, patterns, on-screen text, or fine visual details from "
    "captions alone; verify with frames.\n"
    "3. If the question is visual and you don't know where to look, call "
    "survey to sweep the whole video cheaply, then get_frames to zoom into "
    "the relevant range. Use detail='high' when small text, colors, or "
    "patterns matter.\n"
    "4. ENUMERATION: for 'how many', 'first/second time', or ordering "
    "questions, sweep systematically: survey, identify every candidate "
    "occurrence, then verify each with frames before counting or ordering. "
    "For 'which option is incorrect/true' questions, check each option "
    "against evidence, not just the first plausible one.\n"
    "5. EVIDENCE RULE: for visual questions, your answer must be supported by "
    "something you actually saw in a fetched frame or thumbnail; cite its "
    "timestamp. If the frames you fetched do not show the answer, fetch a "
    "different range, higher density, or higher detail instead of guessing. "
    "Never substitute general world knowledge for missing visual evidence.\n"
    "Give the final answer directly and cite timestamps."
)


def extract_on_demand(
    index_dir: Path,
    video_path: Path,
    ctx: VideoContext,
    start_s: float,
    end_s: float,
    max_frames: int,
    max_edge: int = 768,
) -> list[Keyframe]:
    """Extract evenly spaced frames across [start_s, end_s] from the source
    video at query time. Unlike select_frames (pre-extracted shot midpoints),
    this reaches any moment at any density. Extracted frames are cached in the
    index so repeat questions reuse them."""
    from .frames import extract_frame

    # Clamp the whole range inside the video; models sometimes request
    # timestamps past the end (hallucinated durations, off-by-one math).
    duration = ctx.video.duration_s
    start_s = min(max(0.0, start_s), max(0.0, duration - 1.0))
    end_s = min(max(end_s, start_s + 0.5), duration)
    span = end_s - start_s
    times = [start_s + (i + 0.5) * span / max_frames for i in range(max_frames)]

    frames_dir = index_dir / "frames"
    out: list[Keyframe] = []
    for t in times:
        path = frames_dir / f"ondemand_e{max_edge}_t{t:08.2f}.jpg"
        if path.exists():
            from .frames import _image_size
            w, h = _image_size(path)
            out.append(Keyframe(t_s=round(t, 3), file=f"frames/{path.name}", width=w, height=h))
        else:
            out.append(extract_frame(video_path, t, path, max_edge=max_edge))
    return out


def select_frames(ctx: VideoContext, start_s: float, end_s: float, max_frames: int) -> list[Keyframe]:
    candidates = [
        kf for shot in ctx.shots for kf in shot.keyframes if start_s <= kf.t_s <= end_s
    ]
    if not candidates:
        # Nothing strictly inside the range: return the nearest keyframe.
        all_kfs = [kf for shot in ctx.shots for kf in shot.keyframes]
        if not all_kfs:
            return []
        mid = (start_s + end_s) / 2
        return [min(all_kfs, key=lambda kf: abs(kf.t_s - mid))]
    if len(candidates) <= max_frames:
        return candidates
    step = (len(candidates) - 1) / (max_frames - 1) if max_frames > 1 else 0
    return [candidates[round(i * step)] for i in range(max_frames)]


def _thumb(index_dir: Path, kf: Keyframe, edge: int = 256) -> Path:
    """Downscaled cached copy of a keyframe for cheap surveying."""
    import subprocess

    src = index_dir / kf.file
    dst = src.with_name(f"thumb_{src.name}")
    if not dst.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-vf", f"scale='min(iw,{edge})':-2", "-q:v", "6", str(dst)],
            check=True,
        )
    return dst


MAX_SURVEY_THUMBS = 24


def _survey_blocks(index_dir: Path, ctx: VideoContext) -> list[dict]:
    shots = [s for s in ctx.shots if s.keyframes]
    # Fast-cut videos can have hundreds of shots; subsample evenly so a survey
    # stays ~1-2k tokens instead of blowing up the context.
    if len(shots) > MAX_SURVEY_THUMBS:
        step = (len(shots) - 1) / (MAX_SURVEY_THUMBS - 1)
        shots = [shots[round(i * step)] for i in range(MAX_SURVEY_THUMBS)]
    blocks: list[dict] = []
    for shot in shots:
        kf = shot.keyframes[len(shot.keyframes) // 2]
        thumb = _thumb(index_dir, kf)
        data = base64.standard_b64encode(thumb.read_bytes()).decode()
        blocks.append({"type": "text", "text": f"shot {shot.id} @ {kf.t_s:.1f}s:"})
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })
    return blocks or [{"type": "text", "text": "No keyframes available."}]


def _frame_blocks(index_dir: Path, frames: list[Keyframe]) -> list[dict]:
    blocks: list[dict] = []
    for kf in frames:
        data = base64.standard_b64encode((index_dir / kf.file).read_bytes()).decode()
        blocks.append({"type": "text", "text": f"frame at t={kf.t_s:.1f}s:"})
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })
    if not blocks:
        blocks.append({"type": "text", "text": "No keyframes available in that range."})
    return blocks


def ask(
    index_dir: Path,
    ctx: VideoContext,
    question: str,
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
    video_path: Path | None = None,
) -> tuple[str, CostMeter, int]:
    """Answer a question about an indexed video.

    Returns (answer, cost meter, frames fetched)."""
    client = client or anthropic.Anthropic()
    meter = CostMeter(model=model)
    frames_fetched = 0

    # The text view is a stable prefix across every question about the same
    # video: mark it cacheable so repeat questions bill at ~0.1x input price.
    # (Prefixes under the model's minimum, ~4k tokens on Opus, silently skip
    # caching; harmless for short videos.) Large views get the 1-hour TTL:
    # the 2x write premium is paid once per video, and the cache then survives
    # rate-limit stalls and whole question batches, where the default 5-minute
    # TTL was observed to lapse and force premium re-writes.
    tv = text_view(ctx)
    cache_ctl = {"type": "ephemeral", "ttl": "1h"} if len(tv) > 20000 else {"type": "ephemeral"}
    messages: list[dict] = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"VIDEO INDEX:\n{tv}", "cache_control": cache_ctl},
            {"type": "text", "text": f"QUESTION: {question}"},
        ],
    }]

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=SYSTEM,
            tools=[GET_FRAMES_TOOL, SURVEY_TOOL],
            messages=messages,
        )
        meter.add(response.usage)

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            return answer.strip(), meter, frames_fetched

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "survey":
                frames_fetched += len(ctx.shots)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _survey_blocks(index_dir, ctx),
                })
                continue
            args = block.input
            max_frames = min(int(args.get("max_frames", 4)), 8)
            edge = 1536 if args.get("detail") == "high" else 768
            start, end = float(args["start_s"]), float(args["end_s"])
            if video_path is not None and video_path.exists():
                frames = extract_on_demand(
                    index_dir, video_path, ctx, start, end, max_frames, max_edge=edge)
            else:
                frames = select_frames(ctx, start, end, max_frames)
            frames_fetched += len(frames)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _frame_blocks(index_dir, frames),
            })
        # Keep one moving cache breakpoint on the newest tool result so each
        # loop iteration reads the prior history (incl. images) from cache.
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for blk in msg["content"]:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        blk.pop("cache_control", None)
        results[-1]["cache_control"] = {"type": "ephemeral"}
        messages.append({"role": "user", "content": results})

    return "(no final answer: tool-turn limit reached)", meter, frames_fetched
