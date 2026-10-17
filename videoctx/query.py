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
        },
        "required": ["start_s", "end_s"],
    },
}

SYSTEM = (
    "You answer questions about a video using its index: a shot list with "
    "timestamps, interleaved with the speech transcript. Answer from the text "
    "index alone when it is sufficient. When the question depends on visual "
    "content the text does not capture, call get_frames on the narrowest "
    "relevant time range. Be decisive; do not fetch frames you do not need. "
    "Give the final answer directly and cite timestamps."
)


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
) -> tuple[str, CostMeter, int]:
    """Answer a question about an indexed video.

    Returns (answer, cost meter, frames fetched)."""
    client = client or anthropic.Anthropic()
    meter = CostMeter(model=model)
    frames_fetched = 0

    # The text view is a stable prefix across every question about the same
    # video: mark it cacheable so repeat questions bill at ~0.1x input price.
    # (Prefixes under the model's minimum, ~4k tokens on Opus, silently skip
    # caching; harmless for short videos.)
    messages: list[dict] = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"VIDEO INDEX:\n{text_view(ctx)}",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": f"QUESTION: {question}"},
        ],
    }]

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=SYSTEM,
            tools=[GET_FRAMES_TOOL],
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
            args = block.input
            max_frames = min(int(args.get("max_frames", 4)), 8)
            frames = select_frames(ctx, float(args["start_s"]), float(args["end_s"]), max_frames)
            frames_fetched += len(frames)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _frame_blocks(index_dir, frames),
            })
        messages.append({"role": "user", "content": results})

    return "(no final answer: tool-turn limit reached)", meter, frames_fetched
