from __future__ import annotations

import base64
from pathlib import Path

import anthropic

from .pricing import CostMeter
from .schema import VideoContext

CAPTION_MODEL = "claude-haiku-4-5"

CAPTION_PROMPT = (
    "Describe this video frame in ONE dense line: subjects, actions, setting, "
    "notable objects, and any readable on-screen text (quote it). No preamble."
)


def caption_shots(
    index_dir: Path,
    ctx: VideoContext,
    model: str = CAPTION_MODEL,
    client: anthropic.Anthropic | None = None,
    max_captions: int = 150,
    progress=lambda *_: None,
) -> CostMeter:
    """Caption shots' middle keyframes with a cheap VLM and write the captions
    back into index.json. One-time cost, amortized over all queries.

    Long fast-cut videos can have many hundreds of shots; captioning is capped
    at max_captions, subsampled evenly, to bound both the one-time cost and
    the text-view size."""
    client = client or anthropic.Anthropic()
    meter = CostMeter(model=model)

    todo = [s for s in ctx.shots if not s.caption and s.keyframes]
    if len(todo) > max_captions:
        step = (len(todo) - 1) / (max_captions - 1)
        todo = [todo[round(i * step)] for i in range(max_captions)]

    for shot in todo:
        kf = shot.keyframes[len(shot.keyframes) // 2]
        data = base64.standard_b64encode((index_dir / kf.file).read_bytes()).decode()
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": data}},
                    {"type": "text", "text": CAPTION_PROMPT},
                ],
            }],
        )
        meter.add(response.usage)
        shot.caption = "".join(b.text for b in response.content if b.type == "text").strip()
        shot.caption_model = model
        progress(f"  shot {shot.id}: {shot.caption[:70]}")

    (index_dir / "index.json").write_text(ctx.model_dump_json(indent=2))
    return meter
