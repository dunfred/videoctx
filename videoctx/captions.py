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

    # max_captions is a per-INDEX total, not per-call: repeat calls must not
    # keep adding captions past the cap (bloats the text view).
    budget = max_captions - sum(1 for s in ctx.shots if s.caption)
    if budget <= 0:
        return meter
    todo = [s for s in ctx.shots if not s.caption and s.keyframes]
    if len(todo) > budget:
        step = (len(todo) - 1) / (budget - 1) if budget > 1 else 0
        todo = [todo[round(i * step)] for i in range(budget)]

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


def caption_dense(
    index_dir: Path,
    ctx: VideoContext,
    model: str = CAPTION_MODEL,
    client: anthropic.Anthropic | None = None,
    min_shot_s: float = 45.0,
    max_dense: int = 150,
    progress=lambda *_: None,
) -> CostMeter:
    """Caption the non-midpoint keyframes of long shots, so a multi-minute
    shot contributes a timestamped caption roughly every 20s instead of one
    line total. Enrichment layer: optional, idempotent, provenance recorded
    on the shot's caption_model.

    max_dense is a per-INDEX total (same discipline as caption_shots): repeat
    calls never push past the cap."""
    client = client or anthropic.Anthropic()
    meter = CostMeter(model=model)

    budget = max_dense - sum(
        1 for s in ctx.shots for kf in s.keyframes if kf.caption)
    if budget <= 0:
        return meter
    todo = []
    for shot in ctx.shots:
        if shot.end_s - shot.start_s < min_shot_s or len(shot.keyframes) < 2:
            continue
        mid = shot.keyframes[len(shot.keyframes) // 2]
        todo += [(shot, kf) for kf in shot.keyframes
                 if kf is not mid and not kf.caption]
    if len(todo) > budget:
        step = (len(todo) - 1) / (budget - 1) if budget > 1 else 0
        todo = [todo[round(i * step)] for i in range(budget)]

    for shot, kf in todo:
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
        kf.caption = "".join(b.text for b in response.content if b.type == "text").strip()
        shot.caption_model = shot.caption_model or model
        progress(f"  kf @{kf.t_s:.0f}s: {kf.caption[:70]}")

    (index_dir / "index.json").write_text(ctx.model_dump_json(indent=2))
    return meter
