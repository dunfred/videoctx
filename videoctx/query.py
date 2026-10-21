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

# Range-capable survey, enabled with VIDEOCTX_SWEEP=1 while under dev A/B.
SURVEY_RANGED_TOOL = {
    "name": "survey",
    "description": (
        "Get a low-resolution thumbnail of shots in order, each labeled with "
        "its timestamp. Cheap (~50 tokens per shot). Covers the whole video "
        "by default but subsamples to at most 24 shots; pass start_s/end_s "
        "to survey a sub-range instead. The response header says how many "
        "shots the range contains versus how many are shown, so you know "
        "whether you have seen every shot."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start_s": {"type": ["number", "string"], "description": "range start (seconds or 'm:ss')"},
            "end_s": {"type": ["number", "string"], "description": "range end (seconds or 'm:ss')"},
        },
    },
}

SWEEP_RULES = (
    "\n8. EXHAUSTIVE SWEEP: a full-video survey subsamples to at most 24 "
    "shots. For counting or 'how many times' questions, and for ordering "
    "questions on fast-cut videos, survey consecutive sub-ranges until the "
    "coverage headers confirm you have seen EVERY shot, then count or order "
    "from that complete timeline. Never count from a subsampled survey.\n"
    "9. PEOPLE AND ROLES: to identify who a person is or their role, fetch "
    "frames at their appearances and read on-screen name titles and captions "
    "(lower thirds); prefer what the overlay says over inference from "
    "context."
)

FIND_MOMENTS_TOOL = {
    "name": "find_moments",
    "description": (
        "Visual search over the whole video: describe what you are looking "
        "for in plain language (an object, scene, action, or visual style) "
        "and get the timestamps of the best-matching shots plus their "
        "thumbnails. Much cheaper than scanning with get_frames; use it to "
        "locate the right moment, then zoom with get_frames."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {"type": "string",
                            "description": "What to look for, visually"},
            "top_k": {"type": "integer", "description": "Matches to return (default 5)"},
        },
        "required": ["description"],
    },
}

SYSTEM = (
    "You answer questions about a video using its index: a shot list with "
    "timestamps and visual captions, interleaved with the speech transcript "
    "and 'screen text' lines (verbatim on-screen text captured by OCR). "
    "Screen text is RELIABLE but NOT EXHAUSTIVE: it samples one frame per "
    "shot, so treat it as proof of what it contains, never as proof of "
    "absence or completeness. Enumeration, absence ('which does NOT appear'), "
    "and visual-causation questions still require frames, even when screen "
    "text seems relevant.\n"
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

# Text-path evidence discipline, default-on since the v0.3 freeze
# (disable with VIDEOCTX_NO_TEXT_EVIDENCE=1); mirrors the visual evidence
# rule for transcript claims. Validated on medium dev2 (84.4% vs 80.0%,
# replicated, zero regressions) and long-dev (94.4% vs 88.9%).
TEXT_EVIDENCE_RULES = (
    "\n6. TEXT EVIDENCE RULE: claims about what the video says or mentions "
    "(names, numbers, dates, topic lists, 'which was/wasn't mentioned') must "
    "be backed by a verbatim quote from the transcript or screen text; cite "
    "the quote. Gather EVERY mention of the asked entity or quantity before "
    "choosing: transcripts often contain several related statements (a "
    "claimed date, an origin date, a registration date) and the first "
    "relevant quote is not necessarily the one the question asks for. Never "
    "substitute outside knowledge about the subject for what the video "
    "itself says; when they conflict, the video wins.\n"
    "7. FINAL CHECK: before giving the letter, restate exactly what quantity "
    "or position the question asks for (a change vs a resulting value, "
    "'third mentioned' vs 'third shown', mentioned vs not mentioned), then "
    "confirm the letter against your own cited evidence. If your evidence "
    "identifies option X, your answer must be X."
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


def _parse_ts(value) -> float:
    """Models pass timestamps as numbers or clock strings ('1:12', '0:01:05')."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        total = 0.0
        for p in parts:
            total = total * 60 + p
        return total
    return float(s)


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
    # Edge size keys the cache file; the default keeps its legacy name so
    # existing thumb caches stay valid.
    stem = f"thumb_{src.name}" if edge == 256 else f"thumb{edge}_{src.name}"
    dst = src.with_name(stem)
    if not dst.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-vf", f"scale='min(iw,{edge})':-2", "-q:v", "6", str(dst)],
            check=True,
        )
    return dst


MAX_SURVEY_THUMBS = 24


def _survey_blocks(index_dir: Path, ctx: VideoContext,
                   start_s: float | None = None,
                   end_s: float | None = None) -> list[dict]:
    shots = [s for s in ctx.shots if s.keyframes]
    ranged = start_s is not None or end_s is not None
    if ranged:
        lo = max(0.0, start_s or 0.0)
        hi = end_s if end_s is not None else ctx.video.duration_s
        shots = [s for s in shots
                 if lo <= s.keyframes[len(s.keyframes) // 2].t_s <= hi]
    total = len(shots)
    # Fast-cut videos can have hundreds of shots; subsample evenly so a survey
    # stays ~1-2k tokens instead of blowing up the context.
    if len(shots) > MAX_SURVEY_THUMBS:
        step = (len(shots) - 1) / (MAX_SURVEY_THUMBS - 1)
        shots = [shots[round(i * step)] for i in range(MAX_SURVEY_THUMBS)]
    blocks: list[dict] = []
    if ranged:
        # Coverage header so exhaustive sweeps know when they have seen all
        # shots; emitted only on ranged calls to keep the default survey
        # output identical to the frozen config's.
        blocks.append({"type": "text", "text":
                       f"showing {len(shots)} of {total} shots in range"
                       + (" (subsampled)" if len(shots) < total else " (complete)")})
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


import os as _os

CASCADE_MODEL = _os.environ.get("VIDEOCTX_CASCADE_MODEL", "claude-haiku-4-5")

CASCADE_SYSTEM = (
    "You answer questions about a video using its index: shot captions, "
    "verbatim on-screen text (OCR), and the speech transcript, interleaved by "
    "timestamp. You cannot view frames. If the index gives you enough "
    "evidence to answer confidently, answer directly (follow any format the "
    "question requests) and cite timestamps.\n"
    "Escalate when the question requires seeing frames, depends on visual "
    "details the index does not capture, or you are not confident.\n"
    "Your reply MUST begin with either 'ANSWER:' followed by your answer, or "
    "the single word 'ESCALATE'. Never hedge in prose; if you would hedge, "
    "escalate."
)


class CascadeMeter:
    """Combines per-stage CostMeters (different models bill differently)."""

    def __init__(self, stages: list[CostMeter]):
        self.stages = stages

    @property
    def usd(self) -> float:
        return sum(m.usd for m in self.stages)

    def summary(self) -> dict:
        agg = {
            "usd": round(self.usd, 6),
            "calls": sum(m.calls for m in self.stages),
            "input_tokens": sum(m.input_tokens for m in self.stages),
            "output_tokens": sum(m.output_tokens for m in self.stages),
            "cache_read_tokens": sum(m.cache_read_tokens for m in self.stages),
            "cache_write_tokens": sum(m.cache_write_tokens for m in self.stages),
            "cache_write_1h_tokens": sum(m.cache_write_1h_tokens for m in self.stages),
            "tool_calls": {
                name: sum(m.tool_calls.get(name, 0) for m in self.stages)
                for m in self.stages for name in m.tool_calls
            },
            "stages": [m.summary() for m in self.stages],
        }
        return agg


def ask_cascade(
    index_dir: Path,
    ctx: VideoContext,
    question: str,
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
    video_path: Path | None = None,
) -> tuple[str, CascadeMeter, int]:
    """Two-stage answer: a cheap text-only model first; the full tool loop
    only when it escalates. Returns the same shape as ask()."""
    import os

    client = client or anthropic.Anthropic()
    stage1 = CostMeter(model=CASCADE_MODEL)
    tv = text_view(ctx,
                   include_ocr=os.environ.get("VIDEOCTX_NO_OCR") != "1",
                   include_dense=os.environ.get("VIDEOCTX_NO_DENSE") != "1")
    response = client.messages.create(
        model=CASCADE_MODEL,
        max_tokens=2048,
        system=CASCADE_SYSTEM,
        messages=[{"role": "user",
                   "content": f"VIDEO INDEX:\n{tv}\n\nQUESTION: {question}"}],
    )
    stage1.add(response.usage)
    answer = "".join(b.text for b in response.content if b.type == "text").strip()

    # Accept only an explicit confident answer; anything else (ESCALATE,
    # prose hedging, format non-compliance) falls through to the full loop.
    if answer.startswith("ANSWER:"):
        candidate = answer[len("ANSWER:"):].strip()
        if _os.environ.get("VIDEOCTX_CASCADE_VERIFY") != "1":
            return candidate, CascadeMeter([stage1]), 0
        # Cheap second opinion: audit the answer against the index; any doubt
        # (unsupported claim, pixels-needed question) escalates.
        verify = client.messages.create(
            model=CASCADE_MODEL,
            max_tokens=200,
            system=(
                "You audit an answer produced ONLY from a video's text index "
                "(captions, OCR screen text, transcript), with no access to "
                "frames. Reply PASS only if (a) the index lines quoted or "
                "referenced genuinely support the answer, and (b) the "
                "question is answerable without looking at pixels (not a "
                "color/count/pattern/appearance question the captions merely "
                "hint at). Otherwise reply FAIL. Reply with exactly one "
                "word: PASS or FAIL."
            ),
            messages=[{"role": "user", "content":
                       f"VIDEO INDEX:\n{tv}\n\nQUESTION: {question}\n\n"
                       f"PROPOSED ANSWER: {candidate}"}],
        )
        stage1.add(verify.usage)
        vtext = "".join(b.text for b in verify.content if b.type == "text").strip().upper()
        if vtext.startswith("PASS"):
            return candidate, CascadeMeter([stage1]), 0

    full_answer, stage2, frames = ask(
        index_dir, ctx, question, model=model, client=client, video_path=video_path)
    return full_answer, CascadeMeter([stage1, stage2]), frames


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
    import os

    tv = text_view(ctx,
                   include_ocr=os.environ.get("VIDEOCTX_NO_OCR") != "1",
                   include_dense=os.environ.get("VIDEOCTX_NO_DENSE") != "1")
    cache_ctl = {"type": "ephemeral", "ttl": "1h"} if len(tv) > 20000 else {"type": "ephemeral"}
    prefix: list[dict] = [{"type": "text", "text": f"VIDEO INDEX:\n{tv}"}]
    if os.environ.get("VIDEOCTX_CONTACT_SHEET") == "1":
        # Unconditional visual grounding: a subsampled thumbnail strip baked
        # into the cached prefix, so every question starts with the same
        # coarse view of the whole video at cache-read prices instead of
        # relying on the model's decision to survey. Density and resolution
        # are config knobs: coverage belongs to the amortized index layer.
        n_thumbs = int(os.environ.get("VIDEOCTX_SHEET_THUMBS", "24"))
        edge = int(os.environ.get("VIDEOCTX_SHEET_EDGE", "256"))
        sheet_shots = [s for s in ctx.shots if s.keyframes]
        if len(sheet_shots) > n_thumbs:
            step = (len(sheet_shots) - 1) / (n_thumbs - 1)
            sheet_shots = [sheet_shots[round(i * step)] for i in range(n_thumbs)]
        prefix.append({"type": "text", "text":
                       "CONTACT SHEET (one thumbnail per shot, subsampled "
                       "evenly; timestamps label each):"})
        for shot in sheet_shots:
            kf = shot.keyframes[len(shot.keyframes) // 2]
            data = base64.standard_b64encode(
                _thumb(index_dir, kf, edge=edge).read_bytes()).decode()
            prefix.append({"type": "text", "text": f"shot {shot.id} @ {kf.t_s:.1f}s:"})
            prefix.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": data}})
    # The breakpoint on the final block caches the whole stable prefix.
    prefix[-1]["cache_control"] = cache_ctl
    messages: list[dict] = [{
        "role": "user",
        "content": [*prefix, {"type": "text", "text": f"QUESTION: {question}"}],
    }]

    sweep = os.environ.get("VIDEOCTX_SWEEP") == "1"
    tools = [GET_FRAMES_TOOL, SURVEY_RANGED_TOOL if sweep else SURVEY_TOOL]
    if (index_dir / "embeddings.npz").exists() and os.environ.get("VIDEOCTX_NO_EMBED") != "1":
        tools.append(FIND_MOMENTS_TOOL)

    extra_rules = ""
    if os.environ.get("VIDEOCTX_NO_TEXT_EVIDENCE") != "1":
        extra_rules += TEXT_EVIDENCE_RULES
    if sweep:
        extra_rules += SWEEP_RULES
    system = SYSTEM
    if extra_rules:
        system = SYSTEM.replace(
            "\nGive the final answer directly and cite timestamps.",
            extra_rules + "\nGive the final answer directly and cite timestamps.")

    for _ in range(MAX_TOOL_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            tools=tools,
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
            meter.count_tool(block.name)
            if block.name == "find_moments":
                from .embeddings import find_moments
                hits = find_moments(index_dir, str(block.input["description"]),
                                    top_k=min(int(block.input.get("top_k", 5)), 10))
                content: list = []
                shot_by_id = {s.id: s for s in ctx.shots}
                for h in hits:
                    shot = shot_by_id.get(h["shot_id"])
                    if not shot or not shot.keyframes:
                        continue
                    kf = shot.keyframes[len(shot.keyframes) // 2]
                    content.append({"type": "text", "text":
                                    f"match @ {h['t_s']:.1f}s (score {h['score']:.2f}):"})
                    data = base64.standard_b64encode(
                        _thumb(index_dir, kf).read_bytes()).decode()
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg", "data": data}})
                frames_fetched += len(hits)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": content or [{"type": "text", "text": "no matches"}]})
                continue
            if block.name == "survey":
                if sweep:
                    args = block.input or {}
                    start = _parse_ts(args["start_s"]) if "start_s" in args else 0.0
                    end = _parse_ts(args["end_s"]) if "end_s" in args else ctx.video.duration_s
                    content = _survey_blocks(index_dir, ctx, start, end)
                    # count thumbnails actually sent, not total shots
                    frames_fetched += sum(1 for b in content if b["type"] == "image")
                else:
                    content = _survey_blocks(index_dir, ctx)
                    frames_fetched += len(ctx.shots)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                })
                continue
            args = block.input
            max_frames = min(int(args.get("max_frames", 4)), 8)
            edge = 1536 if args.get("detail") == "high" else 768
            start, end = _parse_ts(args["start_s"]), _parse_ts(args["end_s"])
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
