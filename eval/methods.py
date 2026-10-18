"""Eval methods. Each method answers a multiple-choice question about a video
and reports its own cost. The harness compares accuracy per dollar.

Method contract:
    answer(video_path, question, options) -> MethodResult
"""
from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from pathlib import Path

MC_PROMPT = (
    "{question}\n\nOptions:\n{options}\n\n"
    "The FIRST character of your reply must be the letter of the best option "
    "(A, B, C, or D), followed by a period and one sentence of justification."
)


@dataclass
class MethodResult:
    answer_text: str
    predicted_letter: str | None
    usd: float
    detail: dict


def _format_options(options: list[str]) -> str:
    return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))


def _extract_letter(text: str, n_options: int) -> str | None:
    import re

    valid = "".join(chr(65 + i) for i in range(n_options))
    text = text.strip()
    # 1. Leading letter ("B." / "**B**" / "(B)").
    m = re.match(rf"^\W{{0,2}}([{valid}])(?![A-Za-z])", text)
    if m:
        return m.group(1)
    # 2. Explicit verdict ("the answer is C", "option B", "Answer: D").
    m = re.search(rf"(?:answer|option|choice)\s*(?:is\s*)?:?\s*\**([{valid}])(?![A-Za-z])",
                  text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3. Last standalone letter followed by option punctuation ("B. Black").
    hits = re.findall(rf"(?<![A-Za-z])([{valid}])[\.\):]", text)
    return hits[-1] if hits else None


# ---------------------------------------------------------------- videoctx

def method_videoctx(video_path: Path, question: str, options: list[str],
                    model: str = "claude-opus-4-8", index_cache: Path | None = None) -> MethodResult:
    """Index once (cached on disk), then query via the frames-on-demand loop."""
    from videoctx.captions import caption_shots
    from videoctx.indexer import build_index, load_index
    from videoctx.query import ask

    index_dir = (index_cache or video_path.parent) / f"{video_path.stem}.vctx"
    if not (index_dir / "index.json").exists():
        build_index(video_path, out_dir=index_dir, progress=lambda *_: None)

    ctx = load_index(index_dir)
    if any(s.caption is None for s in ctx.shots):
        caption_shots(index_dir, ctx)  # one-time; written back into the index
    prompt = MC_PROMPT.format(question=question, options=_format_options(options))
    import os

    if os.environ.get("VIDEOCTX_CASCADE") == "1":
        from videoctx.query import ask_cascade
        answer, meter, frames = ask_cascade(index_dir, ctx, prompt, model=model, video_path=video_path)
    else:
        answer, meter, frames = ask(index_dir, ctx, prompt, model=model, video_path=video_path)
    return MethodResult(
        answer_text=answer,
        predicted_letter=_extract_letter(answer, len(options)),
        usd=meter.usd,
        detail={**meter.summary(), "frames_fetched": frames},
    )


# ---------------------------------------------------------------- naive sampling

def method_naive(video_path: Path, question: str, options: list[str],
                 model: str = "claude-opus-4-8", fps: float = 1 / 10) -> MethodResult:
    """Baseline: uniform frame sampling (default one frame per 10s), all frames
    in one request. No transcript, no structure."""
    import anthropic

    from videoctx.pricing import CostMeter
    from videoctx.probe import probe

    info = probe(video_path)
    n = max(4, min(60, int(info.duration_s * fps)))
    content: list[dict] = []
    for i in range(n):
        t = (i + 0.5) * info.duration_s / n
        jpg = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", "scale='min(iw,768)':-2", "-q:v", "4",
             "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
            capture_output=True, check=True,
        ).stdout
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpg).decode(),
        }})
    content.append({"type": "text", "text": MC_PROMPT.format(
        question=question, options=_format_options(options))})

    client = anthropic.Anthropic()
    meter = CostMeter(model=model)
    response = client.messages.create(
        model=model, max_tokens=8192,
        messages=[{"role": "user", "content": content}],
    )
    meter.add(response.usage)
    text = "".join(b.text for b in response.content if b.type == "text")
    return MethodResult(
        answer_text=text,
        predicted_letter=_extract_letter(text, len(options)),
        usd=meter.usd,
        detail={**meter.summary(), "frames_sent": n},
    )


# ---------------------------------------------------------------- gemini native

# USD per million tokens (input, output). Pro rate verified Oct 2026
# (<=200k prompt); flash rate is an estimate: verify before publishing.
GEMINI_PRICES = {
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3-flash-preview": (0.50, 3.00),   # TODO verify
    "gemini-3.1-flash-lite": (0.10, 0.40),    # TODO verify
}


def method_gemini(video_path: Path, question: str, options: list[str],
                  model: str = "gemini-3-flash-preview") -> MethodResult:
    """Native video ingestion baseline. Key from GEMINI_API_KEY, GOOGLE_API_KEY,
    or AISTUDIO_API_KEY. Free-tier aware: waits for upload processing and backs
    off on rate limits."""
    import os
    import time

    from google import genai
    from google.genai import errors as genai_errors

    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
               or os.environ.get("AISTUDIO_API_KEY"))
    client = genai.Client(api_key=api_key)

    uploaded = client.files.upload(file=str(video_path))
    # Video uploads are processed asynchronously; generate_content rejects
    # files that are not ACTIVE yet.
    for _ in range(60):
        if uploaded.state and uploaded.state.name == "ACTIVE":
            break
        if uploaded.state and uploaded.state.name == "FAILED":
            raise RuntimeError(f"gemini upload failed for {video_path.name}")
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)

    prompt = MC_PROMPT.format(question=question, options=_format_options(options))
    response = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=model, contents=[uploaded, prompt])
            break
        except genai_errors.APIError as e:
            # 429s on the free tier consume daily-quota slots per ATTEMPT —
            # never retry them here; the outer drip loop is the retry.
            if e.code == 503 and attempt < 4:
                wait = 20 * (attempt + 1)
                print(f"  gemini {e.code}, backing off {wait}s")
                time.sleep(wait)
            else:
                raise
    assert response is not None
    usage = response.usage_metadata
    inp_rate, out_rate = GEMINI_PRICES.get(model, (0.0, 0.0))
    usd = (usage.prompt_token_count * inp_rate
           + (usage.candidates_token_count or 0) * out_rate) / 1_000_000
    return MethodResult(
        answer_text=response.text or "",
        predicted_letter=_extract_letter(response.text or "", len(options)),
        usd=usd,
        detail={
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
        },
    )


def method_gemini_lite(video_path: Path, question: str, options: list[str]) -> MethodResult:
    """Native ingestion on the free 500-requests/day lite tier."""
    return method_gemini(video_path, question, options, model="gemini-3.1-flash-lite")


METHODS = {
    "videoctx": method_videoctx,
    "naive": method_naive,
    "gemini": method_gemini,
    "gemini-lite": method_gemini_lite,
}
