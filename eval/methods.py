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
    "Answer with the single letter of the best option, then one sentence of justification."
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
    valid = {chr(65 + i) for i in range(n_options)}
    for ch in text.strip().upper():
        if ch in valid:
            return ch
    return None


# ---------------------------------------------------------------- videoctx

def method_videoctx(video_path: Path, question: str, options: list[str],
                    model: str = "claude-opus-4-8", index_cache: Path | None = None) -> MethodResult:
    """Index once (cached on disk), then query via the frames-on-demand loop."""
    from videoctx.indexer import build_index, load_index
    from videoctx.query import ask

    index_dir = (index_cache or video_path.parent) / f"{video_path.stem}.vctx"
    if not (index_dir / "index.json").exists():
        build_index(video_path, out_dir=index_dir, progress=lambda *_: None)

    ctx = load_index(index_dir)
    prompt = MC_PROMPT.format(question=question, options=_format_options(options))
    answer, meter, frames = ask(index_dir, ctx, prompt, model=model)
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

def method_gemini(video_path: Path, question: str, options: list[str],
                  model: str = "gemini-2.5-pro") -> MethodResult:
    """Native video ingestion baseline. Requires GEMINI_API_KEY."""
    from google import genai  # requires `pip install google-genai` and an API key

    client = genai.Client()
    uploaded = client.files.upload(file=str(video_path))
    prompt = MC_PROMPT.format(question=question, options=_format_options(options))
    response = client.models.generate_content(model=model, contents=[uploaded, prompt])
    usage = response.usage_metadata
    # Gemini 2.5 Pro pricing (<=200k prompt): $1.25/M input, $10/M output.
    usd = (usage.prompt_token_count * 1.25 + (usage.candidates_token_count or 0) * 10) / 1_000_000
    return MethodResult(
        answer_text=response.text or "",
        predicted_letter=_extract_letter(response.text or "", len(options)),
        usd=usd,
        detail={
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
        },
    )


METHODS = {
    "videoctx": method_videoctx,
    "naive": method_naive,
    "gemini": method_gemini,
}
