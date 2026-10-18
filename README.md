# videoctx

<!-- mcp-name: io.github.dunfred/videoctx -->

**Index a video once. Query it cheaply, forever, with any LLM.**

`videoctx` ("video context") turns any video into an open, portable **Video
Context** index (`.vctx`, see [SPEC.md](https://github.com/dunfred/videoctx/blob/main/SPEC.md)) using only local, model-free
tools: ffmpeg keyframes, PySceneDetect shot boundaries, faster-whisper
transcripts, and optional one-line visual captions from any cheap VLM. An
agent answers most questions from the index's text layer for fractions of a
cent, and pulls individual frames on demand only when a question needs pixels.

```
videoctx index lecture.mp4          # one-time, local, free
videoctx show  lecture.vctx         # the text view the model reads
videoctx ask   lecture.vctx "What diagram is shown when she explains backprop?"
```

## What people use it for

Ask questions about videos the way you'd ask about a document:

- **Lectures and courses:** "Summarize this 50-minute lecture." "Where does
  she explain backpropagation, and what's on the slide at that moment?"
- **Meetings and interviews:** "List the action items with timestamps."
  "Find the moment the client pushed back on pricing."
- **Screen recordings:** "At what time does the error appear, and what does
  the dialog say?"
- **Content creation:** "Find the three strongest moments for short clips."
  "Generate chapter markers."
- **Video archives:** index a folder of talks once, then every future
  question across all of them costs pennies and answers in seconds.
- **AI agents:** give any MCP-capable agent (Claude Code, Cursor, ...) the
  ability to read video at all.

**Who this is for:** developers and anyone comfortable running a few
terminal commands. You'll need Python, ffmpeg, and an Anthropic API key for
question-answering (indexing itself is free and offline). If that's not you
yet, star/watch the repo: a no-install version is on the roadmap.

## Why

Native video ingestion re-tokenizes the whole video on every question: an
hour of footage is ~250-350k tokens, every single time. Proprietary video
platforms index into closed representations behind per-minute pricing.
videoctx is the third option: an open on-disk format any model can consume,
produced by free tools, where the marginal cost of a follow-up question on an
hour-long video is about **$0.02**.

## Benchmark results (Video-MME, Oct 2026)

Full protocol and honest cost accounting (including prompt-cache write
premiums) in [paper/draft.md](https://github.com/dunfred/videoctx/blob/main/paper/draft.md). Query model: Claude Opus 4.8.

**Short videos, held-out test half (n=150):**

| Method | Accuracy | $/question |
|---|---|---|
| **videoctx** | **83.3%** | **$0.0067** |
| Uniform frame sampling | 74.0% | $0.0172 |

**Identical 30-question subset, all methods:**

| Method | Accuracy | Latency/question |
|---|---|---|
| **videoctx** | **76.7%** | 2-20s |
| Uniform frame sampling | 60.0% | ~5s |
| Gemini Flash native ingestion | 53.3% | ~40s |

**Long videos, 30-60 minutes (n=30):**

| Method | Accuracy | $/question | Follow-up $/question |
|---|---|---|---|
| **videoctx** | **80.0%** | **$0.096** | **$0.024** |
| Uniform frame sampling (60-frame cap) | 70.0% | $0.133 | $0.133 |
| Native ingestion (published Pro rates, est.) | pending | ~$0.50 | ~$0.50 |

One-time index cost: ~$0.10 per hour of video (captions), pennies otherwise.

## Install

```
git clone <repo> && cd videoctx
uv venv && uv pip install -e .
```

Requires `ffmpeg` on PATH. `videoctx index` and the MCP server's core tools
run fully locally with no API key. `videoctx ask` uses the Anthropic API
(`ANTHROPIC_API_KEY` or an `ant auth login` profile).

## MCP server: let any agent read videos

```
claude mcp add videoctx -- $(pwd)/.venv/bin/videoctx-mcp
```

The calling agent's own model reads the text layer and frames directly; the
server makes no model calls and needs no key. Tools: `video_overview` (cheap
text layer), `survey_video` (one thumbnail per shot), `get_frames(path, t0,
t1, detail)`, `transcript_search`, `index_video`. The server's instructions
carry the evidence rule: visual answers must cite fetched frames, never
general knowledge.

## Reproduce the benchmark

```
uv pip install -e ".[eval]"
python eval/prepare_videomme.py --duration short --videos 100 --out eval/data
python eval/split_dev_test.py eval/data/videomme_short.jsonl
python eval/run.py eval/data/videomme_short_test.jsonl --methods videoctx,naive
```

`eval/methods.py` documents each baseline. Cost accounting counts cache
writes at their real premium (1.25x / 2x by TTL); see the paper's Section 7.3
for the accounting traps this avoids.

## Format

The `.vctx` directory is plain JSON + JPEG: `manifest.json` (provenance,
parameters, source hash), `index.json` (shots, keyframes, transcript,
captions), `frames/`. Consumers must ignore unknown keys; enrichment layers
(OCR, embeddings) are reserved in [SPEC.md](https://github.com/dunfred/videoctx/blob/main/SPEC.md) v0.1.

## Status and roadmap

Working and benchmarked: indexing, query loop, MCP server, eval harness.
Next: medium-duration subset, Pro-tier native baseline, cross-model
portability runs (same indexes via non-Claude models), OCR and per-shot
embedding layers, spec v1.0.

## License

Apache-2.0. Copyright 2026 Fred Dunyo.
