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

Numbers below are single-touch runs of the frozen configs (v0.2: captions +
OCR + evidence rules, Opus 4.8 loop; v0.3, 2026-10-21: adds text-evidence
rules, verbatim transcript quotes and a final letter-vs-evidence check)
under full cost accounting. Where we measured the same configuration more
than once, we report the observed range: on n=150, single runs wobble by
2-3 questions; on n=30, by 2-3 questions (= 7-10 points). Treat
sub-3-point deltas on this benchmark as noise.

Subset composition caveat: videos were taken in Video-MME dataset order,
which clusters by domain. The medium and long cells are entirely
Knowledge-domain videos; the short cells are 76-88% Knowledge (remainder
Film & Television). Read these as Knowledge-dominated subsets of Video-MME,
not the full benchmark; future splits will be domain-stratified.

**Short videos, held-out test half (n=150):**

| Method | Accuracy | $/question |
|---|---|---|
| **videoctx v0.4** | **84.7%** | $0.025 |
| videoctx v0.3 | 82.0% | $0.025 |
| videoctx v0.2 | 80.7% (equivalent configs ranged 80.7-85.3%) | $0.025 |
| Uniform frame sampling | 74.0% | $0.0172 |
| Gemini Flash-Lite native (pilot 30) | 70.0% | $0.0009 |
| Gemini Flash native (pilot 30) | 53.3% | $0.0047 |

**Medium videos, 4-15 min (n=30, wide error bars):**

| Method | Accuracy | $/question |
|---|---|---|
| Uniform frame sampling | 80.0% | $0.081 |
| videoctx v0.7 (contact sheet 60@512 in cached prefix) | 70.0% | $0.092 first-touch, ~$0.02 follow-up |
| videoctx v0.2 through v0.6 | 66.7% (near-identical failure set) | $0.051-0.062 |
| Gemini Flash-Lite native | 70.0% | $0.0034 |

Medium is our weakest cell and remains the named target: naive leads on
raw accuracy. Ten questions failed identically across five configs
(prompt rules, a text-view bug fix, dense captions, a 24-thumb contact
sheet); the sixth config, a 60-thumbnail 512px contact sheet baked into
the cached prefix, recovered one. The receipts tell a consistent story:
these videos are fast-cut and fully captioned, and the remaining gap is
visual detail bandwidth at question time, naive's one structural
advantage. We publish the ladder rather than only its best rung.

Correction (2026-10-21): a text-view bug had been dropping the header
line (duration, shot count) on every OCR-enriched index since OCR
shipped; all v0.2/v0.3 numbers were measured under that bug. v0.4 is the
corrected configuration (short +2.7 points, long back to 80.0%, medium
unchanged).

**Long videos, 30-60 minutes (n=30):**

| Method | Accuracy | $/question | Follow-up $/question |
|---|---|---|---|
| **videoctx v0.5 (dense captions)** | **83.3%** | **$0.109** cold (measured $0.047, see note) | **$0.021** |
| videoctx v0.4 | 80.0% | $0.099 | $0.019 |
| videoctx v0.2 / v0.3 | 80.0% / 76.7% (n=30 noise band) | $0.098 | $0.019 |

Cost correction (2026-10-22): the v0.5 long run started 57 minutes after
the v0.4 run, and for the 7 of 10 videos whose text views were unchanged
by dense captioning, it silently inherited v0.4's still-live 1-hour
prompt caches, flattering first-question costs. $0.109/q is the
cold-equivalent (first question $0.285, follow-ups $0.021 as measured).
Benchmark runs are now cache-salted per config so this cannot recur.
| Uniform frame sampling (60-frame cap) | 70.0% | $0.133 | $0.133 |
| Gemini Flash-Lite native | 76.7% | $0.022 | $0.022 (5-9 min latency/q) |
| Native ingestion (published Pro rates, est.) | pending | ~$0.50 | ~$0.50 |

One-time index cost: ~$0.10-0.25 per hour of video (captions + local OCR +
local embeddings), pennies otherwise.

*Cost correction history: early short-video figures ($0.0067/q) undercounted
prompt-cache write premiums; the metering was fixed 2026-10-18 and all
figures above are fully loaded. Details in the paper's Section 7.3.*

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
t1, detail)`, `transcript_search`, `find_moments` (semantic visual search,
needs `videoctx embed` once per index), `index_video`. The server's instructions
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
