# videoctx

**Index a video once. Query it cheaply, forever, with any LLM.**

Most LLMs can't ingest video, and the ones that can charge for every second of
footage on every question. `videoctx` turns any video into an open, portable
**Video Context** index (see [SPEC.md](SPEC.md)) using only local, model-free
tools: ffmpeg for keyframes, PySceneDetect for shot boundaries, faster-whisper
for the transcript. An agent then answers most questions from the cheap text
layer and pulls individual frames only when it actually needs eyes.

```
videoctx index lecture.mp4          # one-time, local, free
videoctx show lecture.vctx          # the text view the model reads
videoctx ask lecture.vctx "What diagram is shown when she explains backprop?"
```

## Why

- **Cost.** Native video ingestion tokenizes every frame on every request.
  Indexing once and fetching 4 frames instead of 3,600 is a 10-50x token
  saving on typical long-video questions.
- **Portability.** The index is plain JSON + JPEG. It works with Claude, GPT,
  Gemini, or a local VLM, and it works offline.
- **Agents.** The query loop is a single tool (`get_frames`) away from any
  agent framework. MCP server: coming next.

## Status

Phase 0: pipeline + eval harness. The claim under test: **what fraction of
native-video accuracy does this retain, at what fraction of the cost?**
Benchmarked on long-video QA (Video-MME) against native ingestion and naive
frame sampling. Numbers will be published here, including where this approach
loses.

## Install

```
uv venv && uv pip install -e .
```

Requires `ffmpeg` on PATH. Querying uses the Anthropic API (`ANTHROPIC_API_KEY`
or an `ant auth login` profile).

## Eval

```
python eval/run.py questions.jsonl --methods videoctx,naive
```

See `eval/run.py` for the question file format.

## License

Apache-2.0
