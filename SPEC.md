# Video Context Format — Spec v0.1 (draft)

Status: draft. Everything here may change until v1.0.

## Purpose

A portable, model-free representation of a video, designed for LLM agents under
token budgets. A video is indexed **once** (locally, with open tools) into a
Video Context directory. Any model can then answer questions about the video by
reading the cheap text layer first and pulling individual frames only when
needed.

Design principles:

1. **Model-free.** Producing an index requires no proprietary model. ffmpeg,
   scene detection, and open speech-to-text are sufficient. Optional layers
   (captions, embeddings) may use any model the producer chooses, and the index
   records which one.
2. **Text-first, frames-on-demand.** The text layer (structure + transcript +
   captions) should answer most questions alone. Frames are addressable by
   timestamp so a consumer fetches only what a question requires.
3. **Portable.** An index is a plain directory of JSON and JPEG. No database,
   no server, no SDK required to consume it.
4. **Honest provenance.** The index records tool versions, parameters, and any
   model calls used to produce it, so results are reproducible and costs
   attributable.

## Directory layout

```
<name>.vctx/
  manifest.json      # format version, provenance, indexing parameters
  index.json         # the context document: shots, keyframes, transcript, captions
  frames/            # extracted keyframes, JPEG
    shot0003_t142.50.jpg
```

## manifest.json

| field | type | notes |
|---|---|---|
| `format` | string | `"vctx"` |
| `format_version` | string | `"0.1"` |
| `created_at` | string | ISO 8601 |
| `producer` | object | `{name, version}` of the indexing tool |
| `params` | object | indexing parameters (detector, thresholds, whisper model, frame max edge, etc.) |
| `source` | object | `{filename, sha256, size_bytes}` of the source video |

## index.json

### `video`

`{duration_s, width, height, fps, has_audio}`

### `shots[]`

Visually coherent segments, ordered, non-overlapping, covering the full
duration.

| field | type | notes |
|---|---|---|
| `id` | int | 0-based |
| `start_s`, `end_s` | float | seconds |
| `keyframes[]` | array | `{t_s, file, width, height}` — `file` relative to index root |
| `caption` | string? | optional, one-line visual description |
| `caption_model` | string? | model that produced `caption`, if any |

### `transcript`

`{language, model, segments: [{start_s, end_s, text}]}` — empty `segments` if
the video has no speech or no audio.

### Reserved for later versions

`embeddings` (per-shot visual/text vectors), `entities`, `ocr`. Consumers must
ignore unknown keys.

## Consumer contract

A consumer (agent harness, MCP server, eval script) should expose three
operations over an index:

1. `text_view()` → compact plain-text rendering of shots + transcript, suitable
   for direct inclusion in a prompt. This is the cheap path.
2. `get_frames(start_s, end_s, max_frames)` → the keyframes within a time
   range, for when the question needs eyes.
3. `frame_at(t_s)` → nearest keyframe to a timestamp.

The intended agent loop: answer from `text_view()` if possible; otherwise call
`get_frames` on the narrowest relevant span; escalate density only if still
uncertain.
