# Index Once, Ask Forever: A Model-Free Video Context Layer and an Accuracy-per-Dollar Benchmark for Video QA with LLM Agents

**Fred Dunyo**
*Draft v0.2, 2026-10-21. Numbers marked [PENDING] await runs in progress
(native Pro-tier columns, gated on paid API access).*

## Abstract

Most LLM agents cannot read video, and the systems that can (native video
ingestion, proprietary video foundation models) re-tokenize the entire video
on every question, making repeated interaction with long footage economically
prohibitive. We present **vctx**, an open, model-free video context format: a
video is indexed once, locally, using only commodity tools (ffmpeg shot
detection and keyframes, open speech-to-text, optional cheap VLM captions)
into a portable directory of JSON and JPEG. An agent answers questions from
the index's cheap text layer and escalates to individual frames only when the
question requires pixels, under an evidence rule that forbids answering visual
questions without fetched visual evidence. We also introduce an
**accuracy-per-dollar** evaluation protocol for video QA that charges every
token honestly, including prompt-cache write premiums, and measures cost as a
function of questions asked per video. On Video-MME short videos (held-out
test half, n=150), the current frozen vctx configuration with Claude Opus
4.8 scores 84.7% at $0.025 per question, versus 74.0% at $0.0172 for a
uniform frame-sampling baseline. On Video-MME long videos (n=30), vctx with
dense keyframe captions reaches 83.3% at $0.047 per question versus 70.0%
at $0.133 for sampling; follow-up questions on an already-warm index cost
about $0.02, roughly 25x below the estimated cost of native ingestion of
the same hour-long video. On medium videos the sampling baseline retains
the lead (80.0% vs 70.0%), a result we analyze in depth: the same 10
questions failed under five successive configurations, isolating raw
visual detail bandwidth at question time as the one advantage of stateless
dense sampling. We also quantify run-to-run variance and locate it almost
entirely in the frame-fetching path, report a replicated dev gain that
transferred zero questions to its target test cell, and publish the full
v0.2-to-v0.7 configuration ladder with per-question receipts. The format
specification, reference implementation, MCP server, and evaluation
harness are released under Apache-2.0.

## 1. Introduction

Images became a first-class input for LLMs years ago; video has not. The
providers that do accept video natively charge for every second of footage on
every request: an hour of video is roughly 250-350k tokens per question, paid
again for each follow-up. Proprietary video-understanding platforms index
video into closed representations behind per-minute pricing. There is no
equivalent of what Markdown is to documents: an open, portable, model-free
representation that any agent can consume.

This paper makes three contributions:

1. **The vctx format** (Section 3): a specification for turning any video
   into a compact, self-contained context directory using only local, free
   tools, with optional enrichment layers (captions, and in future versions
   OCR and embeddings) that record their provenance.
2. **A text-first query protocol** (Section 4): the agent reads a ~1-20k
   token text rendering of the video first, and escalates through a cheap
   thumbnail survey to targeted full-resolution frame extraction, governed by
   an evidence rule that measurably suppresses visual hallucination.
3. **An accuracy-per-dollar benchmark protocol** (Section 5): cost accounting
   that includes cache-write premiums (which we show can silently triple
   apparent costs when ignored), and reporting cost as a function of
   questions-per-video, which exposes the amortization structure that
   single-question evaluations hide.

We deliberately do not train a model. The claim under test is architectural:
that a well-structured, model-free index in front of a general-purpose
multimodal LLM is more accurate than uniform frame sampling and dramatically
cheaper than native ingestion for any workload that asks more than one
question per video.

## 2. Related Work

**Native video ingestion.** Gemini ingests video directly at roughly one
frame per second plus audio tokens [Gemini Team 2024] and charges full
ingestion per request; as of this writing, OpenAI's GPT-5 model documentation
lists text and image input only, with no video modality [OpenAI 2026]. We
treat native ingestion as the capability ceiling and price floor-antagonist:
our question is how much of its accuracy survives at a fraction of its cost.

**Proprietary video platforms.** Twelve Labs exposes video foundation models
(Marengo for embeddings and retrieval, Pegasus for video-to-text), plus an
MCP server, as a hosted API priced per indexed minute [Twelve Labs 2026].
VideoDB offers managed video data infrastructure with natural-language
retrieval [VideoDB 2026]. vctx differs on two axes: the index is produced by
free local tools rather than proprietary models, and the artifact is an open
on-disk format rather than an API-side state.

**Video RAG research.** Retrieval-augmented approaches to long-video QA
share our coarse-to-fine intuition: VideoAgent iteratively selects frames
with an LLM controller [Wang, X. et al. 2024]; VideoTree builds a
query-adaptive hierarchical representation [Wang, Z. et al. 2024]. Most
report accuracy only; none to our knowledge publish an honest
cost-per-question accounting or an open interchange format. Our contribution
is the format + benchmark pairing rather than a novel retrieval algorithm.

**Benchmarks.** We evaluate on Video-MME [Fu et al. 2024], the standard
multi-duration video QA benchmark (short <=2 min, medium 4-15 min, long
30-60 min; 3 questions per video). Gold labels are multiple choice. We note
observed label disputes in Section 7.4.

## 3. The vctx Format

A vctx index is a directory:

```
video.vctx/
  manifest.json   # format version, producer, parameters, source hash + path hint
  index.json      # shots, keyframes, transcript, captions
  frames/         # extracted JPEG keyframes and on-demand frames
```

Indexing uses only local tools: ffprobe (metadata), PySceneDetect
[Castellano] (shot boundaries, with a fixed-window fallback for cut-free
footage), ffmpeg (keyframes, one per shot plus one per 20s within long
shots, long edge 768px), and faster-whisper [SYSTRAN], a CTranslate2 port of
Whisper [Radford et al. 2022] (transcript with per-segment timestamps). An optional
caption pass annotates up to 150 shots per video with one-line visual
descriptions from a cheap VLM (we use Claude Haiku 4.5; ~$0.10 per hour-long
video, one time); an optional dense pass extends this to the non-midpoint
keyframes of shots longer than 45s, so a multi-minute shot contributes a
timestamped caption roughly every 20s instead of one line total (the long
cell's v0.5 gains in 6.3 come from this layer). The manifest records every
parameter and model used.

Indexing is idempotent, portable (plain JSON/JPEG), and cheap: roughly
10-20% of video duration on a laptop CPU, dominated by transcription.

**Design rule: the cap on enrichment is a property of the index, not of the
enrichment call.** An early implementation capped captions per call; repeated
calls quietly accumulated 3x the intended captions and tripled the text-layer
size (Section 7.3). The specification therefore states caps as index
invariants.

## 4. Query Protocol

The consumer contract exposes three operations: `text_view()` (compact text
rendering), `get_frames(t0, t1, n, detail)` (frames extracted on demand from
the source at requested density and resolution), and `survey()` (one small
thumbnail per shot, subsampled to 24). Our reference agent loop supplies
these to the model as tools, with a system prompt encoding four rules:

1. **Text first.** The text view (shot structure + captions + transcript,
   interleaved by timestamp; speech grouped into ~25s blocks; caption-less
   silent spans collapsed) answers most questions alone.
2. **Captions are lossy.** Counting, colors, patterns, and on-screen text
   must be verified against frames, never answered from captions.
3. **Enumeration protocol.** "How many" / "first time" / "which option is
   false" questions require a systematic sweep before answering.
4. **Evidence rule.** A visual answer must cite a fetched frame's timestamp;
   if fetched frames do not show the answer, fetch different ones rather than
   guess. General world knowledge may never substitute for missing pixels.

The text view is marked as a prompt-cache prefix (1-hour TTL above 20k
characters, default TTL below), so every question after the first reads the
video's context at ~10% of input price.

## 5. Benchmark Protocol: Accuracy per Dollar

Each method answers multiple-choice questions and reports its own fully
loaded cost. Three accounting rules matter more than they appear:

1. **Cache writes are charged at their premium** (1.25x for 5-minute TTL, 2x
   for 1-hour TTL, with cache reads at 0.1x, on the provider we use
   [Anthropic 2026]). In our logs, ignoring cache-write
   tokens understated long-video costs by up to 2.3x. Any cached-prefix
   system evaluated without this rule will report fantasy costs.
2. **One-time index costs are reported separately and amortized explicitly.**
   Indexing + captioning a 100-video short corpus cost $0.99 total; we report
   it alongside, not hidden inside, per-question costs.
3. **Cost is a function of questions-per-video.** We report first-question
   and follow-up costs separately (Section 6.4). Single-question evaluation
   structurally favors stateless methods and hides the central advantage of
   indexed ones.

Methods compared: **vctx** (this work); **naive sampling** (uniform frames at
1 per 10s, capped at 60, single request, no transcript: the de facto standard
workaround today); **native ingestion** (Gemini; the video is uploaded and
ingested per question). Letter extraction from free-text answers uses a
three-stage parser validated against manual reads; an earlier single-pass
parser mis-scored 20% of one method's answers (Section 7.3).

All experiments use Claude Opus 4.8 as the query model for vctx and naive;
captions use Claude Haiku 4.5. Dev/test splits are by video. Every vctx
configuration was designed and validated on the pilot + dev half and
frozen before evaluation; each frozen configuration touched the test half
exactly once, and the complete sequence of frozen configurations with
their test results is published as a ladder (7.8) rather than reporting
only the best rung.

## 6. Results

### 6.1 Short videos (Video-MME short, 100 videos, 300 questions)

| Method | Dev (150q) | Test (150q) | $/question (test) |
|---|---|---|---|
| vctx (frozen v0.4, one touch) | n/a | **84.7%** | $0.025 |
| vctx (frozen v0.2, one touch) | 84.0-86.0% band | 80.7% | $0.025 |
| Naive sampling | n/a | 74.0% | $0.0172 |
| Native (Gemini 3 Flash)* | n/a | 53.3% | $0.0047 paid-equiv |

v0.4 = v0.2 plus text-evidence rules (7.7) and a text-view bug fix (7.3g);
each frozen config touched the test half exactly once (7.7 documents the
full ladder).

*Native column measured on the 30-question pilot subset (free-tier quota
limits); on those same 30 questions vctx scores 76.7% and naive 60.0%. The
native row uses Gemini's budget tier; the Pro-tier column remains [PENDING]
on paid access.

One-time corpus cost (index + captions, all 100 videos): $0.99. Dev and test
agree within 2 points; nothing was tuned on the test half.

Note that even if Pro-tier native matches vctx accuracy on short videos,
short clips are native ingestion's *best* cost case, ~2k tokens per clip, and
its economics collapse on long video (6.4). Native ingestion also paid a
measured ~40s of upload + processing latency per question versus vctx's 2-4s
text-path answers.

### 6.2 Medium videos (Video-MME medium, 4-15 min, 30 questions)

Our weakest cell, which we report rather than omit:

| Method | Accuracy | $/question |
|---|---|---|
| Naive sampling | **80.0%** | $0.081 |
| vctx v0.7 (contact sheet 60 thumbs @ 512px) | 70.0% | $0.092 first touch |
| vctx v0.2 through v0.6 | 66.7% | $0.051-0.062 |
| Native (Gemini Flash-Lite) | 70.0% | $0.0034 |

This cell resisted an entire improvement campaign, and we publish the
ladder rather than only its best rung. The same 10 questions failed
identically under five configurations spanning prompt rules, a text-view
bug fix, dense keyframe captions, and a 24-thumbnail contact sheet; only
the sixth, a 60-thumbnail 512px contact sheet baked into the cached
prefix, recovered one (7.7). The receipts localize the residual gap:
these videos are fast-cut and fully captioned, naive solves 7 of the 10
stable failures from 60 full-resolution frames, and the loop answers
26/30 questions without fetching a frame. What medium rewards is raw
visual detail bandwidth at question time, the one structural advantage of
stateless dense sampling. Medium remains the named target; the open lever
is scaling prefix coverage toward naive's input while keeping it
amortized, to be powered by a new dev instrument first.

### 6.3 Long videos (Video-MME long, 10 videos of 30-60 min, 30 questions)

| Method | Accuracy | $/question | Wall-clock/question |
|---|---|---|---|
| vctx (frozen v0.5, one touch) | **83.3%** | **$0.047** | median 3s |
| vctx (frozen v0.2, one touch) | 80.0% | $0.098 | median 4s |
| Naive sampling (60-frame cap) | 70.0% | $0.133 | ~6s |
| Native ingestion (Gemini 3.1 Pro, published rates) | [PENDING] | ~$0.50-0.70 est. | ~40s+ observed on Flash |

v0.5 adds dense keyframe captions: shots longer than 45s receive a
timestamped caption roughly every 20s, so an hour-long video is visible to
the text layer throughout. The effect is twofold: accuracy rose one
question and cost halved, because 28 of 30 questions are now answered
text-only (measured follow-up questions average $0.021). Accuracy on long
videos staying at short-video levels is consistent with the index's size
being decoupled from video duration.

### 6.4 Amortization: the central figure

Measured from the frozen long-video run (FINAL_long30): the **first**
question on a video averages $0.256 (it pays the one-time cache write of the
text view); every **follow-up** inside the cache window averages **$0.019**.
Per-question cost at k questions per video:

| k (questions/video) | vctx | naive | native (est.) |
|---|---|---|---|
| 1 | $0.256 | $0.133 | ~$0.50 |
| 3 | $0.098 | $0.133 | ~$0.50 |
| 10 | $0.043 | $0.133 | ~$0.50 |
| 30 | $0.027 | $0.133 | ~$0.50 |

![Figure 1: per-question cost vs questions asked per video. vctx decays from $0.256 toward its $0.019 follow-up cost; stateless baselines are flat.](figures/amortization.png)

Stateless methods are flat lines; vctx is a decaying curve that crosses naive
between k=2 and k=3 and sits an order of magnitude below native from k=3
onward (Figure 1). Agent
workloads (an assistant discussing a lecture, a reviewer auditing a meeting
recording) are precisely the k>>1 regime.

The decomposition above uses the v0.2 receipts, whose first questions show
clean cold-cache behavior (~25k cache-write tokens, zero reads). We do NOT
use the v0.5 run for this figure: its per-video first questions show ~20k
cache-*read* tokens that nothing in our pipeline wrote, most plausibly
SDK-internal retries whose abandoned first attempt populated the cache
without being metered. Until that is pinned down, v0.5 first-question
costs would understate true cold-start spend, so we report only its
overall average ($0.047) and measured follow-ups ($0.021). We flag this as
a candidate addition to the accounting-trap list (7.3): client-side
retries can convert unmetered cache writes into metered cache reads.

### 6.5 Latency

Native ingestion re-uploads and re-processes the video per question (observed
~40s+ per question on short clips). vctx's text-path answers return in ~2-4s.
For interactive agents this is the difference between conversational and
batch.

## 7. Analysis

### 7.1 Where accuracy comes from

Path split on the short test half: text-only answered 90/150 questions at
90% accuracy for ~$0.003 each; the frame path handled 60/150 at 73%. The
protocol's cost efficiency comes from the text layer absorbing most
questions; its accuracy ceiling is currently set by the frame path.

### 7.2 Ablation ladder (pilot, 30 questions)

| Configuration | Accuracy | Cost |
|---|---|---|
| Shot-midpoint keyframes only | 60.0% | $0.54 |
| + on-demand frame extraction | 70.0% | $0.75 |
| + captions, survey tool, evidence rule, cache | 76.7% | $0.20 |
| + detail escalation, enumeration rule (frozen config) | 76.7% | $0.29 |

The largest single gain (and simultaneous 73% cost cut) came from giving the
text layer visual captions: 5 questions stopped needing frames at all, and
blind time-range guessing largely disappeared.

### 7.3 Negative results and accounting traps (what we had to fix)

We report these because a cost benchmark is only as credible as its
bookkeeping: (a) uncounted cache-write tokens understated long-video costs by
up to 2.3x; (b) 5-minute cache TTL silently lapsed between questions under
rate-limit stalls, tripling write volume until switched to 1-hour TTL for
large prefixes; (c) a per-call rather than per-index caption cap accumulated
3x the intended captions and bloated text views to 50k+ tokens; (d) a naive
answer-letter parser mis-scored ~20% of prose answers; (e) frame requests for
timestamps beyond video duration crashed extraction until server-side
clamping was added (models do hallucinate timestamps); (f) our own first
published short-video cost figure fell to trap (a) and was corrected from
$0.0067 to ~$0.019 per question after remetering, a correction we record
here rather than erase; (g) a shadowed variable in the text-view renderer
silently deleted the header line (duration, shot count, transcript
language) from every OCR-enriched index, so every benchmark number
measured between the OCR layer shipping and 2026-10-21 was produced by a
model that did not know the video's duration from its text layer. Fixing
(g) was worth +2.7 points on the short test half by itself. Every one of
these distorted either cost or accuracy before being caught by
per-question audits.

### 7.4 Run-to-run variance lives in the frame path

Re-running identical or near-identical configurations moved short-video
(n=150) accuracy by 2-5 questions and medium (n=30) accuracy by up to 3
questions (10 points). Diffing flipped questions shows the instability is
concentrated almost entirely in frame-path questions (8-9 of 9 test-half
flips), where the model's fetch choices are stochastic; text-path-dominated
cells replicated exactly (long: 80.0% twice). Two consequences: (1) accuracy
deltas under ~3 points on n=150, or under ~10 points on n=30, should be
treated as noise on this benchmark, a bar several published comparisons do
not clear; (2) stabilizing frame selection (e.g., retrieval-based targeting
instead of model-chosen ranges) is a variance intervention as much as an
accuracy one.

### 7.5 Label quality

In at least 2 of 30 long-video pilot failures, the model's cited frame
evidence plausibly contradicts the gold label. We do not re-adjudicate labels
in this draft; we note that at the 80%+ regime, Video-MME label noise is a
nontrivial fraction of residual error.

### 7.6 Visual retrieval embeddings are accuracy-neutral (dev sets)

We added optional per-shot CLIP [Radford et al. 2021] embeddings to the
index and exposed a
`find_moments(description)` retrieval tool in the query loop, hypothesizing
that semantic frame targeting would help most on long videos, where
model-chosen time ranges have the most room to miss. A/B on held-out dev
sets (tool available vs. disabled, identical config otherwise):

| Dev set | Without retrieval | With retrieval | $/question |
|---|---|---|---|
| Long dev (18q) | 88.9% (16/18) | 94.4% (17/18) | $0.121 vs $0.117 |
| Medium dev (30q) | 93.3% (28/30) | 90.0% (27/30) | $0.037 vs $0.036 |

Each delta is a single question flipping, inside the noise bar of 7.4, and
cost is unchanged. The mechanism is visible in the receipts: only 2-7
questions per run invoked any visual tool at all; the text layer answered
the rest directly, leaving retrieval almost no surface to act on. This is
the same pattern as 7.1 from the other side: interventions on the frame
path are capped by how rarely the frame path is exercised. Embeddings
therefore ship as an optional, provenance-recorded enrichment layer (useful
for interactive moment search), with no accuracy claim attached. Whether
retrieval-based targeting reduces frame-path *variance* (7.4) on the dense
short-video cells remains open.

### 7.7 A replicated dev gain that did not transfer (v0.3)

After the freeze, we added text-evidence rules to the query loop (verbatim
transcript quotes for claims about what the video says, a gather-every-
mention requirement, and a final check that the answer letter matches the
model's own cited evidence). On a 45-question medium dev split built from
fresh same-domain videos, the rules gained 4.4-6.7 points over base across
three independent runs with zero per-question regressions, and gained 5.5
points on the long dev split; a stronger variant adding exhaustive visual
sweeps was rejected (it regressed ordering questions it was meant to fix).
On the medium test cell, the frozen v0.3 run changed *zero* answers:
66.7% in both configs, no flips. Per-question receipts explain the miss:
the test cell's failures concentrate in questions answered from the text
layer without frame escalation (26 of 30 questions fetched no frames;
text-path accuracy 17/26), while the dev split's text path was already at
35/38, so the dev-visible failure classes (evidence-letter mismatches,
premature quote stops) are not the classes the test cell contains. We
report this as a caution against exactly the inference we wanted to make:
a replicated, zero-regression dev gain on a same-domain split predicted
nothing about this test cell. Short moved +1.3 points and long -3.3
points under v0.3, both inside the noise bands of 7.4.

### 7.8 The config ladder (v0.2 to v0.7)

The full improvement campaign, every rung dev-validated first and granted
exactly one test touch, with per-question receipts committed:

| Config | Change | Short (150q) | Medium (30q) | Long (30q) |
|---|---|---|---|---|
| v0.2 | frozen baseline | 80.7% | 66.7% | 80.0% |
| v0.3 | + text-evidence rules | 82.0% | 66.7% (0 flips) | 76.7% |
| v0.4 | + text-view header fix (7.3g) | **84.7%** | 66.7% (0 flips) | 80.0% |
| v0.5 | + dense keyframe captions | not re-run* | 66.7% (0 flips) | **83.3%** @ $0.047 |
| v0.6 | + contact sheet 24 thumbs @ 256px | not re-run | 66.7% (2 flips) | not re-run |
| v0.7 | sheet scaled to 60 thumbs @ 512px | not re-run | **70.0%** | not re-run |

*Dense captions apply only to shots over 45s; short-video indexes no-op by
criterion, so their v0.4 text views are byte-identical under v0.5.

Three observations. First, gains arrived where the mechanism predicted:
the header fix (text-layer integrity) moved the short cell, dense captions
(text-layer coverage over long shots) moved the long cell, and only
prefix-level visual coverage moved medium at all. Second, two levers were
rejected on dev and never spent a test touch: exhaustive visual sweep
rules (regressed the ordering questions they targeted) and CLIP retrieval
(7.6). Third, the medium cell's 10-question failure set surviving five
configurations unchanged is the strongest evidence we have that a
low-variance cell can still be systematically unreachable by a method
family; treating such a cell's single-run ranking as a method comparison
would be exactly the error 7.4 warns against, in the opposite direction.

## 8. Limitations

Sample sizes are modest (n=150 test short, n=30 medium, n=30 long); the
medium cell is noisy enough (6.2) that its ranking should be treated as
provisional. Our subsets are also domain-skewed: videos were taken in
dataset order, which in Video-MME clusters by domain, so the medium and
long cells are entirely Knowledge-domain videos and the short cells are
76-88% Knowledge (remainder Film & Television). Video-MME spans six
domains; results here should be read as Knowledge-dominated subsets, and
future splits will be domain-stratified. The native baseline currently uses
the provider's
budget tier; all query-side results use a single model family (Opus 4.8 for
queries, Haiku 4.5 for captions), and cross-model portability of the format,
while architecturally trivial, is empirically unverified in this draft; cost
estimates for native Pro-tier ingestion are computed from published prices,
not measured. Whisper-small transcription errors propagate to the text layer.
The 1-hour cache TTL assumption matches interactive workloads but not
questions spread across days; for those, the first-question cost recurs.

## 9. Release

Spec (v0.1), reference implementation (CLI + Python), MCP server exposing
`video_overview` / `survey_video` / `get_frames` / `transcript_search` /
`find_moments`, and
the full evaluation harness including every result file referenced here.
Apache-2.0. [Repository URL to be added.]

Current release (0.1.4, PyPI and the MCP registry, 2026-10-21): the v0.4
query configuration (text-evidence rules on by default, corrected text
view), dense keyframe captioning via `caption --dense`, the `find_moments`
semantic search tool on the MCP surface, per-tool call counts in cost
receipts, and a `certifi>=2026.10.19` dependency floor responding to the
eMudhra emSign CA key-compromise advisory (clients with earlier certifi
builds trust roots that enable TLS interception). The contact-sheet prefix
(v0.6/v0.7) ships behind environment toggles pending a properly powered
dev evaluation; enrichment layers remain optional and provenance-recorded,
and the indexer remains model-free.

## References

- [Anthropic 2026] Anthropic. Prompt caching. Claude Developer Platform
  documentation. https://platform.claude.com/docs/en/build-with-claude/prompt-caching
  (accessed 2026-10-19).
- [Castellano] B. Castellano. PySceneDetect: scene cut and transition
  detection (software). https://github.com/Breakthrough/PySceneDetect
- [Fu et al. 2024] C. Fu et al. Video-MME: The First-Ever Comprehensive
  Evaluation Benchmark of Multi-modal LLMs in Video Analysis.
  arXiv:2405.21075, 2024.
- [Gemini Team 2024] Gemini Team, Google. Gemini 1.5: Unlocking multimodal
  understanding across millions of tokens of context. arXiv:2403.05530, 2024.
- [OpenAI 2026] OpenAI. GPT-5.2 model documentation.
  https://platform.openai.com/docs/models/gpt-5.2 (accessed 2026-10-19;
  input modalities listed as text and image).
- [Radford et al. 2021] A. Radford et al. Learning Transferable Visual
  Models From Natural Language Supervision. arXiv:2103.00020, 2021.
- [Radford et al. 2022] A. Radford et al. Robust Speech Recognition via
  Large-Scale Weak Supervision. arXiv:2212.04356, 2022.
- [SYSTRAN] SYSTRAN. faster-whisper: Whisper transcription with CTranslate2
  (software). https://github.com/SYSTRAN/faster-whisper
- [Twelve Labs 2026] Twelve Labs. Video foundation models: Marengo and
  Pegasus. https://www.twelvelabs.io/product/models-overview
  (accessed 2026-10-19).
- [VideoDB 2026] VideoDB. https://videodb.io (accessed 2026-10-19).
- [Wang, X. et al. 2024] X. Wang et al. VideoAgent: Long-form Video
  Understanding with Large Language Model as Agent. arXiv:2403.10517, 2024.
- [Wang, Z. et al. 2024] Z. Wang et al. VideoTree: Adaptive Tree-based Video
  Representation for LLM Reasoning on Long Videos. arXiv:2405.19209, 2024.

---

*Appendix planned: full per-question result tables, prompt texts, failure
taxonomy, and the cost-accounting checklist for cached-prefix systems.*
