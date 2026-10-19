"""Figure 1: per-question cost vs questions asked per video (amortization).

Derives first-question and follow-up costs from the frozen long-run receipts
(eval/results/FINAL_long30.json) and the naive baseline from
eval/results/long_naive.json, so the figure can never drift from the
committed numbers. Native is a flat estimate from published Pro-tier prices.

Usage: python paper/figures/amortization.py   (writes amortization.pdf/.png here)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

ACCENT = "#2a78d6"   # vctx (the subject)
GRAY_1 = "#52514e"   # naive (context)
GRAY_2 = "#898781"   # native estimate (context)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
NATIVE_EST = 0.50

rows = json.loads((ROOT / "eval/results/FINAL_long30.json").read_text())
by_video = defaultdict(list)
for r in rows:
    by_video[r["qid"].rsplit("-", 1)[0]].append(r["usd"])
firsts = [v[0] for v in by_video.values()]
follows = [u for v in by_video.values() for u in v[1:]]
first = sum(firsts) / len(firsts)
follow = sum(follows) / len(follows)

naive_rows = json.loads((ROOT / "eval/results/long_naive.json").read_text())
naive = sum(r["usd"] for r in naive_rows) / len(naive_rows)

ks = list(range(1, 31))
vctx = [(first + (k - 1) * follow) / k for k in ks]

matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=200)
ax.set_yscale("log")

ax.plot(ks, [NATIVE_EST] * len(ks), color=GRAY_2, lw=1.6, ls=":")
ax.plot(ks, [naive] * len(ks), color=GRAY_1, lw=1.6, ls="--")
ax.plot(ks, vctx, color=ACCENT, lw=2.2)

ax.annotate(f"native ingestion (est. published rates)  ~${NATIVE_EST:.2f}",
            xy=(30, NATIVE_EST), xytext=(29.5, NATIVE_EST * 1.18),
            ha="right", color=GRAY_2, fontsize=8)
ax.annotate(f"naive frame sampling  ${naive:.3f}",
            xy=(30, naive), xytext=(29.5, naive * 1.15),
            ha="right", color=GRAY_1, fontsize=8)
ax.annotate(f"vctx (ours)  ${vctx[-1]:.3f} at k=30",
            xy=(30, vctx[-1]), xytext=(29.5, vctx[-1] * 0.78),
            ha="right", color=ACCENT, fontsize=8, fontweight="bold")
ax.annotate(f"first question ${first:.2f}\n(pays the one-time cache write)",
            xy=(1, vctx[0]), xytext=(2.2, vctx[0] * 1.25),
            color=MUTED, fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))

ax.set_xlabel("questions asked per video (k)")
ax.set_ylabel("cost per question (USD, log scale)")
ax.set_xlim(1, 30)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
ax.set_yticklabels(["$0.02", "$0.05", "$0.10", "$0.20", "$0.50"])
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.tick_params(which="minor", length=0)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.tick_params(length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig(OUT / "amortization.pdf")
fig.savefig(OUT / "amortization.png")
print(f"first ${first:.3f}, follow-up ${follow:.3f}, naive ${naive:.3f}; "
      f"vctx at k=3 ${vctx[2]:.3f}, k=10 ${vctx[9]:.3f}, k=30 ${vctx[29]:.3f}")
