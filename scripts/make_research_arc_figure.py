#!/usr/bin/env python3
"""Cross-study narrative figure: how the four stablecoin papers form one program.

This is a repo-level storytelling visual (not in any single 8pp paper) showing the
research arc — correlation -> causation -> measurement -> executability — across the
four coordinated stablecoin studies. Same Columbia theme as the paper figures.

Output: docs/figures/research_program_arc.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1D4F91"
INK = "#0A1F44"
COLBLUE = "#B9D9EB"
MID = "#6CA6CD"
LBLUE = "#EAF2FA"
GREY = "#8895A7"
AMBER = "#E08E2B"

plt.rcParams.update({"font.family": "serif", "text.color": INK})

STAGES = [
    ("contagion-gnn", "1 · PREDICT", "Graph-attention network\nflags the correlational\ncontagion hubs",
     "precision@10:  GAT 0.40  vs  XGB 0.00"),
    ("contagion-abm", "2 · TEST CAUSALITY", "Calibrated counterfactual\nknockout asks if the hub\nactually transmits",
     "top hub BUSD:  zero causal effect"),
    ("contagion-network", "3 · MEASURE", "Provenance-gated on-chain\nHMM detects mechanism-\nspecific stress",
     "7 episodes (2022–25):  mechanism-split"),
    ("stressbench", "4 · EXECUTE", "Book-walking labels separate\nvisible from tradeable\ndislocations",
     "12× optical-to-executable gap"),
]
CONNECT = [
    "“BUSD is the hub”",
    "“…but it isn’t causal”",
    "“…here’s where stress\nactually flows”",
]

fig, ax = plt.subplots(figsize=(13.0, 4.7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 4.7)
ax.axis("off")

box_w, box_h = 2.7, 2.35
gap = (13 - 4 * box_w) / 5
y0 = 1.15
xs = []
for i, (repo, stage, body, metric) in enumerate(STAGES):
    x = gap + i * (box_w + gap)
    xs.append(x)
    # card
    ax.add_patch(FancyBboxPatch((x, y0), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=1.4, edgecolor=NAVY, facecolor=LBLUE, zorder=2))
    # header strip
    ax.add_patch(FancyBboxPatch((x, y0 + box_h - 0.46), box_w, 0.46,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=0, facecolor=NAVY, zorder=3))
    ax.text(x + box_w / 2, y0 + box_h - 0.23, stage, ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold", zorder=4)
    ax.text(x + box_w / 2, y0 + box_h - 0.72, repo, ha="center", va="center",
            color=NAVY, fontsize=9.5, family="monospace", fontweight="bold", zorder=4)
    ax.text(x + box_w / 2, y0 + 0.92, body, ha="center", va="center",
            color=INK, fontsize=8.7, zorder=4)
    # metric chip
    ax.add_patch(FancyBboxPatch((x + 0.14, y0 + 0.12), box_w - 0.28, 0.42,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                linewidth=0, facecolor=COLBLUE, zorder=3))
    ax.text(x + box_w / 2, y0 + 0.33, metric, ha="center", va="center",
            color=INK, fontsize=7.0, fontweight="bold", zorder=4)

# arrows between stages
for i in range(3):
    x1 = xs[i] + box_w
    x2 = xs[i + 1]
    ymid = y0 + box_h / 2
    ax.add_patch(FancyArrowPatch((x1 + 0.04, ymid), (x2 - 0.04, ymid),
                                 arrowstyle="-|>", mutation_scale=20, lw=2.2,
                                 color=AMBER, zorder=5))

ax.text(6.5, 4.45, "A four-study program on stablecoin systemic risk",
        ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
ax.text(6.5, 4.06, "from  correlation  →  causation  →  measurement  →  executability",
        ha="center", va="center", fontsize=10.5, color=NAVY, style="italic")
ax.text(6.5, 0.42,
        "Each study is a standalone paper; together they trace one dislocation signal from a predictor's "
        "ranking down to what is actually tradeable.",
        ha="center", va="center", fontsize=8.3, color=GREY)

out = OUT / "research_program_arc.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
