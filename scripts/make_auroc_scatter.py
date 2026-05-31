#!/usr/bin/env python3
"""AUROC vs net bps scatter for the model ladder."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent
OUT = REPO / "results" / "paper" / "figures" / "figure_auroc_scatter.png"

C_NAVY = "#003057"
C_RED = "#d73027"
C_GOLD = "#F2A900"
C_GREY = "#888888"

MODELS = [
    ("PriceBasis10bps", 0.833, -270.0, C_RED, "o"),
    ("PriceBasis25bps", 0.746, -347.3, C_RED, "s"),
    ("GrossArb", 0.531, -136.4, C_RED, "^"),
    ("Logistic", 0.143, -75.8, C_GREY, "D"),
    ("LightGBM*", 0.717, +17.2, C_NAVY, "o"),
    ("GRU (seq)", 0.798, -239.0, C_GREY, "P"),
]

fig, ax = plt.subplots(figsize=(7, 4.2))

# Quadrant shading
ax.axhspan(0, 220, alpha=0.06, color="green")
ax.axhspan(-400, 0, alpha=0.06, color="#d73027")
ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.4)

# Per-point label positions: (dx, dy) offsets from data point
LABEL_OFFSETS = {
    "PriceBasis10bps": (0.013, -28),  # below, to avoid overlap with PriceBasis25bps
    "PriceBasis25bps": (-0.07, -20),  # left of point, above its own marker
    "GrossArb": (0.013, 10),
    "Logistic": (0.013, 10),
    "LightGBM*": (-0.07, 12),  # left so it doesn't run into GRU label
    "GRU (seq)": (-0.075, 10),  # left of cluster
}

for name, auroc, net, col, mrk in MODELS:
    ax.scatter(auroc, net, color=col, marker=mrk, s=80, zorder=5)
    dx, dy = LABEL_OFFSETS.get(name, (0.013, 8))
    ax.annotate(
        name,
        (auroc, net),
        xytext=(auroc + dx, net + dy),
        fontsize=7.5,
        color=col,
        arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.5),
    )

# Quadrant text: moved away from data clusters
ax.text(
    0.30, 155, "Accurate\n& profitable", ha="left", fontsize=8, color="green", alpha=0.8
)
ax.text(
    0.12,
    -300,
    "Accurate but\nunprofitable",
    ha="left",
    fontsize=8,
    color="#d73027",
    alpha=0.8,
)

# Oracle reference line + label
ax.axhline(162.2, color=C_GOLD, lw=1.2, ls="-", alpha=0.7)
ax.text(0.93, 170, "Oracle ceiling (+162 bps)", ha="right", fontsize=7.5, color=C_GOLD)

# LightGBM callout
ax.annotate(
    "Only model\nabove zero",
    xy=(0.717, 17.2),
    xytext=(0.58, 80),
    fontsize=7.5,
    color=C_NAVY,
    arrowprops=dict(arrowstyle="->", color=C_NAVY, lw=0.8),
)

ax.set_xlabel("AUROC", fontsize=11)
ax.set_ylabel("Net bps per trade", fontsize=11)
ax.set_title("Ranking accuracy does not imply positive returns", fontsize=11)
ax.set_xlim(0.05, 1.0)
ax.set_ylim(-400, 220)
ax.grid(alpha=0.2)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}  size={fig.get_size_inches()}")
plt.close(fig)
