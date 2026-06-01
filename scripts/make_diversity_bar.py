#!/usr/bin/env python3
"""Figure B: Multi-mechanism oracle capture bar chart.

Grouped bars: net bps and oracle capture % for four training conditions.
A dashed horizontal line marks the 50% ceiling. FTX bar is hatched.

Output: results/paper_addon/figures/figure_diversity_bar.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).parent.parent
OUT = REPO / "results" / "paper_addon" / "figures" / "figure_diversity_bar.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Results from run_multi_event_diversity.py (seed=42)
CONDITIONS = [
    ("Terra/LUNA\nonly", 78.5, 123, 48.4, False),
    ("Celsius/3AC\nonly", 79.7, 221, 49.1, False),
    ("FTX\nonly", 36.5, 1, 22.5, True),  # near-degenerate
    ("All four\npooled", 83.7, 163, 51.6, False),
]
ORACLE_NET = 161.7
ORACLE_CAP = 100.0

LABELS = [c[0] for c in CONDITIONS]
NET_BPS = [c[1] for c in CONDITIONS]
ORK_CAP = [c[3] for c in CONDITIONS]
DEGENERATE = [c[4] for c in CONDITIONS]

C_NORMAL = "#2ca02c"  # green
C_DEGEN = "#cccccc"  # light grey
C_ORACLE = "#F2A900"  # gold

plt.rcParams.update({"font.size": 11})
fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
ax2 = ax1.twinx()

x = np.arange(len(LABELS))
w = 0.38

bars_net = []
for i, (nb, degen) in enumerate(zip(NET_BPS, DEGENERATE)):
    color = C_DEGEN if degen else C_NORMAL
    b = ax1.bar(
        x[i] - w / 2,
        nb,
        w,
        color=color,
        hatch="////" if degen else "",
        edgecolor="white" if not degen else "#888888",
        linewidth=0.8,
        zorder=3,
    )
    bars_net.append(b)

bars_cap = []
for i, (oc, degen) in enumerate(zip(ORK_CAP, DEGENERATE)):
    color = C_DEGEN if degen else "#1f77b4"
    b = ax2.bar(
        x[i] + w / 2,
        oc,
        w,
        color=color,
        alpha=0.75,
        hatch="////" if degen else "",
        edgecolor="white" if not degen else "#888888",
        linewidth=0.8,
        zorder=3,
    )
    bars_cap.append(b)

# Oracle reference lines
ax1.axhline(
    ORACLE_NET,
    color=C_ORACLE,
    linestyle="--",
    linewidth=1.2,
    zorder=2,
    label=f"Oracle +{ORACLE_NET:.1f} bps",
)
ax2.axhline(
    50, color="#444444", linestyle=":", linewidth=1.0, zorder=2, label="50% ceiling"
)
ax2.axhline(
    ORACLE_CAP, color=C_ORACLE, linestyle="--", linewidth=1.2, zorder=2, alpha=0.6
)

# Axes labels and limits
ax1.set_ylabel("Net bps (left)", fontsize=11)
ax2.set_ylabel("Oracle capture % (right)", fontsize=11)
ax1.set_ylim(-5, 200)
ax2.set_ylim(-5, 115)
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, fontsize=10.5)
ax1.yaxis.set_tick_params(labelsize=10)
ax2.yaxis.set_tick_params(labelsize=10)

# Annotate FTX bar
ax2.annotate(
    "1 trade\n(degenerate)",
    xy=(x[2] + w / 2, ORK_CAP[2] + 1.5),
    fontsize=9,
    ha="center",
    color="#666666",
    arrowprops=dict(arrowstyle="-", lw=0.6),
)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=C_NORMAL, label="Net bps"),
    mpatches.Patch(facecolor="#1f77b4", alpha=0.75, label="Oracle cap. %"),
    mpatches.Patch(
        facecolor=C_DEGEN, hatch="////", edgecolor="#888888", label="Near-degenerate"
    ),
    plt.Line2D(
        [0], [0], color="#444444", linestyle=":", linewidth=1.0, label="50% ceiling"
    ),
    plt.Line2D([0], [0], color=C_ORACLE, linestyle="--", linewidth=1.2, label="Oracle"),
]
ax1.legend(
    handles=legend_elements,
    fontsize=9.5,
    loc="upper left",
    framealpha=0.85,
    ncol=1,
    borderpad=0.5,
)

ax1.set_title(
    "Training diversity: oracle capture on SVB test split", fontsize=11, pad=4
)
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)

plt.tight_layout(pad=0.8)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
