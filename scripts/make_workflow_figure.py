#!/usr/bin/env python3
"""Figure: StressBench benchmark workflow.

Compact single-column flowchart showing the pipeline from the 18-event
historical catalogue through execution labels, model evaluation, and key
results.

Output: results/paper_addon/figures/figure_workflow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).parent.parent
OUT = REPO / "results" / "paper_addon" / "figures" / "figure_workflow.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
C_NAVY = "#003057"
C_GOLD = "#F2A900"
C_BLUE = "#75B2DD"
C_GREY = "#DDDDDD"
C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_WHITE = "#FFFFFF"
C_DARK = "#333333"

fig, ax = plt.subplots(figsize=(4.6, 5.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Helper ───────────────────────────────────────────────────────────────────


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc,
    ec=C_DARK,
    fs=7.2,
    tc=C_WHITE,
    bold=False,
    style="round,pad=0.04",
):
    bx = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=style,
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
        zorder=3,
    )
    ax.add_patch(bx)
    weight = "bold" if bold else "normal"
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        fontweight=weight,
        zorder=4,
        multialignment="center",
    )


def arrow(ax, x0, y0, x1, y1, color=C_DARK):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9),
        zorder=2,
    )


def arrow_split(ax, xsrc, ysrc, targets, color=C_DARK):
    """Vertical line down then horizontal branches to targets."""
    y_mid = (ysrc + targets[0][1]) / 2
    ax.plot([xsrc, xsrc], [ysrc, y_mid], color=color, lw=0.9, zorder=2)
    xs = [t[0] for t in targets]
    ax.plot([min(xs), max(xs)], [y_mid, y_mid], color=color, lw=0.9, zorder=2)
    for tx, ty in targets:
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(tx, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9),
            zorder=2,
        )


# ── Layer 1: catalogue ───────────────────────────────────────────────────────
box(
    ax,
    0.50,
    0.945,
    0.90,
    0.085,
    "18-event historical catalogue\n7 mechanism classes · Tier A / B / C",
    C_NAVY,
    bold=True,
    fs=7.4,
)

# ── Arrow split ──────────────────────────────────────────────────────────────
arrow_split(ax, 0.50, 0.902, [(0.24, 0.820), (0.76, 0.820)])

# ── Layer 2: two tracks ──────────────────────────────────────────────────────
box(
    ax,
    0.24,
    0.780,
    0.44,
    0.075,
    "Tier B/C transfer pool\nTerra/LUNA · Celsius · FTX · BUSD",
    C_BLUE,
    tc=C_DARK,
    fs=6.9,
)

box(
    ax,
    0.76,
    0.780,
    0.44,
    0.075,
    "2 Tier-A test windows\nSVB stress + recovery",
    C_GOLD,
    tc=C_DARK,
    fs=6.9,
    bold=True,
)

# ── Layer 3: labels ──────────────────────────────────────────────────────────
arrow(ax, 0.24, 0.742, 0.24, 0.690)
arrow(ax, 0.76, 0.742, 0.76, 0.690)

box(
    ax,
    0.24,
    0.655,
    0.44,
    0.072,
    "Optical basis label\n"
    r"$b = 10^{4}\,(P_{\mathrm{stable}}/P_{\mathrm{real}} - 1)$  bps",
    C_GREY,
    tc=C_DARK,
    fs=6.6,
)

box(
    ax,
    0.76,
    0.655,
    0.44,
    0.072,
    "VWAP net-profit label\n"
    r"$\mathrm{net} = 10^{4}\!\left(\frac{\mathrm{VWAP_{sell}}-\mathrm{VWAP_{buy}}}{P_{\mathrm{ref}}} - f - \delta\right)$",
    C_GOLD,
    tc=C_DARK,
    fs=6.4,
)

# merge arrows
ax.plot([0.24, 0.24], [0.622, 0.587], color=C_DARK, lw=0.9)
ax.plot([0.76, 0.76], [0.622, 0.587], color=C_DARK, lw=0.9)
ax.plot([0.24, 0.76], [0.587, 0.587], color=C_DARK, lw=0.9)
ax.annotate(
    "",
    xy=(0.50, 0.552),
    xytext=(0.50, 0.587),
    arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=0.9),
    zorder=2,
)

# ── Layer 4: model ladder ────────────────────────────────────────────────────
box(
    ax,
    0.50,
    0.514,
    0.90,
    0.070,
    "Model ladder  ·  SVB test split\nprice rules · ML · seq · meta-labeling · PPO-GRU",
    C_NAVY,
    bold=True,
    fs=7.0,
)

# ── Arrow to results ─────────────────────────────────────────────────────────
arrow_split(ax, 0.50, 0.479, [(0.18, 0.415), (0.50, 0.415), (0.82, 0.415)])

# ── Layer 5: three result boxes ──────────────────────────────────────────────
box(
    ax,
    0.18,
    0.375,
    0.27,
    0.070,
    "12× optical\nvs executable gap",
    C_RED,
    tc=C_WHITE,
    fs=6.2,
    bold=True,
)

box(
    ax,
    0.50,
    0.375,
    0.27,
    0.070,
    "Calm models fail\nMeta-label: 51% oracle",
    C_GREEN,
    tc=C_WHITE,
    fs=6.2,
    bold=True,
)

box(
    ax,
    0.82,
    0.375,
    0.27,
    0.070,
    "PPO-GRU −29.2 bps\nSupervision format\nis binding",
    C_DARK,
    tc=C_WHITE,
    fs=6.0,
    bold=False,
)

# ── Layer 6: artefact release ────────────────────────────────────────────────
ax.plot([0.18, 0.18], [0.340, 0.300], color=C_DARK, lw=0.9)
ax.plot([0.50, 0.50], [0.340, 0.300], color=C_DARK, lw=0.9)
ax.plot([0.82, 0.82], [0.340, 0.300], color=C_DARK, lw=0.9)
ax.plot([0.18, 0.82], [0.300, 0.300], color=C_DARK, lw=0.9)
ax.annotate(
    "",
    xy=(0.50, 0.265),
    xytext=(0.50, 0.300),
    arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=0.9),
    zorder=2,
)

box(
    ax,
    0.50,
    0.230,
    0.90,
    0.065,
    "Released artefacts · CC-BY 4.0\ncatalogue · labels · oracle · model harness · leaderboard",
    "#555555",
    tc=C_WHITE,
    fs=6.8,
)

plt.tight_layout(pad=0.2)
fig.savefig(str(OUT), dpi=240, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
