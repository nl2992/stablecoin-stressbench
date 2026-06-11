#!/usr/bin/env python3
"""Figure: StressBench benchmark workflow.

Compact single-column flowchart showing the pipeline from the 18-event
historical catalogue through execution labels, model evaluation, and key
results. Plain academic styling: light boxes, thin black borders, black
serif text, and LaTeX-style math for the two label definitions.

Output: results/paper_addon/figures/figure_workflow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).parent.parent
OUT = REPO / "results" / "paper_addon" / "figures" / "figure_workflow.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Academic styling: serif text, Computer-Modern math ───────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
    }
)

# ── Palette: grayscale only ──────────────────────────────────────────────────
C_LINE = "#000000"  # borders, arrows, text
C_LEAF = "#ffffff"  # leaf boxes
C_STAGE = "#e9e9e9"  # stage-header boxes (light gray)

fig, ax = plt.subplots(figsize=(4.7, 5.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Helpers ──────────────────────────────────────────────────────────────────


def box(ax, x, y, w, h, text, fc=C_LEAF, fs=7.4, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="square,pad=0.012",
            facecolor=fc,
            edgecolor=C_LINE,
            linewidth=0.8,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=C_LINE,
        fontweight="bold" if bold else "normal",
        zorder=4,
        multialignment="center",
    )


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=C_LINE, lw=0.8),
        zorder=2,
    )


def arrow_split(ax, xsrc, ysrc, targets):
    """Vertical drop then horizontal branches to each target."""
    y_mid = (ysrc + targets[0][1]) / 2
    ax.plot([xsrc, xsrc], [ysrc, y_mid], color=C_LINE, lw=0.8, zorder=2)
    xs = [t[0] for t in targets]
    ax.plot([min(xs), max(xs)], [y_mid, y_mid], color=C_LINE, lw=0.8, zorder=2)
    for tx, ty in targets:
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(tx, y_mid),
            arrowprops=dict(arrowstyle="-|>", color=C_LINE, lw=0.8),
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
    fc=C_STAGE,
    bold=True,
    fs=7.6,
)
arrow_split(ax, 0.50, 0.902, [(0.26, 0.820), (0.74, 0.820)])

# ── Layer 2: two tracks ──────────────────────────────────────────────────────
box(
    ax,
    0.26,
    0.780,
    0.44,
    0.075,
    "Tier B/C transfer pool\nTerra/LUNA · Celsius · FTX · BUSD",
    fs=7.0,
)
box(
    ax,
    0.74,
    0.780,
    0.44,
    0.075,
    "Two Tier-A test windows\nSVB stress + recovery",
    fs=7.0,
)
arrow(ax, 0.26, 0.742, 0.26, 0.690)
arrow(ax, 0.74, 0.742, 0.74, 0.690)

# ── Layer 3: label definitions (real formulas) ───────────────────────────────
box(
    ax,
    0.26,
    0.652,
    0.44,
    0.076,
    "Optical basis label\n"
    r"$b = 10^{4}\,(P_{\mathrm{stable}}/P_{\mathrm{real}} - 1)$  bps",
    fs=7.0,
)
box(
    ax,
    0.74,
    0.652,
    0.44,
    0.076,
    "VWAP net-profit label\n"
    r"$\mathrm{net} = 10^{4}\!\left(\dfrac{\mathrm{VWAP}_{\mathrm{sell}}"
    r"-\mathrm{VWAP}_{\mathrm{buy}}}{P_{\mathrm{ref}}} - f - \delta\right)$",
    fs=7.0,
)

# merge into model ladder
ax.plot([0.26, 0.26], [0.614, 0.585], color=C_LINE, lw=0.8)
ax.plot([0.74, 0.74], [0.614, 0.585], color=C_LINE, lw=0.8)
ax.plot([0.26, 0.74], [0.585, 0.585], color=C_LINE, lw=0.8)
arrow(ax, 0.50, 0.585, 0.50, 0.552)

# ── Layer 4: model ladder ────────────────────────────────────────────────────
box(
    ax,
    0.50,
    0.514,
    0.90,
    0.070,
    "Model ladder  ·  SVB test split\nprice rules · ML · sequence · meta-labeling · PPO-GRU",
    fc=C_STAGE,
    bold=True,
    fs=7.2,
)
arrow_split(ax, 0.50, 0.479, [(0.20, 0.415), (0.50, 0.415), (0.80, 0.415)])

# ── Layer 5: three result boxes ──────────────────────────────────────────────
box(
    ax,
    0.20,
    0.375,
    0.29,
    0.072,
    "12$\\times$ optical-to-\nexecutable gap",
    fs=7.0,
)
box(
    ax,
    0.50,
    0.375,
    0.29,
    0.072,
    "Calm-trained models\nfail (51% oracle)",
    fs=7.0,
)
box(
    ax,
    0.80,
    0.375,
    0.29,
    0.072,
    "PPO-GRU $-$29.2 bps:\nsupervision binds",
    fs=7.0,
)

# merge into release
ax.plot([0.20, 0.20], [0.339, 0.300], color=C_LINE, lw=0.8)
ax.plot([0.50, 0.50], [0.339, 0.300], color=C_LINE, lw=0.8)
ax.plot([0.80, 0.80], [0.339, 0.300], color=C_LINE, lw=0.8)
ax.plot([0.20, 0.80], [0.300, 0.300], color=C_LINE, lw=0.8)
arrow(ax, 0.50, 0.300, 0.50, 0.267)

# ── Layer 6: artefact release ────────────────────────────────────────────────
box(
    ax,
    0.50,
    0.229,
    0.90,
    0.065,
    "Released artefacts (CC-BY 4.0)\ncatalogue · labels · oracle · model harness · leaderboard",
    fc=C_STAGE,
    fs=7.0,
)

plt.tight_layout(pad=0.2)
fig.savefig(str(OUT), dpi=240, bbox_inches="tight")
plt.close()
print(f"Saved {OUT}")
