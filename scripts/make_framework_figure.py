#!/usr/bin/env python3
"""Three-layer framework figure: optical -> executable -> predictable.

Panel A: Funnel/bar chart showing what fraction of SVB test minutes
         survive each layer, with Gap 1 and Gap 2 annotated.
Panel B: Worked example -- one representative minute showing the
         optical-to-executable collapse in detail.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#003057"
BLUE = "#75B2DD"
GOLD = "#F2A900"
RED = "#C4122F"
LGREY = "#E8E8E8"
MGREY = "#AAAAAA"

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.bottom": False,
        "ytick.left": False,
    }
)

# ── data ──────────────────────────────────────────────────────────────────────
N_TOTAL = 15_832  # SVB test minutes
OPT_FRAC = 0.3433  # Layer 1: optical (max-abs basis > 10 bps)
EXE_FRAC = 0.0288  # Layer 2: executable after VWAP + fees + latency
# Layer 3: captured by model (meta-label ~1.47% of total = 51% of executable)
META_FRAC = EXE_FRAC * 0.510
CALM_FRAC = EXE_FRAC * 0.060  # best calm model (~6% hit rate)

ROWS = [
    ("Layer 1\nOptical", OPT_FRAC, GOLD, "34.3%\n(5,430 min)"),
    ("Layer 2\nExecutable", EXE_FRAC, NAVY, "2.88%\n(456 min)"),
    ("Layer 3\nCaptured\n(meta-label)", META_FRAC, BLUE, "1.47%\n(233 min)"),
]

fig, axes = plt.subplots(
    1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.35, 1]}
)
plt.subplots_adjust(wspace=0.06)

# ── Panel A: three-layer horizontal bars ──────────────────────────────────────
ax = axes[0]

bar_h = 0.52
y_positions = [2.1, 1.2, 0.3]

for i, (label, frac, colour, txt) in enumerate(ROWS):
    y = y_positions[i]
    # background (remaining fraction)
    ax.barh(y, 1.0, height=bar_h, color=LGREY, left=0, zorder=1)
    # filled fraction
    ax.barh(y, frac, height=bar_h, color=colour, left=0, zorder=2)
    # label on left
    ax.text(-0.015, y, label, ha="right", va="center", fontsize=8.5, color="#333333")
    # percentage inside bar or just outside
    x_txt = frac + 0.01 if frac < 0.15 else frac * 0.5
    ha_txt = "left" if frac < 0.15 else "center"
    c_txt = "#333333" if frac < 0.15 else "white"
    ax.text(
        x_txt,
        y,
        txt,
        ha=ha_txt,
        va="center",
        fontsize=8.5,
        color=c_txt,
        fontweight="bold",
    )


# Gap annotations
def gap_arrow(ax, x, y_top, y_bot, label):
    ax.annotate(
        "",
        xy=(x, y_bot + bar_h / 2 + 0.03),
        xytext=(x, y_top - bar_h / 2 - 0.03),
        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2),
    )
    ax.text(
        x + 0.015,
        (y_top + y_bot) / 2,
        label,
        ha="left",
        va="center",
        fontsize=7.5,
        color=RED,
        style="italic",
    )


gap_arrow(ax, 0.38, y_positions[0], y_positions[1], "Gap 1\n12x")
gap_arrow(ax, 0.06, y_positions[1], y_positions[2], "Gap 2\n~50% of exec.")

ax.set_xlim(-0.42, 0.62)
ax.set_ylim(-0.15, 2.65)
ax.set_title("(a) Three layers: SVB test window", fontsize=9.5, pad=4)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_xticklabels(["0", "10%", "20%", "30%", "40%"], fontsize=7.5)
ax.set_yticks([])
ax.tick_params(bottom=True)
ax.spines["bottom"].set_visible(True)
ax.set_xlabel("Fraction of 15,832 test minutes", fontsize=8.5)

# ── Panel B: worked example as a cost-stack waterfall ─────────────────────────
# One representative false-positive minute (Mar 11 2023). The chart shows a
# +112 bps USDT spike, but the tradeable USDC route starts near peg and the
# execution-cost stack pushes it to a net loss. A waterfall reads far more
# cleanly than a text dump: each grey bar is one cost, the red bar is the net.
ax2 = axes[1]

steps = [
    ("USDC route gross", -2.8),
    ("Taker fees (4+4 bps)", -8.0),
    ("Settlement latency", -5.0),
]
y_pos = [3, 2, 1]
bar_h_b = 0.58
cum = 0.0
for (label, delta), y in zip(steps, y_pos):
    left = cum
    cum += delta
    ax2.barh(
        y,
        delta,
        height=bar_h_b,
        left=left,
        color=MGREY,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax2.text(
        cum - 0.5,
        y,
        f"{delta:+.1f}",
        ha="right",
        va="center",
        fontsize=8.0,
        color="#333333",
    )
    ax2.text(0.5, y, label, ha="left", va="center", fontsize=8.0, color="#333333")

net = cum  # -15.8
ax2.barh(0, net, height=bar_h_b, left=0, color=RED, zorder=3)
ax2.text(
    net - 0.5,
    0,
    f"{net:+.1f} bps",
    ha="right",
    va="center",
    fontsize=8.5,
    color=RED,
    fontweight="bold",
)
ax2.text(
    0.5,
    0,
    "Net (not executable)",
    ha="left",
    va="center",
    fontsize=8.0,
    color=RED,
    fontweight="bold",
)

ax2.axvline(0, color="#888888", lw=0.7, zorder=1)
ax2.set_xlim(-20, 9)
ax2.set_ylim(-0.6, 3.95)
ax2.set_yticks([])
ax2.set_xlabel("Net basis points", fontsize=8.5)
ax2.tick_params(labelsize=7.5)
for sp in ("top", "right", "left"):
    ax2.spines[sp].set_visible(False)

# Context line: the chart signal that triggered the (false) entry
ax2.text(
    0.5,
    1.04,
    r"Chart: USDT $+112$ bps; tradeable USDC route $+1$ bps",
    transform=ax2.transAxes,
    ha="center",
    va="bottom",
    fontsize=7.0,
    color=GOLD,
    fontweight="bold",
)
ax2.set_title(
    r"(b) One false-positive minute: optical $\neq$ executable", fontsize=9.5, pad=16
)

fig.tight_layout(pad=0.5)
out = OUT / "figure_framework.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
