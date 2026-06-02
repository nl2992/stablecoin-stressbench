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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT  = Path(__file__).parent.parent
OUT   = ROOT / "results" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY  = "#003057"
BLUE  = "#75B2DD"
GOLD  = "#F2A900"
RED   = "#C4122F"
LGREY = "#E8E8E8"
MGREY = "#AAAAAA"

plt.rcParams.update({
    "font.family":         "serif",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    False,
    "axes.spines.bottom":  False,
    "xtick.bottom":        False,
    "ytick.left":          False,
})

# ── data ──────────────────────────────────────────────────────────────────────
N_TOTAL  = 15_832          # SVB test minutes
OPT_FRAC = 0.3433          # Layer 1: optical (max-abs basis > 10 bps)
EXE_FRAC = 0.0288          # Layer 2: executable after VWAP + fees + latency
# Layer 3: captured by model (meta-label ~1.47% of total = 51% of executable)
META_FRAC = EXE_FRAC * 0.510
CALM_FRAC = EXE_FRAC * 0.060   # best calm model (~6% hit rate)

ROWS = [
    ("Layer 1\nOptical",      OPT_FRAC,  GOLD,  "34.3%\n(5,430 min)"),
    ("Layer 2\nExecutable",   EXE_FRAC,  NAVY,  "2.88%\n(456 min)"),
    ("Layer 3\nCaptured\n(meta-label)", META_FRAC, BLUE, "1.47%\n(233 min)"),
]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                         gridspec_kw={"width_ratios": [1.35, 1]})
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
    ax.text(-0.015, y, label, ha="right", va="center", fontsize=6.5,
            color="#333333")
    # percentage inside bar or just outside
    x_txt = frac + 0.01 if frac < 0.15 else frac * 0.5
    ha_txt = "left"   if frac < 0.15 else "center"
    c_txt  = "#333333" if frac < 0.15 else "white"
    ax.text(x_txt, y, txt, ha=ha_txt, va="center",
            fontsize=6.5, color=c_txt, fontweight="bold")

# Gap annotations
def gap_arrow(ax, x, y_top, y_bot, label):
    ax.annotate("", xy=(x, y_bot + bar_h / 2 + 0.03),
                xytext=(x, y_top - bar_h / 2 - 0.03),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax.text(x + 0.015, (y_top + y_bot) / 2, label,
            ha="left", va="center", fontsize=6, color=RED, style="italic")

gap_arrow(ax, 0.38, y_positions[0], y_positions[1], "Gap 1\n12x")
gap_arrow(ax, 0.06, y_positions[1], y_positions[2], "Gap 2\n~50% of exec.")

ax.set_xlim(-0.42, 0.62)
ax.set_ylim(-0.15, 2.65)
ax.set_title("(a) Three layers: SVB test window", fontsize=7.5, pad=4)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_xticklabels(["0", "10%", "20%", "30%", "40%"], fontsize=6)
ax.set_yticks([])
ax.tick_params(bottom=True)
ax.spines["bottom"].set_visible(True)
ax.set_xlabel("Fraction of 15,832 test minutes", fontsize=7)

# ── Panel B: worked example ────────────────────────────────────────────────────
ax2 = axes[1]
ax2.axis("off")

# Box showing one window
title_y = 0.96
lines = [
    ("11:54 PM, Mar 11 2023 -- a typical false positive", 7.0, NAVY, "bold"),
    ("", 4, "white", "normal"),
    ("USDT basis:     +112 bps  (chart shows opportunity)", 6.3, GOLD, "normal"),
    ("USDC basis:       +1 bps  (actual route: near zero)", 6.3, MGREY, "normal"),
    ("", 3, "white", "normal"),
    ("Why the USDC route fails:", 6.3, "#333333", "bold"),
    ("  Primary filter fires on max-abs basis (USDT spike)", 6.0, "#555555", "normal"),
    ("  BTC-USDC book: thin at $24,810; VWAP walk = $24,819", 6.0, "#555555", "normal"),
    ("  BTC-USD (sell): $24,812", 6.0, "#555555", "normal"),
    ("  Gross USDC margin: (24812-24819)/24810 x 10000 = -2.8 bps", 6.0, "#555555", "normal"),
    ("  Taker fees (4+4): -8 bps", 6.0, "#555555", "normal"),
    ("  Settlement latency: -5 bps", 6.0, "#555555", "normal"),
    ("", 3, "white", "normal"),
    ("Net result:  -15.8 bps  (not executable)", 6.8, RED, "bold"),
    ("", 3, "white", "normal"),
    ("Root cause: route-direction mismatch.", 6.0, "#333333", "normal"),
    ("USDT dislocated; USDC route stays near peg.", 6.0, "#333333", "normal"),
]

y_cur = title_y
line_h = 0.068
for text, fs, color, weight in lines:
    if text == "":
        y_cur -= fs / 200
        continue
    ax2.text(0.04, y_cur, text, transform=ax2.transAxes,
             fontsize=fs, color=color, fontweight=weight,
             va="top", ha="left", fontfamily="monospace" if "bps" in text and "$" not in text else "serif")
    y_cur -= line_h * (fs / 6.5)

ax2.set_title("(b) Worked example: optical != executable", fontsize=7.5, pad=4)
# Border box
rect = mpatches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.015", linewidth=0.8,
    edgecolor=NAVY, facecolor="#FAFAFA",
    transform=ax2.transAxes, zorder=0)
ax2.add_patch(rect)

fig.tight_layout(pad=0.5)
out = OUT / "figure_framework.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
