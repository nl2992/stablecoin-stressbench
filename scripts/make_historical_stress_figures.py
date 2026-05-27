#!/opt/anaconda3/bin/python
"""
make_historical_stress_figures.py
----------------------------------
Produce historical stress coverage figures for the paper addon.

Outputs:
  results/paper_addon/figures/figure_29_event_coverage_heatmap.png
    — Data-type coverage matrix: which stress events have which data types.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "results", "paper_addon", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Columbia academic palette ─────────────────────────────────────────────────
NAVY   = "#003057"
BLUE   = "#75B2DD"
GOLD   = "#F2A900"
LGRAY  = "#D9D9D9"
WHITE  = "#FFFFFF"

# ── Coverage matrix definition ───────────────────────────────────────────────
# Values: 0 = unavailable, 1 = partial, 2 = available
# Rows: events (most recent last so chart reads top-to-bottom chronologically)
# Columns: data types

EVENTS = [
    "DAI Black Thurs. 2020",
    "IRON/TITAN 2021",
    "Terra/UST 2022",
    "Celsius/3AC 2022",
    "FTX 2022",
    "BUSD 2023",
    "USDT/Curve 2023",
    "USDC/SVB 2023",     # Tier A — highlighted
    "USDC recovery 2023",# Tier A — highlighted
]

COLS = [
    "Price\n(OHLCV)",
    "Trade\ntape",
    "L2 order\nbook",
    "DEX\npool",
    "On-chain\ndata",
    "Exec.\nlabels",
    "Model\neval.",
]

# coverage[row][col]: 0=none, 1=partial, 2=full
COVERAGE = np.array([
    # DAI Black Thursday 2020 — Tier B (collateral/liquidation)
    [2, 1, 0, 1, 1, 0, 0],
    # IRON/TITAN 2021 — Tier C (Polygon DEX only)
    [1, 0, 0, 1, 0, 0, 0],
    # Terra/UST 2022 — Tier B (validation split in benchmark)
    [2, 2, 0, 2, 1, 1, 0],
    # Celsius/3AC 2022 — Tier B (exchange credit)
    [2, 1, 0, 1, 1, 0, 0],
    # FTX 2022 — Tier B (exchange credit/liquidity)
    [2, 2, 0, 1, 1, 0, 0],
    # BUSD 2023 — Tier B (regulatory winddown)
    [2, 1, 0, 0, 1, 0, 0],
    # USDT/Curve 2023 — Tier B (DeFi pool imbalance)
    [2, 1, 0, 2, 1, 0, 0],
    # USDC/SVB 2023 — Tier A (PRIMARY benchmark)
    [2, 2, 2, 1, 1, 2, 2],
    # USDC recovery 2023 — Tier A comparator
    [2, 2, 2, 1, 1, 2, 1],
], dtype=float)

TIER_A_ROWS = {7, 8}  # USDC/SVB and USDC recovery


def make_heatmap():
    n_rows, n_cols = COVERAGE.shape

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    fig.patch.set_facecolor(WHITE)

    # Colour map: 0=light gray, 1=benchmark blue, 2=navy
    cmap = ListedColormap([LGRAY, BLUE, NAVY])

    im = ax.imshow(COVERAGE, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COLS, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(EVENTS, fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # ── Cell annotations ──────────────────────────────────────────────────────
    label_map = {0: "", 1: "partial", 2: "✓"}
    for r in range(n_rows):
        for c in range(n_cols):
            v = int(COVERAGE[r, c])
            text = label_map[v]
            color = WHITE if v == 2 else (NAVY if v == 1 else LGRAY)
            fontcolor = WHITE if v == 2 else (NAVY if v == 1 else "#999999")
            if text:
                ax.text(c, r, text, ha="center", va="center",
                        fontsize=7, color=fontcolor, fontweight="bold" if v == 2 else "normal")

    # ── Highlight Tier-A rows ─────────────────────────────────────────────────
    for r in TIER_A_ROWS:
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.5, r - 0.5), n_cols, 1.0,
            boxstyle="square,pad=0",
            linewidth=1.8, edgecolor=GOLD, facecolor="none",
            zorder=5
        ))
        ax.text(-0.6, r, "★", ha="right", va="center",
                fontsize=9, color=GOLD, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=NAVY, label="Available"),
        mpatches.Patch(facecolor=BLUE, label="Partial"),
        mpatches.Patch(facecolor=LGRAY, edgecolor="#aaaaaa", label="Unavailable"),
        mpatches.Patch(facecolor="none", edgecolor=GOLD, linewidth=1.8,
                       label="Tier-A (★)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              bbox_to_anchor=(1.0, -0.18), ncol=4,
              fontsize=7, frameon=True, framealpha=0.9)

    ax.set_title(
        "Historical stress-event data coverage.\n"
        r"Stablecoin stress is historically broad, but execution-grade labels (★) require scarce order-book depth.",
        fontsize=8, pad=12, color=NAVY
    )

    plt.tight_layout(rect=[0.05, 0.08, 1.0, 1.0])

    out_path = os.path.join(OUT_DIR, "figure_29_event_coverage_heatmap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    make_heatmap()
    print("Done.")
