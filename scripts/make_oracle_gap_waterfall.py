#!/usr/bin/env python3
"""Oracle-gap waterfall decomposition figure (T1.3)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).parent.parent
OUT = REPO / "results" / "paper" / "figures" / "figure_oracle_gap_waterfall.png"

ORACLE_BPS = 225.0
BEST_MODEL = -41.1
GAP = ORACLE_BPS - BEST_MODEL
TIMING_MID = 60.0
TIMING_ERR = 20.0
SIZING_MID = 70.0
SIZING_ERR = 30.0
FP_COST = GAP - TIMING_MID - SIZING_MID  # 136.1 bps

STEPS = [
    0,
    ORACLE_BPS,
    ORACLE_BPS - FP_COST,
    ORACLE_BPS - FP_COST - TIMING_MID,
    ORACLE_BPS - FP_COST - TIMING_MID - SIZING_MID,
]

C_ORACLE = "#F2A900"
C_FP = "#d73027"
C_TIMING = "#fc8d59"
C_SIZING = "#fee090"
C_NAVY = "#003057"
C_GREY = "#888888"


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = [
        "Oracle\nceiling",
        "FP drag\n(residual est.)",
        "Timing gap\n(40-80 bps)",
        "Sizing gap\n(40-100 bps)",
        "Best model",
    ]
    bar_bottoms = [0, STEPS[1], STEPS[2], STEPS[3], BEST_MODEL]
    bar_heights = [ORACLE_BPS, -FP_COST, -TIMING_MID, -SIZING_MID, 0]
    bar_colors = [C_ORACLE, C_FP, C_TIMING, C_SIZING, C_NAVY]
    hatches = [None, None, "///", "///", None]

    for i, (bot, h, col, hatch) in enumerate(
        zip(bar_bottoms, bar_heights, bar_colors, hatches)
    ):
        if i == 4:
            ax.axhline(BEST_MODEL, color=col, lw=2, ls="--", zorder=3)
            ax.text(
                i,
                BEST_MODEL - 6,
                f"{BEST_MODEL:+.1f} bps",
                ha="center",
                va="top",
                fontsize=8.5,
                color=col,
                fontweight="bold",
            )
        else:
            ax.bar(
                i,
                abs(h),
                bottom=min(bot, bot + h),
                color=col,
                alpha=0.85,
                width=0.58,
                edgecolor="white",
                linewidth=1.2,
                hatch=hatch,
                zorder=2,
            )
            ax.text(
                i,
                bot + h / 2,
                f"{'+'if h>0 else ''}{h:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if col != C_SIZING else C_NAVY,
            )
            if i == 2:
                ax.errorbar(
                    i,
                    bot + h / 2,
                    yerr=TIMING_ERR,
                    fmt="none",
                    color="black",
                    capsize=6,
                    lw=2,
                    zorder=4,
                )
            if i == 3:
                ax.errorbar(
                    i,
                    bot + h / 2,
                    yerr=SIZING_ERR,
                    fmt="none",
                    color="black",
                    capsize=6,
                    lw=2,
                    zorder=4,
                )

    for i in range(4):
        ax.plot(
            [i + 0.29, i + 0.71],
            [STEPS[i + 1]] * 2,
            color=C_GREY,
            lw=0.9,
            ls="--",
            alpha=0.6,
            zorder=1,
        )

    ax.axhline(ORACLE_BPS, color=C_ORACLE, lw=1.2, ls=":", alpha=0.7)
    ax.axhline(0, color="black", lw=0.7, alpha=0.4)
    ax.set_xticks(range(5))
    ax.set_xticklabels(categories, fontsize=8.8)
    ax.set_ylabel("Net bps per trade", fontsize=10.5)
    ax.set_title(
        "Oracle Gap Decomposition  (executable arb., $10K notional, SVB test split)",
        fontsize=10.5,
    )
    ax.set_ylim(-80, 290)
    ax.set_xlim(-0.5, 4.5)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    legend_handles = [
        mpatches.Patch(
            color=C_FP,
            alpha=0.85,
            label=f"FP drag (residual = gap minus timing/sizing: {FP_COST:.0f} bps)",
        ),
        mpatches.Patch(
            color=C_TIMING,
            alpha=0.85,
            hatch="///",
            label="Timing/sizing estimates: error bars show stated ranges",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=7.8, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
