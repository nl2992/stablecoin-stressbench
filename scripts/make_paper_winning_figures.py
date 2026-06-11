#!/usr/bin/env python3
"""Generate publication-quality figures for the winning-tier paper additions.

Produces:
  1. figure_auroc_pnl_scatter.png  — AUROC vs net bps (all calm-trained models)
  2. figure_shap_top10.png         — Top-10 SHAP importance bar chart
  3. (copies both to results/paper_addon/figures/ and results/paper/figures/)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
FIG_DIR = ROOT / "results" / "paper_addon" / "figures"
PAPER_FIG_DIR = ROOT / "results" / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Columbia palette ────────────────────────────────────────────────────────
NAVY = "#003057"
BLUE = "#75B2DD"
GOLD = "#F2A900"
RED = "#C4122F"
GREY = "#9E9E9E"

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


# ── 1. AUROC vs net-bps scatter ─────────────────────────────────────────────


def make_auroc_scatter() -> None:
    """AUROC vs net bps for all calm-trained models + oracle + meta-labeling."""

    # Data from paper (Table 5 ablation + text)
    models = [
        # label, auroc, net_bps, marker, colour, size
        ("Price rule\n(10 bps)", 0.833, -269.5, "o", RED, 70),
        ("Price rule\n(25 bps)", 0.746, -347.3, "o", RED, 70),
        ("Gross arb", 0.531, -136.0, "o", RED, 70),
        ("Logistic", 0.133, -49.1, "s", GREY, 60),
        ("LightGBM", 0.653, -125.3, "s", GREY, 60),
        ("XGBoost", 0.756, -102.1, "s", GREY, 60),
        ("RF", 0.840, 0.0, "s", GREY, 60),  # 0 trades → NoTrade
        ("GRU\n(seq, AUROC 0.80)", 0.800, -239.0, "^", BLUE, 90),
        ("MLP-Window", 0.652, -341.0, "^", BLUE, 90),
        ("Meta-label\n(Terra→SVB)", 0.000, 82.5, "D", GOLD, 110),  # no AUROC concept
        ("Oracle\n(hindsight)", 0.519, 162.2, "*", NAVY, 140),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for label, auroc, net_bps, marker, colour, size in models:
        ax.scatter(
            auroc,
            net_bps,
            marker=marker,
            color=colour,
            s=size,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate key points
    ax.annotate(
        "GRU\n(AUROC 0.80,\n−239 bps)",
        xy=(0.800, -239.0),
        xytext=(0.58, -200),
        fontsize=6,
        color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
    )

    ax.annotate(
        "Meta-label\n+82.5 bps",
        xy=(0.0, 82.5),
        xytext=(0.12, 110),
        fontsize=6,
        color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
    )

    ax.annotate(
        "Oracle\n+162 bps",
        xy=(0.519, 162.2),
        xytext=(0.30, 150),
        fontsize=6,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8),
    )

    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("AUROC (calm-trained)", fontsize=8)
    ax.set_ylabel("Net bps per triggered trade", fontsize=8)
    ax.set_title(
        "Discrimination ≠ Profitability Under Class Imbalance", fontsize=8, pad=6
    )
    ax.tick_params(labelsize=7)

    legend_elements = [
        mpatches.Patch(color=RED, label="Price rules"),
        mpatches.Patch(color=GREY, label="Tabular ML"),
        mpatches.Patch(color=BLUE, label="Sequence models"),
        mpatches.Patch(color=GOLD, label="Meta-labeling"),
        mpatches.Patch(color=NAVY, label="Oracle ceiling"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=6,
        loc="lower left",
        framealpha=0.8,
        edgecolor="none",
    )

    fig.tight_layout(pad=0.5)
    out = FIG_DIR / "figure_auroc_pnl_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(
        PAPER_FIG_DIR / "figure_auroc_pnl_scatter.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. SHAP Top-10 bar chart ────────────────────────────────────────────────


def make_shap_top10() -> None:
    """Horizontal bar chart of top-10 SHAP features coloured by group."""

    shap_path = ROOT / "results" / "paper_addon" / "table_shap_importance.csv"
    df = pd.read_csv(shap_path)
    total = df["mean_abs_shap"].sum()
    df["pct"] = df["mean_abs_shap"] / total * 100

    top10 = df.head(10).copy()

    colour_map = {
        "price": NAVY,
        "book / frag": BLUE,
        "settle": GOLD,
    }
    colours = [colour_map.get(c, GREY) for c in top10["category"]]

    # Clean labels
    label_map = {
        "cross_quote_basis_usdc_bps": "USDC basis",
        "cross_quote_basis_primary_bps": "Primary basis",
        "deviation_from_1_usd_bps": "Deviation from \$1",
        "cross_quote_basis_usdt_bps": "USDT basis",
        "depth_ask_10bp_mean": "Ask depth (10bp)",
        "spread_bps_mean": "Bid-ask spread",
        "transfer_volume_1m": "Transfer volume",
        "imbalance_1bp_mean": "Order imbalance",
        "trade_volume_1m_total": "Trade volume",
        "depth_bid_10bp_mean": "Bid depth (10bp)",
    }
    labels = [label_map.get(f, f) for f in top10["feature"]]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    y = np.arange(len(top10))
    bars = ax.barh(
        y, top10["pct"].values, color=colours, height=0.65, edgecolor="white"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels[::-1] if False else labels, fontsize=7)
    ax.invert_yaxis()  # highest importance at top

    for bar, val in zip(bars, top10["pct"].values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=6.5,
        )

    ax.set_xlabel("SHAP importance (%)", fontsize=8)
    ax.set_title(
        "Feature Attribution: Meta-Labeling Secondary Classifier", fontsize=8, pad=5
    )
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, max(top10["pct"]) * 1.18)

    # Group legend
    legend_elements = [
        mpatches.Patch(color=NAVY, label=f"Price/basis (94.8%)"),
        mpatches.Patch(color=BLUE, label=f"Book/fragmentation (3.4%)"),
        mpatches.Patch(color=GOLD, label=f"Settlement (1.8%)"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=6,
        loc="lower right",
        framealpha=0.85,
        edgecolor="none",
    )

    fig.tight_layout(pad=0.5)
    out = FIG_DIR / "figure_shap_top10.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "figure_shap_top10.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_auroc_scatter()
    make_shap_top10()
    print("Done.")
