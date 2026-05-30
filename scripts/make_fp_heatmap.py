#!/usr/bin/env python3
"""Generate false-positive structural diagnosis profile chart.

Creates results/paper_addon/figures/figure_fp_profile.png — a grouped bar chart
comparing TP vs FP vs FN vs TN prediction groups on three key economic dimensions:
  1. Average |basis_usdc| (magnitude of the USDC cross-quote basis)
  2. Average bid depth at 10 bp ($)
  3. Average net profit (bps, $10K notional)

Data source: results/paper_addon/table_5_false_positive_diagnosis.csv

Key finding is printed to stdout.

Usage
-----
    python scripts/make_fp_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Hardcoded values from table_5_false_positive_diagnosis.csv
# (sourced from the CSV so the script also works when the CSV is absent)
# ---------------------------------------------------------------------------

_CSV_PATH = Path("results/paper_addon/table_5_false_positive_diagnosis.csv")
_OUT_PATH = Path("results/paper_addon/figures/figure_fp_profile.png")


def load_fp_data(csv_path: Path) -> dict[str, dict]:
    """Load false-positive diagnosis data from CSV, with inline fallback."""
    # Fallback values from the task specification
    _FALLBACK: dict[str, dict] = {
        "TP": {
            "n": 1421,
            "avg_basis_usdc": -344.0859,
            "avg_depth_bid": 59_961.0678,
            "avg_net_profit": -35.1513,   # net_profit_bps_q10000
        },
        "FP": {
            "n": 581,
            "avg_basis_usdc": 0.5597,
            "avg_depth_bid": 72_805.4624,
            "avg_net_profit": -38.7411,
        },
        "FN": {
            "n": 583,
            "avg_basis_usdc": -0.5275,
            "avg_depth_bid": 71_634.8858,
            "avg_net_profit": -41.8483,
        },
        "TN": {
            "n": 10_672,
            "avg_basis_usdc": -0.4309,
            "avg_depth_bid": 70_672.8404,
            "avg_net_profit": -83.0179,
        },
    }

    if not csv_path.exists():
        print(f"[data] {csv_path} not found — using hardcoded values from paper.")
        return _FALLBACK

    try:
        import csv as _csv

        groups: dict[str, dict] = {}
        with open(csv_path, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                grp = row["group"].strip()
                groups[grp] = {
                    "n": int(row["n"]),
                    "avg_basis_usdc": float(row["avg_cross_quote_basis_usdc_bps"]),
                    "avg_depth_bid": float(row["avg_depth_bid_10bp_mean"]),
                    "avg_net_profit": float(row["avg_net_profit_bps_q10000"]),
                }
        print(f"[data] Loaded {len(groups)} groups from {csv_path}")
        return groups
    except Exception as exc:
        print(f"[data] Failed to parse CSV ({exc}) — using hardcoded values.")
        return _FALLBACK


def build_finding(groups: dict[str, dict]) -> str:
    """Derive the key finding narrative from the group statistics."""
    tp = groups["TP"]
    fp = groups["FP"]

    basis_tp = abs(tp["avg_basis_usdc"])
    basis_fp = abs(fp["avg_basis_usdc"])
    depth_tp = tp["avg_depth_bid"]
    depth_fp = fp["avg_depth_bid"]
    profit_tp = tp["avg_net_profit"]
    profit_fp = fp["avg_net_profit"]

    finding = (
        f"KEY FINDING — False-Positive Structural Profile\n"
        f"{'='*60}\n"
        f"TP predictions (n={tp['n']:,}) are characterised by:\n"
        f"  |basis_usdc|  = {basis_tp:>8.1f} bps  (vs FP: {basis_fp:.1f} bps,  ratio {basis_tp/max(basis_fp,0.01):.0f}x)\n"
        f"  bid depth     = {depth_tp:>8,.0f} $   (vs FP: {depth_fp:,.0f} $,  {(depth_fp-depth_tp)/depth_tp*100:+.1f}%)\n"
        f"  net profit    = {profit_tp:>8.1f} bps (vs FP: {profit_fp:.1f} bps)\n"
        f"\n"
        f"False positives (n={fp['n']:,}) cluster in low-basis, high-depth regimes:\n"
        f"  The basis magnitude is {basis_tp/max(basis_fp,0.01):.0f}x LOWER than true positives,\n"
        f"  confirming that FP trades occur when the model fires on liquidity noise\n"
        f"  rather than genuine cross-quote dislocations.\n"
        f"  FP bid depth exceeds TP by {(depth_fp-depth_tp)/depth_tp*100:.1f}%, suggesting the model\n"
        f"  conflates deep, liquid markets with arb-rich markets.\n"
        f"{'='*60}"
    )
    return finding


def make_profile_chart(groups: dict[str, dict], out_path: Path) -> None:
    """Create a grouped bar chart of TP/FP/FN/TN on three dimensions."""
    group_keys = ["TP", "FP", "FN", "TN"]
    group_labels = [
        f"TP\n(n={groups['TP']['n']:,})",
        f"FP\n(n={groups['FP']['n']:,})",
        f"FN\n(n={groups['FN']['n']:,})",
        f"TN\n(n={groups['TN']['n']:,})",
    ]

    # -----------------------------------------------------------------------
    # Three panels:  basis magnitude  |  bid depth  |  net profit
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7, 4))
    fig.suptitle(
        "False-Positive Structural Diagnosis: TP/FP/FN/TN Profile\n"
        r"$\mathit{price\_threshold\_10bps}$ — $\mathit{label\_basis\_usdc\_1m\_gt10bps}$",
        fontsize=9,
        y=1.02,
    )

    # Colour palette: TP=navy, FP=crimson, FN=darkorange, TN=steelblue
    palette = {
        "TP": "#1f4e79",
        "FP": "#c00000",
        "FN": "#e07b00",
        "TN": "#2e74b5",
    }
    colors = [palette[g] for g in group_keys]

    x = np.arange(len(group_keys))
    bar_width = 0.55

    # ------------------------------------------------------------------
    # Panel 1: |basis_usdc| magnitude
    # ------------------------------------------------------------------
    ax1 = axes[0]
    vals1 = [abs(groups[g]["avg_basis_usdc"]) for g in group_keys]
    bars1 = ax1.bar(x, vals1, width=bar_width, color=colors, edgecolor="white", linewidth=0.6)
    ax1.set_title("Basis Magnitude\n|avg basis_usdc| (bps)", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels, fontsize=7)
    ax1.set_ylabel("bps", fontsize=8)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax1.tick_params(axis="y", labelsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax1.set_axisbelow(True)
    # Annotate bars
    for bar, v in zip(bars1, vals1):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals1) * 0.015,
            f"{v:.0f}",
            ha="center", va="bottom", fontsize=6.5,
        )

    # ------------------------------------------------------------------
    # Panel 2: Bid depth (divide by 1000 for readability → k$)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    vals2 = [groups[g]["avg_depth_bid"] / 1_000 for g in group_keys]
    bars2 = ax2.bar(x, vals2, width=bar_width, color=colors, edgecolor="white", linewidth=0.6)
    ax2.set_title("Avg Bid Depth\nat 10 bp ($k)", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(group_labels, fontsize=7)
    ax2.set_ylabel("$k", fontsize=8)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}k"))
    ax2.tick_params(axis="y", labelsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax2.set_axisbelow(True)
    # Set y-axis to start near the minimum for visibility
    y2_min = min(vals2) * 0.97
    y2_max = max(vals2) * 1.04
    ax2.set_ylim(y2_min, y2_max)
    for bar, v in zip(bars2, vals2):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (y2_max - y2_min) * 0.01,
            f"{v:.0f}k",
            ha="center", va="bottom", fontsize=6.5,
        )

    # ------------------------------------------------------------------
    # Panel 3: Net profit bps (q10000)
    # ------------------------------------------------------------------
    ax3 = axes[2]
    vals3 = [groups[g]["avg_net_profit"] for g in group_keys]
    # Use diverging colour: negative=red-tinted, positive=green-tinted
    bar_colors3 = [
        "#c00000" if v < 0 else "#1a9641"
        for v in vals3
    ]
    bars3 = ax3.bar(x, vals3, width=bar_width, color=bar_colors3, edgecolor="white", linewidth=0.6)
    ax3.axhline(0, color="black", linewidth=0.6, linestyle="-")
    ax3.set_title("Avg Net Profit\n(bps, $10K notional)", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(group_labels, fontsize=7)
    ax3.set_ylabel("bps", fontsize=8)
    ax3.tick_params(axis="y", labelsize=7)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax3.set_axisbelow(True)
    v_range = max(vals3) - min(vals3)
    for bar, v in zip(bars3, vals3):
        offset = v_range * 0.02 if v >= 0 else -v_range * 0.04
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            v + offset,
            f"{v:.1f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=6.5,
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=palette["TP"], label="TP — True Positive"),
        Patch(color=palette["FP"], label="FP — False Positive"),
        Patch(color=palette["FN"], label="FN — False Negative"),
        Patch(color=palette["TN"], label="TN — True Negative"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=7,
        framealpha=0.85,
        edgecolor="lightgrey",
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved → {out_path}")


def main() -> None:
    groups = load_fp_data(_CSV_PATH)

    # Build and print key finding
    finding = build_finding(groups)
    print()
    print(finding)
    print()

    # Generate the profile chart
    make_profile_chart(groups, _OUT_PATH)

    print(f"\nDone. Figure written to {_OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
