#!/usr/bin/env python3
"""Generate paper figures from committed benchmark outputs.

Figures produced:
    Figure 1 — USDC basis during SVB crisis (time-series)
    Figure 2 — Price-only vs executable opportunity count (bar chart)
    Figure 3 — Spread and depth deterioration during stress (twin-axis)
    Figure 4 — Feature-set ablation AUROC across models (heatmap)
    Figure 5 — Oracle gap: oracle ceiling vs ML models (grouped bars)

Usage:
    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --data-dir data/gold --output-dir results/figures
    python scripts/make_paper_figures.py --format pdf   # default: png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_STRESS_START_NS = 1_678_406_400_000_000_000  # 2023-03-10T00:00:00Z
_STRESS_END_NS = 1_679_356_740_000_000_000  # 2023-03-20T23:59:00Z  (full test split)

_COLORS = {
    "price": "#2166ac",
    "exec_10k": "#d73027",
    "exec_50k": "#fc8d59",
    "spread": "#1a9641",
    "depth": "#a6d96a",
    "oracle": "#4d4d4d",
    "no_trade": "#bababa",
}


def _ns_to_dt(ns_series):
    import pandas as pd

    return pd.to_datetime(ns_series, unit="ns", utc=True)


def _savefig(fig, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 1: USDC basis during SVB crisis
# ---------------------------------------------------------------------------


def figure_1_usdc_basis(dataset_path: Path, output_dir: Path, fmt: str) -> None:
    """Time-series of USDC/USDT cross-quote basis during the full SVB test window."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    if not dataset_path.exists():
        logger.warning("dataset.parquet not found — skipping Figure 1.")
        return

    df = pl.read_parquet(str(dataset_path))
    stress = df.filter(pl.col("split") == "test").sort("ts_1m_ns")

    required = ["cross_quote_basis_usdc_bps", "cross_quote_basis_usdt_bps",
                "cross_quote_basis_maxabs_bps"]
    missing = [c for c in required if c not in stress.columns]
    if stress.is_empty() or missing:
        logger.warning("Missing columns for Figure 1: %s", missing)
        return

    dt = _ns_to_dt(stress["ts_1m_ns"].to_list())
    usdc = np.array(stress["cross_quote_basis_usdc_bps"].to_list(), dtype=float)
    usdt = np.array(stress["cross_quote_basis_usdt_bps"].to_list(), dtype=float)
    maxabs = np.array(stress["cross_quote_basis_maxabs_bps"].to_list(), dtype=float)

    n_outliers = int(np.sum(np.abs(usdc) > 200))

    # Clip values so matplotlib doesn't draw artifacts at the axis boundary
    clip = 200
    usdc_plot = np.where(np.abs(usdc) > clip, np.nan, usdc)
    usdt_plot = np.where(np.abs(usdt) > clip, np.nan, usdt)
    maxabs_plot = np.where(np.abs(maxabs) > clip, np.nan, maxabs)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(dt, usdc_plot, lw=0.7, color="#003057", label="USDC basis", zorder=3)
    ax.plot(dt, usdt_plot, lw=0.7, color="#d73027", label="USDT basis", zorder=2)
    ax.plot(dt, maxabs_plot, lw=0.9, color="#F2A900", label="Max-abs basis", zorder=1)
    ax.axhline(10, ls="--", lw=0.8, color="#F2A900", alpha=0.8, label="+10 bps threshold")
    ax.set_ylim(-200, 200)
    ax.set_xlabel("UTC date (SVB stress window, Mar 10–20 2023)")
    ax.set_ylabel("Basis (bps)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    if n_outliers > 0:
        ax.annotate(
            f"{n_outliers} windows beyond ±200 bps not shown",
            xy=(0.01, 0.04), xycoords="axes fraction",
            fontsize=7, color="gray", style="italic",
        )
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_1_usdc_basis_svb", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Price-only vs executable opportunity count
# ---------------------------------------------------------------------------


def figure_2_price_vs_exec(paper_dir: Path, output_dir: Path, fmt: str) -> None:
    """Bar chart comparing price-signal and executable-profit window counts."""
    import csv

    import matplotlib.pyplot as plt
    import numpy as np

    table_path = paper_dir / "table_2_price_execution_gap.csv"
    if not table_path.exists():
        logger.warning(
            "table_2 not found — run make_paper_tables.py first. Skipping Figure 2."
        )
        return

    with open(table_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]

    thresholds = [int(r["threshold_bps"]) for r in rows]
    price_pct = [
        float(r.get("price_pct_bps") or r.get("price_pct_usdc") or 0) for r in rows
    ]
    exec_pct = [float(r.get("exec_pct_q10000") or 0) for r in rows]

    x = np.arange(len(thresholds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x - width / 2,
        price_pct,
        width,
        label="Price signal (|basis| > thr)",
        color=_COLORS["price"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        exec_pct,
        width,
        label="Executable at $10K (net > thr)",
        color=_COLORS["exec_10k"],
        alpha=0.85,
    )

    for i, (p, e) in enumerate(zip(price_pct, exec_pct)):
        if p > 0:
            ratio = p / e if e > 0 else float("inf")
            ax.text(
                i,
                p + 0.5,
                f"{ratio:.0f}×",
                ha="center",
                fontsize=8,
                color=_COLORS["price"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f">{t} bps" for t in thresholds])
    ax.set_ylabel("% of 1-min windows (test split)")
    ax.set_title("Figure 2 — Price Signal vs Executable Opportunity (SVB Test Split)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_2_price_vs_exec", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Spread and depth deterioration
# ---------------------------------------------------------------------------


def figure_3_spread_depth(dataset_path: Path, output_dir: Path, fmt: str) -> None:
    """Twin-axis: bid-ask spread and book depth during the stress window."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import polars as pl

    if not dataset_path.exists():
        logger.warning("dataset.parquet not found — skipping Figure 3.")
        return

    df = pl.read_parquet(str(dataset_path))
    stress = df.filter(
        (pl.col("ts_1m_ns") >= _STRESS_START_NS)
        & (pl.col("ts_1m_ns") <= _STRESS_END_NS)
    ).sort("ts_1m_ns")

    if stress.is_empty():
        logger.warning("No stress-window data for Figure 3.")
        return

    dt = _ns_to_dt(stress["ts_1m_ns"].to_list())

    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax2 = ax1.twinx()

    if "spread_bps_mean" in stress.columns:
        ax1.plot(
            dt,
            stress["spread_bps_mean"].to_list(),
            lw=0.8,
            color=_COLORS["spread"],
            label="Bid-ask spread (bps)",
        )
        ax1.set_ylabel("Spread (bps)", color=_COLORS["spread"])

    if "depth_bid_10bp_mean" in stress.columns:
        ax2.plot(
            dt,
            stress["depth_bid_10bp_mean"].to_list(),
            lw=0.8,
            color=_COLORS["depth"],
            alpha=0.7,
            label="Bid depth within 10 bps (BTC)",
        )
        ax2.set_ylabel("Depth (BTC)", color=_COLORS["depth"])

    ax1.set_xlabel("UTC time")
    ax1.set_title("Figure 3 — Spread and Depth Deterioration During SVB Stress")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_3_spread_depth", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Feature-set ablation AUROC heatmap
# ---------------------------------------------------------------------------


def figure_4_ablation_heatmap(
    experiments_dir: Path, output_dir: Path, fmt: str
) -> None:
    """Heatmap of AUROC across (model × feature_set) for the primary task."""
    import csv

    import matplotlib.pyplot as plt
    import numpy as np

    all_results = experiments_dir / "all_results.csv"
    if not all_results.exists():
        logger.warning("all_results.csv not found — skipping Figure 4.")
        return

    with open(all_results) as fh:
        all_rows = list(csv.DictReader(fh))

    # Prefer USDC-specific primary task; fall back to generic for backward compat
    for primary_task in ("basis_usdc_1m_gt10bps", "basis_1m_gt10bps"):
        rows = [
            r for r in all_rows if r["task"] == primary_task and r["model"] != "oracle"
        ]
        if rows:
            break

    if not rows:
        logger.warning("No basis_*_1m_gt10bps results — skipping Figure 4.")
        return

    models = sorted({r["model"] for r in rows})
    feat_sets = sorted({r["feature_set"] for r in rows})

    data = np.full((len(models), len(feat_sets)), float("nan"))
    for r in rows:
        mi = models.index(r["model"])
        fi = feat_sets.index(r["feature_set"])
        try:
            data[mi, fi] = float(r["auroc"])
        except (ValueError, KeyError):
            pass

    fig, ax = plt.subplots(figsize=(6, max(3, len(models) * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.80)
    plt.colorbar(im, ax=ax, label="AUROC")
    ax.set_xticks(range(len(feat_sets)))
    ax.set_xticklabels(feat_sets, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    task_label = rows[0]["task"] if rows else "basis_usdc_1m_gt10bps"
    ax.set_title(f"Figure 4 — AUROC Ablation: {task_label}")
    for i in range(len(models)):
        for j in range(len(feat_sets)):
            v = data[i, j]
            if v == v:
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < 0.55 or v > 0.75 else "black",
                )
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_4_ablation_heatmap", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Oracle gap grouped bars
# ---------------------------------------------------------------------------


def figure_5_oracle_gap(experiments_dir: Path, output_dir: Path, fmt: str) -> None:
    """Grouped bars: oracle ceiling vs best ML model net bps per task."""
    import csv

    import matplotlib.pyplot as plt
    import numpy as np

    table_path = experiments_dir.parent / "paper" / "table_4_oracle_gap.csv"
    if not table_path.exists():
        logger.warning("table_4_oracle_gap.csv not found — skipping Figure 5.")
        return

    with open(table_path) as fh:
        rows = list(csv.DictReader(fh))

    tasks = [r["task"].replace("_", "\n") for r in rows]
    oracle = [float(r["oracle_net_bps"]) for r in rows]
    best = [
        float(r["best_model_net_bps"]) if r["best_model_net_bps"] else float("nan")
        for r in rows
    ]

    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x - width / 2,
        oracle,
        width,
        label="Oracle upper bound",
        color=_COLORS["oracle"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        best,
        width,
        label="Best ML model",
        color=_COLORS["price"],
        alpha=0.85,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylabel("Net bps (test split)")
    ax.set_title("Figure 5 — Oracle Ceiling vs Best ML Model (SVB Test Split)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, output_dir / "figure_5_oracle_gap", fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--experiments-dir", default="results/experiments")
    parser.add_argument("--paper-dir", default="results/paper")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    experiments_dir = Path(args.experiments_dir)
    paper_dir = Path(args.paper_dir)
    output_dir = Path(args.output_dir)
    fmt = args.format

    dataset_path = data_dir / "dataset.parquet"

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        logger.error("matplotlib not installed — pip install matplotlib")
        return

    logger.info("Generating paper figures → %s", output_dir)
    figure_1_usdc_basis(dataset_path, output_dir, fmt)
    figure_2_price_vs_exec(paper_dir, output_dir, fmt)
    figure_3_spread_depth(dataset_path, output_dir, fmt)
    figure_4_ablation_heatmap(experiments_dir, output_dir, fmt)
    figure_5_oracle_gap(experiments_dir, output_dir, fmt)
    logger.info("Figures complete.")


if __name__ == "__main__":
    main()
