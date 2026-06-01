#!/usr/bin/env python3
"""Figure C: Depth withdrawal signature — Terra/LUNA and SVB.

Two vertically stacked panels showing ask-side depth (blue) and bid-ask spread
(orange) in primary-signal windows (|basis_usdc| > 10 bps), aggregated per
hour of event window.  Both events show depth collapsing and spread widening
despite different underlying triggers.

Uses the gold dataset (data/gold/dataset.parquet).

Output: results/paper_addon/figures/figure_depth_withdrawal.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).parent.parent
PARQUET = REPO / "data" / "gold" / "dataset.parquet"
OUT = REPO / "results" / "paper_addon" / "figures" / "figure_depth_withdrawal.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

_BASIS_COL  = "cross_quote_basis_usdc_bps"
_DEPTH_COL  = "depth_ask_10bp_mean"
_SPREAD_COL = "spread_bps_mean"
_TS_COL     = "ts_1m_ns"
_PRIMARY_THRESHOLD = 10.0

PANELS = [
    ("validation", "Terra/LUNA  May 2022  (algorithmic)"),
    ("test",       "USDC/SVB   Mar 2023  (fiat-reserve)"),
]

C_DEPTH  = "#1f77b4"   # blue
C_SPREAD = "#d62728"   # red
C_FILL   = "#aec7e8"


def load_panel(df, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hour_idx, mean_depth, mean_spread) for primary-signal windows."""
    sub = df.filter(
        (df["split"] == split) & (df[_BASIS_COL].abs() > _PRIMARY_THRESHOLD)
    )
    if sub.is_empty():
        return np.array([]), np.array([]), np.array([])

    ts_ns = sub[_TS_COL].to_numpy()
    depth  = sub[_DEPTH_COL].to_numpy() if _DEPTH_COL in sub.columns else np.zeros(len(sub))
    spread = sub[_SPREAD_COL].to_numpy() if _SPREAD_COL in sub.columns else np.zeros(len(sub))

    # Hour index relative to event start
    ts_min = ts_ns.min()
    hour_idx = ((ts_ns - ts_min) / (3_600 * 1e9)).astype(int)

    unique_hours = np.unique(hour_idx)
    d_mean = np.array([depth[hour_idx == h].mean() for h in unique_hours])
    s_mean = np.array([spread[hour_idx == h].mean() for h in unique_hours])

    # Normalise: divide by first-hour value so both panels are comparable
    if len(d_mean) > 0 and d_mean[0] > 0:
        d_mean = d_mean / d_mean[0]
    if len(s_mean) > 0 and s_mean[0] > 0:
        s_mean = s_mean / s_mean[0]

    return unique_hours, d_mean, s_mean


def main() -> None:
    try:
        import polars as pl
    except ImportError:
        print("polars not available; skipping figure_depth_withdrawal.py")
        return

    if not PARQUET.exists():
        print(f"Gold dataset not found at {PARQUET}; skipping.")
        return

    df = pl.read_parquet(str(PARQUET))
    missing = [c for c in [_BASIS_COL, _DEPTH_COL, _SPREAD_COL, _TS_COL, "split"]
               if c not in df.columns]
    if missing:
        print(f"Missing columns {missing}; skipping figure_depth_withdrawal.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.2), sharex=False,
                             gridspec_kw={"hspace": 0.52})

    for ax, (split, title) in zip(axes, PANELS):
        hours, d_norm, s_norm = load_panel(df, split)
        if len(hours) == 0:
            ax.text(0.5, 0.5, f"No primary-signal data\nfor split={split!r}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=8.5, pad=3)
            continue

        ax2 = ax.twinx()
        ax.plot(hours, d_norm, color=C_DEPTH, linewidth=1.4, label="Ask depth (norm.)")
        ax.fill_between(hours, 1.0, d_norm, where=d_norm < 1.0,
                        color=C_FILL, alpha=0.4)
        ax2.plot(hours, s_norm, color=C_SPREAD, linewidth=1.4,
                 linestyle="--", label="Spread (norm.)")

        ax.axhline(1.0, color="#aaaaaa", linewidth=0.7, linestyle=":")
        ax.set_ylabel("Ask depth\n(norm. to hr 0)", fontsize=7.5, color=C_DEPTH)
        ax2.set_ylabel("Spread\n(norm. to hr 0)", fontsize=7.5, color=C_SPREAD)
        ax.yaxis.set_tick_params(labelsize=7, labelcolor=C_DEPTH)
        ax2.yaxis.set_tick_params(labelsize=7, labelcolor=C_SPREAD)
        ax.set_xlabel("Hours from event start", fontsize=7.5)
        ax.xaxis.set_tick_params(labelsize=7)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # Inline legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5,
                  loc="upper right", framealpha=0.8, borderpad=0.4)

    fig.suptitle("Depth withdrawal signature in primary-signal windows\n"
                 "(ask depth ↓, spread ↑ — consistent across mechanism classes)",
                 fontsize=8.5, y=1.01)
    plt.tight_layout(pad=0.7)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
