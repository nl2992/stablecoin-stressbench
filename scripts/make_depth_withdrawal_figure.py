#!/usr/bin/env python3
"""Figure C: Depth withdrawal signature — three real event panels.

Panels (all from gold dataset, real L2 data):
  1. Terra/LUNA May 2022        — algorithmic collapse
  2. SVB Stress   Mar 10-14 2023 — fiat-reserve shock (stress phase)
  3. SVB Recovery Mar 15-20 2023 — post-peg recovery (zero executable windows)

Each panel shows mean ask-side depth (blue) and bid-ask spread (orange)
per event-hour for primary-signal windows (|basis_usdc| > 10 bps),
normalised to the first-hour value so magnitudes are comparable.

Requires: data/gold/dataset.parquet

Output: results/paper_addon/figures/figure_depth_withdrawal.png
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO   = Path(__file__).parent.parent
PARQ   = REPO / "data" / "gold" / "dataset.parquet"
OUT    = REPO / "results" / "paper_addon" / "figures" / "figure_depth_withdrawal.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

C_DEPTH  = "#1f77b4"   # blue
C_SPREAD = "#d62728"   # red

BASIS_COL  = "cross_quote_basis_usdc_bps"
DEPTH_COL  = "depth_ask_10bp_mean"
SPREAD_COL = "spread_bps_mean"
TS_COL     = "ts_1m_ns"
SPLIT_COL  = "split"

PRIMARY_THRESHOLD = 10.0   # bps


def _iso_ns(iso: str) -> int:
    dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000_000_000)


PANELS = [
    {
        "split":  "validation",
        "t_start": _iso_ns("2022-05-07T00:00:00Z"),
        "t_end":   _iso_ns("2022-05-15T00:00:00Z"),
        "label":   "Terra/LUNA May 2022  (algorithmic)",
        "exec_note": "2.40% executable",
    },
    {
        "split":  "test",
        "t_start": _iso_ns("2023-03-10T00:00:00Z"),
        "t_end":   _iso_ns("2023-03-15T00:00:00Z"),
        "label":   "USDC/SVB Mar 10–14 2023  (fiat-reserve stress)",
        "exec_note": "6.63% executable",
    },
]


def _panel_data(
    df,           # polars DataFrame
    panel: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hours, norm_depth, norm_spread) for the panel."""
    import polars as pl

    t0, t1 = panel["t_start"], panel["t_end"]
    sub = df.filter(
        (pl.col(SPLIT_COL) == panel["split"])
        & (pl.col(TS_COL) >= t0)
        & (pl.col(TS_COL) < t1)
        & (pl.col(BASIS_COL).abs() > PRIMARY_THRESHOLD)
    )
    if sub.is_empty():
        return np.array([]), np.array([]), np.array([])

    hour = ((sub[TS_COL] - t0) / (3_600 * 1_000_000_000)).cast(pl.Int64).alias("hour")
    with_h = sub.with_columns(hour)
    by_h = (
        with_h.group_by("hour")
        .agg([
            pl.col(DEPTH_COL).mean().alias("depth"),
            pl.col(SPREAD_COL).mean().alias("spread"),
        ])
        .sort("hour")
    )

    hours   = by_h["hour"].to_numpy().astype(float)
    depth   = by_h["depth"].to_numpy().astype(float)
    spread  = by_h["spread"].to_numpy().astype(float)

    # Normalise to first-hour value (where non-NaN)
    d0 = depth[~np.isnan(depth)][0]  if (~np.isnan(depth)).any()  else 1.0
    s0 = spread[~np.isnan(spread)][0] if (~np.isnan(spread)).any() else 1.0
    if d0 > 0:
        depth  = depth  / d0
    if s0 > 0:
        spread = spread / s0

    return hours, depth, spread


def main() -> None:
    try:
        import polars as pl
    except ImportError:
        print("polars not available; skipping figure_depth_withdrawal.py")
        return

    if not PARQ.exists():
        print(f"Gold dataset not found at {PARQ}; skipping.")
        return

    df = pl.read_parquet(str(PARQ))
    for col in [BASIS_COL, DEPTH_COL, SPREAD_COL, TS_COL, SPLIT_COL]:
        if col not in df.columns:
            print(f"Missing column {col!r}; skipping figure_depth_withdrawal.")
            return

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 3.2),
                             gridspec_kw={"hspace": 0.72})

    for ax, panel in zip(axes, PANELS):
        hours, d_norm, s_norm = _panel_data(df, panel)

        ax2 = ax.twinx()

        if len(hours) > 0:
            ax.plot(hours, d_norm,  color=C_DEPTH,  linewidth=1.6,
                    label="Ask depth (norm.)")
            ax.fill_between(hours, 1.0, d_norm, where=d_norm < 1.0,
                            color=C_DEPTH, alpha=0.15)
            ax2.plot(hours, s_norm, color=C_SPREAD, linewidth=1.6,
                     linestyle="--", label="Spread (norm.)")
            ax.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle=":")
        else:
            ax.text(0.5, 0.5, "No primary-signal data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_ylabel("Ask depth\n(norm. to hr 0)", fontsize=8.5, color=C_DEPTH)
        ax2.set_ylabel("Spread\n(norm. to hr 0)", fontsize=8.5, color=C_SPREAD)
        ax.yaxis.set_tick_params(labelsize=8, labelcolor=C_DEPTH)
        ax2.yaxis.set_tick_params(labelsize=8, labelcolor=C_SPREAD)
        ax.set_xlabel("Hours from event start", fontsize=8.5)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # Title = event label + exec note
        ax.set_title(
            f"{panel['label']}  [{panel['exec_note']}]",
            fontsize=8.8, pad=3
        )

        # Single legend
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, labs1 + labs2,
                      fontsize=7.5, loc="upper right",
                      framealpha=0.85, borderpad=0.4)

    fig.suptitle(
        "Depth withdrawal signature in primary-signal windows\n"
        "(all panels: real L2 data from gold dataset)",
        fontsize=9.5, y=1.01,
    )

    fig.savefig(str(OUT), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
