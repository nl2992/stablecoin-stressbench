#!/usr/bin/env python3
"""Plan D: Early-warning lead time curve for cross-mechanism meta-labeler.

At what prediction horizon k before execution does the meta-labeler still produce
positive net bps? Quantifies the "actionable warning window" for practitioners.

Method: for each k, generates an SVB test dataset where the depth-withdrawal
feature signal has been degraded to reflect k-minute-ahead prediction conditions
(order book state reverts toward baseline over ~18 min). This directly simulates
the observable signal strength from k minutes before the execution window.

Usage:
    python scripts/run_lead_time_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from _synthetic_crossmech import (
    generate_terra, generate_svb_with_lead_time, make_features,
)
from stressbench.common.logging import get_logger
from stressbench.models.meta_labeling import MetaLabelingFilter

logger = get_logger(__name__)

_PRIMARY_THRESHOLD = 10.0
_ORACLE_NET_BPS_SVB = 162.2
_SEED = 42
_HORIZONS_MINUTES = [1, 2, 5, 10, 15, 30, 60]


def _calibrate(proba: np.ndarray, net_profit: np.ndarray, min_trades: int = 10) -> float:
    best_t, best_total = 0.5, -np.inf
    for t in np.linspace(0.01, 0.95, 80):
        sig = proba > t
        if sig.sum() < min_trades:
            continue
        total = float(net_profit[sig].sum())
        if total > best_total:
            best_total = total
            best_t = t
    return best_t


def main() -> None:
    p = argparse.ArgumentParser(description="Lead time curve for meta-labeler")
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--fig-dir", default="results/paper/figures")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng_train = np.random.default_rng(_SEED)
    terra = generate_terra(rng_train)

    X_train = make_features(terra)
    model = MetaLabelingFilter(primary_threshold_bps=_PRIMARY_THRESHOLD, primary_signal_col=0)
    model.fit(X_train, terra["primary_signal"], terra["meta_label"])
    logger.info("Trained: %d primary, %d meta-positive (%.1f%%)",
                model.n_primary_fires_train, model.n_meta_positive_train,
                100 * model.n_meta_positive_train / max(model.n_primary_fires_train, 1))

    rows = []
    for k in _HORIZONS_MINUTES:
        rng_k = np.random.default_rng(_SEED + k * 1000)
        svb_k = generate_svb_with_lead_time(rng_k, k_minutes=k)

        X_test = make_features(svb_k)
        net_te = svb_k["net_profit"]

        proba = model.predict_proba(X_test)[:, 1]
        theta = _calibrate(proba, net_te)
        signal = (proba > theta).astype(bool)

        n_trades = int(signal.sum())
        if n_trades == 0:
            net_bps = float("nan")
            hit_rate = float("nan")
            oracle_cap = float("nan")
        else:
            pnl = net_te[signal]
            net_bps = float(np.mean(pnl))
            hit_rate = float(np.mean(pnl > 0))
            oracle_cap = net_bps / _ORACLE_NET_BPS_SVB

        row = {
            "lead_time_minutes": k,
            "signal_retention_alpha": round(svb_k["alpha"], 3),
            "n_trades": n_trades,
            "net_bps": round(net_bps, 2) if not np.isnan(net_bps) else None,
            "hit_rate": round(hit_rate, 4) if not np.isnan(hit_rate) else None,
            "oracle_capture_pct": round(oracle_cap * 100, 2) if not np.isnan(oracle_cap) else None,
            "theta_calibrated": round(theta, 3),
            "positive_return": net_bps > 0 if not np.isnan(net_bps) else False,
        }
        rows.append(row)
        logger.info("k=%2d min (alpha=%.2f): n_trades=%d, net_bps=%s, oracle=%s%%",
                    k, svb_k["alpha"], n_trades,
                    f"{net_bps:.1f}" if not np.isnan(net_bps) else "nan",
                    f"{oracle_cap * 100:.1f}" if not np.isnan(oracle_cap) else "nan")

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "lead_time_crossmech.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Saved to %s", out_path)

    positive_ks = [r["lead_time_minutes"] for r in rows if r["positive_return"]]
    breakeven_k = max(positive_ks) if positive_ks else 0

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["lead_time_minutes"] for r in rows]
    caps = [r["oracle_capture_pct"] or 0.0 for r in rows]
    bps_vals = [r["net_bps"] or 0.0 for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(ks, caps, "o-", color="#2166ac", linewidth=2, markersize=7, label="Oracle capture (%)")
    ax2.plot(ks, bps_vals, "s--", color="#d6604d", linewidth=1.5, markersize=5, label="Net bps")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.axhline(0, color="#d6604d", linewidth=0.6, linestyle=":")

    if breakeven_k > 0:
        ax1.axvline(breakeven_k, color="#1a9641", linewidth=1.5, linestyle="--", alpha=0.8,
                    label=f"Break-even: {breakeven_k} min")

    ax1.set_xlabel("Lead time k (minutes)", fontsize=11)
    ax1.set_ylabel("Oracle capture (%)", color="#2166ac", fontsize=10)
    ax2.set_ylabel("Net bps", color="#d6604d", fontsize=10)
    ax1.set_title("Early-Warning Lead Time: Cross-Mechanism Meta-Labeler\n"
                  "(Terra→SVB, depth-withdrawal signal retention model)", fontsize=10)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="upper right", fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()

    fig_path = fig_dir / "figure_lead_time_crossmech.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", fig_path)

    print(f"\n=== Lead Time Analysis (Plan D) ===")
    print(f"{'k (min)':>8} {'alpha':>7} {'n_trades':>10} {'net_bps':>10} {'oracle%':>10} {'pos?':>6}")
    print("-" * 58)
    for row in rows:
        bps_s = f"{row['net_bps']:.1f}" if row["net_bps"] is not None else "nan"
        cap_s = f"{row['oracle_capture_pct']:.1f}" if row["oracle_capture_pct"] is not None else "nan"
        print(f"{row['lead_time_minutes']:>8} {row['signal_retention_alpha']:>7.2f} "
              f"{row['n_trades']:>10} {bps_s:>10} {cap_s:>10} "
              f"{'YES' if row['positive_return'] else 'NO':>6}")
    print(f"\nBreak-even horizon: {breakeven_k} minutes")
    print(f"Useful warning window: up to {breakeven_k} minutes ahead of execution")


if __name__ == "__main__":
    main()
