#!/usr/bin/env python3
"""Plan E: Calibration curve / reliability diagram for cross-mechanism meta-labeler.

Shows the meta-labeler is well-calibrated on the SVB test split.
Computes ECE and applies isotonic regression for post-hoc calibration comparison.

Usage:
    python scripts/run_calibration_curve.py
    python scripts/run_calibration_curve.py --output-dir results/experiments_addon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from _synthetic_crossmech import generate_terra, generate_svb, make_features
from stressbench.common.logging import get_logger
from stressbench.models.meta_labeling import MetaLabelingFilter

logger = get_logger(__name__)

_PRIMARY_THRESHOLD = 10.0
_SEED = 42
_N_BINS = 10


def _ece(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(proba)
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(proba[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _calibration_bins(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> list:
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (proba >= lo) & (proba < hi)
        n_bin = int(mask.sum())
        result.append({
            "bin_low": round(lo, 2), "bin_high": round(hi, 2),
            "n": n_bin,
            "mean_predicted_prob": round(float(np.mean(proba[mask])), 4) if n_bin > 0 else None,
            "actual_profitable_rate": round(float(np.mean(y_true[mask])), 4) if n_bin > 0 else None,
        })
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration curve for meta-labeler")
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--fig-dir", default="results/paper/figures")
    args = p.parse_args()

    from sklearn.isotonic import IsotonicRegression

    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(_SEED)
    terra = generate_terra(rng)
    svb = generate_svb(rng)

    X_train = make_features(terra)
    model = MetaLabelingFilter(primary_threshold_bps=_PRIMARY_THRESHOLD, primary_signal_col=0)
    model.fit(X_train, terra["primary_signal"], terra["meta_label"])
    logger.info("Trained: %d primary, %d meta-positive",
                model.n_primary_fires_train, model.n_meta_positive_train)

    X_test = make_features(svb)
    y_prim_svb = svb["primary_signal"]
    prim_mask = y_prim_svb.astype(bool)
    X_fires = X_test[prim_mask]
    y_true_fires = (svb["net_profit"][prim_mask] > 0).astype(int)

    proba_all = model.predict_proba(X_test)[:, 1]
    proba_fires = proba_all[prim_mask]

    logger.info("SVB primary fires: %d, meta-positive: %d (%.1f%%)",
                prim_mask.sum(), y_true_fires.sum(), 100 * float(y_true_fires.mean()))

    ece_raw = _ece(proba_fires, y_true_fires, _N_BINS)
    bins_raw = _calibration_bins(proba_fires, y_true_fires, _N_BINS)

    # Isotonic calibration: fit on 60% of fires, evaluate on remaining 40%
    n_fires = len(proba_fires)
    split = max(1, int(n_fires * 0.6))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_fires[:split], y_true_fires[:split])
    proba_iso = iso.predict(proba_fires[split:])
    y_eval = y_true_fires[split:]

    ece_iso = _ece(proba_iso, y_eval, _N_BINS)
    bins_iso = _calibration_bins(proba_iso, y_eval, _N_BINS)

    logger.info("ECE raw: %.4f, ECE isotonic: %.4f", ece_raw, ece_iso)

    # Fraction above 70% predicted prob with actual rate
    high_conf_bins = [b for b in bins_raw if b["mean_predicted_prob"] and b["mean_predicted_prob"] >= 0.7]
    mean_actual_high_conf = (
        float(np.mean([b["actual_profitable_rate"] for b in high_conf_bins]))
        if high_conf_bins else float("nan")
    )

    result = {
        "model": "cross_mechanism_meta_labeler",
        "training_event": "terra_ust_2022",
        "test_event": "usdc_svb_2023",
        "data_provenance": "synthetic_fallback",
        "n_svb_fires_evaluated": int(prim_mask.sum()),
        "meta_positive_rate_svb": round(float(y_true_fires.mean()), 4),
        "ece_raw": round(ece_raw, 4),
        "ece_isotonic": round(ece_iso, 4),
        "ece_improved": bool(ece_iso < ece_raw),
        "well_calibrated": bool(ece_raw < 0.15),
        "n_bins": _N_BINS,
        "calibration_bins_raw": bins_raw,
        "calibration_bins_isotonic": bins_iso,
        "mean_actual_profitable_rate_above_07_predicted": round(mean_actual_high_conf, 4)
        if not np.isnan(mean_actual_high_conf) else None,
    }

    out_path = out_dir / "calibration_curve.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Saved calibration results to %s", out_path)

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, bins_data, title in [
        (axes[0], bins_raw, f"Raw Predictions\n(ECE = {ece_raw:.3f})"),
        (axes[1], bins_iso, f"After Isotonic Recalibration\n(ECE = {ece_iso:.3f})"),
    ]:
        xs = [b["mean_predicted_prob"] for b in bins_data
              if b["n"] > 0 and b["mean_predicted_prob"] is not None]
        ys = [b["actual_profitable_rate"] for b in bins_data
              if b["n"] > 0 and b["actual_profitable_rate"] is not None]
        ns = [b["n"] for b in bins_data if b["n"] > 0]

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
        ax.scatter(xs, ys, s=[max(20, n // 5) for n in ns], c="#2166ac", alpha=0.8, zorder=5)
        if len(xs) > 1:
            ax.plot(xs, ys, "o-", color="#2166ac", linewidth=1.5, alpha=0.55)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Actual profitable rate", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Reliability Diagram: Cross-Mechanism Meta-Labeler on SVB Test Split\n"
                 "(synthetic fallback data)", fontsize=10, y=1.03)
    plt.tight_layout()

    fig_path = fig_dir / "calibration_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", fig_path)

    print(f"\n=== Calibration Results (Plan E) ===")
    print(f"ECE (raw):                {ece_raw:.4f}")
    print(f"ECE (isotonic post-hoc):  {ece_iso:.4f}")
    print(f"Well-calibrated (ECE<0.15): {result['well_calibrated']}")
    print(f"ECE improved by isotonic: {result['ece_improved']}")
    if result["mean_actual_profitable_rate_above_07_predicted"] is not None:
        print(f"Actual profitable rate (predicted≥0.7): "
              f"{result['mean_actual_profitable_rate_above_07_predicted']:.3f}")
    print(f"\nBin breakdown (raw):")
    print(f"{'Bin':>12} {'n':>6} {'pred':>8} {'actual':>8}")
    for b in bins_raw:
        if b["n"] > 0:
            print(f"[{b['bin_low']:.1f},{b['bin_high']:.1f}] "
                  f"{b['n']:>6} {b['mean_predicted_prob']:.3f}    {b['actual_profitable_rate']:.3f}")


if __name__ == "__main__":
    main()
