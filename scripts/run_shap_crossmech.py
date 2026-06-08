#!/usr/bin/env python3
"""Plan B: SHAP cross-event feature attribution for mechanism invariance.

Computes SHAP values on both the Terra/LUNA training split and the SVB test split.
Shows that depth_bid, spread, depth_ask dominate in both splits — the "why it transfers"
proof for mechanism invariance.

Usage:
    python scripts/run_shap_crossmech.py
    python scripts/run_shap_crossmech.py --output-dir results/experiments_addon
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
_FEATURE_NAMES = ["basis", "depth_bid", "depth_ask", "spread", "imbalance"]


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    corr, _ = spearmanr(a, b)
    return float(corr)


def _jaccard_top_k(rank_a: list, rank_b: list, k: int = 5) -> float:
    sa, sb = set(rank_a[:k]), set(rank_b[:k])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    p = argparse.ArgumentParser(description="SHAP cross-event attribution")
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--fig-dir", default="results/paper/figures")
    args = p.parse_args()

    import shap

    rng = np.random.default_rng(_SEED)
    out_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    terra = generate_terra(rng)
    svb = generate_svb(rng)

    X_train = make_features(terra)
    y_prim_train = terra["primary_signal"]
    y_meta_train = terra["meta_label"]

    model = MetaLabelingFilter(primary_threshold_bps=_PRIMARY_THRESHOLD, primary_signal_col=0)
    model.fit(X_train, y_prim_train, y_meta_train)
    lgbm_clf = model._meta_clf
    logger.info("Trained: %d primary fires, %d meta-positive (%.1f%%)",
                model.n_primary_fires_train, model.n_meta_positive_train,
                100 * model.n_meta_positive_train / max(model.n_primary_fires_train, 1))

    # SHAP on Terra primary fires (training distribution)
    terra_fires_mask = y_prim_train.astype(bool)
    X_terra_fires = X_train[terra_fires_mask]

    # SHAP on SVB primary fires (test / out-of-sample distribution)
    X_test = make_features(svb)
    y_prim_svb = svb["primary_signal"]
    svb_fires_mask = y_prim_svb.astype(bool)
    X_svb_fires = X_test[svb_fires_mask]

    logger.info("SHAP: Terra fires=%d, SVB fires=%d",
                X_terra_fires.shape[0], X_svb_fires.shape[0])

    explainer = shap.TreeExplainer(lgbm_clf)
    shap_terra = explainer.shap_values(X_terra_fires)
    shap_svb = explainer.shap_values(X_svb_fires)

    if isinstance(shap_terra, list):
        shap_terra = shap_terra[1]
    if isinstance(shap_svb, list):
        shap_svb = shap_svb[1]

    mean_abs_terra = np.abs(shap_terra).mean(axis=0)
    mean_abs_svb = np.abs(shap_svb).mean(axis=0)

    rank_terra = list(np.argsort(-mean_abs_terra))
    rank_svb = list(np.argsort(-mean_abs_svb))
    feat_rank_terra = [_FEATURE_NAMES[i] for i in rank_terra]
    feat_rank_svb = [_FEATURE_NAMES[i] for i in rank_svb]

    spearman_rho = _spearman_rho(mean_abs_terra, mean_abs_svb)
    jaccard_top3 = _jaccard_top_k(feat_rank_terra, feat_rank_svb, k=3)
    jaccard_top5 = _jaccard_top_k(feat_rank_terra, feat_rank_svb, k=5)

    logger.info("Terra top-5: %s", feat_rank_terra[:5])
    logger.info("SVB   top-5: %s", feat_rank_svb[:5])
    logger.info("Spearman rho: %.3f, Jaccard@3: %.3f", spearman_rho, jaccard_top3)

    result = {
        "model": "cross_mechanism_meta_labeler",
        "training_event": "terra_ust_2022",
        "test_event": "usdc_svb_2023",
        "data_provenance": "synthetic_fallback",
        "feature_names": _FEATURE_NAMES,
        "terra_mean_abs_shap": [round(float(v), 6) for v in mean_abs_terra],
        "svb_mean_abs_shap": [round(float(v), 6) for v in mean_abs_svb],
        "terra_rank_by_importance": feat_rank_terra,
        "svb_rank_by_importance": feat_rank_svb,
        "terra_n_fires": int(X_terra_fires.shape[0]),
        "svb_n_fires": int(X_svb_fires.shape[0]),
        "spearman_rho": round(spearman_rho, 4),
        "jaccard_top3": round(jaccard_top3, 4),
        "jaccard_top5": round(jaccard_top5, 4),
        "top3_features_align": feat_rank_terra[:3] == feat_rank_svb[:3],
        "mechanism_invariance_confirmed": spearman_rho > 0.60,
    }

    out_path = out_dir / "shap_crossmech.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Saved SHAP results to %s", out_path)

    # Figure: side-by-side SHAP bars sorted by Terra importance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_by_terra = rank_terra
    feat_labels = [_FEATURE_NAMES[i] for i in ordered_by_terra][::-1]  # bottom→top
    vals_terra = mean_abs_terra[ordered_by_terra][::-1]
    vals_svb = mean_abs_svb[ordered_by_terra][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, title, color in [
        (axes[0], vals_terra, "Terra/LUNA\n(Training Split)", "#2166ac"),
        (axes[1], vals_svb, "USDC/SVB\n(Test Split)", "#d6604d"),
    ]:
        ax.barh(feat_labels, vals, color=color, alpha=0.82)
        ax.set_xlabel("Mean |SHAP value|", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"SHAP Feature Importance: Terra/LUNA vs USDC/SVB\n"
        f"(Spearman ρ = {spearman_rho:.3f}, Jaccard@3 = {jaccard_top3:.2f})",
        fontsize=11, y=1.03
    )
    plt.tight_layout()
    fig_path = fig_dir / "shap_crossmech_fig.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", fig_path)

    print(f"\n=== SHAP Cross-Event Attribution (Plan B) ===")
    print(f"Terra top-5: {feat_rank_terra[:5]}")
    print(f"SVB   top-5: {feat_rank_svb[:5]}")
    print(f"Spearman rho: {spearman_rho:.4f}")
    print(f"Jaccard top-3: {jaccard_top3:.4f}")
    print(f"Top-3 align: {result['top3_features_align']}")
    print(f"Mechanism invariance confirmed: {result['mechanism_invariance_confirmed']}")


if __name__ == "__main__":
    main()
