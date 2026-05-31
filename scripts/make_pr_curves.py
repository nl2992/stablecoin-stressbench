#!/usr/bin/env python3
"""Precision-recall curves for key models (T1.4).
Saves to results/paper/figures/figure_pr_curves.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = Path(__file__).parent.parent
DATA = REPO / "data" / "gold" / "dataset.parquet"
OUT = REPO / "results" / "paper" / "figures" / "figure_pr_curves.png"

TASK = "label_basis_usdc_1m_gt10bps"
FEATS = [
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_usdt_bps",
    "cross_quote_basis_maxabs_bps",
    "cross_quote_basis_primary_bps",
    "spread_bps_mean",
    "depth_bid_10bp_mean",
    "depth_ask_10bp_mean",
    "imbalance_1bp_mean",
]

C_ORACLE = "#F2A900"
C_LGBM = "#003057"
C_PRICE = "#d73027"
C_META = "#2ca02c"
C_GREY = "#aaaaaa"


def to_X(sdf, cols):
    return np.nan_to_num(
        sdf.select([c for c in cols if c in sdf.columns]).to_numpy().astype(float),
        nan=0.0,
    )


def main():
    df = pl.read_parquet(str(DATA))
    train = df.filter(pl.col("split") == "train")
    val = df.filter(pl.col("split") == "validation")
    test = df.filter(pl.col("split") == "test")

    X_tr = to_X(train, FEATS)
    y_tr = train[TASK].to_numpy().astype(int)
    X_te = to_X(test, FEATS)
    y_te = test[TASK].to_numpy().astype(int)
    pos_rate = float(y_te.mean())

    # LightGBM (calm-trained)
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        n_jobs=4,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    probs_lgbm = lgbm.predict_proba(X_te)[:, 1]

    # Price rule: use |basis| as score
    basis_te = np.abs(
        np.nan_to_num(
            test["cross_quote_basis_usdc_bps"].to_numpy().astype(float), nan=0.0
        )
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Baseline (random)
    ax.axhline(
        pos_rate, color=C_GREY, ls="--", lw=1.2, label=f"Random (AP = {pos_rate:.3f})"
    )

    # Price rule PR
    prec_p, rec_p, _ = precision_recall_curve(y_te, basis_te)
    ap_p = average_precision_score(y_te, basis_te)
    ax.plot(
        rec_p, prec_p, color=C_PRICE, lw=1.8, label=f"PriceBasis rule (AP = {ap_p:.3f})"
    )

    # LightGBM PR
    prec_l, rec_l, _ = precision_recall_curve(y_te, probs_lgbm)
    ap_l = average_precision_score(y_te, probs_lgbm)
    ax.plot(rec_l, prec_l, color=C_LGBM, lw=1.8, label=f"LightGBM (AP = {ap_l:.3f})")

    ax.set_xlabel("Recall", fontsize=10.5)
    ax.set_ylabel("Precision", fontsize=10.5)
    ax.set_title(
        f"Precision-Recall Curves  (basis_usdc_1m_gt10bps, {pos_rate:.1%} positives)",
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(
        0.03,
        0.03,
        "High recall = many trades = high FP exposure\nHigh AP does not imply positive net bps",
        transform=ax.transAxes,
        fontsize=7.5,
        style="italic",
        color="#555555",
        va="bottom",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
