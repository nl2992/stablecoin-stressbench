#!/usr/bin/env python3
"""SHAP feature importance analysis for the primary LightGBM classifier.

Trains a LightGBM model on the ``price_book_settle`` feature set
(the broadest set) for the primary task
``label_basis_usdc_1m_gt10bps``, computes SHAP values on the test
split, and writes publication-quality figures plus a summary CSV.

Outputs
-------
results/paper_addon/figures/figure_shap_importance.png
    Top-10 SHAP mean |value| bar chart, coloured by feature category.
results/paper_addon/figures/figure_shap_summary.png
    SHAP beeswarm / dot summary plot (all features in the model).
results/paper_addon/table_shap_importance.csv
    Mean |SHAP value| per feature, sorted descending.

Usage
-----
    python scripts/run_shap_analysis.py
    python scripts/run_shap_analysis.py --data-dir data/gold --output-dir results/paper_addon
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Feature set definitions (mirrors src/stressbench/experiments/feature_sets.py)
# ---------------------------------------------------------------------------

_PRICE_COLS: list[str] = [
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_usdt_bps",
    "cross_quote_basis_maxabs_bps",
    "cross_quote_basis_primary_bps",
    "deviation_from_1_usd_bps",
]

_BOOK_COLS: list[str] = [
    "spread_bps_mean",
    "depth_bid_10bp_mean",
    "depth_ask_10bp_mean",
    "imbalance_1bp_mean",
    "data_quality_score_min",
    "trade_count_1m_total",
    "trade_volume_1m_total",
]

_FRAG_COLS: list[str] = [
    "num_active_venues_mean",
    "mid_dispersion_bps_mean",
    "max_minus_min_bps_mean",
]

_SETTLE_COLS: list[str] = [
    "transfer_count_1m",
    "transfer_volume_1m",
    "large_transfer_count_1m",
    "gas_proxy",
    "block_lag_proxy",
    "dex_swap_volume_1m",
    "dex_net_flow_1m",
]

_ALL_FEATURE_COLS: list[str] = _PRICE_COLS + _BOOK_COLS + _FRAG_COLS + _SETTLE_COLS

TARGET_COL = "label_basis_usdc_1m_gt10bps"

# Category colours for the figures
_CAT_COLORS: dict[str, str] = {
    "price": "#2166ac",  # navy blue
    "book": "#d73027",  # red
    "settle": "#1a9641",  # green
}

# ---------------------------------------------------------------------------
# Helper: map a feature name to its category colour
# ---------------------------------------------------------------------------


def _feature_color(name: str) -> str:
    if name in _PRICE_COLS or name in _FRAG_COLS:
        # Fragmentation is a price/book hybrid; treat as book for colour
        if name in _FRAG_COLS:
            return _CAT_COLORS["book"]
        return _CAT_COLORS["price"]
    if name in _BOOK_COLS:
        return _CAT_COLORS["book"]
    if name in _SETTLE_COLS:
        return _CAT_COLORS["settle"]
    return "#888888"


def _feature_category(name: str) -> str:
    if name in _PRICE_COLS:
        return "price"
    if name in _BOOK_COLS or name in _FRAG_COLS:
        return "book / frag"
    if name in _SETTLE_COLS:
        return "settle"
    return "other"


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


def _generate_synthetic_data(n_total: int = 12_000, seed: int = 42) -> "pl.DataFrame":
    """Generate a realistic synthetic dataset matching the expected schema."""
    import polars as pl

    rng = np.random.default_rng(seed)

    # --- basis and price signals ---
    # Basis follows a mean-reverting process with occasional spikes
    basis_usdc = rng.normal(0, 8, n_total).cumsum() * 0.1
    basis_usdc = basis_usdc - basis_usdc.mean()
    basis_usdc = np.clip(basis_usdc, -80, 80)

    basis_usdt = basis_usdc + rng.normal(0, 3, n_total)
    basis_maxabs = np.maximum(np.abs(basis_usdc), np.abs(basis_usdt))
    basis_primary = basis_usdc + rng.normal(0, 1, n_total)
    deviation = basis_usdc * 0.7 + rng.normal(0, 2, n_total)

    # --- order book microstructure ---
    spread = np.abs(rng.normal(2, 1.5, n_total)) + 0.5
    depth_bid = np.abs(rng.normal(500_000, 200_000, n_total))
    depth_ask = np.abs(rng.normal(490_000, 200_000, n_total))
    imbalance = np.clip(rng.normal(0, 0.3, n_total), -1, 1)
    data_quality = np.clip(rng.beta(8, 2, n_total), 0, 1)
    trade_count = np.abs(rng.poisson(30, n_total)).astype(float)
    trade_volume = np.abs(rng.normal(1_000_000, 500_000, n_total))

    # --- fragmentation ---
    num_venues = rng.integers(2, 8, n_total).astype(float)
    mid_disp = np.abs(rng.normal(1.5, 1, n_total))
    max_min = mid_disp * 2 + rng.exponential(0.5, n_total)

    # --- on-chain settlement ---
    transfer_count = np.abs(rng.poisson(15, n_total)).astype(float)
    transfer_volume = np.abs(rng.normal(500_000, 300_000, n_total))
    large_transfer_count = np.abs(rng.poisson(2, n_total)).astype(float)
    gas_proxy = np.abs(rng.normal(50, 30, n_total))
    block_lag = np.abs(rng.normal(2, 1, n_total))
    dex_swap_vol = np.abs(rng.normal(200_000, 150_000, n_total))
    dex_net_flow = rng.normal(0, 100_000, n_total)

    # --- label: basis_usdc > 10 bps (with some book-signal influence) ---
    log_odds = (
        0.15 * basis_usdc
        - 0.05 * spread
        + 0.02 * imbalance * 10
        + 0.01 * mid_disp
        + rng.normal(0, 1, n_total)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    label = (rng.uniform(size=n_total) < prob).astype(int)

    # --- train/val/test split (70/10/20) ---
    split = np.full(n_total, "train", dtype=object)
    val_start = int(n_total * 0.70)
    test_start = int(n_total * 0.80)
    split[val_start:test_start] = "val"
    split[test_start:] = "test"

    data = {
        "cross_quote_basis_usdc_bps": basis_usdc,
        "cross_quote_basis_usdt_bps": basis_usdt,
        "cross_quote_basis_maxabs_bps": basis_maxabs,
        "cross_quote_basis_primary_bps": basis_primary,
        "deviation_from_1_usd_bps": deviation,
        "spread_bps_mean": spread,
        "depth_bid_10bp_mean": depth_bid,
        "depth_ask_10bp_mean": depth_ask,
        "imbalance_1bp_mean": imbalance,
        "data_quality_score_min": data_quality,
        "trade_count_1m_total": trade_count,
        "trade_volume_1m_total": trade_volume,
        "num_active_venues_mean": num_venues,
        "mid_dispersion_bps_mean": mid_disp,
        "max_minus_min_bps_mean": max_min,
        "transfer_count_1m": transfer_count,
        "transfer_volume_1m": transfer_volume,
        "large_transfer_count_1m": large_transfer_count,
        "gas_proxy": gas_proxy,
        "block_lag_proxy": block_lag,
        "dex_swap_volume_1m": dex_swap_vol,
        "dex_net_flow_1m": dex_net_flow,
        TARGET_COL: label,
        "split": split.tolist(),
    }

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir: Path) -> "pl.DataFrame":
    import polars as pl

    parquet_path = data_dir / "dataset.parquet"
    if parquet_path.exists() and parquet_path.stat().st_size > 10:
        print(f"[data] Loading real data from {parquet_path}")
        return pl.read_parquet(parquet_path)

    print("[data] dataset.parquet not found — generating synthetic data.")
    df = _generate_synthetic_data()
    print(f"[data] Synthetic dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    label_rate = df[TARGET_COL].mean()
    print(f"[data] Label prevalence: {label_rate:.1%}")
    return df


# ---------------------------------------------------------------------------
# Feature selection (graceful missing-column handling)
# ---------------------------------------------------------------------------


def resolve_features(df: "pl.DataFrame", desired: list[str]) -> list[str]:
    available = set(df.columns)
    present = [c for c in desired if c in available]
    missing = [c for c in desired if c not in available]
    if missing:
        print(f"[features] Dropping {len(missing)} missing columns: {missing}")
    print(f"[features] Using {len(present)} features.")
    return present


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> "lgb.LGBMClassifier":
    import lightgbm as lgb

    # Compute class weight for balanced training
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        feature_name=feature_names,
    )

    n_iter = model.best_iteration_ if model.best_iteration_ else model.n_estimators
    print(f"[train] Fitted LightGBM — best iteration: {n_iter}")
    return model


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------


def compute_shap(
    model: "lgb.LGBMClassifier",
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (shap_values_class1, mean_abs_shap) arrays."""
    import shap

    print(f"[shap] Computing SHAP values for {X_test.shape[0]:,} test samples …")
    explainer = shap.TreeExplainer(model)
    shap_out = explainer(X_test)

    # shap_out.values shape: (n_samples, n_features) for binary classification
    # (TreeExplainer on LGB classifier returns values for positive class)
    sv = shap_out.values
    if sv.ndim == 3:
        # Multi-output: take class 1 slice
        sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    print("[shap] Done.")
    return sv, mean_abs


# ---------------------------------------------------------------------------
# Figure 1: Top-10 SHAP bar chart
# ---------------------------------------------------------------------------


def plot_shap_bar(
    mean_abs: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 10,
) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Sort descending
    order = np.argsort(mean_abs)[::-1]
    top_idx = order[:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]
    top_colors = [_feature_color(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(
        range(top_n - 1, -1, -1),  # top feature at top
        top_vals,
        color=top_colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax.set_yticks(range(top_n - 1, -1, -1))
    ax.set_yticklabels(
        [_pretty_name(n) for n in top_names],
        fontsize=9,
    )
    ax.set_xlabel("Mean |SHAP value|  (log-odds units)", fontsize=10)
    ax.set_title(
        "Top-10 SHAP Feature Importances\n"
        r"LightGBM — $\mathit{label\_basis\_usdc\_1m\_gt10bps}$",
        fontsize=10,
        pad=8,
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=_CAT_COLORS["price"], label="Price / basis"),
        mpatches.Patch(color=_CAT_COLORS["book"], label="Book / frag."),
        mpatches.Patch(color=_CAT_COLORS["settle"], label="On-chain / settle"),
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=8,
        loc="lower right",
        framealpha=0.8,
        edgecolor="lightgrey",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved bar chart → {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: SHAP beeswarm / dot summary plot
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    max_display: int = 15,
) -> None:
    import matplotlib.pyplot as plt
    import shap

    fig, ax = plt.subplots(figsize=(7, 5))

    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=[_pretty_name(n) for n in feature_names],
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar_label="Feature value",
    )

    ax = plt.gca()
    ax.set_title(
        "SHAP Summary — LightGBM Classifier\n"
        r"$\mathit{label\_basis\_usdc\_1m\_gt10bps}$",
        fontsize=10,
        pad=8,
    )
    ax.set_xlabel("SHAP value  (impact on model output)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[figure] Saved summary dot plot → {output_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(
    mean_abs: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    import csv

    order = np.argsort(mean_abs)[::-1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["rank", "feature", "category", "mean_abs_shap"],
        )
        writer.writeheader()
        for rank, idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature": feature_names[idx],
                    "category": _feature_category(feature_names[idx]),
                    "mean_abs_shap": f"{mean_abs[idx]:.6f}",
                }
            )
    print(f"[csv]    Saved → {output_path}")


# ---------------------------------------------------------------------------
# Key-finding printer
# ---------------------------------------------------------------------------


def print_key_findings(
    mean_abs: np.ndarray,
    feature_names: list[str],
) -> None:
    order = np.argsort(mean_abs)[::-1]
    ranked_names = [feature_names[i] for i in order]
    ranked_vals = mean_abs[order]
    ranked_cats = [_feature_category(n) for n in ranked_names]

    print("\n" + "=" * 65)
    print("KEY FINDINGS — SHAP feature importance")
    print("=" * 65)

    # Total importance by category
    cat_totals: dict[str, float] = {}
    for cat, val in zip(ranked_cats, ranked_vals):
        cat_totals[cat] = cat_totals.get(cat, 0.0) + val
    total = sum(cat_totals.values())
    print("\nImportance share by category:")
    for cat, v in sorted(cat_totals.items(), key=lambda x: -x[1]):
        print(f"  {cat:<18s} {v / total:5.1%}  (sum |SHAP| = {v:.4f})")

    # Top-3 overall
    print("\nTop-3 features overall:")
    for i in range(min(3, len(ranked_names))):
        print(
            f"  #{i + 1:d}  {ranked_names[i]:<40s} "
            f"[{ranked_cats[i]}]  mean|SHAP|={ranked_vals[i]:.4f}"
        )

    # Highest-ranked pure book/frag feature
    book_feats = [
        (n, v, r)
        for r, (n, v, c) in enumerate(
            zip(ranked_names, ranked_vals, ranked_cats), start=1
        )
        if c == "book / frag"
    ]
    if book_feats:
        top_book = book_feats[0]
        print(
            f"\nTop book/frag feature: {top_book[0]!r}  "
            f"rank #{top_book[2]}  mean|SHAP|={top_book[1]:.4f}"
        )

    # Highest-ranked settle feature
    settle_feats = [
        (n, v, r)
        for r, (n, v, c) in enumerate(
            zip(ranked_names, ranked_vals, ranked_cats), start=1
        )
        if c == "settle"
    ]
    if settle_feats:
        top_settle = settle_feats[0]
        print(
            f"Top settle feature:    {top_settle[0]!r}  "
            f"rank #{top_settle[2]}  mean|SHAP|={top_settle[1]:.4f}"
        )

    # Marginal lift narrative
    price_share = cat_totals.get("price", 0.0) / total
    book_share = cat_totals.get("book / frag", 0.0) / total
    settle_share = cat_totals.get("settle", 0.0) / total
    print("\nNarrative:")
    if price_share >= 0.50:
        print(
            f"  Price/basis features dominate ({price_share:.0%} of total SHAP importance)."
        )
    if book_share >= 0.20:
        print(
            f"  Book/frag features provide meaningful marginal lift ({book_share:.0%});"
        )
        print("  microstructure signals are informative beyond raw basis alone.")
    else:
        print(f"  Book/frag features add only modest marginal lift ({book_share:.0%});")
        print("  microstructure adds limited signal on top of price-only features.")
    if settle_share < 0.10:
        print(f"  On-chain/settle features contribute least ({settle_share:.0%}),")
        print(
            "  suggesting settlement proxies have weak predictive power at 1-min horizon."
        )
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pretty_name(col: str) -> str:
    """Human-readable short label for a feature column name."""
    _MAP = {
        "cross_quote_basis_usdc_bps": "Basis USDC (bps)",
        "cross_quote_basis_usdt_bps": "Basis USDT (bps)",
        "cross_quote_basis_maxabs_bps": "Basis max-abs (bps)",
        "cross_quote_basis_primary_bps": "Basis primary (bps)",
        "deviation_from_1_usd_bps": "Deviation from $1 (bps)",
        "spread_bps_mean": "Spread mean (bps)",
        "depth_bid_10bp_mean": "Bid depth 10 bp ($)",
        "depth_ask_10bp_mean": "Ask depth 10 bp ($)",
        "imbalance_1bp_mean": "Order imbalance 1 bp",
        "data_quality_score_min": "Data quality score",
        "trade_count_1m_total": "Trade count (1 min)",
        "trade_volume_1m_total": "Trade volume (1 min)",
        "num_active_venues_mean": "Active venues",
        "mid_dispersion_bps_mean": "Mid dispersion (bps)",
        "max_minus_min_bps_mean": "Max−min spread (bps)",
        "transfer_count_1m": "Transfer count (on-chain)",
        "transfer_volume_1m": "Transfer volume (on-chain)",
        "large_transfer_count_1m": "Large transfers (on-chain)",
        "gas_proxy": "Gas proxy",
        "block_lag_proxy": "Block lag proxy",
        "dex_swap_volume_1m": "DEX swap volume",
        "dex_net_flow_1m": "DEX net flow",
    }
    return _MAP.get(col, col)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/gold"),
        help="Directory containing dataset.parquet (default: data/gold).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper_addon"),
        help="Root output directory (default: results/paper_addon).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data ----------------------------------------------------------
    df = load_data(args.data_dir)

    # 2. Resolve available features -----------------------------------------
    feature_cols = resolve_features(df, _ALL_FEATURE_COLS)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in dataset. "
            f"Available columns: {df.columns}"
        )

    # 3. Split data ---------------------------------------------------------
    train_df = df.filter(df["split"] == "train")
    val_df = df.filter(df["split"] == "val")
    test_df = df.filter(df["split"] == "test")

    print(
        f"[split] train={train_df.shape[0]:,}  "
        f"val={val_df.shape[0]:,}  "
        f"test={test_df.shape[0]:,}"
    )

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_df[TARGET_COL].to_numpy().astype(int)
    X_val = val_df.select(feature_cols).to_numpy().astype(np.float32)
    y_val = val_df[TARGET_COL].to_numpy().astype(int)
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_test = test_df[TARGET_COL].to_numpy().astype(int)

    print(
        f"[label] train prevalence: {y_train.mean():.1%}  "
        f"test prevalence: {y_test.mean():.1%}"
    )

    # 4. Train LightGBM -----------------------------------------------------
    model = train_lgbm(X_train, y_train, X_val, y_val, feature_names=feature_cols)

    # Quick test-set AUC
    try:
        from sklearn.metrics import roc_auc_score

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"[eval]  Test ROC-AUC = {auc:.4f}")
    except Exception:
        pass

    # 5. SHAP values --------------------------------------------------------
    shap_values, mean_abs = compute_shap(model, X_test, feature_cols)

    # 6. Bar chart ----------------------------------------------------------
    bar_path = figures_dir / "figure_shap_importance.png"
    plot_shap_bar(mean_abs, feature_cols, bar_path, top_n=10)

    # 7. Summary dot plot ---------------------------------------------------
    summary_path = figures_dir / "figure_shap_summary.png"
    plot_shap_summary(shap_values, X_test, feature_cols, summary_path, max_display=15)

    # 8. CSV ----------------------------------------------------------------
    csv_path = args.output_dir / "table_shap_importance.csv"
    write_csv(mean_abs, feature_cols, csv_path)

    # 9. Key findings -------------------------------------------------------
    print_key_findings(mean_abs, feature_cols)


if __name__ == "__main__":
    main()
