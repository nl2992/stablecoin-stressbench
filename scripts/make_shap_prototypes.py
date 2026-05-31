#!/usr/bin/env python3
"""SHAP-feature prototype analysis: Terra/LUNA → SVB transfer.

Finds K=4 "stress prototypes" from the Terra/LUNA validation split using
KMeans clustering on the top-5 SHAP features, then identifies the 3 nearest
SVB test windows to each prototype in that same feature space.

Method
------
1. Load data/gold/dataset.parquet.
2. Extract Terra/LUNA windows (split == 'validation') and SVB test windows
   (split == 'test').
3. Load top-5 SHAP features from results/paper_addon/table_shap_importance.csv.
4. Build feature matrices for both splits (StandardScaler fit on Terra/LUNA).
5. KMeans(k=4) on Terra/LUNA feature matrix; pick the Terra window closest to
   each centroid as the prototype.
6. For each prototype, find the 3 nearest SVB windows by Euclidean distance in
   scaled feature space.
7. Report: prototype Terra window features + label, matched SVB window features
   + label + basis + net_profit.
8. Write results/paper_addon/shap_prototypes.csv.
9. Plot a 4×2 grid figure (4 prototypes × [Terra prototype, nearest SVB window])
   as horizontal bar charts of feature values.  Save to
   results/paper/figures/figure_shap_prototypes.png.

Outputs
-------
results/paper_addon/shap_prototypes.csv
results/paper/figures/figure_shap_prototypes.png

Usage
-----
    python scripts/make_shap_prototypes.py
    python scripts/make_shap_prototypes.py \\
        --data-dir data/gold \\
        --shap-csv results/paper_addon/table_shap_importance.csv \\
        --output-dir results/paper_addon \\
        --figures-dir results/paper/figures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_K_PROTOTYPES = 4
_N_NEAREST_SVB = 3
_TOP_N_SHAP = 5

# Human-readable short labels (mirrors run_shap_analysis.py)
_PRETTY_NAMES: dict[str, str] = {
    "cross_quote_basis_usdc_bps": "Basis USDC (bps)",
    "cross_quote_basis_primary_bps": "Basis primary (bps)",
    "deviation_from_1_usd_bps": "Deviation from $1 (bps)",
    "cross_quote_basis_usdt_bps": "Basis USDT (bps)",
    "depth_ask_10bp_mean": "Ask depth 10 bp ($)",
    "spread_bps_mean": "Spread mean (bps)",
    "transfer_volume_1m": "Transfer volume (on-chain)",
    "imbalance_1bp_mean": "Order imbalance 1 bp",
    "trade_volume_1m_total": "Trade volume (1 min)",
    "depth_bid_10bp_mean": "Bid depth 10 bp ($)",
    "block_lag_proxy": "Block lag proxy",
    "mid_dispersion_bps_mean": "Mid dispersion (bps)",
    "gas_proxy": "Gas proxy",
    "num_active_venues_mean": "Active venues",
    "cross_quote_basis_maxabs_bps": "Basis max-abs (bps)",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "gold",
        help="Directory containing dataset.parquet.",
    )
    p.add_argument(
        "--shap-csv",
        type=Path,
        default=_REPO_ROOT / "results" / "paper_addon" / "table_shap_importance.csv",
        help="Path to table_shap_importance.csv.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "paper_addon",
        help="Output directory for shap_prototypes.csv.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "paper" / "figures",
        help="Output directory for figure_shap_prototypes.png.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=_K_PROTOTYPES,
        help="Number of KMeans clusters / prototypes (default: 4).",
    )
    p.add_argument(
        "--n-nearest",
        type=int,
        default=_N_NEAREST_SVB,
        help="Number of nearest SVB windows to report per prototype (default: 3).",
    )
    p.add_argument(
        "--top-n-shap",
        type=int,
        default=_TOP_N_SHAP,
        help="Number of top SHAP features to use (default: 5).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(data_dir: Path) -> "pl.DataFrame":
    """Load dataset.parquet with a clear error if it is missing."""
    import polars as pl

    parquet_path = data_dir / "dataset.parquet"
    if not parquet_path.exists():
        print(
            "\n[ERROR] dataset.parquet not found.\n"
            f"  Expected path: {parquet_path}\n"
            "\n"
            "  This file requires a Tardis Machine subscription and is not\n"
            "  included in the repository.  To obtain it:\n"
            "    1. Subscribe to Tardis Machine at https://tardis.dev/\n"
            "    2. Run:  python scripts/pull_data.py\n"
            "    3. Run:  python scripts/build_features.py --start ... --end ...\n"
            "    4. The pipeline outputs data/gold/dataset.parquet automatically.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[data] Loading {parquet_path} …", flush=True)
    df = pl.read_parquet(str(parquet_path))
    print(f"[data] Loaded {len(df):,} rows × {len(df.columns)} columns.", flush=True)
    return df


def load_shap_features(shap_csv: Path, top_n: int) -> list[str]:
    """Load the top-N features from table_shap_importance.csv."""
    if not shap_csv.exists():
        print(
            f"[ERROR] SHAP CSV not found: {shap_csv}\n"
            "  Run:  python scripts/run_shap_analysis.py  to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    features = []
    with open(shap_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            features.append(row["feature"])
            if len(features) >= top_n:
                break

    print(f"[shap] Top-{top_n} features: {features}", flush=True)
    return features


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_feature_matrix(
    df: "pl.DataFrame",
    feature_cols: list[str],
    split: str,
) -> tuple[np.ndarray, "pl.DataFrame"]:
    """Extract the feature matrix and the corresponding slice of df for a split.

    Returns:
        X:   (n, len(feature_cols)) float32 array with NaN → column median imputed.
        sdf: The filtered/projected polars DataFrame (original columns preserved).
    """
    import polars as pl

    sdf = df.filter(pl.col("split") == split)

    if sdf.is_empty():
        print(
            f"[warn] No rows found for split='{split}'.  "
            f"Available splits: {df['split'].unique().to_list()}",
            file=sys.stderr,
        )
        return np.empty((0, len(feature_cols)), dtype=np.float32), sdf

    # Drop feature columns that are absent
    present = [c for c in feature_cols if c in sdf.columns]
    missing = [c for c in feature_cols if c not in sdf.columns]
    if missing:
        print(
            f"[warn] {len(missing)} SHAP feature(s) absent from dataset: {missing}",
            file=sys.stderr,
        )

    if not present:
        return np.empty((0, len(feature_cols)), dtype=np.float32), sdf

    X_raw = sdf.select(present).to_numpy().astype(np.float32)

    # Impute NaN with column median
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        X_raw = np.where(nan_mask, col_medians[None, :], X_raw)

    # Pad with zeros for missing columns to keep shape consistent
    if len(present) < len(feature_cols):
        n_missing = len(feature_cols) - len(present)
        X_raw = np.concatenate(
            [X_raw, np.zeros((len(X_raw), n_missing), dtype=np.float32)], axis=1
        )

    print(f"[split] '{split}': {len(sdf):,} rows", flush=True)
    return X_raw, sdf


# ---------------------------------------------------------------------------
# KMeans prototypes
# ---------------------------------------------------------------------------


def find_prototypes(
    X_terra: np.ndarray,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run KMeans and find the Terra window closest to each centroid.

    Returns:
        prototype_indices: (k,) array of row indices into X_terra.
        cluster_labels:    (n,) cluster assignment for every Terra row.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    cluster_labels = km.fit_predict(X_terra)
    centroids = km.cluster_centers_  # (k, n_features)

    prototype_indices = np.empty(k, dtype=int)
    for c in range(k):
        mask = cluster_labels == c
        if not mask.any():
            # Degenerate cluster: pick global closest point to this centroid
            dists = np.linalg.norm(X_terra - centroids[c], axis=1)
            prototype_indices[c] = int(np.argmin(dists))
        else:
            cluster_rows = np.where(mask)[0]
            dists = np.linalg.norm(X_terra[cluster_rows] - centroids[c], axis=1)
            prototype_indices[c] = cluster_rows[int(np.argmin(dists))]

    return prototype_indices, cluster_labels


def find_nearest_svb(
    X_terra: np.ndarray,
    X_svb: np.ndarray,
    prototype_idx: int,
    n_nearest: int,
) -> np.ndarray:
    """Return indices of the n_nearest SVB rows to a Terra prototype row."""
    proto_vec = X_terra[prototype_idx]
    dists = np.linalg.norm(X_svb - proto_vec, axis=1)
    # argsort ascending; take top-n
    nearest = np.argsort(dists)[:n_nearest]
    return nearest


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------


def _safe_get(df: "pl.DataFrame", col: str, idx: int) -> float | None:
    """Safely extract a scalar value from a polars DataFrame by row index."""
    if col not in df.columns:
        return None
    val = df[col][idx]
    if val is None:
        return None
    try:
        v = float(val)
        return None if (v != v) else round(v, 4)  # NaN check
    except (TypeError, ValueError):
        return None


def build_result_rows(
    terra_sdf: "pl.DataFrame",
    svb_sdf: "pl.DataFrame",
    X_terra_scaled: np.ndarray,
    X_svb_scaled: np.ndarray,
    prototype_indices: np.ndarray,
    cluster_labels: np.ndarray,
    feature_cols: list[str],
    n_nearest: int,
    k: int,
) -> list[dict]:
    """Build one output row per (prototype, SVB match) pair."""
    rows = []

    # Columns to report beyond the SHAP features
    _EXTRA_COLS = [
        "label_arb_q10000_5m_gt0bps",
        "label_arb_q50000_5m_gt0bps",
        "label_basis_usdc_1m_gt10bps",
        "net_profit_bps_q10000",
        "net_profit_bps_q50000",
        "cross_quote_basis_usdc_bps",
        "cross_quote_basis_maxabs_bps",
        "ts_1m_ns",
    ]

    for proto_rank, proto_idx in enumerate(prototype_indices):
        # Terra prototype info
        cluster_size = int((cluster_labels == proto_rank).sum())
        proto_row: dict = {
            "prototype_rank": proto_rank + 1,
            "cluster_size": cluster_size,
            "source": "terra_luna_2022",
            "type": "prototype",
            "match_rank": 0,
            "dist_to_prototype": 0.0,
        }
        for feat in feature_cols:
            proto_row[f"feat_{feat}"] = _safe_get(terra_sdf, feat, proto_idx)
        for col in _EXTRA_COLS:
            proto_row[col] = _safe_get(terra_sdf, col, proto_idx)

        rows.append(proto_row)

        # SVB nearest matches
        svb_nearest_idxs = find_nearest_svb(
            X_terra_scaled, X_svb_scaled, proto_idx, n_nearest
        )
        for match_rank, svb_idx in enumerate(svb_nearest_idxs, start=1):
            dist = float(
                np.linalg.norm(X_terra_scaled[proto_idx] - X_svb_scaled[svb_idx])
            )
            svb_row: dict = {
                "prototype_rank": proto_rank + 1,
                "cluster_size": cluster_size,
                "source": "usdc_svb_2023",
                "type": "svb_match",
                "match_rank": match_rank,
                "dist_to_prototype": round(dist, 4),
            }
            for feat in feature_cols:
                svb_row[f"feat_{feat}"] = _safe_get(svb_sdf, feat, svb_idx)
            for col in _EXTRA_COLS:
                svb_row[col] = _safe_get(svb_sdf, col, svb_idx)

            rows.append(svb_row)

    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(
    rows: list[dict],
    feature_cols: list[str],
    k: int,
    output_path: Path,
) -> None:
    """4×2 grid: each row = one prototype Terra window + its nearest SVB match.

    Left column: Terra/LUNA prototype.
    Right column: Nearest SVB window.
    x-axis: feature values (horizontal bar chart).
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Palette
    _TERRA_COLOR = "#d73027"  # red for Terra
    _SVB_COLOR = "#2166ac"  # blue for SVB

    # Group rows by prototype_rank
    prototypes: dict[int, dict] = {}
    svb_matches: dict[int, list[dict]] = {}
    for r in rows:
        pr = r["prototype_rank"]
        if r["type"] == "prototype":
            prototypes[pr] = r
        else:
            svb_matches.setdefault(pr, []).append(r)

    # Pretty feature labels
    feat_labels = [_PRETTY_NAMES.get(f, f) for f in feature_cols]

    n_rows = k
    n_cols = 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11, 2.6 * n_rows),
        sharey=True,
    )
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx in range(k):
        pr = row_idx + 1
        proto = prototypes.get(pr)
        svb_nearest = svb_matches.get(pr, [{}])[0]  # first (nearest) SVB match

        for col_idx, (record, color, title_prefix, split_label) in enumerate(
            [
                (proto, _TERRA_COLOR, f"Prototype {pr}", "Terra/LUNA val"),
                (svb_nearest, _SVB_COLOR, f"SVB match (proto {pr})", "USDC/SVB test"),
            ]
        ):
            ax = axes[row_idx, col_idx]
            if not record:
                ax.set_visible(False)
                continue

            vals = []
            for feat in feature_cols:
                v = record.get(f"feat_{feat}")
                vals.append(float(v) if v is not None else 0.0)

            y_pos = range(len(feat_labels) - 1, -1, -1)
            ax.barh(list(y_pos), vals, color=color, alpha=0.78, height=0.65)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(feat_labels, fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

            # Build subtitle with key metrics
            exec_label_val = record.get("label_arb_q10000_5m_gt0bps")
            basis_val = record.get("cross_quote_basis_usdc_bps")
            net_val = record.get("net_profit_bps_q10000")

            exec_str = (
                "exec=1"
                if exec_label_val == 1
                else ("exec=0" if exec_label_val == 0 else "exec=?")
            )
            basis_str = f"basis={basis_val:.1f}bps" if basis_val is not None else ""
            net_str = f"net={net_val:.1f}bps" if net_val is not None else ""

            subtitle_parts = [exec_str]
            if basis_str:
                subtitle_parts.append(basis_str)
            if net_str:
                subtitle_parts.append(net_str)
            subtitle = "  ".join(subtitle_parts)

            ax.set_title(
                f"{title_prefix}\n({split_label})  {subtitle}",
                fontsize=8,
                loc="left",
                pad=4,
            )
            ax.set_xlabel("Feature value", fontsize=7)

    # Legend
    legend_patches = [
        mpatches.Patch(color=_TERRA_COLOR, label="Terra/LUNA prototype"),
        mpatches.Patch(color=_SVB_COLOR, label="Nearest SVB window"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=9,
        framealpha=0.85,
        edgecolor="lightgrey",
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"SHAP-Space Stress Prototypes: Terra/LUNA → SVB Transfer\n"
        f"(top-{len(feature_cols)} SHAP features, K={k} prototypes)",
        fontsize=10,
        y=1.01,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] Saved → {output_path}", flush=True)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        print("[warn] No results to write.", file=sys.stderr)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] Saved {len(rows)} rows → {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(rows: list[dict], feature_cols: list[str]) -> None:
    print("\n" + "=" * 72)
    print("SHAP PROTOTYPE SUMMARY")
    print("=" * 72)

    prototypes = [r for r in rows if r["type"] == "prototype"]
    svb_matches = [r for r in rows if r["type"] == "svb_match"]

    print(f"  Prototypes found: {len(prototypes)}")
    print(f"  SVB matches total: {len(svb_matches)}")
    print()

    for p in prototypes:
        pr = p["prototype_rank"]
        cs = p["cluster_size"]
        exec_v = p.get("label_arb_q10000_5m_gt0bps")
        basis_v = p.get("cross_quote_basis_usdc_bps")
        net_v = p.get("net_profit_bps_q10000")
        print(
            f"  Prototype {pr}  cluster_size={cs}  "
            f"exec_label={exec_v}  basis={basis_v}  net_profit={net_v}"
        )
        matches = [r for r in svb_matches if r["prototype_rank"] == pr]
        for m in matches:
            m_exec = m.get("label_arb_q10000_5m_gt0bps")
            m_basis = m.get("cross_quote_basis_usdc_bps")
            m_net = m.get("net_profit_bps_q10000")
            m_dist = m["dist_to_prototype"]
            print(
                f"    SVB match {m['match_rank']}  dist={m_dist:.3f}  "
                f"exec_label={m_exec}  basis={m_basis}  net_profit={m_net}"
            )
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # 1. Load data -----------------------------------------------------------
    df = load_dataset(args.data_dir)

    # 2. Load top-N SHAP features --------------------------------------------
    feature_cols = load_shap_features(args.shap_csv, args.top_n_shap)

    # 3. Extract Terra/LUNA and SVB feature matrices -------------------------
    from sklearn.preprocessing import StandardScaler

    X_terra_raw, terra_sdf = extract_feature_matrix(df, feature_cols, "validation")
    X_svb_raw, svb_sdf = extract_feature_matrix(df, feature_cols, "test")

    if X_terra_raw.shape[0] == 0:
        print(
            "[ERROR] No Terra/LUNA rows found (split='validation').  "
            "Check that dataset.parquet contains the validation split.",
            file=sys.stderr,
        )
        sys.exit(1)

    if X_svb_raw.shape[0] == 0:
        print(
            "[ERROR] No SVB rows found (split='test').  "
            "Check that dataset.parquet contains the test split.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 4. Scale features (fit on Terra, apply to both) -------------------------
    scaler = StandardScaler()
    X_terra_scaled = scaler.fit_transform(X_terra_raw)
    X_svb_scaled = scaler.transform(X_svb_raw)

    print(
        f"[scale] StandardScaler fit on Terra ({X_terra_scaled.shape[0]:,} rows).  "
        f"Applied to SVB ({X_svb_scaled.shape[0]:,} rows).",
        flush=True,
    )

    # 5. KMeans prototypes ---------------------------------------------------
    k = args.k
    print(f"[kmeans] Fitting KMeans(k={k}) on Terra/LUNA …", flush=True)
    prototype_indices, cluster_labels = find_prototypes(X_terra_scaled, k, args.seed)

    cluster_sizes = [(cluster_labels == c).sum() for c in range(k)]
    for c, (pidx, csize) in enumerate(zip(prototype_indices, cluster_sizes)):
        print(
            f"  Cluster {c+1}: {csize:,} rows  " f"prototype row_idx={pidx}",
            flush=True,
        )

    # 6. Build result rows ---------------------------------------------------
    rows = build_result_rows(
        terra_sdf=terra_sdf,
        svb_sdf=svb_sdf,
        X_terra_scaled=X_terra_scaled,
        X_svb_scaled=X_svb_scaled,
        prototype_indices=prototype_indices,
        cluster_labels=cluster_labels,
        feature_cols=feature_cols,
        n_nearest=args.n_nearest,
        k=k,
    )

    # 7. Write CSV -----------------------------------------------------------
    csv_path = args.output_dir / "shap_prototypes.csv"
    write_csv(rows, csv_path)

    # 8. Make figure ---------------------------------------------------------
    fig_path = args.figures_dir / "figure_shap_prototypes.png"
    try:
        make_figure(rows, feature_cols, k, fig_path)
    except Exception as exc:
        print(
            f"[warn] Could not produce figure: {exc}  "
            "(matplotlib may not be available in this environment)",
            file=sys.stderr,
        )

    # 9. Summary -------------------------------------------------------------
    print_summary(rows, feature_cols)


if __name__ == "__main__":
    main()
