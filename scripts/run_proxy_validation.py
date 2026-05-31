#!/usr/bin/env python3
"""Kline-proxy depth validation against real L2 snapshots.

The benchmark uses kline-proxy depth for the BTC-USDC buy leg in the 2023 SVB
window.  2024 control windows have real L2 depth.  This script asks: if we had
used the kline proxy instead of real L2 during those 2024 windows, how wrong
would the labels have been?

Method
------
For each 2024 row that already has real L2 depth (depth_source in
{'real_l2_snapshot', 'real_l2_incremental'}):

1. Compute arbitrage labels from the *real* net_profit_bps columns
   (pre-computed in dataset.parquet, no re-simulation needed).

2. Simulate the kline proxy by multiplying the buy-leg depth columns
   (depth_bid_10bp_mean, depth_ask_10bp_mean) by a retention factor:
   1.0 - scale_factor.  Four scale factors are tried: 0.20, 0.40, 0.60, 0.80
   (i.e. retaining 80%, 60%, 40%, 20% of real depth).

   When effective depth < notional, execution is infeasible: we set
   net_profit_bps_q{Q} to NaN for that row to match the real labelling
   convention for depth-limited windows.

3. Re-derive forward-looking arbitrage labels from the depth-adjusted
   net_profit_bps columns.

4. Report per-scale-factor:
   - label_agreement     : fraction of rows where real == proxy label (excl. nulls)
   - precision_proxy     : of proxy=1, fraction that are real=1
   - recall_proxy        : of real=1, fraction that are proxy=1
   - f1_proxy            : harmonic mean of precision/recall
   - exec_rate_real      : fraction of rows with real label = 1
   - exec_rate_proxy     : fraction of rows with proxy label = 1
   - exec_rate_diff      : exec_rate_proxy - exec_rate_real
   - oracle_return_real  : mean net_profit_bps for real=1 rows
   - oracle_return_proxy : mean net_profit_bps (real column) for proxy=1 rows
   - oracle_return_diff  : oracle_return_proxy - oracle_return_real

Outputs
-------
results/experiments_addon/proxy_validation.csv

Usage
-----
    python scripts/run_proxy_validation.py
    python scripts/run_proxy_validation.py --data-dir data/gold --output-dir results/experiments_addon
    python scripts/run_proxy_validation.py --task executable_arb_q10000_5m
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_L2_SOURCES = {"real_l2_snapshot", "real_l2_incremental"}

# Scale factors: fraction of real depth *removed* by the proxy
_SCALE_FACTORS = [0.20, 0.40, 0.60, 0.80]

# Notional sizes and the depth columns that gate their executability
_NOTIONAL_DEPTH_MAP: dict[int, float] = {
    10_000: 10_000.0,
    50_000: 50_000.0,
    100_000: 100_000.0,
    500_000: 500_000.0,
}

# Depth feature columns that represent buy-leg liquidity
_DEPTH_COLS = ["depth_bid_10bp_mean", "depth_ask_10bp_mean"]

# Net-profit columns that will be adjusted for depth constraints
_NET_PROFIT_TEMPLATES = {q: f"net_profit_bps_q{q}" for q in _NOTIONAL_DEPTH_MAP}

# Default evaluation task: executable_arb at $10k within 5 minutes
_DEFAULT_TASK = "executable_arb_q10000_5m"
_TASK_CONFIG = {
    "executable_arb_q10000_5m": {
        "label_col": "label_arb_q10000_5m_gt0bps",
        "net_profit_col": "net_profit_bps_q10000",
        "notional": 10_000,
        "horizon": "5m",
        "threshold_bps": 0.0,
    },
    "executable_arb_q50000_5m": {
        "label_col": "label_arb_q50000_5m_gt0bps",
        "net_profit_col": "net_profit_bps_q50000",
        "notional": 50_000,
        "horizon": "5m",
        "threshold_bps": 0.0,
    },
    "executable_arb_q10000_1m": {
        "label_col": "label_arb_q10000_1m_gt0bps",
        "net_profit_col": "net_profit_bps_q10000",
        "notional": 10_000,
        "horizon": "1m",
        "threshold_bps": 0.0,
    },
}

_HORIZON_NS = {"1m": 60_000_000_000, "5m": 300_000_000_000, "15m": 900_000_000_000}


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
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results" / "experiments_addon",
        help="Output directory for proxy_validation.csv.",
    )
    p.add_argument(
        "--task",
        choices=list(_TASK_CONFIG),
        default=_DEFAULT_TASK,
        help="Evaluation task (determines label column, notional, horizon).",
    )
    p.add_argument(
        "--split-filter",
        default="2024",
        help=(
            "String that must appear in the 'split' column value to select "
            "2024 windows.  Default: '2024'.  Pass 'train' to use training split."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(data_dir: Path) -> pl.DataFrame:
    """Load dataset.parquet with a clear error if it is missing."""
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


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_real_l2_2024(df: pl.DataFrame, split_filter: str) -> pl.DataFrame:
    """Return rows from 2024 splits that have real L2 depth.

    Rows from 2024 training windows where depth_source is real_l2_snapshot
    or real_l2_incremental are selected.  The split_filter string is matched
    case-insensitively against the split column.
    """
    if "depth_source" not in df.columns:
        print(
            "[warn] 'depth_source' column not found in dataset; "
            "cannot filter by depth provenance.  "
            "Proceeding with all rows matching split_filter.",
            file=sys.stderr,
        )
        mask = pl.col("split").str.contains(split_filter)
    else:
        mask = pl.col("split").str.contains(split_filter) & pl.col(
            "depth_source"
        ).is_in(list(_REAL_L2_SOURCES))

    out = df.filter(mask)
    n_total_2024 = df.filter(pl.col("split").str.contains(split_filter)).shape[0]
    print(
        f"[filter] split contains '{split_filter}': {n_total_2024:,} rows",
        flush=True,
    )
    if "depth_source" in df.columns:
        print(
            f"[filter] of those, real-L2 depth: {len(out):,} rows "
            f"({100*len(out)/max(n_total_2024,1):.1f}%)",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Proxy label derivation
# ---------------------------------------------------------------------------


def _compute_forward_label(
    df: pl.DataFrame,
    net_col: str,
    horizon_name: str,
    threshold_bps: float,
    ts_col: str = "ts_1m_ns",
) -> pl.Series:
    """Re-derive a forward-looking binary label from a (possibly modified) net_profit column.

    Uses the same reversed-frame rolling-max trick as arbitrage_labels.py.
    Returns an Int8 Series of 0/1/None values aligned to df row order.
    """
    if net_col not in df.columns:
        return pl.Series(name="proxy_label", values=[None] * len(df), dtype=pl.Int8)

    work = df.select([ts_col, net_col]).with_row_index("_row_idx")

    # Replace NaN with null so rolling_max ignores depth-limited rows
    work = work.with_columns(pl.col(net_col).fill_nan(None).alias("_clean_net"))

    ts_max = work[ts_col].max()
    ts_min = work[ts_col].min()

    rev = work.sort(ts_col, descending=True).with_columns(
        (pl.lit(ts_max + ts_min) - pl.col(ts_col))
        .cast(pl.Datetime("ns"))
        .alias("_ts_proxy")
    )

    rev = rev.with_columns(
        pl.col("_clean_net")
        .rolling_max_by(
            by="_ts_proxy",
            window_size=horizon_name,
            closed="right",
            min_periods=1,
        )
        .alias("_max_net")
    )

    rev = rev.with_columns(
        pl.when(pl.col("_max_net").is_null())
        .then(pl.lit(None).cast(pl.Int8))
        .when(pl.col("_max_net") > threshold_bps)
        .then(pl.lit(1).cast(pl.Int8))
        .otherwise(pl.lit(0).cast(pl.Int8))
        .alias("_label")
    )

    result = rev.sort("_row_idx").get_column("_label")
    return result


def apply_depth_scaling(
    df: pl.DataFrame,
    net_col: str,
    notional: float,
    scale_factor: float,
) -> pl.DataFrame:
    """Return df with net_profit column zeroed out where scaled depth < notional.

    The kline proxy underestimates real depth.  Simulate this by multiplying
    both depth columns by (1 - scale_factor).  When the *minimum* of the two
    scaled depth columns falls below the required notional, execution is
    infeasible: set that row's net_profit_bps to NaN (matching the convention
    in arbitrage_labels.py for depth-limited rows).

    Args:
        df:           DataFrame (rows already filtered to 2024/real-L2 windows).
        net_col:      Name of the net_profit column to modify.
        notional:     Required execution notional in USD.
        scale_factor: Fraction of depth to *remove* (0.20 = 80% retained).

    Returns:
        New DataFrame with a ``proxy_net`` column replacing depth-limited rows
        with NaN.  Original columns are not mutated.
    """
    retention = 1.0 - scale_factor

    depth_available = False
    scaled_min_expr = None

    present_depth_cols = [c for c in _DEPTH_COLS if c in df.columns]
    if present_depth_cols:
        depth_available = True
        # Minimum of scaled depth columns: represents the tightest constraint
        exprs = [pl.col(c) * retention for c in present_depth_cols]
        scaled_min_expr = exprs[0]
        for e in exprs[1:]:
            scaled_min_expr = (
                pl.when(scaled_min_expr < e).then(scaled_min_expr).otherwise(e)
            )

    if not depth_available or net_col not in df.columns:
        # Cannot simulate depth gating; return net column unchanged
        df = df.with_columns(pl.col(net_col).alias("proxy_net"))
        return df

    # Where scaled depth < notional: proxy_net = NaN (infeasible)
    df = df.with_columns(
        pl.when(scaled_min_expr < notional)
        .then(pl.lit(float("nan")))
        .otherwise(pl.col(net_col))
        .alias("proxy_net")
    )
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    real_labels: np.ndarray,
    proxy_labels: np.ndarray,
    real_net: np.ndarray,
    scale_factor: float,
    task_name: str,
) -> dict:
    """Compute agreement, precision, recall, F1, exec-rate, and oracle-return metrics."""
    # Exclude rows where either label is null/nan
    valid = ~(
        np.isnan(real_labels.astype(float)) | np.isnan(proxy_labels.astype(float))
    )
    r = real_labels[valid].astype(int)
    p = proxy_labels[valid].astype(int)
    net_v = real_net[valid]

    n = len(r)
    if n == 0:
        return {
            "task": task_name,
            "scale_factor": scale_factor,
            "depth_retained_pct": round((1 - scale_factor) * 100, 0),
            "n_rows": 0,
            "label_agreement": float("nan"),
            "precision_proxy": float("nan"),
            "recall_proxy": float("nan"),
            "f1_proxy": float("nan"),
            "exec_rate_real": float("nan"),
            "exec_rate_proxy": float("nan"),
            "exec_rate_diff": float("nan"),
            "oracle_return_real_bps": float("nan"),
            "oracle_return_proxy_bps": float("nan"),
            "oracle_return_diff_bps": float("nan"),
        }

    agreement = float(np.mean(r == p))

    # Precision: of proxy=1, fraction that are real=1
    proxy_pos = p == 1
    if proxy_pos.sum() > 0:
        precision = float(np.mean(r[proxy_pos] == 1))
    else:
        precision = float("nan")

    # Recall: of real=1, fraction that are proxy=1
    real_pos = r == 1
    if real_pos.sum() > 0:
        recall = float(np.mean(p[real_pos] == 1))
    else:
        recall = float("nan")

    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    exec_rate_real = float(r.mean())
    exec_rate_proxy = float(p.mean())

    # Oracle return: mean net_profit for rows signalled as executable
    # (always computed using the *real* net_profit to be an apples-to-apples
    # comparison of which windows were selected, not how the net_profit changes)
    net_clean = np.where(np.isnan(net_v), np.nan, net_v)

    real_returns = net_clean[real_pos] if real_pos.sum() > 0 else np.array([])
    proxy_returns = net_clean[proxy_pos] if proxy_pos.sum() > 0 else np.array([])

    oracle_return_real = (
        float(np.nanmean(real_returns)) if len(real_returns) > 0 else float("nan")
    )
    oracle_return_proxy = (
        float(np.nanmean(proxy_returns)) if len(proxy_returns) > 0 else float("nan")
    )

    oracle_return_diff = (
        oracle_return_proxy - oracle_return_real
        if not (np.isnan(oracle_return_real) or np.isnan(oracle_return_proxy))
        else float("nan")
    )

    return {
        "task": task_name,
        "scale_factor": scale_factor,
        "depth_retained_pct": round((1 - scale_factor) * 100, 0),
        "n_rows": n,
        "label_agreement": round(agreement, 4),
        "precision_proxy": (
            round(precision, 4) if not np.isnan(precision) else float("nan")
        ),
        "recall_proxy": round(recall, 4) if not np.isnan(recall) else float("nan"),
        "f1_proxy": round(f1, 4) if not np.isnan(f1) else float("nan"),
        "exec_rate_real": round(exec_rate_real, 4),
        "exec_rate_proxy": round(exec_rate_proxy, 4),
        "exec_rate_diff": round(exec_rate_proxy - exec_rate_real, 4),
        "oracle_return_real_bps": (
            round(oracle_return_real, 2)
            if not np.isnan(oracle_return_real)
            else float("nan")
        ),
        "oracle_return_proxy_bps": (
            round(oracle_return_proxy, 2)
            if not np.isnan(oracle_return_proxy)
            else float("nan")
        ),
        "oracle_return_diff_bps": (
            round(oracle_return_diff, 2)
            if not np.isnan(oracle_return_diff)
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("[warn] No results to write.", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] Saved {len(rows)} rows → {output_path}", flush=True)


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("PROXY VALIDATION SUMMARY")
    print("=" * 72)
    print(
        f"{'Scale':>7}  {'Retained':>9}  {'Agreement':>10}  "
        f"{'Precision':>10}  {'Recall':>8}  {'F1':>6}  "
        f"{'ExecDiff':>9}  {'OracleDiff':>11}"
    )
    print("-" * 72)
    for r in rows:
        sf = r["scale_factor"]
        ret = r["depth_retained_pct"]
        ag = r["label_agreement"]
        pr = r["precision_proxy"]
        rc = r["recall_proxy"]
        f1 = r["f1_proxy"]
        ed = r["exec_rate_diff"]
        od = r["oracle_return_diff_bps"]

        def _fmt(v: float, decimals: int = 3) -> str:
            return (
                f"{v:.{decimals}f}"
                if not (isinstance(v, float) and np.isnan(v))
                else "  nan"
            )

        print(
            f"{sf:>7.0%}  {ret:>8.0f}%  {_fmt(ag):>10}  "
            f"{_fmt(pr):>10}  {_fmt(rc):>8}  {_fmt(f1):>6}  "
            f"{_fmt(ed, 4):>9}  {_fmt(od, 1):>11}"
        )
    print("=" * 72)
    print(
        "Interpretation:\n"
        "  scale_factor  = fraction of real depth removed by kline proxy\n"
        "  label_agreement = fraction of windows where real and proxy labels agree\n"
        "  precision_proxy = of proxy=1, fraction are actually executable (real=1)\n"
        "  recall_proxy    = of real=1, fraction correctly identified by proxy\n"
        "  exec_rate_diff  = proxy executable rate minus real executable rate\n"
        "  oracle_return_diff_bps = mean net bps difference (proxy selected vs real)\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    task_cfg = _TASK_CONFIG[args.task]
    label_col = task_cfg["label_col"]
    net_col = task_cfg["net_profit_col"]
    notional = task_cfg["notional"]
    horizon = task_cfg["horizon"]
    threshold_bps = task_cfg["threshold_bps"]

    print(f"[config] Task: {args.task}", flush=True)
    print(f"[config] Label col: {label_col}", flush=True)
    print(f"[config] Net profit col: {net_col}", flush=True)
    print(f"[config] Notional: ${notional:,}", flush=True)
    print(f"[config] Horizon: {horizon}", flush=True)
    print(f"[config] Split filter: '{args.split_filter}'", flush=True)

    # 1. Load ----------------------------------------------------------------
    df = load_dataset(args.data_dir)

    # 2. Filter to 2024 real-L2 windows -------------------------------------
    df_2024 = filter_real_l2_2024(df, args.split_filter)

    if df_2024.is_empty():
        print(
            f"[warn] No rows matched split_filter='{args.split_filter}' with "
            "real-L2 depth.  Check that the dataset contains 2024 training windows.",
            file=sys.stderr,
        )
        sys.exit(0)

    # Validate required columns
    ts_col = "ts_1m_ns"
    if ts_col not in df_2024.columns:
        # Fall back to ts_ns
        if "ts_ns" in df_2024.columns:
            ts_col = "ts_ns"
        else:
            print(
                "[ERROR] Neither 'ts_1m_ns' nor 'ts_ns' found in dataset.",
                file=sys.stderr,
            )
            sys.exit(1)

    if label_col not in df_2024.columns:
        print(
            f"[ERROR] Label column '{label_col}' not found in dataset.\n"
            f"  Available label columns: "
            f"{[c for c in df_2024.columns if c.startswith('label_')]}",
            file=sys.stderr,
        )
        sys.exit(1)

    if net_col not in df_2024.columns:
        print(
            f"[ERROR] Net profit column '{net_col}' not found in dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Extract real labels and net profit ---------------------------------
    # Real labels are read directly from pre-computed columns
    real_label_series = df_2024[label_col]
    real_net_series = df_2024[net_col]

    real_labels_np = real_label_series.cast(pl.Float32).to_numpy().astype(float)
    real_net_np = real_net_series.cast(pl.Float32).to_numpy().astype(float)

    real_label_rate = float(np.nanmean(real_labels_np == 1))
    print(
        f"[real] Real label prevalence (exec_rate): {real_label_rate:.3%}  "
        f"n_rows={len(df_2024):,}",
        flush=True,
    )

    # 4. Iterate over scale factors -----------------------------------------
    rows: list[dict] = []
    for sf in _SCALE_FACTORS:
        print(f"[proxy] Scale factor {sf:.0%}  (retained {(1-sf):.0%}) …", flush=True)

        # Apply depth scaling to produce a modified net_profit column
        df_scaled = apply_depth_scaling(df_2024, net_col, notional, sf)

        # Re-derive forward-looking labels from the scaled net_profit.
        # Rename proxy_net to a unique column name so _compute_forward_label
        # does not conflict with the original net_col in the DataFrame.
        _proxy_col = net_col + "__proxy"
        df_for_labels = df_scaled.rename({"proxy_net": _proxy_col})
        proxy_label_series = _compute_forward_label(
            df_for_labels,
            net_col=_proxy_col,
            horizon_name=horizon,
            threshold_bps=threshold_bps,
            ts_col=ts_col,
        )

        proxy_labels_np = proxy_label_series.cast(pl.Float32).to_numpy().astype(float)

        metrics = compute_metrics(
            real_labels_np,
            proxy_labels_np,
            real_net_np,
            sf,
            args.task,
        )
        rows.append(metrics)

        print(
            f"  agreement={metrics['label_agreement']:.3f}  "
            f"precision={metrics['precision_proxy']:.3f}  "
            f"recall={metrics['recall_proxy']:.3f}  "
            f"exec_rate_diff={metrics['exec_rate_diff']:+.4f}",
            flush=True,
        )

    # 5. Write results -------------------------------------------------------
    output_path = args.output_dir / "proxy_validation.csv"
    write_csv(rows, output_path)
    print_summary(rows)


if __name__ == "__main__":
    main()
