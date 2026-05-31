#!/usr/bin/env python3
"""Loss attribution analysis for the basis_usdc_1m_gt10bps task.

Reads results/experiments/all_results.csv and decomposes each model's
underperformance relative to the oracle into three components:

  1. FP loss         — losses incurred from false-positive trades
                       (directly from the false_positive_cost column)
  2. Slippage loss   — modelled as the spread between average net profit on
                       winning trades vs the oracle rate; proxy for execution
                       degradation on legitimate signals
  3. Signal absence  — opportunity cost: trades the oracle would take but the
                       model abstains from (oracle_n_trades - model_n_trades)
                       scaled by the oracle's per-trade bps

Output
------
results/paper_addon/table_loss_attribution.csv
    Columns: model, feature_set, net_bps, n_trades, fp_loss_bps,
             signal_absence_bps

Usage
-----
    python scripts/make_loss_attribution.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_RESULTS = Path("results/experiments/all_results.csv")
_OUT_CSV = Path("results/paper_addon/table_loss_attribution.csv")
TASK = "basis_usdc_1m_gt10bps"
ORACLE_NET_BPS = 161.72755272090245  # from all_results oracle row
ORACLE_N_TRADES = 316  # oracle trades in test set

_OUT_FIELDS = [
    "model",
    "feature_set",
    "net_bps",
    "n_trades",
    "fp_loss_bps",
    "signal_absence_bps",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_results(csv_path: Path) -> list[dict]:
    """Load all rows from all_results.csv as a list of dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"all_results.csv not found at {csv_path}. "
            "Run scripts/run_experiments.py first."
        )
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val: str, default: float = float("nan")) -> float:
    """Parse a CSV string value to float, returning `default` on failure."""
    if val is None or val.strip() in ("", "nan", "NaN", "None"):
        return default
    try:
        return float(val)
    except ValueError:
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Loss attribution computation
# ---------------------------------------------------------------------------


def compute_loss_attribution(rows: list[dict]) -> list[dict]:
    """Compute loss attribution for each model on the target task.

    Only includes models that have attempted trades (n_trades > 0) or that
    returned a net_bps_captured value, excluding the oracle and no_trade rows.

    Attribution components
    ----------------------
    fp_loss_bps
        = false_positive_cost (per-trade mean bps on false-positive trades)
          This is already reported in the CSV; we surface it directly.
          NaN if the model had zero FP trades.

    signal_absence_bps
        = opportunity cost of missing oracle-reachable trades.
        Modelled as:
          (oracle_n_trades - model_n_trades) * oracle_per_trade_bps / oracle_n_trades
        where oracle_per_trade_bps = ORACLE_NET_BPS.
        Capped at ORACLE_NET_BPS (can't have negative signal-absence).
        If n_trades > oracle_n_trades (model overtrades), signal_absence = 0.
    """
    oracle_per_trade_bps = ORACLE_NET_BPS  # average bps per oracle trade

    attribution_rows: list[dict] = []

    seen_keys: set[tuple] = set()  # deduplicate across feature sets where appropriate

    for row in rows:
        if row["task"] != TASK:
            continue

        model = row["model"]
        feature_set = row["feature_set"]

        if model in ("oracle", "no_trade"):
            continue

        net_bps = safe_float(row["net_bps_captured"])
        n_trades = safe_int(row["n_trades"])
        fp_cost = safe_float(row["false_positive_cost"])  # negative bps (mean FP trade)

        # Signal absence: oracle trades the model didn't take
        missing_trades = max(0, ORACLE_N_TRADES - n_trades)
        # Opportunity cost in bps (relative to total position = oracle_n_trades)
        signal_absence_bps = (missing_trades / ORACLE_N_TRADES) * oracle_per_trade_bps

        # FP loss: expose false_positive_cost directly (it is already a loss, i.e. negative)
        fp_loss_bps = fp_cost  # e.g. -293 bps for price_threshold_10bps

        attribution_rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "net_bps": round(net_bps, 2) if not math.isnan(net_bps) else "nan",
                "n_trades": n_trades,
                "fp_loss_bps": (
                    round(fp_loss_bps, 2) if not math.isnan(fp_loss_bps) else "nan"
                ),
                "signal_absence_bps": round(signal_absence_bps, 2),
            }
        )

    return attribution_rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_attribution_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] Wrote {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Formatted table
# ---------------------------------------------------------------------------


def print_formatted_table(rows: list[dict]) -> None:
    """Print a readable loss attribution table to stdout."""
    print()
    print("=" * 90)
    print(f"LOSS ATTRIBUTION — Task: {TASK}")
    print(
        f"Oracle benchmark: {ORACLE_NET_BPS:.2f} net bps  |  Oracle n_trades: {ORACLE_N_TRADES}"
    )
    print("=" * 90)

    hdr = (
        f"{'Model':<30} {'FeatureSet':<22} {'NetBps':>9} {'N_Trades':>9} "
        f"{'FP_Loss':>10} {'SigAbsence':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        net = r["net_bps"]
        fp = r["fp_loss_bps"]
        sa = r["signal_absence_bps"]

        net_s = (
            f"{net:>+.1f}"
            if isinstance(net, (int, float))
            and not math.isnan(float(str(net).replace("nan", "0")))
            else "   nan"
        )
        fp_s = (
            f"{fp:>+.1f}"
            if isinstance(fp, (int, float))
            and not math.isnan(float(str(fp).replace("nan", "0")))
            else "    nan"
        )
        sa_s = f"{sa:>+.1f}"

        try:
            net_s = f"{float(net):>+.1f}"
        except Exception:
            net_s = "   nan"
        try:
            fp_s = f"{float(fp):>+.1f}"
        except Exception:
            fp_s = "    nan"

        print(
            f"{r['model']:<30} {r['feature_set']:<22} {net_s:>9} "
            f"{r['n_trades']:>9} {fp_s:>10} {sa_s:>12}"
        )

    print("=" * len(hdr))
    print()

    # Narrative summary
    rows_with_trades = [
        r
        for r in rows
        if isinstance(r["net_bps"], str) is False or r["net_bps"] != "nan"
    ]
    rows_with_trades = [r for r in rows_with_trades if r["n_trades"] > 10]

    if rows_with_trades:
        # Model with smallest FP loss
        fp_vals = []
        for r in rows_with_trades:
            try:
                fp_vals.append((r["model"], r["feature_set"], float(r["fp_loss_bps"])))
            except Exception:
                pass
        if fp_vals:
            fp_vals_valid = [(m, fs, v) for m, fs, v in fp_vals if not math.isnan(v)]
            if fp_vals_valid:
                best_fp = max(fp_vals_valid, key=lambda x: x[2])  # least negative
                worst_fp = min(fp_vals_valid, key=lambda x: x[2])
                print(
                    f"Best FP cost:  {best_fp[0]} / {best_fp[1]}  → {best_fp[2]:+.1f} bps per FP trade"
                )
                print(
                    f"Worst FP cost: {worst_fp[0]} / {worst_fp[1]}  → {worst_fp[2]:+.1f} bps per FP trade"
                )
                print()

    print(
        "Interpretation:\n"
        "  fp_loss_bps    = mean bps on false-positive trades (negative = loss per FP trade)\n"
        "  signal_absence = opportunity cost of trades the oracle takes that the model misses\n"
        "                   computed as (oracle_n - model_n) / oracle_n × oracle_net_bps\n"
        "  Models with many trades may have low signal_absence but high fp_loss.\n"
        "  The oracle achieves signal_absence=0 by construction."
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"[main] Reading {_ALL_RESULTS} …")
    all_rows = load_all_results(_ALL_RESULTS)

    print(f"[main] Loaded {len(all_rows)} total rows; filtering task={TASK!r}")
    attribution_rows = compute_loss_attribution(all_rows)
    print(
        f"[main] Computed attribution for {len(attribution_rows)} model-feature-set pairs"
    )

    write_attribution_csv(attribution_rows, _OUT_CSV)
    print_formatted_table(attribution_rows)


if __name__ == "__main__":
    main()
