#!/usr/bin/env python3
"""Run the full Stablecoin StressBench experiment grid.

Iterates over all (task × feature_set × model) combinations, trains on the
training split, evaluates on the test split, and writes one CSV per task.

Usage:
    # Full grid (all tasks × all feature sets × all models)
    python scripts/run_experiments.py --data-dir data/gold

    # Subset: specific tasks and feature sets
    python scripts/run_experiments.py \\
        --data-dir data/gold \\
        --tasks basis_1m_gt10bps basis_usdc_1m_gt10bps executable_arb_q10000_5m \\
        --feature-sets price_only price_plus_book all \\
        --models no_trade logistic lasso rf

    # Include oracle (requires realized net profit in the dataset)
    python scripts/run_experiments.py --data-dir data/gold --include-oracle

Output:
    results/experiments/{task_name}__results.csv   — one row per (model, feature_set)
    results/experiments/all_results.csv            — combined table across all tasks
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from stressbench.common.logging import get_logger
from stressbench.experiments.feature_sets import FEATURE_SETS
from stressbench.experiments.tasks import TASKS
from stressbench.experiments.experiment_runner import run_experiment

logger = get_logger(__name__)

_DEFAULT_MODELS = [
    "no_trade",
    "price_threshold_10bps",
    "last_value",
    "rolling_mean",
    "ar1",
    "logistic",
    "ridge",
    "lasso",
    "lgbm",
    "xgb",
    "rf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark experiment grid.")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--output-dir", default="results/experiments")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Task names. Defaults to all. Available: {list(TASKS)}",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="*",
        default=None,
        help=f"Feature set names. Defaults to all. Available: {list(FEATURE_SETS)}",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=_DEFAULT_MODELS,
        help="Model names to run.",
    )
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Include NetProfitOracleUpperBound (uses future net profit — upper bound only).",
    )
    parser.add_argument(
        "--model-save-dir",
        default=None,
        help="If set, save fitted model pickles here.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _make_model(name: str, df_cols: list[str]):
    """Instantiate a model by name."""
    from stressbench.models.rule_baselines import (
        NoTradeBaseline,
        PriceBasisThresholdBaseline,
        GrossArbThresholdBaseline,
    )

    if name == "no_trade":
        return NoTradeBaseline()
    if name == "price_threshold_10bps":
        col_idx = df_cols.index("cross_quote_basis_primary_bps") if "cross_quote_basis_primary_bps" in df_cols else 0
        return PriceBasisThresholdBaseline(col_index=col_idx, threshold_bps=10.0)
    if name == "price_threshold_25bps":
        col_idx = df_cols.index("cross_quote_basis_primary_bps") if "cross_quote_basis_primary_bps" in df_cols else 0
        return PriceBasisThresholdBaseline(col_index=col_idx, threshold_bps=25.0)
    if name == "gross_arb_threshold":
        col_idx = df_cols.index("cross_quote_basis_maxabs_bps") if "cross_quote_basis_maxabs_bps" in df_cols else 0
        return GrossArbThresholdBaseline(col_index=col_idx, threshold_bps=20.0)
    if name == "last_value":
        from stressbench.models.baselines import LastValueBaseline
        return LastValueBaseline()
    if name == "rolling_mean":
        from stressbench.models.baselines import RollingMeanBaseline
        return RollingMeanBaseline()
    if name == "ar1":
        from stressbench.models.baselines import AR1Baseline
        return AR1Baseline()
    if name == "logistic":
        from stressbench.models.baselines import LogisticBaseline
        return LogisticBaseline()
    if name == "ridge":
        from stressbench.models.baselines import RidgeBaseline
        return RidgeBaseline()
    if name == "lasso":
        from stressbench.models.baselines import LassoBaseline
        return LassoBaseline()
    if name == "lgbm":
        from stressbench.models.tree_models import LGBMWrapper
        return LGBMWrapper(task="classification")
    if name == "xgb":
        from stressbench.models.tree_models import XGBWrapper
        return XGBWrapper(task="classification")
    if name == "rf":
        from stressbench.models.tree_models import RandomForestWrapper
        return RandomForestWrapper(task="classification")
    raise ValueError(f"Unknown model: {name!r}")


def _flatten_result(result: dict) -> dict:
    """Flatten a nested backtest result dict into one CSV row."""
    ml = result.get("ml_metrics", {})
    econ = result.get("economic_metrics", {})
    return {
        "task": result.get("task_name", result.get("task", "")),
        "feature_set": result.get("feature_set", ""),
        "model": result.get("model", ""),
        "n_train": result.get("n_train", ""),
        "n_test": result.get("n_test", ""),
        # ML metrics
        "auroc": ml.get("auroc", ""),
        "auprc": ml.get("auprc", ""),
        "f1": ml.get("f1", ""),
        "balanced_accuracy": ml.get("balanced_accuracy", ""),
        "brier_score": ml.get("brier_score", ""),
        # Economic metrics
        "net_bps_captured": econ.get("net_bps_captured", ""),
        "hit_rate_above_cost": econ.get("hit_rate_above_cost", ""),
        "false_positive_cost": econ.get("false_positive_cost", ""),
        "n_trades": econ.get("n_trades", ""),
        "final_pnl_usd": econ.get("final_pnl_usd", ""),
        "max_drawdown_usd": econ.get("max_drawdown_usd", ""),
        "sharpe_ratio": econ.get("sharpe_ratio", ""),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = args.tasks or list(TASKS.keys())
    feat_sets_to_run = args.feature_sets or list(FEATURE_SETS.keys())

    dataset_path = Path(args.data_dir) / "dataset.parquet"
    if not dataset_path.exists():
        # Fall back to glob for partitioned datasets
        dataset_path = Path(args.data_dir)

    all_rows: list[dict] = []

    for task_name in tasks_to_run:
        if task_name not in TASKS:
            logger.warning("Unknown task '%s'; skipping.", task_name)
            continue

        task_rows: list[dict] = []
        logger.info("=== Task: %s ===", task_name)

        for fs_name in feat_sets_to_run:
            if fs_name not in FEATURE_SETS:
                logger.warning("Unknown feature set '%s'; skipping.", fs_name)
                continue

            logger.info("  Feature set: %s", fs_name)

            models_to_run = list(args.models)
            if args.include_oracle:
                models_to_run.append("oracle")

            for model_name in models_to_run:
                if args.dry_run:
                    logger.info("  [DRY RUN] Would run: %s / %s / %s", task_name, fs_name, model_name)
                    continue

                try:
                    if model_name == "oracle":
                        from stressbench.models.rule_baselines import NetProfitOracleUpperBound
                        model = NetProfitOracleUpperBound(threshold_bps=0.0)
                    else:
                        # Feature cols are resolved inside run_experiment; pass dummy list here
                        model = _make_model(model_name, [])

                    result = run_experiment(
                        dataset_path=dataset_path,
                        task_name=task_name,
                        feature_set_name=fs_name,
                        model=model,
                        model_name=model_name,
                        model_dir=args.model_save_dir,
                    )
                    row = _flatten_result(result)
                    task_rows.append(row)
                    all_rows.append(row)
                    logger.info(
                        "    %s: AUROC=%.3f  net_bps=%.1f  n_trades=%s",
                        model_name,
                        row.get("auroc") or float("nan"),
                        row.get("net_bps_captured") or float("nan"),
                        row.get("n_trades", "—"),
                    )
                except Exception as exc:
                    logger.error(
                        "  FAILED: %s / %s / %s — %s", task_name, fs_name, model_name, exc
                    )

        if task_rows:
            task_csv = output_dir / f"{task_name}__results.csv"
            _write_csv(task_rows, task_csv)
            logger.info("Wrote %s (%d rows)", task_csv, len(task_rows))

    if all_rows:
        all_csv = output_dir / "all_results.csv"
        _write_csv(all_rows, all_csv)
        logger.info("Combined results: %s (%d rows)", all_csv, len(all_rows))

    logger.info("Experiment grid complete — %s", datetime.now(timezone.utc).isoformat())


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
