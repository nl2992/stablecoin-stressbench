"""Single-experiment runner for the Stablecoin StressBench grid.

Loads train/test data for a given (task, feature_set) pair, trains a model,
runs the backtest, and returns the result dict.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.evaluation.backtest import run_backtest
from stressbench.experiments.feature_sets import FEATURE_SETS
from stressbench.experiments.tasks import TASKS

logger = get_logger(__name__)

_EXCLUDE_COLS = {"split", "ts_1m_ns", "basis_primary_asset", "buy_venue", "sell_venue", "depth_source"}


def _resolve_feature_cols(
    df: pl.DataFrame,
    feature_set_name: str,
    label_col: str,
) -> list[str]:
    """Return the feature column list for a feature set, dropping absent columns."""
    requested = FEATURE_SETS.get(feature_set_name)
    if requested is None:
        # "all" mode: every non-label, non-metadata column
        return [
            c for c in df.columns
            if not c.startswith("label_") and c not in _EXCLUDE_COLS
        ]
    available = set(df.columns)
    missing = [c for c in requested if c not in available]
    if missing:
        logger.warning(
            "Feature set '%s': dropping %d absent columns: %s",
            feature_set_name, len(missing), missing,
        )
    return [c for c in requested if c in available]


def _load_split(
    df: pl.DataFrame,
    split: str,
    label_col: str,
    feature_cols: list[str],
    net_profit_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (X, y, y_net) arrays for a given split."""
    sdf = df.filter(pl.col("split") == split).filter(pl.col(label_col).is_not_null())
    if sdf[label_col].dtype.is_float():
        sdf = sdf.filter(~pl.col(label_col).is_nan())

    X_raw = sdf.select(feature_cols).to_numpy().astype(np.float32)
    y = sdf[label_col].to_numpy()

    # Impute NaN features: column median, all-NaN columns → 0
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        with np.errstate(all="ignore"):
            col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        X = np.where(nan_mask, col_medians[None, :], X_raw)
    else:
        X = X_raw

    # Net profit for economic scoring
    if net_profit_col and net_profit_col in sdf.columns:
        y_net_raw = sdf[net_profit_col].to_numpy().astype(np.float64)
        y_net = np.where(np.isnan(y_net_raw), -999.0, y_net_raw)
    else:
        y_net = np.zeros(len(y), dtype=np.float64)

    return X, y, y_net


def run_experiment(
    dataset_path: str | Path,
    task_name: str,
    feature_set_name: str,
    model: Any,
    model_name: str,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate one (task, feature_set, model) experiment.

    Args:
        dataset_path: Path to ``dataset.parquet`` (or a directory containing it).
        task_name: Key from :data:`~stressbench.experiments.tasks.TASKS`.
        feature_set_name: Key from
            :data:`~stressbench.experiments.feature_sets.FEATURE_SETS`.
        model: Unfitted model with ``fit`` / ``predict`` / ``predict_proba``.
        model_name: Display name for leaderboard rows.
        model_dir: Optional directory to save the fitted model as a pickle.

    Returns:
        Dict with keys: ``task``, ``feature_set``, ``model``, ``ml_metrics``,
        ``economic_metrics``, ``n_train``, ``n_test``.
    """
    task_cfg = TASKS.get(task_name)
    if task_cfg is None:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {list(TASKS)}")

    label_col = task_cfg["label"]
    ml_task = task_cfg["task"]
    notional_usd = task_cfg["notional_usd"]
    net_profit_col = task_cfg.get("net_profit_col")

    # Load dataset
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        parquet_files = list(dataset_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {dataset_path}")
        df = pl.read_parquet(str(dataset_path / "*.parquet"))
    else:
        df = pl.read_parquet(str(dataset_path))

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found in dataset. "
            f"Available label columns: {[c for c in df.columns if c.startswith('label_')]}"
        )

    feature_cols = _resolve_feature_cols(df, feature_set_name, label_col)
    if not feature_cols:
        raise ValueError(f"No feature columns available for feature set '{feature_set_name}'")

    logger.info(
        "Experiment: task=%s  features=%s (%d cols)  model=%s",
        task_name, feature_set_name, len(feature_cols), model_name,
    )

    X_train, y_train, _ = _load_split(df, "train", label_col, feature_cols, net_profit_col)
    X_test, y_test, y_net_test = _load_split(df, "test", label_col, feature_cols, net_profit_col)

    # Special case: oracle needs y_net at fit time
    if hasattr(model, "fit"):
        import inspect
        sig = inspect.signature(model.fit)
        if "y_net_profit" in sig.parameters:
            model.fit(X_test, y_test, y_net_profit=y_net_test)
        else:
            model.fit(X_train, y_train)

    # Optionally save the fitted model
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        pkl_name = f"{model_name}__{task_name}__{feature_set_name}.pkl"
        with open(model_dir / pkl_name, "wb") as fh:
            pickle.dump(model, fh)

    result = run_backtest(
        model=model,
        X_test=X_test,
        y_test=y_test,
        y_net_profit=y_net_test,
        task=ml_task,
        notional_usd=notional_usd,
        model_name=model_name,
    )
    result["task_name"] = task_name
    result["feature_set"] = feature_set_name
    result["n_train"] = len(y_train)
    result["n_test"] = len(y_test)
    result["feature_cols"] = feature_cols

    return result
