"""Single-experiment runner for the Stablecoin StressBench grid.

Loads train/validation/test data for a given (task, feature_set) pair,
trains a model, calibrates the probability threshold on the validation split,
evaluates on test, and returns the result dict.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl

from stressbench.common.logging import get_logger
from stressbench.evaluation.backtest import run_backtest
from stressbench.evaluation.economic_metrics import economic_summary
from stressbench.evaluation.ml_metrics import classification_metrics
from stressbench.experiments.feature_sets import FEATURE_SETS
from stressbench.experiments.tasks import TASKS

logger = get_logger(__name__)

_EXCLUDE_COLS = {
    "split", "ts_1m_ns", "basis_primary_asset",
    "buy_venue", "sell_venue", "depth_source",
}

# Sources treated as real executable L2 depth (not synthetic)
_REAL_L2_DEPTH_SOURCES = {"real_l2_snapshot", "real_l2_incremental"}


def _resolve_feature_cols(
    df: pl.DataFrame,
    feature_set_name: str,
    label_col: str,
) -> list[str]:
    """Return the feature column list for a feature set, dropping absent columns."""
    requested = FEATURE_SETS.get(feature_set_name)
    if requested is None:
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

    if sdf.is_empty():
        n_feat = len(feature_cols)
        return np.empty((0, n_feat), dtype=np.float32), np.empty(0), np.empty(0)

    X_raw = sdf.select(feature_cols).to_numpy().astype(np.float32)
    y = sdf[label_col].to_numpy()

    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        with np.errstate(all="ignore"):
            col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        X = np.where(nan_mask, col_medians[None, :], X_raw)
    else:
        X = X_raw

    if net_profit_col and net_profit_col in sdf.columns:
        y_net_raw = sdf[net_profit_col].to_numpy().astype(np.float64)
        y_net = np.where(np.isnan(y_net_raw), -999.0, y_net_raw)
    else:
        y_net = np.zeros(len(y), dtype=np.float64)

    return X, y, y_net


def _calibrate_threshold(
    y_proba: np.ndarray,
    y_net: np.ndarray,
    n_candidates: int = 17,
) -> tuple[float, float]:
    """Find the probability threshold maximising net_bps_captured on a held-out set.

    Returns:
        (best_threshold, best_net_bps) — the threshold and the net bps achieved
        at that threshold on the calibration set.
    """
    best_t, best_net = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, n_candidates):
        signal = (y_proba > t).astype(np.int8)
        if signal.sum() == 0:
            continue
        net = float(np.mean(y_net[signal == 1]))
        if net > best_net:
            best_net = net
            best_t = float(t)
    return best_t, best_net if best_net > -np.inf else float("nan")


def run_experiment(
    dataset_path: str | Path,
    task_name: str,
    feature_set_name: str,
    model_name: str,
    model_factory: Callable[[str, list[str]], Any],
    model_dir: str | Path | None = None,
    calibrate_threshold: bool = True,
) -> dict[str, Any]:
    """Train and evaluate one (task, feature_set, model) experiment.

    The model is built *after* feature columns are resolved so that rule-based
    baselines (e.g. ``PriceBasisThresholdBaseline``) receive the correct column
    index for whichever feature set is active.

    Validation split is used to calibrate the probability threshold before
    final test evaluation when ``calibrate_threshold=True``.

    Args:
        dataset_path: Path to ``dataset.parquet`` or a directory containing it.
        task_name: Key from :data:`~stressbench.experiments.tasks.TASKS`.
        feature_set_name: Key from
            :data:`~stressbench.experiments.feature_sets.FEATURE_SETS`.
        model_name: Display name used in leaderboard rows.
        model_factory: Callable ``(model_name, feature_cols) → model``. Called
            after feature columns are resolved so index-based baselines receive
            the correct column index.
        model_dir: Optional directory to save the fitted model as a pickle.
        calibrate_threshold: If True, use the validation split to find the
            probability threshold that maximises net bps before test evaluation.

    Returns:
        Dict with keys: ``task_name``, ``feature_set``, ``model``,
        ``ml_metrics``, ``economic_metrics``, ``n_train``, ``n_val``,
        ``n_test``, ``validation_threshold``, ``validation_net_bps``.
    """
    task_cfg = TASKS.get(task_name)
    if task_cfg is None:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {list(TASKS)}")

    label_col = task_cfg["label"]
    ml_task = task_cfg["task"]
    notional_usd = task_cfg["notional_usd"]
    net_profit_col = task_cfg.get("net_profit_col")

    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        parquet_files = list(dataset_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {dataset_path}")
        df = pl.read_parquet(str(dataset_path / "*.parquet"))
    else:
        df = pl.read_parquet(str(dataset_path))

    if label_col not in df.columns:
        available = [c for c in df.columns if c.startswith("label_")]
        raise ValueError(
            f"Label column '{label_col}' not found in dataset. "
            f"Available label columns: {available}"
        )

    feature_cols = _resolve_feature_cols(df, feature_set_name, label_col)
    if not feature_cols:
        raise ValueError(f"No feature columns available for feature set '{feature_set_name}'")

    logger.info(
        "Experiment: task=%s  features=%s (%d cols)  model=%s",
        task_name, feature_set_name, len(feature_cols), model_name,
    )

    # Build model now that feature_cols are known — critical for index-based baselines
    model = model_factory(model_name, feature_cols)

    X_train, y_train, _ = _load_split(df, "train", label_col, feature_cols, net_profit_col)
    X_val, y_val, y_net_val = _load_split(df, "validation", label_col, feature_cols, net_profit_col)
    X_test, y_test, y_net_test = _load_split(df, "test", label_col, feature_cols, net_profit_col)

    # Fit: oracle gets y_net at fit time; others use train labels
    import inspect
    sig = inspect.signature(model.fit)
    if "y_net_profit" in sig.parameters:
        model.fit(X_test, y_test, y_net_profit=y_net_test)
    else:
        model.fit(X_train, y_train)

    # Threshold calibration on validation split
    val_threshold = 0.5
    val_net_bps = float("nan")
    if calibrate_threshold and len(X_val) > 0 and ml_task == "classification":
        try:
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_val_proba = np.clip(y_val_proba, 0.0, 1.0)
            val_threshold, val_net_bps = _calibrate_threshold(y_val_proba, y_net_val)
            logger.info("  Val threshold: %.2f  (val net_bps=%.1f)", val_threshold, val_net_bps)
        except (AttributeError, ValueError):
            pass

    # Optionally save model
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
        threshold=val_threshold,
        model_name=model_name,
    )
    result["task_name"] = task_name
    result["feature_set"] = feature_set_name
    result["n_train"] = len(y_train)
    result["n_val"] = len(y_val)
    result["n_test"] = len(y_test)
    result["feature_cols"] = feature_cols
    result["validation_threshold"] = round(val_threshold, 4)
    result["validation_net_bps"] = round(val_net_bps, 2) if not np.isnan(val_net_bps) else None

    return result
