#!/usr/bin/env python3
"""Train all benchmark models on the training split.

Usage:
    python scripts/train_models.py --data-dir data/gold --model-dir models/trained
    python scripts/train_models.py --data-dir data/gold --model-dir models/trained --models lgbm xgb rf
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_ALL_MODELS = ["last_value", "rolling_mean", "ar1", "logistic", "ridge", "lasso", "lgbm", "xgb", "rf"]


_SEQ_MODELS = ["temporal_cnn", "transformer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train benchmark models.")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--model-dir", default="models/trained")
    parser.add_argument(
        "--models",
        nargs="*",
        default=_ALL_MODELS,
        help=f"Flat models to train. Options: {_ALL_MODELS}",
    )
    parser.add_argument(
        "--seq-models",
        nargs="*",
        default=None,
        choices=_SEQ_MODELS,
        help=f"Sequence models to train: {_SEQ_MODELS}. Requires PyTorch.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sliding window length (minutes) for sequence models (default: 60).",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Prediction task type.",
    )
    parser.add_argument(
        "--label",
        default="label_basis_1m_gt10bps",
        help="Label column to predict.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=None,
        help="Feature columns (default: all non-label columns).",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Generate synthetic demo data if dataset.parquet is missing. "
             "For CI/demo only — never use for paper results.",
    )
    return parser.parse_args()


def load_train_data(
    data_dir: str,
    label_col: str,
    sort_by_time: bool = False,
    allow_synthetic: bool = False,
):
    """Load training data from Gold Parquet files.

    Args:
        data_dir: Path to the Gold directory containing dataset.parquet.
        label_col: Column name to use as the prediction target.
        sort_by_time: If True, sort by ``ts_1m_ns`` before returning arrays.
            Required for sequence models to ensure windows are time-ordered.
        allow_synthetic: If True, generate synthetic demo data when no Parquet
            files are found instead of raising an error.
    """
    import polars as pl

    gold_path = Path(data_dir)
    parquet_files = list(gold_path.glob("**/*.parquet"))
    if not parquet_files:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"No Parquet files found in '{data_dir}'. "
                "Run scripts/build_features.py first to produce dataset.parquet, "
                "or pass --allow-synthetic for demo/CI mode."
            )
        logger.warning("No data found in %s; generating synthetic demo data (--allow-synthetic).", data_dir)
        rng = np.random.default_rng(42)
        n = 10_000
        X = rng.standard_normal((n, 20)).astype(np.float32)
        y = (rng.standard_normal(n) > 0).astype(np.int8)
        return X, y, None

    df = pl.read_parquet(str(gold_path / "*.parquet"))
    label_df = df.filter(pl.col("split") == "train")
    if sort_by_time and "ts_1m_ns" in label_df.columns:
        label_df = label_df.sort("ts_1m_ns")

    # Drop rows where the label is null/NaN (end-of-window forward-look overflow)
    label_df = label_df.filter(pl.col(label_col).is_not_null())
    if label_df[label_col].dtype.is_float():
        label_df = label_df.filter(~pl.col(label_col).is_nan())

    # Exclude the time index and split key — not useful as float32 features
    _EXCLUDE = {"split", "ts_1m_ns"}
    feature_cols = [
        c for c in df.columns
        if not c.startswith("label_") and c not in _EXCLUDE
    ]
    X_raw = label_df.select(feature_cols).to_numpy().astype(np.float32)
    y = label_df[label_col].to_numpy()

    # Impute NaN features with column median (NaN from thin-market depth rows).
    # For all-NaN columns (e.g. $500K net-profit in normal periods), fall back to 0.
    # Tree models handle NaN natively but sklearn linear/logistic models do not.
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        with np.errstate(all="ignore"):
            col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)  # all-NaN cols → 0
        nan_cols = int(nan_mask.any(axis=0).sum())
        logger.info("Imputing NaN in %d feature columns (median or 0 for all-NaN).", nan_cols)
        X = np.where(nan_mask, col_medians[None, :], X_raw)
    else:
        X = X_raw

    return X, y, feature_cols


def make_sequence_windows(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences from a time-ordered flat feature matrix.

    Args:
        X: ``(n_samples, n_features)`` array, must be time-ordered.
        y: ``(n_samples,)`` label array.
        seq_len: Number of consecutive time steps per window.

    Returns:
        X_seq: ``(n_windows, seq_len, n_features)`` — one row per window.
        y_seq: ``(n_windows,)`` — label at the last timestep of each window.
    """
    n, f = X.shape
    if n < seq_len:
        logger.warning(
            "Fewer samples (%d) than seq_len (%d); no sequence windows created.", n, seq_len
        )
        return np.empty((0, seq_len, f), dtype=X.dtype), np.empty(0, dtype=y.dtype)
    n_windows = n - seq_len + 1
    # Vectorised index construction — avoids an explicit Python loop
    idx = np.arange(seq_len)[None, :] + np.arange(n_windows)[:, None]  # (n_windows, seq_len)
    X_seq = X[idx]                 # (n_windows, seq_len, n_features)
    y_seq = y[idx[:, -1]]          # label at the end of each window
    return X_seq, y_seq


def get_model(name: str, task: str):
    if name == "last_value":
        from stressbench.models.baselines import LastValueBaseline
        return LastValueBaseline()
    elif name == "rolling_mean":
        from stressbench.models.baselines import RollingMeanBaseline
        return RollingMeanBaseline()
    elif name == "ar1":
        from stressbench.models.baselines import AR1Baseline
        return AR1Baseline()
    elif name == "logistic":
        from stressbench.models.baselines import LogisticBaseline
        return LogisticBaseline()
    elif name == "ridge":
        from stressbench.models.baselines import RidgeBaseline
        return RidgeBaseline()
    elif name == "lasso":
        from stressbench.models.baselines import LassoBaseline
        return LassoBaseline()
    elif name == "lgbm":
        from stressbench.models.tree_models import LGBMWrapper
        return LGBMWrapper(task=task)
    elif name == "xgb":
        from stressbench.models.tree_models import XGBWrapper
        return XGBWrapper(task=task)
    elif name == "rf":
        from stressbench.models.tree_models import RandomForestWrapper
        return RandomForestWrapper(task=task)
    else:
        raise ValueError(f"Unknown model: {name}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data from %s", args.data_dir)
    X, y, feature_cols = load_train_data(
        args.data_dir, args.label, allow_synthetic=args.allow_synthetic
    )
    logger.info("Training data shape: X=%s, y=%s", X.shape, y.shape)

    # Flat models
    for model_name in args.models:
        logger.info("Training model: %s", model_name)
        try:
            model = get_model(model_name, args.task)
            model.fit(X, y)
            model_path = model_dir / f"{model_name}_{args.label}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info("Saved model to %s", model_path)
        except Exception as exc:
            logger.error("Failed to train %s: %s", model_name, exc)

    # Sequence models: TemporalCNN, TransformerEncoder
    seq_models_to_train = args.seq_models or []
    if seq_models_to_train:
        from stressbench.models.sequence_models import TemporalCNN, TransformerEncoder

        logger.info(
            "Building sequence windows (seq_len=%d) for: %s",
            args.seq_len, seq_models_to_train,
        )
        # Reload time-sorted so windows are chronological
        X_sorted, y_sorted, _ = load_train_data(
            args.data_dir, args.label,
            sort_by_time=True, allow_synthetic=args.allow_synthetic,
        )
        X_seq, y_seq = make_sequence_windows(X_sorted, y_sorted, args.seq_len)
        logger.info("Sequence windows: X_seq=%s  y_seq=%s", X_seq.shape, y_seq.shape)

        if X_seq.shape[0] == 0:
            logger.warning("No sequence windows; skipping sequence model training.")
        else:
            y_seq_f = y_seq.astype(np.float32)
            n_features = X_sorted.shape[1]
            for model_name in seq_models_to_train:
                logger.info("Training sequence model: %s", model_name)
                try:
                    if model_name == "temporal_cnn":
                        model = TemporalCNN(input_dim=n_features, task=args.task)
                    elif model_name == "transformer":
                        model = TransformerEncoder(input_dim=n_features, task=args.task)
                    else:
                        logger.warning("Unknown sequence model %s; skipping.", model_name)
                        continue
                    model.fit(X_seq, y_seq_f)
                    model_path = model_dir / f"{model_name}_{args.label}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    logger.info("Saved sequence model to %s", model_path)
                except Exception as exc:
                    logger.error("Failed to train sequence model %s: %s", model_name, exc)

    # Save feature column metadata
    meta = {
        "label": args.label,
        "task": args.task,
        "feature_cols": feature_cols or [],
        "models_trained": args.models,
        "seq_models_trained": seq_models_to_train,
        "seq_len": args.seq_len,
    }
    with open(model_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete. Models saved to %s", model_dir)


if __name__ == "__main__":
    main()
