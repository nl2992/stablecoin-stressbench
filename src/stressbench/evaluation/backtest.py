"""Simple event-based backtest framework.

Runs a model on the test split and computes both ML and economic metrics.
Uses event-based splits to prevent temporal leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from stressbench.evaluation.economic_metrics import economic_summary
from stressbench.evaluation.ml_metrics import classification_metrics, regression_metrics
from stressbench.common.logging import get_logger

logger = get_logger(__name__)


def run_backtest(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_net_profit: np.ndarray,
    task: str = "classification",
    notional_usd: float = 50_000.0,
    cost_threshold_bps: float = 0.0,
    threshold: float = 0.5,
    model_name: str = "model",
) -> dict[str, Any]:
    """Run a backtest for a single model on the test set.

    Args:
        model: Fitted model with ``predict`` (and optionally ``predict_proba``) method.
        X_test: Test feature matrix.
        y_test: Ground-truth labels.
        y_net_profit: Ground-truth net profit in basis points (for economic metrics).
        task: ``"classification"`` or ``"regression"``.
        notional_usd: Notional size per trade in USD.
        cost_threshold_bps: Cost threshold for economic metrics.
        model_name: Name of the model for reporting.

    Returns:
        Dict with ML metrics, economic metrics, and model name.
    """
    logger.info("Running backtest for model: %s", model_name)

    y_pred = model.predict(X_test)

    if task == "classification":
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except (AttributeError, ValueError):
            y_proba = y_pred.astype(float)

        # Clip to [0, 1] — regression models may produce out-of-range scores
        y_proba = np.clip(y_proba, 0.0, 1.0)

        # Apply calibrated threshold from validation split (default 0.5)
        unique_vals = np.unique(y_pred[~np.isnan(y_pred)])
        if len(unique_vals) > 2 or (len(unique_vals) > 0 and not set(unique_vals.tolist()).issubset({0, 1, 0.0, 1.0})):
            y_pred_binary = (y_proba > threshold).astype(np.int8)
        else:
            y_pred_binary = y_pred.astype(np.int8)

        ml = classification_metrics(y_test, y_proba, y_pred_binary)
        signal = y_pred_binary
    else:
        ml = regression_metrics(y_test, y_pred)
        signal = (y_pred > cost_threshold_bps).astype(int)

    econ = economic_summary(
        y_true_net_profit=y_net_profit,
        y_pred_signal=signal,
        notional_usd=notional_usd,
        cost_threshold_bps=cost_threshold_bps,
    )

    return {
        "model": model_name,
        "task": task,
        "ml_metrics": ml,
        "economic_metrics": econ,
    }
