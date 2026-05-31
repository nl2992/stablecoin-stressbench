"""Rule-based baselines for the Stablecoin StressBench benchmark.

These baselines do not learn from data. They provide anchors for economic
evaluation:

- ``NoTradeBaseline``: never trade — economic P&L is zero by construction.
  Any model that loses money is worse than this.

- ``PriceBasisThresholdBaseline``: trade when a specified feature column
  (typically the cross-quote basis) exceeds a threshold in absolute value.
  Represents a naive "buy when price diverges" policy.

- ``GrossArbThresholdBaseline``: trade when gross spread exceeds a threshold.
  Equivalent to price-only arbitrage before execution costs.

- ``NetProfitOracleUpperBound``: uses realized future net profit as the
  signal. This is a cheater model that defines the theoretical ceiling on
  economic performance. Must be ``fit`` on the same split it predicts.
"""

from __future__ import annotations

import numpy as np


class NoTradeBaseline:
    """Never trade. Produces zero P&L and zero drawdown by construction.

    This is the critical anchor: any model that loses money on average is
    strictly worse than doing nothing.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NoTradeBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])


class PriceBasisThresholdBaseline:
    """Trade when |feature[col_index]| > threshold_bps.

    Intended to be wired to the cross-quote basis column so it models a
    "buy when price diverges" policy that ignores all execution costs.

    Args:
        col_index: Index of the basis feature in the X matrix.
        threshold_bps: Absolute basis threshold in basis points.
    """

    def __init__(self, col_index: int = 0, threshold_bps: float = 10.0) -> None:
        self.col_index = col_index
        self.threshold_bps = threshold_bps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PriceBasisThresholdBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.abs(X[:, self.col_index]) > self.threshold_bps).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])


class GrossArbThresholdBaseline:
    """Trade when a gross-spread feature exceeds a threshold.

    Structurally identical to ``PriceBasisThresholdBaseline``; separated
    semantically so leaderboard rows distinguish price-signal from gross-arb
    policies.

    Args:
        col_index: Index of the gross-spread feature in X.
        threshold_bps: Gross spread threshold in basis points.
    """

    def __init__(self, col_index: int = 0, threshold_bps: float = 20.0) -> None:
        self.col_index = col_index
        self.threshold_bps = threshold_bps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GrossArbThresholdBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.col_index] > self.threshold_bps).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])


class RouteConsistentBasisRule:
    """Trade when the USDC-route basis exceeds a threshold.

    Isolates the USDC route to prevent route-direction mismatch false
    positives (FP windows show USDC basis ~0.56 bps while USDT is elevated).

    Args:
        col_index: Index of cross_quote_basis_usdc_bps in X.
        threshold_bps: USDC basis threshold in basis points.
    """

    def __init__(self, col_index: int = 0, threshold_bps: float = 10.0) -> None:
        self.col_index = col_index
        self.threshold_bps = threshold_bps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RouteConsistentBasisRule":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.col_index] > self.threshold_bps).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])


class DepthAdjustedBasisRule:
    """Trade when USDC basis exceeds spread + fee estimate.

    Filters windows where the basis cannot survive current spread plus fees.

    Args:
        basis_col_index: Index of cross_quote_basis_usdc_bps in X.
        spread_col_index: Index of spread_bps_mean in X.
        fee_estimate_bps: Round-trip fee estimate in basis points.
    """

    def __init__(
        self,
        basis_col_index: int = 0,
        spread_col_index: int = 1,
        fee_estimate_bps: float = 8.0,
    ) -> None:
        self.basis_col_index = basis_col_index
        self.spread_col_index = spread_col_index
        self.fee_estimate_bps = fee_estimate_bps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DepthAdjustedBasisRule":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        basis = X[:, self.basis_col_index]
        spread = X[:, self.spread_col_index]
        return (basis > spread + self.fee_estimate_bps).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])


class SpreadDepthRule:
    """Trade when USDC basis exceeds a threshold AND ask depth is sufficient.

    Combines a basis filter with a minimum book depth requirement.
    The depth threshold defaults to 0 (no filter) and can be calibrated
    via fit() — set to the median ask depth of positive training labels.

    Args:
        basis_col_index: Index of cross_quote_basis_usdc_bps in X.
        depth_col_index: Index of depth_ask_10bp_mean in X.
        basis_threshold_bps: Basis threshold in basis points.
        depth_threshold: Minimum ask depth; overwritten by fit() if > 0 labels exist.
    """

    def __init__(
        self,
        basis_col_index: int = 0,
        depth_col_index: int = 1,
        basis_threshold_bps: float = 10.0,
        depth_threshold: float = 0.0,
    ) -> None:
        self.basis_col_index = basis_col_index
        self.depth_col_index = depth_col_index
        self.basis_threshold_bps = basis_threshold_bps
        self.depth_threshold = depth_threshold
        self._depth_threshold: float = depth_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpreadDepthRule":
        pos_mask = y == 1
        if pos_mask.any():
            self._depth_threshold = float(np.median(X[pos_mask, self.depth_col_index]))
        else:
            self._depth_threshold = self.depth_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        basis_ok = X[:, self.basis_col_index] > self.basis_threshold_bps
        depth_ok = X[:, self.depth_col_index] > self._depth_threshold
        return (basis_ok & depth_ok).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])


class LatencyHaircutRule:
    """Trade only when USDC basis exceeds threshold at both t and t-1.

    Persistence filter: requires two consecutive bars above threshold to
    reduce latency-sensitive false positives. First bar always = 0.

    Args:
        col_index: Index of cross_quote_basis_usdc_bps in X.
        threshold_bps: Basis threshold in basis points.
    """

    def __init__(self, col_index: int = 0, threshold_bps: float = 10.0) -> None:
        self.col_index = col_index
        self.threshold_bps = threshold_bps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LatencyHaircutRule":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        above = X[:, self.col_index] > self.threshold_bps
        signal = np.zeros(len(X), dtype=np.int8)
        signal[1:] = (above[1:] & above[:-1]).astype(np.int8)
        return signal

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])


class NetProfitOracleUpperBound:
    """Oracle upper bound: signal = 1 when realized net profit > threshold.

    This is a cheater model that uses future information. It defines the
    theoretical ceiling on economic performance — the best any model could
    achieve if it had perfect foresight.

    Usage:
        oracle = NetProfitOracleUpperBound(threshold_bps=0.0)
        oracle.fit(X_test, y_test, y_net_profit=y_net_test)
        signal = oracle.predict(X_test)

    Args:
        threshold_bps: Net profit threshold in basis points. Trades are
            signalled when realized net profit exceeds this value.
    """

    def __init__(self, threshold_bps: float = 0.0) -> None:
        self.threshold_bps = threshold_bps
        self._oracle_signal: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_net_profit: np.ndarray | None = None,
    ) -> "NetProfitOracleUpperBound":
        if y_net_profit is not None:
            # NaN net profit means insufficient book depth → not executable → 0
            net = np.where(np.isnan(y_net_profit), -np.inf, y_net_profit)
            self._oracle_signal = (net > self.threshold_bps).astype(np.int8)
        else:
            self._oracle_signal = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._oracle_signal is not None and len(self._oracle_signal) == len(X):
            return self._oracle_signal
        return np.zeros(len(X), dtype=np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        signal = self.predict(X).astype(float)
        return np.column_stack([1.0 - signal, signal])
