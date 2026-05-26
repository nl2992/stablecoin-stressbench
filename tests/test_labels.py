"""Tests for label generation modules."""

from __future__ import annotations

import polars as pl
import pytest

from stressbench.labels.basis_labels import add_basis_labels
from stressbench.labels.regime_labels import add_regime_labels
from stressbench.labels.recovery_labels import compute_recovery_halflife


def _make_basis_df(n: int = 100) -> pl.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    ts = [i * 60_000_000_000 for i in range(n)]
    basis = rng.normal(0, 10, n).tolist()
    return pl.DataFrame({"ts_ns": ts, "cross_quote_basis_primary_bps": basis})


def test_add_basis_labels_adds_columns():
    df = _make_basis_df(200)
    result = add_basis_labels(df)
    assert "label_basis_1m" in result.columns
    assert "label_basis_5m" in result.columns
    assert "label_basis_1m_gt10bps" in result.columns
    assert "label_basis_5m_gt25bps" in result.columns


def test_add_basis_labels_classification_binary():
    df = _make_basis_df(200)
    result = add_basis_labels(df)
    col = result["label_basis_1m_gt10bps"].drop_nulls()
    assert set(col.unique().to_list()).issubset({0, 1})


def test_add_basis_labels_label_prefix():
    df = _make_basis_df(100)
    result = add_basis_labels(df, label_prefix="basis_usdc")
    assert "label_basis_usdc_1m" in result.columns
    assert "label_basis_usdc_1m_gt10bps" in result.columns
    assert "label_basis_1m" not in result.columns


def test_add_basis_labels_explicit_col():
    import numpy as np
    rng = np.random.default_rng(7)
    ts = [i * 60_000_000_000 for i in range(100)]
    df = pl.DataFrame({
        "ts_ns": ts,
        "cross_quote_basis_maxabs_bps": rng.normal(0, 15, 100).tolist(),
    })
    result = add_basis_labels(
        df,
        basis_col="cross_quote_basis_maxabs_bps",
        label_prefix="basis_maxabs",
    )
    assert "label_basis_maxabs_1m_gt10bps" in result.columns


def test_add_basis_labels_empty_df():
    df = pl.DataFrame({"ts_ns": [], "cross_quote_basis_primary_bps": []})
    result = add_basis_labels(df)
    assert result.is_empty()


def test_add_regime_labels_normal():
    df = pl.DataFrame(
        {
            "ts_ns": [i * 60_000_000_000 for i in range(10)],
            "deviation_from_1_usd_bps": [0.5] * 10,
            "spread_bps": [1.0] * 10,
            "depth_bid_10bp": [100_000.0] * 10,
            "transfer_count_1m": [10] * 10,
            "is_issuer_event_window": [False] * 10,
        }
    )
    result = add_regime_labels(df)
    assert "label_regime" in result.columns
    assert all(r == "normal" for r in result["label_regime"].to_list())


def test_add_regime_labels_peg_pressure():
    df = pl.DataFrame(
        {
            "ts_ns": [i * 60_000_000_000 for i in range(5)],
            "deviation_from_1_usd_bps": [30.0] * 5,
            "spread_bps": [1.0] * 5,
            "depth_bid_10bp": [100_000.0] * 5,
            "transfer_count_1m": [10] * 5,
            "is_issuer_event_window": [False] * 5,
        }
    )
    result = add_regime_labels(df)
    assert all(r == "peg_pressure" for r in result["label_regime"].to_list())


def test_add_regime_labels_issuer_event_priority():
    df = pl.DataFrame(
        {
            "ts_ns": [i * 60_000_000_000 for i in range(5)],
            "deviation_from_1_usd_bps": [30.0] * 5,  # peg_pressure
            "spread_bps": [1.0] * 5,
            "depth_bid_10bp": [100_000.0] * 5,
            "transfer_count_1m": [10] * 5,
            "is_issuer_event_window": [True] * 5,  # issuer_event takes priority
        }
    )
    result = add_regime_labels(df)
    assert all(r == "issuer_event_window" for r in result["label_regime"].to_list())


def test_compute_recovery_halflife_basic():
    import polars as pl
    ts = [i * 60_000_000_000 for i in range(20)]
    # Deviation peaks at t=5 then decays
    deviation = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 40.0, 30.0, 20.0, 10.0,
                 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    df = pl.DataFrame({"ts_ns": ts, "deviation_from_1_usd_bps": deviation})
    halflife = compute_recovery_halflife(df)
    # Peak at t=5 (300s), half=25bps, first time <=25 is t=8 (480s)
    # halflife = (480 - 300) / 60_000_000_000 * 60 = 3 minutes
    assert halflife is not None
    assert halflife > 0


def test_compute_recovery_halflife_no_recovery():
    ts = [i * 60_000_000_000 for i in range(5)]
    deviation = [10.0, 20.0, 30.0, 40.0, 50.0]  # Never recovers
    df = pl.DataFrame({"ts_ns": ts, "deviation_from_1_usd_bps": deviation})
    halflife = compute_recovery_halflife(df)
    assert halflife is None
