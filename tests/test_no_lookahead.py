"""No-lookahead assertions for label generation.

Guarantees:
  1. label at row t equals basis at t + horizon_ns (no future leakage).
  2. Labels derived from a strictly future timestamp, not the current row.
  3. The last horizon rows have NaN/null labels (forward look overflows the window).
"""

from __future__ import annotations

import polars as pl
import pytest

from stressbench.labels.basis_labels import add_basis_labels

_1M_NS = 60_000_000_000
_5M_NS = 300_000_000_000


def _make_df(n: int = 15, spacing_ns: int = _1M_NS) -> pl.DataFrame:
    """Evenly spaced 1-minute basis series with known values."""
    ts = [i * spacing_ns for i in range(n)]
    # basis values are distinct integers so misalignment is obvious
    basis = [float(i * 10) for i in range(n)]
    return pl.DataFrame({"ts_ns": ts, "cross_quote_basis_primary_bps": basis})


def test_label_1m_equals_next_row_basis() -> None:
    """label_basis_1m at row i must equal basis at row i+1 (1-minute spacing)."""
    df = _make_df(n=10, spacing_ns=_1M_NS)
    labeled = add_basis_labels(df, ts_col="ts_ns")

    for i in range(len(labeled) - 1):
        expected = labeled["cross_quote_basis_primary_bps"][i + 1]
        actual = labeled["label_basis_1m"][i]
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Row {i}: label_basis_1m={actual} but basis[{i+1}]={expected}"
        )


def test_label_5m_equals_row_5_steps_ahead() -> None:
    """label_basis_5m at row i must equal basis at row i+5 (1-minute rows)."""
    df = _make_df(n=20, spacing_ns=_1M_NS)
    labeled = add_basis_labels(df, ts_col="ts_ns")

    for i in range(len(labeled) - 5):
        expected = labeled["cross_quote_basis_primary_bps"][i + 5]
        actual = labeled["label_basis_5m"][i]
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"Row {i}: label_basis_5m={actual} but basis[{i+5}]={expected}"
        )


def test_label_does_not_equal_current_row() -> None:
    """The label must come from a future row, not the same timestamp."""
    df = _make_df(n=10, spacing_ns=_1M_NS)
    labeled = add_basis_labels(df, ts_col="ts_ns")

    # Every interior row should have a label != its own basis (values are distinct)
    for i in range(len(labeled) - 1):
        current_basis = labeled["cross_quote_basis_primary_bps"][i]
        label_val = labeled["label_basis_1m"][i]
        if label_val is not None:
            assert label_val != pytest.approx(current_basis, abs=1e-9), (
                f"Row {i}: label_basis_1m equals current basis={current_basis} — "
                "this would indicate the label is reading the present, not the future."
            )


def test_tail_rows_have_null_labels() -> None:
    """The last rows whose t + horizon overflows the series must have null labels.

    Uses the 15m horizon (900 s) with a 5-row 1m-spaced series.  All rows are
    within [0, 4m], so t + 15m = [15m, 19m] — far beyond the series end.
    The join_asof 90 s tolerance is too small to bridge the 11m+ gap, so every
    label_basis_15m value must be null.
    """
    _15M_NS = 15 * _1M_NS
    df = _make_df(n=5, spacing_ns=_1M_NS)
    labeled = add_basis_labels(df, ts_col="ts_ns")

    # All rows: t + 15m overflows the series (max ts=4m, nearest future entry is at
    # 4m-15m=-11m, distance=4m+11m=15m >> 90s tolerance) → all labels should be null
    for i in range(len(labeled)):
        val = labeled["label_basis_15m"][i]
        assert val is None, (
            f"Row {i}: label_basis_15m should be null for a 5-row series "
            f"(15m horizon overflows entirely) but got {val}"
        )


def test_label_prefix_no_lookahead() -> None:
    """USDC-specific labels with a custom prefix also respect the shift direction."""
    df = _make_df(n=12, spacing_ns=_1M_NS).rename(
        {"cross_quote_basis_primary_bps": "cross_quote_basis_usdc_bps"}
    )
    labeled = add_basis_labels(
        df,
        basis_col="cross_quote_basis_usdc_bps",
        ts_col="ts_ns",
        label_prefix="basis_usdc",
    )

    for i in range(len(labeled) - 1):
        expected = labeled["cross_quote_basis_usdc_bps"][i + 1]
        actual = labeled["label_basis_usdc_1m"][i]
        if actual is not None:
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"Row {i}: label_basis_usdc_1m={actual} but usdc_basis[{i+1}]={expected}"
            )


def test_binary_label_consistent_with_regression_label() -> None:
    """label_basis_1m_gt10bps must be 1 iff abs(label_basis_1m) > 10."""
    df = _make_df(n=12, spacing_ns=_1M_NS)
    labeled = add_basis_labels(df, ts_col="ts_ns")

    for i in range(len(labeled) - 1):
        reg = labeled["label_basis_1m"][i]
        binary = labeled["label_basis_1m_gt10bps"][i]
        if reg is not None and binary is not None:
            expected_binary = int(abs(reg) > 10.0)
            assert binary == expected_binary, (
                f"Row {i}: label_basis_1m={reg}, expected gt10bps={expected_binary} "
                f"but got {binary}"
            )
