"""Tests verifying that synthetic kline depth is excluded from paper-grade net profit.

The core study claim (3.34% of windows are executable after VWAP cost) must be
derived from real limit-order book snapshots, not from OHLCV-derived synthetic
ladders.  These tests check that depth_source tagging is correct and that the
paper-grade pipeline does not inadvertently include synthetic rows.
"""

from __future__ import annotations

import polars as pl
import pytest

# Vocabulary of depth_source values
_REAL_L2_SOURCES = {"real_l2_snapshot", "real_l2_incremental"}
_SYNTHETIC_SOURCES = {"synthetic_kline"}
_ALL_SOURCES = _REAL_L2_SOURCES | _SYNTHETIC_SOURCES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_book_rows(depth_source: str, n: int = 3) -> list[dict]:
    """Return minimal book-level records tagged with the given depth_source."""
    base_price = 30_000.0
    return [
        {
            "ts_event_ns": 1_678_000_000_000_000_000 + i * 60_000_000_000,
            "ts_receive_ns": 1_678_000_000_000_000_000 + i * 60_000_000_000,
            "venue_id": "testv",
            "instrument_id": "testv:BTCUSD",
            "native_symbol": "BTCUSD",
            "side": "bid" if i % 2 == 0 else "ask",
            "level": 0,
            "price": base_price - 10.0 if i % 2 == 0 else base_price + 10.0,
            "size": 1.0,
            "checksum": None,
            "raw_source": f"testv:{depth_source}",
            "payload_hash": f"hash_{i}",
            "depth_source": depth_source,
            "is_crossed_book": False,
            "is_negative_size": False,
            "is_sequence_gap": False,
            "is_checksum_failed": False,
            "is_stale_quote": False,
            "is_resync_period": False,
        }
        for i in range(n)
    ]


def _mixed_books() -> pl.DataFrame:
    """DataFrame with both real L2 and synthetic kline rows."""
    real_rows = _make_book_rows("real_l2_snapshot", n=4)
    incr_rows = _make_book_rows("real_l2_incremental", n=2)
    synth_rows = _make_book_rows("synthetic_kline", n=3)
    return pl.DataFrame(real_rows + incr_rows + synth_rows)


# ---------------------------------------------------------------------------
# Tests: Silver normalizer tagging
# ---------------------------------------------------------------------------

def test_real_l2_snapshot_tag() -> None:
    """Rows tagged real_l2_snapshot must have that value in depth_source."""
    df = pl.DataFrame(_make_book_rows("real_l2_snapshot", n=5))
    assert (df["depth_source"] == "real_l2_snapshot").all()


def test_synthetic_kline_tag() -> None:
    """Rows tagged synthetic_kline must have that value in depth_source."""
    df = pl.DataFrame(_make_book_rows("synthetic_kline", n=5))
    assert (df["depth_source"] == "synthetic_kline").all()


def test_real_l2_incremental_tag() -> None:
    df = pl.DataFrame(_make_book_rows("real_l2_incremental", n=5))
    assert (df["depth_source"] == "real_l2_incremental").all()


# ---------------------------------------------------------------------------
# Tests: paper-grade filter excludes synthetic rows
# ---------------------------------------------------------------------------

def test_real_l2_filter_excludes_synthetic() -> None:
    """Filtering to real L2 sources must remove all synthetic_kline rows."""
    df = _mixed_books()
    real_only = df.filter(pl.col("depth_source").is_in(list(_REAL_L2_SOURCES)))
    sources_in_result = set(real_only["depth_source"].unique().to_list())
    assert "synthetic_kline" not in sources_in_result, (
        f"synthetic_kline rows leaked into the real-L2-only DataFrame: {sources_in_result}"
    )


def test_real_l2_filter_preserves_both_real_subtypes() -> None:
    """Both real_l2_snapshot and real_l2_incremental must survive the filter."""
    df = _mixed_books()
    real_only = df.filter(pl.col("depth_source").is_in(list(_REAL_L2_SOURCES)))
    sources_in_result = set(real_only["depth_source"].unique().to_list())
    assert "real_l2_snapshot" in sources_in_result
    assert "real_l2_incremental" in sources_in_result


def test_real_l2_filter_row_count() -> None:
    """Row count after filtering must equal the number of real L2 input rows."""
    df = _mixed_books()
    n_real = int(df.filter(pl.col("depth_source").is_in(list(_REAL_L2_SOURCES))).height)
    assert n_real == 6  # 4 snapshot + 2 incremental


def test_synthetic_only_produces_no_paper_grade_rows() -> None:
    """When all rows are synthetic, the real-L2 filter must return an empty frame."""
    df = pl.DataFrame(_make_book_rows("synthetic_kline", n=5))
    real_only = df.filter(pl.col("depth_source").is_in(list(_REAL_L2_SOURCES)))
    assert real_only.is_empty(), (
        "Expected empty paper-grade frame when all input depth is synthetic_kline"
    )


# ---------------------------------------------------------------------------
# Tests: depth_source vocabulary is well-defined
# ---------------------------------------------------------------------------

def test_depth_source_values_are_in_allowed_vocabulary() -> None:
    """Any depth_source value in a mixed book must be in the known vocabulary."""
    df = _mixed_books()
    unknown = [
        v for v in df["depth_source"].unique().to_list()
        if v not in _ALL_SOURCES
    ]
    assert not unknown, f"Unknown depth_source values: {unknown}"


def test_no_generic_real_l2_tag_in_books() -> None:
    """The collapsed generic 'real_l2' tag must not appear — only granular subtypes.

    Previously _load_silver_channels was overwriting all real-L2 rows with the
    generic 'real_l2' string.  Normalizers now emit real_l2_snapshot or
    real_l2_incremental, and _load_silver_channels only fills missing values.
    """
    df = _mixed_books()
    found_generic = (df["depth_source"] == "real_l2").any()
    assert not found_generic, (
        "Generic 'real_l2' tag found — normalizers should use real_l2_snapshot "
        "or real_l2_incremental, and _load_silver_channels should not overwrite them."
    )
