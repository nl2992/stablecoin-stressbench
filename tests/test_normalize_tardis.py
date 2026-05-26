"""Tests for normalize_tardis — Tardis CSV archive Silver normalizers.

Each test creates a minimal Bronze-schema DataFrame whose ``payload`` column
contains the flat Tardis CSV field names that ``archive_to_bronze.tardis_to_bronze``
preserves after renaming:

    localTimestamp  →  excluded (used as ts_receive_ns in the Bronze wrapper)
    timestamp       →  ts_exchange
    exchange        →  _exchange
    symbol          →  _symbol

All other Tardis CSV columns are stored verbatim in the payload JSON.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from stressbench.normalization.normalize_tardis import (
    normalize_tardis_book_snapshot_1s,
    normalize_tardis_incremental_book_l2,
    normalize_tardis_trades,
)

_TS_NS = 1_704_067_200_000_000_000   # 2024-01-01 00:00:00 UTC
_TS_ISO = "2024-01-01T00:00:00.000000Z"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tardis_trades_df() -> pl.DataFrame:
    """Single Bronze row representing one Tardis trades CSV record."""
    payload = {
        "ts_exchange": _TS_ISO,
        "_exchange": "coinbase",
        "_symbol": "BTC-USD",
        "id": "trade-001",
        "side": "buy",
        "price": 42_500.75,
        "amount": 0.25,
    }
    return pl.DataFrame({
        "source": ["coinbase"],
        "channel": ["tardis_trades"],
        "symbol": ["BTC-USD"],
        "ts_exchange": [_TS_ISO],
        "ts_receive_ns": [_TS_NS],
        "payload": [json.dumps(payload)],
        "payload_hash": [""],
        "schema_version": ["raw.v1"],
        "ingest_batch_id": ["tardis_archive"],
    })


@pytest.fixture()
def tardis_book_snapshot_df() -> pl.DataFrame:
    """Single Bronze row representing one Tardis book_snapshot_1s CSV record (5 levels)."""
    payload: dict = {
        "ts_exchange": _TS_ISO,
        "_exchange": "coinbase",
        "_symbol": "BTC-USD",
        "isSnapshot": "true",
    }
    for i in range(5):
        payload[f"bids[{i}].price"] = 42_500.0 - i * 5
        payload[f"bids[{i}].amount"] = 1.0 + i * 0.1
        payload[f"asks[{i}].price"] = 42_510.0 + i * 5
        payload[f"asks[{i}].amount"] = 0.8 + i * 0.1
    return pl.DataFrame({
        "source": ["coinbase"],
        "channel": ["tardis_book_snapshot_1s"],
        "symbol": ["BTC-USD"],
        "ts_exchange": [_TS_ISO],
        "ts_receive_ns": [_TS_NS],
        "payload": [json.dumps(payload)],
        "payload_hash": [""],
        "schema_version": ["raw.v1"],
        "ingest_batch_id": ["tardis_archive"],
    })


@pytest.fixture()
def tardis_incremental_l2_df() -> pl.DataFrame:
    """Three Bronze rows representing Tardis incremental_book_L2 updates."""
    rows = []
    for side, price, amount in [
        ("bid", 42_498.0, 2.0),
        ("ask", 42_502.0, 1.5),
        ("bid", 42_495.0, 3.0),
    ]:
        payload = {
            "ts_exchange": _TS_ISO,
            "_exchange": "kraken",
            "_symbol": "BTC/USD",
            "isSnapshot": "false",
            "side": side,
            "price": price,
            "amount": amount,
        }
        rows.append({
            "source": "kraken",
            "channel": "tardis_incremental_book_l2",
            "symbol": "BTC/USD",
            "ts_exchange": _TS_ISO,
            "ts_receive_ns": _TS_NS,
            "payload": json.dumps(payload),
            "payload_hash": "",
            "schema_version": "raw.v1",
            "ingest_batch_id": "tardis_archive",
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# normalize_tardis_trades
# ---------------------------------------------------------------------------

def test_normalize_tardis_trades_non_empty(tardis_trades_df):
    result = normalize_tardis_trades(tardis_trades_df)
    assert not result.is_empty()


def test_normalize_tardis_trades_venue_id(tardis_trades_df):
    result = normalize_tardis_trades(tardis_trades_df)
    assert result["venue_id"][0] == "coinbase"


def test_normalize_tardis_trades_instrument_id(tardis_trades_df):
    result = normalize_tardis_trades(tardis_trades_df)
    assert "coinbase" in result["instrument_id"][0]
    assert "BTC" in result["instrument_id"][0]


def test_normalize_tardis_trades_price_positive(tardis_trades_df):
    result = normalize_tardis_trades(tardis_trades_df)
    assert result["price"][0] == pytest.approx(42_500.75)
    assert result["price"][0] > 0


def test_normalize_tardis_trades_size_positive(tardis_trades_df):
    result = normalize_tardis_trades(tardis_trades_df)
    assert result["size"][0] == pytest.approx(0.25)
    assert result["size"][0] > 0


def test_normalize_tardis_trades_side(tardis_trades_df):
    result = normalize_tardis_trades(tardis_trades_df)
    assert result["side"][0] == "buy"


def test_normalize_tardis_trades_empty_df():
    empty = pl.DataFrame({
        "source": pl.Series([], dtype=pl.Utf8),
        "symbol": pl.Series([], dtype=pl.Utf8),
        "ts_receive_ns": pl.Series([], dtype=pl.Int64),
        "payload": pl.Series([], dtype=pl.Utf8),
        "payload_hash": pl.Series([], dtype=pl.Utf8),
        "ingest_batch_id": pl.Series([], dtype=pl.Utf8),
    })
    result = normalize_tardis_trades(empty)
    assert result.is_empty()


# ---------------------------------------------------------------------------
# normalize_tardis_book_snapshot_1s
# ---------------------------------------------------------------------------

def test_normalize_tardis_book_snapshot_1s_non_empty(tardis_book_snapshot_df):
    result = normalize_tardis_book_snapshot_1s(tardis_book_snapshot_df)
    assert not result.is_empty()


def test_normalize_tardis_book_snapshot_1s_has_bid_and_ask(tardis_book_snapshot_df):
    result = normalize_tardis_book_snapshot_1s(tardis_book_snapshot_df)
    sides = set(result["side"].to_list())
    assert "bid" in sides
    assert "ask" in sides


def test_normalize_tardis_book_snapshot_1s_level0_exists(tardis_book_snapshot_df):
    result = normalize_tardis_book_snapshot_1s(tardis_book_snapshot_df)
    levels = result["level"].to_list()
    assert 0 in levels


def test_normalize_tardis_book_snapshot_1s_prices_positive(tardis_book_snapshot_df):
    result = normalize_tardis_book_snapshot_1s(tardis_book_snapshot_df)
    assert (result["price"] > 0).all()


def test_normalize_tardis_book_snapshot_1s_venue(tardis_book_snapshot_df):
    result = normalize_tardis_book_snapshot_1s(tardis_book_snapshot_df)
    assert (result["venue_id"] == "coinbase").all()


def test_normalize_tardis_book_snapshot_1s_row_count(tardis_book_snapshot_df):
    # 5 bid levels + 5 ask levels = 10 rows
    result = normalize_tardis_book_snapshot_1s(tardis_book_snapshot_df)
    assert len(result) == 10


# ---------------------------------------------------------------------------
# normalize_tardis_incremental_book_l2
# ---------------------------------------------------------------------------

def test_normalize_tardis_incremental_book_l2_non_empty(tardis_incremental_l2_df):
    result = normalize_tardis_incremental_book_l2(tardis_incremental_l2_df)
    assert not result.is_empty()


def test_normalize_tardis_incremental_book_l2_has_bid_and_ask(tardis_incremental_l2_df):
    result = normalize_tardis_incremental_book_l2(tardis_incremental_l2_df)
    sides = set(result["side"].to_list())
    assert "bid" in sides
    assert "ask" in sides


def test_normalize_tardis_incremental_book_l2_prices_positive(tardis_incremental_l2_df):
    result = normalize_tardis_incremental_book_l2(tardis_incremental_l2_df)
    assert (result["price"] > 0).all()


def test_normalize_tardis_incremental_book_l2_venue_id(tardis_incremental_l2_df):
    result = normalize_tardis_incremental_book_l2(tardis_incremental_l2_df)
    assert (result["venue_id"] == "kraken").all()


def test_normalize_tardis_incremental_book_l2_row_count(tardis_incremental_l2_df):
    result = normalize_tardis_incremental_book_l2(tardis_incremental_l2_df)
    assert len(result) == 3


def test_normalize_tardis_incremental_book_l2_level_indices(tardis_incremental_l2_df):
    """Level counter increments within (ts, venue, symbol, side) groups."""
    result = normalize_tardis_incremental_book_l2(tardis_incremental_l2_df)
    # bid has 2 rows at same ts → levels 0 and 1
    bid_levels = sorted(result.filter(pl.col("side") == "bid")["level"].to_list())
    assert bid_levels == [0, 1]
    # ask has 1 row → level 0
    ask_levels = result.filter(pl.col("side") == "ask")["level"].to_list()
    assert ask_levels == [0]
