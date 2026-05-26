"""Tests for archive_to_bronze — vendor CSV → canonical Bronze Parquet conversion.

Each test writes a minimal in-memory CSV to a temporary directory, calls the
appropriate canonicalizer, and asserts the expected Hive-partitioned output.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_aggtrades_csv(tmp_path: Path, symbol: str, date: str) -> Path:
    """Write a minimal Binance aggTrades archive CSV (no header)."""
    rows = [
        # agg_trade_id, price, qty, first_trade_id, last_trade_id, transact_time, is_buyer_maker
        [1, "42500.00", "0.10", 1000, 1001, 1704067200000, "False"],
        [2, "42501.00", "0.20", 1002, 1003, 1704067260000, "True"],
    ]
    path = tmp_path / f"{symbol}-aggTrades-{date}.csv"
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def _make_klines_csv(tmp_path: Path, symbol: str, date: str) -> Path:
    """Write a minimal Binance 1-minute klines archive CSV (no header)."""
    rows = [
        # open_time, open, high, low, close, volume, close_time, quote_vol, trades, taker_base, taker_quote, ignore
        [1704067200000, "42500", "42600", "42400", "42550", "10.0",
         1704067259999, "425500", 120, "5.0", "212750", "0"],
        [1704067260000, "42550", "42700", "42450", "42620", "12.0",
         1704067319999, "510240", 140, "6.0", "255120", "0"],
    ]
    path = tmp_path / f"{symbol}-1m-{date}.csv"
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return path


def _make_tardis_trades_csv(tmp_path: Path, exchange: str, symbol: str) -> Path:
    """Write a minimal Tardis trades CSV."""
    rows = [
        {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": "2024-01-01T00:00:00.000000Z",
            "localTimestamp": "1704067200000000",
            "id": "t001",
            "side": "buy",
            "price": "42500.00",
            "amount": "0.10",
        },
        {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": "2024-01-01T00:00:01.000000Z",
            "localTimestamp": "1704067201000000",
            "id": "t002",
            "side": "sell",
            "price": "42510.00",
            "amount": "0.05",
        },
    ]
    path = tmp_path / f"{exchange}_{symbol}_trades.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_tardis_book_snapshot_csv(tmp_path: Path, exchange: str, symbol: str) -> Path:
    """Write a minimal Tardis book_snapshot_1s CSV (2 levels each side)."""
    row = {
        "exchange": exchange,
        "symbol": symbol,
        "timestamp": "2024-01-01T00:00:00.000000Z",
        "localTimestamp": "1704067200000000",
        "isSnapshot": "true",
        "bids[0].price": "42500.00",
        "bids[0].amount": "1.0",
        "bids[1].price": "42495.00",
        "bids[1].amount": "2.0",
        "asks[0].price": "42510.00",
        "asks[0].amount": "0.8",
        "asks[1].price": "42515.00",
        "asks[1].amount": "1.5",
    }
    path = tmp_path / f"{exchange}_{symbol}_book_snapshot_1s.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Tests — Binance aggTrades
# ---------------------------------------------------------------------------

def test_aggtrades_writes_canonical_channel(tmp_path):
    from stressbench.ingestion.archive_to_bronze import binance_aggtrades_to_bronze

    symbol, date = "USDCUSDT", "2024-01-01"
    csv_path = _make_aggtrades_csv(tmp_path, symbol, date)
    n = binance_aggtrades_to_bronze(csv_path, symbol, date, bronze_root=tmp_path)

    assert n > 0
    parquets = list(tmp_path.glob("venue=binance/channel=aggTrades/**/*.parquet"))
    assert parquets, "No Bronze Parquet files written for aggTrades"


def test_aggtrades_hive_layout(tmp_path):
    from stressbench.ingestion.archive_to_bronze import binance_aggtrades_to_bronze

    symbol, date = "USDCUSDT", "2024-01-01"
    csv_path = _make_aggtrades_csv(tmp_path, symbol, date)
    binance_aggtrades_to_bronze(csv_path, symbol, date, bronze_root=tmp_path)

    parquets = list(tmp_path.glob(
        f"venue=binance/channel=aggTrades/symbol={symbol}/date={date}/hour=*/part-0.parquet"
    ))
    assert parquets


def test_aggtrades_payload_is_json(tmp_path):
    from stressbench.ingestion.archive_to_bronze import binance_aggtrades_to_bronze

    symbol, date = "USDCUSDT", "2024-01-01"
    csv_path = _make_aggtrades_csv(tmp_path, symbol, date)
    binance_aggtrades_to_bronze(csv_path, symbol, date, bronze_root=tmp_path)

    p = next(tmp_path.glob("venue=binance/channel=aggTrades/**/*.parquet"))
    df = pl.read_parquet(p)
    payload = json.loads(df["payload"][0])
    assert "p" in payload   # price field in WS aggTrade format
    assert "q" in payload   # qty
    assert "T" in payload   # transaction time


def test_aggtrades_ingest_batch_id(tmp_path):
    from stressbench.ingestion.archive_to_bronze import binance_aggtrades_to_bronze

    symbol, date = "USDCUSDT", "2024-01-01"
    csv_path = _make_aggtrades_csv(tmp_path, symbol, date)
    binance_aggtrades_to_bronze(csv_path, symbol, date, bronze_root=tmp_path)

    p = next(tmp_path.glob("venue=binance/channel=aggTrades/**/*.parquet"))
    df = pl.read_parquet(p)
    assert df["ingest_batch_id"][0] == "binance_vision_archive"


# ---------------------------------------------------------------------------
# Tests — Binance klines
# ---------------------------------------------------------------------------

def test_klines_writes_canonical_channel(tmp_path):
    from stressbench.ingestion.archive_to_bronze import binance_klines_to_bronze

    symbol, date = "BTCUSDT", "2024-01-01"
    csv_path = _make_klines_csv(tmp_path, symbol, date)
    n = binance_klines_to_bronze(csv_path, symbol, date, bronze_root=tmp_path)

    assert n > 0
    parquets = list(tmp_path.glob("venue=binance/channel=klines/**/*.parquet"))
    assert parquets, "No Bronze Parquet files written for klines"


def test_klines_payload_has_k_object(tmp_path):
    from stressbench.ingestion.archive_to_bronze import binance_klines_to_bronze

    symbol, date = "BTCUSDT", "2024-01-01"
    csv_path = _make_klines_csv(tmp_path, symbol, date)
    binance_klines_to_bronze(csv_path, symbol, date, bronze_root=tmp_path)

    p = next(tmp_path.glob("venue=binance/channel=klines/**/*.parquet"))
    df = pl.read_parquet(p)
    payload = json.loads(df["payload"][0])
    assert "k" in payload
    k = payload["k"]
    assert "o" in k and "h" in k and "l" in k and "c" in k


# ---------------------------------------------------------------------------
# Tests — Tardis trades
# ---------------------------------------------------------------------------

def test_tardis_trades_writes_tardis_channel(tmp_path):
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    exchange, symbol, date = "coinbase", "BTC-USD", "2024-01-01"
    csv_path = _make_tardis_trades_csv(tmp_path, exchange, symbol)
    n = tardis_to_bronze(csv_path, exchange, symbol, "trades", date, bronze_root=tmp_path)

    assert n > 0
    # Channel must be tardis_trades, not "trade" or "depth"
    parquets = list(tmp_path.glob("venue=coinbase/channel=tardis_trades/**/*.parquet"))
    assert parquets, "Channel should be tardis_trades, not 'trade'"


def test_tardis_trades_channel_not_generic(tmp_path):
    """Tardis channels must not collide with live WebSocket channel names."""
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    exchange, symbol, date = "coinbase", "BTC-USD", "2024-01-01"
    csv_path = _make_tardis_trades_csv(tmp_path, exchange, symbol)
    tardis_to_bronze(csv_path, exchange, symbol, "trades", date, bronze_root=tmp_path)

    bad_channels = list(tmp_path.glob("venue=coinbase/channel=trade/**/*.parquet"))
    assert not bad_channels, "Tardis trade data landed in generic 'trade' channel"


def test_tardis_trades_ingest_batch_id(tmp_path):
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    exchange, symbol, date = "coinbase", "BTC-USD", "2024-01-01"
    csv_path = _make_tardis_trades_csv(tmp_path, exchange, symbol)
    tardis_to_bronze(csv_path, exchange, symbol, "trades", date, bronze_root=tmp_path)

    p = next(tmp_path.glob("venue=coinbase/channel=tardis_trades/**/*.parquet"))
    df = pl.read_parquet(p)
    assert df["ingest_batch_id"][0] == "tardis_archive"


def test_tardis_trades_payload_is_json(tmp_path):
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    exchange, symbol, date = "coinbase", "BTC-USD", "2024-01-01"
    csv_path = _make_tardis_trades_csv(tmp_path, exchange, symbol)
    tardis_to_bronze(csv_path, exchange, symbol, "trades", date, bronze_root=tmp_path)

    p = next(tmp_path.glob("venue=coinbase/channel=tardis_trades/**/*.parquet"))
    df = pl.read_parquet(p)
    payload = json.loads(df["payload"][0])
    assert "price" in payload or "ts_exchange" in payload


# ---------------------------------------------------------------------------
# Tests — Tardis book_snapshot_1s
# ---------------------------------------------------------------------------

def test_tardis_book_snapshot_writes_tardis_channel(tmp_path):
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    exchange, symbol, date = "coinbase", "BTC-USD", "2024-01-01"
    csv_path = _make_tardis_book_snapshot_csv(tmp_path, exchange, symbol)
    n = tardis_to_bronze(csv_path, exchange, symbol, "book_snapshot_1s", date, bronze_root=tmp_path)

    assert n > 0
    parquets = list(tmp_path.glob(
        "venue=coinbase/channel=tardis_book_snapshot_1s/**/*.parquet"
    ))
    assert parquets, "Channel should be tardis_book_snapshot_1s, not 'depth'"


def test_tardis_book_snapshot_channel_not_depth(tmp_path):
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    exchange, symbol, date = "coinbase", "BTC-USD", "2024-01-01"
    csv_path = _make_tardis_book_snapshot_csv(tmp_path, exchange, symbol)
    tardis_to_bronze(csv_path, exchange, symbol, "book_snapshot_1s", date, bronze_root=tmp_path)

    bad = list(tmp_path.glob("venue=coinbase/channel=depth/**/*.parquet"))
    assert not bad, "Tardis book snapshot landed in generic 'depth' channel"
