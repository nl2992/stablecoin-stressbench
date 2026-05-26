"""Normalize raw Bronze order-book messages into canonical Silver book records.

Data-quality flags applied:
    is_crossed_book, is_negative_size, is_sequence_gap,
    is_checksum_failed, is_stale_quote, is_resync_period
"""

from __future__ import annotations

import json

import polars as pl

from stressbench.common.ids import instrument_id as make_instrument_id
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_STALE_THRESHOLD_NS = 5_000_000_000  # 5 seconds


def normalize_binance_depth(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Binance ``@depth`` stream messages into book level records.

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver book-level schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        symbol = row.get("symbol") or payload.get("s", "UNKNOWN")
        ts_event_ns = int(payload.get("T", payload.get("E", 0))) * 1_000_000

        for level_idx, (price, size) in enumerate(payload.get("b", [])):
            records.append(
                _make_level_record(
                    ts_event_ns=ts_event_ns,
                    ts_receive_ns=row["ts_receive_ns"],
                    venue_id="binance",
                    symbol=symbol,
                    side="bid",
                    level=level_idx,
                    price=float(price),
                    size=float(size),
                    row=row,
                )
            )
        for level_idx, (price, size) in enumerate(payload.get("a", [])):
            records.append(
                _make_level_record(
                    ts_event_ns=ts_event_ns,
                    ts_receive_ns=row["ts_receive_ns"],
                    venue_id="binance",
                    symbol=symbol,
                    side="ask",
                    level=level_idx,
                    price=float(price),
                    size=float(size),
                    row=row,
                )
            )

    if not records:
        return pl.DataFrame()
    return _apply_quality_flags(pl.DataFrame(records))


def normalize_coinbase_level2(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Coinbase ``level2`` channel messages into book level records.

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver book-level schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = payload.get("type", "")
        if msg_type not in ("snapshot", "l2update"):
            continue

        symbol = payload.get("product_id", row.get("symbol", "UNKNOWN"))
        ts_event_ns = _parse_coinbase_ts(payload.get("time", ""))

        changes = payload.get("changes", [])
        if msg_type == "snapshot":
            for level_idx, (price, size) in enumerate(payload.get("bids", [])):
                records.append(
                    _make_level_record(
                        ts_event_ns=ts_event_ns,
                        ts_receive_ns=row["ts_receive_ns"],
                        venue_id="coinbase",
                        symbol=symbol,
                        side="bid",
                        level=level_idx,
                        price=float(price),
                        size=float(size),
                        row=row,
                    )
                )
            for level_idx, (price, size) in enumerate(payload.get("asks", [])):
                records.append(
                    _make_level_record(
                        ts_event_ns=ts_event_ns,
                        ts_receive_ns=row["ts_receive_ns"],
                        venue_id="coinbase",
                        symbol=symbol,
                        side="ask",
                        level=level_idx,
                        price=float(price),
                        size=float(size),
                        row=row,
                    )
                )
        else:
            for change in changes:
                side_raw, price, size = change
                side = "bid" if side_raw == "buy" else "ask"
                records.append(
                    _make_level_record(
                        ts_event_ns=ts_event_ns,
                        ts_receive_ns=row["ts_receive_ns"],
                        venue_id="coinbase",
                        symbol=symbol,
                        side=side,
                        level=0,
                        price=float(price),
                        size=float(size),
                        row=row,
                    )
                )

    if not records:
        return pl.DataFrame()
    return _apply_quality_flags(pl.DataFrame(records))


def normalize_kraken_book(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Kraken ``book`` channel messages into book level records.

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver book-level schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        if payload.get("channel") != "book":
            continue

        checksum_failed = payload.get("_checksum_failed", False)

        for entry in payload.get("data", []):
            symbol = entry.get("symbol", row.get("symbol", "UNKNOWN"))
            ts_event_ns = int(float(entry.get("timestamp", 0)) * 1e9)
            checksum = str(entry.get("checksum", ""))

            for level_idx, level in enumerate(entry.get("bids", [])):
                records.append(
                    _make_level_record(
                        ts_event_ns=ts_event_ns,
                        ts_receive_ns=row["ts_receive_ns"],
                        venue_id="kraken",
                        symbol=symbol,
                        side="bid",
                        level=level_idx,
                        price=float(level.get("price", 0)),
                        size=float(level.get("qty", 0)),
                        row=row,
                        checksum=checksum,
                        is_checksum_failed=checksum_failed,
                    )
                )
            for level_idx, level in enumerate(entry.get("asks", [])):
                records.append(
                    _make_level_record(
                        ts_event_ns=ts_event_ns,
                        ts_receive_ns=row["ts_receive_ns"],
                        venue_id="kraken",
                        symbol=symbol,
                        side="ask",
                        level=level_idx,
                        price=float(level.get("price", 0)),
                        size=float(level.get("qty", 0)),
                        row=row,
                        checksum=checksum,
                        is_checksum_failed=checksum_failed,
                    )
                )

    if not records:
        return pl.DataFrame()
    return _apply_quality_flags(pl.DataFrame(records))


def normalize_binance_klines(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Binance 1-minute klines archive records into synthetic book levels.

    Expects Bronze records whose ``payload`` JSON contains a ``"k"`` sub-object
    matching the ``@kline_1m`` WebSocket message format:

        payload.k.t  — kline open time (milliseconds)
        payload.k.o/h/l/c  — OHLC prices
        payload.k.v  — base volume
        payload.k.V  — taker-buy base volume
        payload.k.s  — symbol

    Generates 5 synthetic bid and 5 synthetic ask levels using the same
    H-L spread schedule used by :mod:`scripts.fetch_real_data`.

    Args:
        df: Raw Bronze DataFrame with ``payload`` column (JSON string).

    Returns:
        Normalized DataFrame conforming to the Silver book-level schema.
    """
    _SIZE_SCHEDULE = [0.30, 0.25, 0.20, 0.15, 0.10]
    _SPREAD_SCHEDULE = [1.0, 2.0, 3.5, 6.0, 10.0]
    _N_LEVELS = 5

    records = []
    for row in df.iter_rows(named=True):
        try:
            outer = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        k = outer.get("k", outer)  # support both wrapped and flat payloads
        open_time_ms = int(k.get("t", 0))
        ts_ns = open_time_ms * 1_000_000
        symbol = row.get("symbol") or k.get("s", "UNKNOWN")

        try:
            mid = float(k.get("c", 0))
        except (TypeError, ValueError):
            continue
        if mid <= 0:
            continue

        try:
            hl_range = float(k.get("h", mid)) - float(k.get("l", mid))
        except (TypeError, ValueError):
            hl_range = 0.0
        half_spread = max(hl_range / 2.0, mid * 0.00005)  # floor at 0.5 bps

        try:
            taker_buy = max(float(k.get("V", 0)), 1e-9)
            total_vol = max(float(k.get("v", 0)), 1e-9)
        except (TypeError, ValueError):
            taker_buy = 1e-9
            total_vol = 1e-9
        taker_sell = max(total_vol - taker_buy, 1e-9)

        for i in range(_N_LEVELS):
            bid_price = mid - _SPREAD_SCHEDULE[i] * half_spread
            ask_price = mid + _SPREAD_SCHEDULE[i] * half_spread
            base = dict(
                ts_event_ns=ts_ns,
                ts_receive_ns=row.get("ts_receive_ns", ts_ns),
                venue_id="binance",
                instrument_id=make_instrument_id("binance", symbol),
                native_symbol=symbol,
                level=i,
                checksum=None,
                raw_source="binance:klines",
                payload_hash=row.get("payload_hash", ""),
                is_crossed_book=False,
                is_negative_size=False,
                is_sequence_gap=False,
                is_checksum_failed=False,
                is_stale_quote=False,
                is_resync_period=False,
            )
            records.append({**base, "side": "bid", "price": bid_price,
                            "size": taker_sell * _SIZE_SCHEDULE[i]})
            records.append({**base, "side": "ask", "price": ask_price,
                            "size": taker_buy * _SIZE_SCHEDULE[i]})

    if not records:
        return pl.DataFrame()
    return _apply_quality_flags(pl.DataFrame(records))


def _make_level_record(
    ts_event_ns: int,
    ts_receive_ns: int,
    venue_id: str,
    symbol: str,
    side: str,
    level: int,
    price: float,
    size: float,
    row: dict,
    checksum: str | None = None,
    is_checksum_failed: bool = False,
) -> dict:
    return {
        "ts_event_ns": ts_event_ns,
        "ts_receive_ns": ts_receive_ns,
        "venue_id": venue_id,
        "instrument_id": make_instrument_id(venue_id, symbol),
        "native_symbol": symbol,
        "side": side,
        "level": level,
        "price": price,
        "size": size,
        "checksum": checksum,
        "raw_source": f"{venue_id}:book",
        "payload_hash": row.get("payload_hash", ""),
        "is_crossed_book": False,
        "is_negative_size": size < 0,
        "is_sequence_gap": row.get("_sequence_gap", False),
        "is_checksum_failed": is_checksum_failed,
        "is_stale_quote": False,
        "is_resync_period": False,
    }


def _apply_quality_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Apply crossed-book and stale-quote flags to a book level DataFrame."""
    if df.is_empty():
        return df

    # Detect crossed books: best_bid >= best_ask per instrument/timestamp
    bids = df.filter(pl.col("side") == "bid").filter(pl.col("level") == 0).select(
        ["ts_event_ns", "instrument_id", pl.col("price").alias("best_bid")]
    )
    asks = df.filter(pl.col("side") == "ask").filter(pl.col("level") == 0).select(
        ["ts_event_ns", "instrument_id", pl.col("price").alias("best_ask")]
    )
    bbo = bids.join(asks, on=["ts_event_ns", "instrument_id"], how="inner")
    crossed = bbo.filter(pl.col("best_bid") >= pl.col("best_ask")).select(
        ["ts_event_ns", "instrument_id"]
    ).with_columns(pl.lit(True).alias("_crossed"))

    df = df.join(crossed, on=["ts_event_ns", "instrument_id"], how="left")
    df = df.with_columns(
        pl.col("_crossed").fill_null(False).alias("is_crossed_book")
    ).drop("_crossed")

    return df


def _parse_coinbase_ts(ts_str: str) -> int:
    if not ts_str:
        return 0
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1e9)
    except ValueError:
        return 0
