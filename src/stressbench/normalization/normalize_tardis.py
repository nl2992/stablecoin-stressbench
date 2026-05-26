"""Normalize Tardis historical archive CSV records into canonical Silver schema.

Tardis CSV rows are stored in Bronze as JSON payloads whose keys are the
original flat CSV column names.  These differ from live WebSocket message
shapes, so dedicated normalizers are required instead of routing through
the Coinbase/Kraken WebSocket normalizers.

Supported Tardis data types
---------------------------
* ``trades``              → :func:`normalize_tardis_trades`
* ``book_snapshot_1s``    → :func:`normalize_tardis_book_snapshot_1s`
* ``incremental_book_L2`` → :func:`normalize_tardis_incremental_book_l2`

Column conventions (Tardis CSV)
--------------------------------
trades
    exchange, symbol, timestamp, localTimestamp, id, side, price, amount

book_snapshot_1s
    exchange, symbol, timestamp, localTimestamp, isSnapshot,
    bids[0].price, bids[0].amount, bids[1].price, bids[1].amount, …,
    asks[0].price, asks[0].amount, …

incremental_book_L2
    exchange, symbol, timestamp, localTimestamp, isSnapshot, side, price, amount

After the Bronze canonicalizer runs (``archive_to_bronze.tardis_to_bronze``),
column names are preserved in the payload JSON with the following renames:
    localTimestamp  → excluded (used as ts_receive_ns in the Bronze wrapper)
    timestamp       → ts_exchange
    exchange        → _exchange
    symbol          → _symbol
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from stressbench.common.ids import instrument_id as make_instrument_id
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# Maximum number of bid/ask levels to scan in book_snapshot_1s rows
_MAX_BOOK_LEVELS = 25


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _ts_to_ns(ts: str | float | int | None) -> int:
    """Convert a Tardis timestamp to nanoseconds since epoch.

    Accepts ISO 8601 strings (``2024-01-15T00:00:01.123456Z``) or
    numeric seconds / milliseconds / nanoseconds.
    """
    if ts is None:
        return 0
    if isinstance(ts, (int, float)):
        v = int(ts)
        # Heuristic: ns > 1e18, ms > 1e12, s otherwise
        if v > 1_000_000_000_000_000_000:
            return v
        if v > 1_000_000_000_000:
            return v * 1_000_000  # ms → ns
        return v * 1_000_000_000  # s → ns
    # String ISO 8601
    try:
        ts_str = str(ts).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # datetime has microsecond resolution; multiply to nanoseconds
        return int(dt.timestamp() * 1_000) * 1_000_000
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Trades normalizer
# ---------------------------------------------------------------------------

def normalize_tardis_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Tardis ``trades`` CSV records into Silver trade-level rows.

    Bronze payload fields used::

        ts_exchange  — exchange-reported ISO 8601 timestamp
        _exchange    — original Tardis exchange name (e.g. ``coinbase``)
        _symbol      — original Tardis symbol (e.g. ``USDC-USD``)
        id           — Tardis trade ID string
        side         — ``"buy"`` or ``"sell"``
        price        — trade price (numeric)
        amount       — trade size in base currency (numeric)

    The ``source`` and ``symbol`` columns on the Bronze row are used as
    fallback venue_id / native_symbol when the payload fields are absent.

    Returns:
        DataFrame conforming to the Silver ``fact_trade`` schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        venue_id = row.get("source") or payload.get("_exchange", "unknown")
        native_symbol = row.get("symbol") or payload.get("_symbol", "UNKNOWN")
        ts_event_ns = _ts_to_ns(payload.get("ts_exchange") or payload.get("timestamp"))

        try:
            price = float(payload.get("price", 0))
            size = float(payload.get("amount", 0))
        except (TypeError, ValueError):
            continue
        if price <= 0 or size <= 0:
            continue

        side_raw = str(payload.get("side", "")).lower()
        side = side_raw if side_raw in ("buy", "sell") else "unknown"

        records.append({
            "ts_event_ns": ts_event_ns,
            "ts_receive_ns": row.get("ts_receive_ns", ts_event_ns),
            "venue_id": venue_id,
            "instrument_id": make_instrument_id(venue_id, native_symbol),
            "native_symbol": native_symbol,
            "trade_id": str(payload.get("id", "")),
            "side": side,
            "price": price,
            "size": size,
            "notional_usd": None,
            "raw_source": f"{venue_id}:tardis_trades",
            "payload_hash": row.get("payload_hash", ""),
            "ingest_batch_id": row.get("ingest_batch_id", ""),
            "is_outlier_price": False,
        })

    if not records:
        return pl.DataFrame()

    result = pl.DataFrame(records)
    # Outlier flag: price > 5% from median
    med = float(result["price"].median() or 0)
    if med > 0:
        result = result.with_columns(
            ((pl.col("price") - med).abs() / med > 0.05).alias("is_outlier_price")
        )
    return result


# ---------------------------------------------------------------------------
# Book snapshot normalizer (book_snapshot_1s)
# ---------------------------------------------------------------------------

def normalize_tardis_book_snapshot_1s(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Tardis ``book_snapshot_1s`` CSV records into Silver book rows.

    Each Bronze row represents one full order-book snapshot.  Bid and ask
    levels are stored as flat payload keys ``bids[i].price`` / ``bids[i].amount``
    and ``asks[i].price`` / ``asks[i].amount`` for ``i`` in 0 … N-1.

    Returns:
        DataFrame conforming to the Silver ``fact_book_level`` schema.
    """
    records = []
    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        venue_id = row.get("source") or payload.get("_exchange", "unknown")
        native_symbol = row.get("symbol") or payload.get("_symbol", "UNKNOWN")
        ts_event_ns = _ts_to_ns(payload.get("ts_exchange") or payload.get("timestamp"))
        ts_receive_ns = row.get("ts_receive_ns", ts_event_ns)

        for i in range(_MAX_BOOK_LEVELS):
            for side, price_key, amt_key in (
                ("bid", f"bids[{i}].price", f"bids[{i}].amount"),
                ("ask", f"asks[{i}].price", f"asks[{i}].amount"),
            ):
                raw_price = payload.get(price_key)
                raw_amt = payload.get(amt_key)
                if raw_price is None:
                    break  # no more levels on this side
                try:
                    price = float(raw_price)
                    size = float(raw_amt or 0)
                except (TypeError, ValueError):
                    continue
                if price <= 0:
                    continue

                records.append({
                    "ts_event_ns": ts_event_ns,
                    "ts_receive_ns": ts_receive_ns,
                    "venue_id": venue_id,
                    "instrument_id": make_instrument_id(venue_id, native_symbol),
                    "native_symbol": native_symbol,
                    "side": side,
                    "level": i,
                    "price": price,
                    "size": size,
                    "checksum": None,
                    "raw_source": f"{venue_id}:tardis_book_snapshot",
                    "payload_hash": row.get("payload_hash", ""),
                    "depth_source": "real_l2_snapshot",
                    "is_crossed_book": False,
                    "is_negative_size": size < 0,
                    "is_sequence_gap": False,
                    "is_checksum_failed": False,
                    "is_stale_quote": False,
                    "is_resync_period": False,
                })

    if not records:
        return pl.DataFrame()
    return _apply_crossed_flag(pl.DataFrame(records))


# ---------------------------------------------------------------------------
# Incremental L2 normalizer (incremental_book_L2)
# ---------------------------------------------------------------------------

def normalize_tardis_incremental_book_l2(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize Tardis ``incremental_book_L2`` CSV records into Silver book rows.

    Each Bronze row is one price-level update with columns::

        side   — ``"bid"`` or ``"ask"``
        price  — price level
        amount — size at that level (0 = delete)

    Rows with the same ``ts_exchange`` and ``isSnapshot == "true"`` form a
    full snapshot; subsequent rows are incremental updates.

    Returns:
        DataFrame conforming to the Silver ``fact_book_level`` schema.
    """
    # Collect all rows grouped by (ts, is_snapshot)
    # Assign a monotonically increasing level index within each (ts, side) group
    level_counters: dict[tuple, int] = {}  # (ts_ns, venue, symbol, side) → next level
    records = []

    for row in df.iter_rows(named=True):
        try:
            payload = json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            continue

        venue_id = row.get("source") or payload.get("_exchange", "unknown")
        native_symbol = row.get("symbol") or payload.get("_symbol", "UNKNOWN")
        ts_event_ns = _ts_to_ns(payload.get("ts_exchange") or payload.get("timestamp"))
        ts_receive_ns = row.get("ts_receive_ns", ts_event_ns)

        side_raw = str(payload.get("side", "")).lower()
        side = side_raw if side_raw in ("bid", "ask") else None
        if side is None:
            continue

        try:
            price = float(payload.get("price", 0))
            size = float(payload.get("amount", 0))
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue

        key = (ts_event_ns, venue_id, native_symbol, side)
        level = level_counters.get(key, 0)
        level_counters[key] = level + 1

        records.append({
            "ts_event_ns": ts_event_ns,
            "ts_receive_ns": ts_receive_ns,
            "venue_id": venue_id,
            "instrument_id": make_instrument_id(venue_id, native_symbol),
            "native_symbol": native_symbol,
            "side": side,
            "level": level,
            "price": price,
            "size": size,
            "checksum": None,
            "raw_source": f"{venue_id}:tardis_incremental_book",
            "payload_hash": row.get("payload_hash", ""),
            "depth_source": "real_l2_incremental",
            "is_crossed_book": False,
            "is_negative_size": size < 0,
            "is_sequence_gap": False,
            "is_checksum_failed": False,
            "is_stale_quote": False,
            "is_resync_period": False,
        })

    if not records:
        return pl.DataFrame()
    return _apply_crossed_flag(pl.DataFrame(records))


# ---------------------------------------------------------------------------
# Quality flag helper
# ---------------------------------------------------------------------------

def _apply_crossed_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Mark rows where best bid >= best ask at the same timestamp/instrument."""
    if df.is_empty():
        return df
    bids = (
        df.filter((pl.col("side") == "bid") & (pl.col("level") == 0))
        .select(["ts_event_ns", "instrument_id", pl.col("price").alias("best_bid")])
    )
    asks = (
        df.filter((pl.col("side") == "ask") & (pl.col("level") == 0))
        .select(["ts_event_ns", "instrument_id", pl.col("price").alias("best_ask")])
    )
    bbo = bids.join(asks, on=["ts_event_ns", "instrument_id"], how="inner")
    crossed = (
        bbo.filter(pl.col("best_bid") >= pl.col("best_ask"))
        .select(["ts_event_ns", "instrument_id"])
        .with_columns(pl.lit(True).alias("_crossed"))
    )
    df = df.join(crossed, on=["ts_event_ns", "instrument_id"], how="left")
    df = df.with_columns(
        pl.col("_crossed").fill_null(False).alias("is_crossed_book")
    ).drop("_crossed")
    return df
