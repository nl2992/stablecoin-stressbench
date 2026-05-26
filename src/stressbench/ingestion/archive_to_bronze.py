"""Canonicalizer: convert historical archive files into the canonical Bronze layout.

Live WebSocket collectors write through ``raw_writer.py``, which produces:

    data/bronze/
      venue=<venue>/
        channel=<channel>/
          symbol=<symbol>/
            date=YYYY-MM-DD/
              hour=HH/
                part-<uuid>.parquet

Historical archive downloads (Binance Vision, Tardis) write to vendor-specific
paths that the Silver builder cannot scan.  This module reads those files and
re-emits them into the canonical structure so the full Bronze → Silver → Gold
pipeline works identically for both live and archive data.

Each output Parquet file has the same schema as ``raw_writer.write_raw_batch``:
    source, channel, symbol, ts_exchange, ts_receive_ns,
    payload (JSON string), payload_hash, schema_version, ingest_batch_id

Supported conversions
---------------------
* Binance Vision aggTrades CSV  →  ``venue=binance / channel=aggTrades``
* Binance Vision klines CSV     →  ``venue=binance / channel=klines``
* Tardis CSV.gz                 →  ``venue=<exchange> / channel=<channel>``

Etherscan transfers are handled in
:func:`stressbench.ingestion.etherscan_loader.save_transfers_to_bronze`,
which has been updated to write to the canonical path directly.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import polars as pl

from stressbench.common.config import bronze_root as _bronze_root_fn
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_SCHEMA_VERSION = "raw.v1"
_ARCHIVE_BATCH_ID = "binance_vision_archive"
_TARDIS_BATCH_ID = "tardis_archive"

# Tardis data_type → canonical Bronze channel name
_TARDIS_CHANNEL_MAP: dict[str, str] = {
    "trades": "trade",
    "incremental_book_L2": "depth",
    "book_snapshot_1s": "depth",
    "quotes": "quote",
    "book_ticker": "bookTicker",
}

# Binance aggTrades CSV columns (no header row in archive files)
_AGG_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match",
]
_AGG_DTYPES: dict[str, type] = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "transact_time": pl.Int64,
    "is_buyer_maker": pl.Utf8,
    "is_best_match": pl.Utf8,
}

# Binance klines CSV columns (no header row)
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
_KLINE_DTYPES: dict[str, type] = {c: pl.Float64 for c in _KLINE_COLS}
_KLINE_DTYPES["open_time"] = pl.Int64   # type: ignore[assignment]
_KLINE_DTYPES["close_time"] = pl.Int64  # type: ignore[assignment]
_KLINE_DTYPES["trade_count"] = pl.Int64  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_bronze_partition(
    df: pl.DataFrame,
    venue: str,
    channel: str,
    symbol: str,
    date: str,
    hour: int,
    bronze_root: Path,
    overwrite: bool = False,
) -> bool:
    """Write one hour-partition of Bronze Parquet.  Returns True if written."""
    out_dir = (
        bronze_root
        / f"venue={venue}"
        / f"channel={channel}"
        / f"symbol={symbol}"
        / f"date={date}"
        / f"hour={hour:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "part-0.parquet"
    if out_file.exists() and not overwrite:
        logger.debug("Already exists, skipping: %s", out_file)
        return False
    df.write_parquet(out_file)
    return True


def _rows_to_bronze_df(
    rows: list[dict],
    venue: str,
    channel: str,
    symbol: str,
    batch_id: str,
) -> pl.DataFrame:
    """Convert a list of raw record dicts to a Bronze-schema DataFrame."""
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([
        {
            "source": venue,
            "channel": channel,
            "symbol": symbol,
            "ts_exchange": r.get("ts_exchange", ""),
            "ts_receive_ns": r["ts_receive_ns"],
            "payload": json.dumps(r["payload"], sort_keys=True),
            "payload_hash": r.get("payload_hash", ""),
            "schema_version": _SCHEMA_VERSION,
            "ingest_batch_id": batch_id,
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Public API — Binance aggTrades
# ---------------------------------------------------------------------------

def binance_aggtrades_to_bronze(
    csv_path: Path,
    symbol: str,
    date: str,
    bronze_root: Path | None = None,
    overwrite: bool = False,
) -> int:
    """Convert a Binance aggTrades archive CSV to canonical Bronze Parquet.

    Constructs a ``@aggTrade``-compatible JSON payload for each row so the
    existing :func:`~stressbench.normalization.normalize_trades.normalize_binance_trades`
    normalizer can process the file unchanged.

    Args:
        csv_path: Path to the ``{SYMBOL}-aggTrades-{DATE}.csv`` file.
        symbol: Binance symbol string (e.g. ``"BTCUSDT"``).
        date: ISO date string ``YYYY-MM-DD``.
        bronze_root: Override for Bronze root directory.
        overwrite: Re-write even if the partition already exists.

    Returns:
        Number of rows written.
    """
    root = bronze_root or _bronze_root_fn()

    df = pl.read_csv(
        csv_path,
        has_header=False,
        new_columns=_AGG_COLS[:df_col_count(csv_path)],
        schema_overrides=_AGG_DTYPES,
        ignore_errors=True,
    )
    # Some archive files include 8 columns (with is_best_match), some 7
    # Normalise to always have at least the 7 required columns
    for col in _AGG_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))

    # Build per-row payload dicts (WS @aggTrade format expected by normalizer)
    records: list[dict] = []
    for row in df.iter_rows(named=True):
        ts_ms = int(row["transact_time"])
        is_buyer_maker = str(row.get("is_buyer_maker", "false")).strip().lower() == "true"
        payload = {
            "T": ts_ms,
            "p": str(row["price"]),
            "q": str(row["quantity"]),
            "t": int(row["agg_trade_id"]),
            "m": is_buyer_maker,
            "s": symbol,
        }
        records.append({
            "ts_receive_ns": ts_ms * 1_000_000,
            "ts_exchange": str(ts_ms),
            "payload": payload,
        })

    if not records:
        return 0

    # Partition by hour and write one file per hour
    hour_buckets: dict[int, list[dict]] = {}
    for rec in records:
        h = (rec["ts_receive_ns"] // 3_600_000_000_000) % 24
        hour_buckets.setdefault(h, []).append(rec)

    written = 0
    for hour, recs in sorted(hour_buckets.items()):
        bronze_df = _rows_to_bronze_df(recs, "binance", "aggTrades", symbol, _ARCHIVE_BATCH_ID)
        if _write_bronze_partition(bronze_df, "binance", "aggTrades", symbol, date, hour, root, overwrite):
            written += len(recs)

    logger.info("[archive_to_bronze] aggTrades %s %s → %d rows", symbol, date, written)
    return written


# ---------------------------------------------------------------------------
# Public API — Binance klines
# ---------------------------------------------------------------------------

def binance_klines_to_bronze(
    csv_path: Path,
    symbol: str,
    date: str,
    bronze_root: Path | None = None,
    overwrite: bool = False,
) -> int:
    """Convert a Binance 1-minute klines archive CSV to canonical Bronze Parquet.

    Constructs a ``@kline``-compatible JSON payload for each row so the
    :func:`~stressbench.normalization.normalize_books.normalize_binance_klines`
    normalizer can produce synthetic book level records.

    Args:
        csv_path: Path to the ``{SYMBOL}-1m-{DATE}.csv`` file.
        symbol: Binance symbol string.
        date: ISO date string ``YYYY-MM-DD``.
        bronze_root: Override for Bronze root directory.
        overwrite: Re-write even if the partition already exists.

    Returns:
        Number of rows written.
    """
    root = bronze_root or _bronze_root_fn()

    df = pl.read_csv(
        csv_path,
        has_header=False,
        new_columns=_KLINE_COLS,
        schema_overrides=_KLINE_DTYPES,
        ignore_errors=True,
    )

    records: list[dict] = []
    for row in df.iter_rows(named=True):
        open_time_ms = int(row["open_time"])
        payload = {
            "k": {
                "t": open_time_ms,
                "T": int(row["close_time"]),
                "s": symbol,
                "o": str(row["open"]),
                "h": str(row["high"]),
                "l": str(row["low"]),
                "c": str(row["close"]),
                "v": str(row["volume"]),
                "q": str(row["quote_volume"]),
                "n": int(row["trade_count"]),
                "V": str(row["taker_buy_base"]),
                "Q": str(row["taker_buy_quote"]),
            }
        }
        records.append({
            "ts_receive_ns": open_time_ms * 1_000_000,
            "ts_exchange": str(open_time_ms),
            "payload": payload,
        })

    if not records:
        return 0

    hour_buckets: dict[int, list[dict]] = {}
    for rec in records:
        h = (rec["ts_receive_ns"] // 3_600_000_000_000) % 24
        hour_buckets.setdefault(h, []).append(rec)

    written = 0
    for hour, recs in sorted(hour_buckets.items()):
        bronze_df = _rows_to_bronze_df(recs, "binance", "klines", symbol, _ARCHIVE_BATCH_ID)
        if _write_bronze_partition(bronze_df, "binance", "klines", symbol, date, hour, root, overwrite):
            written += len(recs)

    logger.info("[archive_to_bronze] klines %s %s → %d rows", symbol, date, written)
    return written


# ---------------------------------------------------------------------------
# Public API — Tardis
# ---------------------------------------------------------------------------

def tardis_to_bronze(
    file_path: Path,
    exchange: str,
    symbol: str,
    data_type: str,
    date: str,
    bronze_root: Path | None = None,
    overwrite: bool = False,
) -> int:
    """Convert a Tardis CSV.gz download to canonical Bronze Parquet.

    Maps Tardis ``data_type`` to a canonical Bronze ``channel`` using
    ``_TARDIS_CHANNEL_MAP``.  Rows are wrapped in a JSON payload whose
    structure mirrors the native exchange WebSocket message as closely as
    possible so the existing Silver normalizers can process them.

    Args:
        file_path: Path to the Tardis ``*.csv.gz`` file.
        exchange: Tardis exchange name (e.g. ``"coinbase"``).
        symbol: Tardis symbol string (e.g. ``"USDC-USD"``).
        data_type: Tardis data type (e.g. ``"trades"``).
        date: ISO date string ``YYYY-MM-DD``.
        bronze_root: Override for Bronze root directory.
        overwrite: Re-write even if the partition already exists.

    Returns:
        Number of rows written.
    """
    root = bronze_root or _bronze_root_fn()
    channel = _TARDIS_CHANNEL_MAP.get(data_type, data_type)

    try:
        df = pl.read_csv(file_path, infer_schema_length=10000)
    except Exception as exc:
        logger.warning("Failed to read Tardis file %s: %s", file_path, exc)
        return 0

    if df.is_empty():
        return 0

    # Rename common Tardis columns to canonical names before building payload
    rename_map = {
        "localTimestamp": "ts_receive_us",
        "timestamp": "ts_exchange",
        "exchange": "_exchange",
        "symbol": "_symbol",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(existing)

    records: list[dict] = []
    for row in df.iter_rows(named=True):
        # ts_receive_us is Tardis local receive time in microseconds
        ts_receive_ns = int(row.get("ts_receive_us", 0)) * 1000
        if ts_receive_ns == 0:
            continue

        # Build payload from all row fields (preserves full Tardis record)
        payload = {k: v for k, v in row.items()
                   if k not in ("ts_receive_us",) and v is not None}

        records.append({
            "ts_receive_ns": ts_receive_ns,
            "ts_exchange": str(row.get("ts_exchange", "")),
            "payload": payload,
        })

    if not records:
        return 0

    hour_buckets: dict[int, list[dict]] = {}
    for rec in records:
        h = (rec["ts_receive_ns"] // 3_600_000_000_000) % 24
        hour_buckets.setdefault(h, []).append(rec)

    written = 0
    for hour, recs in sorted(hour_buckets.items()):
        bronze_df = _rows_to_bronze_df(recs, exchange, channel, symbol, _TARDIS_BATCH_ID)
        if _write_bronze_partition(bronze_df, exchange, channel, symbol, date, hour, root, overwrite):
            written += len(recs)

    logger.info("[archive_to_bronze] tardis %s/%s %s %s → %d rows",
                exchange, symbol, data_type, date, written)
    return written


# ---------------------------------------------------------------------------
# Utility: detect CSV column count without reading the full file
# ---------------------------------------------------------------------------

def df_col_count(csv_path: Path) -> int:
    """Return the number of columns in the first row of a CSV file."""
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        return len(first_line.split(","))
    except Exception:
        return len(_AGG_COLS)
