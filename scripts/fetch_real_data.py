#!/usr/bin/env python3
"""Fetch real historical data from public APIs for all benchmark event windows.

Data sources (all free, no API key required):
  - Binance Vision spot archive: aggTrades, klines (1-minute OHLCV → synthetic book)
  - Binance Vision USDM futures archive: bookDepth (real percentage-based depth snapshots)
  - Coinbase REST API: 1-minute candles for BTC-USD, USDC-USD, USDT-USD

Writes directly to the Silver Hive-partitioned Parquet layer so that
build_features.py can be run with --skip-silver.

For BTCUSDT, --futures-bookdepth replaces kline-derived synthetic book with real
futures bookDepth snapshots (~30s cadence, cumulative depth at ±1–5% from mid).
For other symbols, kline-derived synthetic book is used as an approximation.

Usage:
    python scripts/fetch_real_data.py
    python scripts/fetch_real_data.py --futures-bookdepth
    python scripts/fetch_real_data.py --windows usdc_depeg_2023 terra_luna_2022
    python scripts/fetch_real_data.py --symbols BTCUSDT USDCUSDT --dry-run
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_BINANCE_BASE = "https://data.binance.vision/data/spot/daily"
_COINBASE_BASE = "https://api.exchange.coinbase.com"

# Binance symbols to attempt — 404s are silently skipped
_BINANCE_SYMBOLS = ["BTCUSDT", "USDCUSDT", "ETHUSDT", "BTCUSDC"]

# Coinbase products that give us USD-quoted reference prices
_COINBASE_PRODUCTS = ["BTC-USD", "USDT-USD"]  # USDC-USD not listed on Coinbase Exchange

# All benchmark event windows (mirrors event_windows.yaml)
_ALL_WINDOWS: dict[str, tuple[str, str]] = {
    "usdc_depeg_2023":        ("2023-03-10", "2023-03-14"),
    "usdc_depeg_recovery":    ("2023-03-15", "2023-03-20"),
    "terra_luna_2022":        ("2022-05-07", "2022-05-14"),
    "normal_control_feb2023": ("2023-02-01", "2023-02-07"),
    "normal_control_jan2022": ("2022-01-10", "2022-01-16"),
    "normal_control_q12024":  ("2024-01-15", "2024-01-21"),
}

# Silver schemas
_TRADE_COLS = [
    "ts_event_ns", "ts_receive_ns", "venue_id", "instrument_id",
    "native_symbol", "trade_id", "side", "price", "size",
    "notional_usd", "raw_source", "payload_hash", "ingest_batch_id",
    "is_outlier_price",
]
_BOOK_COLS = [
    "ts_event_ns", "ts_receive_ns", "venue_id", "instrument_id",
    "native_symbol", "side", "level", "price", "size",
    "checksum", "raw_source", "payload_hash",
    "is_crossed_book", "is_negative_size", "is_sequence_gap",
    "is_checksum_failed", "is_stale_quote", "is_resync_period",
]

# Synthetic book: number of depth levels to generate per side from kline data
_BOOK_LEVELS = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _date_range(start: str, end: str) -> list[str]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _write_silver(
    silver_root: Path,
    venue: str,
    channel: str,
    symbol: str,
    date_str: str,
    hour: int,
    df: pl.DataFrame,
    overwrite: bool,
) -> None:
    if df.is_empty():
        return
    out_dir = (
        silver_root
        / f"venue={venue}"
        / f"channel={channel}"
        / f"symbol={symbol}"
        / f"date={date_str}"
        / f"hour={hour:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "part-0.parquet"
    if out.exists() and not overwrite:
        return
    df.write_parquet(out)


def _get_zip_csv(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            return None
        return r.content
    except Exception as exc:
        logger.warning("Download failed %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Binance aggTrades → Silver trades
# ---------------------------------------------------------------------------

_AGG_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id",
    "transact_time", "is_buyer_maker", "is_best_match",
]
_AGG_DTYPES = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "transact_time": pl.Int64,
    "is_buyer_maker": pl.Utf8,
    "is_best_match": pl.Utf8,
}


def ingest_binance_aggtrades(
    symbol: str,
    date_str: str,
    silver_root: Path,
    overwrite: bool,
) -> int:
    url = f"{_BINANCE_BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    content = _get_zip_csv(url)
    if not content:
        return 0

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pl.read_csv(
                f,
                has_header=False,
                new_columns=_AGG_COLS,
                schema_overrides=_AGG_DTYPES,
            )

    instrument_id = f"binance:{symbol}"
    med = float(df["price"].median() or 0)

    df = (
        df
        .with_columns([
            (pl.col("transact_time") * 1_000_000).alias("ts_event_ns"),
            pl.lit(instrument_id).alias("instrument_id"),
            pl.lit("binance").alias("venue_id"),
            pl.lit(symbol).alias("native_symbol"),
            pl.col("agg_trade_id").cast(pl.Utf8).alias("trade_id"),
            pl.when(pl.col("is_buyer_maker").str.to_lowercase() == "true")
              .then(pl.lit("sell"))
              .otherwise(pl.lit("buy"))
              .alias("side"),
            pl.col("quantity").alias("size"),
            pl.lit(None).cast(pl.Float64).alias("notional_usd"),
            pl.lit("binance:aggTrades").alias("raw_source"),
            pl.lit("").alias("payload_hash"),
            pl.lit("binance_vision_archive").alias("ingest_batch_id"),
            (
                ((pl.col("price") - med).abs() / med > 0.05)
                if med > 0
                else pl.lit(False)
            ).alias("is_outlier_price"),
        ])
        .with_columns(pl.col("ts_event_ns").alias("ts_receive_ns"))
        .with_columns((pl.col("ts_event_ns") // 3_600_000_000_000).alias("_hour_idx"))
    )

    total = 0
    for (hour_idx,), grp in df.group_by(["_hour_idx"]):
        hour = int(hour_idx) % 24
        out = grp.select(_TRADE_COLS)
        _write_silver(silver_root, "binance", "aggTrades", symbol, date_str, hour, out, overwrite)
        total += len(out)

    logger.info("  [binance aggTrades] %s %s → %d rows", symbol, date_str, total)
    return total


# ---------------------------------------------------------------------------
# Binance klines → Silver books (synthetic BBO + depth levels)
# ---------------------------------------------------------------------------

_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trade_count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
_KLINE_DTYPES = {c: pl.Float64 for c in _KLINE_COLS}
_KLINE_DTYPES["open_time"] = pl.Int64
_KLINE_DTYPES["close_time"] = pl.Int64
_KLINE_DTYPES["trade_count"] = pl.Int64


def _kline_to_book_records(
    row: dict,
    instrument_id: str,
    venue_id: str,
    symbol: str,
) -> list[dict]:
    """Convert one 1-minute kline to synthetic book level records.

    Spread is estimated from the minute's high-low range.  Depth levels use
    exponentially decreasing sizes centred on close price.  This is a
    kline-based approximation — not real order-book data.
    """
    ts_ns = int(row["open_time"]) * 1_000_000
    mid = float(row["close"])
    if mid <= 0:
        return []

    hl_range = float(row["high"]) - float(row["low"])
    half_spread = max(hl_range / 2, mid * 0.00005)  # floor at 0.5bps

    taker_buy = max(float(row["taker_buy_base"]), 1e-9)
    taker_sell = max(float(row["volume"]) - taker_buy, 1e-9)

    records: list[dict] = []
    size_schedule = [0.30, 0.25, 0.20, 0.15, 0.10]  # sums to 1.0
    spread_schedule = [1.0, 2.0, 3.5, 6.0, 10.0]    # multiples of half_spread

    for i in range(_BOOK_LEVELS):
        bid_price = mid - spread_schedule[i] * half_spread
        ask_price = mid + spread_schedule[i] * half_spread
        bid_size = taker_sell * size_schedule[i]
        ask_size = taker_buy * size_schedule[i]

        base = dict(
            ts_event_ns=ts_ns,
            ts_receive_ns=ts_ns,
            venue_id=venue_id,
            instrument_id=instrument_id,
            native_symbol=symbol,
            level=i,
            checksum=None,
            raw_source=f"{venue_id}:klines",
            payload_hash="",
            is_crossed_book=False,
            is_negative_size=False,
            is_sequence_gap=False,
            is_checksum_failed=False,
            is_stale_quote=False,
            is_resync_period=False,
        )
        records.append({**base, "side": "bid", "price": bid_price, "size": bid_size})
        records.append({**base, "side": "ask", "price": ask_price, "size": ask_size})

    return records


def ingest_binance_klines(
    symbol: str,
    date_str: str,
    silver_root: Path,
    overwrite: bool,
) -> int:
    url = f"{_BINANCE_BASE}/klines/{symbol}/1m/{symbol}-1m-{date_str}.zip"
    content = _get_zip_csv(url)
    if not content:
        return 0

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pl.read_csv(
                f,
                has_header=False,
                new_columns=_KLINE_COLS,
                schema_overrides=_KLINE_DTYPES,
            )

    instrument_id = f"binance:{symbol}"
    all_records: list[dict] = []
    for row in df.iter_rows(named=True):
        all_records.extend(_kline_to_book_records(row, instrument_id, "binance", symbol))

    if not all_records:
        return 0

    book_df = (
        pl.DataFrame(all_records)
        .with_columns((pl.col("ts_event_ns") // 3_600_000_000_000).alias("_hour_idx"))
    )

    total = 0
    for (hour_idx,), grp in book_df.group_by(["_hour_idx"]):
        hour = int(hour_idx) % 24
        out = grp.select(_BOOK_COLS)
        _write_silver(silver_root, "binance", "depth", symbol, date_str, hour, out, overwrite)
        total += len(out)

    logger.info("  [binance klines→book] %s %s → %d records", symbol, date_str, total)
    return total


# ---------------------------------------------------------------------------
# Binance USDM futures bookDepth → Silver books (real ~30s depth snapshots)
# ---------------------------------------------------------------------------

_FUTURES_BOOKDEPTH_BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"

# Futures USDM symbols with bookDepth archive — all three route legs.
# Archive coverage (Binance Vision public):
#   BTCUSDT:  2023-01-01 → present  (sell leg, all benchmark windows)
#   USDCUSDT: 2023-03-12 → present  (cross leg, SVB test days 3-10 + all 2024 windows)
#   BTCUSDC:  2024-01-04 → present  (buy leg, 2024 calm-control only; perp didn't exist in 2023)
_FUTURES_BOOKDEPTH_SYMBOLS = ["BTCUSDT", "USDCUSDT", "BTCUSDC"]


def ingest_binance_futures_bookdepth(
    symbol: str,
    date_str: str,
    silver_root: Path,
    overwrite: bool,
) -> int:
    """Download Binance USDM futures bookDepth and write to Silver channel=depth.

    Snapshot format: cumulative depth at ±1–5% from mid, every ~30 seconds.
    Converted to incremental level depth with band-average prices via
    price = cumulative_notional / cumulative_depth at each level.

    Writes to the same channel=depth path as kline-derived books, so real
    data takes precedence when overwrite=True or files don't exist.
    """
    url = f"{_FUTURES_BOOKDEPTH_BASE}/{symbol}/{symbol}-bookDepth-{date_str}.zip"
    content = _get_zip_csv(url)
    if not content:
        return 0

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pl.read_csv(f)

    # Columns: timestamp (str "YYYY-MM-DD HH:MM:SS"), percentage (int), depth (float), notional (float)
    # Convert timestamp to nanoseconds (timestamps are UTC)
    raw = raw.with_columns(
        (
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="us")
            .cast(pl.Int64)
            * 1000  # microseconds → nanoseconds
        ).alias("ts_event_ns")
    )

    # Tag bid/ask sides and compute absolute percentage level
    raw = raw.with_columns([
        pl.when(pl.col("percentage") < 0)
          .then(pl.lit("bid"))
          .otherwise(pl.lit("ask"))
          .alias("side"),
        pl.col("percentage").abs().alias("abs_pct"),
    ])

    # Sort so shift() computes incremental depth correctly within each (ts, side) group
    raw = raw.sort(["ts_event_ns", "side", "abs_pct"])

    # Cumulative → incremental depth and notional via shift within group
    raw = raw.with_columns([
        pl.col("depth").shift(1).over(["ts_event_ns", "side"]).alias("prev_depth"),
        pl.col("notional").shift(1).over(["ts_event_ns", "side"]).alias("prev_notional"),
    ]).with_columns([
        pl.when(pl.col("abs_pct") == 1)
          .then(pl.col("depth"))
          .otherwise(pl.col("depth") - pl.col("prev_depth"))
          .alias("incr_depth"),
        pl.when(pl.col("abs_pct") == 1)
          .then(pl.col("notional"))
          .otherwise(pl.col("notional") - pl.col("prev_notional"))
          .alias("incr_notional"),
    ])

    # Band-average price = total notional in band / BTC depth in band
    raw = raw.with_columns(
        pl.when(pl.col("incr_depth") > 0)
          .then(pl.col("incr_notional") / pl.col("incr_depth"))
          .otherwise(pl.lit(None).cast(pl.Float64))
          .alias("price")
    ).filter(pl.col("incr_depth") > 0)

    # level index: abs_pct 1→0, 2→1, 3→2, 4→3, 5→4 (matches kline convention)
    instrument_id = f"binance:{symbol}"
    raw = raw.with_columns([
        (pl.col("abs_pct") - 1).cast(pl.Int64).alias("level"),
        pl.col("ts_event_ns").alias("ts_receive_ns"),
        pl.lit("binance").alias("venue_id"),
        pl.lit(instrument_id).alias("instrument_id"),
        pl.lit(symbol).alias("native_symbol"),
        pl.col("incr_depth").alias("size"),
        pl.lit(None).alias("checksum"),
        pl.lit("binance:futures_bookdepth").alias("raw_source"),
        pl.lit("").alias("payload_hash"),
        pl.lit(False).alias("is_crossed_book"),
        pl.lit(False).alias("is_negative_size"),
        pl.lit(False).alias("is_sequence_gap"),
        pl.lit(False).alias("is_checksum_failed"),
        pl.lit(False).alias("is_stale_quote"),
        pl.lit(False).alias("is_resync_period"),
    ]).with_columns(
        (pl.col("ts_event_ns") // 3_600_000_000_000).alias("_hour_idx")
    )

    total = 0
    for (hour_idx,), grp in raw.group_by(["_hour_idx"]):
        hour = int(hour_idx) % 24
        out = grp.select(_BOOK_COLS)
        _write_silver(silver_root, "binance", "depth", symbol, date_str, hour, out, overwrite)
        total += len(out)

    logger.info("  [binance futures bookdepth] %s %s → %d records", symbol, date_str, total)
    return total


# ---------------------------------------------------------------------------
# Coinbase REST candles → Silver trades + books
# ---------------------------------------------------------------------------

def _fetch_coinbase_candles(
    product_id: str,
    start_dt: datetime,
    end_dt: datetime,
) -> list[list]:
    """Fetch 1-minute candles from Coinbase REST API (300 per call, paginated)."""
    all_candles: list[list] = []
    cur = start_dt
    step = timedelta(minutes=299)  # 300 candles per request

    while cur < end_dt:
        batch_end = min(cur + step, end_dt)
        url = (
            f"{_COINBASE_BASE}/products/{product_id}/candles"
            f"?start={cur.isoformat()}&end={batch_end.isoformat()}&granularity=60"
        )
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    all_candles.extend(data)
            elif r.status_code == 429:
                time.sleep(2)
                continue
            else:
                logger.warning("Coinbase %s: HTTP %d", url, r.status_code)
        except Exception as exc:
            logger.warning("Coinbase fetch error: %s", exc)

        cur = batch_end + timedelta(minutes=1)
        time.sleep(0.35)  # respect rate limit (3 req/s)

    return all_candles


def ingest_coinbase_candles(
    product_id: str,
    date_str: str,
    silver_root: Path,
    overwrite: bool,
) -> int:
    """Fetch one day of 1-minute candles and write Silver trades + books."""
    d = date.fromisoformat(date_str)
    start_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=1)

    candles = _fetch_coinbase_candles(product_id, start_dt, end_dt)
    if not candles:
        logger.info("  [coinbase] no candles: %s %s", product_id, date_str)
        return 0

    # Coinbase candles: [timestamp, open, high, low, close, volume] (reversed)
    rows = sorted(candles, key=lambda x: x[0])
    instrument_id = f"coinbase:{product_id}"
    symbol = product_id.replace("-", "")

    trade_records: list[dict] = []
    book_records: list[dict] = []

    for ts, low_p, high_p, open_p, close_p, vol in rows:  # Coinbase: [time, low, high, open, close, vol]
        ts_ns = int(ts) * 1_000_000_000
        mid = float(close_p)
        if mid <= 0:
            continue
        hl_range = float(high_p) - float(low_p)
        half_spread = max(hl_range / 2, mid * 0.00005)
        vol_f = max(float(vol), 1e-9)

        trade_records.append(dict(
            ts_event_ns=ts_ns,
            ts_receive_ns=ts_ns,
            venue_id="coinbase",
            instrument_id=instrument_id,
            native_symbol=product_id,
            trade_id=str(ts),
            side="buy",
            price=mid,
            size=vol_f / 2,
            notional_usd=None,
            raw_source="coinbase:candles",
            payload_hash="",
            ingest_batch_id="coinbase_api",
            is_outlier_price=False,
        ))

        size_schedule = [0.30, 0.25, 0.20, 0.15, 0.10]
        spread_schedule = [1.0, 2.0, 3.5, 6.0, 10.0]
        base = dict(
            ts_event_ns=ts_ns, ts_receive_ns=ts_ns,
            venue_id="coinbase", instrument_id=instrument_id,
            native_symbol=product_id, checksum=None,
            raw_source="coinbase:candles", payload_hash="",
            is_crossed_book=False, is_negative_size=False,
            is_sequence_gap=False, is_checksum_failed=False,
            is_stale_quote=False, is_resync_period=False,
        )
        for i in range(_BOOK_LEVELS):
            book_records.append({
                **base, "side": "bid", "level": i,
                "price": mid - spread_schedule[i] * half_spread,
                "size": vol_f * size_schedule[i] / 2,
            })
            book_records.append({
                **base, "side": "ask", "level": i,
                "price": mid + spread_schedule[i] * half_spread,
                "size": vol_f * size_schedule[i] / 2,
            })

    total = 0
    if trade_records:
        tdf = (
            pl.DataFrame(trade_records)
            .with_columns((pl.col("ts_event_ns") // 3_600_000_000_000).alias("_h"))
        )
        for (h,), grp in tdf.group_by(["_h"]):
            out = grp.select(_TRADE_COLS)
            _write_silver(
                silver_root, "coinbase", "matches", symbol, date_str, int(h) % 24, out, overwrite
            )
            total += len(out)

    if book_records:
        bdf = (
            pl.DataFrame(book_records)
            .with_columns((pl.col("ts_event_ns") // 3_600_000_000_000).alias("_h"))
        )
        for (h,), grp in bdf.group_by(["_h"]):
            out = grp.select(_BOOK_COLS)
            _write_silver(
                silver_root, "coinbase", "level2", symbol, date_str, int(h) % 24, out, overwrite
            )

    logger.info("  [coinbase candles] %s %s → %d trade rows", product_id, date_str, total)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch real data from Binance Vision + Coinbase REST into Silver layer."
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        default=list(_ALL_WINDOWS.keys()),
        choices=list(_ALL_WINDOWS.keys()),
        help="Event windows to fetch (default: all).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=_BINANCE_SYMBOLS,
        help="Binance symbols to download (default: all configured).",
    )
    parser.add_argument(
        "--coinbase-products",
        nargs="*",
        default=_COINBASE_PRODUCTS,
        help="Coinbase products to fetch (default: BTC-USD USDT-USD).",
    )
    parser.add_argument(
        "--silver-dir",
        default="data/silver",
        help="Silver output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite existing Silver files.",
    )
    parser.add_argument(
        "--skip-coinbase",
        action="store_true",
        help="Skip Coinbase API fetching.",
    )
    parser.add_argument(
        "--futures-bookdepth",
        action="store_true",
        help=(
            "Download real Binance USDM futures bookDepth snapshots for "
            f"{_FUTURES_BOOKDEPTH_SYMBOLS} and write to channel=depth "
            "(replaces/augments kline-derived synthetic book for those symbols)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    silver_root = Path(args.silver_dir)
    silver_root.mkdir(parents=True, exist_ok=True)

    # Collect all dates across selected windows
    all_dates: set[str] = set()
    for w in args.windows:
        start, end = _ALL_WINDOWS[w]
        for d in _date_range(start, end):
            all_dates.add(d)

    logger.info(
        "Fetching %d date(s) across %d window(s): %s",
        len(all_dates), len(args.windows), sorted(args.windows),
    )
    logger.info("Binance symbols: %s", args.symbols)
    if not args.skip_coinbase:
        logger.info("Coinbase products: %s", args.coinbase_products)

    if args.dry_run:
        logger.info("[DRY RUN] Would fetch %d × %d Binance files + Coinbase candles.",
                    len(sorted(all_dates)), len(args.symbols))
        if getattr(args, "futures_bookdepth", False):
            logger.info("[DRY RUN] Would also fetch futures bookDepth for %s.",
                        _FUTURES_BOOKDEPTH_SYMBOLS)
        return

    total_rows = 0

    # Binance: one date at a time to limit memory
    for date_str in sorted(all_dates):
        for symbol in args.symbols:
            total_rows += ingest_binance_aggtrades(
                symbol, date_str, silver_root, args.overwrite
            )
            total_rows += ingest_binance_klines(
                symbol, date_str, silver_root, args.overwrite
            )

    # Binance USDM futures bookDepth: real depth snapshots for liquid symbols
    if getattr(args, "futures_bookdepth", False):
        logger.info("Fetching Binance USDM futures bookDepth for %s", _FUTURES_BOOKDEPTH_SYMBOLS)
        for date_str in sorted(all_dates):
            for symbol in _FUTURES_BOOKDEPTH_SYMBOLS:
                total_rows += ingest_binance_futures_bookdepth(
                    symbol, date_str, silver_root, overwrite=True  # always overwrite kline-derived
                )

    # Coinbase: paginated REST candles
    if not args.skip_coinbase:
        for date_str in sorted(all_dates):
            for product in args.coinbase_products:
                total_rows += ingest_coinbase_candles(
                    product, date_str, silver_root, args.overwrite
                )

    # Summary
    n_parquet = sum(1 for _ in silver_root.glob("**/*.parquet"))
    logger.info(
        "Fetch complete: %d total rows written to %d Silver parquet files.",
        total_rows, n_parquet,
    )


if __name__ == "__main__":
    main()
