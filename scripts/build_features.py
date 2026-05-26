#!/usr/bin/env python3
"""Build Silver and Gold feature tables from Bronze raw data.

Pipeline:
    Bronze (raw JSON/Parquet) →
    Silver (normalised trades, books, on-chain) →
    Gold (microstructure features, basis, fragmentation, settlement, labels)

Usage:
    python scripts/build_features.py --start 2024-01-01 --end 2024-01-07
    python scripts/build_features.py --start 2024-01-01 --end 2024-01-07 --skip-silver
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import polars as pl

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# Notional sizes (USD) used for net-profit and arbitrage-window labels
_NOTIONAL_SIZES_USD = [10_000, 50_000, 100_000, 500_000]

# Map (venue, channel) → short key used in _get_normalizer
_CHANNEL_MAP: dict[tuple[str, str], str] = {
    # Live WebSocket channels
    ("binance", "trade"): "binance_trades",
    ("binance", "depth"): "binance_depth",
    ("coinbase", "matches"): "coinbase_trades",
    ("coinbase", "level2"): "coinbase_level2",
    ("kraken", "trade"): "kraken_trades",
    ("kraken", "book"): "kraken_book",
    ("uniswap_v3", "swap"): "uniswap_swaps",
    # Archive / historical channels (written by archive_to_bronze.py)
    ("binance", "aggTrades"): "binance_trades",   # aggTrade WS-format payload
    ("binance", "klines"): "binance_klines",       # kline WS-format payload → synthetic book
    # Tardis archive channels — source-identity preserved, routed to Tardis normalizers
    # (Tardis CSV payload shape differs from live WS message shape)
    ("coinbase", "tardis_trades"): "tardis_trades",
    ("coinbase", "tardis_book_snapshot_1s"): "tardis_book_snapshot",
    ("coinbase", "tardis_incremental_book_l2"): "tardis_incremental_book",
    ("kraken", "tardis_trades"): "tardis_trades",
    ("kraken", "tardis_book_snapshot_1s"): "tardis_book_snapshot",
    ("kraken", "tardis_incremental_book_l2"): "tardis_incremental_book",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature tables from Bronze data.")
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
    )
    parser.add_argument("--bronze-dir", default="data/bronze")
    parser.add_argument("--silver-dir", default="data/silver")
    parser.add_argument("--gold-dir", default="data/gold")
    parser.add_argument("--skip-silver", action="store_true",
                        help="Skip Silver normalisation (use existing Silver data).")
    parser.add_argument("--skip-labels", action="store_true",
                        help="Skip label generation.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _date_range(start: datetime, end: datetime) -> list[str]:
    days: list[str] = []
    cur = start.date()
    while cur < end.date():
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def _get_normalizer(venue: str, channel: str) -> Callable | None:
    key = _CHANNEL_MAP.get((venue, channel))
    if key == "binance_trades":
        from stressbench.normalization.normalize_trades import normalize_binance_trades
        return normalize_binance_trades
    if key == "binance_depth":
        from stressbench.normalization.normalize_books import normalize_binance_depth
        return normalize_binance_depth
    if key == "binance_klines":
        from stressbench.normalization.normalize_books import normalize_binance_klines
        return normalize_binance_klines
    if key == "tardis_trades":
        from stressbench.normalization.normalize_tardis import normalize_tardis_trades
        return normalize_tardis_trades
    if key == "tardis_book_snapshot":
        from stressbench.normalization.normalize_tardis import normalize_tardis_book_snapshot_1s
        return normalize_tardis_book_snapshot_1s
    if key == "tardis_incremental_book":
        from stressbench.normalization.normalize_tardis import normalize_tardis_incremental_book_l2
        return normalize_tardis_incremental_book_l2
    if key == "coinbase_trades":
        from stressbench.normalization.normalize_trades import normalize_coinbase_trades
        return normalize_coinbase_trades
    if key == "coinbase_level2":
        from stressbench.normalization.normalize_books import normalize_coinbase_level2
        return normalize_coinbase_level2
    if key == "kraken_trades":
        from stressbench.normalization.normalize_trades import normalize_kraken_trades
        return normalize_kraken_trades
    if key == "kraken_book":
        from stressbench.normalization.normalize_books import normalize_kraken_book
        return normalize_kraken_book
    if key == "uniswap_swaps":
        from stressbench.normalization.normalize_onchain import normalize_uniswap_swaps
        return normalize_uniswap_swaps
    return None


def _ts_to_ns(iso_str: str) -> int:
    """Parse an ISO-8601 timestamp string to nanoseconds since epoch."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1_000_000_000)


def _write_gold(gold_root: Path, table: str, date_str: str, df: pl.DataFrame) -> None:
    if df is None or df.is_empty():
        return
    out = gold_root / table / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "part-0.parquet")
    logger.debug("Wrote %s/%s: %d rows", table, date_str, len(df))


# ---------------------------------------------------------------------------
# Silver build
# ---------------------------------------------------------------------------

def build_silver(bronze_dir: str, silver_dir: str, start: datetime, end: datetime, dry_run: bool) -> None:
    """Normalise Bronze raw data into Silver Parquet files."""
    logger.info("Building Silver layer: %s → %s", start.date(), end.date())
    if dry_run:
        logger.info("[DRY RUN] Skipping Silver build.")
        return

    from stressbench.normalization.normalize_onchain import normalize_etherscan_transfers

    bronze_root = Path(bronze_dir)
    silver_root = Path(silver_dir)
    dates = set(_date_range(start, end))
    total_read = total_written = 0

    for venue_dir in sorted(bronze_root.glob("venue=*")):
        venue = venue_dir.name.split("=", 1)[1]
        for channel_dir in sorted(venue_dir.glob("channel=*")):
            channel = channel_dir.name.split("=", 1)[1]
            is_etherscan = (venue == "etherscan" and channel == "transfer")
            normalizer = None if is_etherscan else _get_normalizer(venue, channel)
            if normalizer is None and not is_etherscan:
                logger.debug("No normalizer for %s/%s; skipping.", venue, channel)
                continue

            for symbol_dir in sorted(channel_dir.glob("symbol=*")):
                symbol = symbol_dir.name.split("=", 1)[1]
                for date_dir in sorted(symbol_dir.glob("date=*")):
                    date_str = date_dir.name.split("=", 1)[1]
                    if date_str not in dates:
                        continue
                    for hour_dir in sorted(date_dir.glob("hour=*")):
                        hour = hour_dir.name.split("=", 1)[1]
                        files = sorted(hour_dir.glob("*.parquet"))
                        if not files:
                            continue
                        total_read += len(files)
                        try:
                            df = pl.concat([pl.read_parquet(f) for f in files])
                        except Exception as exc:
                            logger.warning("Read error %s: %s", hour_dir, exc)
                            continue
                        try:
                            normalized = (
                                normalize_etherscan_transfers(df, token_symbol=symbol)
                                if is_etherscan
                                else normalizer(df)
                            )
                        except Exception as exc:
                            logger.warning(
                                "Normalize error %s/%s/%s: %s", venue, channel, symbol, exc
                            )
                            continue
                        if normalized.is_empty():
                            continue
                        out = (
                            silver_root
                            / f"venue={venue}"
                            / f"channel={channel}"
                            / f"symbol={symbol}"
                            / f"date={date_str}"
                            / f"hour={hour}"
                        )
                        out.mkdir(parents=True, exist_ok=True)
                        normalized.write_parquet(out / "part-0.parquet")
                        total_written += 1

    logger.info(
        "Silver build complete: %d Bronze files → %d Silver partitions.",
        total_read, total_written,
    )


# ---------------------------------------------------------------------------
# Gold helpers — called once per date
# ---------------------------------------------------------------------------

def _load_silver_channels(
    silver_root: Path,
    channels: list[str],
    dates: set[str],
    depth_source: str | None = None,
) -> pl.DataFrame:
    """Load and concatenate Silver Parquet files matching given channels and dates.

    Args:
        silver_root: Root of the Silver layer.
        channels: Channel names to scan.
        dates: ISO date strings to include.
        depth_source: Fallback ``depth_source`` value to add when the column is
            absent in a loaded frame (or to fill nulls when it is present).
            Use granular values like ``"real_l2_snapshot"``,
            ``"real_l2_incremental"``, or ``"synthetic_kline"`` to preserve
            provenance from Silver normalizers.  A frame that already carries a
            ``depth_source`` column (written by the normalizer) is not
            overwritten — only null entries are filled.
    """
    frames: list[pl.DataFrame] = []
    for channel in channels:
        for f in sorted(silver_root.glob(f"venue=*/channel={channel}/**/date=*/hour=*/part-0.parquet")):
            date_part = next(
                (p.split("=", 1)[1] for p in f.parts if p.startswith("date=")), None
            )
            if date_part not in dates:
                continue
            try:
                frame = pl.read_parquet(f)
                if depth_source is not None and not frame.is_empty():
                    if "depth_source" not in frame.columns:
                        frame = frame.with_columns(pl.lit(depth_source).alias("depth_source"))
                    else:
                        frame = frame.with_columns(
                            pl.col("depth_source").fill_null(depth_source).alias("depth_source")
                        )
                frames.append(frame)
            except Exception as exc:
                logger.warning("Skip %s: %s", f, exc)
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


def _compute_book_1m(books_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Silver book levels into 1-minute microstructure features.

    Uses the last book snapshot within each minute window per
    (venue_id, instrument_id) to avoid stale-state bias.
    """
    if books_df.is_empty():
        return pl.DataFrame()

    df = books_df.with_columns(
        ((pl.col("ts_event_ns") // 60_000_000_000) * 60_000_000_000).alias("ts_1m_ns")
    )

    # Identify the last tick timestamp per (minute, venue, instrument)
    last_ts = df.group_by(["ts_1m_ns", "venue_id", "instrument_id"]).agg(
        pl.col("ts_event_ns").max().alias("_ref_ts")
    )
    snap = (
        df.join(last_ts, on=["ts_1m_ns", "venue_id", "instrument_id"], how="left")
        .filter(pl.col("ts_event_ns") == pl.col("_ref_ts"))
        .drop("_ref_ts")
    )

    # Level-0 best bid / ask
    _has_depth_source = "depth_source" in snap.columns
    bids0_cols = [
        "ts_1m_ns", "venue_id", "instrument_id",
        pl.col("price").alias("best_bid"),
        pl.col("size").alias("_bid0_sz"),
        pl.col("is_checksum_failed").cast(pl.Boolean).fill_null(False).alias("_chk"),
        pl.col("is_resync_period").cast(pl.Boolean).fill_null(False).alias("_resync"),
    ]
    if _has_depth_source:
        bids0_cols.append(pl.col("depth_source").first().alias("depth_source"))
    bids0 = snap.filter(
        (pl.col("side") == "bid") & (pl.col("level") == 0)
    ).select(bids0_cols)
    asks0 = snap.filter(
        (pl.col("side") == "ask") & (pl.col("level") == 0)
    ).select([
        "ts_1m_ns", "venue_id", "instrument_id",
        pl.col("price").alias("best_ask"),
        pl.col("size").alias("_ask0_sz"),
    ])
    bbo = bids0.join(asks0, on=["ts_1m_ns", "venue_id", "instrument_id"], how="inner")

    # Depth within 10 bps of mid
    snap_aug = snap.join(
        bbo.select(["ts_1m_ns", "venue_id", "instrument_id", "best_bid", "best_ask"]),
        on=["ts_1m_ns", "venue_id", "instrument_id"],
        how="inner",
    )
    bid_depth = (
        snap_aug.filter(
            (pl.col("side") == "bid") & (pl.col("price") >= pl.col("best_bid") * 0.999)
        ).group_by(["ts_1m_ns", "venue_id", "instrument_id"])
        .agg(pl.col("size").sum().alias("depth_bid_10bp"))
    )
    ask_depth = (
        snap_aug.filter(
            (pl.col("side") == "ask") & (pl.col("price") <= pl.col("best_ask") * 1.001)
        ).group_by(["ts_1m_ns", "venue_id", "instrument_id"])
        .agg(pl.col("size").sum().alias("depth_ask_10bp"))
    )

    result = (
        bbo
        .join(bid_depth, on=["ts_1m_ns", "venue_id", "instrument_id"], how="left")
        .join(ask_depth, on=["ts_1m_ns", "venue_id", "instrument_id"], how="left")
    )

    mid_expr = (pl.col("best_bid") + pl.col("best_ask")) / 2.0
    result = result.with_columns([
        mid_expr.alias("mid"),
        ((pl.col("best_ask") - pl.col("best_bid")) / mid_expr * 10_000.0).alias("spread_bps"),
        (
            (pl.col("_bid0_sz") - pl.col("_ask0_sz"))
            / (pl.col("_bid0_sz") + pl.col("_ask0_sz") + 1e-10)
        ).alias("imbalance_1bp"),
        (
            pl.lit(1.0)
            - pl.col("_chk").cast(pl.Float64) * 0.3
            - pl.col("_resync").cast(pl.Float64) * 0.5
        ).clip(0.0, 1.0).alias("data_quality_score"),
    ]).drop(["_bid0_sz", "_ask0_sz", "_chk", "_resync"])

    return result


def _compute_trade_1m(trades_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Silver trades into 1-minute trade statistics."""
    if trades_df.is_empty():
        return pl.DataFrame()

    df = (
        trades_df
        .filter(~pl.col("is_outlier_price").cast(pl.Boolean).fill_null(False))
        .with_columns(
            ((pl.col("ts_event_ns") // 60_000_000_000) * 60_000_000_000).alias("ts_1m_ns")
        )
    )
    return df.group_by(["ts_1m_ns", "venue_id", "instrument_id"]).agg([
        pl.len().alias("trade_count_1m"),
        pl.col("size").sum().alias("trade_volume_1m"),
    ])


def _annotate_instruments(df: pl.DataFrame, instruments: list[dict]) -> pl.DataFrame:
    meta = pl.DataFrame([
        {
            "instrument_id": i["instrument_id"],
            "base_asset": i["base_asset"],
            "quote_asset": i["quote_asset"],
        }
        for i in instruments
    ])
    return df.join(meta, on="instrument_id", how="left")


def _compute_stablecoin_prices(book_1m: pl.DataFrame, instruments: list[dict]) -> pl.DataFrame:
    """Return per-venue stablecoin USD prices and deviations from $1."""
    df = _annotate_instruments(book_1m, instruments)

    usdt_ref = (
        df.filter((pl.col("base_asset") == "USDT") & (pl.col("quote_asset") == "USD"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("usdt_usd_ref"))
    )
    usdc_ref = (
        df.filter((pl.col("base_asset") == "USDC") & (pl.col("quote_asset") == "USD"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("usdc_usd_ref"))
    )
    df = df.join(usdt_ref, on="ts_1m_ns", how="left").join(usdc_ref, on="ts_1m_ns", how="left")

    df = df.with_columns(
        pl.when(pl.col("quote_asset") == "USD")
        .then(pl.col("mid"))
        .when(pl.col("quote_asset") == "USDT")
        .then(pl.col("mid") * pl.col("usdt_usd_ref"))
        .when(pl.col("quote_asset") == "USDC")
        .then(pl.col("mid") * pl.col("usdc_usd_ref"))
        .otherwise(None)
        .alias("price_usd")
    )

    sc = df.filter(
        pl.col("base_asset").is_in(["USDC", "USDT", "DAI"])
        & pl.col("price_usd").is_not_null()
    )
    return sc.select([
        "ts_1m_ns", "venue_id", "instrument_id",
        pl.col("base_asset").alias("stablecoin"),
        "price_usd",
        ((pl.col("price_usd") - 1.0) * 10_000.0).alias("deviation_from_1_usd_bps"),
    ])


def _compute_basis_features(book_1m: pl.DataFrame, instruments: list[dict]) -> pl.DataFrame:
    """Compute cross-quote BTC/ETH basis per minute (the benchmark's core feature)."""
    df = _annotate_instruments(book_1m, instruments)

    usdt_ref = (
        df.filter((pl.col("base_asset") == "USDT") & (pl.col("quote_asset") == "USD"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("usdt_usd_ref"))
    )
    usdc_ref = (
        df.filter((pl.col("base_asset") == "USDC") & (pl.col("quote_asset") == "USD"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("usdc_usd_ref"))
    )

    # Fallback: derive USDC-USD from USDC-USDT × USDT-USD when no direct pair exists.
    # This is the implied price a trader would use when no USDC/USD orderbook is available.
    usdc_usdt = (
        df.filter((pl.col("base_asset") == "USDC") & (pl.col("quote_asset") == "USDT"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("_usdc_usdt"))
    )
    usdc_ref_implied = (
        usdc_usdt.join(usdt_ref, on="ts_1m_ns", how="inner")
        .with_columns((pl.col("_usdc_usdt") * pl.col("usdt_usd_ref")).alias("usdc_usd_ref"))
        .select(["ts_1m_ns", "usdc_usd_ref"])
    )
    if usdc_ref.is_empty():
        usdc_ref = usdc_ref_implied
    elif not usdc_ref_implied.is_empty():
        usdc_ref = (
            usdc_ref
            .join(usdc_ref_implied, on="ts_1m_ns", how="full", coalesce=True, suffix="_imp")
            .with_columns(
                pl.col("usdc_usd_ref").fill_null(pl.col("usdc_usd_ref_imp"))
            )
            .select(["ts_1m_ns", "usdc_usd_ref"])
        )

    df = df.join(usdt_ref, on="ts_1m_ns", how="left").join(usdc_ref, on="ts_1m_ns", how="left")

    btc_direct = (
        df.filter((pl.col("base_asset") == "BTC") & (pl.col("quote_asset") == "USD"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("btc_usd_direct"))
    )

    # Primary: BTCUSDC × USDC-USD. Fallback: BTCUSDT × USDCUSDT × USDT-USD (3-leg implied).
    btc_via_usdc_direct = (
        df.filter((pl.col("base_asset") == "BTC") & (pl.col("quote_asset") == "USDC"))
        .with_columns((pl.col("mid") * pl.col("usdc_usd_ref")).alias("_imp"))
        .group_by("ts_1m_ns").agg(pl.col("_imp").mean().alias("btc_usd_via_usdc"))
    )
    # Implied: BTCUSDT × USDCUSDT × USDT-USD (3-leg route — valid when BTCUSDC not available)
    btc_usdt_mid = (
        df.filter((pl.col("base_asset") == "BTC") & (pl.col("quote_asset") == "USDT"))
        .group_by("ts_1m_ns").agg(pl.col("mid").mean().alias("_btc_usdt_mid"))
    )
    btc_via_usdc_implied = (
        btc_usdt_mid
        .join(usdc_usdt, on="ts_1m_ns", how="inner")
        .join(usdt_ref, on="ts_1m_ns", how="inner")
        .with_columns(
            (pl.col("_btc_usdt_mid") * pl.col("_usdc_usdt") * pl.col("usdt_usd_ref")).alias("btc_usd_via_usdc")
        )
        .select(["ts_1m_ns", "btc_usd_via_usdc"])
    )
    if btc_via_usdc_direct.is_empty():
        btc_via_usdc = btc_via_usdc_implied
    elif not btc_via_usdc_implied.is_empty():
        btc_via_usdc = (
            btc_via_usdc_direct
            .join(btc_via_usdc_implied, on="ts_1m_ns", how="full", coalesce=True, suffix="_imp")
            .with_columns(
                pl.col("btc_usd_via_usdc").fill_null(pl.col("btc_usd_via_usdc_imp"))
            )
            .select(["ts_1m_ns", "btc_usd_via_usdc"])
        )
    else:
        btc_via_usdc = btc_via_usdc_direct

    btc_via_usdt = (
        df.filter((pl.col("base_asset") == "BTC") & (pl.col("quote_asset") == "USDT"))
        .with_columns((pl.col("mid") * pl.col("usdt_usd_ref")).alias("_imp"))
        .group_by("ts_1m_ns").agg(pl.col("_imp").mean().alias("btc_usd_via_usdt"))
    )

    basis = (
        btc_direct
        .join(btc_via_usdc, on="ts_1m_ns", how="full", coalesce=True)
        .join(btc_via_usdt, on="ts_1m_ns", how="full", coalesce=True)
    )

    d = pl.col("btc_usd_direct")
    safe = d.is_not_null() & (d > 0)
    basis = basis.with_columns([
        pl.when(safe)
        .then(10_000.0 * (pl.col("btc_usd_via_usdc") - d) / d)
        .otherwise(None).alias("cross_quote_basis_usdc_bps"),
        pl.when(safe)
        .then(10_000.0 * (pl.col("btc_usd_via_usdt") - d) / d)
        .otherwise(None).alias("cross_quote_basis_usdt_bps"),
    ])

    # Max-absolute basis: generic stress detector (larger of USDC vs USDT deviation)
    basis = basis.with_columns(
        pl.when(
            pl.col("cross_quote_basis_usdc_bps").abs()
            >= pl.col("cross_quote_basis_usdt_bps").abs()
        )
        .then(pl.col("cross_quote_basis_usdc_bps"))
        .otherwise(pl.col("cross_quote_basis_usdt_bps"))
        .alias("cross_quote_basis_maxabs_bps")
    )

    # Primary basis: USDC-specific (for SVB/USDC event analysis).
    # Falls back to max-absolute when USDC basis is unavailable.
    basis = basis.with_columns([
        pl.col("cross_quote_basis_usdc_bps")
        .fill_null(pl.col("cross_quote_basis_maxabs_bps"))
        .alias("cross_quote_basis_primary_bps"),
        pl.when(pl.col("cross_quote_basis_usdc_bps").is_not_null())
        .then(pl.lit("USDC"))
        .otherwise(pl.lit("max_abs"))
        .alias("basis_primary_asset"),
    ])

    return basis


def _compute_fragmentation(book_1m: pl.DataFrame, instruments: list[dict]) -> pl.DataFrame:
    """Cross-venue fragmentation metrics per stablecoin per minute."""
    df = _annotate_instruments(book_1m, instruments)
    sc = df.filter(
        pl.col("base_asset").is_in(["USDC", "USDT", "DAI"])
        & (pl.col("quote_asset") == "USD")
    )
    if sc.is_empty():
        return pl.DataFrame()

    mid_mean = pl.col("mid").mean()
    return (
        sc.group_by(["ts_1m_ns", "base_asset"]).agg([
            pl.len().alias("num_active_venues"),
            (pl.col("mid").std() / mid_mean * 10_000.0).alias("mid_dispersion_bps"),
            ((pl.col("mid").max() - pl.col("mid").min()) / mid_mean * 10_000.0).alias("max_minus_min_bps"),
            pl.col("depth_bid_10bp").mean().alias("depth_bid_mean"),
            pl.col("depth_ask_10bp").mean().alias("depth_ask_mean"),
        ])
        .rename({"base_asset": "stablecoin"})
    )


def _compute_net_profit_1m(
    books_df: pl.DataFrame,
    instruments: list[dict],
    fee_cfg: dict,
) -> pl.DataFrame:
    """Compute executable net-profit via full VWAP order-book walk for each notional size.

    Reconstructs one OrderBook per (venue, instrument) from the last snapshot
    within each 1-minute window, then calls ``net_profit_bps()`` at each
    notional size. This is the benchmark's central claim: executable arbitrage
    is materially smaller than naive best-price comparison because large orders
    walk through the book and face market impact.
    """
    from stressbench.book.order_book import OrderBook
    from stressbench.book.vwap import net_profit_bps as vwap_net_profit

    if books_df.is_empty():
        return pl.DataFrame()

    df = _annotate_instruments(books_df, instruments)
    btc = df.filter(pl.col("base_asset") == "BTC")
    if btc.is_empty():
        return pl.DataFrame()

    btc = btc.with_columns(
        ((pl.col("ts_event_ns") // 60_000_000_000) * 60_000_000_000).alias("ts_1m_ns")
    )

    # Use the last book snapshot per (ts_1m_ns, venue_id, instrument_id)
    last_ts = btc.group_by(["ts_1m_ns", "venue_id", "instrument_id"]).agg(
        pl.col("ts_event_ns").max().alias("_ref_ts")
    )
    snap = (
        btc.join(last_ts, on=["ts_1m_ns", "venue_id", "instrument_id"], how="inner")
        .filter(pl.col("ts_event_ns") == pl.col("_ref_ts"))
        .drop("_ref_ts")
    )

    fee_sched = fee_cfg.get("fee_schedules", {})
    eth_cfg = fee_sched.get("ethereum_mainnet", {})
    gas_usd = (
        eth_cfg.get("gas_gwei_estimate", 20.0)
        * eth_cfg.get("uniswap_swap_gas", 150_000)
        / 1e9
        * 2_000.0
    )

    rows: list[dict] = []
    for ts_val in snap["ts_1m_ns"].unique().sort().to_list():
        slice_t = snap.filter(pl.col("ts_1m_ns") == ts_val)

        # Build one OrderBook per (venue_id, instrument_id) using all depth levels
        venue_books: dict[tuple[str, str], OrderBook] = {}
        for vi, ii in slice_t.select(["venue_id", "instrument_id"]).unique().rows():
            grp = slice_t.filter(
                (pl.col("venue_id") == vi) & (pl.col("instrument_id") == ii)
            )
            bid_rows = grp.filter(pl.col("side") == "bid")
            ask_rows = grp.filter(pl.col("side") == "ask")
            bids = list(zip(bid_rows["price"].to_list(), bid_rows["size"].to_list()))
            asks = list(zip(ask_rows["price"].to_list(), ask_rows["size"].to_list()))
            book = OrderBook()
            book.apply_snapshot(bids, asks)
            if book.best_bid() is not None or book.best_ask() is not None:
                venue_books[(vi, ii)] = book

        venues = list(venue_books.items())
        if len(venues) < 2:
            continue

        row: dict = {"ts_1m_ns": ts_val}
        best_buy_v: str | None = None
        best_sell_v: str | None = None
        found_any = False

        for q in _NOTIONAL_SIZES_USD:
            best_net: float | None = None

            for (buy_venue, _buy_inst), buy_book in venues:
                buy_ask = buy_book.best_ask()
                if buy_ask is None or buy_ask <= 0:
                    continue
                qty_btc = float(q) / buy_ask  # approx BTC qty at this notional
                t_buy = fee_sched.get(buy_venue, {}).get("taker_bps", 10.0)
                wfee = fee_sched.get(buy_venue, {}).get("withdrawal_fee_usdc", 1.0)

                for (sell_venue, _sell_inst), sell_book in venues:
                    if buy_venue == sell_venue:
                        continue
                    t_sell = fee_sched.get(sell_venue, {}).get("taker_bps", 10.0)
                    net = vwap_net_profit(
                        buy_book, sell_book, qty_btc,
                        taker_fee_buy_bps=t_buy,
                        taker_fee_sell_bps=t_sell,
                        withdrawal_fee_usd=wfee,
                        gas_fee_usd=gas_usd,
                        settlement_delay_penalty_bps=2.0,
                        notional_usd=float(q),
                    )
                    if net is not None and (best_net is None or net > best_net):
                        best_net = net
                        best_buy_v = buy_venue
                        best_sell_v = sell_venue

            row[f"net_profit_bps_q{q}"] = best_net if best_net is not None else float("nan")
            if best_buy_v:
                found_any = True

        if found_any:
            row["buy_venue"] = best_buy_v or ""
            row["sell_venue"] = best_sell_v or ""
            # Propagate depth quality: real_l2 if any real-L2 book was used for
            # this minute window, otherwise synthetic_kline (not paper-grade).
            if "depth_source" in snap.columns:
                ts_sources = (
                    slice_t["depth_source"].drop_nulls().unique().to_list()
                )
                row["depth_source"] = (
                    "real_l2" if "real_l2" in ts_sources else "synthetic_kline"
                )
            rows.append(row)

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _compute_settlement_1m(
    transfers_df: pl.DataFrame,
    swaps_df: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate on-chain data into 1-minute settlement proxy features."""
    from stressbench.features.settlement import compute_settlement_features_1m

    no_transfers = transfers_df is None or transfers_df.is_empty()
    no_swaps = swaps_df is None or swaps_df.is_empty()
    if no_transfers and no_swaps:
        return pl.DataFrame()

    minutes: set[int] = set()
    if not no_transfers and "ts_unix_seconds" in transfers_df.columns:
        for ts in transfers_df["ts_unix_seconds"].cast(pl.Int64).to_list():
            minutes.add((ts // 60) * 60 * 1_000_000_000)

    rows: list[dict] = []
    for ts_1m_ns in sorted(minutes):
        s0 = ts_1m_ns // 1_000_000_000
        s1 = s0 + 60

        t_slice = (
            pl.DataFrame() if no_transfers
            else transfers_df.filter(
                (pl.col("ts_unix_seconds").cast(pl.Int64) >= s0)
                & (pl.col("ts_unix_seconds").cast(pl.Int64) < s1)
            )
        )
        s_slice = (
            pl.DataFrame() if no_swaps
            else swaps_df.filter(
                (pl.col("ts_unix_seconds").cast(pl.Int64) >= s0)
                & (pl.col("ts_unix_seconds").cast(pl.Int64) < s1)
            )
        )
        rows.append(compute_settlement_features_1m(t_slice, s_slice, ts_1m_ns=ts_1m_ns))

    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Gold build (public)
# ---------------------------------------------------------------------------

def build_gold_features(
    silver_dir: str,
    gold_dir: str,
    start: datetime,
    end: datetime,
    dry_run: bool,
) -> None:
    """Build Gold microstructure, basis, fragmentation, and settlement tables."""
    logger.info("Building Gold feature tables: %s → %s", start.date(), end.date())
    if dry_run:
        logger.info("[DRY RUN] Skipping Gold feature build.")
        return

    from stressbench.common.config import load_instruments, load_fee_schedules

    silver_root = Path(silver_dir)
    gold_root = Path(gold_dir)
    gold_root.mkdir(parents=True, exist_ok=True)

    instruments = load_instruments()
    fee_cfg = load_fee_schedules()
    dates = _date_range(start, end)

    for date_str in dates:
        logger.info("Gold: processing %s", date_str)
        one_day = {date_str}

        # Load books in two passes so depth quality can be tracked downstream.
        # Real L2 (executable-quality) is preferred for net-profit calculations.
        # Synthetic klines are acceptable for microstructure feature snapshots only.
        _REAL_L2_CHANNELS = [
            "depth", "level2", "book",
            "tardis_book_snapshot_1s", "tardis_incremental_book_l2",
        ]
        books_real = _load_silver_channels(silver_root, _REAL_L2_CHANNELS, one_day, depth_source="real_l2_snapshot")
        books_synth = _load_silver_channels(silver_root, ["klines"], one_day, depth_source="synthetic_kline")
        book_frames = [df for df in [books_real, books_synth] if not df.is_empty()]
        books = pl.concat(book_frames, how="diagonal") if book_frames else pl.DataFrame()

        trades = _load_silver_channels(
            silver_root,
            ["trade", "aggTrades", "matches", "tardis_trades"],
            one_day,
        )
        transfers = _load_silver_channels(silver_root, ["transfer"], one_day)
        swaps = _load_silver_channels(silver_root, ["swap"], one_day)

        book_1m = _compute_book_1m(books)
        trade_1m = _compute_trade_1m(trades)

        if not book_1m.is_empty():
            feat_book = (
                book_1m.join(
                    trade_1m, on=["ts_1m_ns", "venue_id", "instrument_id"], how="left"
                )
                if not trade_1m.is_empty()
                else book_1m
            )
            _write_gold(gold_root, "feat_book_1m", date_str, feat_book)

            basis = _compute_basis_features(book_1m, instruments)
            _write_gold(gold_root, "feat_basis_1m", date_str, basis)

            sc_dev = _compute_stablecoin_prices(book_1m, instruments)
            _write_gold(gold_root, "feat_stablecoin_dev_1m", date_str, sc_dev)

            frag = _compute_fragmentation(book_1m, instruments)
            _write_gold(gold_root, "feat_fragmentation_1m", date_str, frag)

        # Paper-grade net profit: real L2 books only (real_l2_snapshot / real_l2_incremental).
        # Synthetic kline books are NOT allowed here — they do not represent executable depth.
        if not books_real.is_empty():
            net_profit = _compute_net_profit_1m(books_real, instruments, fee_cfg)
            _write_gold(gold_root, "feat_net_profit_1m", date_str, net_profit)

        # Proxy net profit: all books including synthetic klines.
        # Written only when no real L2 is available — for CI / smoke / demo mode only.
        # Never use this table for paper results.
        if books_real.is_empty() and not books.is_empty():
            net_profit_proxy = _compute_net_profit_1m(books, instruments, fee_cfg)
            _write_gold(gold_root, "feat_net_profit_1m_proxy", date_str, net_profit_proxy)

        settle = _compute_settlement_1m(transfers, swaps)
        _write_gold(gold_root, "feat_settlement_1m", date_str, settle)

    logger.info("Gold feature build complete.")


# ---------------------------------------------------------------------------
# Labels build (public)
# ---------------------------------------------------------------------------

def _add_split_column(df: pl.DataFrame, event_windows: dict, ts_col: str) -> pl.DataFrame:
    df = df.with_columns(pl.lit("train").alias("split"))
    for _name, ev in event_windows.items():
        if not isinstance(ev, dict) or "start" not in ev:
            continue
        split = ev.get("split", "train")
        s_ns = _ts_to_ns(ev["start"])
        e_ns = _ts_to_ns(ev["end"])
        df = df.with_columns(
            pl.when((pl.col(ts_col) >= s_ns) & (pl.col(ts_col) <= e_ns))
            .then(pl.lit(split))
            .otherwise(pl.col("split"))
            .alias("split")
        )
    return df


def build_labels(gold_dir: str, start: datetime, end: datetime, dry_run: bool) -> None:
    """Generate forward-looking labels from Gold feature tables and write dataset.parquet."""
    logger.info("Building labels: %s → %s", start.date(), end.date())
    if dry_run:
        logger.info("[DRY RUN] Skipping label build.")
        return

    from stressbench.labels.basis_labels import add_basis_labels
    from stressbench.labels.regime_labels import add_regime_labels
    from stressbench.labels.recovery_labels import add_recovery_labels
    from stressbench.labels.arbitrage_labels import add_arbitrage_labels
    from stressbench.labels.profitability_labels import (
        add_profitability_rank_label,
        add_is_profitable_label,
    )
    from stressbench.common.config import load_event_windows

    gold_root = Path(gold_dir)
    dates_set = set(_date_range(start, end))
    TS = "ts_1m_ns"

    def _load_table(name: str) -> pl.DataFrame:
        frames = []
        for f in sorted(gold_root.glob(f"{name}/date=*/part-0.parquet")):
            d = f.parent.name.split("=", 1)[1]
            if d not in dates_set:
                continue
            try:
                frames.append(pl.read_parquet(f))
            except Exception as exc:
                logger.warning("Skip %s: %s", f, exc)
        return pl.concat(frames) if frames else pl.DataFrame()

    basis_df = _load_table("feat_basis_1m")
    settle_df = _load_table("feat_settlement_1m")
    net_df = _load_table("feat_net_profit_1m")
    sc_dev_df = _load_table("feat_stablecoin_dev_1m")
    frag_df = _load_table("feat_fragmentation_1m")
    book_feat_df = _load_table("feat_book_1m")

    if basis_df.is_empty():
        logger.warning("No basis features found for the given date range; skipping labels.")
        return

    df = basis_df.clone()

    # Microstructure features: aggregate across venues per minute
    if not book_feat_df.is_empty():
        book_agg_exprs = [pl.col("spread_bps").mean().alias("spread_bps_mean")]
        for col in ("depth_bid_10bp", "depth_ask_10bp"):
            if col in book_feat_df.columns:
                book_agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        if "imbalance_1bp" in book_feat_df.columns:
            book_agg_exprs.append(pl.col("imbalance_1bp").mean().alias("imbalance_1bp_mean"))
        if "data_quality_score" in book_feat_df.columns:
            book_agg_exprs.append(
                pl.col("data_quality_score").min().alias("data_quality_score_min")
            )
        if "trade_count_1m" in book_feat_df.columns:
            book_agg_exprs.append(pl.col("trade_count_1m").sum().alias("trade_count_1m_total"))
        if "trade_volume_1m" in book_feat_df.columns:
            book_agg_exprs.append(pl.col("trade_volume_1m").sum().alias("trade_volume_1m_total"))
        book_agg = book_feat_df.group_by(TS).agg(book_agg_exprs)
        df = df.join(book_agg, on=TS, how="left")

    if not settle_df.is_empty():
        settle_cols = [TS] + [
            c for c in [
                "transfer_count_1m", "transfer_volume_1m", "large_transfer_count_1m",
                "gas_proxy", "block_lag_proxy", "dex_swap_volume_1m", "dex_net_flow_1m",
                "mint_count_1h", "burn_count_1h",
            ]
            if c in settle_df.columns
        ]
        df = df.join(settle_df.select(settle_cols), on=TS, how="left")

    if not net_df.is_empty():
        net_cols = [TS] + [c for c in net_df.columns if c.startswith("net_profit_bps_")]
        df = df.join(net_df.select(net_cols), on=TS, how="left")

    # Mean absolute deviation across stablecoins and venues → regime detection
    if not sc_dev_df.is_empty():
        avg_dev = sc_dev_df.group_by(TS).agg(
            pl.col("deviation_from_1_usd_bps").abs().mean().alias("deviation_from_1_usd_bps")
        )
        df = df.join(avg_dev, on=TS, how="left")

    # Mean fragmentation metrics across stablecoins
    if not frag_df.is_empty():
        avg_frag = frag_df.group_by(TS).agg([
            pl.col("num_active_venues").mean().alias("num_active_venues_mean"),
            pl.col("mid_dispersion_bps").mean().alias("mid_dispersion_bps_mean"),
            pl.col("max_minus_min_bps").mean().alias("max_minus_min_bps_mean"),
        ])
        df = df.join(avg_frag, on=TS, how="left")

    # --- Apply labels ---
    # Primary (USDC-specific, fallback to max-abs) → label_basis_*  [backward compat]
    if "cross_quote_basis_primary_bps" in df.columns:
        df = add_basis_labels(df, basis_col="cross_quote_basis_primary_bps", ts_col=TS, label_prefix="basis")
    # USDC-specific → label_basis_usdc_*
    if "cross_quote_basis_usdc_bps" in df.columns:
        df = add_basis_labels(df, basis_col="cross_quote_basis_usdc_bps", ts_col=TS, label_prefix="basis_usdc")
    # Max-absolute (generic stress) → label_basis_maxabs_*
    if "cross_quote_basis_maxabs_bps" in df.columns:
        df = add_basis_labels(df, basis_col="cross_quote_basis_maxabs_bps", ts_col=TS, label_prefix="basis_maxabs")

    df = add_regime_labels(df, ts_col=TS)

    event_windows = load_event_windows()
    ev_list = [
        {"start_ns": _ts_to_ns(v["start"]), "end_ns": _ts_to_ns(v["end"])}
        for v in (event_windows.values() if isinstance(event_windows, dict) else [])
        if isinstance(v, dict) and "start" in v and "end" in v
    ]
    if ev_list and "deviation_from_1_usd_bps" in df.columns:
        df = add_recovery_labels(df, event_windows=ev_list, ts_col=TS)

    df = add_arbitrage_labels(df, ts_col=TS)

    for q in _NOTIONAL_SIZES_USD:
        col = f"net_profit_bps_q{q}"
        if col in df.columns:
            df = add_profitability_rank_label(df, net_profit_col=col, ts_col=TS)
            df = add_is_profitable_label(df, net_profit_col=col)
            break

    # Attach split column for train_models.py compatibility
    if isinstance(event_windows, dict):
        df = _add_split_column(df, event_windows, TS)

    out_path = gold_root / "dataset.parquet"
    df.write_parquet(out_path)
    logger.info(
        "Dataset written to %s  (%d rows, %d columns).",
        out_path, len(df), len(df.columns),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.skip_silver:
        build_silver(args.bronze_dir, args.silver_dir, args.start, args.end, args.dry_run)

    build_gold_features(args.silver_dir, args.gold_dir, args.start, args.end, args.dry_run)

    if not args.skip_labels:
        build_labels(args.gold_dir, args.start, args.end, args.dry_run)

    logger.info("Feature build pipeline complete.")


if __name__ == "__main__":
    main()
