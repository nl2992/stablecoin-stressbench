#!/usr/bin/env python3
"""Pull raw market data from all configured venues and write to Bronze layer.

Usage:
    python scripts/pull_data.py --start 2024-01-01 --end 2024-01-07
    python scripts/pull_data.py --start 2024-01-01 --end 2024-01-07 --venues binance coinbase
    python scripts/pull_data.py --start 2024-01-01 --end 2024-01-07 --mode archive
    python scripts/pull_data.py --start 2024-01-15 --end 2024-01-16 --venues binance --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from stressbench.common.config import load_config, load_token_addresses
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# Binance data types pulled in archive mode
_BINANCE_DATA_TYPES = ["aggTrades", "klines"]

# Tardis data types to fetch for Coinbase / Kraken (requires API key)
_TARDIS_DATA_TYPES = ["trades", "book_snapshot_1s"]

# Map from instruments.yaml venue_id to Tardis exchange name
_TARDIS_EXCHANGE_MAP = {
    "coinbase": "coinbase",
    "kraken": "kraken",
}


def _date_range(start: datetime, end: datetime) -> list[str]:
    """Return list of ISO date strings [start.date, ..., end.date] inclusive."""
    result = []
    cur = start.date()
    stop = end.date()
    while cur <= stop:
        result.append(cur.isoformat())
        cur += timedelta(days=1)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull raw market data to Bronze layer.")
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        help="Start datetime (ISO 8601, UTC). Example: 2024-01-01",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        help="End datetime (ISO 8601, UTC). Example: 2024-01-07",
    )
    parser.add_argument(
        "--venues",
        nargs="*",
        default=None,
        help="Venues to pull (default: all configured venues).",
    )
    parser.add_argument(
        "--mode",
        choices=["archive", "tardis"],
        default="archive",
        help="Data source mode (default: archive). Use 'tardis' for Coinbase/Kraken historical data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/bronze",
        help="Bronze output directory (default: data/bronze).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pulled without actually downloading.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Canonicalization helpers — vendor staging → canonical Bronze layout
# ---------------------------------------------------------------------------

def _canonicalize_binance(csv_paths: list[Path], symbol: str, bronze_root: Path) -> None:
    """Convert downloaded Binance archive CSVs into canonical Bronze Parquet."""
    from stressbench.ingestion.archive_to_bronze import (
        binance_aggtrades_to_bronze,
        binance_klines_to_bronze,
    )
    for csv_path in csv_paths:
        if not csv_path or not csv_path.exists():
            continue
        # Derive date from file name e.g. BTCUSDT-aggTrades-2024-01-15.csv
        parts = csv_path.stem.split("-")
        try:
            date = "-".join(parts[-3:])  # last three segments = YYYY-MM-DD
        except IndexError:
            logger.warning("Cannot parse date from %s; skipping canonicalizer.", csv_path)
            continue
        name_lower = csv_path.name.lower()
        if "aggtrades" in name_lower:
            binance_aggtrades_to_bronze(csv_path, symbol, date, bronze_root)
        elif "klines" in name_lower or "1m" in name_lower:
            binance_klines_to_bronze(csv_path, symbol, date, bronze_root)
        else:
            logger.debug("No canonicalizer for Binance file %s; skipping.", csv_path.name)


def _canonicalize_tardis(
    file_path: Path | None,
    exchange: str,
    symbol: str,
    data_type: str,
    date: str,
    bronze_root: Path,
) -> None:
    """Convert a downloaded Tardis CSV.gz file into canonical Bronze Parquet."""
    if not file_path or not Path(file_path).exists():
        return
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze
    tardis_to_bronze(Path(file_path), exchange, symbol, data_type, date, bronze_root)


# ---------------------------------------------------------------------------
# Binance archive
# ---------------------------------------------------------------------------

def pull_binance_archive(
    start: datetime,
    end: datetime,
    output_dir: str,
    dry_run: bool,
) -> None:
    """Download Binance Vision spot+futures archive files for all configured symbols."""
    from stressbench.ingestion.binance_archive import pull_event_window

    cfg = load_config()
    instruments = cfg.get("instruments", [])
    binance_symbols = [
        inst["native_symbol"]
        for inst in instruments
        if inst.get("venue_id") == "binance"
    ]
    if not binance_symbols:
        logger.warning("No Binance symbols found in instruments.yaml; skipping.")
        return

    root = Path(output_dir)
    start_date = start.date().isoformat()
    end_date = end.date().isoformat()

    for symbol in binance_symbols:
        if dry_run:
            logger.info(
                "[DRY RUN] Would pull Binance archive: symbol=%s, %s → %s, types=%s",
                symbol, start_date, end_date, _BINANCE_DATA_TYPES,
            )
            continue
        logger.info("Pulling Binance archive: symbol=%s, %s → %s", symbol, start_date, end_date)
        csv_paths = pull_event_window(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_types=_BINANCE_DATA_TYPES,
            root=root / "vendor=binance_archive",  # vendor staging area
        )
        # Canonicalize: convert vendor CSVs → venue=binance/channel=.../... Bronze
        _canonicalize_binance(csv_paths, symbol, root)


# ---------------------------------------------------------------------------
# Tardis (Coinbase / Kraken historical data)
# ---------------------------------------------------------------------------

def pull_tardis(
    start: datetime,
    end: datetime,
    venues: list[str],
    output_dir: str,
    dry_run: bool,
) -> None:
    """Download historical data from Tardis for the requested venues."""
    from stressbench.ingestion.tardis_loader import pull_tardis_day

    cfg = load_config()
    instruments = cfg.get("instruments", [])
    root = Path(output_dir)
    dates = _date_range(start, end)

    for venue in venues:
        tardis_exchange = _TARDIS_EXCHANGE_MAP.get(venue)
        if not tardis_exchange:
            logger.debug("No Tardis mapping for venue '%s'; skipping.", venue)
            continue

        symbols = [
            inst["native_symbol"]
            for inst in instruments
            if inst.get("venue_id") == venue
        ]
        if not symbols:
            logger.warning("No symbols for venue '%s' in instruments.yaml; skipping.", venue)
            continue

        for symbol in symbols:
            for data_type in _TARDIS_DATA_TYPES:
                for day in dates:
                    if dry_run:
                        logger.info(
                            "[DRY RUN] Would pull Tardis: exchange=%s symbol=%s type=%s date=%s",
                            tardis_exchange, symbol, data_type, day,
                        )
                        continue
                    logger.info(
                        "Pulling Tardis: exchange=%s symbol=%s type=%s date=%s",
                        tardis_exchange, symbol, data_type, day,
                    )
                    vendor_path = pull_tardis_day(
                        exchange=tardis_exchange,
                        symbol=symbol,
                        data_type=data_type,
                        date=day,
                        root=root / "vendor=tardis",  # vendor staging area
                    )
                    # Canonicalize into venue=<exchange>/channel=<ch>/...
                    _canonicalize_tardis(
                        file_path=Path(vendor_path) if vendor_path else None,
                        exchange=tardis_exchange,
                        symbol=symbol,
                        data_type=data_type,
                        date=day,
                        bronze_root=root,
                    )


# ---------------------------------------------------------------------------
# Etherscan (on-chain stablecoin transfers)
# ---------------------------------------------------------------------------

def pull_etherscan(
    start: datetime,
    end: datetime,
    output_dir: str,
    dry_run: bool,
) -> None:
    """Download ERC-20 transfer events for all configured stablecoin tokens."""
    from stressbench.ingestion.etherscan_loader import (
        fetch_block_by_timestamp,
        fetch_token_transfers,
        save_transfers_to_bronze,
    )

    tokens = load_token_addresses()
    if not tokens:
        logger.warning("No token addresses configured; skipping Etherscan pull.")
        return

    root = Path(output_dir)
    dates = _date_range(start, end)

    if dry_run:
        for token_symbol, chain_map in tokens.items():
            if "ethereum" not in chain_map:
                continue
            logger.info(
                "[DRY RUN] Would pull Etherscan: token=%s, %s → %s",
                token_symbol, start.date(), end.date(),
            )
        return

    # Resolve block numbers for the full range once
    start_block = fetch_block_by_timestamp(int(start.timestamp()))
    end_block = fetch_block_by_timestamp(int(end.timestamp()))

    if start_block is None or end_block is None:
        logger.error(
            "Could not resolve block numbers for range %s → %s. "
            "Check ETHERSCAN_API_KEY.",
            start.date(), end.date(),
        )
        return

    logger.info(
        "Etherscan block range: %d → %d (%s → %s)",
        start_block, end_block, start.date(), end.date(),
    )

    for token_symbol, chain_map in tokens.items():
        if "ethereum" not in chain_map:
            continue

        logger.info("Fetching Etherscan transfers: token=%s", token_symbol)
        transfers = fetch_token_transfers(
            token_symbol=token_symbol,
            start_block=start_block,
            end_block=end_block,
        )

        if not transfers:
            logger.info("No transfers found for %s in range.", token_symbol)
            continue

        # Partition by date and save to Bronze
        from collections import defaultdict
        by_date: dict[str, list] = defaultdict(list)
        for tx in transfers:
            # Etherscan timeStamp is Unix seconds
            try:
                tx_date = datetime.fromtimestamp(
                    int(tx.get("timeStamp", 0)), tz=timezone.utc
                ).date().isoformat()
            except (ValueError, TypeError):
                tx_date = dates[0]  # fallback to start date
            by_date[tx_date].append(tx)

        for day, day_transfers in by_date.items():
            save_transfers_to_bronze(
                transfers=day_transfers,
                token_symbol=token_symbol,
                date=day,
                root=root,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config()

    all_venues = list(cfg.get("instruments", [{}]))
    # Deduplicate venue_ids from instruments list
    all_venues = list(dict.fromkeys(
        inst.get("venue_id", "") for inst in cfg.get("instruments", [])
    ))
    venues = args.venues or all_venues

    logger.info(
        "Pulling data: mode=%s, venues=%s, start=%s, end=%s",
        args.mode, venues, args.start.date(), args.end.date(),
    )

    if args.mode == "archive":
        if "binance" in venues:
            pull_binance_archive(args.start, args.end, args.output_dir, args.dry_run)
        if "coinbase" in venues or "kraken" in venues:
            logger.info(
                "Coinbase/Kraken have no public archive; redirecting to Tardis mode "
                "(requires TARDIS_API_KEY in environment)."
            )
            tardis_venues = [v for v in venues if v in ("coinbase", "kraken")]
            pull_tardis(args.start, args.end, tardis_venues, args.output_dir, args.dry_run)

    elif args.mode == "tardis":
        pull_tardis(args.start, args.end, venues, args.output_dir, args.dry_run)

    # Always pull on-chain data regardless of mode
    pull_etherscan(args.start, args.end, args.output_dir, args.dry_run)

    logger.info("Data pull complete.")


if __name__ == "__main__":
    main()
