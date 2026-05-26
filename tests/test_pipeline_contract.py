"""Pipeline-contract integration test.

Verifies the full Bronze → Silver → Gold path for Tardis archive data using
only synthetic in-memory data — no network calls, no real API keys.

Synthetic Tardis CSV
  → archive_to_bronze.tardis_to_bronze        (Bronze Parquet)
  → build_features.build_silver               (Silver Parquet)
  → build_features.build_gold_features        (Gold Parquet + dataset.parquet)

Acceptance criteria
-------------------
* Silver trade rows > 0
* Silver book rows > 0
* feat_book_1m rows > 0
* feat_net_profit_1m either has rows or fails only because fewer than two
  venues are present (the function returns an empty DataFrame rather than
  raising in that case)

This test is the regression guard: if any routing layer silently breaks —
wrong channel name, wrong normalizer key, column-order mismatch — it will
surface here before reaching production.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest


_DATE = "2024-01-01"
_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
_END = datetime(2024, 1, 2, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Synthetic data writers
# ---------------------------------------------------------------------------

def _write_tardis_trades(exchange: str, symbol: str, root: Path) -> Path:
    """Write 60 trade rows (one per minute across midnight) to a Tardis CSV."""
    path = root / f"{exchange}_{symbol}_trades.csv"
    fieldnames = ["exchange", "symbol", "timestamp", "localTimestamp",
                  "id", "side", "price", "amount"]
    base_ts_us = 1_704_067_200_000_000  # 2024-01-01T00:00:00 UTC in microseconds
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(60):
            ts_us = base_ts_us + i * 60_000_000  # one per minute
            ts_iso = f"2024-01-01T00:{i:02d}:00.000000Z"
            writer.writerow({
                "exchange": exchange,
                "symbol": symbol,
                "timestamp": ts_iso,
                "localTimestamp": str(ts_us),
                "id": f"t{i:04d}",
                "side": "buy" if i % 2 == 0 else "sell",
                "price": str(42_500.0 + i * 0.5),
                "amount": "0.10",
            })
    return path


def _write_tardis_book_snapshot(exchange: str, symbol: str, root: Path) -> Path:
    """Write 60 book snapshot rows (one per minute) to a Tardis CSV."""
    path = root / f"{exchange}_{symbol}_book_snapshot_1s.csv"
    fieldnames = ["exchange", "symbol", "timestamp", "localTimestamp", "isSnapshot",
                  "bids[0].price", "bids[0].amount", "bids[1].price", "bids[1].amount",
                  "asks[0].price", "asks[0].amount", "asks[1].price", "asks[1].amount"]
    base_ts_us = 1_704_067_200_000_000
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(60):
            ts_us = base_ts_us + i * 60_000_000
            ts_iso = f"2024-01-01T00:{i:02d}:00.000000Z"
            mid = 42_500.0 + i * 0.5
            writer.writerow({
                "exchange": exchange,
                "symbol": symbol,
                "timestamp": ts_iso,
                "localTimestamp": str(ts_us),
                "isSnapshot": "true",
                "bids[0].price": str(mid - 1),
                "bids[0].amount": "1.0",
                "bids[1].price": str(mid - 2),
                "bids[1].amount": "2.0",
                "asks[0].price": str(mid + 1),
                "asks[0].amount": "0.8",
                "asks[1].price": str(mid + 2),
                "asks[1].amount": "1.5",
            })
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_dirs(tmp_path):
    """Return (bronze_root, silver_root, gold_root, vendor_staging)."""
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    vendor = tmp_path / "vendor"
    for d in (bronze, silver, gold, vendor):
        d.mkdir(parents=True)
    return bronze, silver, gold, vendor


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_tardis_pipeline_contract(pipeline_dirs):
    """Tardis CSV → Bronze → Silver → Gold runs without errors and produces rows."""
    from stressbench.ingestion.archive_to_bronze import tardis_to_bronze

    bronze, silver, gold, vendor = pipeline_dirs
    exchange, symbol = "coinbase", "BTC-USD"

    # 1. Write synthetic Tardis CSVs
    trades_csv = _write_tardis_trades(exchange, symbol, vendor)
    book_csv = _write_tardis_book_snapshot(exchange, symbol, vendor)

    # 2. Canonicalise to Bronze
    n_trades = tardis_to_bronze(
        trades_csv, exchange, symbol, "trades", _DATE, bronze_root=bronze
    )
    n_books = tardis_to_bronze(
        book_csv, exchange, symbol, "book_snapshot_1s", _DATE, bronze_root=bronze
    )
    assert n_trades > 0, "tardis_to_bronze wrote 0 trade rows"
    assert n_books > 0, "tardis_to_bronze wrote 0 book rows"

    # Verify Tardis-specific channel names in Bronze layout
    trade_bronze = list(bronze.glob("venue=coinbase/channel=tardis_trades/**/*.parquet"))
    book_bronze = list(bronze.glob(
        "venue=coinbase/channel=tardis_book_snapshot_1s/**/*.parquet"
    ))
    assert trade_bronze, "Bronze trade files not found under channel=tardis_trades"
    assert book_bronze, "Bronze book files not found under channel=tardis_book_snapshot_1s"

    # 3. Build Silver (import and call build_silver directly)
    import sys
    from pathlib import Path as _P
    scripts_dir = str(_P(__file__).parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_features import build_silver  # type: ignore[import]

    build_silver(str(bronze), str(silver), _START, _END, dry_run=False)

    # 4. Assert Silver rows
    silver_trades = list(silver.glob("venue=coinbase/channel=tardis_trades/**/*.parquet"))
    silver_books = list(silver.glob(
        "venue=coinbase/channel=tardis_book_snapshot_1s/**/*.parquet"
    ))
    assert silver_trades, "No Silver trade files written"
    assert silver_books, "No Silver book files written"

    trade_df = pl.concat([pl.read_parquet(p) for p in silver_trades])
    book_df = pl.concat([pl.read_parquet(p) for p in silver_books])

    assert len(trade_df) > 0, "Silver trade rows = 0"
    assert len(book_df) > 0, "Silver book rows = 0"

    # Verify Silver schema basics
    assert "price" in trade_df.columns and "side" in trade_df.columns
    assert "price" in book_df.columns and "level" in book_df.columns

    # 5. Build Gold features + labels (dataset.parquet written by build_labels)
    from build_features import build_gold_features, build_labels  # type: ignore[import]

    build_gold_features(str(silver), str(gold), _START, _END, dry_run=False)
    build_labels(str(gold), _START, _END, dry_run=False)

    # 6. Assert Gold rows
    book_feat_files = list(gold.glob("feat_book_1m/**/*.parquet"))
    assert book_feat_files, "feat_book_1m produced no output files"

    feat_df = pl.concat([pl.read_parquet(p) for p in book_feat_files])
    assert len(feat_df) > 0, "feat_book_1m rows = 0"

    # feat_net_profit_1m requires ≥2 venues; must either have rows or be absent/empty
    net_files = list(gold.glob("feat_net_profit_1m/**/*.parquet"))
    if net_files:
        net_df = pl.concat([pl.read_parquet(p) for p in net_files])
        # If present, allow empty (single-venue pipeline) but not an error
        assert isinstance(net_df, pl.DataFrame)

    # dataset.parquet must exist
    dataset_path = gold / "dataset.parquet"
    assert dataset_path.exists(), "dataset.parquet not written"
    dataset_df = pl.read_parquet(dataset_path)
    assert len(dataset_df) > 0, "dataset.parquet is empty"
