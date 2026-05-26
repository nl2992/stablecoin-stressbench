#!/usr/bin/env python3
"""Patch dataset.parquet to add missing derived columns and USDC/maxabs labels.

Adds:
    cross_quote_basis_maxabs_bps   — max-absolute of USDC and USDT basis
    cross_quote_basis_primary_bps  — USDC basis with maxabs fallback

    label_basis_usdc_*             — USDC-specific forward-looking labels
    label_basis_maxabs_*           — max-absolute forward-looking labels

These columns are required for the full experiment grid (basis_usdc_* and
basis_maxabs_* tasks). They can be derived entirely from the existing
cross_quote_basis_usdc_bps and cross_quote_basis_usdt_bps columns without
rebuilding from raw data.

Usage:
    python scripts/patch_dataset.py
    python scripts/patch_dataset.py --data-dir data/gold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from stressbench.common.logging import get_logger
from stressbench.labels.basis_labels import add_basis_labels

logger = get_logger(__name__)

_TS_COL = "ts_1m_ns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch dataset.parquet with derived columns.")
    parser.add_argument("--data-dir", default="data/gold")
    return parser.parse_args()


def patch(dataset_path: Path) -> None:
    logger.info("Loading %s", dataset_path)
    df = pl.read_parquet(str(dataset_path))
    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Derived basis columns
    # ------------------------------------------------------------------
    needed_feat = {
        "cross_quote_basis_maxabs_bps",
        "cross_quote_basis_primary_bps",
    }
    missing_feat = needed_feat - set(df.columns)

    if "cross_quote_basis_usdc_bps" not in df.columns or "cross_quote_basis_usdt_bps" not in df.columns:
        raise RuntimeError(
            "Dataset is missing cross_quote_basis_usdc_bps or cross_quote_basis_usdt_bps; "
            "rebuild from source."
        )

    if "cross_quote_basis_maxabs_bps" in missing_feat:
        df = df.with_columns(
            pl.when(
                pl.col("cross_quote_basis_usdc_bps").abs()
                >= pl.col("cross_quote_basis_usdt_bps").abs()
            )
            .then(pl.col("cross_quote_basis_usdc_bps"))
            .otherwise(pl.col("cross_quote_basis_usdt_bps"))
            .alias("cross_quote_basis_maxabs_bps")
        )
        logger.info("Added cross_quote_basis_maxabs_bps")

    if "cross_quote_basis_primary_bps" in missing_feat:
        df = df.with_columns(
            pl.col("cross_quote_basis_usdc_bps")
            .fill_null(pl.col("cross_quote_basis_maxabs_bps"))
            .alias("cross_quote_basis_primary_bps")
        )
        logger.info("Added cross_quote_basis_primary_bps")

    # ------------------------------------------------------------------
    # USDC-specific labels
    # ------------------------------------------------------------------
    if "label_basis_usdc_1m_gt10bps" not in df.columns:
        logger.info("Adding label_basis_usdc_* columns …")
        before = set(df.columns)
        df = add_basis_labels(
            df,
            basis_col="cross_quote_basis_usdc_bps",
            ts_col=_TS_COL,
            label_prefix="basis_usdc",
        )
        added = set(df.columns) - before
        logger.info("  Added %d columns: %s", len(added), sorted(added))
    else:
        logger.info("label_basis_usdc_* already present — skipping.")

    # ------------------------------------------------------------------
    # Max-absolute basis labels
    # ------------------------------------------------------------------
    if "label_basis_maxabs_1m_gt10bps" not in df.columns:
        logger.info("Adding label_basis_maxabs_* columns …")
        before = set(df.columns)
        df = add_basis_labels(
            df,
            basis_col="cross_quote_basis_maxabs_bps",
            ts_col=_TS_COL,
            label_prefix="basis_maxabs",
        )
        added = set(df.columns) - before
        logger.info("  Added %d columns: %s", len(added), sorted(added))
    else:
        logger.info("label_basis_maxabs_* already present — skipping.")

    logger.info("Final dataset: %d rows × %d cols", len(df), len(df.columns))
    df.write_parquet(str(dataset_path))
    logger.info("Saved %s", dataset_path)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.data_dir) / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.parquet not found at {dataset_path}")
    patch(dataset_path)


if __name__ == "__main__":
    main()
