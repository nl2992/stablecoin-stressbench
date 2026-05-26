"""Feature set definitions for the Stablecoin StressBench experiment grid.

Each feature set is a named list of column names from ``dataset.parquet``.
``None`` means "use all non-label, non-metadata columns" (same as the default
train_models.py behaviour).

Sets are ordered from narrowest to broadest signal:
    price_only          → basis and stablecoin deviation columns only
    price_plus_book     → add microstructure (spread, depth, imbalance, volume)
    price_book_frag     → add cross-venue fragmentation metrics
    price_book_settle   → add on-chain settlement proxies
    all                 → all available features (no filter)

When a column in a feature set is absent from the dataset (e.g. settlement
columns when on-chain data was not pulled), the experiment runner will drop it
silently and log a warning.
"""

from __future__ import annotations

# ------------------------------------------------------------------
# Basis / price-signal columns
# ------------------------------------------------------------------
_PRICE_COLS: list[str] = [
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_usdt_bps",
    "cross_quote_basis_maxabs_bps",
    "cross_quote_basis_primary_bps",
    "deviation_from_1_usd_bps",
]

# ------------------------------------------------------------------
# Microstructure columns (1-minute aggregates across venues)
# ------------------------------------------------------------------
_BOOK_COLS: list[str] = [
    "spread_bps_mean",
    "depth_bid_10bp_mean",
    "depth_ask_10bp_mean",
    "imbalance_1bp_mean",
    "data_quality_score_min",
    "trade_count_1m_total",
    "trade_volume_1m_total",
]

# ------------------------------------------------------------------
# Cross-venue fragmentation columns
# ------------------------------------------------------------------
_FRAG_COLS: list[str] = [
    "num_active_venues_mean",
    "mid_dispersion_bps_mean",
    "max_minus_min_bps_mean",
]

# ------------------------------------------------------------------
# On-chain settlement proxy columns
# ------------------------------------------------------------------
_SETTLE_COLS: list[str] = [
    "transfer_count_1m",
    "transfer_volume_1m",
    "large_transfer_count_1m",
    "gas_proxy",
    "block_lag_proxy",
    "dex_swap_volume_1m",
    "dex_net_flow_1m",
]

# ------------------------------------------------------------------
# Exported feature sets
# ------------------------------------------------------------------
FEATURE_SETS: dict[str, list[str] | None] = {
    "price_only": _PRICE_COLS,
    "price_plus_book": _PRICE_COLS + _BOOK_COLS,
    "price_book_frag": _PRICE_COLS + _BOOK_COLS + _FRAG_COLS,
    "price_book_settle": _PRICE_COLS + _BOOK_COLS + _FRAG_COLS + _SETTLE_COLS,
    "all": None,
}
