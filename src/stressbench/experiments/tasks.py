"""Task definitions for the Stablecoin StressBench experiment grid.

Each task entry specifies:
    label       : column name in dataset.parquet
    task        : "classification" or "regression"
    horizon     : forecast horizon string (for display / docs)
    notional_usd: notional size used for economic metrics
    basis_asset : which stablecoin's basis drives the label (for docs)
    net_profit_col: which net_profit_bps column to use for economic scoring

Primary tasks (basis classification):
    basis_1m_gt10bps          — USDC primary basis, 1-minute, >10 bps  [default]
    basis_usdc_1m_gt10bps     — USDC-specific, 1-minute, >10 bps
    basis_maxabs_1m_gt10bps   — max-absolute, 1-minute, >10 bps

Extended horizon tasks:
    basis_usdc_5m_gt25bps     — USDC-specific, 5-minute, >25 bps
    basis_usdc_15m_gt25bps    — USDC-specific, 15-minute, >25 bps

Executable arbitrage tasks (uses realized net profit labels):
    executable_arb_q10000_5m  — profitable at $10K within 5 minutes
    executable_arb_q50000_5m  — profitable at $50K within 5 minutes
"""

from __future__ import annotations

TASKS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Primary benchmark task (backward-compat label name)
    # ------------------------------------------------------------------
    "basis_1m_gt10bps": {
        "label": "label_basis_1m_gt10bps",
        "task": "classification",
        "horizon": "1m",
        "basis_asset": "primary",
        "notional_usd": 50_000,
        "net_profit_col": "net_profit_bps_q50000",
        "description": "USDC primary basis >10 bps in 1 minute (benchmark default)",
    },

    # ------------------------------------------------------------------
    # USDC-specific basis tasks (for SVB / USDC depeg event analysis)
    # ------------------------------------------------------------------
    "basis_usdc_1m_gt10bps": {
        "label": "label_basis_usdc_1m_gt10bps",
        "task": "classification",
        "horizon": "1m",
        "basis_asset": "USDC",
        "notional_usd": 50_000,
        "net_profit_col": "net_profit_bps_q50000",
        "description": "USDC-specific basis >10 bps in 1 minute",
    },
    "basis_usdc_5m_gt25bps": {
        "label": "label_basis_usdc_5m_gt25bps",
        "task": "classification",
        "horizon": "5m",
        "basis_asset": "USDC",
        "notional_usd": 50_000,
        "net_profit_col": "net_profit_bps_q50000",
        "description": "USDC-specific basis >25 bps in 5 minutes",
    },
    "basis_usdc_15m_gt25bps": {
        "label": "label_basis_usdc_15m_gt25bps",
        "task": "classification",
        "horizon": "15m",
        "basis_asset": "USDC",
        "notional_usd": 50_000,
        "net_profit_col": "net_profit_bps_q50000",
        "description": "USDC-specific basis >25 bps in 15 minutes",
    },

    # ------------------------------------------------------------------
    # Max-absolute basis tasks (generic stress detector)
    # ------------------------------------------------------------------
    "basis_maxabs_1m_gt10bps": {
        "label": "label_basis_maxabs_1m_gt10bps",
        "task": "classification",
        "horizon": "1m",
        "basis_asset": "max_abs",
        "notional_usd": 50_000,
        "net_profit_col": "net_profit_bps_q50000",
        "description": "Max-absolute cross-quote basis >10 bps in 1 minute",
    },

    # ------------------------------------------------------------------
    # Executable arbitrage tasks (label derived from net_profit_bps)
    # ------------------------------------------------------------------
    "executable_arb_q10000_1m": {
        "label": "label_arb_q10000_1m_gt0bps",
        "task": "classification",
        "horizon": "1m",
        "basis_asset": None,
        "notional_usd": 10_000,
        "net_profit_col": "net_profit_bps_q10000",
        "description": "Executable arbitrage at $10K notional within 1 minute",
    },
    "executable_arb_q10000_5m": {
        "label": "label_arb_q10000_5m_gt0bps",
        "task": "classification",
        "horizon": "5m",
        "basis_asset": None,
        "notional_usd": 10_000,
        "net_profit_col": "net_profit_bps_q10000",
        "description": "Executable arbitrage at $10K notional within 5 minutes",
    },
    "executable_arb_q50000_5m": {
        "label": "label_arb_q50000_5m_gt0bps",
        "task": "classification",
        "horizon": "5m",
        "basis_asset": None,
        "notional_usd": 50_000,
        "net_profit_col": "net_profit_bps_q50000",
        "description": "Executable arbitrage at $50K notional within 5 minutes",
    },
}
