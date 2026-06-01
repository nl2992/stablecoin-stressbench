# Data Card — Stablecoin StressBench

## Overview

StressBench is a transaction-cost-aware benchmark dataset for detecting, forecasting, and economically ranking stablecoin settlement-risk dislocations across centralized and decentralized venues. It covers three stress episodes and three calm control periods between January 2022 and January 2024.

---

## Dataset Summary

| Property | Value |
|---|---|
| **Version** | 0.1.0 |
| **Date range** | 2022-01-10 – 2024-01-21 |
| **Gold rows** | 56,134 (1-minute bars in the current committed dataset) |
| **Gold columns** | 125 (features + labels + meta) |
| **Venues** | Binance spot/futures, Coinbase REST/WS support, Kraken WS support |
| **Instruments** | 18 (USDC, USDT, DAI, BTC cross-pairs) |
| **On-chain** | Ethereum mainnet ERC-20 transfers (USDC, USDT, DAI) |
| **License** | MIT |

---

## Event Windows

### Test split
| Window | Dates | Reason |
|---|---|---|
| USDC SVB depeg | 2023-03-10 – 2023-03-14 | USDC reserve-bank stress / SVB collapse; peak cross-quote BTC-route basis ~350 bps; raw USDC/USD spot peak ~−1300 bps (see note) |
| USDC recovery | 2023-03-15 – 2023-03-20 | Recovery window post SVB resolution |

> **Note on peak deviation figures.** Two distinct measurements appear in this codebase:
> - **Cross-quote basis** (`cross_quote_basis_usdc_bps`): the BTC-route triangular basis, measuring USDC price via `(BTC_USDC − BTC_USDT)/BTC_USDT × 10,000`. Peak during SVB: approximately −350 bps on this metric.
> - **Raw USDC/USD spot deviation**: the direct USDC-to-dollar exchange rate on secondary markets. Peak during SVB: approximately −1,300 bps ($0.87), widely reported and verified by Circle. This figure does NOT appear in the cross-quote basis columns.
> 
> The benchmark's classification labels (`label_basis_usdc_*`) use the cross-quote BTC-route basis.
> The −1,300 bps figure is the secondary-market USDC/USD deviation and is cited for historical context only.

### Validation split
| Window | Dates | Reason |
|---|---|---|
| Terra/Luna collapse | 2022-05-07 – 2022-05-14 | UST/LUNA collapse and contagion stress on other stablecoins |

### Train split (normal control)
| Window | Dates | Reason |
|---|---|---|
| Normal control (Feb 2023) | 2023-02-01 – 2023-02-07 | Calm window immediately before SVB stress |
| Normal control (Jan 2022) | 2022-01-10 – 2022-01-16 | Calm window in early 2022 |
| Normal control (Q1 2024) | 2024-01-15 – 2024-01-21 | Calm window in Q1 2024 |

---

## Data Sources

### Centralized exchange (CeFi)

| Source | Symbols | Channels | Frequency |
|---|---|---|---|
| Binance Vision spot archive | USDCUSDT, BTCUSDT, BTCUSDC, ETHUSDT, ETHUSDC, DAIUSDT | aggTrades, klines (1m OHLCV) | Per-trade / 1-minute |
| Binance Vision USDM futures archive | BTCUSDT | bookDepth (±1–5% bands) | ~30 s snapshots |
| Coinbase REST candles | BTC-USD, USDT-USD | OHLCV (1m) | 1-minute |

> **Note:** Coinbase and Kraken do not publish free public historical archives. The benchmark uses Coinbase REST candles for price reference. Full-depth historical data for these venues requires a Tardis subscription.

### On-chain (DeFi)

| Source | Data type | Chains |
|---|---|---|
| Etherscan V2 API | ERC-20 transfer events | Ethereum mainnet |
| The Graph (Uniswap v3) | Pool-level swap events, liquidity, TVL | Ethereum mainnet |

### Issuer events

Manually curated issuer-level events for USDC (Circle):
- SVB collapse / USDC reserve impact (2023-03-10 – 2023-03-13)

---

## Data Source Provenance

| Source | Bronze ingestor | Silver normalizer | Gold table | depth_source tag |
|---|---|---|---|---|
| Binance Vision aggTrades archive | `archive_to_bronze.py` | `normalize_binance_trades` | `feat_book_1m` (trade stats) | — |
| Binance Vision klines (1m OHLCV) | `archive_to_bronze.py` | `normalize_binance_klines` | `feat_book_1m`, `feat_net_profit_1m` | `synthetic_kline` |
| Binance Vision bookDepth (futures) | `archive_to_bronze.py` | `normalize_binance_depth` | `feat_book_1m`, `feat_net_profit_1m` | `real_l2_incremental` |
| Coinbase WebSocket level2 | `start_live_capture.py` | `normalize_coinbase_level2` | `feat_book_1m`, `feat_net_profit_1m` | `real_l2_snapshot` / `real_l2_incremental` |
| Kraken WebSocket book | `start_live_capture.py` | `normalize_kraken_book` | `feat_book_1m`, `feat_net_profit_1m` | `real_l2_snapshot` |
| Tardis book_snapshot_1s | `archive_to_bronze.py` | `normalize_tardis_book_snapshot_1s` | `feat_book_1m`, `feat_net_profit_1m` | `real_l2_snapshot` |
| Tardis incremental_book_L2 | `archive_to_bronze.py` | `normalize_tardis_incremental_book_l2` | `feat_book_1m`, `feat_net_profit_1m` | `real_l2_incremental` |
| Etherscan ERC-20 transfers | `fetch_real_data.py` | `normalize_etherscan_transfers` | `feat_settlement_1m` | — |
| The Graph Uniswap v3 swaps | `fetch_real_data.py` | `normalize_uniswap_swaps` | `feat_settlement_1m` | — |

> **depth_source** tags each Silver book row by data quality. Valid values: `real_l2_snapshot` (full book snapshot from WebSocket/REST), `real_l2_incremental` (incremental diff updates reconstructed into full book), and `synthetic_kline` (5-level synthetic ladder inferred from OHLCV klines). Net-profit rows carry provenance through `depth_source`, `depth_sources_used`, and `is_paper_grade_depth`; claim scope for proxy legs is documented in `docs/execution_route_coverage.md`.

---

## Silver Layer Schema

Normalized per-venue Parquet files (Hive-partitioned: `venue=*/channel=*/symbol=*/date=*/hour=*/`).

**Common columns:** `ts_event_ns`, `ts_receive_ns`, `venue_id`, `instrument_id`, `native_symbol`, `payload_hash`

**Trades (`fact_trade`):** `price`, `size`, `side`, `trade_id`, `is_outlier_price`, `raw_source`

**Book levels (`fact_book_level`):** `side`, `level` (0 = best), `price`, `size`, `checksum`, `depth_source`,
`is_crossed_book`, `is_negative_size`, `is_sequence_gap`, `is_checksum_failed`, `is_stale_quote`, `is_resync_period`

**On-chain transfers:** `block_number`, `tx_hash`, `from_address`, `to_address`, `value`, `token_symbol`, `ts_unix_seconds`

---

## Gold Layer Schema

One row per UTC minute; produced by `scripts/build_features.py`.

### Core feature tables

| Table | Key columns | Description |
|---|---|---|
| `feat_book_1m` | `ts_1m_ns`, `venue_id`, `instrument_id` | BBO, spread, depth, imbalance, data quality score per venue per minute |
| `feat_basis_1m` | `ts_1m_ns` | Three cross-quote basis columns, stablecoin price tables |
| `feat_net_profit_1m` | `ts_1m_ns` | VWAP round-trip net profit at four notional sizes; `depth_source` column distinguishes real-L2 vs synthetic rows |
| `feat_fragmentation_1m` | `ts_1m_ns`, `stablecoin` | Cross-venue price dispersion per stablecoin per minute |
| `feat_settlement_1m` | `ts_1m_ns` | On-chain settlement proxy (transfer count, gas, Uniswap liquidity) |

### Cross-quote basis columns (feat_basis_1m)

| Column | Description |
|---|---|
| `cross_quote_basis_usdc_bps` | `10000 × (BTC_via_USDC − BTC_direct) / BTC_direct` — USDC-specific basis, primary signal for SVB/USDC stress events |
| `cross_quote_basis_usdt_bps` | `10000 × (BTC_via_USDT − BTC_direct) / BTC_direct` — USDT-specific basis |
| `cross_quote_basis_maxabs_bps` | Max absolute of USDC and USDT basis — generic stress detector |
| `cross_quote_basis_primary_bps` | USDC basis with max-abs fallback — backward-compatible label driver |
| `basis_primary_asset` | `"USDC"` or `"max_abs"` — indicates which route drove `cross_quote_basis_primary_bps` |

### Net-profit columns (feat_net_profit_1m)

| Column | Description |
|---|---|
| `net_profit_bps_q10000` | Round-trip net profit at $10K after taker fees and VWAP price impact |
| `net_profit_bps_q50000` | Same at $50K notional |
| `net_profit_bps_q100000` | Same at $100K notional |
| `net_profit_bps_q500000` | Same at $500K notional |
| `buy_venue` / `sell_venue` | Best route identified by the VWAP walk |
| `depth_source` | One of `"real_l2_snapshot"`, `"real_l2_incremental"`, or `"synthetic_kline"` — indicates book data quality for the selected route |

### Book microstructure columns (feat_book_1m, aggregated in dataset.parquet)

| Column | Description |
|---|---|
| `spread_bps_mean` | Mean bid–ask spread across active venues |
| `depth_bid_10bp_mean` | Mean BTC depth within 10 bps of bid across venues |
| `depth_ask_10bp_mean` | Mean BTC depth within 10 bps of ask across venues |
| `imbalance_1bp_mean` | Mean level-0 quote imbalance `(bid_sz − ask_sz) / (bid_sz + ask_sz)` |
| `data_quality_score_min` | Min data quality score across venues (penalizes checksum failures and resync periods) |

### Labels

**Basis labels** — three families, one per basis variant:

| Family | Column pattern | Basis source |
|---|---|---|
| `label_basis_*` | `label_basis_{1m,5m,15m,1h}[_gt{5,10,25,50}bps]` | `cross_quote_basis_primary_bps` (backward-compat driver) |
| `label_basis_usdc_*` | `label_basis_usdc_{1m,5m,15m,1h}[_gt{5,10,25,50}bps]` | `cross_quote_basis_usdc_bps` (USDC-specific) |
| `label_basis_maxabs_*` | `label_basis_maxabs_{1m,5m,15m,1h}[_gt{5,10,25,50}bps]` | `cross_quote_basis_maxabs_bps` (generic stress) |

Binary columns (`_gt{N}bps`) are `int8` — 1 if `|future_basis| > N bps`.  Regression targets (no threshold suffix) are `float64` in basis points.

**Executable arbitrage labels** (notional × horizon): `label_arb_q{10000,50000,100000,500000}_{1m,5m,15m}_gt{0,5,10}bps` — 1 if net profit exceeds the threshold over the next horizon at the given notional size.

**Regime labels**: `label_regime_{calm,stress,recovery}` — manual event-based regime tags.

**Recovery label**: `label_recovery_within_{1h,4h,24h}` — 1 if basis returns within 5 bps threshold within H.

---

## Row Counts by Split

| Split | Rows | Dates |
|---|---|---|
| Train (3 control windows) | 28,776 | calm-control windows |
| Validation (Terra/Luna 2022) | 11,526 | May 2022 stress window |
| Test (USDC/SVB depeg 2023) | 15,832 | Mar 2023 stress and recovery window |
| **Total** | **56,134** | current `data/gold/dataset.parquet` |

The frozen baseline paper tables use the filtered benchmark-freeze view recorded
in `results/paper/table_1_data_coverage.csv` (47,487 rows). Current add-on
experiments read the committed Gold dataset directly.

---

## Known Limitations

1. **Depth coverage by route leg**: The Gold dataset primarily uses Binance depth and kline-derived proxy depth for BTC-route VWAP execution labels. The Silver pipeline supports Coinbase WebSocket L2, Kraken WebSocket book (via live capture), and Tardis snapshots (via archive), but **full historical L2 for Coinbase and Kraken during the SVB period (March 2023) requires a Tardis subscription not included in the committed dataset.parquet**. Rows where full L2 is unavailable retain price-reference and proxy-depth provenance; net-profit labels are computed on the available route books. See `depth_source`, `depth_sources_used`, and `is_paper_grade_depth` columns for per-row provenance. The phrase "execution-grade" refers to committed VWAP labels with documented route-level depth provenance — not necessarily full three-venue L2 coverage.

2. **Execution cost overestimate**: Net profit computations use BTCUSDT futures band-average prices as a proxy for the BTCUSDC buy side. This overstates price impact for small trades but is directionally correct for stress periods.

3. **Issuer events**: Only USDC SVB events are manually curated. USDT and DAI issuer events are absent.

4. **Kraken checksum**: The Kraken WebSocket book checksum implementation is simplified. For production-grade book reconstruction, implement Kraken's exact checksum spec.

5. **Data gaps**: Binance Vision archives occasionally have missing minutes (exchange maintenance). Gaps are treated as missing values and imputed at model training time.

---

## Intended Use

- **Benchmark**: Evaluate ML models on their ability to predict and economically exploit stablecoin settlement dislocations.
- **Research**: Study the price-to-execution gap and how execution costs reduce apparent arbitrage opportunities.
- **Not intended for**: Live trading, regulatory reporting, or real-time risk management without substantial additional validation.

---

## Maintenance

**Author:** Nigel Li (nl2992@columbia.edu)  
**Affiliation:** Columbia University, Master of Arts in Financial Mathematics (MAFN)  
**Paper:** Submitted to ICAIF 2026
