# Data Card — Stablecoin StressBench

## Overview

StressBench is a transaction-cost-aware benchmark dataset for detecting, forecasting, and economically ranking stablecoin settlement-risk dislocations across centralized and decentralized venues. It covers three stress episodes and three calm control periods between January 2022 and January 2024.

---

## Dataset Summary

| Property | Value |
|---|---|
| **Version** | 0.1.0 |
| **Date range** | 2022-01-10 – 2024-01-21 |
| **Gold rows** | 57,719 (1-minute bars) |
| **Gold columns** | 83 (21 features + 60 labels + 2 meta) |
| **Venues** | Binance spot, Coinbase REST, Kraken WS |
| **Instruments** | 18 (USDC, USDT, DAI, BTC cross-pairs) |
| **On-chain** | Ethereum mainnet ERC-20 transfers (USDC, USDT, DAI) |
| **License** | MIT |

---

## Event Windows

### Test split
| Window | Dates | Reason |
|---|---|---|
| USDC SVB depeg | 2023-03-10 – 2023-03-14 | USDC reserve-bank stress / SVB collapse; peak deviation 350 bps |
| USDC recovery | 2023-03-15 – 2023-03-20 | Recovery window post SVB resolution |

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

## Silver Layer Schema

Normalized per-venue Parquet files (Hive-partitioned: `venue/channel/symbol/date/hour/`).

**Common columns:** `ts_event_ns`, `ts_receive_ns`, `venue_id`, `native_symbol`, `channel`

**Trades:** `price`, `size`, `side`, `trade_id`, `is_buyer_maker`

**Book snapshots:** `bid_price_l{1..5}`, `bid_size_l{1..5}`, `ask_price_l{1..5}`, `ask_size_l{1..5}`

**On-chain transfers:** `block_number`, `tx_hash`, `from_address`, `to_address`, `value`, `token_symbol`

---

## Gold Layer Schema

One row per UTC minute; produced by `scripts/build_features.py`.

### Features (21 columns)

| Column | Description |
|---|---|
| `cross_quote_basis_bps` | USDT–USDC price gap in basis points (Binance spot) |
| `vwap_buy_bps` | VWAP cost of buying USDC with USDT at $10K notional |
| `vwap_sell_bps` | VWAP revenue of selling USDC for USDT at $10K notional |
| `bid_ask_spread_bps` | Best bid–ask spread in basis points |
| `depth_1pct_bps` | Book depth within ±1% of mid |
| `order_imbalance` | (bid_depth – ask_depth) / (bid_depth + ask_depth) |
| `trade_flow_imbalance` | Buy volume minus sell volume over trailing 5 minutes |
| `btc_usdt_mid` | BTC/USDT mid price (regime proxy) |
| `btc_usdc_mid` | BTC/USDC mid price |
| `btc_basis_bps` | BTC cross-venue basis (USDT vs USDC quote) |
| `net_profit_bps_q10000` | Round-trip net profit at $10K after fees and price impact |
| `net_profit_bps_q50000` | Round-trip net profit at $50K |
| `net_profit_bps_q100000` | Round-trip net profit at $100K |
| `net_profit_bps_q500000` | Round-trip net profit at $500K |
| `mint_event` | 1 if a USDC mint event occurred in this minute |
| `burn_event` | 1 if a USDC burn event occurred in this minute |
| `block_lag_proxy` | Estimated Ethereum block confirmation lag (seconds) |
| `fragmentation_hhi` | Herfindahl index of cross-venue price fragmentation |
| `venue_count` | Number of active venues with valid quotes |
| `return_1m` | 1-minute log return of USDC/USDT |
| `volatility_5m` | Rolling 5-minute realized volatility |

### Labels (60 columns)

**Basis labels** (horizon × threshold): `label_basis_{1m,5m,15m,1h}_gt{5,10,25,50}bps` — binary classification; regression targets `label_basis_{horizon}`.

**Arbitrage window labels** (notional × horizon × threshold): `label_arb_q{10000,50000,100000,500000}_{1m,5m,15m}_gt{0,5,10}bps` — 1 if max executable net profit over next H minutes exceeds threshold.

**Regime labels**: `label_regime_{calm,stress,recovery}` — manual event-based regime tags.

**Recovery label**: `label_recovery_within_{1h,4h,24h}` — 1 if basis returns within 5 bps threshold within H.

---

## Row Counts by Split

| Split | Rows | Dates |
|---|---|---|
| Train (3 control windows) | 30,252 | 21 days |
| Validation (Terra/Luna 2022) | 11,568 | 8 days |
| Test (SVB depeg + recovery 2023) | 15,899 | 11 days |
| **Total** | **57,719** | **40 days** |

---

## Known Limitations

1. **Binance-only depth data**: Deep order-book reconstruction uses Binance USDM futures bookDepth for BTC and synthetic 1m klines for stablecoin pairs. Depth data for Coinbase and Kraken is unavailable without a Tardis subscription.

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
**Paper:** Submitted to ICAIF 2024 competition track
