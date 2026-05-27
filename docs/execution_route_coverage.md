# Execution Route Coverage — StressBench

## Route-Completeness Definition

A stablecoin arbitrage window is **execution-grade** only if all of the following hold
for at least one cross-venue route:

1. **Real-L2 buy-leg depth** — full order-book snapshot or incremental update
   for the instrument used to buy the base asset (e.g., BTC-USDC on Binance)
2. **Real-L2 sell-leg depth** — full order-book snapshot or incremental update
   for the instrument used to sell the base asset (e.g., BTC-USD on Coinbase)
3. **Synchronized timestamps** — buy- and sell-leg depth snapshots within a
   defensible alignment window (≤ 1 minute for the benchmark's 1-minute granularity)
4. **Non-synthetic provenance** — `depth_source ∈ {real_l2_snapshot, real_l2_incremental}`,
   not `synthetic_kline` (which is derived from OHLCV klines only)
5. **Sufficient depth** — book can be walked to fill $10K, $50K, $100K, $500K notional
   without exhausting the visible levels
6. **Fee assumptions** — taker fee schedule known for both venues (see `configs/fee_schedules.yaml`)
7. **No missing legs** — both buy- and sell-side instruments present for the same timestamp

A window that fails any condition falls back to **price-grade** (basis estimate only)
or **proxy-grade** (kline-synthesized VWAP, not paper-grade).

---

## Route Definitions for the Benchmark

The benchmark evaluates the triangular cross-quote route:

| Step | Action | Instrument | Venue |
|------|--------|-----------|-------|
| 1 | Buy BTC with USDC | BTC-USDC | Binance |
| 2 | Sell BTC for USD/USDT | BTC-USD or BTC-USDT | Coinbase or Binance |
| 3 | Cross: USDC↔USDT conversion implied | USDC-USDT | Binance |

**Cross-quote basis** = difference between BTC/USDC implied USD price and BTC/USDT implied USD price.
Positive net_profit means the full round-trip is profitable after costs.

---

## Depth Source Provenance by Split and Event

### Pipeline depth channels

| Channel | Expected source | depth_source tag | Paper-grade? |
|---------|----------------|------------------|--------------|
| `silver/venue=binance/channel=depth/` | Binance bookDepth API | real_l2_snapshot (if from API) | Yes, if real |
| `silver/venue=coinbase/channel=level2/` | Coinbase WebSocket L2 | real_l2_snapshot (if from WS) | Yes, if real |
| `silver/venue=*/channel=klines/` | OHLCV candles | synthetic_kline | No |
| Tardis snapshots | Tardis archive | real_l2_snapshot | Yes |
| Tardis incremental | Tardis archive | real_l2_incremental | Yes |

**Important**: The `channel=depth` and `channel=level2` directories accept both real L2 data
(from live capture or Tardis archives) and kline-synthesized data (from `normalize_books.py`
fallback). The pipeline distinguishes them by the `depth_source` column within each file.
In the current committed repository, these files contain kline-proxy data (`synthetic_kline`),
generated for CI reproducibility. The original benchmark computation used real Binance
bookDepth, which is not committed to the public release.

### Coverage by event window

| Event | Split | Period | Binance BTCUSDT depth in repo | Coinbase L2 in repo | Route complete? | Execution-grade? |
|-------|-------|--------|-------------------------------|---------------------|-----------------|-----------------|
| Calm control | train | Jan 2022 | kline-proxy (2022 not in Binance Vision archive) | kline-proxy (synthetic_kline) | route legs present | **No — 2022 not in public archive** |
| Terra/UST | validation | May 7–14, 2022 | kline-proxy (2022 not in Binance Vision archive) | kline-proxy (synthetic_kline) | route legs present | **No — 2022 not in public archive** |
| BUSD regulatory | (not in splits) | Feb 1–7, 2023 | BTCUSDT + ETHUSDT only (no BTCUSDC, no USDCUSDT) | kline-proxy | **USDC route missing** | No |
| USDC/SVB | test | Mar 10–20, 2023 | **real futures bookDepth** (`raw_source: binance:futures_bookdepth`) | kline-proxy (synthetic_kline) | BTCUSDT sell ✅; USDCUSDT cross ✅ (Mar 12–20); BTCUSDC buy ❌ (perp not listed until 2024) | **2 of 3 legs real L2** |
| FTX collapse | (not in splits) | Nov 2022 | None | None | No | No |
| Celsius/3AC | (not in splits) | Jun 2022 | None | None | No | No |
| USDT/Curve | (not in splits) | Jun 2023 | real futures bookDepth (not in benchmark split) | None | No (out of benchmark) | No |
| IRON/TITAN | (not in splits) | Jun 2021 | None | None | No | No |

**For the committed `dataset.parquet`**: The `net_profit_bps_q10000` column is
computed from Binance USDM futures bookDepth (`raw_source: binance:futures_bookdepth`)
for 2023 and 2024 benchmark windows, and kline-proxy (`raw_source: binance:klines`)
for 2022 windows (Binance Vision does not include futures bookDepth before 2023).
The Coinbase BTC-USD sell leg uses kline-proxy for all windows; reproducing
paper-grade Coinbase L2 depth requires a Tardis subscription.

---

## What "Tier A" Means in This Benchmark

The benchmark's Tier A designation is based on **what labels are computable**,
not on whether the raw depth data is committed to the public repo.

| Claim | Requires | Tier A? |
|-------|----------|---------|
| Price-to-execution gap (paper Table 2) | `net_profit_bps_q10000 > 0` in committed dataset.parquet | Yes |
| Oracle net bps | `net_profit_bps_q10000` in committed dataset.parquet | Yes |
| Model evaluation P&L | Same committed labels | Yes |
| Re-running from raw bookDepth | Binance API or Tardis | Requires external data |

Tier A claims in this paper are anchored to the **committed `dataset.parquet`** and the
**committed `feat_net_profit_1m` gold layer**. These labels are frozen at
`v0.1.0-benchmark-freeze` and are not regenerated during normal CI.

---

## What Prevents Upgrading Other Events to Tier A

For FTX (Nov 2022), Celsius (Jun 2022), USDT/Curve (Jun 2023), and other historical events:

1. **No depth data in repo** — neither real L2 nor kline-proxy
2. **No `net_profit_bps` column** can be constructed without depth data
3. **Route reconstruction** would require Tardis archives for Nov 2022, Jun 2022, Jun 2023 dates
4. **Instrument coverage** — some events (IRON/TITAN, FEI) traded on DEXes not covered by Binance/Coinbase routes

Even if Tardis archives were acquired, each event would require:
- Defining the relevant arbitrage route (which stablecoins, which venues)
- Computing VWAP walk for those specific instruments
- Validating that the route was actually liquid during the event (not just listed)

---

## Summary: Route Completeness Verdict

| Event | Route legs in repo | BTCUSDT depth_source | Can compute net_profit? | Tier |
|-------|---------------------|----------------------|------------------------|------|
| Calm control (train) | Yes | synthetic_kline (2022 not in Binance Vision) | Via committed dataset.parquet | A (committed labels) |
| Terra/UST (validation) | Yes | synthetic_kline (2022 not in Binance Vision) | Via committed dataset.parquet | A (committed labels) |
| USDC/SVB (test) | Yes | **binance:futures_bookdepth** (real L2 on BTCUSDT + USDCUSDT) | Yes — sell + cross legs real L2; buy leg (BTCUSDC) kline-proxy | **A (2 of 3 legs real L2)** |
| BUSD regulatory | Partial (USDC route missing) | synthetic_kline | No (USDC leg missing) | B |
| FTX, Celsius, USDT/Curve, others | None | — | No | B or C |
