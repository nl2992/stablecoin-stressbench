# ICAIF 2026 Paper Outline

**Title**: Stablecoin StressBench: An Execution-Aware Benchmark for Stablecoin Dislocations

**Venue**: ACM ICAIF 2026 — Milan, Italy, November 14–17, 2026
**Format**: 8 pages max (ACM sigconf double-column), no appendices, no supplementary material
**Submission mode**: Double-blind (anonymous parameter in documentclass)
**Contribution type**: Benchmark / empirical study

---

## Abstract

Stablecoin stress events generate frequent price dislocations, but many apparent arbitrage opportunities disappear once executable depth, VWAP book walking, fees, and market impact are included. We introduce Stablecoin StressBench, a transaction-cost-aware benchmark for evaluating whether AI and econometric models can distinguish optical arbitrage from executable arbitrage. During the March 2023 USDC/SVB test window, 35.1% of 1-minute windows exceed a 10 bps primary/max price-basis threshold (12.65% for the USDC-specific basis), but only 2.88% remain profitable at $10K notional after execution costs — a 12× price-to-execution gap. A hindsight oracle earns positive net bps, but all tested rule-based and ML models lose money out of sample, revealing a large oracle gap (210+ bps). The benchmark reframes stablecoin arbitrage prediction as an execution-barrier problem rather than a simple price-signal detection problem, and provides an open evaluation framework for future models.

*~150 words — fits ACM abstract requirement*

---

## 1. Introduction (~0.5 pages)

**Opening hook**: Stablecoin de-peg events create dramatic price gaps visible on any chart. But chart prices and executable prices are different things.

**Problem statement**:
- Stablecoin stress produces visible price dislocations (cross-quote basis, VWAP quote fragmentation)
- Most apparent arbitrage disappears once order-book depth, VWAP slippage, taker fees, and settlement delays are modeled
- Standard ML models classify price dislocations reasonably well statistically but fail economically

**Gap in existing literature**:
- Cruz et al. (2024) document USDC/SVB price fragmentation but do not evaluate ex-ante predictability
- Hautsch et al. (2018) model blockchain limits to arbitrage theoretically; we measure them empirically in real order-book data
- No existing benchmark provides execution-aware labels with real-L2 depth for stablecoin arbitrage

**Contributions** (bulleted):
1. Execution-aware stablecoin dislocation benchmark with real-L2 VWAP labels
2. Empirical measurement of the price-to-execution gap during the SVB stress event
3. Oracle-gap evaluation: confirms profitable windows exist but are not identified ex ante
4. Open benchmark for future AI model evaluation on this task

---

## 2. Related Work (~0.5 pages)

| Topic | Key references | Our contribution vs. prior work |
|---|---|---|
| Stablecoin de-pegs | Cruz et al. 2024; Kwon et al. 2023 | Execution-aware follow-on: not just price gap but tradable gap |
| Limits to arbitrage | Hautsch et al. 2018; Shleifer & Vishny 1997 | Empirical L2 measurement vs. theoretical framing |
| Market microstructure | Cont & Kukanov 2013; Glosten & Milgrom 1985 | Real-time VWAP walk with depth provenance tracking |
| Financial ML benchmarks | López de Prado 2018; Makarov & Schoar 2020 | Meta-labeling for execution filtering; execution barriers in crypto arbitrage |
| Stablecoin flows | Griffin & Shams 2020 | Microstructure execution focus, not flow manipulation |

---

## 3. Benchmark Design (~1.5 pages)

### 3.1 Data sources
- Binance Vision archive (public, kline aggregates as cross-check; real-L2 where available)
- Tardis.dev real-time L2 snapshots: Coinbase, Kraken (real_l2_snapshot, real_l2_incremental)
- Instruments: BTC-USDC, BTC-USDT, ETH-USDC, ETH-USDT across venues
- Date range: Aug 2022 – Apr 2023

### 3.2 Event-based train/validation/test split

| Split | Event | Period | Minutes |
|---|---|---|---|
| Train | Calm control baseline | Aug 2022 – Jan 2023 | 20,125 |
| Validation | Terra/LUNA de-peg | May 2022 | 11,523 |
| Test | USDC/SVB de-peg | Mar 10–15, 2023 | 15,839 |

No information from the test split is used in training or calibration. No-lookahead is formally verified (see §3.4).

### 3.3 Historical event catalogue and data tiering

The benchmark catalogues **18 stress events** (2020–2023) across **7 mechanism classes**:
algorithmic/reflexive (N=5), fiat-reserve bank shock (N=2, Tier A), regulatory winddown (N=2),
exchange credit/liquidity (N=3), DeFi pool imbalance (N=3), collateral/liquidation (N=1),
RWA/niche stablecoin (N=2).

Tier A (execution-grade, N=2): real L2 depth; VWAP labels and oracle gap computable.
Tier B (price-grade, N=11): OHLCV/DEX; basis estimates only.
Tier C (context-grade, N=5): taxonomy only; no numerical claims.

All execution-aware claims are anchored to Tier A. Tier B figures use "est." notation.

### 3.4 Feature sets
Four nested sets from narrowest to broadest:
- `price_only`: 5 cross-quote basis and stablecoin deviation columns
- `price_plus_book`: + 7 microstructure columns (spread, depth, imbalance, volume)
- `price_book_frag`: + 3 cross-venue fragmentation columns
- `price_book_settle`: + 7 on-chain settlement proxies

### 3.5 Integrity guarantees
- No-lookahead: labels constructed by `join_asof` at `t + horizon`; formally tested
- Split integrity: no cross-split overlap; tested against `configs/event_windows.yaml`
- Depth provenance: `depth_source ∈ {real_l2_snapshot, real_l2_incremental}` for net-profit labels; `is_paper_grade_depth` flag per row; tested
- Source verification: `use_in_paper=True` claims require `verified=True` in `source_verification.py`; enforced by `test_historical_layer.py`

---

## 4. Execution-Aware Labels (~1 page)

### 4.1 Cross-quote basis labels
For horizon $h \in \{1\text{m}, 5\text{m}, 15\text{m}, 1\text{h}\}$ and threshold $\tau$:
```
label_basis_usdc_{h}_gt{τ}bps(t) = 1 iff |basis_usdc(t+h)| > τ
```
Primary task: `basis_usdc_1m_gt10bps` (SVB de-peg event signal).

### 4.2 VWAP execution walk
For each minute and notional $q \in \{10\text{K}, 50\text{K}, 100\text{K}, 500\text{K}\}$:
1. Walk the real-L2 bid/ask ladder at each venue to fill $q$ notional
2. Compute VWAP buy and sell prices cross-venue
3. Deduct taker fee and model market impact as depth consumed above top-of-book

### 4.3 Net executable arbitrage labels
```
net_profit_bps(q) = gross_spread_bps(q) − taker_fees_bps − settlement_penalty_bps
label_arb_q{N}_{h}_gt0bps(t) = 1 iff future net_profit_bps(q) > 0 within h
```
Primary economic task: `executable_arb_q10000_5m`.

### 4.4 Oracle upper bound
The oracle trades every window where `net_profit_bps > 0` in hindsight. It sets the theoretical performance ceiling and is not deployable.

---

## 5. Models and Evaluation (~1 page)

### 5.1 Model stack

**Rule baselines**: NoTrade (economic anchor), PriceBasis10bps, PriceBasis25bps, GrossArbThreshold

**Statistical baselines**: LastValue (AR0), RollingMean, AR1

**ML models**: Logistic (L2), Lasso (L1), Random Forest, XGBoost, LightGBM

**Add-on**: ExpectedNetProfitRegressor — directly predicts `future_net_profit_bps_q10000` and trades when prediction exceeds a calibrated floor

### 5.2 Threshold calibration
Threshold chosen on validation by maximizing total net P&L subject to ≥ 25 trades.
Ablation confirms result is robust across threshold rules (fixed 0.5/0.7, F1, mean bps, total P&L with different minimums).

### 5.3 Evaluation protocol
Primary: net_bps_captured, hit_rate_above_cost, final_pnl_usd, oracle_capture_pct
Secondary: AUROC, AUPRC, Brier score, n_trades

---

## 6. Results (~2 pages)

### 6.1 Price-to-execution gap (Figure 2, Table 2)

| Threshold | Price signal | Executable ($10K) | Ratio |
|---|---|---|---|
| 0 bps | 97.15% | 3.34% | 29× |
| 10 bps | 35.09% (primary/max basis); 12.65% (USDC only) | 2.88% | 12× |
| 25 bps | 9.63% | 2.62% | 3.7× |

*The gap is large and persistent across thresholds.*

### 6.2 Signal waterfall (Figure 11)

```
Total test minutes:           15,839  (100%)
Price dislocation (>10 bps):   5,560  (35.1%, primary/max basis); 2,004 (12.65%, USDC-specific)
Executable at $10K:              456  (2.88%)
Best model trades:              ≈XXX
Best model profitable trades:   ≈XXX
Oracle profitable trades:        351  (2.22%)
```

*The waterfall shows where value is lost at each stage.*

### 6.3 Oracle gap (Figure 5, Table 4)

| Task | Oracle | Best ML | Gap |
|---|---|---|---|
| basis_usdc_1m_gt10bps | +161.7 bps | −49.1 bps | 211 bps |
| executable_arb_q10000_5m | +224.6 bps | −42.9 bps | 267 bps |

*All non-oracle models are economically negative on the test split.*

### 6.4 Robustness (Figures 8, 9)
The price-to-execution gap persists across all tested cost assumptions (forward rolling max of adjusted net profit over each horizon window; fee and settlement parameters genuinely affect executable percentages):
- Across notionals ($10K–$500K): smaller notionals are easier to fill but opportunity count drops at $50K+
- Across fee regimes (base → high fee, −2 bps adjustment): 5m executable drops from 5.64% → 5.46%; ratio rises from 2.24× → 2.32×
- Across settlement penalties (0 → 10 bps): 5m executable drops from 5.64% → 4.88%; ratio rises to 2.59×
- Combined worst case (high fee + 10 bps settlement): gap is monotonically larger than base case

### 6.5 Expected net-profit model
Directly predicting net_profit_bps improves calibration vs classification models but does not cross into positive net bps territory on the test split.

---

## 7. Limitations and Future Work (~0.5 pages)

**Limitations**:
- Single Tier-A stress event (SVB Mar 2023); 18-event catalogue documents mechanism diversity but other events lack L2 depth for execution-grade analysis
- Models trained on fiat-reserve bank shock may not generalise to algorithmic, exchange-credit, or DeFi-pool failure modes
- CEX internal book settlement is proxied; true on-chain settlement latency not modeled
- Partial fills and latency-induced execution slippage are simplified
- Feature set does not include order flow imbalance at sub-minute resolution

**Future work**:
- Tier A expansion: acquire L2 archives for Terra/UST, FTX stress, USDT/Curve events
- Expected net-profit models with uncertainty-aware abstention
- Confidence-weighted position sizing (scale notional by predicted probability)
- Graph-based venue fragmentation features during stress
- Cross-mechanism model generalisation testing

---

## 8. Conclusion (~0.25 pages)

Stablecoin StressBench introduces a benchmark for execution-aware stablecoin arbitrage prediction. The central finding is that price-only signals dramatically overstate the frequency of profitable arbitrage: the price-to-execution ratio is 12× at 10 bps during the SVB crisis. A hindsight oracle confirms profitable windows exist (161–225 net bps), but every tested ML and rule-based model loses money out of sample, revealing a 200+ bps oracle gap. This benchmark establishes stablecoin arbitrage prediction as an execution-barrier problem rather than a signal-detection problem, and provides an open evaluation framework for future models to measure their progress against.

---

## Figure Budget (8-page ACM paper)

| Figure | Content | Size | Priority |
|---|---|---|---|
| Figure 1 | USDC basis event study | Half page | Must include |
| Figure 2 | Price-to-execution gap bar chart | Quarter page | Must include |
| Figure 5 | Oracle gap grouped bars | Quarter page | Must include |
| Figure 11 | Signal waterfall | Quarter page | Must include |
| Figure 3 | Spread/depth deterioration (context) | Quarter page | Optional |
| Figure 8 | Robustness by notional | Quarter page | Optional |

*Total for must-include: ~1.5 pages. Leaves room for tables.*

## Table Budget

| Table | Content | Rows | Priority |
|---|---|---|---|
| Table 1 | Dataset coverage | 6 | Must include |
| Table 2 | Price-to-execution gap | 4 rows × 6 cols | Must include |
| Table 3 | Model ablation (abridged) | 8 rows | Must include |
| Table 4 | Oracle gap | 4 rows | Must include |

---

## References (target ~15–20 citations)

1. Cruz et al. (2024) — USDC/SVB de-peg analysis
2. Hautsch, Scheuch & Voigt (2018) — blockchain limits to arbitrage
3. Kwon, Minegishi & Nishi (2023) — Terra/LUNA de-peg mechanics
4. Griffin & Shams (2020) — Tether stablecoin flows
5. Cont & Kukanov (2013) — optimal order routing
6. Glosten & Milgrom (1985) — adverse selection in market microstructure
7. López de Prado (2018) — Advances in Financial Machine Learning (meta-labeling)
8. Shleifer & Vishny (1997) — limits to arbitrage
9. Gromb & Vayanos (2010) — limits of arbitrage: state of the theory
10. Makarov & Schoar (2020) — trading and arbitrage in cryptocurrency markets
11. Vidal-Tomàs, Briola & Aste (2023) — FTX downfall and Binance consolidation
13. Pennington et al. (2022) — algorithmic stablecoin failure modes
14. Adams et al. (2021) — Uniswap v3 and AMM liquidity
15. Prokopenko et al. (2023) — on-chain settlement latency under stress
