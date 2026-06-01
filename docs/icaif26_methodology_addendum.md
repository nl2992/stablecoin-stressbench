# ICAIF 2026 Methodology Addendum

## 1. Problem

Stablecoin dislocations are often measured using quoted price deviations, but quoted price gaps are not necessarily executable. A dislocation is economically meaningful only if it survives order-book depth, VWAP execution, taker fees, market impact, and settlement frictions.

During the March 2023 USDC/SVB de-peg — the primary test event in this benchmark — 34.3% of current-dataset 1-minute windows showed a primary/max cross-quote basis exceeding 10 bps on price alone (12.45% for the USDC-specific basis). After a full VWAP order-book walk at $10K notional including taker fees and market impact, only 2.88% exceeded the executable-profit threshold. This **12× price-to-execution gap** defines the core measurement challenge.

## 2. Research Question

**Can AI and econometric models identify stablecoin dislocations that remain profitable after realistic execution costs?**

The benchmark answers this with a structured null result and one transfer result: a hindsight oracle earns 161–225 net bps per trade on the test split, confirming profitable windows exist; calm-trained executable-arbitrage models are negative out of sample; Terra/LUNA-trained meta-labeling is positive on the SVB basis task.

## 3. Benchmark Contribution

Stablecoin StressBench introduces an execution-aware benchmark that separates four distinct layers:

| Layer | Measure | Empirical result (test split) |
|---|---|---|
| 1. Price-only dislocation | `|cross_quote_basis_maxabs_bps| > 10 bps` (primary/max basis; USDC-specific: 12.45%) | 34.3% of minutes |
| 2. Gross arbitrage | Raw buy/sell spread | Positive in ~35% of minutes |
| 3. Net executable arbitrage | VWAP walk + fees + impact | 2.88% of minutes at $10K |
| 4. Predictable executable arbitrage | Ex-ante model identifies net-profitable minutes | Calm-trained executable models negative; Terra/LUNA-trained meta-labeling positive on the basis task |

This decomposition is the central methodological contribution. Layers 1–3 are measured from data; layer 4 is the open benchmark challenge.

## 4. Execution-Aware Label

For each 1-minute window at notional size $q$, we reconstruct executable prices
using the available route books and retain row-level depth provenance. Real-L2
and proxy legs are auditable through `depth_source`, `depth_sources_used`, and
`is_paper_grade_depth`. The net-profit label is:

```
net_profit_bps(q) =
    gross_spread_bps(q)           [VWAP buy vs sell cross venues]
  − buy_taker_fee_bps             [maker/taker fee schedule]
  − sell_taker_fee_bps
  − fixed_transfer_cost_bps       [settlement / withdrawal cost proxy]
  − settlement_delay_penalty_bps  [opportunity cost of transfer latency]
```

A window is labelled **executable** (`label_arb_q{N}_{horizon}_gt0bps = 1`) iff `future net_profit_bps(q) > 0` within the prediction horizon.

**Depth provenance guarantee**: net-profit labels carry route provenance in `depth_sources_used`, `is_paper_grade_depth`, and `depth_source`. Real-L2 and proxy legs are separated in the data, and claim scope is governed by `docs/execution_route_coverage.md`.

## 5. Model Evaluation

We evaluate models using both statistical and economic metrics:

### Statistical metrics
- AUROC: classification skill across thresholds
- AUPRC: precision–recall skill on imbalanced positive class
- Brier score: calibration quality
- Balanced accuracy at chosen threshold

### Economic metrics (primary success criteria)
- **net_bps_captured**: mean net profit bps on trades taken
- **hit_rate_above_cost**: fraction of trades with positive net profit
- **false_positive_cost**: mean net profit bps on trades where model predicted positive but outcome was negative
- **n_trades**: trade count (test split)
- **final_pnl_usd**: cumulative P&L at benchmark notional
- **oracle_capture_pct**: `net_bps_captured / oracle_net_bps` — the headline gap metric

The no-trade baseline (`net_bps_captured = 0`) is the economic anchor; models that lose money are worse than abstaining entirely.

### Threshold calibration

The decision threshold is calibrated on the **validation split** by maximizing total net P&L subject to ≥ 25 trades:

```
threshold* = argmax_{t ∈ [0.05, 0.95]}  Σ_{i: proba_i > t} net_profit_bps_i
             subject to |{i : proba_i > t}| ≥ 25
```

This is economically grounded: a strategy must have sufficient trade count to distinguish signal from sampling noise.

## 6. Core Finding

The benchmark establishes three empirical facts:

1. **Price dislocations are frequent.** 34.3% of current SVB test-split minutes exceed 10 bps primary/max cross-quote basis (12.45% for the USDC-specific basis alone).
2. **Executable opportunities are rare.** Only 2.88% survive the $10K VWAP executable-profit threshold (12× price-to-execution ratio).
3. **The oracle gap is large.** The hindsight oracle earns +161 bps; the best ML model loses −49 bps; the gap is 210 bps.

The conclusion is not that stablecoin arbitrage is impossible — the oracle proves otherwise — but that **standard classification and regression models do not yet solve the execution-identification problem**.

## 7. Add-On Contributions

The following extensions supplement the core benchmark result:

### 7a. Robustness over costs and notionals
The price-to-execution gap is recomputed across notional sizes ($10K–$500K), fee regimes (±50% on base fees), settlement penalties (0–10 bps), and prediction horizons (1m, 5m, 15m). This tests whether the core finding depends on specific parameter choices.

### 7b. Expected net-profit regressor
Rather than classifying whether a threshold will be exceeded, the `ExpectedNetProfitRegressor` directly predicts `future_net_profit_bps_q10000`. It trades when predicted net profit exceeds a validation-calibrated floor. This targets the economic objective directly.

### 7c. Uncertainty-aware abstention (future work)
The uncertainty module (`src/stressbench/experiments/uncertainty.py`) implements bootstrap ensemble and quantile regression models that abstain when prediction uncertainty is high. Experiments comparing these abstention strategies against the no-trade baseline are reserved for future work due to the computational cost of bootstrap ensembles.

### 7d. Threshold calibration sensitivity
The primary threshold rule maximizes validation total P&L subject to ≥ 25 trades. Sensitivity to this choice is covered by `results/paper_addon/table_9_threshold_ablation.csv`: fixed thresholds, F1-style rules, and economic-threshold rules do not remove the execution gap.

### 7f. Cross-mechanism meta-labeling and RL diagnostic
The current paper draft adds a transfer test. A meta-labeling filter trained on Terra/LUNA primary-signal windows earns +82.5 bps on the SVB basis task, while a conditioned PPO-GRU trained on the same positive-density window set earns −29.2 bps. This separates the value of stress-like binary supervision from reward-only policy learning under sparse profitable windows.

### 7e. False-positive diagnosis
Feature profiles of true positives vs false positives are compared to explain why models trade bad windows (large basis but insufficient depth, high spread, or unfavorable fee conditions).

## 8. Relation to ICAIF 2026

This work contributes to the following ICAIF topic areas:

- **Financial benchmark construction**: systematic train/validation/test split with event-based design and benchmark-freeze protocol
- **Blockchain and cryptocurrency**: stablecoin de-peg mechanics, cross-venue arbitrage, on-chain settlement frictions
- **Market microstructure**: route-level order-book depth provenance, VWAP execution, spread/depth deterioration during stress
- **Trading and execution**: execution-aware label construction, transaction cost modeling, oracle gap evaluation
- **Validation and calibration of financial AI models**: threshold calibration on economic objectives, out-of-sample robustness, no-lookahead guarantees
- **Uncertainty quantification**: abstention under model uncertainty, confidence-weighted trading signals

## 9. Historical Coverage and Data Tiering

### 9.1 Tier Classification

The benchmark distinguishes three data availability tiers for historical stress events:

| Tier | Description | What is computable | Benchmark use |
|------|-------------|-------------------|--------------|
| **A** | Execution-grade: committed VWAP labels with route-level depth provenance | `net_profit_bps` labels; oracle bound; execution-gap claims within documented scope | Primary benchmark tasks |
| **B** | Price-grade: OHLCV / CEX trades / DEX pool data, no full L2 | Price-basis labels; depeg magnitude; frequency claims | Secondary analysis only |
| **C** | Context-grade: partial data, post-hoc reconstruction only | Historical taxonomy; qualitative mechanism description | Literature framing |

Tier classification is formally encoded in `configs/event_windows_historical.yaml` and
`src/stressbench/history/event_catalog.py` (DataTier enum). Each event carries a
`coverage_score` (0.25/0.50/0.75/1.00) reflecting the completeness of available data.

### 9.2 Which Claims are Execution-Grade vs Price-Grade

**Execution-grade claims (Tier A only):**
- The 12× price-to-execution gap (34.3% price-basis positive vs 2.88% executable-threshold positive at $10K)
- Oracle net bps per trade (161–225 bps)
- All `oracle_capture_pct` values
- Model `net_bps_captured` and `test_final_pnl_usd`
- The core null result: all models produce negative economic value

These claims are anchored exclusively to the USDC/SVB March 10–15 2023 event (`usdc_svb_2023`)
and its recovery window (`usdc_svb_recovery_2023`), which are the only Tier A events with full
L2 data captured during the event.

**Price-grade claims (Tier B, illustrative):**
- Historical depeg magnitudes for Terra/UST, FTX, BUSD, USDT/Curve events
- Frequency of stablecoin stress events over time
- Cross-event comparison of depeg severity and duration
- Mechanism taxonomy (algorithmic vs collateralized vs exchange-specific vs regulatory)

Price-grade claims must be explicitly labeled as "price-grade" in paper text and cannot
be used to make execution-gap or profitability arguments.

### 9.3 The Event Catalogue Approach

The full event catalogue (`docs/stablecoin_stress_event_catalog.md`) documents **18 events**
spanning March 2020 to October 2023 across **7 mechanism classes**:

**Algorithmic / Reflexive (5 events, Tier B–C):**
- FEI Launch Stress Apr 2021 (C): Partial-reserve algorithmic design fails at launch.
- IRON/TITAN Jun 2021 (C): Canonical death spiral. Terminal depeg in <24h.
- MIM/Wonderland Jan–Feb 2022 (B): Governance-confidence shock via DeFi collateral links.
- Terra/UST May 2022 (B): Large-scale algorithmic failure. $40B destruction. Validation split.
- USDD/TRON Jun 2022 (B): UST-contagion sentiment on competing algorithmic stablecoin.

**Fiat-Reserve Bank Shock (2 events, Tier A):**
- **USDC/SVB Mar 10–15 2023 (A): PRIMARY benchmark event. Raw USDC/USD spot peak −1300 bps; cross-quote BTC-route basis peak ~−350 bps (benchmark metric). Binance L2 captured; Coinbase/Kraken historical L2 requires Tardis subscription (not in committed dataset.parquet).**
- USDC Recovery Mar 15–Apr 1 2023 (A partial): Normal conditions; low label density.

**Regulatory / Issuer Winddown (2 events, Tier B–C):**
- Binance USDC→BUSD Conversion Sep 2022 (C): Policy event; 0 bps depeg.
- BUSD Regulatory Feb–Mar 2023 (B): NYDFS order; −30 bps managed dislocation.

**Exchange Credit / Liquidity (3 events, Tier B):**
- Celsius/3AC Contagion Jun 2022 (B): DeFi credit cascade; est. −100 bps CEX contagion.
- HUSD Issuer Failure Aug 2022 (B): Exchange-issuer credit risk; est. −800 bps.
- FTX Collapse Nov 2022 (B): Exchange insolvency; small CEX contagion (−20 bps verified).

**DeFi Pool Imbalance (3 events, Tier B):**
- Curve 3Pool/UST May 2022 (B): 12h leading indicator before UST collapse.
- USDC/DAI Secondary DeFi SVB 2023 (B): MakerDAO PSM co-incident with Tier A event.
- USDT/Curve Jun 2023 (B): Brief pool imbalance. −8 bps. Hours not days.

**Collateral / Liquidation (1 event, Tier B):**
- DAI Black Thursday Mar 2020 (B): Collateral crash; DAI **above** peg (+150 bps verified).

**RWA / Niche Stablecoin (2 events, Tier B–C):**
- Acala aUSD Exploit Aug 2022 (C): 1.28B unbacked aUSD minted; protocol halt.
- USDR RWA Failure Oct 2023 (B): Real-estate illiquidity; est. −5000 bps.

The event taxonomy serves three purposes:
1. **Generalizability framing**: Benchmark tests on the SVB (fiat-reserve) event. Models trained on
   this mechanism class may not generalise to algorithmic, exchange-credit, or DeFi-pool failures.
2. **Mechanism classification**: Each mechanism class represents a structurally distinct failure mode
   with different depth-deterioration profiles, contagion chains, and signal characteristics.
3. **Future data roadmap**: The 18-event catalogue is the target for Tier A expansion as additional
   exchange L2 archives are acquired.

### 9.4 Coverage Score Protocol

A formal coverage_score (0.25–1.00) summarizes data availability:

| Score | Criteria | Typical tier |
|-------|----------|-------------|
| 0.25 | Price data only, or context-grade reconstruction | C |
| 0.50 | Price + trades, OR price + DEX/on-chain | B |
| 0.75 | Price + trades + L2 (partial — not all venues) | A (partial) |
| 1.00 | Price + trades + L2 (Binance confirmed; Coinbase/Kraken partial, Tardis required) + supplementary | A (full) |

The coverage_score is computed by `src/stressbench/history/data_availability.py` and exposed in
`DataAvailabilityProfile`. The predefined profiles in `PREDEFINED_COVERAGE` are the authoritative
source for all table and figure claims about data availability.
