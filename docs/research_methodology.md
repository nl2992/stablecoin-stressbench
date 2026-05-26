# Research Methodology — Stablecoin StressBench

## Study in One Paragraph

This paper introduces **Stablecoin StressBench**, a transaction-cost-aware benchmark for evaluating machine learning models on stablecoin dislocation detection and arbitrage prediction. Using the March 2023 USDC/SVB de-peg as the primary test event, we show that while price-level signals fire frequently during stress (35.1% of 1-minute windows exceed 10 bps primary/max cross-quote basis; 12.65% for the USDC-specific basis), almost none survive a realistic execution filter: only 2.88% of those windows remain profitable at $10K notional after a full VWAP order-book walk inclusive of taker fees and market impact. The oracle upper bound confirms profitable windows exist (161–225 net bps on average), but every ML and rule-based model tested produces negative net bps on the test split, yielding a large and persistent **oracle gap**. The benchmark thus functions as a rigorous null result: it establishes that standard classification and regression models, even with rich microstructure features, do not yet solve the execution-barrier problem in stablecoin arbitrage.

---

## Prior Literature

| Paper | Contribution | Relation to this work |
|---|---|---|
| Cruz et al. (2024) | Documents USDC/SVB de-peg mechanics and cross-venue price fragmentation | Primary event motivation; calibrates our stress window (Mar 10–15 2023) |
| Hautsch, Scheuch & Voigt (2018) | Blockchain-specific limits to arbitrage: latency, settlement finality, gas cost | Justifies VWAP + fee + settlement frictions in our execution label |
| Kwon, Minegishi & Nishi (2023) | Terra/LUNA de-peg dynamics; algorithmic stablecoin collapse | Motivates the validation split (May 2022 Terra event) |
| Griffin & Shams (2020) | Tether/BTC flows as price manipulation signal | Background on stablecoin market microstructure; informs feature design |
| Cont & Kukanov (2013) | Optimal order routing across fragmented markets | Theoretical basis for VWAP depth-walk as execution cost measure |
| López de Prado (2018) | Meta-labeling framework: separating primary signal from execution filtering | Motivates MetaLabelingFilter design and two-stage evaluation |
| Makarov & Schoar (2020) | Persistent crypto price gaps across venues; limits to triangular arbitrage | Establishes baseline expectation for execution barriers in crypto markets |

---

## Event-Study Design

### Dataset timeline

| Split | Event | Period | Minutes |
|---|---|---|---|
| Train | Calm control (pre-stress baseline) | ~Aug 2022 – Jan 2023 | 20,125 |
| Validation | Terra/LUNA de-peg | May 2022 | 11,523 |
| Test | USDC/SVB de-peg | Mar 10–15 2023 | 15,839 |

Splits are non-overlapping by construction (verified by `tests/test_split_integrity.py`). The validation event is used exclusively for threshold calibration and model selection; no test-split information leaks into any training or calibration step.

### No-lookahead guarantee

All labels are constructed by joining the basis/profit series at `t + horizon` back to each row at time `t` using a 90-second `join_asof` tolerance. Rows within one horizon of the split boundary receive null labels (verified by `tests/test_no_lookahead.py`).

---

## Three Cross-Quote Basis Columns

| Column | Definition | Use |
|---|---|---|
| `cross_quote_basis_usdc_bps` | USDC mid-price deviation from par, bps | Primary SVB event signal; drives `label_basis_usdc_*` |
| `cross_quote_basis_usdt_bps` | USDT mid-price deviation from par, bps | Control series |
| `cross_quote_basis_maxabs_bps` | `max(|usdc|, |usdt|)` with sign of argmax | Generic stress detector; drives `label_basis_maxabs_*` |
| `cross_quote_basis_primary_bps` | USDC with maxabs fallback for nulls | Backward-compat label for `label_basis_*` (existing) |

---

## VWAP Execution-Label Methodology

The executable arbitrage label is the central methodological innovation. For each 1-minute window:

1. **Book walk**: Traverse real-L2 order-book snapshots (`depth_source ∈ {real_l2_snapshot, real_l2_incremental}`) using a VWAP walk at notional sizes $10K, $50K, $100K, $500K.
2. **Fee deduction**: Apply taker fee schedule per venue (configured in `configs/fees.yaml`).
3. **Market impact**: Depth consumed above the top-of-book is priced at the marginal level reached.
4. **Net profit label**: `net_profit_bps_q{N} = gross_basis_bps - vwap_slippage_bps - fee_bps`. A window is labelled executable (`label_arb_q{N}_{horizon}_gt0bps = 1`) iff `net_profit_bps_q{N} > 0`.

Synthetic kline depth (`depth_source = synthetic_kline`) is never used in paper-grade net-profit calculations. The Gold pipeline writes `feat_net_profit_1m` only when real-L2 data is available; a proxy file (`feat_net_profit_1m_proxy`) is written for CI/smoke-test runs only.

Depth-source provenance is auditable per row via `depth_sources_used`, `is_paper_grade_depth`, and `depth_source` columns (verified by `tests/test_depth_source_filter.py`).

---

## Model Stack

### Rule baselines

| Model | Description |
|---|---|
| `no_trade` | Never trades. Economic anchor (0 bps, 0 trades). |
| `price_threshold_10bps` | Trade when `|cross_quote_basis_primary_bps| > 10`. |
| `price_threshold_25bps` | Trade when `|cross_quote_basis_primary_bps| > 25`. |
| `gross_arb_threshold` | Trade when gross basis exceeds estimated fee floor. |

### Statistical baselines

| Model | Description |
|---|---|
| `last_value` | Persistence (AR0). |
| `rolling_mean` | 10-period rolling mean. |
| `ar1` | AR(1) regression label. |

### ML models

| Model | Description |
|---|---|
| `logistic` | L2-regularized logistic regression. |
| `lasso` | L1-regularized logistic regression. |
| `ridge` | Ridge-regularized logistic regression. |
| `rf` | Random forest (100 trees, max depth 6). |
| `xgb` | XGBoost (100 rounds, early stopping on validation). |
| `lgbm` | LightGBM (100 rounds, early stopping on validation). |

### Oracle

`NetProfitOracleUpperBound` — trades every window where `net_profit_bps_q{N} > 0` in hindsight. Sets the theoretical ceiling; not deployable.

---

## Feature Sets

| Set | Columns | Purpose |
|---|---|---|
| `price_only` | 5 basis/deviation cols | Isolate pure price signal |
| `price_plus_book` | + 7 microstructure cols | Add spread, depth, imbalance, volume |
| `price_book_frag` | + 3 fragmentation cols | Add cross-venue dispersion |
| `price_book_settle` | + 7 on-chain settlement proxies | Add settlement frictions |

Column definitions: see `src/stressbench/experiments/feature_sets.py`.

---

## Threshold Calibration

For each (task × feature_set × model) triple, the decision threshold is chosen on the **validation split** by maximizing **total net P&L** (not mean bps per trade), subject to a minimum of 25 trades:

```
threshold* = argmax_{t ∈ [0.05, 0.95]} Σ_i net_profit_bps_i · 1[proba_i > t]
             subject to |{i : proba_i > t}| ≥ 25
```

This objective is economically grounded: a strategy with high mean bps but few trades has lower total utility than one with moderate mean bps and many trades.

---

## Core Empirical Results

### Price-to-execution gap (test split, Table 2)

| Threshold | Price signal (% windows) | Executable at $10K (% windows) | Ratio |
|---|---|---|---|
| 0 bps | 97.15% | 3.34% | 29× |
| 5 bps | 63.22% | 3.03% | 21× |
| 10 bps | 35.09% | 2.88% | 12× |
| 25 bps | 9.63% | 2.62% | 3.7× |

Source: `results/paper/table_2_price_execution_gap.csv`.

### Oracle gap (test split, Table 4)

| Task | Oracle net bps | Best ML net bps | Capture % |
|---|---|---|---|
| `basis_usdc_1m_gt10bps` | 161.73 | −49.09 (logistic@price\_plus\_book) | −30.4% |
| `basis_usdc_5m_gt25bps` | 161.73 | −136.04 (gross\_arb@price\_only) | −84.1% |
| `executable_arb_q10000_5m` | 224.57 | −42.90 (threshold\_25bps@price\_only) | −19.1% |
| `executable_arb_q50000_5m` | 146.75 | −112.73 (gross\_arb@price\_only) | −76.8% |

Source: `results/paper/table_4_oracle_gap.csv`.

All non-oracle models produce **negative net bps** on the test split. The oracle gap is not a model-selection problem — it is an execution-barrier problem.

---

## Four Paper Claims

1. **Price dislocations are frequent during stress.** In the SVB test split, 35.1% of 1-minute windows exceed a 10 bps primary/max cross-quote basis (12.65% for the USDC-specific basis). The signal is strong and persistent for days.

2. **Most dislocations are not executable after costs.** Only 2.88% of windows remain profitable at $10K notional after a VWAP depth walk, taker fees, and market impact. The price-to-execution ratio is 12× at 10 bps and 29× unconditionally.

3. **ML models classify but do not execute.** All tested models achieve above-chance AUROC on the classification task but produce negative economic returns on the test split. Microstructure features do not close the gap: `price_only` and `price_plus_book` perform similarly.

4. **The oracle gap is real and large.** A hindsight oracle yields 162–225 net bps per trade on average. The gap between oracle and best model is 200+ bps, establishing that profitable windows exist but that ex-ante identification is an open problem.

---

## Benchmark-Freeze Checklist

- [x] Bronze → Silver → Gold pipeline reproduced from raw data
- [x] Depth-source provenance tagged per row (`depth_sources_used`, `is_paper_grade_depth`)
- [x] Real-L2-only net-profit labels (`feat_net_profit_1m`)
- [x] No-lookahead labels verified (`tests/test_no_lookahead.py`)
- [x] Split integrity verified (`tests/test_split_integrity.py`)
- [x] Depth-source filter verified (`tests/test_depth_source_filter.py`)
- [x] **198 tests passing** (`pytest tests/ -q`)
- [x] Full experiment grid committed (`results/experiments/all_results.csv`, 136 rows)
- [x] Paper tables committed (`results/paper/table_{1-4}_*.csv`)
- [x] Paper figures committed (`results/paper/figures/figure_{1-5}_*.png`)
- [x] Tag: `v0.1.0-benchmark-freeze`

---

## Historical Event Catalogue

Stablecoin stress is not a single phenomenon. The benchmark catalogues **18 events** across
**7 mechanism classes** (2020–2023):

| Class | N | Representative event | Tier |
|---|---|---|---|
| Algorithmic / Reflexive | 5 | Terra/UST May 2022 | B |
| **Fiat-Reserve Bank Shock** | **2** | **USDC/SVB Mar 2023** | **A** |
| Regulatory Winddown | 2 | BUSD Feb 2023 | B |
| Exchange Credit / Liquidity | 3 | FTX Nov 2022 | B |
| DeFi Pool Imbalance | 3 | USDT/Curve Jun 2023 | B |
| Collateral / Liquidation | 1 | DAI Black Thursday 2020 | B |
| RWA / Niche Stablecoin | 2 | USDR Oct 2023 | B |

**Claim permissions by tier:**
- Tier A only: execution-gap claims, oracle bps, model net bps, oracle capture %
- Tier B: price-grade estimates ("est.") for depeg magnitude and frequency only
- Tier C: taxonomy and mechanism context; no numerical claims

Source verification: `src/stressbench/history/source_verification.py` (26 records; `use_in_paper=True` requires `verified=True`).
Price-grade feature summaries: `results/paper_addon/table_17_historical_price_grade_summary.csv`.
Mechanism taxonomy: `results/paper_addon/table_18_mechanism_taxonomy_summary.csv`.

## Benchmark-Freeze Checklist

- [x] Bronze → Silver → Gold pipeline reproduced from raw data
- [x] Depth-source provenance tagged per row (`depth_sources_used`, `is_paper_grade_depth`)
- [x] Real-L2-only net-profit labels (`feat_net_profit_1m`)
- [x] No-lookahead labels verified (`tests/test_no_lookahead.py`)
- [x] Split integrity verified (`tests/test_split_integrity.py`)
- [x] Depth-source filter verified (`tests/test_depth_source_filter.py`)
- [x] **198 tests passing** (`pytest tests/ -q`)
- [x] Full experiment grid committed (`results/experiments/all_results.csv`, 136 rows)
- [x] Paper tables committed (`results/paper/table_{1-4}_*.csv`)
- [x] Paper figures committed (`results/paper/figures/figure_{1-5}_*.png`)
- [x] Add-on tables committed (`results/paper_addon/table_{5,8,8b,9,10,14-19}_*.csv`)
- [x] Historical catalogue: 18 events, 7 mechanism classes, source verification registry
- [x] Tag: `v0.1.0-benchmark-freeze`

## Open Problems (future work)

- **Tier A expansion**: Acquire L2 archives for Terra/UST (Binance), FTX stress (Kraken), USDT/Curve to enable cross-mechanism execution-gap comparison.
- **Uncertainty-aware abstention**: Bootstrap ensemble models that abstain when uncertainty is high; computational cost deferred.
- **Threshold ablation**: Full multi-rule ablation (fixed 0.5/0.7, validation F1, mean bps) to confirm null result is threshold-rule-independent.
- **Meta-labeling on validation split**: Current meta-labeling fails due to zero profitable primary-signal windows in the calm training split; re-train on Terra/LUNA validation data.
