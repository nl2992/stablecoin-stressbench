# Research Methodology — Stablecoin StressBench

## Study in One Paragraph

This paper introduces **Stablecoin StressBench**, a transaction-cost-aware benchmark for evaluating stablecoin dislocation models. In the March 2023 USDC/SVB test event, price-level signals fire often: 34.3% of current-dataset 1-minute windows exceed a 10 bps primary/max cross-quote basis, and 12.45% exceed that threshold on the USDC-specific basis. Only 2.88% exceed the $10K executable-profit threshold after a VWAP order-book walk, taker fees, and market impact. The oracle upper bound confirms profitable windows exist (161–225 net bps on average), but calm-trained executable-arbitrage models remain negative out of sample. The current paper draft adds one positive transfer result: meta-labeling trained on Terra/LUNA earns +82.5 bps on SVB, while a conditioned PPO-GRU with the same positive-label density earns -29.2 bps. The benchmark therefore frames the problem as execution-aware stress transfer, not simple price-signal detection.

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
| Train | Calm control windows | 2022, 2023, 2024 controls | 28,776 |
| Validation | Terra/LUNA de-peg | May 2022 | 11,526 |
| Test | USDC/SVB de-peg and recovery | Mar 2023 | 15,832 |

These are the row counts in the current committed `data/gold/dataset.parquet`.
The frozen baseline paper tables use the filtered benchmark-freeze view in
`results/paper/table_1_data_coverage.csv` (47,487 rows).

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

1. **Book walk**: Traverse the available route books using a VWAP walk at notional sizes $10K, $50K, $100K, $500K. Rows retain `depth_source`, `depth_sources_used`, and `is_paper_grade_depth` so real-L2 and proxy legs can be audited separately.
2. **Fee deduction**: Apply taker fee schedule per venue (configured in `configs/fee_schedules.yaml`).
3. **Market impact**: Depth consumed above the top-of-book is priced at the marginal level reached.
4. **Net profit label**: `net_profit_bps_q{N} = gross_basis_bps - vwap_slippage_bps - fee_bps`. A window is labelled executable (`label_arb_q{N}_{horizon}_gt0bps = 1`) iff `net_profit_bps_q{N} > 0`.

Synthetic kline depth (`depth_source = synthetic_kline`) is separated from real-L2 depth and treated as proxy depth. The Gold pipeline writes `feat_net_profit_1m` when real-L2 books are available and writes `feat_net_profit_1m_proxy` only for CI/smoke-test runs that have no real L2.

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
| 0 bps | 95.86% | 3.34% | 29× |
| 5 bps | 61.99% | 3.03% | 20× |
| 10 bps | 34.33% | 2.88% | 12× |
| 25 bps | 9.54% | 2.62% | 3.6× |

Source: current `data/gold/dataset.parquet` and the paper draft. The frozen
baseline artifact `results/paper/table_2_price_execution_gap.csv` records the
earlier benchmark-freeze view (35.09% primary/max and 12.65% USDC-specific at
the 10 bps threshold).

### Oracle gap (test split, Table 4)

| Task | Oracle net bps | Best ML net bps | Capture % |
|---|---|---|---|
| `basis_usdc_1m_gt10bps` | 161.73 | −49.09 (logistic@price\_plus\_book) | −30.4% |
| `basis_usdc_5m_gt25bps` | 161.73 | −136.04 (gross\_arb@price\_only) | −84.1% |
| `executable_arb_q10000_5m` | 224.57 | −42.90 (threshold\_25bps@price\_only) | −19.1% |
| `executable_arb_q50000_5m` | 146.75 | −112.73 (gross\_arb@price\_only) | −76.8% |

Source: `results/paper/table_4_oracle_gap.csv`.

Frozen calm-trained executable-arbitrage models produce **negative net bps** on the test split. The oracle gap is not a simple model-selection problem; it reflects the execution barrier and the lack of stress-like training examples.

### Cross-mechanism transfer and policy diagnostics

| Method | Training | Test result |
|---|---|---|
| Meta-labeling, `price_plus_book` | Terra/LUNA validation split | +82.45 bps, 397 trades, 50.8% oracle capture |
| Conditioned PPO-GRU | Terra/LUNA primary-signal windows | -29.24 bps, 919 trades |

Source files: `results/experiments_addon/meta_labeling_crossmech_results.csv`
and `results/experiments/conditioned_rl_results.csv`.

---

## Four Paper Claims

1. **Price dislocations are frequent during stress.** In the current SVB test split, 34.3% of 1-minute windows exceed a 10 bps primary/max cross-quote basis (12.45% for the USDC-specific basis). The signal is strong and persistent for days.

2. **Most dislocations are not executable after costs.** Only 2.88% of windows exceed the $10K executable-profit threshold after a VWAP depth walk, taker fees, and market impact. The price-to-execution ratio is 12× at 10 bps and 29× unconditionally.

3. **Calm-trained models classify but do not execute.** The frozen executable-arbitrage models produce negative economic returns on the test split. Microstructure features help in places, but they do not close the oracle gap without stress-like training data.

4. **The oracle gap is real and large.** A hindsight oracle yields 162–225 net bps per trade on average. Terra/LUNA-trained meta-labeling narrows the basis-task gap, but executable-arbitrage prediction remains an open problem.

---

## Benchmark-Freeze Checklist

- [x] Bronze → Silver → Gold pipeline reproduced from raw data
- [x] Depth-source provenance tagged per row (`depth_sources_used`, `is_paper_grade_depth`)
- [x] Route-level depth provenance for net-profit labels (`feat_net_profit_1m`)
- [x] No-lookahead labels verified (`tests/test_no_lookahead.py`)
- [x] Split integrity verified (`tests/test_split_integrity.py`)
- [x] Depth-source filter verified (`tests/test_depth_source_filter.py`)
- [x] Full test suite available via `pytest tests/ -q`
- [x] Full experiment grid committed (`results/experiments/all_results.csv`, 136 rows)
- [x] Paper tables committed (`results/paper/table_{1-4}_*.csv`)
- [x] Paper figures committed in `results/paper/figures/`
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

## Open Problems (future work)

- **Tier A expansion**: Acquire L2 archives for Terra/UST (Binance), FTX stress (Kraken), USDT/Curve to enable cross-mechanism execution-gap comparison.
- **Uncertainty-aware abstention**: Bootstrap ensemble models that abstain when uncertainty is high; computational cost deferred.
- **Meta-labeling beyond one training event**: Terra/LUNA-to-SVB transfer works in the current draft; the next question is whether it survives more mechanisms once Tier-A depth is available.
- **Reactive RL simulation**: The conditioned PPO-GRU result is a diagnostic, not a production execution simulator.
