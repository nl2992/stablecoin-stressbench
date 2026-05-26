# Economic Benchmark Card — Stablecoin StressBench

## Identity

| Field | Value |
|---|---|
| Benchmark name | Stablecoin StressBench |
| Version | v0.1.0-benchmark-freeze |
| Task type | Binary classification (primary); economic evaluation (primary success criterion) |
| Domain | Stablecoin arbitrage / market microstructure |
| Primary stress event | USDC/SVB de-peg, March 10–15, 2023 |

## Primary Task

**Predict executable stablecoin dislocations after costs.**

A window is a positive example iff a trade placed at the start of the window earns positive net profit within the prediction horizon, after VWAP order-book depth walk, taker fees, and market impact.

| Task name | Label column | Notional | Horizon |
|---|---|---|---|
| `basis_usdc_1m_gt10bps` | `label_basis_usdc_1m_gt10bps` | — | 1 min |
| `executable_arb_q10000_5m` | `label_arb_q10000_5m_gt0bps` | $10,000 | 5 min |
| `executable_arb_q50000_5m` | `label_arb_q50000_5m_gt0bps` | $50,000 | 5 min |

## Primary Metric

**`final_pnl_usd` and `net_bps_captured`**

These measure whether a model strategy is economically viable after costs. A model that loses money is worse than the no-trade baseline regardless of its AUROC.

## Secondary Metrics

| Metric | Type | Use |
|---|---|---|
| AUROC | Statistical | Classification skill across thresholds |
| AUPRC | Statistical | Skill on imbalanced positive class |
| Brier score | Statistical | Calibration quality |
| `hit_rate_above_cost` | Economic | Fraction of trades with net positive outcome |
| `false_positive_cost` | Economic | Mean net bps on incorrectly predicted positive windows |
| `n_trades` | Economic | Trade count (must meet minimum 25 on validation for threshold calibration) |
| `oracle_capture_pct` | Economic | `net_bps_captured / oracle_net_bps` — headline gap metric |

## Baselines

| Model | Expected net bps | Role |
|---|---|---|
| `no_trade` | 0 (by definition) | Economic lower bound; floor for comparison |
| `oracle` | +161 to +225 bps | Theoretical ceiling; not deployable |
| `price_threshold_10bps` | Negative (test split) | Naive rule baseline |
| `logistic@price_plus_book` | −49 bps (test split) | Best non-oracle ML result |

## Oracle

The `NetProfitOracleUpperBound` trades every window where `net_profit_bps_q{N} > 0` in hindsight. It uses future information not available at prediction time. It serves as the theoretical performance ceiling.

**Oracle net bps (test split)**:
- `basis_usdc_1m_gt10bps`: +161.7 bps
- `executable_arb_q10000_5m`: +224.6 bps
- `executable_arb_q50000_5m`: +146.8 bps

## No-Trade Baseline

The `NoTradeBaseline` never trades. It returns exactly 0 net bps and 0 final P&L. Any model that loses money is economically worse than abstaining. This is the **economic lower-risk benchmark**.

## Main Result

**Price dislocations are common; executable opportunities are rare; oracle is positive; deployable models are negative.**

- 35.1% of test-split minutes show |primary/max basis| > 10 bps (12.65% for USDC-specific basis)
- 2.88% are executable at $10K after costs (12× price-to-execution ratio)
- Oracle earns +161 bps; best ML model loses −49 bps; oracle gap = 210 bps
- All non-oracle models produce negative net bps on the SVB test split

## Dataset

| Split | Event | Minutes | Positive rate (executable 5m $10K) |
|---|---|---|---|
| Train | Calm baseline | 20,125 | 0.0% (no real-L2 depth data in train) |
| Validation | Terra/LUNA May 2022 | 11,523 | 2.45% |
| Test | USDC/SVB Mar 2023 | 15,839 | 2.88% |
| **Total** | | **47,487** | |

## Depth Provenance

Net-profit labels use only real L2 order-book depth:
- `real_l2_snapshot`: Coinbase, Kraken full-book snapshots
- `real_l2_incremental`: Tardis incremental updates

Synthetic kline depth (`synthetic_kline`) is excluded from paper-grade calculations. Provenance is auditable per row via `is_paper_grade_depth` and `depth_sources_used` columns.

## Historical Event Catalogue

The benchmark catalogues **18 stress events** (2020–2023) across **7 mechanism classes**.
Execution-grade claims are anchored to the two Tier A events (USDC/SVB 2023).
Source verification status for every event is in `src/stressbench/history/source_verification.py`.

| Tier | N | Example | Execution claims? |
|---|---|---|---|
| A | 2 | USDC/SVB Mar 2023 | Yes — all oracle gap, net bps, model evaluation |
| B | 11 | Terra/UST, FTX, BUSD, USDT/Curve | Price-grade estimates only ("est.") |
| C | 5 | IRON/TITAN, FEI, Acala, Binance conversion | Taxonomy context only |

## Reproducibility

| Item | Status |
|---|---|
| Raw data ingestor | `scripts/pull_data.py` (Binance archive + Tardis modes) |
| Bronze → Silver → Gold pipeline | `scripts/build_features.py` |
| Experiment runner | `scripts/run_experiments.py` |
| All results | `results/experiments/all_results.csv` (136 rows) |
| Paper tables | `results/paper/table_{1-4}_*.csv` |
| Paper figures | `results/paper/figures/figure_{1-5}_*.png` |
| Historical tables (14–19) | `results/paper_addon/table_{14..19}_*.csv` |
| Historical figures (25–28) | `results/paper_addon/figures/figure_{25..28}_*.png` |
| Tests | **187 passing** (`pytest tests/ -q`) |

## Benchmark-Freeze Tag

`v0.1.0-benchmark-freeze` — committed results; baseline files are read-only for add-on work.
All add-on experiments write to `results/experiments_addon/` and `results/paper_addon/`.

## Limitations

A reviewer can understand the benchmark from this card without reading source code. Known limitations:

1. **Single Tier-A primary stress event** — SVB/USDC Mar 2023. The 18-event catalogue documents mechanism diversity; Tier A expansion to other mechanisms requires additional L2 data acquisition.
2. **CEX settlement is proxied** — true on-chain settlement latency not modeled.
3. **Latency simplified** — partial fills and network latency not included.
4. **Feature set** — no sub-minute order flow imbalance.
5. **Mechanism generalisation** — models trained on fiat-reserve bank shock may not generalise to algorithmic or exchange-credit failure modes.
