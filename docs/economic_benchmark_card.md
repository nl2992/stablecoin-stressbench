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
| `logistic@price_plus_book` | −49 bps (basis task, frozen baseline) | Best frozen non-oracle ML result for the basis task |
| `meta_labeling_crossmech@price_plus_book` | +82.5 bps (basis task, add-on) | Terra/LUNA-trained transfer result; not part of the calm-trained baseline grid |

## Oracle

The `NetProfitOracleUpperBound` trades every window where `net_profit_bps_q{N} > 0` in hindsight. It uses future information not available at prediction time. It serves as the theoretical performance ceiling.

**Oracle net bps (test split)**:
- `basis_usdc_1m_gt10bps`: +161.7 bps
- `executable_arb_q10000_5m`: +224.6 bps
- `executable_arb_q50000_5m`: +146.8 bps

## No-Trade Baseline

The `NoTradeBaseline` never trades. It returns exactly 0 net bps and 0 final P&L. Any model that loses money is economically worse than abstaining. This is the **economic lower-risk benchmark**.

## Main Result

**Price dislocations are common; executable opportunities are rare; the oracle is positive; calm-trained executable-task models are negative.**

- 34.3% of current test-split minutes show |primary/max basis| > 10 bps (12.45% for USDC-specific basis); the frozen baseline table records 35.09% and 12.65%
- 2.88% exceed the $10K executable-profit threshold after costs (12× price-to-execution ratio)
- Oracle earns +161 to +225 bps depending on task
- Frozen executable-arbitrage models are negative out of sample
- Cross-mechanism meta-labeling trained on Terra/LUNA is the positive add-on result: +82.5 bps on the SVB basis task

## Dataset

| Split | Event | Minutes | Positive rate (executable 5m $10K) |
|---|---|---|---|
| Train | Calm baseline | 28,776 | 0.0% |
| Validation | Terra/LUNA May 2022 | 11,526 | 2.3% |
| Test | USDC/SVB Mar 2023 | 15,832 | Current dataset uses the same split; headline 2.88% comes from the paper-freeze evaluation view |
| **Total** | | **56,134** | |

The frozen baseline paper tables use the 47,487-row benchmark-freeze view in
`results/paper/table_1_data_coverage.csv`; current add-on scripts read the
committed 56,134-row `data/gold/dataset.parquet`.

## Depth Provenance

Net-profit labels carry route-level depth provenance:
- `real_l2_snapshot`: full-book snapshots when available
- `real_l2_incremental`: reconstructed incremental depth when available
- `synthetic_kline`: proxy depth inferred from OHLCV

Synthetic kline depth is treated as proxy depth, not as full L2. Provenance is auditable per row via `depth_source`, `is_paper_grade_depth`, and `depth_sources_used` columns, and claim scope is documented in `docs/execution_route_coverage.md`.

## Historical Event Catalogue

The benchmark catalogues **18 stress events** (2020–2023) across **7 mechanism classes**.
Execution-grade headline claims are anchored to the USDC/SVB test event.
Source verification status for every event is in `src/stressbench/history/source_verification.py`.

| Tier | N | Example | Execution claims? |
|---|---|---|---|
| A | 2 | USDC/SVB Mar 2023 and recovery/comparator windows | Yes, with claim scope documented in `docs/execution_route_coverage.md` |
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
| Paper figures | `results/paper/figures/` |
| Historical tables (14–19) | `results/paper_addon/table_{14..19}_*.csv` |
| Historical figures (25–28) | `results/paper_addon/figures/figure_{25..28}_*.png` |
| Tests | Run `pytest tests/ -q`; current count depends on the local pytest/plugin version |

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
