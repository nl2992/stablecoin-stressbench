# Benchmark Card — Stablecoin StressBench

## Overview

StressBench evaluates stablecoin dislocation models on transaction-cost-aware
outcomes. The benchmark separates three layers that are often conflated:

| Layer | Meaning | Main field |
|---|---|---|
| Optical | A price-basis signal is visible | `cross_quote_basis_*_bps` |
| Executable | The route is profitable after VWAP, fees, and settlement costs | `net_profit_bps_q*` |
| Predictable | A model identifies the executable windows ex ante | experiment outputs |

The headline result is the optical-to-executable gap in the March 2023 USDC/SVB
test window. In the current paper dataset, 34.33% of test minutes exceed a
10 bps primary/max basis threshold and 12.45% exceed the USDC-specific threshold,
while 2.88% exceed the $10K executable-profit threshold after costs. The frozen
baseline table in `results/paper/table_2_price_execution_gap.csv` records the
earlier benchmark-freeze view (35.09% primary/max; 12.65% USDC-specific).

## Tasks

| Task | Label | Primary use |
|---|---|---|
| `basis_usdc_1m_gt10bps` | `label_basis_usdc_1m_gt10bps` | USDC-specific basis forecasting |
| `executable_arb_q10000_5m` | `label_arb_q10000_5m_gt0bps` | $10K executable-arbitrage prediction |
| `executable_arb_q50000_5m` | `label_arb_q50000_5m_gt0bps` | $50K executable-arbitrage prediction |
| `cross_mech_transfer` | meta-label on primary fires | Terra/LUNA-to-SVB transfer |
| `policy_entry` | enter/wait reward | RL timing diagnostic |

## Splits

The current committed Gold dataset has 56,134 rows and 125 columns.

| Split | Event | Rows |
|---|---|---|
| Train | Calm-control windows | 28,776 |
| Validation | Terra/LUNA May 2022 | 11,526 |
| Test | USDC/SVB March 2023 | 15,832 |

The frozen baseline paper tables use the benchmark-freeze view in
`results/paper/table_1_data_coverage.csv` (47,487 rows). This is why some
baseline result files show 15,839 test rows.

## Metrics

Statistical metrics are AUROC, AUPRC, balanced accuracy, and Brier score.
Economic metrics are primary: `net_bps_captured`, `hit_rate_above_cost`,
`n_trades`, `final_pnl_usd`, false-positive cost, and oracle capture.

The no-trade baseline is 0 bps. The oracle trades only hindsight-profitable
windows and is not deployable.

## Main Results

| Result | Value | Source |
|---|---:|---|
| Primary/max basis > 10 bps, SVB test | 34.33% current; 35.09% frozen | `data/gold/dataset.parquet`; `results/paper/table_2_price_execution_gap.csv` |
| USDC-specific basis > 10 bps, SVB test | 12.45% current; 12.65% frozen | `data/gold/dataset.parquet`; `results/paper/table_2_price_execution_gap.csv` |
| Executable threshold at $10K, 1m | 2.88% | `data/gold/dataset.parquet`; `results/paper/table_2_price_execution_gap.csv` |
| Price-to-execution ratio | 12x | `data/gold/dataset.parquet`; `results/paper/table_2_price_execution_gap.csv` |
| Oracle, `basis_usdc_1m_gt10bps` | +161.7 bps | `results/paper/table_4_oracle_gap.csv` |
| Oracle, `executable_arb_q10000_5m` | +224.6 bps | `results/paper/table_4_oracle_gap.csv` |
| Best frozen executable-task model | -42.9 bps | `results/paper/table_4_oracle_gap.csv` |
| Cross-mechanism meta-labeling | +82.5 bps | `results/experiments_addon/meta_labeling_crossmech_results.csv` |
| Conditioned PPO-GRU diagnostic | -29.2 bps | `results/experiments/conditioned_rl_results.csv` |

The frozen executable-arbitrage tasks remain negative for deployable
calm-trained models. The current paper draft adds a separate positive result:
meta-labeling trained on Terra/LUNA transfers to SVB and recovers about half of
the basis-task oracle return.

## Reproduction

Baseline grid:

```bash
python scripts/run_experiments.py --data-dir data/gold
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py --output-dir results/paper/figures
```

Add-ons:

```bash
python scripts/run_robustness_grid.py
python scripts/run_addon_experiments.py
python scripts/run_meta_labeling_crossmech.py
python scripts/run_conditioned_rl.py
python scripts/make_addon_tables.py
```

## Scope

Execution-grade claims are anchored to the SVB test event. Terra/LUNA is used as
a validation and cross-mechanism training event, but its executable labels rely
on proxy depth and are directional rather than Tier-A headline evidence. Other
catalogue events are Tier B/C unless sufficient route-level depth data is added.
