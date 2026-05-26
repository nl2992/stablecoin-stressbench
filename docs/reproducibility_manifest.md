# Reproducibility Manifest — Stablecoin StressBench

**Version:** v0.1.0-benchmark-freeze  
**Date:** 2026-05-26  
**Author:** Nigel Li (nl2992@columbia.edu)

This manifest records every script that contributes to the paper results, the order in which to run them, and which output files each produces. A reviewer should be able to reproduce all tables and figures by following this sequence.

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Python ≥ 3.11 required. All dependencies are pinned in `pyproject.toml`.

---

## Benchmark-Freeze Tag

The core benchmark results were frozen at commit `v0.1.0-benchmark-freeze`. All add-on work writes only to `results/experiments_addon/` and `results/paper_addon/` — baseline files in `results/paper/` and `results/experiments/` are never modified by add-on scripts.

---

## Step-by-Step Reproduction

### Step 0 — Data

The gold-layer dataset is checked into the repository at `data/gold/dataset.parquet`.  
No raw data download is required for reproducing paper results.

| File | Description |
|---|---|
| `data/gold/dataset.parquet` | 47,487 rows × 125 columns; train/validation/test split column |

To rebuild from raw sources (optional; requires API keys in `.env`):

```bash
python scripts/archive_to_bronze.py    # pull Binance Vision archives
python scripts/build_silver.py         # normalize to Silver Parquet
python scripts/build_features.py       # build Gold feature tables
python scripts/patch_dataset.py        # add derived columns + labels
```

### Step 1 — Baseline experiment grid

```bash
python scripts/run_experiments.py
```

**Outputs** (written to `results/experiments/`):

| File | Description |
|---|---|
| `results/experiments/all_results.csv` | 136-row experiment grid (task × feature set × model) |

### Step 2 — Baseline paper tables

```bash
python scripts/make_paper_tables.py
```

**Outputs** (written to `results/paper/`):

| File | Description |
|---|---|
| `results/paper/table_1_dataset_coverage.csv` | Dataset split coverage |
| `results/paper/table_2_price_execution_gap.csv` | Price-to-execution gap across thresholds |
| `results/paper/table_3_model_ablation.csv` | Model ablation results |
| `results/paper/table_4_oracle_gap.csv` | Oracle gap by task |

### Step 3 — Baseline paper figures

```bash
python scripts/make_paper_figures.py --output-dir results/paper/figures
```

**Outputs** (written to `results/paper/figures/`):

| File | Content |
|---|---|
| `figure_1_usdc_basis_event_study.png` | USDC basis time-series over SVB stress window |
| `figure_2_price_execution_gap.png` | Price-to-execution gap bar chart |
| `figure_3_spread_depth.png` | Spread and depth deterioration |
| `figure_4_model_comparison.png` | Model comparison (test net bps) |
| `figure_5_oracle_gap.png` | Oracle gap grouped bars |

### Step 4 — Robustness grid (add-on)

```bash
python scripts/run_robustness_grid.py
```

**Output:**

| File | Description |
|---|---|
| `results/experiments_addon/robustness_price_execution_gap.csv` | 960-row grid: fee × settlement × notional × horizon |

### Step 5 — Add-on experiments (ExpectedNetProfitRegressor)

```bash
python scripts/run_addon_experiments.py
```

**Output:**

| File | Description |
|---|---|
| `results/experiments_addon/expected_net_profit_results.csv` | 6 rows: regressor results across feature sets |

### Step 6 — False-positive diagnosis

```bash
python scripts/analyze_false_positives.py
```

**Output:**

| File | Description |
|---|---|
| `results/paper_addon/table_5_false_positive_diagnosis.csv` | TP/FP/FN/TN profiles for price_threshold_10bps rule |

### Step 7 — Add-on tables

```bash
python scripts/make_addon_tables.py
```

**Outputs** (written to `results/paper_addon/`):

| File | Description |
|---|---|
| `table_8_robustness_summary.csv` | Robustness by notional/horizon (base fee, 0 settlement) |
| `table_8b_cost_robustness_summary.csv` | Robustness by fee regime × settlement penalty ($10K, 5m, 10bps) |
| `table_9_threshold_ablation.csv` | Threshold calibration sensitivity |
| `table_10_expected_net_profit.csv` | ExpectedNetProfitRegressor vs baseline |
| `table_model_failure_summary.csv` | Compact oracle gap by model family |

### Step 8 — Add-on figures

```bash
python scripts/make_addon_figures.py
```

**Outputs** (written to `results/paper_addon/figures/`):

| File | Content |
|---|---|
| `figure_8_robustness_by_notional.png` | Price-to-execution ratio by notional size |
| `figure_9_robustness_by_cost.png` | Executable % under fee/settlement variation |
| `figure_11_signal_waterfall.png` | Signal waterfall: price → executable → oracle |
| `figure_12_expected_net_profit.png` | ExpectedNetProfitRegressor vs baseline |

### Step 9 — Extended paper figures (Columbia academic theme)

```bash
python scripts/make_paper_figures_extended.py
```

**Outputs** (written to `results/paper_addon/figures/`):

| File | Content |
|---|---|
| `figure_14_event_timeline.png` | Event window calendar |
| `figure_15_model_comparison.png` | All models ranked by net bps |
| `figure_16_cumulative_pnl.png` | Oracle vs ML vs no-trade cumulative P&L |
| `figure_17_roc_curves.png` | ROC curves (logistic + LGBM) |
| `figure_18_feature_importance.png` | LGBM feature importance by family |
| `figure_19_basis_heatmap.png` | USDC basis intensity heatmap |
| `figure_20_cost_decomposition.png` | Cost decomposition waterfall |
| `figure_21_horizon_ratio.png` | Price-to-execution ratio by horizon |
| `figure_22_calibration.png` | Reliability / calibration diagram |

### Step 10 — Historical catalogue tables (18-event expansion)

```bash
python scripts/generate_source_audit_table.py
python scripts/rebuild_historical_tables.py
python scripts/build_price_grade_event_features.py
python scripts/make_mechanism_taxonomy_table.py
python scripts/run_event_robustness.py
```

**Outputs** (written to `results/paper_addon/`):

| File | Description |
|---|---|
| `table_14_historical_event_catalog.csv` | 18 events: mechanism class, tier, dates, empirical use |
| `table_15_event_data_coverage.csv` | 18 events × 5 data source type columns |
| `table_16_event_source_audit.csv` | 26 source records: verified / use_in_paper flags |
| `table_17_historical_price_grade_summary.csv` | Price-grade summaries (16 synthetic, 2 Tier A empirical) |
| `table_18_mechanism_taxonomy_summary.csv` | 7 mechanism classes: event counts, tier distribution, max depeg |
| `table_19_event_robustness.csv` | Structural price-to-execution gap characterisation by mechanism |

### Step 11 — Expanded historical figures (mechanism taxonomy, 18-event coverage)

```bash
python scripts/make_expanded_historical_figures.py
```

**Outputs** (written to `results/paper_addon/figures/`):

| File | Content |
|---|---|
| `figure_25_mechanism_taxonomy.png` | Event count + max depeg by mechanism class |
| `figure_26_coverage_matrix.png` | 18-event × data source type coverage matrix |
| `figure_27_event_timeline_expanded.png` | Event timeline 2020–2023 by mechanism class and tier |
| `figure_28_tierb_depeg_panel.png` | Tier B depeg severity vs. Tier A benchmark |

---

## Integrity Tests

Run the full test suite to verify no-overwrite guards, cost sensitivity, and model correctness:

```bash
pytest tests/ -q
```

Key test files:

| Test file | What it verifies |
|---|---|
| `test_addon_outputs_do_not_overwrite.py` | Add-on scripts never write to `results/paper/` |
| `test_robustness_cost_sensitivity.py` | Fee/settlement changes monotonically affect executable % |
| `test_robustness_grid.py` | Robustness grid schema and monotone properties |
| `test_expected_net_profit_model.py` | ExpectedNetProfitRegressor interface and calibration |
| `test_uncertainty_abstention.py` | Uncertainty module API |
| `test_historical_layer.py` | 18-event YAML count, mechanism classes, Tier A identity, source registry coverage, synthetic flag correctness |

---

## Key Empirical Numbers (paper cross-reference)

| Claim | Value | Source file |
|---|---|---|
| Primary/max basis > 10 bps (test split) | 35.09% | `table_2_price_execution_gap.csv` |
| USDC-specific basis > 10 bps (test split) | 12.65% | `table_2_price_execution_gap.csv` |
| Executable at $10K / 5m (test split) | 5.644% | `robustness_price_execution_gap.csv` |
| Executable at $10K / 1m (test split, same-minute) | 2.88% | `table_2_price_execution_gap.csv` |
| Price-to-execution ratio (1m, 10 bps, primary) | 12× | `table_2_price_execution_gap.csv` |
| Oracle net bps (basis_usdc_1m_gt10bps) | +161.7 | `table_4_oracle_gap.csv` |
| Oracle net bps (executable_arb_q10000_5m) | +224.6 | `table_4_oracle_gap.csv` |
| Best ML net bps (test split) | −49.1 (logistic@price_plus_book) | `table_4_oracle_gap.csv` |
| Oracle gap (basis task) | 210 bps | `table_4_oracle_gap.csv` |
| ExpectedNetProfitRegressor best | −61.4 bps | `expected_net_profit_results.csv` |
| False positives (price_threshold_10bps) | 581 (mean −38.7 bps) | `table_5_false_positive_diagnosis.csv` |

---

## Notes on Claim Consistency

- **35.1% figure**: refers to `cross_quote_basis_maxabs_bps > 10 bps` (primary/max basis). The USDC-specific figure is **12.65%**. All paper text and docs use "primary/max basis" when citing 35.1%.
- **12× ratio**: computed as 35.09% / 2.88% at 1-minute same-window horizon.
- **Depth provenance**: `depth_source ∈ {real_l2_snapshot, real_l2_incremental}` for all paper-grade net-profit labels. Synthetic kline depth is excluded.
- **No lookahead**: labels constructed by `join_asof` at `t + horizon`; formally verified in `tests/test_label_integrity.py`.
