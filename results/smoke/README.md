# Smoke-Run Results

Verification outputs from a single end-to-end pipeline run on Binance archive
data (2024-01-15).  All commands ran on a local machine; no network calls are
required to reproduce this from the cached Bronze/Silver data.

## How to reproduce

```bash
# 1. Pull Binance archive for one day (downloads ~10 MB)
python scripts/pull_data.py \
  --start 2024-01-15 --end 2024-01-15 \
  --venues binance --mode archive

# 2. Build Silver + Gold (skip-silver reuses existing Silver from all event windows)
python scripts/build_features.py \
  --start 2022-01-10 --end 2023-03-21 \
  --skip-silver

# 3. Train models
python scripts/train_models.py \
  --data-dir data/gold --model-dir models/trained

# 4. Evaluate and produce leaderboard
python scripts/evaluate_models.py \
  --data-dir data/gold --model-dir models/trained \
  --output results/smoke/leaderboard_smoke.csv
```

## Files

| File | Description |
|---|---|
| `leaderboard_smoke.csv` | Model performance on the test split (15 387 rows) |
| `leaderboard_binance_2024_01_15.csv` | Same leaderboard, named by data date |
| `pipeline_row_counts.csv` | Row counts at each pipeline stage |

## Key numbers

| Stage | Table | Rows |
|---|---|---|
| Bronze | aggTrades | 2 182 116 |
| Bronze | klines | 7 200 |
| Silver | aggTrades | 220 254 497 |
| Silver | level2 | 1 136 940 |
| Gold | dataset | 47 487 |

## Leaderboard summary (test split, label = `label_basis_1m_gt10bps`)

| Model | AUROC | Notes |
|---|---|---|
| Lasso | 0.733 | Best discriminative signal |
| Logistic | 0.706 | |
| XGBoost | 0.548 | |
| RF | 0.594 | 0 trades above cost threshold |
| Baselines (last\_value, AR1, rolling\_mean) | 0.500 | No predictive signal |

All models produce **negative PnL** on the basis-arbitrage task.
This is the central result: the price-to-execution gap makes the theoretical
basis non-tradeable in practice — consistent with the ICAIF paper finding.

## Data coverage note

The Binance-only smoke run (2024-01-15) uses synthetic kline-derived book
features.  The multi-event-window leaderboard (all dates) uses real Coinbase
level2 and Binance depth data collected via `fetch_real_data.py`.
Tardis-sourced multi-venue real L2 data requires a Tardis API key; see
`scripts/pull_data.py --mode tardis`.
