# Stablecoin StressBench Status

This repository contains the code, dataset artifacts, experiment outputs, and
paper support files for the ICAIF Stablecoin StressBench project.

## Positioning

Stablecoin StressBench is a benchmark for settlement-risk dislocations in
stablecoin markets. It sits at the intersection of financial benchmark
construction, blockchain and cryptocurrency, market microstructure, trading and
smart order routing, risk management, financial time-series analysis, and
graph/network modeling.

## In Place

- Core Python package under `src/stressbench`.
- Order-book reconstruction, spread, depth, imbalance, crossed-book checks, and
  executable VWAP net-profit calculations.
- Bronze raw writer, manifest utilities, and live WebSocket collectors for
  Binance, Coinbase, and Kraken.
- Archive and live normalization for trades, books, klines, Tardis payloads,
  metadata, and on-chain transfers.
- Feature modules for microstructure, basis, fragmentation, settlement proxies,
  issuer events, and graph snapshots.
- Labels for basis forecasting, arbitrage windows, profitability, regimes, and
  recovery.
- Rule, linear, tree, sequence, cost-sensitive, regime-detection,
  meta-labeling, and graph-model wrappers.
- Evaluation metrics, backtest wrapper, and leaderboard builder.
- ClickHouse DDL files and YAML configs.
- Bronze-to-Silver-to-Gold orchestration in `scripts/build_features.py`.
- Experiment runners for the baseline grid, robustness, expected-net-profit
  regression, meta-labeling, RL diagnostics, SHAP prototypes, settlement
  validation, and paper tables/figures.
- Committed Gold dataset and paper result artifacts under `data/gold/`,
  `results/paper/`, `results/experiments/`, `results/paper_addon/`, and
  `results/experiments_addon/`.
- Pytest coverage for book logic, normalization, labels, feature contracts,
  no-lookahead checks, split integrity, robustness, add-on overwrite guards,
  meta-labeling, uncertainty APIs, and historical-layer checks.

## Research State

- The baseline freeze is tagged `v0.1.0-benchmark-freeze`.
- **Gold dataset**: `data/gold/dataset.parquet` is committed and ship-ready
  (56,134+ rows, 125 columns, SVB test + Terra/LUNA validation + 3 calm-control
  training splits).
- **Experiment results**: all paper tables and figures are reproduced by running
  `scripts/make_paper_tables.py` and `scripts/make_paper_figures.py`.
  The reproducibility manifest (`results/paper/`) is the authoritative guide.
- **Key empirical numbers** (from `results/experiments/all_results.csv`):
  - Oracle: +161.7 bps, 316 trades (basis_usdc_1m_gt10bps, SVB test split)
  - Every calibrated ML model (LightGBM, logistic, XGBoost) produces 0 or
    degenerate trades on the SVB test split; no calm-trained model beats zero.
  - Cross-mechanism meta-labeling (Terra/LUNA train, SVB test): +82.5 bps,
    397 trades, 50.9% oracle capture.
  - Pooled four-event training (Terra/LUNA + Celsius/3AC + FTX + BUSD): +83.7
    bps, 163 trades, 51.6% oracle capture (from
    `results/experiments_addon/multi_event_diversity_results.csv`).
  - Conditioned PPO-GRU RL: −29.2 bps, 919 trades.
- Add-on experiments are isolated under `results/experiments_addon/` and
  `results/paper_addon/`.
- The paper (8 pages, compiled PDF at `paper/main.pdf`) uses the SVB test event
  for execution-grade claims, Terra/LUNA as the primary cross-mechanism training
  event, and an 18-event catalogue for tiering and governance.

## Known Gaps

- Public historical Coinbase/Kraken L2 replay is not available without Tardis or
  equivalent paid archives.
- Tier B/C events in the catalogue lack route-complete L2 depth; their numbers
  are OHLCV price-grade estimates and carry "est." notation.
- The SVB and all training-event BTC-USDC buy legs use kline-proxy depth (the
  perpetual contract did not exist on Binance until January 2024); a 20% depth
  haircut sensitivity confirms the execution-gap conclusion is robust.
- Graph models and reactive RL simulation remain research extensions beyond
  the paper's headline results.
