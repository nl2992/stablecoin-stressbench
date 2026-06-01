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
- The current Gold dataset has 56,134 rows and 125 columns.
- Frozen baseline paper tables are in `results/paper/`.
- Current add-on work is isolated under `results/experiments_addon/` and
  `results/paper_addon/`.
- The paper draft uses the SVB test event for execution-grade claims, Terra/LUNA
  as a stress validation/cross-mechanism training event, and an 18-event
  catalogue for tiering and governance.

## Known Gaps

- Public historical Coinbase/Kraken L2 replay is not available without Tardis or
  equivalent paid archives.
- Some stress events in the catalogue remain Tier B/C because they lack
  sufficient route-level depth provenance for VWAP labels; their numbers are
  price-grade or taxonomy-only.
- The SVB BTC-USDC buy leg uses proxy depth in the committed benchmark view;
  sensitivity checks apply a depth haircut and keep the main execution-gap
  conclusion intact.
- Graph models and reactive RL simulation remain research extensions rather than
  paper-grade headline results.
