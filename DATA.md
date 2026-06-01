# Data Notes

Stablecoin StressBench does not commit raw market or on-chain archives to Git.
The repository tracks code, schemas, configs, notebooks, tests, paper result
artifacts, and the current Gold benchmark dataset used by the paper draft.
Large raw Bronze and normalized Silver archives should live locally under
`data/` or in an external artifact store.

## Local Layout

```text
data/
  bronze/   # immutable raw vendor messages and downloaded archive files
  silver/   # normalized canonical trade, book, metadata, and on-chain tables
  gold/     # benchmark features, labels, splits, model-ready dataset.parquet
```

## Sources

- Binance public archives from `https://data.binance.vision`
- Live Binance, Coinbase, and Kraken WebSocket captures
- Tardis historical crypto market data, if credentials are available
- Etherscan token transfer data for stablecoin settlement proxies
- Issuer event timelines and venue metadata from the YAML configs

## Git Policy

- Keep large raw and generated data out of Git.
- Commit only tiny examples or fixtures when needed for tests.
- Store API keys in `.env`, never in committed files.
- Prefer reproducible pull/build scripts over manually shared files.

## Current Status

The committed Gold dataset is `data/gold/dataset.parquet` with 56,134 rows and
125 columns. It contains calm-control, Terra/LUNA validation, and USDC/SVB test
splits.

The frozen baseline paper tables in `results/paper/` were generated from the
benchmark-freeze dataset view documented in `results/paper/table_1_data_coverage.csv`.
The current paper draft also uses add-on artifacts in `results/experiments_addon/`
and `results/paper_addon/`, including cross-mechanism meta-labeling, RL
diagnostics, robustness grids, source-audit tables, and settlement checks.

Rebuilding from raw sources is supported by `scripts/pull_data.py`,
`scripts/archive_to_bronze.py`, and `scripts/build_features.py`, but exact
historical Coinbase/Kraken L2 replay requires external Tardis credentials.
