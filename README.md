# Stablecoin StressBench

**Stablecoin StressBench** is a transaction-cost-aware benchmark for detecting, forecasting, and economically ranking stablecoin dislocations across venues, quote currencies, and settlement rails.

This repository implements a full-featured, reproducible benchmark and research environment designed for evaluating machine learning and econometric models on stablecoin stress-testing and dislocation scenarios.

## Research Objective

Can AI models identify stablecoin dislocations that are still economically meaningful after spreads, depth, fees, transfer frictions, and settlement risk are included?

Stablecoin StressBench is designed to test three hypotheses:
1. **Cross-quote and cross-venue dislocations** are measurable and severe during stablecoin stress events.
2. **Naive price-only signals overstate arbitrage** because they ignore executable depth, fees, latency, and transfer constraints.
3. **Joint feature models** (combining order-book state, cross-venue basis, stablecoin FX deviation, venue status, and on-chain settlement features) outperform price/candle-only baselines.

### Preliminary empirical result

During the SVB-crisis test window (March 2023), **35.1% of 1-minute windows** showed a cross-quote basis exceeding 10 bps on price alone — yet only **3.34% remained profitable** after a full VWAP order-book walk at $10K notional (including taker fees and price impact). This price-to-execution gap is the core quantitative claim of the benchmark.

## Repository Structure

```text
stablecoin-stressbench/
  README.md
  pyproject.toml
  .env.example
  Makefile
  docker-compose.yml

  configs/               # YAML configurations for venues, instruments, event windows, fees
  src/stressbench/       # Source code for ingestion, normalization, book reconstruction, features, labels, models, evaluation
  sql/clickhouse/        # ClickHouse DDL schemas for dim, fact, feature, and label tables
  scripts/               # Operational scripts for data capture, pipeline building, training, and evaluation
  notebooks/             # Jupyter notebooks for analysis and visualization
  tests/                 # Pytest suite for core components
```

## Quick Start

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Build Bronze → Silver → Gold (historical data)
```bash
# Pull Binance Vision archive (public, no API key needed)
python scripts/pull_data.py --start 2023-03-10 --end 2023-03-15 \
  --venues binance --mode archive

# Pull Coinbase / Kraken via Tardis (requires TARDIS_API_KEY in .env)
python scripts/pull_data.py --start 2023-03-10 --end 2023-03-15 \
  --venues coinbase kraken --mode tardis

# Normalize Bronze to Silver, then build Gold feature tables
python scripts/build_features.py --start 2023-03-10 --end 2023-03-15
```

### 3. Run the experiment grid
```bash
# Train all models and evaluate across all tasks × feature sets
python scripts/run_experiments.py --data-dir data/gold

# Results written to results/experiments/all_results.csv
```

### 4. Run Live Capture (optional)
```bash
python scripts/start_live_capture.py
```

### 5. Run tests
```bash
pytest tests/ -q
```

## License
MIT License
