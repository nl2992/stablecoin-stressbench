# Benchmark Card — Stablecoin StressBench

## Overview

StressBench is an event-based benchmark for evaluating stablecoin dislocation models on realistic, transaction-cost-aware metrics. The central claim is:

> **Naive price-signal detection of stablecoin arbitrage is misleading. After accounting for VWAP execution costs, exchange fees, and settlement friction, the majority of apparent arbitrage windows are unprofitable.**

---

## Task Definition

**Primary task:** Binary classification — predict whether the cross-quote basis between USDC/USDT will exceed 10 bps in the next 1 minute (`label_basis_1m_gt10bps`).

**Economic task:** Given a model's signal, determine whether predicted opportunities are executable at $50K notional after all transaction costs.

**Secondary tasks:** Regression (predict future basis in bps), multi-horizon (1m, 5m, 15m, 1h), multi-notional ($10K – $500K).

---

## Evaluation Protocol

### Splits

Models are trained on calm control windows and evaluated on stress episodes — reflecting the real deployment scenario where stress is unknown at training time.

| Split | Window | Rows |
|---|---|---|
| Train | Normal (Jan 2022, Feb 2023, Jan 2024) | 30,252 |
| Validation | Terra/Luna collapse (May 2022) | 11,568 |
| Test | USDC SVB depeg + recovery (Mar 2023) | 15,899 |

**No temporal leakage**: labels are forward-looking (basis at `t + horizon`); features use only information available at time `t`. The train/validation/test split is event-based, not time-based, to avoid look-ahead bias from parameter tuning.

### ML Metrics

| Metric | Description |
|---|---|
| AUROC | Area under the ROC curve |
| AUPRC | Area under the precision-recall curve |
| F1 (threshold=0.5) | Harmonic mean of precision and recall |
| Balanced Accuracy | Mean of per-class recall |
| Brier Score | Mean squared probability error |

### Economic Metrics

| Metric | Description |
|---|---|
| Net bps captured | Mean net profit (bps) across all signalled trades |
| Hit rate above cost | Fraction of signalled trades that are executable net-positive |
| False positive cost | Mean loss (bps) on signalled trades that were unprofitable |
| # trades | Total number of trades signalled |
| Final P&L (USD) | Cumulative dollar profit at $50K notional |
| Max drawdown (USD) | Maximum peak-to-trough dollar loss |
| Sharpe ratio | Annualized Sharpe computed on per-trade returns |

---

## Central Finding: The Price-to-Execution Gap

The benchmark quantifies the gap between apparent and executable arbitrage:

| Period | Basis >10bps | Executable net-profit >0 at $10K | Δ (Gap) |
|---|---|---|---|
| Train — normal (calm) | 4.2% | **0.00%** | 4.2 pp |
| Validation — Terra/Luna stress | 13.7% | 2.57% | 11.1 pp |
| Test — SVB depeg stress | 35.1% | **3.34%** | 31.8 pp |

During the SVB crisis, 35% of minutes showed a price dislocation > 10 bps, but only 3.34% were actually executable after VWAP sweep + fees + settlement delay. The execution cost averages ~88 bps for a $10K round-trip through the BTCUSDC book.

---

## Leaderboard (Test Set — `label_basis_1m_gt10bps`, $50K notional)

| Rank | Model | AUROC | AUPRC | F1 | Bal. Acc | Net Bps | Hit Rate | # Trades |
|---|---|---|---|---|---|---|---|---|
| 1 | Random Forest | **0.723** | 0.579 | 0.359 | 0.591 | -30.7 | 12.3% | 1,933 |
| 2 | Logistic | 0.713 | **0.596** | **0.566** | 0.591 | -75.7 | 4.4% | 11,967 |
| 3 | Lasso | 0.705 | 0.594 | 0.487 | 0.633 | **-22.3** | **15.3%** | 3,470 |
| 4 | Ridge | 0.689 | 0.587 | 0.525 | **0.647** | -33.9 | 12.7% | 4,179 |
| 5 | XGBoost | 0.674 | 0.504 | 0.441 | 0.596 | -63.2 | 6.9% | 3,875 |
| 6 | LightGBM | 0.622 | 0.449 | 0.542 | 0.599 | -75.3 | 4.4% | 8,936 |
| — | Last Value | 0.500 | 0.361 | — | 0.500 | — | — | 0 |
| — | Rolling Mean | 0.500 | 0.361 | — | 0.500 | — | — | 0 |
| — | AR1 | 0.500 | 0.361 | — | 0.500 | — | — | 0 |

**Key observations:**

1. **Best ML model (RF AUROC 0.72) still loses money.** The execution cost barrier means no model achieves positive net P&L at $50K notional during the test period.

2. **Lasso is the most economical baseline** (-22.3 net bps, 15.3% hit rate) because L1 regularization selects a sparse, low-false-positive signal. It makes fewer but more targeted predictions.

3. **High AUROC ≠ positive P&L.** Logistic regression achieves AUROC 0.713 but loses -75.7 bps per trade on average due to excessive false positives (11,967 trades vs. 3,470 for Lasso).

4. **Tree models overfit to volume patterns** visible in calm training data that do not generalize to the SVB stress regime structure.

5. **Naïve baselines (Last Value, Rolling Mean, AR1) never trade.** They correctly identify no executable opportunities, equivalent to a trivially conservative policy.

---

## Submission Format

To participate, submit a model with the following interface:

```python
class MyModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    # predict_proba[:,1] is the probability of label=1
```

Evaluation uses `scripts/evaluate_models.py`. Save your model as a pickle file:

```
models/trained/{model_name}_label_basis_1m_gt10bps.pkl
```

Then run:

```bash
python scripts/evaluate_models.py \
    --data-dir data/gold \
    --model-dir models/trained \
    --output my_leaderboard.csv
```

---

## Reproducibility

The full benchmark can be reproduced from raw data archives:

```bash
# 1. Pull raw data (Binance Vision archive — free, no API key required)
python scripts/pull_data.py \
    --start 2022-01-10 --end 2024-01-22 \
    --venues binance --mode archive

# 2. Build Silver + Gold features and labels (40 event-window days)
python scripts/build_features.py \
    --start 2022-01-10 --end 2024-01-22

# 3. Train all baseline models
python scripts/train_models.py \
    --data-dir data/gold --model-dir models/trained

# 4. Evaluate and produce leaderboard
python scripts/evaluate_models.py \
    --data-dir data/gold --model-dir models/trained \
    --output results/benchmark_results.csv
```

Or end-to-end:

```bash
python scripts/run_pipeline.py --start 2022-01-10 --end 2024-01-22
```

---

## Limitations and Caveats

1. **Single stress event (SVB)**: The test set covers only the March 2023 USDC depeg. Generalization to other stress events is not guaranteed.

2. **Binance-centric depth**: Net profit computations rely on Binance book depth. Coinbase/Kraken depth data would change execution cost estimates.

3. **No short selling**: The benchmark considers unidirectional arbitrage (buy cheap USDC, sell expensive USDC). Short constraints are not modeled.

4. **Latency**: The benchmark assumes immediate execution at VWAP. Real-world latency and partial fills would further reduce executable profits.

5. **Regime shift**: Training on calm periods and testing on crisis is by design but limits the amount of in-distribution training data for stress-period patterns.

---

## Citation

```bibtex
@misc{li2024stressbench,
  title   = {Stablecoin StressBench: A Transaction-Cost-Aware Benchmark for Settlement-Risk Dislocations},
  author  = {Nigel Li},
  year    = {2024},
  note    = {ICAIF Competition Track},
  url     = {https://github.com/nl2992/stablecoin-stressbench}
}
```
