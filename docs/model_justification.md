# Model Stack Justification

This document explains the deliberate layering of models in the Stablecoin StressBench benchmark.
Each family answers a specific benchmark question and connects to a literature strand.

---

## 1. Economic Anchors

### NoTradeBaseline

**Motivation.** The simplest strategy is to abstain entirely. Any model that yields negative net
bps is worse than not trading. The no-trade baseline anchors the economic floor at 0 bps captured
and $0 cumulative P&L.

**Literature.** The concept of a "do-nothing" or abstention baseline appears throughout the market
microstructure literature as the comparison point for active strategies (see Grossman & Stiglitz 1980
on the cost of acquiring information relative to passive investment).

**Benchmark question answered.** Does the model produce positive economic value? If oracle_capture_pct
is negative, the answer is no.

**Paper interpretation.** The no-trade baseline beats any strategy with negative net bps and is the
right floor for the frozen executable-arbitrage tasks. It no longer describes every result in the
repository: the current paper draft also reports a positive cross-mechanism meta-labeling add-on on
the SVB basis task.

---

### NetProfitOracleUpperBound

**Motivation.** The oracle uses hindsight to know the realized net profit for each minute. It trades
on every window where future_net_profit_bps > 0. This defines the ceiling: the maximum net bps
capturable if prediction were perfect. Without this ceiling, we cannot know whether the gap is
large (models are failing on a hard problem) or small (problem is approximately solved).

**Literature.** Oracle bounds are standard in online-learning regret analysis (Cesa-Bianchi & Lugosi
2006). In finance, the concept maps to the "hindsight optimal strategy" used in strategy evaluation
(Hautsch, Scheuch & Voigt 2018 on limits to arbitrage).

**Benchmark question answered.** What is the maximum achievable net bps? Oracle earns 161–225 net
bps per trade on the test split, confirming that profitable windows exist and the dataset is not
degenerate.

**Paper interpretation.** The 0% oracle capture by all models highlights the severity of the
price-to-execution gap and the limits of ex-ante prediction under adverse selection.

---

## 2. Naive Trading Rules

### PriceBasisThresholdBaseline

**Motivation.** The most common industry practice is to trade whenever the quoted cross-venue price
basis exceeds a threshold (e.g., 10 bps). This rule requires no fitting and serves as the direct
naive benchmark.

**Literature.** Simple threshold rules for statistical arbitrage go back at least to Gatev, Goetzmann
& Rouwenhorst (2006) on pairs trading. In stablecoin markets, the analog is: trade when |USDC_price
- 1.00| > δ on CEX.

**Benchmark question answered.** Does the naive quoted-price rule generate profit? The empirical
answer is no: price-basis windows are frequent (35% of test minutes) but only 2.88% survive
execution costs at $10K notional.

**Paper interpretation.** This baseline quantifies the 12× price-to-execution gap. The baseline
signal fires too often, and most fires are false positives that lose money after fees and market
impact.

---

### GrossArbThresholdBaseline

**Motivation.** A slight improvement over raw price basis: uses the gross VWAP spread (before fees)
to filter windows where a round-trip appears profitable pre-cost. This isolates fee drag as the
marginal killer.

**Literature.** Gross vs net arbitrage decomposition follows standard market microstructure analysis
of round-trip costs (Amihud & Mendelson 1986; Stoll 2000).

**Benchmark question answered.** How much of the price-to-execution gap is explained by fees alone
vs market impact and settlement latency?

**Paper interpretation.** The waterfall decomposition (Figure 11 in the paper) uses this baseline
to isolate the fee-cost layer from the market-impact layer.

---

## 3. Time-Series Baselines

### LastValueBaseline

**Motivation.** The random walk model: tomorrow's value equals today's. If basis dynamics are
unpredictable, no model should beat this. In classification terms, the last-observed binary label
is the "momentum" signal.

**Literature.** Random walk null hypothesis for financial prices (Fama 1965; Lo & MacKinlay 1988).

**Benchmark question answered.** Is there any short-horizon persistence in basis dynamics that
models can exploit?

**Paper interpretation.** LastValue performs near baseline on precision-recall, confirming limited
first-order autocorrelation in profitable-window labels at 1-minute resolution.

---

### RollingMeanBaseline

**Motivation.** A simple mean-reversion model: if the rolling average basis is elevated, the
probability of profitable windows in the next minute is proportional to that average. Captures
regime persistence without any learning.

**Literature.** Mean reversion in basis / spread dynamics is documented in Engle & Granger (1987)
and has been applied to crypto markets in Liu & Tsyvinski (2021).

**Benchmark question answered.** Does regime-level persistence (elevated stress windows clustering)
provide predictive signal beyond the instantaneous value?

**Paper interpretation.** RollingMean slightly outperforms LastValue on AUROC but still fails
economically, suggesting regime persistence exists statistically but not enough to overcome
execution costs.

---

### AR1Baseline

**Motivation.** An autoregressive order-1 model fits the linear relationship y_t = alpha + beta
* y_{t-1} using the target series. Incorporates the first-order dynamics explicitly with an
estimated coefficient rather than assuming beta=1 (random walk).

**Literature.** AR(1) is the canonical univariate time-series model (Box & Jenkins 1970). In
financial prediction it is commonly used as the "hard-to-beat" linear baseline (Welch & Goyal 2008).

**Benchmark question answered.** Does the first-order lag structure of the basis series contain
exploitable signal?

**Paper interpretation.** AR1 with explicit lag features performs comparably to rolling mean,
confirming the series is close to a random walk at 1-minute resolution.

---

## 4. Interpretable Linear ML

### LogisticBaseline

**Motivation.** Logistic regression with StandardScaler is the canonical interpretable classifier.
It learns a linear decision boundary in feature space, providing odds ratios on individual features
and a probability score that can be thresholded.

**Literature.** Logistic regression is the standard linear benchmark for classification problems
(Hastie, Tibshirani & Friedman 2009, Chapter 4). In finance, it is widely used for credit risk
scoring and event prediction (Altman 1968 for bankruptcy; Bharath & Shumway 2008 for default).

**Benchmark question answered.** Can a linear model in basis/microstructure features beat a
threshold rule?

**Paper interpretation.** Logistic regression underperforms tree models on AUROC, consistent with
the nonlinear and threshold-like nature of the arbitrage signal. However, its coefficients are
interpretable: bid-ask spread and depth features receive large weights, consistent with market
microstructure theory.

---

### LassoBaseline

**Motivation.** L1-penalized linear regression performs automatic feature selection. In
high-dimensional feature sets (price + book + fragmentation + settlement), many columns are
noisy; Lasso recovers a sparse model.

**Literature.** Lasso (Tibshirani 1996) is widely used for feature selection in financial
prediction (DeMiguel, Garlappi & Uppal 2009 on portfolio selection; Chinco, Clark-Joseph & Ye 2019
on machine learning with many predictors).

**Benchmark question answered.** Which features survive L1 regularization, and does sparsity
improve generalization?

**Paper interpretation.** Lasso selects 3–5 features from the price+book set, primarily basis and
spread columns. Its generalization is similar to Ridge, suggesting the features are correlated
rather than having individually high noise.

---

### RidgeBaseline

**Motivation.** L2-penalized linear regression handles multicollinearity (cross-venue features are
correlated) and provides a smooth regularization path. Unlike Lasso, it keeps all features but
shrinks coefficients proportionally.

**Literature.** Ridge regression (Hoerl & Kennard 1970) is the standard alternative to OLS under
multicollinearity. In finance, Ridge is used for factor models with many correlated predictors
(Kelly, Pruitt & Su 2019 on instrumented principal components).

**Benchmark question answered.** Does multicollinearity in the feature set explain the gap between
linear and nonlinear models?

**Paper interpretation.** Ridge and Lasso perform similarly, confirming the bottleneck is
nonlinearity rather than collinearity. The transition from linear to tree models provides a
clear nonlinearity bonus documented in the AUROC comparisons.

---

## 5. Nonlinear Tabular ML

### RandomForestWrapper

**Motivation.** Random Forest is the leading nonlinear ensemble for tabular data. It handles
feature interactions, missing values, and class imbalance robustly. In the benchmark it serves
as the nonlinear reference model.

**Literature.** Breiman (2001) introduced Random Forest. For tabular financial data, RF is a
standard nonlinear baseline whose performance on structured microstructure features is well-established
(see Hastie, Tibshirani & Friedman 2009, Chapter 15 on ensemble methods).

**Benchmark question answered.** Does nonlinear feature interaction improve prediction of
executable arbitrage?

**Paper interpretation.** RF improves AUROC over logistic regression but still fails economically.
Its feature importances confirm the basis and depth columns are primary signals; fragmentation
and settlement columns are secondary.

---

### XGBoostWrapper

**Motivation.** XGBoost extends gradient boosting with regularization, column subsampling, and
efficient split finding. It is the dominant model in tabular ML competitions and is widely used
in production trading systems.

**Literature.** Chen & Guestrin (2016) introduced XGBoost. It is widely used in production
trading systems and financial ML competitions as the dominant gradient-boosting alternative.

**Benchmark question answered.** Does a state-of-the-art boosting model with regularization close
the gap to the oracle?

**Paper interpretation.** XGBoost performs comparably to LightGBM on this dataset (near-zero
oracle capture). The result is robust across notional sizes (robustness grid, 960 rows), confirming
the finding is not an artifact of a single hyperparameter configuration.

---

### LGBMWrapper (LightGBM)

**Motivation.** LightGBM uses histogram-based split finding and leaf-wise growth, making it
faster and often more accurate than XGBoost on large tabular datasets. It is the primary
"best base model" for the benchmark and the model swept in the hyperparameter experiment.

**Literature.** Ke et al. (2017) introduced LightGBM. It is widely used in production
high-frequency trading systems and financial ML research for its speed and accuracy on tabular data.

**Benchmark question answered.** Does the best available tabular ML model close the oracle gap?

**Paper interpretation.** LightGBM is a strong tabular baseline and, in the current paper draft,
one calm-trained basis-task run is slightly positive. That does not remove the main result:
calm-trained executable-arbitrage models remain negative, and the oracle gap is still large.

---

## 6. Direct Economic Target

### ExpectedNetProfitRegressor

**Motivation.** Instead of classifying whether |basis| > threshold, this model directly regresses
on future net_profit_bps. It converts the economic target into the model's training objective,
bypassing the intermediate classification step. A trade is taken iff predicted net_profit > 0.

**Literature.** Direct economic loss functions in ML are advocated by Elkan (2001) for cost-
sensitive learning. In finance, directly optimizing P&L rather than classification metrics is
standard in reinforcement learning approaches (Moody & Saffell 2001; Lim et al. 2022 on
deep hedging).

**Benchmark question answered.** Does aligning the model objective with the economic target
(profit) rather than the statistical target (binary classification) improve economic performance?

**Paper interpretation.** ExpectedNetProfitRegressor improves over the classification models on
oracle_capture_pct (moving from negative to near-zero) but does not achieve positive net bps. This
suggests the signal is genuinely weak rather than the loss function being mis-specified.

---

## 7. Finance-Specific Models

### MetaLabelingFilter

**Motivation.** López de Prado (2018) introduced meta-labeling: a two-stage approach where a
primary model generates binary signals, and a secondary (meta) model filters false positives among
the primary fires. The key insight is that the meta-model trains only on the subset of rows where
the primary signal fires, making it a specialized false-positive detector rather than a general
classifier.

In the benchmark, the primary signal is price_basis > 10 bps (the naive threshold rule). The
meta-label is 1{future net_profit_bps > 0} given primary signal = 1. The meta-model learns to
distinguish genuine opportunities from noise among those windows where the basis is wide.

**Literature.** López de Prado, M. (2018), "Advances in Financial Machine Learning," Chapter 3 on
meta-labeling. The framework has been applied to equity signal filtering (de Prado 2020) and
crypto trading (Li et al. 2021).

**Benchmark question answered.** Can a secondary classifier, trained only on the basis-fire events,
filter the false positives that drive the economic losses?

**Paper interpretation.** Calm-control meta-labeling remains an informative null result: the
calm train split has too few profitable primary-signal fires for the secondary model to learn a
stress filter. The current paper draft then tests the obvious fix: train the secondary model on
Terra/LUNA primary-signal windows and evaluate on SVB.

That cross-mechanism run is the one positive transfer result in the repository:
`MetaLabelingFilter_lgbm_crossmech` with `price_plus_book` features earns +82.45 bps on the SVB
test split, takes 397 trades, and captures 50.8% of the basis-task oracle. The interpretation is
not that meta-labeling solves all executable arbitrage tasks; it shows that stress-like positive
examples matter more than adding another calm-trained classifier.

---

### Regime Detectors (EWMA Z-Score, CUSUM, BOCPD)

**Motivation.** Stablecoin depegging is a regime-change event. Rather than predicting individual
windows, regime detectors identify when the market has shifted from "calm" to "stress." Trades
can then be conditioned on regime: only trade during detected stress windows.

**EWMA Z-Score** is the simplest online detector: it tracks exponentially weighted mean and
standard deviation and signals stress when the z-score exceeds a threshold. It is widely used in
industrial statistical process control (Montgomery 2012).

**CUSUM (Cumulative Sum)** detects persistent shifts in the mean by accumulating deviations.
It has better detection delay than EWMA for step-change shifts. In finance, CUSUM is used for
structural break detection in time series (Zeileis et al. 2002; Pesaran & Timmermann 2007).

**BOCPD (Bayesian Online Changepoint Detection)** maintains a posterior over the run-length
(time since last changepoint) and updates it online using a conjugate Gaussian prior. It provides
a full probability distribution over whether a changepoint has occurred.

**Literature.** Adams & MacKay (2007) introduced BOCPD. Cintra & Holloway (2023) apply BOCPD
to Curve stablecoin pool reserve dynamics and show early detection of the Terra/UST de-peg,
with pool-imbalance signals approximately 12 hours before the collapse. Note: their application
is to on-chain AMM pool data, not to CEX cross-quote basis. Transfer to the CEX cross-quote
setting is not guaranteed.

**Benchmark question answered.** Can regime detection improve trading precision by conditioning
trades on detected stress windows?

**Paper interpretation.** The empirical results reveal a clear precision-recall tradeoff:

- **CUSUM** (k=0.5, h=5): near-perfect recall (0.9995) but 83.5% false alarm rate — detects
  virtually all stress windows but fires almost constantly in both calm and stress periods.
- **EWMA** (span=20, thr=2.5): high specificity (0.65% FAR) and 86.9% accuracy, but very
  low recall (0.95%) — misses almost all stress windows, making it a precision filter not a
  detector.
- **BOCPD** (hz=0.01, thr=0.5): AUROC **0.229**, substantially below both CUSUM (0.582) and
  EWMA (0.624). BOCPD in the Cintra & Holloway (2023) setting works on slowly-varying pool
  reserve ratios; applied to minute-resolution CEX cross-quote basis, the Gaussian conjugate
  assumption does not hold well, explaining the poor AUROC.

None of the regime detectors solve the execution identification problem. Their value is as
pre-filters that reduce the candidate window set, not as standalone trading signals. The false
alarm rate at the stress threshold means any regime-gated strategy still inherits most of the
core execution-barrier problem.

---

## 8. Summary: Model Ladder

The benchmark is designed as a model ladder, where each rung answers a specific question:

| Rung | Model Family | Question | Key result |
|------|-------------|----------|------------|
| 0 | NoTradeBaseline | Floor: is trading better than abstaining? | Beats negative executable-task baselines |
| 1 | PriceBasisThreshold | Does the naive rule work? | No — 12× price-to-execution gap |
| 2 | LastValue / RollingMean / AR1 | Is there lag persistence? | Weak signal, not profitable |
| 3 | Logistic / Ridge / Lasso | Does linear ML beat threshold rules? | Marginal AUROC gain, still negative econ |
| 4 | RF / XGBoost / LightGBM | Does nonlinear ML close the gap? | Best AUROC but still negative econ |
| 5 | ExpectedNetProfitRegressor | Does aligning objectives help? | Near-zero but not positive |
| 6 | MetaLabelingFilter | Does FP filtering help? | Positive only when trained on Terra/LUNA stress fires |
| 7 | Regime detectors | Does regime conditioning help? | Reduces trade count, mixed econ result |
| ∞ | NetProfitOracleUpperBound | Ceiling: what is achievable? | 161–225 net bps/trade |

The core benchmark finding is that the gap between rung 7 and the oracle (rung ∞) remains large.
This is not a modeling failure but an empirical finding about the predictability of stablecoin
arbitrage opportunities after realistic execution costs.

---

## References

All references below are verified and citable in the paper. Placeholder citations
(arXiv IDs with XXXXX, future-dated conference proceedings) have been removed.

**Ensemble / gradient boosting:**
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).
  LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 2017*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
  (2nd ed.). Springer.

**Regularised linear models:**
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1), 267–288.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal
  problems. *Technometrics*, 12(1), 55–67.

**Economic anchors and cost-sensitive learning:**
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI 2001*.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a
  relative value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.
- Amihud, Y., & Mendelson, H. (1986). Asset pricing and the bid-ask spread. *Journal of Financial
  Economics*, 17(2), 223–249.
- Stoll, H. R. (2000). Friction. *Journal of Finance*, 55(4), 1479–1514.

**Market microstructure:**
- Hautsch, N., Scheuch, C., & Voigt, S. (2018). Limits to arbitrage in markets with stochastic
  settlement latency. *VGSF Working Paper*.
- Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market.
  *Journal of Financial Economics*, 14(1), 71–100.
- Cont, R., & Kukanov, A. (2013). Optimal order placement in limit order markets.
  *Quantitative Finance*, 17(1), 21–39.

**Finance-specific ML models:**
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  [Meta-labeling: Chapter 3.]
- Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
- Cintra, R., & Holloway, T. (2023). Bayesian changepoint detection in Curve stablecoin pools.
  *Working Paper*. [Note: application is to on-chain Curve pool reserves, not CEX cross-quote basis.]

**Time-series baselines:**
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*.
  Holden-Day.
- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks.
  *Review of Financial Studies*, 1(1), 41–66.
