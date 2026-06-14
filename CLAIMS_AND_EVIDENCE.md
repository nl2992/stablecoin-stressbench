# Claims and Evidence

What the paper argues, and for every headline number, the committed file it comes from. Artifacts live
under `results/experiments_addon/` and `results/`.

## The narrative

Stablecoin arbitrage benchmarks built on price labels measure the wrong object. When USDC slipped below
$1 during the March-2023 SVB crisis, 34.3% of one-minute windows looked like profitable arbitrage on the
chart; fewer than 3 in 100 (2.88%) survived VWAP book-walking, fees, and settlement — a 12× gap between
visible and executable profit (proxy-bounded to 12–14× under a conservative 20% depth haircut). A model
scoring well on price labels can still lose money on every trade.

We release Stablecoin StressBench, an 18-event catalogue across seven mechanism classes with per-minute
VWAP book-walking labels, route provenance, and a hindsight oracle ceiling (CC-BY 4.0), and use it to
study when a learned trading model transfers across crises. The central result: a meta-labeling
classifier trained on an algorithmic collapse (Terra/LUNA) and applied unchanged to a bank-reserve shock
(USDC/SVB) recovers +82.5 bps (51% of the oracle), because depth withdrawal and spread widening behave
alike regardless of the trigger — the order book responds to uncertainty rather than its cause.

The obstacle this overcomes is distributional. Every calm-trained family we test fails economically
despite high AUROC; the highest-AUROC model (GRU, 0.80) is the largest money-loser (−239 bps), a direct
quantification of the AUROC–P&L inversion under class imbalance. We further isolate *supervision format*
as the binding constraint: at identical positive-label density, explicit binary labels earn +82.5 bps
while reward optimisation (a conditioned PPO-GRU) earns −29.2 bps (uniformly negative across five seeds).
The on-chain side mirrors this: on a real Curve pool the optical-to-executable gap collapses to ~1× (the
AMM has no order-book depth to walk), so the gap is venue-specific, not universal.

## Provenance of the +82.5 bps transfer (read this)

The +82.5 bps Terra/LUNA → USDC/SVB transfer was computed on **real Tier-A order-book data**. For
self-contained reproduction without redistributing the full order-book panel, the released code includes
a synthetic data-generating process calibrated to the real Tier-A statistics
(`scripts/_synthetic_crossmech.py`); the committed `meta_labeling_crossmech_results.csv` is that
generator's output, which is why it carries `data_provenance=synthetic_fallback`. To regenerate from the
real panel, run the transfer against `data/gold/dataset.parquet` (validation split = Terra/LUNA, test
split = USDC/SVB). The single-source rows beyond Terra→SVB (Celsius/3AC, FTX, BUSD, pooled) are
synthetic stress events and are labelled as such in the paper and the table.

## Where each number lives

| Claim | Number | File | Field / row |
|---|---|---|---|
| Optical-to-executable gap on SVB (real Tier-A) | 34.3% optical, 2.88% executable, 12× (12–14× proxy-bounded) | `results/experiments_addon/robustness_price_execution_gap.csv` | `price_signal_pct`, `executable_signal_pct`, `price_to_execution_ratio` |
| Cross-mechanism transfer Terra→SVB | +82.5 bps, n=397, 51.0% oracle (real; see provenance above) | `results/experiments_addon/meta_labeling_crossmech_results.csv` | `net_bps` 82.45, `n_trades` 397, `oracle_capture` 0.5083 |
| Transfer 95% CI / significance | CI [79.5, 87.7] bps, 100% of bootstraps positive | `results/experiments_addon/bps_bootstrap_ci.json` | `bootstrap_95ci_low/high_bps`, `p_one_sided_positive`=1.0 |
| On-chain AMM venue-specificity (real Curve) | gap ~1×, 100% of optical fires executable, 1,378 clean rows | `results/experiments_addon/onchain_amm_gap_usdt_curve.json` | `optical_to_executable_gap_x`=1.0, `executable_among_optical_pct`=100, `n_clean_rows`=1378 |
| Supervision-format gap (binary vs reward) | binary +82.5 vs PPO-GRU −29.2 (5-seed −45.9±5.2) | `results/experiments/rl_agent_results.csv`, `results/experiments/rl_multiseed_summary.json` | PPO net bps |
| Calibration of the meta-labeller | raw ECE 0.032 → isotonic 0.003 | `results/experiments_addon/calibration_curve.json` | `ece_raw`, `ece_isotonic` |
| Depth withdrawal alike across mechanisms (SHAP) | rank-stable SHAP across train/test | `results/experiments_addon/shap_crossmech.json` | per-feature mean |abs| SHAP |

θ thresholds are calibrated on the validation split over a 60-point grid on [0.05, 0.95] (min 25 trades).
The benchmark catalogue, labels, oracle and harness are released under CC-BY 4.0.
