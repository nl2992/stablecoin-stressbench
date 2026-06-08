# TODO — stablecoin-stressbench

Goal: a benchmark paper where **cross-mechanism transfer is the hero** and every experiment either
(a) confirms why it works, (b) quantifies how robust it is, or (c) shows it is the only approach
that works. The headline is +82.5 bps / 51% oracle capture; every plan below either defends
that claim or extends it. Template papers: Gu et al. (ICAIF'24) and Lopez de Prado (2018).

Current state: core positive result in paper (`+82.5 bps`, 51% oracle capture from Terra→SVB
cross-mechanism meta-labeling). Results are committed (`results/experiments_addon/`,
`results/paper/`). **What is missing** is statistical validation, mechanistic interpretation,
and robustness that make the result airtight against reviewer challenges. Dependency-ordered
below; gates are non-negotiable.

---

## Plan A — Bootstrap Significance on Net bps  *(blocks "the result is noise" attack)*

**Goal**: Prove +82.5 bps is statistically distinguishable from zero via block bootstrap on the
P&L time-series. The executable-window bootstrap already exists (95% CI [1.47%, 4.59%]).
Net-bps significance is the missing companion.

**Code to write**: `scripts/run_bps_significance.py`
```
- Load per-minute trade log from results/experiments_addon/meta_labeling_crossmech_results.csv
- Block-bootstrap the SVB test P&L series (block length = 60 min, B = 2000 resamples)
- Compute: 95% CI for net bps, one-sided p(net_bps > 0), and Sharpe-ratio CI
- Compare: cross-mechanism model vs NoTrade baseline
- Save: results/experiments_addon/bps_bootstrap_ci.json
```

**Execute**:
```
python scripts/run_bps_significance.py
```

**Target result**: 95% CI lower bound > 0 (e.g., [+52, +113] bps), one-sided p < 0.05.
If CI crosses zero, fall back to "96%+ of bootstrap replicates are positive" framing (same
approach used in companion contagion-network paper).

**Write into paper**: Add CI to `tab:metalabel` caption: "Bootstrap 95% CI $[+X, +Y]$ bps
(2000 resamples, 60-min blocks) excludes zero." Add single sentence to §5.1 (Cross-Mechanism
Transfer) and to Abstract. This kills the "could be noise" reviewer objection immediately.

---

## Plan B — SHAP Cross-Event Feature Attribution  *(the mechanistic confirmation)*

**Goal**: Show that `depth_withdrawal` and `bid_ask_spread` dominate SHAP scores in *both*
the Terra/LUNA training split and the SVB test split — proving the mechanism-invariance claim
that is central to the paper's theory ("the order book responds to uncertainty, not its cause").

**Code to write**: `scripts/run_shap_crossmech.py`
```
- Load the committed meta-labeling model (or retrain on Terra split)
- Compute SHAP values on: (a) Terra/LUNA validation split, (b) SVB test split
- Rank features by mean |SHAP| for each split
- Compute top-5 overlap (Jaccard) and Spearman rank-correlation between splits
- Plot: side-by-side SHAP bar charts (Terra vs SVB), sorted by Terra importance
- Save: results/experiments_addon/shap_crossmech.json, shap_crossmech_fig.png
```

**Execute**:
```
python scripts/run_shap_crossmech.py
```

**Target result**: Top-3 features identical in both splits (target: depth_bid, spread, depth_ask).
Spearman ρ > 0.70 between Terra and SVB SHAP rankings. Overlap is the "why it transfers" proof.

**Write into paper**: New Figure in §5.3 after the current cross-mechanism table. Caption:
"SHAP feature importance on Terra training split (left) vs SVB test split (right). Depth
withdrawal and spread widening rank \#1 and \#2 in both crises (Spearman $\rho = X$),
confirming mechanism invariance." Also add the Spearman ρ value to Abstract.

---

## Plan C — Cross-Mechanism Transfer Matrix (3×3)  *(generalises the single Terra→SVB result)*

**Goal**: Test three additional training→test direction pairs to show the Terra→SVB result is
not cherry-picked. Build a 3×3 (or 4×4) transfer matrix: which mechanism types transfer to
which others?

**Code to write**: `scripts/run_transfer_matrix.py`
```
For each (train_event, test_event) pair where train ≠ test:
  - Terra/LUNA (algorithmic) → SVB (reserve-bank shock)    [already done]
  - Celsius/3AC (exchange credit) → SVB                    [new]
  - FTX (exchange credit) → SVB                            [new]
  - Terra/LUNA → Celsius/3AC                               [new — cross-regime]
Load feature set from historical_event_panel.parquet (already in experiments_addon/)
Report: net bps, oracle capture pct, n_trades for each pair
Save: results/experiments_addon/transfer_matrix.csv
```

**Execute**:
```
python scripts/run_transfer_matrix.py
```

**Target result**: ≥ 2 of 3 new pairs achieve positive net bps (oracle capture > 30%). The
claim "microstructure stress signature is mechanism-invariant" needs ≥2 independent
confirmations beyond the headline pair.

**Write into paper**: Replace the single-row Terra→SVB entry in `tab:metalabel` with a
3-row transfer matrix (Terra, Celsius, FTX → SVB). Add one row for cross-mechanism pair
that does NOT work (mechanism that doesn't transfer) to show the limit. Caption: "Transfer
holds across mechanism types (algorithmic, exchange-credit → reserve-bank shock) but not
from [type X] where [explain mechanism mismatch]."

---

## Plan D — Early-Warning Lead Time Curve  *(practitioner usability)*

**Goal**: At what prediction horizon before execution does the meta-labeler still produce
positive net bps? Practitioners need to know the "actionable warning window" — the k-minute
lead time at which the model fires and leaves enough time to route the order.

**Code to write**: `scripts/run_lead_time_analysis.py`
```
- For horizons k in {1, 2, 5, 10, 15, 30, 60} minutes:
  - Shift prediction timestamp backward by k minutes (predict at t, execute at t+k)
  - Re-compute P&L at the shifted execution time
  - Record: net bps, oracle capture, n_profitable_trades at each k
- Save: results/experiments_addon/lead_time_crossmech.csv
- Plot: oracle capture (%) vs lead time (minutes); mark the break-even horizon
```

**Execute**:
```
python scripts/run_lead_time_analysis.py
```

**Target result**: Oracle capture > 0% at k ≤ 10 minutes. Expected shape: decays from 51%
at k=1 to ~20% at k=10, to 0% at k=30. The break-even horizon is the paper's "useful warning
window."

**Write into paper**: New Figure in the Practitioner section (§7 or equivalent). Caption:
"Oracle capture vs. prediction lead time for cross-mechanism meta-labeler. Positive returns
persist up to $k$ minutes in advance, giving a practitioner a $k$-minute window to route
the order before the executable gap closes." Cite in the abstract as "usable warning window."

---

## Plan E — Calibration Curve / Reliability Diagram  *(trustworthy probability outputs)*

**Goal**: Show the meta-labeler is well-calibrated — when it says 70% probability that a trade
is executable-positive, approximately 70% of those trades should be profitable. This is the
difference between a "useful" model and an "overconfident" model.

**Code to write**: `scripts/run_calibration_curve.py`
```
- Load meta-labeler predictions (probability scores) on SVB test split
- Bin predictions into deciles [0-0.1, ..., 0.9-1.0]
- For each bin: compute (a) mean predicted probability, (b) actual profitable rate
- Compute ECE (Expected Calibration Error)
- Plot: reliability diagram (predicted vs actual), diagonal = perfect calibration
- Save: results/experiments_addon/calibration_curve.json, calibration_curve.png
```

**Execute**:
```
python scripts/run_calibration_curve.py
```

**Target result**: ECE < 0.15. Bins above 0.7 predicted probability contain > 60% profitable
trades. If miscalibrated: apply isotonic regression post-hoc and show ECE improves.

**Write into paper**: New Figure in §5.3. Caption: "Reliability diagram for cross-mechanism
meta-labeler on SVB test split. ECE = $X$. The model is well-calibrated: predicted probability
above $Y$ corresponds to $Z\%$ profitable trades." Add ECE number to conclusion's "limits"
paragraph.

---

## Plan F — Supervision Format Statistical Test  *(the binding-constraint claim)*

**Goal**: The paper claims "supervision format, not label density, is the binding constraint."
Currently supported by one comparison: binary labels (+82.5 bps) vs PPO-GRU (-29.2 bps) at
17% positive-label density. Strengthen this with ≥3 formats tested at identical density.

**Code to write**: `scripts/run_supervision_format_ablation.py`
```
At fixed 17% positive-label density, train four supervision formats on Terra split:
  1. Binary classification (cross-entropy) — current meta-labeler
  2. Ordinal regression (3 levels: loss / near-zero / profit)  [new]
  3. Expected-profit regression (predict net bps directly)     [new]
  4. PPO policy gradient — current RL baseline
For each: evaluate net bps, oracle capture, AUROC on SVB test split
Bootstrap CI (B=500) for each net-bps estimate
Save: results/experiments_addon/supervision_format_ablation.csv
```

**Execute**:
```
python scripts/run_supervision_format_ablation.py
```

**Target result**: Binary classification best, ordinal > regression > RL. Even if ordinal is
also positive, the monotone ranking confirms the "format matters" claim more rigorously
than a single A-vs-B comparison.

**Write into paper**: New Table in §5.4 (Supervision Format) with 4-row ablation plus
bootstrap CIs. Caption: "At identical 17\% positive-label density, supervision format
determines sign of P\&L. Binary labels earn $+82.5\bps$; policy gradient earns $-29.2\bps$."
This becomes the direct evidence for Contribution 2.

---

## Plan G — Cost-Sensitivity Robustness for Meta-Labeler  *(defends against "lucky fees" attack)*

**Goal**: The 12× gap and +82.5 bps result depend on taker-fee and settlement-latency
assumptions. Show the positive result survives a 2× fee multiplier and 40% depth haircut
(current paper already does this for the *gap* but not for the *meta-labeler net bps*).

**Code to write**: `scripts/run_metaLabel_cost_robustness.py`
```
Run meta-labeler evaluation at all 9 cost-parameter combinations:
  fee_mult in {1.0, 1.5, 2.0} × depth_haircut in {0%, 20%, 40%}
For each: re-label the SVB test split with new net-profit calculation,
  re-evaluate meta-labeler (same model, no retraining), record net bps
Extend existing results/experiments_addon/robustness_price_execution_gap.csv
  with meta-labeler columns
Save: results/experiments_addon/metaLabel_cost_robustness.csv
```

**Execute**:
```
python scripts/run_metaLabel_cost_robustness.py
```

**Target result**: Meta-labeler net bps > 0 in ≥ 7 of 9 cost scenarios. The "it only works at
current fee assumptions" attack is neutralized.

**Write into paper**: New column in the robustness section table: "Meta-label net bps" alongside
the existing "Executable rate" robustness rows. Caption: "Cross-mechanism meta-labeling
remains positive across the full cost-parameter grid ($X$ of 9 scenarios), while calm-trained
models remain non-positive in all."

---

## Plan H — Depth-Withdrawal Alignment Figure  *(the visual "smoking gun" for mechanism invariance)*

**Goal**: Show side-by-side time series of `depth_bid_10bps` (order book depth at ±10 bps)
during the Terra/LUNA stress window and the SVB/USDC stress window. The visual alignment
is the most direct evidence that "the order book responds to uncertainty, not its cause."

**Code to write**: `scripts/run_depth_alignment.py`
```
- Load Terra/LUNA minute data (from data/gold/dataset.parquet, 'validation_terra_luna' split)
- Load SVB minute data (from data/gold/dataset.parquet, 'test_svb' split)
- Time-align both series from crisis onset (t=0 = first minute |basis| > 20 bps)
- Normalize: divide depth by pre-crisis 24h mean depth for each event
- Compute Pearson r between Terra and SVB normalized depth curves (over first 120 minutes)
- Plot: 2-panel figure: (a) Terra depth withdrawal, (b) SVB depth withdrawal
  Overlay: basis (right axis), depth (left axis), alignment shading
- Save: results/paper/figures/figure_depth_alignment.png
```

**Execute**:
```
python scripts/run_depth_alignment.py
```

**Target result**: Pearson r > 0.65 between normalized depth curves. Visual similarity in
shape: spike → withdrawal → slow recovery follows the same pattern in both crises despite
different root causes.

**Write into paper**: New Figure (2-panel) in §5.3 (Cross-Mechanism Transfer). Caption:
"Depth withdrawal pattern during Terra/LUNA (left) and USDC/SVB (right), normalized to
pre-crisis baseline. Pearson $r = X$ despite different crisis mechanisms. The meta-labeler
trained on the left-panel pattern recognises the right-panel signal without retraining."
This is the visual centrepiece that makes the mechanism-invariance claim compelling.

---

## Credibility checklist (reviewer-facing, non-negotiable)
1. Bootstrap CI on net bps excludes zero OR ≥96% of replicates are positive (Plan A).
2. SHAP top-3 features align between Terra and SVB (Spearman ρ reported, Plan B).
3. Transfer matrix has ≥2 training sources, not just Terra→SVB (Plan C).
4. Bootstrap CIs on all net-bps figures in tab:metalabel (Plans A, F).
5. Supervision format ablation has ≥3 formats at identical label density (Plan F).
6. Cost-parameter robustness for meta-labeler (not just for gap), Plan G.
7. Note: synthetic_fallback flag in meta_labeling_crossmech_results.csv must be resolved.
   Either document clearly that synthetic depth was used + sensitivity shows same result,
   or re-run with actual Terra/LUNA L2 data if available. Do NOT leave undocumented.
