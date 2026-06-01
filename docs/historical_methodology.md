# Historical Methodology — Stablecoin StressBench

## 1. Catalogue Construction Philosophy

Stablecoin StressBench separates **historical coverage** from **execution-grade coverage**.

The historical catalogue records stablecoin stress events using whatever data is available.
However, execution-aware arbitrage claims are made **only for Tier A events** where
order-book depth supports VWAP net-profit labels. Tier B events support price, volume,
pool-imbalance, and liquidity-proxy analysis. Tier C events are retained for taxonomy and
motivation only.

**What the paper says:**
> We catalogue stablecoin stress events across history using the best available data,
> and perform execution-aware analysis where order-book depth permits.

**What the paper does not say:**
> We analyze all stablecoin history with execution-aware labels.

**What the paper does not say:**
> Every stablecoin event has arbitrage opportunities.

---

## 2. Data Tier Definitions

| Tier | Name | What is available | What is computable | Paper use |
|---|---|---|---|---|
| **A** | Execution-grade | Route-level depth provenance sufficient for VWAP labels | VWAP execution, `net_profit_bps`, oracle gap, price-to-execution ratio | Execution-scope quantitative claims |
| **B** | Price/liquidity-grade | OHLCV, CEX trades, DEX pool reserves, on-chain flows, or liquidity proxies | Peg deviation, cross-quote basis, liquidity stress, pool imbalance, regime labels | Price-grade summaries only |
| **C** | Context-grade | Partial data, news, issuer reports, or on-chain metrics only | Mechanism taxonomy, historical narrative | Taxonomy and motivation only |

### Tier A acceptance criteria

A window may be tagged Tier A only if ALL of the following are true:
- `depth_source` / `depth_sources_used` document all route legs used for labels
- At least two route legs have overlapping coverage during the event window
- `is_paper_grade_depth = True` for ≥ 80% of rows in the event window
- `net_profit_bps_q10000` is non-null for ≥ 50% of rows

### Tier B acceptance criteria

A window may be tagged Tier B if:
- At least one of: OHLCV, CEX aggTrades, DEX pool price, or on-chain transfer data exists
- Price-grade features (peg deviation, spread, volume) are computable
- The data source is documented in `configs/event_windows_historical.yaml`

### Tier C acceptance criteria

All other events. Data is either missing, unreliable, or covers only the stablecoin in question
without cross-venue comparison.

---

## 3. Source Hierarchy

Sources are ranked by reliability:

1. **Official** — Issuer announcements, protocol governance votes, regulatory orders
2. **Academic** — Peer-reviewed or working papers with verifiable methodology
3. **Market data** — Exchange OHLCV archives, on-chain explorers, DEX subgraphs
4. **News** — Major crypto publications (CoinDesk, The Block, Bloomberg Crypto)
5. **Context** — Wikipedia, community posts, social media (for timeline only, never for numbers)

**Rule:** No exact depeg magnitude, dollar amount, or date enters the paper unless present
in `results/paper_addon/table_16_event_source_audit.csv` with `verified=True` and
`use_in_paper=True`.

**For unverified claims:** use "est." notation or cite only the mechanism (e.g.,
"UST depegged significantly" rather than "UST lost 99% of its value in 48 hours").

---

## 4. Three-Layer Terminology

| Layer | Definition | Example output |
|---|---|---|
| **Historical catalogue** | All 18 documented stress events, any tier | `table_14_historical_event_catalog.csv` |
| **Historical empirical panel** | Events where `dataset.parquet` contains rows | `historical_event_panel.parquet` |
| **Execution benchmark** | Tier A events with VWAP labels | `table_2_price_execution_gap.csv`, oracle gap |

These three must never be conflated. The execution benchmark is a strict subset of the
historical empirical panel, which is a strict subset of the historical catalogue.

---

## 5. Event-Panel Construction

The historical event panel is built by `scripts/build_historical_event_panel.py`:

1. Load `data/gold/dataset.parquet`
2. For each event in `configs/event_windows_historical.yaml`:
   - Filter rows by `ts_1m_ns` within `[start, end]`
   - Tag each row with `event_id`, `data_tier`, `coverage_score`
3. Events not covered by the current dataset receive empty panels (no rows)
4. Write `results/experiments_addon/historical_event_panel.parquet`

**Current dataset coverage:** The gold dataset covers train (calm control windows),
validation (Terra/LUNA May 2022), and test (USDC/SVB Mar 2023). Only these three
splits appear in the empirical panel.

---

## 6. Price-Grade Feature Construction

For Tier B events, `scripts/build_price_grade_event_features.py` computes stress
summaries from whatever price data is available:

| Feature | Source | Computation |
|---|---|---|
| `max_abs_depeg_bps` | Price series | `10000 × max(|price − 1.0|)` |
| `duration_above_Xbps` | Price series | Count of minutes with `|depeg| > X bps` |
| `realized_vol_bps` | Price series | `10000 × std(log returns) × sqrt(1440)` |
| `recovery_time_hours` | Price series | Time from peak depeg to `|depeg| < 10 bps` |
| `pool_share_stablecoin` | DEX pool | Stablecoin fraction of pool reserves |
| `max_pool_imbalance` | DEX pool | Max deviation from equal weighting |

**For events not in the current dataset:** features are derived from `max_depeg_bps_est`
and `duration_class` in the YAML config, clearly tagged as `is_illustrative = True`.
Illustrative values must not be cited as measured data in the paper.

---

## 7. What Claims Are Allowed by Tier

### Tier A — allowed
- "X% of minutes show `|basis| > 10 bps`" ✓
- "Only Y% are executable at $10K after VWAP + fees" ✓
- "Oracle earns +N bps; all models lose money" ✓
- Exact net_profit_bps, hit rates, oracle gap ✓

### Tier B — allowed
- "The event showed a maximum depeg of approximately X bps (est.)" ✓ (if sourced)
- "Liquidity withdrew by ~Y% from the pool during the event" ✓ (if sourced)
- "Cross-venue basis exceeded Z bps for W hours" ✓ (if measured from data)
- Regime/stress indicator patterns ✓

### Tier B — NOT allowed
- Executable net_profit_bps without documented route-level depth provenance ✗
- Oracle gap without depth data ✗
- "Arbitrage was available" without execution measurement ✗

### Tier C — allowed
- Mechanism description and motivation ✓
- Reference to academic literature on the event ✓
- Taxonomy classification ✓

### Tier C — NOT allowed
- Any exact quantitative claim ✗
- Any empirical feature computed from the event ✗

---

## 8. Acceptance Criteria for Future Event Additions

A new event may be added to the catalogue only if:

1. `event_id` is unique and follows the naming convention `{name}_{year}`
2. `mechanism_class` is one of the defined taxonomy values
3. `data_tier` is explicitly assigned (A, B, or C)
4. `verification_status` is one of: `verified`, `needs_source`, `do_not_use_in_paper`
5. At least one entry exists in `source_verification.py` for the event
6. No execution-grade claim is made unless `data_tier = A` and route-level depth provenance is documented
7. `max_depeg_bps_est` is labelled "est." and sourced, or left null

---

## 9. Summary: The Three Sentences

For the paper, use exactly this framing:

> Stablecoin StressBench catalogues 18 stress events across algorithmic, fiat-reserve,
> regulatory, exchange-credit, DeFi-pool, collateral, and niche-stablecoin mechanisms
> from 2020 to 2023. Events are classified by data availability tier: Tier A (execution-grade,
> N=2), Tier B (price/liquidity-grade, N=11), and Tier C (context-grade, N=5).
> Execution-aware arbitrage claims — price-to-execution gap, oracle gap, model evaluation —
> are made only for Tier A events where route-level depth provenance supports VWAP net-profit labels.
