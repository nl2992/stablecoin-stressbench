# Stablecoin Stress Event Catalog

Comprehensive historical catalogue of stablecoin stress events with data-tier classification
for use in the StressBench benchmark. **18 events** spanning 2020–2023 across **7 mechanism classes**.

The authoritative machine-readable source for all events is
`configs/event_windows_historical.yaml`. This document provides narrative context.
Source verification status for each event is in
`src/stressbench/history/source_verification.py`.

---

## Data Tier Classification

| Tier | Description | Computability | Benchmark use |
|------|-------------|--------------|---------------|
| **A** | Execution-grade: full L2 order book + VWAP labels computable | Full `net_profit_bps` labels; oracle bound | Primary benchmark tasks |
| **B** | Price-grade: OHLCV / DEX / trades available, no full L2 | Price-basis labels; depeg magnitude | Secondary analysis; estimates only |
| **C** | Context-grade: partial data or post-hoc reconstruction | Historical taxonomy; qualitative | Literature framing; no empirical claims |

Tier A execution-gap claims are valid **only for USDC/SVB Mar 2023** (the two Tier A events).
All Tier B magnitude figures must use "est." notation in paper text.

---

## Mechanism Classes

| Class | Description | Events (N) | Tier range |
|---|---|---|---|
| Algorithmic / Reflexive | Mint/burn or governance-token backing; bank-run dynamics | 5 | B–C |
| Fiat-Reserve Bank Shock | Reserve bank seizure; redemption uncertainty | 2 | **A** |
| Regulatory / Issuer Winddown | Regulatory order or policy reduces supply | 2 | B–C |
| Exchange Credit / Liquidity | Exchange insolvency or withdrawal freeze | 3 | B |
| DeFi Pool Imbalance | AMM pool imbalance; on-chain vs CEX price divergence | 3 | B |
| Collateral / Liquidation | Collateral crash triggers forced liquidations | 1 | B |
| RWA / Niche Stablecoin | Real-asset backing or protocol exploit | 2 | B–C |

---

## Events by Mechanism Class

---

### CLASS 1: ALGORITHMIC / REFLEXIVE

---

#### Event A-1: FEI Protocol Launch Stress (April 2021)

**Event ID:** `fei_launch_2021`  
**Tier:** C (context-grade)  
**Verification status:** needs_source

**Mechanism:** FEI launched with a Genesis Group event that minted ~1.3B FEI. The peg-defence
mechanism (Uniswap direct incentives) failed to maintain the $1 peg. FEI traded well below $1
for approximately 1–2 weeks. Not a death spiral — FEI later recovered — but a significant
launch-stress event for algorithmic partial-reserve designs.

**Key dates:**
- 2021-04-01: FEI Genesis Group launch; ~1.3B FEI minted
- 2021-04-01 to ~2021-04-10: FEI below $1; peg-defence incentives insufficient

**Affected stablecoins:** FEI (depeg); USDC/USDT: not meaningfully affected

**Peak depeg magnitude:** est. −2000 bps (no primary on-chain confirmation)

**Data availability:**
- CoinGecko OHLCV: available in principle; URL not verified
- Uniswap v2 swap history: available via The Graph
- No CEX L2 data (FEI was DEX-only on Ethereum)

**Coverage score:** 0.25

**Empirical use:** Taxonomy context only. Not in any benchmark split.

**Notes:** The FEI launch stress is noteworthy because it shows that even well-funded algorithmic
designs can fail at launch. Tribe DAO later merged with Fei Protocol. Exact depeg magnitude
not confirmed from primary source — use "est." notation.

---

#### Event A-2: IRON/TITAN Collapse (June 2021)

**Event ID:** `iron_titan_2021`  
**Tier:** C (context-grade)  
**Verification status:** needs_source

**Mechanism:** Algorithmic stablecoin bank run. IRON was a partially-collateralized stablecoin
backed by USDC (75%) and TITAN (25%). A bank-run dynamic emerged: TITAN's price fell, reducing
IRON's effective backing, triggering redemptions which sold TITAN, accelerating the decline.
IRON collapsed to $0 within ~24 hours — the first widely-observed death spiral in this category.

**Key dates:**
- 2021-06-16: TITAN begins sharp decline; IRON depeg starts
- 2021-06-17: TITAN reaches ~$0; IRON collapses to ~$0

**Affected stablecoins:** IRON: 100% terminal depeg; USDC/USDT: negligible contagion (<5 bps)

**Peak depeg magnitude:** IRON: terminal (−10000 bps); benchmark venue stablecoins: negligible

**Data availability:**
- CoinGecko/CMC OHLCV: available (price-grade)
- QuickSwap (Polygon) DEX: partial swap data via The Graph
- No CEX L2 depth (IRON was DEX-only)

**Coverage score:** 0.25

**Literature:** Klages-Mundt & Minca (2022) model this as a canonical algorithmic stablecoin
death spiral. Later replicated at vastly larger scale by UST/LUNA.

**Empirical use:** Taxonomy context only. Not in benchmark splits.

**Notes:** IRON/TITAN established the death-spiral template later replicated by UST/LUNA at
much larger scale. Mark Cuban publicly acknowledged losses, providing press documentation.
Intermediate depeg price path not confirmed from primary on-chain data.

---

#### Event A-3: MIM/Wonderland Confidence Shock (January–February 2022)

**Event ID:** `mim_wonderland_2022`  
**Tier:** B (price-grade)  
**Verification status:** needs_source

**Mechanism:** Confidence shock from governance controversy. Wonderland (TIME/wMEMO) was a
DeFi protocol backed partly by MIM (Magic Internet Money). The pseudonymous identity of
Wonderland's treasury manager (Sifu/Michael Patryn) was revealed as a convicted fraudster.
This triggered a panic unwinding of TIME and wMEMO, pressuring MIM via collateral links.
Curve MIM-3pool became heavily imbalanced as MIM holders exited.

**Key dates:**
- 2022-01-27: Sifu identity revealed; TIME/wMEMO price collapses
- 2022-01-27 to 2022-02-03: MIM basis widens; Curve pool imbalanced
- ~2022-02-03: MIM partially recovers as Abracadabra Protocol stabilises

**Affected stablecoins:**
- MIM: est. −300 bps at peak
- USDC/USDT: minimal direct contagion

**Peak depeg magnitude:** MIM: est. −300 bps (no primary on-chain confirmation)

**Data availability:**
- CoinGecko OHLCV for MIM: available in principle
- Curve MIM-3pool reserves: available via Curve API / The Graph
- Abracadabra Protocol on-chain: available
- No CEX L2 for MIM (primarily on-chain and DEX)

**Coverage score:** 0.50

**Empirical use:** Taxonomy context; illustrates governance-confidence shock mechanism distinct
from reserve-bank or regulatory events.

**Notes:** The MIM/Wonderland episode demonstrates that off-chain governance controversies can
trigger on-chain stablecoin stress. Not in benchmark splits. Magnitude is an estimate from
secondary commentary; exact figures need on-chain Curve data verification.

---

#### Event A-4: Terra/UST Collapse (May 2022)

**Event ID:** `terra_ust_2022`  
**Tier:** B (price-grade)  
**Verification status:** verified

**Mechanism:** Algorithmic stablecoin death spiral at scale. UST was backed by LUNA (Terra's
native governance token) through a burn/mint mechanism. Large UST withdrawals from Anchor
Protocol (~$14B TVL) triggered a bank-run. LUNA collapsed from ~$85 to <$0.01 in 7 days. UST
depegged from $1 to ~$0.02. Total market cap destruction: ~$40B.

**Key dates:**
- 2022-05-07: Large Curve pool UST withdrawals begin; Curve 3pool imbalance develops
- 2022-05-09: UST depeg breaks 0.90; LUNA begins sharp decline
- 2022-05-11: UST at $0.30; emergency LUNA mint begins
- 2022-05-13: LUNA below $0.01; UST at $0.08
- 2022-05-14: Terra chain halted

**Affected stablecoins:**
- UST: terminal depeg (−9800 bps verified)
- USDT: brief stress (<15 bps)
- USDC: minimal (<10 bps)
- DAI: minimal; Curve 3pool rebalancing caused temporary liquidity pressure

**Peak depeg magnitude:** UST: −9800 bps (verified); USDT: −15 bps briefly; DAI: −8 bps briefly

**Data availability:**
- CEX OHLCV (Binance, Coinbase, Kraken): available
- Binance partial order book: available via Binance archive
- Curve pool reserves: available via The Graph
- On-chain Terra transactions: available via Terra Explorer archives
- Full L2 depth tape for benchmark venues: not in benchmark dataset

**Coverage score:** 0.50

**Literature:**
- Briola et al. (2023): network analysis of UST collapse via Curve liquidity pools
- Cintra & Holloway (2023): Curve 3pool imbalance preceded UST collapse by ~12h
- Kwon, Minegishi & Nishi (2023): algorithmic stablecoin collapse post-mortem

**Empirical use:** Validation split (the benchmark's validation event, `split=validation` in
`configs/event_windows.yaml`). Price-basis claims only. The 12h Curve-leading-indicator finding
is verified and citable.

**Notes:** UST terminal depeg is one of the most widely-documented events in crypto history.
Contagion to USDC/USDT was short-lived but real. The Curve pool imbalance as a leading indicator
is now verified in academic literature.

---

#### Event A-5: USDD/TRON Stress (June 2022)

**Event ID:** `usdd_tron_2022`  
**Tier:** B (price-grade)  
**Verification status:** needs_source

**Mechanism:** USDD (TRON algorithmic stablecoin) experienced stress in the weeks following the
UST collapse as sentiment toward algorithmic stablecoins was broadly negative. USDD's reserve
mechanism (TRX backing + overcollateralization claims) was questioned. USDD briefly traded below
$1 on Huobi and Poloniex, primary CEX venues for TRON ecosystem stablecoins.

**Key dates:**
- 2022-06-13 to 2022-06-20: USDD trades below peg on TRON-ecosystem exchanges

**Affected stablecoins:** USDD: est. −200 bps peak; USDC/USDT: not materially affected

**Peak depeg magnitude:** USDD: est. −200 bps (no primary source confirmation)

**Data availability:**
- CoinGecko OHLCV for USDD: available in principle
- Huobi, Poloniex OHLCV: available in principle
- No benchmark-venue (Binance/Coinbase/Kraken) USDD depth

**Coverage score:** 0.50

**Empirical use:** Taxonomy context; UST-contagion sentiment on other algorithmic stablecoins.
Not in benchmark splits.

**Notes:** USDD's TRX reserve backing maintained peg recovery, distinguishing it from UST.
Exact magnitude not confirmed from primary source; "est." notation required.

---

### CLASS 2: FIAT-RESERVE BANK SHOCK

---

#### Event B-1: USDC/SVB Stress (March 10–15, 2023) — PRIMARY BENCHMARK EVENT

**Event ID:** `usdc_svb_2023`  
**Tier:** A (execution-grade)  
**Verification status:** verified

**Mechanism:** Reserve-bank insolvency shock. Silicon Valley Bank (SVB), where Circle held
approximately $3.3B of USDC reserves (~8% of total reserves), was seized by FDIC regulators
on March 10, 2023. USDC depegged to as low as $0.87 on secondary markets. The depeg was driven
by uncertainty about reserve recovery. On March 12, the FDIC announced full deposit insurance
for SVB, resolving the uncertainty. USDC recovered to $0.997 by March 13 and to $1.000 by March 15.

**Key dates:**
- 2023-03-09: SVB announces emergency capital raise; deposits begin leaving
- 2023-03-10 08:30 UTC: FDIC seizure announcement
- 2023-03-10 10:00–18:00 UTC: USDC begins depegging; reaches $0.95 on Coinbase
- 2023-03-11: USDC at $0.87 on peak stress
- 2023-03-12 22:00 UTC: US Treasury + Fed + FDIC joint statement: full SVB deposit insurance
- 2023-03-13: USDC rallies to $0.997
- 2023-03-15: USDC fully restored to $1.000

**Affected stablecoins:**
- USDC: peak −1300 bps (~$0.87; verified)
- DAI: secondary depeg (USDC-backed collateral); peak −200 bps
- USDT: brief premium +50 bps (flight to Tether)

**Peak depeg magnitude:** USDC: −1300 bps (verified); DAI: −200 bps; USDT: +50 bps (premium)

**Data availability:**
- Binance/Coinbase/Kraken real L2 snapshots: YES (captured during event)
- VWAP labels at all notional sizes: YES
- Full 5-day test window: YES

**Coverage score:** 1.0

**Core benchmark results (all execution-grade, test split):**
- 35.1% of minutes exceed 10 bps primary/max cross-quote basis (12.65% USDC-specific)
- 2.88% executable at $10K/1m after VWAP walk + fees → **12× price-to-execution gap**
- Oracle: +161.7 bps (basis task), +224.6 bps (executable arb task)
- Best ML model: −49.1 bps (logistic@price\_plus\_book); oracle gap 211 bps
- All non-oracle models: **negative net bps** out of sample

**Empirical use:** Primary benchmark test split. All execution-grade claims anchored here.

---

#### Event B-2: USDC Recovery Window (March 15 – April 1, 2023)

**Event ID:** `usdc_svb_recovery_2023`  
**Tier:** A (execution-grade for CEX labels; Tier A partial)  
**Verification status:** verified

**Mechanism:** Post-SVB recovery. After the FDIC deposit guarantee (Mar 12), USDC reanchored
to $1.000 within 3 days. The recovery window provides a within-epoch "normal" baseline: low
basis, very few executable arbitrage windows, high model false-positive rate.

**Key dates:**
- 2023-03-15: USDC at $1.000 restored
- 2023-03-20: Volatility returns to pre-SVB levels
- 2023-04-01: Window end

**Affected stablecoins:** USDC: full recovery (<10 bps routine noise); DAI: full recovery

**Coverage score:** 0.75

**Empirical use:** Test split recovery window. Validates model false-positive rate drops in
normal regime. Part of `split=test` in `configs/event_windows.yaml`.

---

### CLASS 3: REGULATORY / ISSUER WINDDOWN

---

#### Event C-1: Binance USDC→BUSD Auto-Conversion (September 2022)

**Event ID:** `binance_stablecoin_conversion_2022`  
**Tier:** C (context-grade)  
**Verification status:** needs_source

**Mechanism:** Binance announced that USDC, USDP, and TUSD balances would be automatically
converted to BUSD effective September 29, 2022. This was a platform policy decision, not a
market stress event. No depeg occurred (conversions at par). Included to document the Binance
stablecoin market structure shift that preceded the BUSD regulatory action of Feb 2023.

**Peak depeg magnitude:** 0 bps (par conversion)

**Empirical use:** Context only; explains Binance BUSD dominance leading into Feb 2023 NYDFS action.

---

#### Event C-2: BUSD Regulatory Winddown (February–March 2023)

**Event ID:** `busd_regulatory_2023`  
**Tier:** B (price-grade)  
**Verification status:** verified

**Mechanism:** Regulatory enforcement. NYDFS ordered Paxos to stop minting new BUSD on
February 13, 2023. SEC issued a Wells notice to Paxos for BUSD being an unregistered security.
Binance auto-converted user BUSD holdings to USDT/USDC. BUSD supply declined from $16B to ~$8B
within 30 days. Brief dislocations occurred during conversion rushes.

**Key dates:**
- 2023-02-13: NYDFS orders Paxos to stop BUSD minting; SEC Wells notice
- 2023-02-15: Binance announces BUSD conversion program
- 2023-03-08: BUSD supply halved

**Affected stablecoins:**
- BUSD: −10 to −30 bps during conversion rushes (verified)
- USDC/USDT: +5 to +10 bps appreciation as conversion destinations

**Peak depeg magnitude:** BUSD: −30 bps (verified); USDC/USDT: +10 bps

**Coverage score:** 0.50

**Literature:** Gorton & Zhang (2023): regulatory approaches to stablecoin issuers.

**Empirical use:** Illustrative regulatory winddown dynamics; not in benchmark splits.
Occurred 3 weeks before the SVB event; may explain elevated Q1 2023 basis volatility.

---

### CLASS 4: EXCHANGE CREDIT / LIQUIDITY

---

#### Event D-1: Celsius/Three Arrows Capital Contagion (June 2022)

**Event ID:** `celsius_3ac_2022`  
**Tier:** B (price-grade)  
**Verification status:** verified (event dates); needs_source (depeg magnitude)

**Mechanism:** Celsius Network froze withdrawals on June 12, 2022. Three Arrows Capital (3AC)
insolvency was announced ~June 17. Both entities had large DeFi positions; forced liquidations
caused stETH/ETH peg stress and modest USDT/USDC basis widening on CEX venues. The primary
stress was in DeFi collateral chains, not stablecoin pegs directly.

**Key dates:**
- 2022-06-12: Celsius freezes withdrawals
- 2022-06-15: stETH/ETH peg stress peaks (~94 cents on the dollar)
- 2022-06-17: 3AC insolvency announced; liquidation cascade begins
- 2022-06-25: Window end; DeFi stabilises

**Affected stablecoins:**
- stETH: off-peg vs ETH (not a stablecoin but the primary stress asset)
- USDT: est. −100 bps maximum (CEX contagion)
- USDC: minimal

**Peak depeg magnitude:** USDT: est. −100 bps (not confirmed from primary market data)

**Coverage score:** 0.50

**Empirical use:** Taxonomy context; illustrates DeFi-credit-contagion mechanism. Not in
benchmark splits.

---

#### Event D-2: HUSD Issuer Failure (August 2022)

**Event ID:** `husd_depeg_2022`  
**Tier:** B (price-grade)  
**Verification status:** needs_source

**Mechanism:** HUSD issuer (Stable Universal) lost Huobi backing due to Huobi Global
restructuring. HUSD redemptions were effectively suspended. HUSD traded at a steep discount
on secondary markets. Huobi later clarified the situation; partial recovery followed. HUSD is
not a benchmark venue stablecoin but illustrates exchange-credit mechanism diversity.

**Key dates:**
- 2022-08-18: HUSD begins trading at steep discount
- 2022-08-25: Partial recovery after Huobi clarification

**Affected stablecoins:** HUSD: est. −800 bps peak; USDC/USDT at benchmark venues: negligible

**Peak depeg magnitude:** HUSD: est. −800 bps (no primary source confirmation)

**Coverage score:** 0.50

**Empirical use:** Taxonomy context for exchange-issuer credit risk. Not in benchmark splits.

---

#### Event D-3: FTX Collapse (November 2022)

**Event ID:** `ftx_collapse_2022`  
**Tier:** B (price-grade)  
**Verification status:** verified

**Mechanism:** Exchange credit and insolvency shock. FTX's balance sheet was revealed to be
insolvent (FTT tokens used as Alameda collateral). CoinDesk's Nov 2 report triggered a bank run.
FTX halted withdrawals on Nov 8. FTX filed for Chapter 11 on Nov 11. USDT briefly depegged
on external CEX due to exchange-specific premium/discount effects.

**Key dates:**
- 2022-11-02: CoinDesk publishes Alameda balance sheet
- 2022-11-06: Binance CEO tweets about FTT liquidation
- 2022-11-08: FTX halts withdrawals
- 2022-11-11: FTX Chapter 11 filing

**Affected stablecoins:**
- USDT: briefly −5 to −20 bps on external CEX (verified)
- USDC: <5 bps
- DAI: <5 bps

**Peak depeg magnitude:** USDT: −20 bps on Kraken briefly (verified); USDC: −5 bps

**Coverage score:** 0.50

**Literature:**
- Vidal-Tomàs, Briola & Aste (2023): FTX downfall and Binance consolidation
- Conlon, Corbet & McGee (2023): FTX contagion using event-study methodology

**Empirical use:** Illustrates exchange-specific credit shocks produce smaller stablecoin
dislocations than reserve-bank shocks (SVB: −1300 bps vs FTX: −20 bps). Not in benchmark splits.

---

### CLASS 5: DEFI POOL IMBALANCE

---

#### Event E-1: Curve 3Pool / UST Imbalance (May 2022)

**Event ID:** `curve_3pool_ust_2022`  
**Tier:** B (price-grade)  
**Verification status:** needs_source

**Mechanism:** Co-incident with the Terra/UST collapse (Event A-4). As UST holders exited into
Curve 3CRV (the USDC+USDT+DAI LP token), the 3pool became heavily UST-imbalanced. This
created within-pool discounts on USDC, USDT, and DAI relative to the pool's posted prices.
Cintra & Holloway (2023) document the 12h lead time of this imbalance before the full UST depeg.

**Peak depeg magnitude:** Pool-internal: est. −500 bps; CEX-side USDC/USDT: <10 bps

**Data availability:**
- Curve pool reserve data: available via The Graph (subgraph)
- CEX OHLCV: available
- No L2 depth for the pool-internal stress

**Coverage score:** 0.50

**Empirical use:** DeFi contagion mechanism illustration; co-incident with `terra_ust_2022`.
Motivates integration of Curve pool reserve data as a feature.

---

#### Event E-2: USDC/DAI Secondary DeFi Stress During SVB (March 10–15, 2023)

**Event ID:** `usdc_dai_secondary_svb_2023`  
**Tier:** B (price-grade, despite coinciding with Tier A window)  
**Verification status:** needs_source

**Mechanism:** Co-incident with USDC/SVB (Event B-1). DAI depegged to approximately −200 bps
due to USDC backing in MakerDAO's Peg Stability Module (PSM). MakerDAO emergency governance
raised PSM fees to stem outflows. FRAX (partially USDC-backed) also faced pressure. The DeFi
side amplified the CEX-side pressure captured in the Tier A dataset.

**Peak depeg magnitude:** DAI: est. −200 bps; FRAX: est. −100 bps (not confirmed from primary MakerDAO logs)

**Coverage score:** 0.75 (Tier A window but Tier B data for DeFi-side)

**Empirical use:** DeFi contagion illustration co-incident with the primary Tier A event.
Do not double-count with `usdc_svb_2023` for benchmark tasks.

---

#### Event E-3: USDT/Curve Pool Stress (June 2023)

**Event ID:** `usdt_curve_2023`  
**Tier:** B (price-grade)  
**Verification status:** verified

**Mechanism:** Curve pool imbalance. Tether reserve concerns re-emerged alongside a large
Curve 3pool imbalance (USDT became overweight ~60% of pool vs. normal ~33%). USDT briefly
traded at −8 bps on Binance. The event was short-lived (hours to days) relative to the SVB event.

**Key dates:**
- 2023-06-12: Curve 3pool imbalance begins
- 2023-06-13: Peak USDT discount −8 bps; rebalancing begins
- 2023-06-14: Pool rebalances; USDT <2 bps discount
- 2023-06-15: Normal conditions restored

**Affected stablecoins:** USDT: −8 bps (verified); USDC: +3 bps; DAI: +2 bps

**Peak depeg magnitude:** USDT: −80 bps (cross-venue conservative estimate); CEX: −8 bps verified

**Coverage score:** 0.50

**Empirical use:** Out-of-sample illustrative. Motivates on-chain Curve integration as feature source.

---

### CLASS 6: COLLATERAL / LIQUIDATION

---

#### Event F-1: DAI Black Thursday (March 12, 2020)

**Event ID:** `dai_black_thursday_2020`  
**Tier:** B (price-grade)  
**Verification status:** verified

**Mechanism:** Collateral liquidation shock. ETH price crashed ~50% on March 12, 2020 (COVID
market sell-off). MakerDAO CDP (Collateralized Debt Position) vaults became undercollateralized.
Liquidation auctions cleared at near-zero bids due to gas congestion. Approximately $4.5M of
DAI was minted with zero collateral backing in these broken auctions. DAI traded **above peg**
(premium) because demand to cover undercollateralized positions exceeded supply.

**Key dates:**
- 2020-03-12: ETH price crashes ~50%; Maker vaults undercollateralized
- 2020-03-12: Gas congestion causes liquidation auctions to clear at near-zero bids
- 2020-03-13: MakerDAO governance declares emergency; MKR dilution begins
- 2020-03-14: System stabilisation; DAI returns to peg

**Affected stablecoins:**
- DAI: +150 bps premium (above peg; verified from MakerDAO post-mortems)
- USDC/USDT: not significantly affected

**Peak depeg magnitude:** DAI: +150 bps (above peg; verified)

**Coverage score:** 0.50

**Literature:** MakerDAO post-mortems document the Black Thursday events in detail. The event
is widely studied in DeFi risk literature (Gudgeon et al. 2020; Klages-Mundt et al. 2021).

**Empirical use:** Taxonomy context. Illustrates collateral-shock mechanism produces above-peg
stress (opposite direction to reserve-bank shocks). Not in benchmark splits.

**Notes:** DAI above-peg stress has different arbitrage mechanics than below-peg: the
optimal trade is to sell DAI (short), not buy it. The execution-barrier framework still applies
but with reversed directional exposure.

---

### CLASS 7: RWA / NICHE STABLECOIN

---

#### Event G-1: Acala aUSD Minting Exploit (August 2022)

**Event ID:** `acala_ausd_2022`  
**Tier:** C (context-grade)  
**Verification status:** needs_source

**Mechanism:** Smart contract exploit. A misconfigured iBTC/aUSD liquidity pool on Acala
Network (Polkadot parachain) allowed unbacked aUSD to be minted. Approximately 1.28 billion
aUSD were illegitimately minted. Acala placed the network in maintenance mode and halted
transactions. Most minted aUSD was identified and burned through governance; partial recovery
followed. Not a market-stress event in the conventional sense — it is an exploit/hack mechanism.

**Key dates:**
- 2022-08-14: Exploit discovered; 1.28B aUSD minted
- 2022-08-14: Acala network in maintenance mode
- 2022-08-21: Governance vote to burn illegitimate aUSD; partial recovery

**Peak depeg magnitude:** aUSD: est. −9900 bps (near-zero on secondary markets)

**Coverage score:** 0.25

**Empirical use:** Taxonomy context only. Demonstrates smart-contract exploit mechanism
distinct from economic stress events.

---

#### Event G-2: USDR Real-Estate Backed Stablecoin Failure (October 2023)

**Event ID:** `usdr_2023`  
**Tier:** B (price-grade)  
**Verification status:** needs_source

**Mechanism:** Real-world-asset (RWA) collateral illiquidity. USDR (Tangible Protocol) was
backed by tokenised real estate on Polygon. DAI reserves (the liquid portion) were depleted
by redemption requests. The remaining real-estate collateral could not be liquidated quickly
enough to meet redemptions. USDR traded at approximately $0.50 on secondary markets.

**Key dates:**
- 2023-10-11: DAI reserves depleted; redemption requests exceed liquid backing
- 2023-10-11 to 2023-10-18: USDR trades at steep discount on Polygon DEX

**Peak depeg magnitude:** USDR: est. −5000 bps (~$0.50 trough)

**Data availability:**
- CoinGecko OHLCV for USDR: available in principle
- Polygon DEX swaps: available via The Graph
- No CEX L2 data (USDR was DEX-only on Polygon)

**Coverage score:** 0.50

**Empirical use:** Taxonomy context. Illustrates RWA illiquidity risk for asset-backed stablecoins.
Not in benchmark splits.

---

## Coverage Summary

| Event ID | Mechanism Class | Tier | Max Depeg (est.) | Benchmark Use |
|---|---|---|---|---|
| fei_launch_2021 | Algorithmic / Reflexive | C | −2000 bps (est.) | Context only |
| iron_titan_2021 | Algorithmic / Reflexive | C | Terminal | Context only |
| mim_wonderland_2022 | Algorithmic / Reflexive | B | −300 bps (est.) | Context only |
| terra_ust_2022 | Algorithmic / Reflexive | B | −9800 bps (verified) | Validation split |
| usdd_tron_2022 | Algorithmic / Reflexive | B | −200 bps (est.) | Context only |
| **usdc_svb_2023** | **Fiat-Reserve Bank Shock** | **A** | **−1300 bps (verified)** | **PRIMARY TEST** |
| usdc_svb_recovery_2023 | Fiat-Reserve Bank Shock | A | −10 bps (normal) | Test recovery window |
| binance_stablecoin_conversion_2022 | Regulatory Winddown | C | 0 bps (par) | Context only |
| busd_regulatory_2023 | Regulatory Winddown | B | −30 bps (verified) | Illustrative |
| celsius_3ac_2022 | Exchange Credit / Liquidity | B | −100 bps (est.) | Context only |
| husd_depeg_2022 | Exchange Credit / Liquidity | B | −800 bps (est.) | Context only |
| ftx_collapse_2022 | Exchange Credit / Liquidity | B | −20 bps (verified) | Illustrative |
| curve_3pool_ust_2022 | DeFi Pool Imbalance | B | −500 bps pool-internal (est.) | Contagion illustration |
| usdc_dai_secondary_svb_2023 | DeFi Pool Imbalance | B | −200 bps (est.) | Co-incident illustration |
| usdt_curve_2023 | DeFi Pool Imbalance | B | −80 bps (est.) | Illustrative |
| dai_black_thursday_2020 | Collateral / Liquidation | B | +150 bps (above peg, verified) | Context only |
| acala_ausd_2022 | RWA / Niche Stablecoin | C | −9900 bps (est.) | Context only |
| usdr_2023 | RWA / Niche Stablecoin | B | −5000 bps (est.) | Context only |

**Totals:** 18 events · Tier A: 2 · Tier B: 11 · Tier C: 5

---

## Implications for Benchmark Claims

**Execution-grade claims** (net_bps_captured, oracle_capture_pct, price-to-execution gap)
are valid **only for Tier A events** (USDC/SVB 2023 primary and recovery).

**Price-grade claims** (depeg magnitude, frequency comparisons) may reference Tier B events
for illustrative purposes but **must** be labelled as "est." or "price-grade" in paper text.

**Taxonomy claims** (mechanism classification, historical precedent) may reference Tier C events
with explicit caveats about data availability. No numerical claims allowed for Tier C.

---

## References

- Briola, A., Vidal-Tomàs, D., Wang, Y., & Aste, T. (2023). Anatomy of a run: The Terra Luna crash. *Finance Research Letters*.
- Catalini, C., & de Gortari, A. (2023). On the economic design of stablecoins. *MIT DCI Working Paper*.
- Cintra, R., & Holloway, C. (2023). Bayesian changepoint detection in Curve stablecoin pools. *Working Paper*.
- Conlon, T., Corbet, S., & McGee, R. (2023). The FTX collapse and systemic crypto risk. *Working Paper*.
- Gorton, G., & Zhang, J. (2023). Taming wildcat stablecoins. *University of Chicago Law Review*.
- Gudgeon, L., et al. (2020). DeFi Protocols for Loanable Funds. *ACM AFT*.
- Hautsch, N., Scheuch, C., & Voigt, S. (2018). Limits to arbitrage in markets with stochastic settlement latency. *VGSF Working Paper*.
- Klages-Mundt, A., & Minca, A. (2022). While stability lasts: A stochastic model of stablecoin pegs. *Operations Research*.
- Kwon, S., Minegishi, K., & Nishi, R. (2023). Terra/LUNA de-peg mechanics. *Working Paper*.
- Lyons, R., & Viswanath-Natraj, G. (2023). What keeps stablecoins stable? *Journal of International Money and Finance*.
- Vidal-Tomàs, D., Briola, A., & Aste, T. (2023). FTX's downfall and Binance's consolidation. *SSRN Working Paper*.
