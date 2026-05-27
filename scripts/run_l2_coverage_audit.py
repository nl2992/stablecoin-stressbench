#!/opt/anaconda3/bin/python
"""
run_l2_coverage_audit.py
------------------------
Audit L2 order-book depth coverage by event, venue, instrument, and route.

Scans silver depth/level2 directories and identifies:
  - Which dates have depth data for each instrument
  - What depth_source tag is present (real_l2_snapshot vs synthetic_kline)
  - Whether the cross-quote arbitrage route is complete (buy-leg + sell-leg)
  - Which historical events fall within the covered date ranges

Also audits whether committed feat_net_profit_1m gold files exist per date.

Outputs:
  results/paper_addon/table_23_l2_coverage_by_event_route.csv
"""

import os
import warnings
import pandas as pd
import pyarrow.parquet as pq
import yaml
from datetime import datetime, date

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILVER_ROOT = os.path.join(REPO_ROOT, "data", "silver")
GOLD_ROOT = os.path.join(REPO_ROOT, "data", "gold")
HIST_YAML = os.path.join(REPO_ROOT, "configs", "event_windows_historical.yaml")
BENCH_YAML = os.path.join(REPO_ROOT, "configs", "benchmark_splits.yaml")
OUT_DIR = os.path.join(REPO_ROOT, "results", "paper_addon")
OUT_PATH = os.path.join(OUT_DIR, "table_23_l2_coverage_by_event_route.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Route definition ─────────────────────────────────────────────────────────
# Cross-quote USDC/USDT arbitrage route uses these instruments:
ROUTE_BUY_LEG  = [("binance", "depth", "BTCUSDC"), ("binance", "depth", "USDCUSDT")]
ROUTE_SELL_LEG = [("binance", "depth", "BTCUSDT"), ("coinbase", "level2", "BTCUSD")]
ALL_ROUTE_INSTRUMENTS = ROUTE_BUY_LEG + ROUTE_SELL_LEG


def _dates_in_silver(venue: str, channel: str, symbol: str) -> set:
    """Return set of date strings available in silver for this venue/channel/symbol."""
    path = os.path.join(SILVER_ROOT, f"venue={venue}", f"channel={channel}", f"symbol={symbol}")
    if not os.path.exists(path):
        return set()
    return {d.replace("date=", "") for d in os.listdir(path) if d.startswith("date=")}


def _sample_depth_source(venue: str, channel: str, symbol: str, date_str: str) -> str:
    """Read one parquet file and return the depth_source value, or 'none'."""
    date_path = os.path.join(
        SILVER_ROOT, f"venue={venue}", f"channel={channel}",
        f"symbol={symbol}", f"date={date_str}"
    )
    if not os.path.exists(date_path):
        return "none"
    # Find first parquet file (any hour)
    for root, _, files in os.walk(date_path):
        for f in files:
            if f.endswith(".parquet"):
                try:
                    pf = pq.read_table(os.path.join(root, f))
                    if "depth_source" in pf.schema.names:
                        col = pf.column("depth_source")
                        vals = set(str(v) for v in col.to_pylist() if v is not None)
                        if vals:
                            # Priority: real_l2_snapshot > real_l2_incremental > synthetic_kline
                            if "real_l2_snapshot" in vals:
                                return "real_l2_snapshot"
                            if "real_l2_incremental" in vals:
                                return "real_l2_incremental"
                            return "synthetic_kline"
                    if "raw_source" in pf.schema.names:
                        col = pf.column("raw_source")
                        vals = set(str(v) for v in col.to_pylist() if v is not None)
                        if any("kline" in v.lower() or "candle" in v.lower() for v in vals):
                            return "synthetic_kline"
                        return "unknown_raw_source"
                    return "no_depth_source_col"
                except Exception:
                    pass
    return "no_files"


def _gold_net_profit_dates() -> set:
    """Return set of date strings for which feat_net_profit_1m gold exists."""
    path = os.path.join(GOLD_ROOT, "feat_net_profit_1m")
    if not os.path.exists(path):
        return set()
    return {d.replace("date=", "") for d in os.listdir(path) if d.startswith("date=")}


def _date_range(start: str, end: str) -> list:
    """Return list of date strings in [start, end] inclusive."""
    s = datetime.strptime(start[:10], "%Y-%m-%d").date()
    e = datetime.strptime(end[:10], "%Y-%m-%d").date()
    days = []
    current = s
    while current <= e:
        days.append(str(current))
        from datetime import timedelta
        current += timedelta(days=1)
    return days


def main():
    # ── Load YAML catalogues ─────────────────────────────────────────────────
    with open(HIST_YAML) as f:
        hist = yaml.safe_load(f)

    # ── Build instrument coverage maps ───────────────────────────────────────
    print("Scanning silver depth/level2 coverage ...")
    instrument_dates = {}
    for venue, channel, symbol in ALL_ROUTE_INSTRUMENTS:
        key = f"{venue}/{channel}/{symbol}"
        instrument_dates[key] = _dates_in_silver(venue, channel, symbol)
        print(f"  {key}: {len(instrument_dates[key])} dates")

    # Sample depth_source for USDC/SVB test (2023-03-14 as representative)
    print("\nDepth source audit (sample date 2023-03-14):")
    depth_sources = {}
    for venue, channel, symbol in ALL_ROUTE_INSTRUMENTS:
        key = f"{venue}/{channel}/{symbol}"
        src = _sample_depth_source(venue, channel, symbol, "2023-03-14")
        depth_sources[key] = src
        print(f"  {key}: {src}")

    gold_dates = _gold_net_profit_dates()
    print(f"\nGold feat_net_profit_1m dates available: {len(gold_dates)}")

    # ── Define event windows for audit ───────────────────────────────────────
    # Key events with date ranges for coverage check
    event_windows = {
        "calm_control_jan2022": ("2022-01-10", "2022-01-16", "train", "Calm control Jan 2022", "A"),
        "terra_ust_2022":      ("2022-05-07", "2022-05-14", "validation", "Terra/UST May 2022", "B/validation"),
        "celsius_3ac_2022":    ("2022-06-10", "2022-06-20", "none", "Celsius/3AC Jun 2022", "B"),
        "ftx_collapse_2022":   ("2022-11-06", "2022-11-12", "none", "FTX Nov 2022", "B"),
        "busd_regulatory_2023":("2023-02-01", "2023-02-07", "none", "BUSD Feb 2023", "B"),
        "usdc_svb_2023":       ("2023-03-10", "2023-03-20", "test", "USDC/SVB Mar 2023", "A"),
        "usdt_curve_2023":     ("2023-06-10", "2023-06-15", "none", "USDT/Curve Jun 2023", "B"),
        "iron_titan_2021":     ("2021-06-16", "2021-06-17", "none", "IRON/TITAN Jun 2021", "C"),
    }

    # ── Build coverage rows ───────────────────────────────────────────────────
    rows = []
    for event_id, (start, end, split, event_name, tier) in event_windows.items():
        event_dates = set(_date_range(start, end))
        n_expected = len(event_dates)

        row_base = {
            "event_id": event_id,
            "event_name": event_name,
            "tier": tier,
            "benchmark_split": split,
            "start_date": start,
            "end_date": end,
            "n_expected_days": n_expected,
        }

        # Check each route instrument
        buy_leg_covered = 0
        sell_leg_covered = 0
        buy_leg_src = []
        sell_leg_src = []

        for venue, channel, symbol in ALL_ROUTE_INSTRUMENTS:
            key = f"{venue}/{channel}/{symbol}"
            avail = instrument_dates[key] & event_dates
            n_avail = len(avail)
            pct = n_avail / n_expected * 100 if n_expected > 0 else 0.0

            # Sample depth_source if any dates available
            if avail:
                sample_date = sorted(avail)[0]
                src = _sample_depth_source(venue, channel, symbol, sample_date)
            else:
                src = "none"

            row_base[f"{venue}_{symbol}_days_available"] = n_avail
            row_base[f"{venue}_{symbol}_coverage_pct"] = round(pct, 1)
            row_base[f"{venue}_{symbol}_depth_source"] = src

            if (venue, channel, symbol) in ROUTE_BUY_LEG:
                buy_leg_covered += n_avail
                buy_leg_src.append(src)
            else:
                sell_leg_covered += n_avail
                sell_leg_src.append(src)

        # Route completeness
        route_buy_leg_ok = buy_leg_covered > 0
        route_sell_leg_ok = sell_leg_covered > 0
        route_complete = route_buy_leg_ok and route_sell_leg_ok

        # Real L2 or synthetic?
        all_src = buy_leg_src + sell_leg_src
        has_real_l2 = any(s in ("real_l2_snapshot", "real_l2_incremental") for s in all_src)
        all_synthetic = all(s == "synthetic_kline" for s in all_src if s != "none")

        # Gold net_profit coverage
        gold_avail = gold_dates & event_dates
        n_gold = len(gold_avail)

        row_base["route_buy_leg_covered"] = route_buy_leg_ok
        row_base["route_sell_leg_covered"] = route_sell_leg_ok
        row_base["route_complete"] = route_complete
        row_base["has_real_l2_depth"] = has_real_l2
        row_base["all_depth_synthetic_kline"] = all_synthetic
        row_base["gold_net_profit_days"] = n_gold
        row_base["used_for_execution_label"] = n_gold > 0

        # Tier after audit
        if route_complete and n_gold > 0:
            tier_after = "A (committed labels present)"
        elif route_complete and all_synthetic:
            tier_after = "A (kline-proxy only; real L2 requires re-ingestion)"
        elif route_buy_leg_ok or route_sell_leg_ok:
            tier_after = "B (partial route; USDC route incomplete)"
        else:
            tier_after = "B/C (no depth in repo)"

        row_base["tier_after_audit"] = tier_after

        note_parts = []
        if n_gold > 0:
            note_parts.append(f"feat_net_profit_1m committed for {n_gold}/{n_expected} days")
        if all_synthetic:
            note_parts.append("silver depth uses kline-proxy; paper-grade labels in committed dataset.parquet only")
        if not route_complete and (route_buy_leg_ok or route_sell_leg_ok):
            note_parts.append("USDC-route instruments missing (BTCUSDC or USDCUSDT)")
        if not route_buy_leg_ok and not route_sell_leg_ok:
            note_parts.append("no depth data in repo; external acquisition required")
        row_base["notes"] = "; ".join(note_parts) if note_parts else "—"

        rows.append(row_base)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows → {OUT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Tier-after-audit summary ──────────────────────────────────────────")
    for _, r in df.iterrows():
        print(f"  {r['event_id']:30s}  {r['tier_after_audit']}")


if __name__ == "__main__":
    main()
