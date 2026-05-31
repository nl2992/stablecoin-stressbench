#!/usr/bin/env python3
"""Validate the delta=5 bps settlement penalty using on-chain Etherscan data.

Fetches USDC ERC-20 transfer events for the SVB stress window (Mar 10-20 2023),
extracts gas prices from each transaction, computes the actual settlement cost
as a fraction of a $10K notional, and compares to the delta=5 bps assumption.

Also reports per-day USDC transfer volume, burn (redemption) events, and
large-transfer counts as on-chain evidence of institutional stress.

Outputs:
    results/paper_addon/settlement_validation.csv
    results/paper_addon/settlement_gas_costs.csv
    results/paper/figures/figure_settlement_validation.png

Usage:
    ETHERSCAN_API_KEY=<key> python scripts/run_settlement_validation.py
    python scripts/run_settlement_validation.py --api-key <key>
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO    = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "results" / "paper_addon"
FIG_DIR = REPO / "results" / "paper" / "figures"

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
CHAIN_ID       = 1
USDC_CONTRACT  = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
ZERO_ADDR      = "0x0000000000000000000000000000000000000000"

# SVB window block range (pre-computed from timestamps to avoid extra API calls)
SVB_START_BLOCK = 16_794_061   # 2023-03-10 00:00 UTC
SVB_END_BLOCK   = 16_865_180   # 2023-03-20 00:00 UTC

# Gas cost constants
GAS_UNITS_ERC20   = 65_000     # standard ERC-20 transfer
NOTIONAL_USD      = 10_000
ETH_PRICE_SVB_USD = 1_580.0    # approximate ETH/USD mid-March 2023


def _get(params: dict, api_key: str) -> dict | None:
    p = dict(params)
    p["chainid"] = CHAIN_ID
    p["apikey"]  = api_key
    try:
        r = requests.get(ETHERSCAN_BASE, params=p, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "1":
            return None
        return data
    except requests.RequestException:
        return None
    finally:
        time.sleep(0.22)


def fetch_transfers_chunk(start: int, end: int, api_key: str) -> list[dict]:
    data = _get({
        "module": "account", "action": "tokentx",
        "contractaddress": USDC_CONTRACT,
        "startblock": start, "endblock": end,
        "page": 1, "offset": 10000, "sort": "asc",
    }, api_key)
    return data.get("result", []) if data else []


def fetch_all_transfers(api_key: str) -> list[dict]:
    """Fetch transfers by splitting the block range into daily chunks."""
    all_txs = []
    blocks_per_day = 7_175   # ~12 sec block time post-merge
    start = SVB_START_BLOCK
    day   = 0
    while start < SVB_END_BLOCK:
        end   = min(start + blocks_per_day, SVB_END_BLOCK)
        chunk = fetch_transfers_chunk(start, end, api_key)
        all_txs.extend(chunk)
        date_approx = datetime(2023, 3, 10 + day, tzinfo=timezone.utc).strftime("%b %d")
        print(f"  {date_approx}: blocks {start:,}-{end:,} -> {len(chunk):,} transfers")
        start = end + 1
        day  += 1
    return all_txs


def gas_to_bps(gas_price_wei: int) -> float:
    gas_eth = GAS_UNITS_ERC20 * gas_price_wei / 1e18
    gas_usd = gas_eth * ETH_PRICE_SVB_USD
    return gas_usd / NOTIONAL_USD * 10_000


def run(api_key: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching USDC transfers for SVB window (Mar 10-20 2023)...")
    txs = fetch_all_transfers(api_key)
    print(f"Total: {len(txs):,} transfers")

    # ----------------------------------------------------------------
    # Per-day stats + per-tx gas cost
    # ----------------------------------------------------------------
    day_stats:  dict[str, dict] = {}
    gas_by_day: dict[str, list[float]] = {}

    for tx in txs:
        try:
            ts   = int(tx.get("timeStamp", 0))
            day  = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except Exception:
            continue

        # Volume (USDC has 6 decimals)
        try:
            amount_usdc = int(tx.get("value", "0")) / 1e6
        except Exception:
            amount_usdc = 0.0

        from_addr = tx.get("from", "").lower()
        to_addr   = tx.get("to",   "").lower()

        if day not in day_stats:
            day_stats[day] = {
                "date": day,
                "transfer_count": 0,
                "transfer_volume_usdc_M": 0.0,
                "large_transfer_count": 0,
                "mint_count":  0,
                "burn_count":  0,
            }
        s = day_stats[day]
        s["transfer_count"]        += 1
        s["transfer_volume_usdc_M"] = round(s["transfer_volume_usdc_M"] + amount_usdc / 1e6, 4)
        if amount_usdc >= 1_000_000:
            s["large_transfer_count"] += 1
        if from_addr == ZERO_ADDR:
            s["mint_count"] += 1
        if to_addr == ZERO_ADDR:
            s["burn_count"] += 1

        # Gas price
        try:
            gp_wei = int(tx.get("gasPrice", "0"))
            if gp_wei > 0:
                gas_by_day.setdefault(day, []).append(gp_wei)
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Gas cost per day
    # ----------------------------------------------------------------
    gas_rows = []
    for day in sorted(gas_by_day.keys()):
        prices_wei = gas_by_day[day]
        if not prices_wei:
            continue
        median_wei = sorted(prices_wei)[len(prices_wei) // 2]
        mean_wei   = sum(prices_wei) / len(prices_wei)
        p95_wei    = sorted(prices_wei)[int(len(prices_wei) * 0.95)]
        gas_rows.append({
            "date":                  day,
            "n_txs":                 len(prices_wei),
            "median_gas_gwei":       round(median_wei / 1e9, 2),
            "mean_gas_gwei":         round(mean_wei   / 1e9, 2),
            "p95_gas_gwei":          round(p95_wei    / 1e9, 2),
            "median_cost_bps_10k":   round(gas_to_bps(median_wei), 3),
            "p95_cost_bps_10k":      round(gas_to_bps(p95_wei),    3),
            "delta_assumption_bps":  5.0,
        })

    val_rows = sorted(day_stats.values(), key=lambda r: r["date"])

    # Save CSVs
    gas_path = OUT_DIR / "settlement_gas_costs.csv"
    val_path = OUT_DIR / "settlement_validation.csv"
    if gas_rows:
        with open(gas_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(gas_rows[0].keys()))
            w.writeheader(); w.writerows(gas_rows)
    if val_rows:
        with open(val_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(val_rows[0].keys()))
            w.writeheader(); w.writerows(val_rows)

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    print("\n=== Settlement Cost vs delta=5 bps assumption ===")
    print(f"{'Date':<12} {'Med gas':>9} {'Med cost bps':>13} {'P95 cost bps':>13} {'<= 5 bps?':>10}")
    print("-" * 60)
    for r in gas_rows:
        flag = "YES" if r["p95_cost_bps_10k"] <= 5.0 else "margin tight"
        print(f"{r['date']:<12} {r['median_gas_gwei']:>9.1f} {r['median_cost_bps_10k']:>13.3f} "
              f"{r['p95_cost_bps_10k']:>13.3f} {flag:>10}")

    print("\n=== USDC On-chain Activity ===")
    print(f"{'Date':<12} {'Transfers':>10} {'Vol ($M)':>10} {'Mints':>7} {'Burns':>7} {'Large':>7}")
    print("-" * 60)
    for r in val_rows:
        print(f"{r['date']:<12} {r['transfer_count']:>10,} {r['transfer_volume_usdc_M']:>10.1f} "
              f"{r['mint_count']:>7} {r['burn_count']:>7} {r['large_transfer_count']:>7}")

    if gas_rows:
        all_p95 = [r["p95_cost_bps_10k"] for r in gas_rows]
        print(f"\nP95 gas cost range: {min(all_p95):.2f} – {max(all_p95):.2f} bps on $10K")
        print(f"delta=5 bps assumption: {'CONSERVATIVE (covers P95)' if max(all_p95) < 5.0 else 'covers median; P95 may be tight during peak congestion'}")

    _make_figure(gas_rows, val_rows)
    print("\nDone.")


def _make_figure(gas_rows: list[dict], val_rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import date as dt_date
    except ImportError:
        print("matplotlib not available — skipping figure")
        return

    if not gas_rows and not val_rows:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # ---- Top: gas cost vs delta ----
    ax1 = axes[0]
    if gas_rows:
        days   = [dt_date.fromisoformat(r["date"]) for r in gas_rows]
        median = [r["median_cost_bps_10k"] for r in gas_rows]
        p95    = [r["p95_cost_bps_10k"]    for r in gas_rows]
        ax1.fill_between(days, median, p95, alpha=0.25, color="#d73027",
                         label="Median–P95 gas cost range")
        ax1.plot(days, median, color="#d73027", marker="o", ms=5, lw=2.0,
                 label="Median gas cost (bps, $10K notional)")
        ax1.plot(days, p95,    color="#d73027", marker="^", ms=5, lw=1.5,
                 ls="--", alpha=0.7)
        ax1.axhline(5.0, color="#666666", ls="--", lw=1.5,
                    label=r"$\delta = 5$ bps assumption")
        ax1.set_ylabel("Settlement cost (bps)", fontsize=10)
        ax1.legend(fontsize=8.5, loc="upper left")
        ax1.set_ylim(0, max(max(p95) * 1.4, 6.5))
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_title(
        "Settlement Penalty Validation: On-Chain Gas Costs vs Benchmark Assumption\n"
        "(USDC transfers, Mar 10–20 2023, ETH ~$1,580)",
        fontsize=10)

    # ---- Bottom: USDC burn volume ----
    ax2 = axes[1]
    if val_rows:
        days2  = [dt_date.fromisoformat(r["date"]) for r in val_rows]
        burns  = [r["burn_count"]          for r in val_rows]
        large  = [r["large_transfer_count"] for r in val_rows]
        ax2.bar(days2, burns, color="#4575b4", alpha=0.75,
                label="USDC burn events (redemptions)")
        ax2.set_ylabel("Burn events", fontsize=10, color="#4575b4")
        ax2.tick_params(axis="y", labelcolor="#4575b4")
        ax2b = ax2.twinx()
        ax2b.plot(days2, large, color="#f4a261", marker="s", ms=5, lw=1.5,
                  label=r"Large transfers ($\geq$\$1M)")
        ax2b.set_ylabel("Large transfers", fontsize=10, color="#f4a261")
        ax2b.tick_params(axis="y", labelcolor="#f4a261")
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left")
    ax2.set_xlabel("Date (Mar 2023)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out = FIG_DIR / "figure_settlement_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved.")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("ETHERSCAN_API_KEY", ""))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        print("ERROR: set ETHERSCAN_API_KEY or pass --api-key")
        raise SystemExit(1)
    run(args.api_key)
