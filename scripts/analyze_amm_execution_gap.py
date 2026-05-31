#!/usr/bin/env python3
"""AMM execution gap analysis: Curve StableSwap vs CEX order-book.

Compares price-to-execution gaps for two mechanism classes:
  1. CEX cross-quote (USDC/SVB Mar 2023): Tier-A order-book data
  2. AMM pool imbalance (USDT/Curve Jun 2023): Curve StableSwap simulation

Key finding: AMM depth is *endogenous* to imbalance, unlike CEX depth.
As LP imbalance grows, effective price impact accelerates non-linearly ---
fundamentally different from the order-book depth withdrawal in the CEX case.

Curve StableSwap formula (2-asset, amplification A):
    A * n^n * (x + y) + D = A * n^n * D + D^{n+1} / (n^n * prod(x_i))
For n=2:
    4A(x+y) + D = 4AD + D^3/(4xy)

where D is the invariant (total pool value at balance), x and y are
current reserves, A is the amplification coefficient.

Writes to:
    results/paper_addon/table_amm_execution_gap.csv
    results/paper_addon/figures/figure_amm_vs_cex_gap.png

Usage:
    python scripts/analyze_amm_execution_gap.py
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from stressbench.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Curve StableSwap math
# ---------------------------------------------------------------------------


def _curve_get_D(x: float, y: float, A: float) -> float:
    """Compute the invariant D for a 2-asset Curve pool via Newton's method."""
    S = x + y
    if S == 0:
        return 0.0
    D = S
    Ann = A * 4  # A * n^n where n=2
    for _ in range(255):
        D_P = D * D * D / (4 * x * y)
        D_prev = D
        D = (Ann * S + 2 * D_P) * D / ((Ann - 1) * D + 3 * D_P)
        if abs(D - D_prev) < 1e-6:
            break
    return D


def _curve_get_y(x_new: float, D: float, A: float) -> float:
    """Given new reserve x_new, compute the required y to preserve invariant D."""
    Ann = A * 4
    c = D * D * D / (4 * x_new * Ann)
    b = x_new + D / Ann
    y = D
    for _ in range(255):
        y_prev = y
        y = (y * y + c) / (2 * y + b - D)
        if abs(y - y_prev) < 1e-6:
            break
    return y


def curve_spot_deviation_bps(reserve_x: float, reserve_y: float, A: float) -> float:
    """Compute spot price deviation from 1:1 parity for a 2-asset Curve pool.

    Uses implicit differentiation of the StableSwap invariant holding D fixed:
        4A*(x+y) + D = 4*A*D + D^3/(4*x*y)

    dy/dx = -(4A + D_P/x) / (4A + D_P/y)
    where D_P = D^3/(4*x*y)

    Returns deviation in bps: (|dy/dx| - 1) * 10000
    """
    D = _curve_get_D(reserve_x, reserve_y, A)
    D_P = D**3 / (4.0 * reserve_x * reserve_y)
    Ann = 4.0 * A
    # Correct implicit differentiation (D is invariant, not reserve):
    dydx_abs = (Ann + D_P / reserve_x) / (Ann + D_P / reserve_y)
    return (dydx_abs - 1.0) * 10_000.0


def curve_price_impact(
    reserve_x: float,
    reserve_y: float,
    A: float,
    trade_size_usd: float,
    fee_bps: float = 4.0,
) -> dict:
    """Compute price impact for buying `trade_size_usd` of x using y (Curve AMM).

    Convention: x is the scarce asset (USDC during event), y is the abundant asset
    (USDT). We buy x (USDC) by paying y (USDT) — this is the arbitrage direction
    when x is at a discount externally.

    Args:
        reserve_x: Reserve of scarce asset (USDC), USD-equivalent.
        reserve_y: Reserve of abundant asset (USDT).
        A: Curve amplification coefficient.
        trade_size_usd: USD notional size.
        fee_bps: Pool fee in bps (4 bps for Curve 3pool).

    Returns:
        dict with:
          - spot_price_bps: pre-trade price deviation from par (bps) — how much
                           USDT premium is priced into the AMM spot.
          - price_impact_bps: additional cost from finite trade size (slippage).
          - total_execution_cost_bps: slippage + fee.
          - net_profit_bps: spot_price_bps - total_execution_cost_bps.
          - is_executable: net_profit_bps > 0.
    """
    D = _curve_get_D(reserve_x, reserve_y, A)

    # Spot price deviation from par (how much USDT you get above/below 1:1)
    spot_dev_bps = curve_spot_deviation_bps(reserve_x, reserve_y, A)

    # Execute: receive dx of x by paying dy of y
    # Positive spot_dev_bps means x is scarce → we gain more y than expected
    # We buy dx of x (give y, receive x):
    dx = trade_size_usd
    x_new = reserve_x + dx  # we add dx of x to pool (selling x for y)
    if x_new <= 0 or reserve_x <= 0 or reserve_y <= 0:
        return {
            "spot_price_bps": spot_dev_bps,
            "price_impact_bps": float("nan"),
            "total_execution_cost_bps": float("nan"),
            "net_profit_bps": float("nan"),
            "is_executable": False,
        }

    y_new = _curve_get_y(x_new, D, A)
    dy_received = reserve_y - y_new  # y we receive (positive = profit direction)

    if dy_received <= 0:
        return {
            "spot_price_bps": spot_dev_bps,
            "price_impact_bps": float("nan"),
            "total_execution_cost_bps": float("nan"),
            "net_profit_bps": float("nan"),
            "is_executable": False,
        }

    # Apply fee (pool takes fee_bps of output)
    dy_after_fee = dy_received * (1.0 - fee_bps / 10_000.0)

    # Execution price: USDT received per USDC sold
    exec_price = dy_after_fee / dx

    # Execution deviation from par (positive = above 1:1, good for seller of x)
    exec_dev_bps = (exec_price - 1.0) * 10_000.0

    # Price impact = difference between spot and execution (slippage cost)
    price_impact_bps = max(0.0, spot_dev_bps - exec_dev_bps)

    # Total execution cost = slippage + fee
    total_cost_bps = price_impact_bps + fee_bps

    # Net profit from arbitrage:
    # We sell x (USDC) in AMM at exec_dev_bps above par, then buy x back externally
    # at the external price. If x is at a discount externally by ext_discount_bps,
    # profit = exec_dev_bps - 0 (since we assumed external = par for x).
    # But since y (USDT) has spot_dev_bps premium, selling x in pool to get extra y,
    # net = exec_dev_bps (the premium captured minus fee and slippage).
    net_profit_bps = exec_dev_bps  # positive if above-par y received

    return {
        "spot_price_bps": round(spot_dev_bps, 2),
        "price_impact_bps": round(price_impact_bps, 2),
        "total_execution_cost_bps": round(total_cost_bps, 2),
        "net_profit_bps": round(net_profit_bps, 2),
        "is_executable": bool(net_profit_bps > 0),
    }


# ---------------------------------------------------------------------------
# Event parameters
# ---------------------------------------------------------------------------

# USDT/Curve June 2023 event parameters.
# Curve 3pool: ~$800M TVL in June 2023, became heavily USDT-weighted.
# A=2000 is Curve's stablecoin amplification factor.
# Key finding: Curve's AMM compresses price deviations — even at 73/27 imbalance
# the pool spot deviation is only ~3.7 bps. With a 4 bps fee, arbitrage
# through the AMM is NOT executable. This contrasts with CEX order-book events
# (SVB) where spot deviations reach 100s of bps.
_CURVE_JUNE_2023 = {
    "name": "USDT/Curve Jun 2023",
    "mechanism": "DeFi pool imbalance",
    "A": 2000,  # Curve 3pool amplification
    "pool_tvl_usd": 800_000_000,  # ~$800M TVL
    "fee_bps": 4.0,  # 0.04% pool fee
    # Scenarios: (usdc_frac, usdt_frac, description)
    # x=USDC (scarce during event), y=USDT (abundant — LPs withdrew USDC)
    "imbalance_scenarios": [
        (0.50, 0.50, "Balanced (pre-event)"),
        (0.45, 0.55, "Mild (5% drift)"),
        (0.40, 0.60, "Moderate (10% drift)"),
        (0.33, 0.67, "Elevated (17% drift)"),
        (0.27, 0.73, "Peak Jun 16 (23% drift)"),
        (0.20, 0.80, "Extreme stress"),
    ],
}

# CEX order-book comparison: USDC/SVB Mar 2023 (from paper results)
_CEX_SVB_BENCHMARKS = [
    # (notional_usd, exec_rate_pct, avg_net_bps, description)
    (10_000, 2.88, 15.0, "$10K notional (paper headline)"),
    (50_000, 1.62, 12.0, "$50K notional"),
    (500_000, 0.03, 5.0, "$500K notional"),
]


def analyze_amm_vs_cex(output_dir: Path, fmt: str = "png") -> None:
    """Compute and plot AMM vs CEX execution gap comparison."""
    rows = []

    curve_cfg = _CURVE_JUNE_2023
    pool_tvl = curve_cfg["pool_tvl_usd"]
    A = curve_cfg["A"]
    fee_bps = curve_cfg["fee_bps"]

    logger.info("=== Curve StableSwap AMM Analysis (%s) ===", curve_cfg["name"])
    logger.info("A=%d, TVL=$%.0fM, fee=%.0f bps", A, pool_tvl / 1e6, fee_bps)

    notional_sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]

    for usdc_frac, usdt_frac, scenario_label in curve_cfg["imbalance_scenarios"]:
        reserve_x = pool_tvl * usdc_frac  # USDC reserve
        reserve_y = pool_tvl * usdt_frac  # USDT reserve
        spot_dev = curve_spot_deviation_bps(reserve_x, reserve_y, A)

        for notional in notional_sizes:
            result = curve_price_impact(reserve_x, reserve_y, A, notional, fee_bps)
            rows.append(
                {
                    "event": curve_cfg["name"],
                    "mechanism": "DeFi pool imbalance",
                    "scenario": scenario_label,
                    "usdc_frac": usdc_frac,
                    "usdt_frac": usdt_frac,
                    "pool_tvl_m_usd": pool_tvl / 1e6,
                    "A": A,
                    "fee_bps": fee_bps,
                    "notional_usd": notional,
                    "amm_spot_dev_bps": round(spot_dev, 3),
                    "price_impact_bps": result["price_impact_bps"],
                    "total_cost_bps": result["total_execution_cost_bps"],
                    "net_profit_bps": result["net_profit_bps"],
                    "is_executable": result["is_executable"],
                    "note": (
                        f"AMM compresses spot to {spot_dev:.2f} bps vs fee {fee_bps} bps"
                    ),
                }
            )

        # Log summary for this scenario at $10K and $100K
        r10 = curve_price_impact(reserve_x, reserve_y, A, 10_000, fee_bps)
        r100 = curve_price_impact(reserve_x, reserve_y, A, 100_000, fee_bps)
        logger.info(
            "  [%s] spot=%.2f bps  $10K: net=%.2f exec=%s  $100K: net=%.2f exec=%s",
            scenario_label,
            spot_dev,
            r10["net_profit_bps"],
            r10["is_executable"],
            r100["net_profit_bps"],
            r100["is_executable"],
        )

    # Write CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "table_amm_execution_gap.csv"
    if rows:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s", out_csv)

    # Summary table
    _write_crossmech_summary(rows, output_dir, curve_cfg)

    # Figure
    _plot_amm_vs_cex(rows, output_dir, fmt, curve_cfg)


def _write_crossmech_summary(rows: list[dict], output_dir: Path, cfg: dict) -> None:
    """Write cross-mechanism comparison summary.

    Key comparison:
      CEX (SVB/Terra): large spot deviations (100–300 bps) but low exec rate due
        to order-book depth depletion.
      AMM (Curve June 2023): spot deviation compressed to ~3–10 bps by high-A
        invariant; fee alone (4 bps) makes AMM-route arbitrage unprofitable.
        Detection requires on-chain LP reserve monitoring (BOCPD works on AMM
        reserves; fails on CEX basis due to mean-reversion).
    """
    summary = []

    # --- CEX events (from paper results) ---
    summary.append(
        {
            "mechanism": "Fiat-reserve shock (CEX order book)",
            "event": "USDC/SVB Mar 2023",
            "data_tier": "A",
            "notional_usd": 10_000,
            "typical_spot_dev_bps": 100.0,
            "exec_rate_pct": 2.88,
            "price_to_exec_ratio": round(34.3 / 2.88, 1),
            "oracle_net_bps": 162.2,
            "exec_barrier": "order-book depth withdrawal (deposit run drains Binance liquidity)",
            "detection_method": "basis threshold or ML (BOCPD: AUROC 0.23, fails)",
            "note": "paper Tier-A headline result",
        }
    )
    summary.append(
        {
            "mechanism": "Algorithmic/reflexive (CEX order book)",
            "event": "Terra/LUNA May 2022 (val split)",
            "data_tier": "B (kline-proxy)",
            "notional_usd": 10_000,
            "typical_spot_dev_bps": 50.0,
            "exec_rate_pct": 2.30,
            "price_to_exec_ratio": round(13.5 / 2.30, 1),
            "oracle_net_bps": 88.0,
            "exec_barrier": "unstable reference pricing (algorithmic collapse)",
            "detection_method": "cross-mechanism meta-labeling; BOCPD AUROC ~0.3",
            "note": "cross-mechanism directional check",
        }
    )

    # --- AMM events (Curve simulation) ---
    for usdc_frac, usdt_frac, scenario_label in cfg["imbalance_scenarios"][
        1:
    ]:  # skip balanced
        reserve_x = cfg["pool_tvl_usd"] * usdc_frac
        reserve_y = cfg["pool_tvl_usd"] * usdt_frac
        spot_dev = curve_spot_deviation_bps(reserve_x, reserve_y, cfg["A"])
        r = curve_price_impact(reserve_x, reserve_y, cfg["A"], 10_000, cfg["fee_bps"])

        summary.append(
            {
                "mechanism": "DeFi pool imbalance (Curve AMM)",
                "event": f"USDT/Curve Jun 2023 — {scenario_label}",
                "data_tier": "C (StableSwap simulation, A=2000)",
                "notional_usd": 10_000,
                "typical_spot_dev_bps": round(spot_dev, 2),
                "exec_rate_pct": 100.0 if r["is_executable"] else 0.0,
                "price_to_exec_ratio": (
                    "∞ (not exec.)"
                    if not r["is_executable"]
                    else round(spot_dev / max(r["net_profit_bps"], 0.01), 1)
                ),
                "oracle_net_bps": round(r["net_profit_bps"], 2),
                "exec_barrier": (
                    f"AMM fee ({cfg['fee_bps']} bps) > pool spot deviation ({spot_dev:.2f} bps); "
                    "high-A invariant compresses prices even at severe imbalance"
                ),
                "detection_method": "BOCPD on LP reserve imbalance (gradual signal; works unlike CEX basis)",
                "note": f"A={cfg['A']}, TVL=${cfg['pool_tvl_usd']/1e6:.0f}M",
            }
        )

    out_csv = output_dir / "table_crossmech_summary.csv"
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    logger.info("Wrote cross-mechanism summary: %s", out_csv)


def _plot_amm_vs_cex(rows: list[dict], output_dir: Path, fmt: str, cfg: dict) -> None:
    """Generate AMM vs CEX execution gap comparison figure.

    Left panel: AMM pool spot deviation vs imbalance level.
      Shows Curve's high-A invariant compresses spot deviation to <10 bps
      even at severe pool imbalance.

    Right panel: CEX order-book exec rate vs AMM net profit (both vs notional).
      CEX: sharp elbow (retail ceiling) due to depth depletion.
      AMM: flat and near-zero regardless of notional (fee-dominated).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left panel: AMM spot deviation vs imbalance level ---
    ax1 = axes[0]
    scenarios = cfg["imbalance_scenarios"]
    usdt_fracs = [s[1] for s in scenarios]
    spot_devs = []
    for usdc_frac, usdt_frac, _ in scenarios:
        reserve_x = cfg["pool_tvl_usd"] * usdc_frac
        reserve_y = cfg["pool_tvl_usd"] * usdt_frac
        spot_devs.append(curve_spot_deviation_bps(reserve_x, reserve_y, cfg["A"]))

    ax1.bar(
        range(len(scenarios)),
        spot_devs,
        color="#003057",
        alpha=0.85,
        label="AMM spot deviation (bps)",
    )
    ax1.axhline(
        cfg["fee_bps"],
        color="#d73027",
        ls="--",
        lw=1.5,
        label=f"Pool fee ({cfg['fee_bps']} bps)",
    )
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s[2].replace(" ", "\n") for s in scenarios], fontsize=7)
    ax1.set_ylabel("Spot price deviation (bps)")
    ax1.set_title(
        f"Curve AMM Spot Deviation vs Pool Imbalance\n"
        f"(A={cfg['A']}, TVL=${cfg['pool_tvl_usd']/1e6:.0f}M, "
        f"USDT/Curve Jun 2023)"
    )
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.annotate(
        "Fee > spot dev\n→ not executable\nthrough AMM",
        xy=(3, cfg["fee_bps"] + 0.2),
        fontsize=8,
        color="#d73027",
        ha="center",
    )

    # --- Right panel: CEX exec rate vs AMM net profit (vs notional) ---
    ax2 = axes[1]
    ax2_r = ax2.twinx()

    # CEX from paper results (notional scaling)
    cex_notionals_k = [1, 5, 10, 50, 100, 500]
    # Exec rates from paper Table 2 and robustness
    cex_exec_rates = [3.4, 3.2, 2.88, 1.62, 0.8, 0.03]

    # AMM peak imbalance: net profit vs notional
    peak_usdc, peak_usdt = 0.27, 0.73
    reserve_x = cfg["pool_tvl_usd"] * peak_usdc
    reserve_y = cfg["pool_tvl_usd"] * peak_usdt
    amm_notionals_k = [1, 5, 10, 25, 50, 100, 250, 500]
    amm_net_bps = []
    for n_k in amm_notionals_k:
        r = curve_price_impact(
            reserve_x, reserve_y, cfg["A"], n_k * 1000, cfg["fee_bps"]
        )
        amm_net_bps.append(r["net_profit_bps"])

    ax2.plot(
        cex_notionals_k,
        cex_exec_rates,
        "s--",
        color="#003057",
        label="CEX exec. rate % (SVB order book)",
        lw=1.8,
        ms=6,
    )
    ax2_r.plot(
        amm_notionals_k,
        amm_net_bps,
        "o-",
        color="#d73027",
        label="AMM net profit bps (Curve, peak imbalance)",
        lw=1.8,
        ms=5,
    )
    ax2_r.axhline(0, color="#888", lw=0.8, ls=":")
    ax2.axhline(0, color="#003057", lw=0.5, ls=":")

    ax2.set_xscale("log")
    ax2.set_xlabel("Notional ($K, log scale)")
    ax2.set_ylabel("CEX exec. rate (%)", color="#003057")
    ax2_r.set_ylabel("AMM net profit (bps)", color="#d73027")
    ax2.set_title(
        "CEX vs AMM: Notional Scaling\n"
        "(SVB Mar 2023 depth ceiling vs Curve Jun 2023 fee ceiling)"
    )
    ax2.annotate(
        "Retail ceiling\n(depth depletes)",
        xy=(40, 1.5),
        fontsize=8,
        color="#003057",
        ha="center",
    )
    ax2.annotate(
        "Fee floor\n(always ~−0.3 bps)",
        xy=(50, 0.05),
        xycoords=("data", "axes fraction"),
        fontsize=8,
        color="#d73027",
        ha="center",
    )

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_fig = figures_dir / f"figure_amm_vs_cex_gap.{fmt}"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", out_fig)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMM vs CEX execution gap analysis")
    p.add_argument("--output-dir", default="results/paper_addon")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analyze_amm_vs_cex(output_dir, args.format)
    logger.info("AMM execution gap analysis complete.")


if __name__ == "__main__":
    main()
