#!/usr/bin/env python3
"""Generate the event universe figures for Section 3.4.

Produces:
  figure_event_universe.png  — two-panel:
    Left:  Bubble timeline of all 18 events (mechanism × year, size=|depeg|, colour=tier)
    Right: Real basis fingerprints — Terra/LUNA vs USDC/SVB (measured data only)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "paper_addon" / "figures"
PAPER = ROOT / "results" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY = "#003057"
BLUE = "#75B2DD"
GOLD = "#F2A900"
RED = "#C4122F"
LGREY = "#D0D0D0"
MGREY = "#909090"

TIER_COLOUR = {"A": NAVY, "B": BLUE, "C": LGREY}

MECH_LABELS = {
    "Algorithmic / Reflexive": "Algorithmic /\nReflexive",
    "Fiat-Reserve Bank Shock": "Fiat-Reserve\nBank Shock",
    "Regulatory / Issuer Winddown": "Regulatory\nWinddown",
    "Exchange Credit / Liquidity": "Exchange Credit\n/ Liquidity",
    "DeFi Pool Imbalance": "DeFi Pool\nImbalance",
    "Collateral / Liquidation": "Collateral /\nLiquidation",
    "RWA / Niche Stablecoin": "RWA /\nNiche",
}
MECH_ORDER = list(MECH_LABELS.keys())


def load_events() -> pd.DataFrame:
    cat = pd.read_csv(
        ROOT / "results/paper_addon/table_14_historical_event_catalog.csv"
    )
    cat["start_dt"] = pd.to_datetime(cat["start"])
    cat["year_frac"] = (
        cat["start_dt"].dt.year + (cat["start_dt"].dt.dayofyear - 1) / 365
    )
    cat["mech_idx"] = cat["mechanism_class"].map(
        {v: i for i, v in enumerate(MECH_ORDER)}
    )
    cat["abs_depeg"] = cat["max_depeg_bps_est"].abs().clip(upper=12000)
    # bubble radius: sqrt of depeg magnitude, scaled
    cat["bubble_r"] = np.sqrt(cat["abs_depeg"]).clip(10, 120) * 0.8
    # tier from data_tier column
    cat["tier_1"] = cat["data_tier"].str[0]
    return cat


def load_real_basis() -> dict[str, pd.DataFrame]:
    panel_path = ROOT / "results/experiments_addon/historical_event_panel.parquet"
    if not panel_path.exists():
        # Right-panel data unavailable in this checkout. The paper crops the
        # right panel out (trim in \includegraphics), so the left bubble panel
        # is what matters; return empty so the figure still builds.
        return {}
    panel = pd.read_parquet(panel_path)
    basis_col = "cross_quote_basis_usdc_bps"
    out = {}
    for ev in ["terra_ust_2022", "usdc_svb_2023"]:
        sub = panel[panel["event_id"] == ev].copy()
        sub = sub.sort_values("ts_1m_ns").reset_index(drop=True)
        sub["minute"] = np.arange(len(sub))
        sub["basis_clip"] = sub[basis_col].clip(-500, 500)
        out[ev] = sub
    return out


def make_figure() -> None:
    cat = load_events()
    basis = load_real_basis()

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    plt.subplots_adjust(wspace=0.12)

    # ── Left panel: bubble timeline ──────────────────────────────────────────
    ax = axes[0]

    for _, row in cat.iterrows():
        colour = TIER_COLOUR.get(row["tier_1"], LGREY)
        # Distinguish above-peg (positive max_depeg_bps_est) vs below-peg
        edge = GOLD if row["max_depeg_bps_est"] > 0 else "white"
        ax.scatter(
            row["year_frac"],
            row["mech_idx"],
            s=row["bubble_r"] ** 1.4,
            color=colour,
            edgecolors=edge,
            linewidths=0.7,
            alpha=0.88,
            zorder=3,
        )
        # Label key events. Offsets are per-event and horizontal-only
        # (va="center") so each label sits on its own row centreline and
        # never overlaps the mechanism rows above or below it.
        LABEL_OFFSETS = {
            "dai_black_thursday_2020": (9, 0),
            "terra_ust_2022": (9, 0),
            "ftx_collapse_2022": (9, 0),
            "busd_regulatory_2023": (9, 0),
            "usdc_svb_2023": (10, 0),
        }
        if row["event_id"] in LABEL_OFFSETS:
            ax.annotate(
                row["display_name"]
                .replace(" (PRIMARY)", "")
                .replace("USDC/SVB Stress", "USDC/SVB")
                .replace("Terra/UST Collapse", "Terra/UST")
                .replace("BUSD Regulatory Winddown", "BUSD Winddown"),
                xy=(row["year_frac"], row["mech_idx"]),
                xytext=LABEL_OFFSETS[row["event_id"]],
                textcoords="offset points",
                fontsize=4.8,
                color="#333333",
                ha="left",
                va="center",
            )

    ax.set_yticks(range(len(MECH_ORDER)))
    ax.set_yticklabels([MECH_LABELS[m] for m in MECH_ORDER], fontsize=5.5)
    ax.set_xlabel("Year", fontsize=7)
    # Extra right-side whitespace so event labels (placed to the right of their
    # bubbles) sit well inside the axis and survive the LaTeX trim crop that
    # removes the right panel.
    ax.set_xlim(2019.7, 2025.4)
    ax.set_xticks([2020, 2021, 2022, 2023, 2024])
    ax.set_xticklabels(["2020", "2021", "2022", "2023", "2024"], fontsize=6)
    ax.tick_params(left=False, labelsize=6)
    ax.set_title("(a) 18-Event Stress Universe", fontsize=7, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Tier legend
    leg = [
        mpatches.Patch(color=NAVY, label="Tier A — execution-grade"),
        mpatches.Patch(color=BLUE, label="Tier B — price/liquidity"),
        mpatches.Patch(color=LGREY, label="Tier C — taxonomy"),
        mpatches.Patch(
            facecolor="white", edgecolor=GOLD, linewidth=0.9, label="Above-peg stress"
        ),
    ]
    ax.legend(
        handles=leg,
        fontsize=5,
        loc="upper left",
        framealpha=0.85,
        edgecolor="none",
        ncol=2,
    )

    # ── Right panel: real basis fingerprints ─────────────────────────────────
    ax2 = axes[1]

    if not basis:
        # Parquet absent in this checkout; paper crops this panel out anyway.
        ax2.axis("off")
        for out_path in [
            OUT / "figure_event_universe.png",
            PAPER / "figure_event_universe.png",
        ]:
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print("Saved figure_event_universe.png (left panel only; parquet absent)")
        return

    terra = basis["terra_ust_2022"]
    svb = basis["usdc_svb_2023"]

    # Normalise time to [0,1]
    t_terra = terra["minute"] / len(terra)
    t_svb = svb["minute"] / len(svb)

    ax2.plot(
        t_terra,
        terra["basis_clip"].clip(-300, 300),
        color=BLUE,
        lw=0.7,
        alpha=0.85,
        label="Terra/UST (Tier B, est.)",
    )
    ax2.plot(
        t_svb,
        svb["basis_clip"].clip(-1200, 300),
        color=NAVY,
        lw=0.9,
        alpha=0.9,
        label="USDC/SVB (Tier A, real)",
    )

    ax2.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax2.axhline(-10, color=RED, lw=0.5, linestyle=":", alpha=0.5)
    ax2.set_xlabel("Normalised episode time", fontsize=7)
    ax2.set_ylabel("USDC basis (bps)", fontsize=7)
    ax2.set_title("(b) Mechanism Fingerprints\n(real data only)", fontsize=7, pad=4)
    ax2.tick_params(labelsize=6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=5, loc="lower left", framealpha=0.85, edgecolor="none")

    ax2.annotate(
        "Algorithmic spiral:\nrapid deep collapse",
        xy=(0.25, terra["basis_clip"].min()),
        xytext=(0.32, -220),
        fontsize=5,
        color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.6),
    )
    ax2.annotate(
        "Reserve-bank run:\nsharp then persistent",
        xy=(0.15, svb["basis_clip"].min()),
        xytext=(0.38, -400),
        fontsize=5,
        color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.6),
    )

    fig.tight_layout(pad=0.4)
    for out_path in [
        OUT / "figure_event_universe.png",
        PAPER / "figure_event_universe.png",
    ]:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure_event_universe.png")


def make_latex_table() -> None:
    """Write the 18-event catalogue as a LaTeX longtable fragment."""
    cat = pd.read_csv(
        ROOT / "results/paper_addon/table_14_historical_event_catalog.csv"
    )
    cat["start_dt"] = pd.to_datetime(cat["start"])

    # Short mechanism labels
    mech_short = {
        "Algorithmic / Reflexive": "Algo./Reflex.",
        "Exchange Credit / Liquidity": "Exch.\ Credit",
        "Regulatory / Issuer Winddown": "Regulatory",
        "Fiat-Reserve Bank Shock": "Fiat Reserve",
        "DeFi Pool Imbalance": "DeFi Pool",
        "Collateral / Liquidation": "Collateral",
        "RWA / Niche Stablecoin": "RWA/Niche",
    }
    # Duration short labels
    dur_map = {"days": "days", "hours": "hrs", "weeks": "wks"}

    rows = []
    for _, r in cat.sort_values("start_dt").iterrows():
        depeg = r["max_depeg_bps_est"]
        if abs(depeg) >= 1000:
            depeg_str = f"${depeg/100:+.0f}$\\,pp$^\\dagger$"
        else:
            depeg_str = f"${int(depeg):+d}$"
        if r["data_tier"] in ("B", "C"):
            depeg_str = "est.\ " + depeg_str  # estimated

        mech = mech_short.get(r["mechanism_class"], r["mechanism_class"])
        dur = dur_map.get(r["duration_class"], r["duration_class"])
        coins = r["stablecoins"].split(",")[0].strip()  # first coin only
        name = (
            r["display_name"]
            .replace(" (PRIMARY)", "")
            .replace("/", "/\\allowbreak ")
            .replace("&", "\\&")
        )

        rows.append(
            f"  {name} & {r['start_dt'].strftime('%b~%Y')} & "
            f"\\texttt{{{coins}}} & {mech} & {depeg_str} & "
            f"{dur} & {r['data_tier']}\\\\"
        )

    latex = (
        r"""\begin{table}[t]
\centering
\caption{StressBench event catalogue: all 18 events.
  Max depeg in bps ($+$=above peg); $^\dagger$=percentage-point collapse.
  ``est.''~=~source-verified estimate; Tier~A~=~execution-grade
  (VWAP labels); Tier~B~=~price/liquidity; Tier~C~=~taxonomy only.}
\label{tab:catalogue}
\setlength{\tabcolsep}{2pt}
{\scriptsize
\begin{tabular}{p{2.4cm}rp{0.9cm}p{1.35cm}rrl}
\toprule
\textbf{Event} & \textbf{Date} & \textbf{Coin} &
\textbf{Mechanism} & \textbf{Max depeg} & \textbf{Dur.} & \textbf{Tier}\\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}
}
\end{table}"""
    )

    out_path = ROOT / "results" / "paper_addon" / "table_catalogue_latex.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_figure()
    make_latex_table()
    print("Done.")
