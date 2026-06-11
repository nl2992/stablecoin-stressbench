#!/usr/bin/env python3
"""Compact single-panel event universe figure for 8-page budget."""
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

NAVY = "#003057"
BLUE = "#75B2DD"
LGREY = "#D0D0D0"
GOLD = "#F2A900"

MECH_LABELS = {
    "Algorithmic / Reflexive": "Algorithmic /\nReflexive",
    "Fiat-Reserve Bank Shock": "Fiat-Reserve\nBank Shock",
    "Regulatory / Issuer Winddown": "Regulatory\nWinddown",
    "Exchange Credit / Liquidity": "Exchange Credit\n/ Liquidity",
    "DeFi Pool Imbalance": "DeFi Pool\nImbalance",
    "Collateral / Liquidation": "Collateral /\nLiquidation",
    "RWA / Niche Stablecoin": "RWA / Niche",
}
MECH_ORDER = list(MECH_LABELS.keys())
TIER_COL = {"A": NAVY, "B": BLUE, "C": LGREY}

cat = pd.read_csv(ROOT / "results/paper_addon/table_14_historical_event_catalog.csv")
cat["start_dt"] = pd.to_datetime(cat["start"])
cat["year_frac"] = cat["start_dt"].dt.year + (cat["start_dt"].dt.dayofyear - 1) / 365
cat["mech_idx"] = cat["mechanism_class"].map({v: i for i, v in enumerate(MECH_ORDER)})
cat["abs_depeg"] = cat["max_depeg_bps_est"].abs().clip(upper=12000)
cat["bubble_r"] = np.sqrt(cat["abs_depeg"]).clip(10, 120) * 0.75
cat["tier_1"] = cat["data_tier"].str[0]

fig, ax = plt.subplots(figsize=(3.5, 2.2))

for _, row in cat.iterrows():
    colour = TIER_COL.get(row["tier_1"], LGREY)
    edge = GOLD if row["max_depeg_bps_est"] > 0 else "white"
    ax.scatter(
        row["year_frac"],
        row["mech_idx"],
        s=row["bubble_r"] ** 1.35,
        color=colour,
        edgecolors=edge,
        linewidths=0.6,
        alpha=0.88,
        zorder=3,
    )

# Label key events only
labels = {
    "usdc_svb_2023": ("USDC/SVB", (-0.05, 0.38)),
    "terra_ust_2022": ("Terra/UST", (0.05, -0.42)),
    "ftx_collapse_2022": ("FTX", (0.05, 0.38)),
    "dai_black_thursday_2020": ("DAI BT", (0.05, 0.38)),
}
for _, row in cat.iterrows():
    if row["event_id"] in labels:
        txt, (dx, dy) = labels[row["event_id"]]
        ax.annotate(
            txt,
            xy=(row["year_frac"], row["mech_idx"]),
            xytext=(row["year_frac"] + dx, row["mech_idx"] + dy),
            fontsize=4.2,
            color="#333",
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
        )

ax.set_yticks(range(len(MECH_ORDER)))
ax.set_yticklabels([MECH_LABELS[m] for m in MECH_ORDER], fontsize=5)
ax.set_xlabel("Year", fontsize=6)
ax.set_xlim(2019.7, 2024.2)
ax.set_xticks([2020, 2021, 2022, 2023, 2024])
ax.set_xticklabels(["'20", "'21", "'22", "'23", "'24"], fontsize=5.5)
ax.tick_params(left=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")

leg = [
    mpatches.Patch(color=NAVY, label="Tier A"),
    mpatches.Patch(color=BLUE, label="Tier B"),
    mpatches.Patch(color=LGREY, label="Tier C"),
    mpatches.Patch(facecolor="white", edgecolor=GOLD, linewidth=0.8, label="Above-peg"),
]
ax.legend(
    handles=leg,
    fontsize=4.5,
    loc="lower right",
    framealpha=0.85,
    edgecolor="none",
    ncol=2,
)

fig.tight_layout(pad=0.3)
for p in [
    OUT / "figure_event_universe_compact.png",
    PAPER / "figure_event_universe_compact.png",
]:
    fig.savefig(p, dpi=220, bbox_inches="tight")
plt.close(fig)
print("Saved figure_event_universe_compact.png")
