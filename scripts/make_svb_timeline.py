#!/usr/bin/env python3
"""Stress-event timeline: 3-panel SVB window figure (T2.2).
Top: USDC price. Middle: cross-quote basis. Bottom: trade entry markers.
Saves to results/paper/figures/figure_svb_timeline.png
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np, polars as pl, pandas as pd
import lightgbm as lgb

REPO = Path(__file__).parent.parent
DATA = REPO / "data" / "gold" / "dataset.parquet"
OUT  = REPO / "results" / "paper" / "figures" / "figure_svb_timeline.png"

C_ORACLE="#F2A900"; C_META="#2ca02c"; C_PRICE="#d73027"; C_NAVY="#003057"; C_GREY="#888888"

FEATS = ["cross_quote_basis_usdc_bps","cross_quote_basis_usdt_bps",
         "cross_quote_basis_maxabs_bps","cross_quote_basis_primary_bps",
         "spread_bps_mean","depth_bid_10bp_mean","depth_ask_10bp_mean","imbalance_1bp_mean"]

def to_X(sdf):
    return np.nan_to_num(sdf.select([c for c in FEATS if c in sdf.columns]).to_numpy().astype(float), nan=0.0)

def main():
    df = pl.read_parquet(str(DATA))
    train = df.filter(pl.col("split")=="train")
    val   = df.filter(pl.col("split")=="validation")
    test  = df.filter(pl.col("split")=="test")

    # Train meta-label on val primary fires
    X_tr = to_X(train); y_tr = train["label_basis_usdc_1m_gt10bps"].to_numpy().astype(int)
    lgbm = lgb.LGBMClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,n_jobs=4,random_state=42,verbose=-1)
    lgbm.fit(X_tr, y_tr)

    # Meta-label secondary model on Terra/LUNA
    prim_va = val["label_basis_usdc_1m_gt10bps"].to_numpy().astype(bool)
    net_va  = np.nan_to_num(val["net_profit_bps_q10000"].to_numpy().astype(float), nan=0.0)
    meta_X  = to_X(val)[prim_va]
    meta_y  = (net_va[prim_va] > 10).astype(int)
    meta_lgbm = lgb.LGBMClassifier(n_estimators=100,max_depth=3,learning_rate=0.05,n_jobs=4,random_state=42,verbose=-1)
    meta_lgbm.fit(meta_X, meta_y)

    # Test split signals
    prim_te = test["label_basis_usdc_1m_gt10bps"].to_numpy().astype(bool)
    net_te  = np.nan_to_num(test["net_profit_bps_q10000"].to_numpy().astype(float), nan=0.0)
    X_te    = to_X(test)

    # Calibrate meta threshold on val
    meta_probs_va = meta_lgbm.predict_proba(meta_X)[:,1]
    best_t, best_obj = 0.5, -np.inf
    for t in np.linspace(0.1,0.9,80):
        m = meta_probs_va > t
        if m.sum() < 25: continue
        obj = float(np.mean(net_va[prim_va][m]))
        if obj > best_obj: best_obj, best_t = obj, t

    # Calibrate lgbm threshold on val
    lgbm_probs_va = lgbm.predict_proba(to_X(val))[:,1]
    net_va_all = np.nan_to_num(val["net_profit_bps_q10000"].to_numpy().astype(float), nan=0.0)
    best_tl, best_objl = 0.5, -np.inf
    for t in np.linspace(0.05,0.95,100):
        m = lgbm_probs_va > t
        if m.sum() < 25: continue
        obj = float(np.mean(net_va_all[m]))
        if obj > best_objl: best_objl, best_tl = obj, t

    # Signals on test
    oracle_sig = net_te > 10
    rule_sig   = test["label_basis_usdc_1m_gt10bps"].to_numpy().astype(bool)
    meta_probs_te = meta_lgbm.predict_proba(X_te[prim_te])[:,1]
    meta_full = np.zeros(len(test), dtype=bool)
    idx = np.where(prim_te)[0]
    meta_full[idx[meta_probs_te > best_t]] = True

    # Timestamps
    ts_ns = test["ts_1m_ns"].to_numpy()
    ts_dt = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("US/Eastern")
    basis = np.clip(test["cross_quote_basis_usdc_bps"].to_numpy().astype(float), -200, 200)
    price = test["btc_usd_via_usdc"].to_numpy().astype(float) if "btc_usd_via_usdc" in test.columns else test["btc_usd_direct"].to_numpy().astype(float)
    price = np.nan_to_num(price, nan=np.nanmean(price[~np.isnan(price)]))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                              gridspec_kw={"height_ratios":[2,2,1.2]})

    # Top: USDC price
    ax0 = axes[0]
    ax0.plot(ts_dt, price/1000, color=C_NAVY, lw=0.8, alpha=0.85)
    ax0.set_ylabel("BTC/USDC price\n(000s USD)", fontsize=9)
    ax0.grid(alpha=0.15)
    ax0.set_title("SVB Stress Window  (Mar 10-20, 2023, ET)", fontsize=10.5)

    # Middle: basis
    ax1 = axes[1]
    ax1.fill_between(ts_dt, basis, 0, where=basis>0, alpha=0.4, color=C_PRICE, label="USDC basis > 0")
    ax1.fill_between(ts_dt, basis, 0, where=basis<0, alpha=0.3, color=C_NAVY)
    ax1.axhline(10, color=C_PRICE, lw=0.9, ls=":", alpha=0.7, label="10 bps threshold")
    ax1.axhline(-10, color=C_NAVY, lw=0.9, ls=":", alpha=0.7)
    ax1.set_ylabel("Cross-quote\nbasis (bps)", fontsize=9)
    ax1.set_ylim(-150, 150)
    ax1.grid(alpha=0.15)
    ax1.legend(fontsize=7.5, loc="upper left")

    # Bottom: trade markers
    ax2 = axes[2]
    ax2.set_ylabel("Trade entries", fontsize=9)
    y_oracle = np.where(oracle_sig, 0.75, np.nan)
    y_meta   = np.where(meta_full,  0.50, np.nan)
    y_rule   = np.where(rule_sig,   0.25, np.nan)
    ax2.scatter(ts_dt[oracle_sig], y_oracle[oracle_sig], color=C_ORACLE, s=8, marker="|", linewidths=2, label="Oracle")
    ax2.scatter(ts_dt[meta_full],  y_meta[meta_full],   color=C_META,   s=8, marker="|", linewidths=2, label=f"Meta-label (cross-mech.)")
    ax2.scatter(ts_dt[rule_sig],   y_rule[rule_sig],    color=C_PRICE,  s=5, marker="|", linewidths=1, alpha=0.5, label="Price rule (fires everywhere)")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.25,0.50,0.75]); ax2.set_yticklabels(["Rule","Meta","Oracle"], fontsize=8)
    ax2.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax2.grid(alpha=0.1, axis="x")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")
    plt.close(fig)

if __name__ == "__main__":
    main()
