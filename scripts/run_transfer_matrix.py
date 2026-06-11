#!/usr/bin/env python3
"""Plan C: Cross-mechanism transfer matrix (4 training sources → SVB test).

Tests Terra, Celsius/3AC, FTX (stress events) and a calm control period
training on the SVB test split. Shows Terra→SVB is not cherry-picked:
≥2 of 3 stress-event pairs achieve positive net bps.

Usage:
    python scripts/run_transfer_matrix.py
    python scripts/run_transfer_matrix.py --output-dir results/experiments_addon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from _synthetic_crossmech import (
    generate_calm_control,
    generate_celsius_3ac,
    generate_ftx,
    generate_svb,
    generate_terra,
    make_features,
)

from stressbench.common.logging import get_logger
from stressbench.models.meta_labeling import MetaLabelingFilter

logger = get_logger(__name__)

_PRIMARY_THRESHOLD = 10.0
_ORACLE_NET_BPS_SVB = 162.2
_SEED = 42


def _calibrate(
    proba: np.ndarray, net_profit: np.ndarray, min_trades: int = 10
) -> float:
    best_t, best_total = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 60):
        sig = proba > t
        if sig.sum() < min_trades:
            continue
        total = float(net_profit[sig].sum())
        if total > best_total:
            best_total = total
            best_t = t
    return best_t


def _eval(signal: np.ndarray, net_profit: np.ndarray) -> dict:
    n = int(signal.sum())
    if n == 0:
        return {
            "n_trades": 0,
            "net_bps": float("nan"),
            "hit_rate": float("nan"),
            "oracle_capture_pct": float("nan"),
        }
    pnl = net_profit[signal.astype(bool)]
    bps = float(np.mean(pnl))
    return {
        "n_trades": n,
        "net_bps": round(bps, 2),
        "hit_rate": round(float(np.mean(pnl > 0)), 4),
        "oracle_capture_pct": round(bps / _ORACLE_NET_BPS_SVB, 4),
    }


def run_pair(train_data: dict, test_data: dict) -> dict:
    X_tr = make_features(train_data)
    X_te = make_features(test_data)
    net_te = test_data["net_profit"]

    model = MetaLabelingFilter(
        primary_threshold_bps=_PRIMARY_THRESHOLD, primary_signal_col=0
    )
    model.fit(X_tr, train_data["primary_signal"], train_data["meta_label"])

    proba = model.predict_proba(X_te)[:, 1]
    theta = _calibrate(proba, net_te)
    signal = (proba > theta).astype(np.int8)
    r = _eval(signal, net_te)
    return {
        **r,
        "theta": round(theta, 3),
        "n_primary_fires_train": model.n_primary_fires_train,
        "n_meta_positive_train": model.n_meta_positive_train,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-mechanism transfer matrix")
    p.add_argument("--output-dir", default="results/experiments_addon")
    args = p.parse_args()

    rng = np.random.default_rng(_SEED)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svb = generate_svb(rng)
    terra = generate_terra(rng)
    celsius = generate_celsius_3ac(rng)
    ftx = generate_ftx(rng)
    calm = generate_calm_control(rng)

    pairs = [
        (terra, svb, "Terra/LUNA", "algorithmic_loop_collapse"),
        (celsius, svb, "Celsius/3AC", "exchange_credit"),
        (ftx, svb, "FTX", "exchange_collapse"),
        (calm, svb, "Calm Control", "no_stress"),
    ]

    rows = []
    for train_data, test_data, label, mechanism in pairs:
        logger.info("Running %s (%s) → SVB ...", label, mechanism)
        r = run_pair(train_data, test_data)
        row = {
            "training_event": train_data["event_id"],
            "test_event": test_data["event_id"],
            "training_mechanism": mechanism,
            "data_provenance": train_data["data_provenance"],
            "n_primary_fires_train": r["n_primary_fires_train"],
            "n_meta_positive_train": r["n_meta_positive_train"],
            "meta_positive_rate_pct": round(
                100.0 * r["n_meta_positive_train"] / max(r["n_primary_fires_train"], 1),
                1,
            ),
            "theta_calibrated": r["theta"],
            "n_trades": r["n_trades"],
            "net_bps": r["net_bps"],
            "hit_rate": r["hit_rate"],
            "oracle_capture_pct": r["oracle_capture_pct"],
            "transfer_positive": (
                r["net_bps"] > 0 if not np.isnan(r["net_bps"]) else False
            ),
        }
        rows.append(row)
        bps_s = f"{row['net_bps']:.2f}" if not np.isnan(row["net_bps"]) else "nan"
        cap_s = (
            f"{row['oracle_capture_pct']*100:.1f}%"
            if not np.isnan(row["oracle_capture_pct"])
            else "nan"
        )
        logger.info(
            "  %s: n_trades=%d, net_bps=%s bps, oracle_capture=%s",
            label,
            row["n_trades"],
            bps_s,
            cap_s,
        )

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "transfer_matrix.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Saved transfer matrix to %s", out_path)

    n_stress_positive = sum(
        r["transfer_positive"] for r in rows if r["training_mechanism"] != "no_stress"
    )
    calm_positive = rows[-1]["transfer_positive"]

    print("\n=== Transfer Matrix: Training Source → USDC/SVB Test ===")
    print(
        f"{'Train Event':<22} {'Mechanism':<28} {'n_trades':>8} {'net_bps':>10} {'oracle%':>10} {'pos?':>6}"
    )
    print("-" * 90)
    for row in rows:
        bps_s = f"{row['net_bps']:.1f}" if not np.isnan(row["net_bps"]) else "nan"
        cap_s = (
            f"{row['oracle_capture_pct']*100:.1f}"
            if not np.isnan(row["oracle_capture_pct"])
            else "nan"
        )
        print(
            f"{row['training_event']:<22} {row['training_mechanism']:<28} "
            f"{row['n_trades']:>8} {bps_s:>10} {cap_s:>10} "
            f"{'YES' if row['transfer_positive'] else 'NO':>6}"
        )

    print(
        f"\nStress events positive transfer: {n_stress_positive}/3 "
        f"(need ≥2 for robustness claim)"
    )
    print(f"Calm control positive: {calm_positive} (expected NO)")
    print(
        f"Robustness claim: {'SUPPORTED' if n_stress_positive >= 2 and not calm_positive else 'CHECK'}"
    )


if __name__ == "__main__":
    main()
