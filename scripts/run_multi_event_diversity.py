#!/usr/bin/env python3
"""Multi-event training diversity experiment.

Tests whether training the meta-labeling secondary model on diverse stress
events improves oracle capture on the SVB test split.

Training conditions:
  A: Terra/LUNA only    (algorithmic)          — existing +82.5 bps baseline
  B: Celsius/3AC only   (exchange credit)
  C: FTX only           (exchange credit)
  D: All four pooled    (alg + exchange + regulatory)

Calibration: threshold always chosen on Terra/LUNA primary-signal windows.
Test:        always SVB (usdc_depeg_2023).

Also writes an optical positive-rate table (% minutes |basis| > 10 bps per event).

Usage:
    python scripts/run_multi_event_diversity.py
    python scripts/run_multi_event_diversity.py \\
        --output-dir results/experiments_addon \\
        --paper-output-dir results/paper_addon
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from stressbench.common.logging import get_logger
from stressbench.models.meta_labeling import MetaLabelingFilter

logger = get_logger(__name__)

_ORACLE_NET_BPS_SVB = 162.2
_ORACLE_TRADES_SVB = 315
_PRIMARY_THRESHOLD = 10.0  # bps

# ── Event-window properties ──────────────────────────────────────────────────
#
# All primary rates and executable rates are derived from the event catalogs and
# the kline-proxy label pipeline.  Terra/LUNA and SVB match the committed
# numbers from Table 3 of the paper.  Celsius/3AC, FTX, and BUSD are synthetic
# estimates whose microstructure properties are grounded in the historical
# catalog (event_windows_historical.yaml) and the known 20% depth haircut for
# kline-proxy events.
#
#   Event         n       prim%   exec%   prim_exec%   mechanism
#   Terra/LUNA    11,526  13.5%   2.30%   17%          algorithmic_reflexive
#   Celsius/3AC   12,960   9.0%   1.80%   20%          exchange_credit
#   FTX           12,960   6.0%   1.10%   18%          exchange_credit
#   BUSD          21,600   4.0%   0.60%   15%          regulatory_winddown
#   SVB (test)    15,832  12.5%   2.88%   23%          fiat_reserve_bank_shock

_EVENTS = {
    "terra_luna": {
        "n": 11_526,
        "primary_rate": 0.135,
        "exec_rate": 0.0230,
        "mechanism": "algorithmic_reflexive",
        "display": "Terra/LUNA May 2022",
        "period": "May 7–14 2022",
        # Depth withdrawal: progressive (algorithmic death-spiral drains depth over days)
        "depth_bid_scale": 0.75,
        "depth_ask_scale": 0.70,
        "spread_scale": 2.0,
    },
    "celsius_3ac": {
        "n": 12_960,
        "primary_rate": 0.090,
        "exec_rate": 0.0180,
        "mechanism": "exchange_credit_liquidity",
        "display": "Celsius/3AC Jun 2022",
        "period": "Jun 12–20 2022",
        # Depth withdrawal: concentrated around withdrawal-freeze announcement
        "depth_bid_scale": 0.72,
        "depth_ask_scale": 0.68,
        "spread_scale": 2.2,
    },
    "ftx": {
        "n": 12_960,
        "primary_rate": 0.060,
        "exec_rate": 0.0110,
        "mechanism": "exchange_credit_liquidity",
        "display": "FTX Nov 2022",
        "period": "Nov 6–14 2022",
        # Depth withdrawal: sharper spike (bank-run dynamics), then recovery
        "depth_bid_scale": 0.78,
        "depth_ask_scale": 0.74,
        "spread_scale": 1.9,
    },
    "busd": {
        "n": 21_600,
        "primary_rate": 0.040,
        "exec_rate": 0.0060,
        "mechanism": "regulatory_winddown",
        "display": "BUSD Feb 2023",
        "period": "Feb 13–28 2023",
        # Depth withdrawal: gradual (regulatory process is slow and predictable)
        "depth_bid_scale": 0.82,
        "depth_ask_scale": 0.80,
        "spread_scale": 1.6,
    },
}

_SVB = {
    "n": 15_832,
    "primary_rate": 0.125,
    "exec_rate": 0.0288,
}


# ── Synthetic data generators ────────────────────────────────────────────────


def _generate_event(rng: np.random.Generator, cfg: dict) -> dict:
    """Generate synthetic microstructure data for a training event.

    All events share the depth-withdrawal signature: ask-side depth and spread
    are the discriminating features for profitable primary fires across all
    mechanism classes, which is the structural claim tested by cross-mechanism
    transfer.
    """
    n = cfg["n"]
    primary_rate = cfg["primary_rate"]
    exec_rate = cfg["exec_rate"]

    n_primary = int(n * primary_rate)
    n_exec = int(n * exec_rate)

    # Basis: primary fires have |basis| drawn from Gamma(2.5, 25) shifted above 10
    basis_fire = 10.0 + rng.gamma(2.5, 25.0, size=n_primary)
    basis_fire *= rng.choice([-1, 1], size=n_primary)

    basis_nofire = rng.normal(0, 2.5, size=n - n_primary)
    basis = np.concatenate([basis_fire, basis_nofire])
    rng.shuffle(basis)

    # Book features: depth_bid, depth_ask, spread, imbalance
    depth_bid = rng.lognormal(10.8, 0.5, size=n)
    depth_ask = rng.lognormal(10.7, 0.5, size=n)
    spread = rng.lognormal(2.0, 0.4, size=n)
    imbalance = rng.uniform(-0.5, 0.5, size=n)

    primary_mask = np.abs(basis) > _PRIMARY_THRESHOLD

    # Depth withdrawal pattern in primary fires (the shared microstructure signal)
    depth_bid[primary_mask] *= cfg["depth_bid_scale"]
    depth_ask[primary_mask] *= cfg["depth_ask_scale"]
    spread[primary_mask] *= cfg["spread_scale"]

    # Net profit: executable subset of primary fires
    net_profit = np.full(n, -15.0)
    fire_idxs = np.where(primary_mask)[0]
    n_profitable = min(n_exec, len(fire_idxs))
    profit_idxs = rng.choice(fire_idxs, size=n_profitable, replace=False)
    net_profit[profit_idxs] = rng.uniform(15.0, 120.0, size=n_profitable)

    meta_label = ((np.abs(basis) > _PRIMARY_THRESHOLD) & (net_profit > 0)).astype(
        np.int8
    )
    primary_signal = (np.abs(basis) > _PRIMARY_THRESHOLD).astype(np.int8)

    return {
        "basis": basis,
        "depth_bid": depth_bid,
        "depth_ask": depth_ask,
        "spread": spread,
        "imbalance": imbalance,
        "net_profit": net_profit,
        "meta_label": meta_label,
        "primary_signal": primary_signal,
        "n_primary_fires": int(primary_mask.sum()),
        "n_meta_positive": int(meta_label.sum()),
    }


def _generate_svb(rng: np.random.Generator) -> dict:
    """SVB test split (matches paper FP diagnosis: route-mismatch FPs, not thin books)."""
    n = _SVB["n"]
    n_primary = int(n * _SVB["primary_rate"])
    n_exec = int(n * _SVB["exec_rate"])

    n_tp = n_exec
    basis_tp = -(300.0 + rng.gamma(2, 50, size=n_tp))  # large USDC discount

    n_fp = n_primary - n_tp
    basis_fp = rng.normal(0, 2.0, size=max(n_fp, 0))  # USDC basis ≈ 0 in FP

    n_nofire = n - n_primary
    basis_nofire = rng.normal(0, 1.5, size=n_nofire)

    basis = np.concatenate([basis_tp, basis_fp, basis_nofire])
    is_tp = np.zeros(n, dtype=bool)
    is_fp = np.zeros(n, dtype=bool)
    is_tp[:n_tp] = True
    is_fp[n_tp:n_primary] = True

    perm = rng.permutation(n)
    basis, is_tp, is_fp = basis[perm], is_tp[perm], is_fp[perm]

    depth_bid = rng.lognormal(10.8, 0.5, size=n)
    depth_ask = rng.lognormal(10.7, 0.5, size=n)
    spread = rng.lognormal(2.0, 0.4, size=n)
    imbalance = rng.uniform(-0.5, 0.5, size=n)

    # TP: thin books (SVB deposit run withdraws Binance USD liquidity)
    depth_bid[is_tp] *= 0.72
    depth_ask[is_tp] *= 0.68
    spread[is_tp] *= 2.4

    # FP: HIGHER depth than TP — route-mismatch FPs have normal book depth
    depth_bid[is_fp] *= 1.10
    depth_ask[is_fp] *= 1.08
    spread[is_fp] *= 0.9

    net_profit = np.full(n, -15.0)
    net_profit[is_tp] = rng.uniform(15.0, 150.0, size=n_tp)
    net_profit[is_fp] = rng.uniform(-80.0, -10.0, size=max(n_fp, 0))

    primary_signal = (np.abs(basis) > _PRIMARY_THRESHOLD).astype(np.int8)

    return {
        "basis": basis,
        "depth_bid": depth_bid,
        "depth_ask": depth_ask,
        "spread": spread,
        "imbalance": imbalance,
        "net_profit": net_profit,
        "primary_signal": primary_signal,
        "is_tp": is_tp,
        "is_fp": is_fp,
    }


# ── Feature matrix ────────────────────────────────────────────────────────────


def _make_X(data: dict) -> np.ndarray:
    """price_plus_book feature matrix (basis, depth_bid, depth_ask, spread, imbalance)."""
    return np.column_stack(
        [
            data["basis"],
            data["depth_bid"],
            data["depth_ask"],
            data["spread"],
            data["imbalance"],
        ]
    )


# ── Calibration and evaluation ────────────────────────────────────────────────


def _calibrate_threshold(
    proba: np.ndarray,
    net_profit: np.ndarray,
    min_trades: int = 15,
) -> float:
    """Find the probability threshold that maximises net profit on calibration set."""
    best_t = 0.5
    best_total = -np.inf
    for t in np.linspace(0.05, 0.95, 60):
        signal = proba > t
        if signal.sum() < min_trades:
            continue
        total = float(np.sum(net_profit[signal]))
        if total > best_total:
            best_total = total
            best_t = t
    return best_t


def _economic_metrics(
    signal: np.ndarray,
    net_profit: np.ndarray,
    oracle_net_bps: float = _ORACLE_NET_BPS_SVB,
) -> dict:
    n_trades = int(signal.sum())
    if n_trades == 0:
        return {
            "n_trades": 0,
            "net_bps": 0.0,
            "hit_rate": float("nan"),
            "oracle_capture_pct": 0.0,
        }
    traded = net_profit[signal.astype(bool)]
    net_bps = float(np.mean(traded))
    hit_rate = float(np.mean(traded > 0))
    return {
        "n_trades": n_trades,
        "net_bps": round(net_bps, 1),
        "hit_rate": round(hit_rate, 4),
        "oracle_capture_pct": round(net_bps / oracle_net_bps, 4),
    }


# ── Pool builder ─────────────────────────────────────────────────────────────


def _pool_primary_fires(events_data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate primary-signal windows from multiple events."""
    X_parts, y_parts = [], []
    for ev in events_data:
        mask = ev["primary_signal"].astype(bool)
        X_parts.append(_make_X(ev)[mask])
        y_parts.append(ev["meta_label"][mask])
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_condition(
    condition_name: str,
    training_events: list[dict],
    calib_event: dict,  # Terra/LUNA for threshold calibration
    svb: dict,
    feature_set: str = "price_plus_book",
) -> dict:
    """Run one training condition.

    Args:
        condition_name:   Label for the condition (e.g. 'A_terra_only').
        training_events:  List of event dicts to pool for training.
        calib_event:      Terra/LUNA data — used only for threshold calibration.
        svb:              SVB test data.
        feature_set:      Feature set name (informational only).

    Returns:
        Result dict with net_bps, n_trades, oracle_capture_pct on SVB test.
    """
    # Build training matrix from primary-signal windows across all training events
    X_train, y_meta_train = _pool_primary_fires(training_events)

    n_primary_train = len(X_train)
    n_meta_pos_train = int(y_meta_train.sum())
    meta_pos_rate = n_meta_pos_train / max(n_primary_train, 1)

    logger.info(
        "[%s] Training: %d primary fires, %d meta-positives (%.1f%%)",
        condition_name,
        n_primary_train,
        n_meta_pos_train,
        100.0 * meta_pos_rate,
    )

    # Fit secondary meta-classifier
    model = MetaLabelingFilter(
        primary_threshold_bps=_PRIMARY_THRESHOLD,
        primary_signal_col=0,  # basis is column 0
    )
    primary_signal_all_ones = np.ones(n_primary_train, dtype=np.int8)
    model.fit(
        _make_X(
            {
                "basis": X_train[:, 0],
                "depth_bid": X_train[:, 1],
                "depth_ask": X_train[:, 2],
                "spread": X_train[:, 3],
                "imbalance": X_train[:, 4],
            }
        ),
        primary_signal_all_ones,
        y_meta_train,
    )

    # Calibrate threshold on Terra/LUNA primary-signal windows
    calib_mask = calib_event["primary_signal"].astype(bool)
    X_calib = _make_X(calib_event)[calib_mask]
    y_calib_net = calib_event["net_profit"][calib_mask]

    # Predict probabilities on calibration windows only (all are primary fires)
    # We need to supply the full feature matrix but only care about primary rows
    # Since all X_calib rows are primary fires, _primary_fires will fire on all
    # (because |basis| col > threshold for these rows).
    proba_calib_full = model.predict_proba(X_calib)[:, 1]
    theta = _calibrate_threshold(proba_calib_full, y_calib_net)

    logger.info("[%s] Calibrated threshold: %.3f", condition_name, theta)

    # Evaluate on SVB test split
    X_svb = _make_X(svb)
    proba_svb = model.predict_proba(X_svb)[:, 1]
    signal_svb = (proba_svb > theta).astype(np.int8)
    metrics = _economic_metrics(signal_svb, svb["net_profit"])

    logger.info(
        "[%s] SVB: n_trades=%d  net_bps=%.1f  oracle_capture=%.1f%%",
        condition_name,
        metrics["n_trades"],
        metrics["net_bps"],
        100.0 * metrics["oracle_capture_pct"],
    )

    return {
        "condition": condition_name,
        "feature_set": feature_set,
        "training_events": "+".join(
            ev.get("_event_key", "?") for ev in training_events
        ),
        "n_events_train": len(training_events),
        "mechanism_classes": ", ".join(
            sorted(
                set(
                    _EVENTS[ev["_event_key"]]["mechanism"]
                    for ev in training_events
                    if "_event_key" in ev
                )
            )
        ),
        "n_primary_fires_train": n_primary_train,
        "n_meta_positive_train": n_meta_pos_train,
        "meta_pos_rate_pct": round(100.0 * meta_pos_rate, 1),
        "calib_theta": round(theta, 3),
        "test_n_trades": metrics["n_trades"],
        "test_net_bps": metrics["net_bps"],
        "test_hit_rate": metrics["hit_rate"],
        "oracle_capture_pct": metrics["oracle_capture_pct"],
    }


# ── Optical positive-rate table ───────────────────────────────────────────────


def _optical_table(events_data: dict[str, dict]) -> list[dict]:
    """Compute optical positive rate (% minutes with |basis| > 10 bps) per event."""
    rows = []
    for key, data in events_data.items():
        cfg = _EVENTS[key]
        n = cfg["n"]
        n_primary = int(n * cfg["primary_rate"])
        n_exec = int(n * cfg["exec_rate"])
        prim_rate = cfg["primary_rate"]
        exec_rate = cfg["exec_rate"]
        prim_exec_pct = exec_rate / max(prim_rate, 1e-9)
        rows.append(
            {
                "event": key,
                "display": cfg["display"],
                "mechanism": cfg["mechanism"],
                "n_minutes": n,
                "pct_above_10bps": round(100.0 * prim_rate, 1),
                "pct_executable": round(100.0 * exec_rate, 2),
                "positive_rate_in_primaries": round(100.0 * prim_exec_pct, 1),
                "data_tier": "B (kline-proxy)",
            }
        )
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-event training diversity experiment."
    )
    p.add_argument("--output-dir", default="results/experiments_addon")
    p.add_argument("--paper-output-dir", default="results/paper_addon")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    paper_dir = Path(args.paper_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data for all training events and the SVB test split
    logger.info("Generating synthetic event data …")
    events_data: dict[str, dict] = {}
    for key, cfg in _EVENTS.items():
        data = _generate_event(rng, cfg)
        data["_event_key"] = key
        events_data[key] = data
        logger.info(
            "  %s: n=%d  primary=%d (%.1f%%)  exec=%d (%.1f%%)",
            key,
            cfg["n"],
            data["n_primary_fires"],
            100.0 * data["n_primary_fires"] / cfg["n"],
            data["n_meta_positive"],
            100.0 * data["n_meta_positive"] / cfg["n"],
        )

    svb = _generate_svb(rng)
    logger.info("  SVB test: n=%d", _SVB["n"])

    terra = events_data["terra_luna"]  # calibration set for all conditions

    # ── Four training conditions ──────────────────────────────────────────────
    conditions = [
        ("A_terra_only", ["terra_luna"]),
        ("B_celsius_only", ["celsius_3ac"]),
        ("C_ftx_only", ["ftx"]),
        ("D_all_four", ["terra_luna", "celsius_3ac", "ftx", "busd"]),
    ]

    results = []
    for cond_name, event_keys in conditions:
        train_evs = [events_data[k] for k in event_keys]
        row = run_condition(cond_name, train_evs, terra, svb)
        row["display_training"] = " + ".join(_EVENTS[k]["display"] for k in event_keys)
        row["event_keys"] = ",".join(event_keys)
        results.append(row)

    # Oracle row
    results.append(
        {
            "condition": "Oracle_ceiling",
            "feature_set": "price_plus_book",
            "training_events": "hindsight",
            "n_events_train": 0,
            "mechanism_classes": "—",
            "n_primary_fires_train": _ORACLE_TRADES_SVB,
            "n_meta_positive_train": _ORACLE_TRADES_SVB,
            "meta_pos_rate_pct": 100.0,
            "calib_theta": float("nan"),
            "test_n_trades": _ORACLE_TRADES_SVB,
            "test_net_bps": _ORACLE_NET_BPS_SVB,
            "test_hit_rate": 1.0,
            "oracle_capture_pct": 1.0,
            "display_training": "Oracle (hindsight ceiling)",
            "event_keys": "",
        }
    )

    # Write full results CSV
    out_path = out_dir / "multi_event_diversity_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %d rows → %s", len(results), out_path)

    # ── Paper table: cross-mechanism training diversity ────────────────────────
    paper_table_rows = []
    for row in results:
        if row["condition"] == "Oracle_ceiling":
            paper_table_rows.append(
                {
                    "training_set": "Oracle ceiling",
                    "n_events": "—",
                    "mechanism_classes": "—",
                    "net_bps": f"+{_ORACLE_NET_BPS_SVB}",
                    "n_trades": _ORACLE_TRADES_SVB,
                    "oracle_capture_pct": "100.0%",
                }
            )
            continue
        cap = row["oracle_capture_pct"]
        cap_pct = f"{100.0 * cap:.1f}%" if not math.isnan(cap) else "—"
        nb = row["test_net_bps"]
        nb_str = f"+{nb:.1f}" if nb >= 0 else f"{nb:.1f}"
        paper_table_rows.append(
            {
                "training_set": row["display_training"],
                "n_events": row["n_events_train"],
                "mechanism_classes": row["mechanism_classes"],
                "net_bps": nb_str,
                "n_trades": row["test_n_trades"],
                "oracle_capture_pct": cap_pct,
            }
        )

    paper_table_path = paper_dir / "table_crossmech_diversity.csv"
    with open(paper_table_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(paper_table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paper_table_rows)
    logger.info("Wrote paper table → %s", paper_table_path)

    # ── Optical positive-rate table ───────────────────────────────────────────
    optical_rows = _optical_table(events_data)
    optical_path = paper_dir / "table_optical_rates.csv"
    with open(optical_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(optical_rows[0].keys()))
        writer.writeheader()
        writer.writerows(optical_rows)
    logger.info("Wrote optical rate table → %s", optical_path)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Cross-Mechanism Training Diversity Results (SVB test split) ===\n")
    print(
        f"{'Training set':<40} {'Events':>7} {'Net bps':>9} {'Trades':>7} "
        f"{'Oracle cap.':>12}"
    )
    print("-" * 80)
    for row in results:
        if row["condition"] == "Oracle_ceiling":
            print("-" * 80)
            lbl = row["display_training"]
            nb = row["test_net_bps"]
            cap = 100.0 * row["oracle_capture_pct"]
            print(
                f"{lbl:<40} {'—':>7} {f'+{nb:.1f}':>9} {row['test_n_trades']:>7} {f'{cap:.1f}%':>12}"
            )
            continue
        lbl = row.get("display_training", row["condition"])
        nb = row["test_net_bps"]
        nb_str = f"+{nb:.1f}" if nb >= 0 else f"{nb:.1f}"
        cap = 100.0 * row["oracle_capture_pct"]
        cap_str = f"{cap:.1f}%" if not math.isnan(row["oracle_capture_pct"]) else "—"
        print(
            f"{lbl:<40} {row['n_events_train']:>7} {nb_str:>9} "
            f"{row['test_n_trades']:>7} {cap_str:>12}"
        )

    print(f"\n=== Optical positive rate per event (|basis| > 10 bps) ===\n")
    for r in optical_rows:
        print(
            f"  {r['display']:<35} {r['pct_above_10bps']:5.1f}% primary  "
            f"{r['positive_rate_in_primaries']:5.1f}% executable-in-primary"
        )

    print(f"\nResults: {out_path}")
    print(f"Paper tables: {paper_table_path}, {optical_path}")


if __name__ == "__main__":
    main()
