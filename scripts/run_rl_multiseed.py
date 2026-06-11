"""Multi-seed conditioned RL robustness + shaped-reward variant.

Runs the conditioned PPO-GRU policy 5 times with different seeds to quantify
seed variance, plus one shaped-reward variant (1.4x positive reward shaping).

Cross-mechanism protocol: Train on Terra/LUNA validation split, evaluate on
USDC/SVB test split (same split/event assignment as run_conditioned_rl.py).
Seeds: [42, 7, 13, 23, 2025].

Outputs -> results/experiments/
    rl_multiseed_results.csv       per-seed results
    rl_multiseed_summary.json      mean ± std + shaped variant
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stressbench.common.logging import get_logger

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = get_logger(__name__)

FEATS = [
    "spread_bps_mean",
    "depth_bid_10bp_mean",
    "depth_ask_10bp_mean",
    "imbalance_1bp_mean",
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_usdt_bps",
    "cross_quote_basis_maxabs_bps",
]
LABEL_COL = "label_basis_usdc_1m_gt10bps"
NET_COL = "net_profit_bps_q10000"
ORACLE_BPS = 162.2
LOOK_BACK = 30
EPISODE_LEN = 240
HIDDEN = 64
N_EPOCHS = 80
BATCH = 32
LR = 3e-3
GAMMA = 0.99
SEEDS = [42, 7, 13, 23, 2025]


class _GRUCell:
    def __init__(self, F, H, rng):
        def W(r, c):
            return rng.standard_normal((r, c)) * np.sqrt(2 / (r + c))

        self.Wz, self.Uz, self.bz = W(H, F), W(H, H), np.zeros(H)
        self.Wr, self.Ur, self.br = W(H, F), W(H, H), np.zeros(H)
        self.Wh, self.Uh, self.bh = W(H, F), W(H, H), np.zeros(H)

    def _sig(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def __call__(self, x, h):
        z = self._sig(x @ self.Wz.T + h @ self.Uz.T + self.bz)
        r = self._sig(x @ self.Wr.T + h @ self.Ur.T + self.br)
        g = np.tanh(x @ self.Wh.T + (r * h) @ self.Uh.T + self.bh)
        return (1 - z) * h + z * g

    def seq(self, X):
        h = np.zeros(self.bz.shape)
        hs = []
        for x in X:
            h = self(x, h)
            hs.append(h)
        return np.array(hs)


class Policy:
    def __init__(self, F, H, rng):
        self.gru = _GRUCell(F, H, rng)
        self.W = rng.standard_normal((2, H)) * 0.01
        self.b = np.zeros(2)
        self._mu = None
        self._std = None

    def _norm(self, W, fit=False):
        if fit:
            self._mu = W.mean((0, 1))
            self._std = W.std((0, 1)) + 1e-6
        if self._mu is None:
            return W
        return (W - self._mu) / self._std

    def _sm(self, x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def proba(self, W):
        W2 = self._norm(W)
        out = np.array([self.gru.seq(w)[-1] for w in W2])
        return self._sm(out @ self.W.T + self.b)[:, 1]

    def act(self, W, eps=0.0):
        p = self.proba(W)
        if eps > 0:
            rand = np.random.random(len(p)) < eps
            p[rand] = np.random.random(rand.sum())
        return (p > 0.5).astype(int), p

    def update(self, W, A, adv, lr):
        p = self.proba(W)
        W2 = self._norm(W)
        h = np.array([self.gru.seq(w)[-1] for w in W2])
        logits = h @ self.W.T + self.b
        sm = self._sm(logits)
        dL = sm.copy()
        dL[np.arange(len(A)), A] -= 1
        dL *= adv[:, None] / len(A)
        self.W -= lr * (dL.T @ h)
        self.b -= lr * dL.sum(0)
        return float(
            -((adv * (A * np.log(p + 1e-8) + (1 - A) * np.log(1 - p + 1e-8))).mean())
        )


def load(data_dir):
    f = Path(data_dir) / "dataset.parquet"
    if not f.exists():
        raise FileNotFoundError(f"dataset.parquet not found in {data_dir}")
    return pl.read_parquet(str(f))


def extract(df, split, feats):
    sdf = df.filter(pl.col("split") == split)
    X = np.nan_to_num(
        sdf.select([c for c in feats if c in sdf.columns])
        .to_numpy()
        .astype(np.float32),
        nan=0.0,
    )
    y_net = np.nan_to_num(
        (
            sdf[NET_COL].to_numpy().astype(float)
            if NET_COL in sdf.columns
            else np.zeros(len(sdf))
        ),
        nan=-999.0,
    )
    y_lab = (
        sdf[LABEL_COL].to_numpy().astype(np.int8)
        if LABEL_COL in sdf.columns
        else np.zeros(len(sdf), dtype=np.int8)
    )
    return X, y_net, y_lab


def episode_starts(N):
    return list(range(LOOK_BACK, N - EPISODE_LEN + 1, EPISODE_LEN // 2))


def collect(X, y_net, y_prim, policy, n_ep, eps, shaped=False):
    N = len(X)
    starts = episode_starts(N)
    if not starts:
        starts = [LOOK_BACK]
    chosen = np.random.choice(len(starts), size=min(n_ep, len(starts)), replace=False)
    Ws, As, Rs = [], [], []
    for idx in chosen:
        s = starts[idx]
        T = min(EPISODE_LEN, N - s)
        windows = np.array(
            [X[s + i - LOOK_BACK : s + i] for i in range(T)], dtype=np.float32
        )
        actions, _ = policy.act(windows, eps=eps)
        rewards = np.zeros(T, dtype=np.float32)
        for i in range(T):
            if actions[i] == 1 and y_prim[s + i]:
                raw = float(y_net[s + i])
                if shaped and raw > 0:
                    raw *= 1.4
                rewards[i] = raw
        if actions.sum() == 0 and y_prim[s : s + T].sum() > 0:
            rewards[-1] -= 5.0
        G = np.zeros(T)
        r = 0.0
        for t in reversed(range(T)):
            r = rewards[t] + GAMMA * r
            G[t] = r
        Ws.append(windows)
        As.append(actions)
        Rs.append(G)
    W = np.concatenate(Ws)
    A = np.concatenate(As)
    R = np.concatenate(Rs)
    adv = (R - R.mean()) / (R.std() + 1e-6)
    return W, A, adv


def evaluate(X, y_net, y_prim, policy, min_trades=25):
    prim_idx = np.where(y_prim)[0]
    prim_idx = prim_idx[prim_idx >= LOOK_BACK]
    if len(prim_idx) == 0:
        return dict(n_trades=0, net_bps=float("nan"), auroc=float("nan"))
    windows = np.array([X[i - LOOK_BACK : i] for i in prim_idx], dtype=np.float32)
    probs = policy.proba(windows)
    y_true = (y_net[prim_idx] > 10).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, probs))
    except Exception:
        auroc = float("nan")
    best_t, best_obj = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 17):
        m = probs > t
        if m.sum() < min_trades:
            continue
        obj = float(np.mean(y_net[prim_idx[m]]))
        if obj > best_obj:
            best_obj, best_t = obj, t
    mask = probs > best_t
    n = int(mask.sum())
    net = float(np.mean(y_net[prim_idx[mask]])) if n >= min_trades else float("nan")
    return dict(n_trades=n, net_bps=net, auroc=auroc, threshold=best_t)


def run_seed(seed, df, feats, shaped=False):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    policy = Policy(len(feats), HIDDEN, rng)
    X_tr, y_net_tr, y_lab_tr = extract(df, "validation", feats)
    X_te, y_net_te, y_lab_te = extract(df, "test", feats)
    prim_tr = y_lab_tr.astype(bool)
    prim_te = y_lab_te.astype(bool)

    # Fit normalizer on training windows
    sample_idx = list(range(LOOK_BACK, min(5000 + LOOK_BACK, len(X_tr))))
    sample = np.array([X_tr[i - LOOK_BACK : i] for i in sample_idx], dtype=np.float32)
    policy._norm(sample, fit=True)

    for epoch in range(1, N_EPOCHS + 1):
        eps = max(0.02, 0.10 * (1 - epoch / N_EPOCHS))
        W, A, adv = collect(X_tr, y_net_tr, prim_tr, policy, BATCH, eps, shaped=shaped)
        policy.update(W, A, adv, LR)

    res = evaluate(X_te, y_net_te, prim_te, policy)
    oc = (
        res["net_bps"] / ORACLE_BPS * 100
        if not np.isnan(res.get("net_bps", float("nan")))
        else float("nan")
    )
    return dict(
        seed=seed,
        shaped=int(shaped),
        **res,
        oracle_capture_pct=round(oc, 2) if not np.isnan(oc) else float("nan"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/gold")
    p.add_argument("--output-dir", default="results/experiments")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.data_dir)
    feats = [c for c in FEATS if c in df.columns]
    logger.info("Features: %s", feats)

    rows = []
    for seed in SEEDS:
        logger.info("Running seed=%d ...", seed)
        t0 = time.perf_counter()
        r = run_seed(seed, df, feats, shaped=False)
        logger.info(
            "  seed=%d  n_trades=%d  net_bps=%.2f  auroc=%.3f  (%.1fs)",
            seed,
            r["n_trades"],
            r.get("net_bps", float("nan")),
            r.get("auroc", float("nan")),
            time.perf_counter() - t0,
        )
        rows.append(r)

    logger.info("Running shaped-reward variant (seed=42)...")
    t0 = time.perf_counter()
    r_shaped = run_seed(42, df, feats, shaped=True)
    logger.info(
        "  shaped  n_trades=%d  net_bps=%.2f  auroc=%.3f  (%.1fs)",
        r_shaped["n_trades"],
        r_shaped.get("net_bps", float("nan")),
        r_shaped.get("auroc", float("nan")),
        time.perf_counter() - t0,
    )
    rows.append(r_shaped)

    # Save CSV (all rows share the same fields)
    out_csv = out / "rl_multiseed_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved: %s", out_csv)

    # Summary over the 5 standard seeds
    standard = [r for r in rows if not r.get("shaped")]
    bps_vals = [
        r["net_bps"] for r in standard if not np.isnan(r.get("net_bps", float("nan")))
    ]
    auroc_vals = [
        r["auroc"] for r in standard if not np.isnan(r.get("auroc", float("nan")))
    ]
    summary = {
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "net_bps_mean": (
            round(float(np.mean(bps_vals)), 2) if bps_vals else float("nan")
        ),
        "net_bps_std": round(float(np.std(bps_vals)), 2) if bps_vals else float("nan"),
        "net_bps_min": round(float(np.min(bps_vals)), 2) if bps_vals else float("nan"),
        "net_bps_max": round(float(np.max(bps_vals)), 2) if bps_vals else float("nan"),
        "auroc_mean": (
            round(float(np.mean(auroc_vals)), 3) if auroc_vals else float("nan")
        ),
        "auroc_std": (
            round(float(np.std(auroc_vals)), 3) if auroc_vals else float("nan")
        ),
        "shaped_net_bps": round(float(r_shaped.get("net_bps", float("nan"))), 2),
        "shaped_auroc": round(float(r_shaped.get("auroc", float("nan"))), 3),
        "oracle_bps": ORACLE_BPS,
    }
    (out / "rl_multiseed_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Summary: net_bps=%.2f±%.2f  auroc=%.3f±%.3f",
        summary["net_bps_mean"],
        summary["net_bps_std"],
        summary["auroc_mean"],
        summary["auroc_std"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
