#!/usr/bin/env python3
"""PPO-GRU RL execution policy for stablecoin arbitrage timing.

Cross-mechanism protocol:
  Train: Terra/LUNA validation split (N=11,526)
  Evaluate: USDC/SVB test split (N=15,832)

Algorithm: REINFORCE with baseline (equivalent to PPO with one epoch
per batch when framed as weighted policy gradient).

The RL framing tests whether sequential timing — wait for depth
withdrawal to mature, then enter — resolves the 40-80 bps timing
component of the oracle gap identified in §6.8.

Usage:
    python scripts/train_rl_agent.py --data-dir data/gold
    python scripts/train_rl_agent.py --data-dir data/gold --verbose
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rl_agent")

# ---------------------------------------------------------------------------
# Feature and task configuration (matches train_temporal_model.py)
# ---------------------------------------------------------------------------

MICROSTRUCTURE_FEATURES = [
    "spread_bps_mean",
    "depth_bid_10bp_mean",
    "depth_ask_10bp_mean",
    "imbalance_1bp_mean",
    "cross_quote_basis_usdc_bps",
    "cross_quote_basis_usdt_bps",
    "cross_quote_basis_maxabs_bps",
]

LABEL_COL = "label_basis_usdc_1m_gt10bps"
NET_PROFIT_COL = "net_profit_bps_q10000"
ORACLE_NET_BPS = 162.2

LOOK_BACK = 30
EPISODE_LEN = 240   # 4-hour episodes
HIDDEN_SIZE = 64
N_EPOCHS = 80
EPISODE_BATCH = 32  # episodes per training batch
LR = 3e-3
GAMMA = 0.99
EPSILON = 0.10      # exploration rate (decays to 0.02)
MIN_TRADES = 25


# ---------------------------------------------------------------------------
# Synthetic data fallback (mirrors train_temporal_model.py)
# ---------------------------------------------------------------------------

def _generate_synthetic_data() -> pl.DataFrame:
    N_TRAIN, N_VAL, N_TEST = 28_776, 11_526, 15_832
    N = N_TRAIN + N_VAL + N_TEST
    rng = np.random.default_rng(42)

    basis_usdc = rng.normal(0, 8, N).cumsum() * 0.08
    basis_usdc -= basis_usdc.mean()
    basis_usdc = np.clip(basis_usdc, -500, 500)
    spike_mask = rng.uniform(size=N) < 0.03
    basis_usdc[spike_mask] += rng.choice([-1, 1], size=spike_mask.sum()) * rng.uniform(15, 80, spike_mask.sum())
    basis_usdt = basis_usdc * 0.6 + rng.normal(0, 4, N)
    basis_maxabs = np.maximum(np.abs(basis_usdc), np.abs(basis_usdt))
    spread = np.abs(rng.normal(2.5, 1.5, N)) + 0.5
    depth_bid = np.abs(rng.normal(65_000, 20_000, N))
    depth_ask = np.abs(rng.normal(63_000, 19_000, N))
    imbalance = np.clip(rng.normal(0, 0.3, N), -1, 1)

    log_odds = 0.18 * basis_usdc - 0.04 * spread + 0.02 * imbalance * 10 + rng.normal(0, 1.1, N)
    prob = 1 / (1 + np.exp(-log_odds))
    label = (rng.uniform(size=N) < prob).astype(np.int32)
    net_profit = np.where(label == 1, rng.normal(45, 60, N), rng.normal(-35, 25, N))

    split = ["train"] * N_TRAIN + ["validation"] * N_VAL + ["test"] * N_TEST
    return pl.DataFrame({
        "spread_bps_mean": spread,
        "depth_bid_10bp_mean": depth_bid,
        "depth_ask_10bp_mean": depth_ask,
        "imbalance_1bp_mean": imbalance,
        "cross_quote_basis_usdc_bps": basis_usdc,
        "cross_quote_basis_usdt_bps": basis_usdt,
        "cross_quote_basis_maxabs_bps": basis_maxabs,
        LABEL_COL: label,
        NET_PROFIT_COL: net_profit,
        "split": split,
    })


def load_dataset(data_dir: Path) -> pl.DataFrame:
    p = data_dir / "dataset.parquet"
    if p.exists() and p.stat().st_size > 1000:
        logger.info("Loading %s", p)
        return pl.read_parquet(str(p))
    logger.warning("No dataset found — generating synthetic data")
    return _generate_synthetic_data()


# ---------------------------------------------------------------------------
# NumPy GRU cell (reused from train_temporal_model.py)
# ---------------------------------------------------------------------------

class NumpyGRUCell:
    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator):
        self.input_size = input_size
        self.hidden_size = hidden_size

        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-lim, lim, (fan_out, fan_in)).astype(np.float32)

        h, d = hidden_size, input_size
        self.W_z = xavier(d, h); self.U_z = xavier(h, h); self.b_z = np.zeros(h, np.float32)
        self.W_r = xavier(d, h); self.U_r = xavier(h, h); self.b_r = np.zeros(h, np.float32)
        self.W_h = xavier(d, h); self.U_h = xavier(h, h); self.b_h = np.zeros(h, np.float32)

    @staticmethod
    def _sigmoid(x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward_sequence(self, x_seq: np.ndarray) -> np.ndarray:
        batch, T, _ = x_seq.shape
        h = np.zeros((batch, self.hidden_size), dtype=np.float32)
        for t in range(T):
            x = x_seq[:, t, :]
            z = self._sigmoid(x @ self.W_z.T + h @ self.U_z.T + self.b_z)
            r = self._sigmoid(x @ self.W_r.T + h @ self.U_r.T + self.b_r)
            ht = np.tanh(x @ self.W_h.T + (r * h) @ self.U_h.T + self.b_h)
            h = (1 - z) * h + z * ht
        return h

    @property
    def params(self):
        return [self.W_z, self.U_z, self.b_z, self.W_r, self.U_r, self.b_r,
                self.W_h, self.U_h, self.b_h]


# ---------------------------------------------------------------------------
# GRU Policy (actor-only, binary action: enter=1, wait=0)
# ---------------------------------------------------------------------------

class GRUPolicy:
    """GRU encoder + linear softmax head. Supports REINFORCE gradient update."""

    def __init__(self, input_size: int, hidden_size: int = 64, rng=None):
        if rng is None:
            rng = np.random.default_rng(42)
        self._rng = rng
        self._gru = NumpyGRUCell(input_size, hidden_size, rng)
        lim = np.sqrt(6.0 / (hidden_size + 2))
        self._W_out = rng.uniform(-lim, lim, (2, hidden_size)).astype(np.float32)
        self._b_out = np.zeros(2, dtype=np.float32)
        self._scaler = StandardScaler()
        self._fitted = False
        # Adam state
        self._adam_t = 0
        self._m = [np.zeros_like(p) for p in self._all_params()]
        self._v = [np.zeros_like(p) for p in self._all_params()]

    def _all_params(self):
        return self._gru.params + [self._W_out, self._b_out]

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def _normalize(self, X_win: np.ndarray, fit: bool = False) -> np.ndarray:
        N, T, F = X_win.shape
        Xf = X_win.reshape(-1, F)
        if fit:
            self._scaler.fit(Xf)
            self._fitted = True
        if not self._fitted:
            return X_win
        return self._scaler.transform(Xf).reshape(N, T, F).astype(np.float32)

    def predict_proba(self, X_win: np.ndarray) -> np.ndarray:
        """Return (N, 2) action probability array."""
        Xn = self._normalize(X_win, fit=not self._fitted)
        h = self._gru.forward_sequence(Xn)        # (N, H)
        logits = h @ self._W_out.T + self._b_out  # (N, 2)
        return self._softmax(logits)               # (N, 2)

    def sample_action(self, X_win: np.ndarray, epsilon: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Sample actions with ε-greedy exploration. Returns (actions, log_probs)."""
        probs = self.predict_proba(X_win)          # (N, 2)
        N = len(probs)
        if epsilon > 0:
            explore = self._rng.uniform(size=N) < epsilon
            random_actions = self._rng.integers(0, 2, size=N)
            greedy_actions = probs.argmax(axis=-1)
            actions = np.where(explore, random_actions, greedy_actions)
        else:
            # Stochastic sampling proportional to probabilities
            actions = np.array([
                self._rng.choice(2, p=probs[i]) for i in range(N)
            ])
        log_probs = np.log(probs[np.arange(N), actions] + 1e-7)
        return actions.astype(np.int8), log_probs

    def reinforce_update(
        self,
        X_win: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        lr: float = 3e-3,
    ) -> float:
        """REINFORCE gradient update: θ += lr * E[A_t * ∇ log π(a_t|s_t)]."""
        Xn = self._normalize(X_win, fit=False)
        N, T, F = Xn.shape
        h = self._gru.forward_sequence(Xn)
        logits = h @ self._W_out.T + self._b_out
        probs = self._softmax(logits)

        # Policy gradient: d_logit = advantage * (action_indicator - probs)
        action_onehot = np.zeros_like(probs)
        action_onehot[np.arange(N), actions.astype(int)] = 1.0
        d_logit = (advantages[:, None] * (action_onehot - probs)) / N  # (N, 2)

        # Loss for logging (negative expected reward)
        loss = -float(np.mean(advantages * np.log(probs[np.arange(N), actions.astype(int)] + 1e-7)))

        # Gradients for output layer
        dW_out = d_logit.T @ h    # (2, H)
        db_out = d_logit.sum(0)   # (2,)
        d_h = d_logit @ self._W_out  # (N, H)

        # One-step TBPTT through GRU (same truncation as NumpyGRUClassifier)
        gru = self._gru
        t_last = T - 1
        x = Xn[:, t_last, :]
        # Recompute last-step intermediates
        hprev = self._gru.forward_sequence(Xn[:, :t_last, :]) if t_last > 0 else np.zeros((N, gru.hidden_size), np.float32)
        z = NumpyGRUCell._sigmoid(x @ gru.W_z.T + hprev @ gru.U_z.T + gru.b_z)
        r = NumpyGRUCell._sigmoid(x @ gru.W_r.T + hprev @ gru.U_r.T + gru.b_r)
        ht = np.tanh(x @ gru.W_h.T + (r * hprev) @ gru.U_h.T + gru.b_h)

        d_ht = d_h * z * (1.0 - ht ** 2)
        d_z  = d_h * (ht - hprev) * z * (1.0 - z)
        d_r  = (d_ht @ gru.U_h) * hprev * r * (1.0 - r)

        grads = [
            d_z.T @ x, d_z.T @ hprev, d_z.sum(0),
            d_r.T @ x, d_r.T @ hprev, d_r.sum(0),
            d_ht.T @ x, d_ht.T @ (r * hprev), d_ht.sum(0),
            dW_out, db_out,
        ]

        # Adam update
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam_t += 1
        t = self._adam_t
        for i, (p, g) in enumerate(zip(self._all_params(), grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            p += lr * (self._m[i] / (1 - beta1 ** t)) / (np.sqrt(self._v[i] / (1 - beta2 ** t)) + eps)

        return loss


# ---------------------------------------------------------------------------
# MDP Environment
# ---------------------------------------------------------------------------

class StressBenchMDPEnv:
    """Episode-based MDP wrapping the benchmark's per-minute data.

    Episodes: fixed-length windows (EPISODE_LEN minutes) drawn from
    the training split in chronological order.

    State: 30-minute look-back window of microstructure features.
    Action: 0=wait, 1=enter arbitrage.
    Reward: net_profit_bps if entering, 0 if waiting.
    """

    def __init__(
        self,
        X: np.ndarray,           # (N, n_features) chronological
        y_net: np.ndarray,       # (N,) net profit bps
        y_label: np.ndarray,     # (N,) binary profitable label
        look_back: int = 30,
        episode_len: int = EPISODE_LEN,
    ):
        self.X = X
        self.y_net = y_net
        self.y_label = y_label
        self.look_back = look_back
        self.episode_len = episode_len
        self._N = len(X)
        self._ep_start = 0
        self._step = 0

    def _valid_starts(self):
        """Starting indices for episodes (need look_back context)."""
        return list(range(self.look_back, self._N - self.episode_len + 1, self.episode_len // 2))

    def reset_episode(self, start_idx: int | None = None) -> None:
        if start_idx is None:
            valid = self._valid_starts()
            start_idx = valid[np.random.randint(len(valid))]
        self._ep_start = start_idx
        self._step = 0

    def get_windows(self) -> np.ndarray:
        """Return all windows for the current episode: (episode_len, look_back, F)."""
        T = min(self.episode_len, self._N - self._ep_start)
        windows = []
        for i in range(T):
            t = self._ep_start + i
            win_start = t - self.look_back
            windows.append(self.X[win_start:t, :])
        return np.array(windows, dtype=np.float32)  # (T, look_back, F)

    def get_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Compute rewards for episode given actions (0=wait, 1=enter)."""
        T = min(self.episode_len, self._N - self._ep_start)
        rewards = np.zeros(T, dtype=np.float32)
        for i in range(T):
            t = self._ep_start + i
            if actions[i] == 1:
                rewards[i] = float(self.y_net[t])
        # Penalty if agent never entered during a window with profitable minutes
        if actions.sum() == 0:
            ep_slice = slice(self._ep_start, self._ep_start + T)
            if self.y_label[ep_slice].sum() > 0:
                rewards[-1] -= 10.0  # discourage pure NoTrade
        return rewards


def compute_returns(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Compute discounted returns G_t = sum_{k=t}^{T} gamma^(k-t) r_k."""
    G = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def collect_batch(
    env: StressBenchMDPEnv,
    policy: GRUPolicy,
    n_episodes: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect n_episodes and return (windows, actions, advantages)."""
    all_windows, all_actions, all_returns = [], [], []

    starts = env._valid_starts()
    chosen = np.random.choice(len(starts), size=min(n_episodes, len(starts)), replace=False)

    for idx in chosen:
        env.reset_episode(starts[idx])
        windows = env.get_windows()               # (T, look_back, F)
        actions, _ = policy.sample_action(windows, epsilon=epsilon)
        rewards = env.get_rewards(actions)
        returns = compute_returns(rewards)
        all_windows.append(windows)
        all_actions.append(actions)
        all_returns.append(returns)

    W = np.concatenate(all_windows)   # (total_steps, look_back, F)
    A = np.concatenate(all_actions)
    R = np.concatenate(all_returns)

    # Baseline-subtracted advantage
    baseline = R.mean()
    std = R.std() + 1e-6
    advantages = (R - baseline) / std

    return W, A, advantages


def calibrate_threshold(
    probs: np.ndarray,
    y_net: np.ndarray,
    min_trades: int = 25,
) -> float:
    best_t, best_total = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 17):
        mask = probs > t
        if mask.sum() < min_trades:
            continue
        total = float(y_net[mask].sum())
        if total > best_total:
            best_total = total
            best_t = t
    return best_t


def evaluate_policy(
    policy: GRUPolicy,
    X: np.ndarray,
    y_net: np.ndarray,
    y_label: np.ndarray,
    look_back: int,
    threshold: float,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate policy on a split, compute economic and ML metrics."""
    N = len(X)
    if N <= look_back:
        return {}

    # Build all windows
    windows = np.array(
        [X[t - look_back:t] for t in range(look_back, N)],
        dtype=np.float32,
    )  # (N-look_back, look_back, F)

    probs = policy.predict_proba(windows)[:, 1]  # enter probability
    aligned_label = y_label[look_back:]
    aligned_net   = y_net[look_back:]

    signal = (probs > threshold).astype(np.int8)
    n_trades = int(signal.sum())

    if n_trades == 0:
        net_bps = float("nan")
        auroc = float("nan")
        hit_rate = float("nan")
        oracle_cap = float("nan")
    else:
        net_bps = float(aligned_net[signal == 1].mean())
        hit_rate = float((aligned_net[signal == 1] > 0).mean())
        try:
            auroc = float(roc_auc_score(aligned_label, probs))
        except ValueError:
            auroc = float("nan")
        oracle_cap = net_bps / ORACLE_NET_BPS * 100 if not np.isnan(net_bps) else float("nan")

    logger.info(
        "[%s] threshold=%.2f  n_trades=%d  net_bps=%.1f  AUROC=%.3f  oracle_cap=%.1f%%",
        split_name, threshold, n_trades,
        net_bps if not np.isnan(net_bps) else -9999,
        auroc if not np.isnan(auroc) else -9999,
        oracle_cap if not np.isnan(oracle_cap) else -9999,
    )

    return {
        "split": split_name,
        "n_trades": n_trades,
        "net_bps_captured": net_bps,
        "auroc": auroc,
        "hit_rate_above_cost": hit_rate,
        "oracle_capture_pct": oracle_cap,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RL execution policy (PPO-GRU, cross-mechanism)")
    parser.add_argument("--data-dir", default="data/gold")
    parser.add_argument("--output-dir", default="results/experiments")
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    logger.info("=" * 60)
    logger.info("RL Execution Policy — StressBench (cross-mechanism)")
    logger.info("  Train: Terra/LUNA (validation split)")
    logger.info("  Eval : USDC/SVB (test split)")
    logger.info("  Algo : REINFORCE with baseline (PPO-equivalent)")
    logger.info("=" * 60)

    # Load data
    df = load_dataset(Path(args.data_dir))
    logger.info("Dataset: %d rows, %d cols", df.height, df.width)

    feature_cols = [c for c in MICROSTRUCTURE_FEATURES if c in df.columns]
    if len(feature_cols) < 3:
        feature_cols = [c for c in df.columns if c in MICROSTRUCTURE_FEATURES]
    if not feature_cols:
        feature_cols = MICROSTRUCTURE_FEATURES[:3]
        logger.warning("Using first 3 features as fallback")
    logger.info("Features: %s", feature_cols)

    def _extract(split: str):
        sdf = df.filter(pl.col("split") == split)
        X = sdf.select(feature_cols).to_numpy().astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        y_net = sdf[NET_PROFIT_COL].to_numpy().astype(np.float64) if NET_PROFIT_COL in sdf.columns else np.zeros(len(sdf))
        y_label = sdf[LABEL_COL].to_numpy().astype(np.int8) if LABEL_COL in sdf.columns else np.zeros(len(sdf), dtype=np.int8)
        return X, np.nan_to_num(y_net, nan=-999.0), y_label

    X_train, y_net_train, y_label_train = _extract("validation")   # Terra/LUNA
    X_test,  y_net_test,  y_label_test  = _extract("test")         # SVB

    logger.info("Train (Terra/LUNA): %d rows, pos=%.2f%%", len(X_train), y_label_train.mean() * 100)
    logger.info("Test  (SVB):        %d rows, pos=%.2f%%", len(X_test),  y_label_test.mean() * 100)

    # Initialize policy
    rng = np.random.default_rng(42)
    policy = GRUPolicy(input_size=len(feature_cols), hidden_size=HIDDEN_SIZE, rng=rng)

    # Fit scaler on training features (windows)
    sample_windows = np.array(
        [X_train[t - LOOK_BACK:t] for t in range(LOOK_BACK, min(5000 + LOOK_BACK, len(X_train)))],
        dtype=np.float32,
    )
    policy._normalize(sample_windows, fit=True)

    # MDP environment on training split (Terra/LUNA)
    env = StressBenchMDPEnv(X_train, y_net_train, y_label_train, look_back=LOOK_BACK)

    valid_starts = env._valid_starts()
    logger.info("Training episodes available: %d", len(valid_starts))

    # Training loop
    episode_returns = []
    epsilon = EPSILON

    t0 = time.perf_counter()
    for epoch in range(1, args.n_epochs + 1):
        epsilon = max(0.02, EPSILON * (1 - epoch / args.n_epochs))
        W, A, advantages = collect_batch(env, policy, n_episodes=EPISODE_BATCH, epsilon=epsilon)

        loss = policy.reinforce_update(W, A, advantages, lr=LR)

        ep_ret = float(advantages.mean())
        episode_returns.append(ep_ret)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t0
            logger.info(
                "Epoch %3d/%d  loss=%.4f  mean_adv=%.3f  eps=%.3f  %.1fs",
                epoch, args.n_epochs, loss, ep_ret, epsilon, elapsed,
            )

    # Calibrate threshold on training (Terra/LUNA) split
    logger.info("\nCalibrating threshold on Terra/LUNA hold-out...")
    cal_windows = np.array(
        [X_train[t - LOOK_BACK:t] for t in range(LOOK_BACK, len(X_train))],
        dtype=np.float32,
    )
    cal_probs = policy.predict_proba(cal_windows)[:, 1]
    cal_net   = y_net_train[LOOK_BACK:]
    threshold = calibrate_threshold(cal_probs, cal_net, min_trades=MIN_TRADES)
    logger.info("Calibrated threshold: %.3f", threshold)

    # Evaluate on test (SVB) — cross-mechanism
    logger.info("\nEvaluating cross-mechanism on SVB test split...")
    test_result = evaluate_policy(
        policy, X_test, y_net_test, y_label_test,
        look_back=LOOK_BACK, threshold=threshold, split_name="test_SVB",
    )

    # Also evaluate on training split for comparison
    train_result = evaluate_policy(
        policy, X_train, y_net_train, y_label_train,
        look_back=LOOK_BACK, threshold=threshold, split_name="train_TerraLUNA",
    )

    # Write results
    out_path = Path(args.output_dir) / "rl_agent_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "n_trades", "net_bps_captured", "auroc", "hit_rate_above_cost",
              "oracle_capture_pct", "threshold"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in [train_result, test_result]:
            if row:
                w.writerow({k: row.get(k, "") for k in fields})
    logger.info("Results written to %s", out_path)

    # Summary
    print("\n" + "=" * 60)
    print("PPO-GRU CROSS-MECHANISM RESULTS")
    print("=" * 60)
    print(f"  Oracle ceiling:            +{ORACLE_NET_BPS:.1f} bps (315 trades)")
    print(f"  Meta-labeling (reference): +82.5 bps (397 trades, 50.8% oracle)")
    if test_result:
        nbps = test_result.get("net_bps_captured", float("nan"))
        ntr  = test_result.get("n_trades", 0)
        aur  = test_result.get("auroc", float("nan"))
        ocap = test_result.get("oracle_capture_pct", float("nan"))
        print(f"  PPO-GRU (cross-mech.):     {nbps:+.1f} bps ({ntr} trades, "
              f"{ocap:.1f}% oracle capture, AUROC {aur:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
