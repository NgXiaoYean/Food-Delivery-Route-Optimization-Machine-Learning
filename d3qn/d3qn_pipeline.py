"""
Algorithm 3: Double DQN & Dueling Double DQN (D3QN) Pipeline (Part A + Part B)
==============================================================================
Why Dueling Double DQN for Food Delivery Route Optimization?
  1. Solves DQN's Maximization Bias (Hasselt et al., 2016):
     Standard DQN uses max_a' Q_target(s', a'), overestimating noisy traffic/hotspot rewards.
     Double DQN decouples selection and evaluation:
         y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
  2. Dueling Network Architecture (Wang et al., 2016):
     Decomposes Q(s, a) into two separate neural streams:
       - State-Value V(s): estimates overall order difficulty (distance, traffic jam, storm)
       - Advantage A(s, a): isolates which turn (Up/Down/Left/Right) avoids adjacent hotspots!
       Combined via: Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
  3. Explicit Regularization & Optimization (Part B):
     Evaluates L2 weight decay (AdamW), Dropout regularization, and Polyak soft target updates.

Run from project root:
    python d3qn/d3qn_pipeline.py
"""
from concurrent.futures import ProcessPoolExecutor
import random
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.dqn_v2 import QNetwork as StandardMLP, ReplayBuffer
from dqn.dqn_v3 import OBS_DIM_V3, features_v3
from env import DeliveryEnv, load_orders
from shared import VAL_SEED_OFFSET, evaluate_policy, safe_argmax, split_orders_single_3way

RESULT_DIR = BASE_DIR / "d3qn" / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
torch.set_num_threads(1)


# ==============================================================================
# DUELING DOUBLE DQN ARCHITECTURE WITH LAYER NORMALIZATION & DROPOUT REGULARIZATION
# ==============================================================================

class DuelingQNetwork(nn.Module):
    """
    Dueling Deep Q-Network separating State-Value V(s) and Action-Advantage A(s, a).
    Includes LayerNorm and configurable Dropout for Part B regularization experiments.
    """

    def __init__(self, obs_dim: int = OBS_DIM_V3, n_actions: int = 4, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        if obs_dim <= 0 or hidden <= 0:
            raise ValueError("`obs_dim` and `hidden` must be positive integers.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("`dropout` probability must be in [0.0, 1.0).")

        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

        # Stream 1: Scalar State-Value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Stream 2: Per-Action Advantage A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_layer(x)
        val = self.value_stream(feat)               # shape: (batch, 1)
        adv = self.advantage_stream(feat)           # shape: (batch, n_actions)
        # Identifiable mean-subtracted aggregation
        return val + (adv - adv.mean(dim=1, keepdim=True))


def make_d3qn_policy(net: nn.Module):
    """Deterministic greedy evaluation policy (disables Dropout via net.eval()) with shared safe_argmax."""
    tracker = {"prev_pos": None}

    def act(env):
        if env.steps == 0:
            tracker["prev_pos"] = None
        net.eval()
        with torch.no_grad():
            state = features_v3(env._obs(), env)
            q_vals = net(torch.tensor(state).unsqueeze(0)).squeeze(0).numpy().copy()

        best_action = safe_argmax(q_vals, env, tracker["prev_pos"])
        tracker["prev_pos"] = env.pos
        return best_action

    return act


def train_d3qn(env, cfg: dict, verbose: bool = False):
    """
    Train either Vanilla Double DQN (`dueling=False`) or Dueling Double DQN (`dueling=True`)
    with L2 regularization (`weight_decay`), Dropout, and Polyak / periodic target updates.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env.rng = np.random.default_rng(SEED)

    if cfg.get("dueling", True):
        online = DuelingQNetwork(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"], dropout=cfg.get("dropout", 0.0))
        target = DuelingQNetwork(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"], dropout=0.0)
    else:
        online = StandardMLP(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"])
        target = StandardMLP(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"])

    target.load_state_dict(online.state_dict())
    target.eval()

    # AdamW applies decoupled L2 weight decay regularization
    optimiser = optim.AdamW(
        online.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    buffer = ReplayBuffer(cfg["buffer_size"])
    total_steps = 0
    episode_rewards = []
    has_dropout = cfg.get("dropout", 0.0) > 0.0
    online_params = list(online.parameters())
    target_params = list(target.parameters())
    tau = cfg.get("tau", None)

    for ep in range(cfg["episodes"]):
        frac = min(1.0, ep / max(1, cfg["eps_decay_episodes"]))
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])

        obs, _ = env.reset()
        state = features_v3(obs, env)
        done, ep_reward = False, 0.0

        while not done:
            if random.random() < eps:
                action = random.randrange(4)
            else:
                if has_dropout:
                    online.eval()
                with torch.no_grad():
                    action = int(online(torch.tensor(state).unsqueeze(0)).argmax(1))
                if has_dropout:
                    online.train()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = features_v3(next_obs, env)
            done = terminated or truncated

            buffer.add(state, action, reward / cfg["reward_scale"], next_state, float(terminated))
            state = next_state
            ep_reward += reward
            total_steps += 1

            if len(buffer) >= max(cfg["warmup"], cfg["batch_size"]):
                s, a, r, s2, d = buffer.sample(cfg["batch_size"])
                q = online(s).gather(1, a)

                with torch.no_grad():
                    # Double Q-Learning target: online selects action, target evaluates Q-value
                    if has_dropout:
                        online.eval()
                    best_actions = online(s2).argmax(1, keepdim=True)
                    if has_dropout:
                        online.train()
                    best_next = target(s2).gather(1, best_actions)
                    y = r + cfg["gamma"] * best_next * (1.0 - d)

                loss = nn.functional.smooth_l1_loss(q, y)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_params, cfg.get("grad_clip", 5.0))
                optimiser.step()

                # Polyak soft target update or periodic hard update
                if tau is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(target_params, online_params, tau)
                elif total_steps % cfg["target_update_steps"] == 0:
                    target.load_state_dict(online.state_dict())

        episode_rewards.append(ep_reward)
        if verbose and (ep + 1) % 500 == 0:
            print(f"    Episode {ep+1:5d} | eps={eps:.2f} | Avg Reward (last 500): {np.mean(episode_rewards[-500:]):6.2f}", flush=True)

    online.eval()
    return online, episode_rewards


def _train_worker(job):
    """Deterministic single-thread worker for parallel candidate training."""
    label, train_orders, train_max_km, cfg = job
    torch.set_num_threads(1)
    t0 = time.time()
    print(f"  [Start] {label}...", flush=True)
    net, rewards = train_d3qn(DeliveryEnv(train_orders, max_km=train_max_km), cfg, verbose=True)
    elapsed = round(time.time() - t0, 1)
    print(f"  [Done]  {label} ({elapsed}s)", flush=True)
    return label, {k: v.cpu() for k, v in net.state_dict().items()}, rewards, elapsed, cfg


def load_best_d3qn(checkpoint_path: Path = RESULT_DIR / "d3qn_model_best.pt") -> nn.Module:
    """Safely load the saved Dueling Double DQN model with exception handling."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"D3QN checkpoint not found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    if cfg.get("dueling", True):
        net = DuelingQNetwork(obs_dim=OBS_DIM_V3, hidden=cfg.get("hidden", 128), dropout=0.0)
    else:
        net = StandardMLP(obs_dim=OBS_DIM_V3, hidden=cfg.get("hidden", 128))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


# ==============================================================================
# MAIN PIPELINE: PART A (DOUBLE DQN (MLP) VS D3QN) + PART B (REGULARIZATION)
# ==============================================================================

def main():
    try:
        data_path = BASE_DIR / "data" / "dataset_clean.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing cleaned dataset at {data_path}. Run `python clean_data.py` first.")
        orders = load_orders(data_path)
        train_orders, val_orders, test_orders, train_max_km = split_orders_single_3way(orders)
        val_env = DeliveryEnv(val_orders, max_km=train_max_km)
        test_env = DeliveryEnv(test_orders, max_km=train_max_km)
    except Exception as exc:
        print(f"[ERROR] Could not start D3QN pipeline: {exc}")
        sys.exit(1)

    base_cfg = dict(
        episodes=3500,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=1800,
        batch_size=64,
        buffer_size=50_000,
        warmup=1000,
        reward_scale=10.0,
        target_update_steps=300,
    )

    print("=" * 72)
    print("ALGORITHM 3: DOUBLE DQN — PART A (DOUBLE DQN (MLP) VS D3QN)")
    print("=" * 72)

    cfg_vanilla = {**base_cfg, "name": "Double DQN (Standard MLP)", "dueling": False, "lr": 1e-3, "gamma": 0.95, "hidden": 128, "weight_decay": 0.0, "dropout": 0.0, "tau": 0.005}
    cfg_dueling = {**base_cfg, "name": "Dueling Double DQN (D3QN)", "dueling": True,  "lr": 1e-3, "gamma": 0.95, "hidden": 128, "weight_decay": 0.0, "dropout": 0.0, "tau": 0.005}

    tuning_candidates = [
        dict(label="D3QN-Reg 1 (L2 wd=1e-4, Dropout=0.0)", lr=1e-3, gamma=0.95, hidden=128, weight_decay=1e-4, dropout=0.00, tau=0.005),
        dict(label="D3QN-Reg 2 (L2 wd=1e-4, Dropout=0.1)", lr=1e-3, gamma=0.95, hidden=128, weight_decay=1e-4, dropout=0.10, tau=0.005),
        dict(label="D3QN-Reg 3 (L2 wd=5e-4, h=192, g=0.94)", lr=8e-4, gamma=0.94, hidden=192, weight_decay=5e-4, dropout=0.05, tau=0.005),
    ]

    jobs = [
        ("Double DQN (MLP)", train_orders, train_max_km, cfg_vanilla),
        ("Dueling Double DQN", train_orders, train_max_km, cfg_dueling),
    ] + [
        (cand["label"], train_orders, train_max_km, {**base_cfg, "dueling": True, **cand})
        for cand in tuning_candidates
    ]

    print(f"\nTraining {len(jobs)} single-order Double-DQN / D3QN variants in parallel (1 CPU thread each)...", flush=True)
    results_by_label = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        for label, state_dict, rewards, elapsed, cfg in pool.map(_train_worker, jobs):
            if cfg.get("dueling", True):
                net = DuelingQNetwork(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"], dropout=0.0)
            else:
                net = StandardMLP(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"])
            net.load_state_dict(state_dict)
            net.eval()
            results_by_label[label] = (net, rewards, elapsed, cfg)

    net_van, rew_van, _, _ = results_by_label["Double DQN (MLP)"]
    res_van = evaluate_policy(make_d3qn_policy(net_van), test_env, episodes=500, name="Double DQN (MLP)")

    net_duel, rew_duel, time_duel, _ = results_by_label["Dueling Double DQN"]
    res_duel = evaluate_policy(make_d3qn_policy(net_duel), test_env, episodes=500, name="Dueling Double DQN")

    df_part_a = pd.DataFrame([res_van, res_duel])
    df_part_a.to_csv(RESULT_DIR / "d3qn_part_a_results.csv", index=False)
    print("\n=== Double DQN Part A Comparison ===")
    print(df_part_a.round(3).to_string(index=False))

    plt.figure(figsize=(9, 4.5))
    plt.plot(pd.Series(rew_van).rolling(100).mean(), label="Vanilla Double DQN (MLP)", color="#ff7f0e", alpha=0.85)
    plt.plot(pd.Series(rew_duel).rolling(100).mean(), label="Dueling Double DQN (Value + Advantage)", color="#2ca02c", lw=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward (100-Episode Moving Avg)")
    plt.title("Algorithm 3 Learning Curves: Double DQN (MLP) vs. D3QN")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "d3qn_learning_curves.png", dpi=160)
    plt.close()

    # --------------------------------------------------------------------------
    # PART B: REGULARIZATION (WEIGHT DECAY, DROPOUT) & HYPERPARAMETER TUNING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ALGORITHM 3: DUELING DOUBLE DQN — PART B (REGULARIZATION & TUNING)")
    print("=" * 72)

    val_duel = evaluate_policy(make_d3qn_policy(net_duel), val_env, episodes=500, name="D3QN Baseline (No Reg)", seed_offset=VAL_SEED_OFFSET)
    best_net = net_duel
    best_cfg = cfg_dueling
    best_val_reward = val_duel["avg_reward"]

    tune_rows = [{
        "config": "D3QN Baseline (No Reg)",
        "lr": cfg_dueling["lr"],
        "gamma": cfg_dueling["gamma"],
        "hidden": cfg_dueling["hidden"],
        "weight_decay_L2": 0.0,
        "dropout": 0.0,
        **{k: v for k, v in val_duel.items() if k != "agent"},
        "train_time_s": time_duel,
    }]

    for i, cand in enumerate(tuning_candidates, 1):
        net_c, _, elapsed, full_cfg = results_by_label[cand["label"]]
        # Select strictly on the 16% VALIDATION split (seeds 1000..1499)
        metrics = evaluate_policy(make_d3qn_policy(net_c), val_env, episodes=500, name=cand["label"], seed_offset=VAL_SEED_OFFSET)

        tune_rows.append({
            "config": cand["label"],
            "lr": cand["lr"],
            "gamma": cand["gamma"],
            "hidden": cand["hidden"],
            "weight_decay_L2": cand["weight_decay"],
            "dropout": cand["dropout"],
            **{k: v for k, v in metrics.items() if k != "agent"},
            "train_time_s": elapsed,
        })
        print(f"  [{i}/{len(tuning_candidates)}] {cand['label']} -> Val Success: {metrics['success_rate']:.1%} | Val Avg Reward: {metrics['avg_reward']:.2f} | Hotspots: {metrics['avg_hotspot_steps']:.2f}")

        if metrics["avg_reward"] > best_val_reward:
            best_val_reward = metrics["avg_reward"]
            best_net = net_c
            best_cfg = full_cfg

    df_tune = pd.DataFrame(tune_rows)
    df_tune.to_csv(RESULT_DIR / "d3qn_part_b_tuning.csv", index=False)
    print("\n=== Dueling Double DQN Part B Validation Regularization & Tuning Summary ===")
    print(df_tune.round(3).to_string(index=False))

    torch.save({
        "state_dict": best_net.state_dict(),
        "config": best_cfg,
    }, RESULT_DIR / "d3qn_model_best.pt")
    test_best = evaluate_policy(make_d3qn_policy(best_net), test_env, episodes=500, name="D3QN")
    print(f"\nSaved best Dueling Double DQN model (Val Reward={best_val_reward:.2f} | Test Reward={test_best['avg_reward']:.2f}) -> {RESULT_DIR / 'd3qn_model_best.pt'}")


if __name__ == "__main__":
    main()
