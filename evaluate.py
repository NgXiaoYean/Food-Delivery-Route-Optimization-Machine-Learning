"""
DQN v3 = DQN v2 + local hotspot awareness.
Run from the PROJECT ROOT (the folder containing env.py):  python -m dqn.dqn_v3

Why? The error analysis showed DQN v2 behaves like the greedy agent: it spends about 1-1.5 steps
per delivery inside congested cells, the planner only about 0.3. The v2 state never tells the
network where the hotspots are. v3 adds 4 flags: "is the cell above / below / left / right a hotspot?".
Traffic level is already in the state, so the network can learn that a hotspot costs more under Jam.

This needs NO change to env.py (the flags are read from env.hotspots), so teammates are unaffected.
"""
import sys
import random
from pathlib import Path

# Add project root to Python's module search path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dqn.dqn_v2 import (
    CONFIG,
    QNetwork,
    ReplayBuffer,
    SEED,
    features as base_features
)

from env import DeliveryEnv, load_orders
from shared import evaluate_policy, split_orders

OBS_DIM_V3 = 8 + 3 + 4  # env state + v2 relative-direction features + 4 hotspot flags
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # same order as the actions: up, down, left, right


def features_v3(obs, env):
    """v2 features plus: is the neighbouring cell in each direction a congested hotspot?"""
    r, c = env.pos
    flags = []
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        inside = 0 <= nr < env.n and 0 <= nc < env.n
        flags.append(float(inside and env.hotspots[nr, nc]))
    return np.concatenate([base_features(obs), flags]).astype(np.float32)


def train_dqn_v3(env, cfg=CONFIG, verbose=True):
    """Same DQN training loop as v2 (replay, target network, epsilon-greedy); only the state differs."""
    online = QNetwork(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"])
    target = QNetwork(obs_dim=OBS_DIM_V3, hidden=cfg["hidden"])
    target.load_state_dict(online.state_dict())
    optimiser = optim.Adam(online.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"])
    total_steps, episode_rewards, episode_success = 0, [], []

    for ep in range(cfg["episodes"]):
        frac = min(1.0, ep / cfg["eps_decay_episodes"])
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])

        obs, _ = env.reset()
        state = features_v3(obs, env)
        done, ep_reward, delivered = False, 0.0, False
        while not done:
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = int(online(torch.tensor(state).unsqueeze(0)).argmax(1))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = features_v3(next_obs, env)  # env has moved, so flags describe the NEW position
            done = terminated or truncated
            delivered = info["delivered"]
            buffer.add(state, action, reward / cfg["reward_scale"], next_state, float(terminated))
            state, ep_reward = next_state, ep_reward + reward
            total_steps += 1

            if len(buffer) >= max(cfg["warmup"], cfg["batch_size"]):
                s, a, r, s2, d = buffer.sample(cfg["batch_size"])
                q = online(s).gather(1, a)
                with torch.no_grad():
                    y = r + cfg["gamma"] * target(s2).max(1, keepdim=True)[0] * (1 - d)
                loss = nn.functional.smooth_l1_loss(q, y)
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimiser.step()

            if total_steps % cfg["target_update_steps"] == 0:
                target.load_state_dict(online.state_dict())

        episode_rewards.append(ep_reward)
        episode_success.append(float(delivered))
        if verbose and (ep + 1) % 250 == 0:
            print(f"episode {ep+1:5d}  eps={eps:.2f}  avg reward={np.mean(episode_rewards[-250:]):7.1f}  "
                  f"success={np.mean(episode_success[-250:]):.0%}  (last 250 episodes)")
    return online, episode_rewards


def dqn_policy_v3(net):
    """Greedy policy (no exploration) used for evaluation."""
    def act(env):
        with torch.no_grad():
            state = features_v3(env._obs(), env)
            return int(net(torch.tensor(state).unsqueeze(0)).argmax(1))
    return act


if __name__ == "__main__":
    RESULT_DIR = BASE_DIR / "dqn" / "result"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    orders = load_orders(BASE_DIR / "data" / "dataset_clean.csv")
    train_orders, test_orders = split_orders(orders)
    train_env = DeliveryEnv(train_orders)
    test_env = DeliveryEnv(test_orders)
    train_env.rng = np.random.default_rng(SEED)

    net, rewards = train_dqn_v3(train_env)

    pd.Series(rewards).rolling(100).mean().plot()
    plt.xlabel("Training episode"); plt.ylabel("Reward (100-episode average)")
    plt.title("DQN v3 learning curve")
    plt.savefig(RESULT_DIR / "dqn_learning_curve_v3.png", dpi=150, bbox_inches="tight")

    result = evaluate_policy(dqn_policy_v3(net), test_env, episodes=500, name="DQN v3")
    print(result)
    pd.DataFrame([result]).to_csv(RESULT_DIR / "dqn_results_v3.csv", index=False)
    torch.save(net.state_dict(), RESULT_DIR / "dqn_model_v3.pt")