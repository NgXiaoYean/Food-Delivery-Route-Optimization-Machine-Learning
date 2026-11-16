"""
DQN agent for the food-delivery environment (v2).   
Changes from v1 (v1 reached only 7% test success, below the random agent):
  - features(): add relative-direction inputs (destination minus position).
  - reward_scale: divide rewards by 10 inside the learning update for steadier training.
  - more training episodes and a slower epsilon decay.
KEEP the v1 results (learning curve + numbers) for the before/after table in the report (Part B).
"""

import random

import sys
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared import evaluate_policy, split_orders
from env import DeliveryEnv, load_orders


RESULT_DIR = Path("./dqn/result")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# All tunable settings in one place -> Part B can search over these automatically.
CONFIG = dict(
    episodes=6000,            # number of training deliveries (v1: 3000)
    gamma=0.99,               # discount: how much future reward matters
    lr=1e-3,                  # learning rate of the optimiser
    batch_size=64,            # transitions sampled from replay per update
    buffer_size=50_000,       # replay memory capacity
    eps_start=1.0,            # start fully random
    eps_end=0.05,             # always keep 5% exploration
    eps_decay_episodes=3000,  # episodes over which epsilon falls linearly (v1: 1500)
    target_update_steps=500,  # copy online net -> target net every N steps
    hidden=128,               # neurons per hidden layer
    warmup=1000,              # collect this many transitions before learning
    reward_scale=10.0,        # learn from reward/10 so Q-values stay small and stable
)

OBS_DIM = 8 + 3  # 8 numbers from the environment + 3 engineered features


def features(obs):
    """Add relative-direction features. The network no longer has to discover that
    'destination minus position' tells it which way to move."""
    d_row = obs[2] - obs[0]   # >0: destination is further down, <0: further up
    d_col = obs[3] - obs[1]   # >0: destination is further right, <0: further left
    dist = abs(d_row) + abs(d_col)  # Manhattan distance left to travel
    return np.concatenate([obs, [d_row, d_col, dist]]).astype(np.float32)


class QNetwork(nn.Module):
    """Maps the state features to 4 Q-values (one per action)."""

    def __init__(self, obs_dim=OBS_DIM, n_actions=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Stores past transitions; sampling randomly breaks correlation between steps."""

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


def train_dqn(env, cfg=CONFIG, verbose=True):
    """Train a DQN agent on `env`. Returns (trained network, list of episode rewards)."""
    online = QNetwork(hidden=cfg["hidden"])
    target = QNetwork(hidden=cfg["hidden"])
    target.load_state_dict(online.state_dict())  # target starts as an exact copy
    optimiser = optim.Adam(online.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"])
    total_steps, episode_rewards, episode_success = 0, [], []

    for ep in range(cfg["episodes"]):
        # Linear epsilon decay: lots of exploration early, mostly exploitation later
        frac = min(1.0, ep / cfg["eps_decay_episodes"])
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])

        obs, _ = env.reset()
        state = features(obs)
        done, ep_reward, delivered = False, 0.0, False
        while not done:
            # Epsilon-greedy action selection
            if random.random() < eps:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = int(online(torch.tensor(state).unsqueeze(0)).argmax(1))

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = features(next_obs)
            done = terminated or truncated
            delivered = info["delivered"]
            # Only a true delivery stops bootstrapping. A timeout is cut-off, not a real end.
            # The network learns from the SCALED reward; ep_reward keeps the true reward for reporting.
            buffer.add(state, action, reward / cfg["reward_scale"], next_state, float(terminated))
            state, ep_reward = next_state, ep_reward + reward
            total_steps += 1

            if len(buffer) >= max(cfg["warmup"], cfg["batch_size"]):
                s, a, r, s2, d = buffer.sample(cfg["batch_size"])
                q = online(s).gather(1, a)  # Q-value of the action actually taken
                with torch.no_grad():       # target uses the frozen network
                    best_next = target(s2).max(1, keepdim=True)[0]
                    y = r + cfg["gamma"] * best_next * (1 - d)
                loss = nn.functional.smooth_l1_loss(q, y)  # Huber loss: robust to outliers
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)  # avoid exploding gradients
                optimiser.step()

            if total_steps % cfg["target_update_steps"] == 0:
                target.load_state_dict(online.state_dict())

        episode_rewards.append(ep_reward)
        episode_success.append(float(delivered))
        if verbose and (ep + 1) % 250 == 0:
            print(f"episode {ep+1:5d}  eps={eps:.2f}  "
                  f"avg reward={np.mean(episode_rewards[-250:]):7.1f}  "
                  f"success={np.mean(episode_success[-250:]):.0%}  (last 250 episodes)")
    return online, episode_rewards


def dqn_policy(net):
    """Greedy policy (no exploration) used for evaluation."""
    def act(env):
        with torch.no_grad():
            state = features(env._obs())
            return int(net(torch.tensor(state).unsqueeze(0)).argmax(1))
    return act


if __name__ == "__main__":
    orders = load_orders("./data/dataset_clean.csv")
    train_orders, test_orders = split_orders(orders)  # 80% train / 20% test
    train_env = DeliveryEnv(train_orders)
    test_env = DeliveryEnv(test_orders)
    train_env.rng = np.random.default_rng(SEED)  # reproducible training orders

    net, rewards = train_dqn(train_env)

    # Learning curve: moving average shows whether the agent is improving
    smooth = pd.Series(rewards).rolling(100).mean()
    plt.plot(smooth)
    plt.xlabel("Training episode"); plt.ylabel("Reward (100-episode average)")
    plt.title("DQN learning curve (v2)")
    plt.savefig(RESULT_DIR / "dqn_learning_curve_v2.png", dpi=150, bbox_inches="tight")

    # Final evaluation on UNSEEN test orders, same seeds as all other agents
    result = evaluate_policy(dqn_policy(net), test_env, episodes=500, name="DQN v2")
    print(result)
    pd.DataFrame([result]).to_csv(RESULT_DIR / "dqn_results_v2.csv", index=False)
    torch.save(net.state_dict(), RESULT_DIR / "dqn_model_v2.pt")