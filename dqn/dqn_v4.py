"""Standard DQN for the isolated dynamic delivery task (no Double/Dueling DQN).

Run experiments with: python -B -m dqn.run_experiments --help
"""
from dataclasses import asdict, dataclass
import random

import numpy as np
import torch
from torch import nn

from dqn.dynamic_env import FEATURE_DIMS, features


@dataclass(frozen=True)
class Config:
    total_steps: int = 60_000
    hidden: int = 128
    lr: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup: int = 1000
    target_update: int = 500
    train_every: int = 4
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    reward_scale: float = 10.0

    def __post_init__(self):
        if min(self.total_steps, self.hidden, self.batch_size, self.buffer_size,
               self.target_update, self.train_every) < 1:
            raise ValueError("Step counts and sizes must be positive")
        if self.buffer_size < self.batch_size or self.warmup < 0:
            raise ValueError("Replay capacity must cover a batch; warmup must be nonnegative")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= 1:
            raise ValueError("gamma and epsilon must be in [0, 1]")
        if self.lr <= 0 or self.reward_scale <= 0 or not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("Invalid learning rate, reward scale or epsilon decay")


class QNetwork(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, state):
        return self.net(state)


def bellman_targets(rewards, terminal, next_q, gamma):
    return rewards + gamma * next_q.max(dim=1).values * (1.0 - terminal)


def train(env, cfg=Config(), *, seed=42, feature_mode="deadline"):
    """Equal environment-step budgets; isolated exploration/replay RNG per run.

    The environment's finite-budget failure is terminal. Exhausting the TRAINING
    budget mid-episode is not: its final transition retains bootstrapping.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    online = QNetwork(FEATURE_DIMS[feature_mode], cfg.hidden)
    target = QNetwork(FEATURE_DIMS[feature_mode], cfg.hidden)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimiser = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    replay, replay_index = [], 0
    episode, episode_return, episode_common, losses = 0, 0.0, 0.0, []
    env.reset(seed=seed * 1_000_000)
    state = features(env, feature_mode)
    rows = []
    updates = 0
    for step in range(1, cfg.total_steps + 1):
        fraction = min(1.0, (step - 1) / max(1, cfg.total_steps * cfg.epsilon_decay_fraction))
        epsilon = 1.0 + fraction * (cfg.epsilon_end - 1.0)
        if rng.random() < epsilon:
            action = rng.randrange(4)
        else:
            with torch.no_grad():
                action = int(online(torch.from_numpy(state)).argmax().item())
        _, reward, terminated, truncated, info = env.step(action)
        next_state = features(env, feature_mode)
        transition = (state, action, reward / cfg.reward_scale, next_state, float(terminated))
        if len(replay) < cfg.buffer_size:
            replay.append(transition)
        else:
            replay[replay_index] = transition
        replay_index = (replay_index + 1) % cfg.buffer_size
        state = next_state
        episode_return += reward
        episode_common += info["common_reward"]
        if len(replay) >= max(cfg.warmup, cfg.batch_size) and step % cfg.train_every == 0:
            batch = rng.sample(replay, cfg.batch_size)
            s, a, r, s2, terminal = zip(*batch)
            s = torch.from_numpy(np.stack(s))
            s2 = torch.from_numpy(np.stack(s2))
            q = online(s).gather(1, torch.tensor(a).unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                y = bellman_targets(torch.tensor(r, dtype=torch.float32),
                                    torch.tensor(terminal), target(s2), cfg.gamma)
            loss = nn.functional.smooth_l1_loss(q, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DQN loss")
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimiser.step()
            losses.append(float(loss.item()))
            updates += 1
        if step % cfg.target_update == 0:
            target.load_state_dict(online.state_dict())
        if terminated or truncated:
            rows.append(dict(episode=episode, total_steps=step, reward=episode_return,
                             common_return=episode_common, delivered=info["delivered"],
                             elapsed_min=info["elapsed_min"], epsilon=epsilon,
                             mean_loss=float(np.mean(losses)) if losses else None))
            episode += 1
            env.reset(seed=seed * 1_000_000 + episode)
            state = features(env, feature_mode)
            episode_return, episode_common, losses = 0.0, 0.0, []
    online.eval()
    return online, rows, {"updates": updates, "steps": cfg.total_steps,
                          "completed_episodes": episode}


def policy(net, feature_mode="deadline"):
    net.eval()

    def act(env):
        with torch.no_grad():
            return int(net(torch.from_numpy(features(env, feature_mode))).argmax().item())
    return act


def save_checkpoint(path, net, cfg, *, seed, feature_mode, metadata):
    torch.save(dict(format_version=1, algorithm="standard_dqn", state_dict=net.state_dict(),
                    config=asdict(cfg), seed=seed, feature_mode=feature_mode,
                    obs_dim=FEATURE_DIMS[feature_mode], metadata=metadata), path)


def load_checkpoint(path):
    item = torch.load(path, map_location="cpu", weights_only=True)
    if item.get("format_version") != 1 or item.get("algorithm") != "standard_dqn":
        raise ValueError("Expected a v4 standard-DQN checkpoint")
    cfg = Config(**item["config"])
    if item["obs_dim"] != FEATURE_DIMS[item["feature_mode"]]:
        raise ValueError("Checkpoint feature dimensions do not match")
    net = QNetwork(item["obs_dim"], cfg.hidden)
    net.load_state_dict(item["state_dict"])
    net.eval()
    return net, item
