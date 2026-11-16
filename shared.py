"""
Shared evaluation protocol and train/test splitting utilities.
Ensures every team member's RL algorithm (Tabular Q-Learning, DQN, Dueling Double DQN)
is trained on the same 80% split and evaluated on the exact same 500 seeded test episodes.
"""
import numpy as np
import pandas as pd


def split_orders(orders: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """
    Deterministically split cleaned delivery orders into train (80%) and test (20%) sets.
    Includes validation and exception handling for robustness.
    """
    if orders is None or not isinstance(orders, pd.DataFrame) or orders.empty:
        raise ValueError("`orders` must be a non-empty pandas DataFrame.")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"`test_frac` must be between 0 and 1, got {test_frac}.")

    test = orders.sample(frac=test_frac, random_state=seed)
    train = orders.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_policy(policy_fn, env, episodes: int = 500, name: str = "agent") -> dict:
    """
    Run `policy_fn(env)` on seeds 0..episodes-1 (identical test scenarios for every model)
    and compute comprehensive performance and domain-specific delivery metrics:
      - success_rate:          Fraction of orders successfully delivered before timeout
      - on_time_rate:          SLA metric — fraction delivered within Zomato's real Time_taken
      - avg_reward:            Mean cumulative MDP reward across episodes
      - std_reward:            Reward standard deviation (measures strategy stability)
      - avg_steps:             Mean grid steps per episode
      - route_efficiency:      Manhattan shortest distance / actual steps (1.0 = minimal steps)
      - avg_delivery_time_min: Mean simulated travel time (minutes) for completed deliveries
      - avg_hotspot_steps:     Mean congested hotspot cells entered per delivery
    """
    if not callable(policy_fn):
        raise TypeError("`policy_fn` must be a callable function taking `env` and returning an action (0..3).")
    if episodes <= 0:
        raise ValueError(f"`episodes` must be positive, got {episodes}.")

    wins, on_time = 0, 0
    rewards, steps, minutes, efficiencies, hotspot_counts = [], [], [], [], []

    for ep in range(episodes):
        env.reset(seed=ep)
        manhattan_dist = abs(env.pos[0] - env.dest[0]) + abs(env.pos[1] - env.dest[1])
        done, ep_r, hot = False, 0.0, 0
        info = {"delivered": False, "steps": 0, "elapsed_min": 0.0}

        while not done:
            try:
                action = int(policy_fn(env))
            except Exception as exc:
                raise RuntimeError(f"Policy '{name}' failed at episode {ep}, step {env.steps}: {exc}") from exc

            if action not in (0, 1, 2, 3):
                raise ValueError(f"Policy '{name}' returned invalid action {action}; expected 0..3.")

            _, rew, term, trunc, info = env.step(action)
            ep_r += rew
            if hasattr(env, "hotspots"):
                hot += int(env.hotspots[env.pos])
            done = term or trunc

        delivered = bool(info["delivered"])
        wins += int(delivered)
        rewards.append(ep_r)
        steps.append(info["steps"])
        hotspot_counts.append(hot)

        if delivered:
            minutes.append(info["elapsed_min"])
            if info["elapsed_min"] <= getattr(env, "benchmark_min", float("inf")):
                on_time += 1
            eff = min(1.0, manhattan_dist / max(1, info["steps"]))
            efficiencies.append(eff)
        else:
            efficiencies.append(0.0)

    return {
        "agent": name,
        "success_rate": wins / episodes,
        "on_time_rate": on_time / episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "route_efficiency": float(np.mean(efficiencies)),
        "avg_delivery_time_min": float(np.mean(minutes)) if minutes else float("nan"),
        "avg_hotspot_steps": float(np.mean(hotspot_counts)),
    }
