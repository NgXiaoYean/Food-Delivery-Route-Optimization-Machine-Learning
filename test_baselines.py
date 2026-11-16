"""
Sanity check for the shared environment: compare a RANDOM agent with a simple
GREEDY agent (walks straight toward the destination, ignores traffic hotspots).

Run:  py test_baselines.py

What to look for:
  - Greedy should reach ~100% success, much better than random. If not, the
    max_steps or the reward values need changing BEFORE anyone trains.
  - Greedy's reward should be clearly higher than random's.
  - These two rows are also your baselines for the final comparison in the report.
"""
import numpy as np
from env import DeliveryEnv, load_orders

orders = load_orders("./data/dataset_clean.csv")
env = DeliveryEnv(orders)


def greedy_action(env):
    """0=up, 1=down, 2=left, 3=right: fix the row difference first, then the column."""
    (r, c), (dr, dc) = env.pos, env.dest
    if r != dr:
        return 1 if dr > r else 0
    return 3 if dc > c else 2


def run(policy, name, episodes=500):
    wins, rewards, steps, minutes = 0, [], [], []
    for ep in range(episodes):
        env.reset(seed=ep)  # same seeds for every agent = fair comparison
        done, ep_r = False, 0.0
        while not done:
            _, rew, term, trunc, info = env.step(policy(env))
            ep_r += rew
            done = term or trunc
        wins += info["delivered"]
        rewards.append(ep_r)
        steps.append(info["steps"])
        minutes.append(info["elapsed_min"])
    print(f"{name:<8} success={wins/episodes:.0%}  avg reward={np.mean(rewards):7.1f}  "
          f"avg steps={np.mean(steps):5.1f}  avg delivery time={np.mean(minutes):5.1f} min")


run(lambda e: int(e.rng.integers(4)), "Random")
run(greedy_action, "Greedy")