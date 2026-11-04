# Every agent must be evaluated on the same test orders, so we split the orders into TRAIN and TEST sets here
import numpy as np


def split_orders(orders, test_frac=0.2, seed=42):
    # 80/20 splits 
    test = orders.sample(frac=test_frac, random_state=seed)
    train = orders.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_policy(policy_fn, env, episodes=500, name="agent"):
    """Run a policy on seeds 0..episodes-1 (same orders for every agent) and return the metrics.
    policy_fn(env) must return an action (0=up, 1=down, 2=left, 3=right)."""
    wins, rewards, steps, minutes = 0, [], [], []
    for ep in range(episodes):
        env.reset(seed=ep)
        done, ep_r = False, 0.0
        while not done:
            _, rew, term, trunc, info = env.step(policy_fn(env))
            ep_r += rew
            done = term or trunc
        wins += info["delivered"]
        rewards.append(ep_r)
        steps.append(info["steps"])
        if info["delivered"]:  # delivery time only makes sense for completed deliveries
            minutes.append(info["elapsed_min"])
    return {
        "agent": name,
        "success_rate": wins / episodes,
        "avg_reward": float(np.mean(rewards)),
        "avg_steps": float(np.mean(steps)),
        "avg_delivery_time_min": float(np.mean(minutes)) if minutes else float("nan"),
    }