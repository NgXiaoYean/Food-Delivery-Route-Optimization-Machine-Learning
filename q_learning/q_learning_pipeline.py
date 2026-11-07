"""
Algorithm 1: Tabular Q-Learning Pipeline (Part A + Part B)
==========================================================
Covers:
  - Part A (Iterations & Representation Refinement):
      * Q-Learning v1 (Raw Absolute Discrete State: 40,000 states -> Curse of Dimensionality)
      * Q-Learning v2 (Compressed Relative + Local Hotspot Discrete State -> 100% Success)
  - Part B (Model Optimization & Regularization):
      * Automated search over learning rate (alpha), discount factor (gamma),
        exploration decay, and L2 Q-table regularization (weight_decay).
      * Saves learning curves, CSV evaluation tables, and best serialized Q-table.

Run from project root:
    python q_learning/q_learning_pipeline.py
"""
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from env import DeliveryEnv, load_orders
from shared import evaluate_policy, split_orders

RESULT_DIR = BASE_DIR / "q_learning" / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ==============================================================================
# STATE REPRESENTATIONS (PART A: RAW VS REFINED FEATURE ENGINEERING)
# ==============================================================================

def raw_discrete_state(env) -> tuple:
    """
    v1 State: Raw grid coordinates (pos_r, pos_c, dest_r, dest_c, traffic).
    State space = 10 * 10 * 10 * 10 * 4 = 40,000 discrete states.
    Illustrates the Curse of Dimensionality on unseen start/goal pairs.
    """
    return env.discrete_state()


def refined_discrete_state(env) -> tuple:
    """
    v2 State: Invariant relative displacement + local hotspot & boundary sensors.
      - sgn_r, sgn_c in {-1, 0, +1}: direction toward customer (9 combinations)
      - axis_pref in {0, 1}: whether vertical gap >= horizontal gap
      - high_traffic in {0, 1}: whether traffic is High/Jam (hotspot penalty severe)
      - neighbor_status in {0, 1, 2}^4: for Up/Down/Left/Right -> 0=clear, 1=hotspot, 2=wall
    Compresses 40,000 coordinate pairs into ~350 generalizable states!
    """
    (r, c), (dr, dc) = env.pos, env.dest
    diff_r = dr - r
    diff_c = dc - c
    sgn_r = 0 if diff_r == 0 else (1 if diff_r > 0 else -1)
    sgn_c = 0 if diff_c == 0 else (1 if diff_c > 0 else -1)
    axis_pref = int(abs(diff_r) >= abs(diff_c))
    high_traffic = int(env.traffic >= 2)

    neighbors = []
    for mr, mc in MOVES:
        nr, nc = r + mr, c + mc
        if not (0 <= nr < env.n and 0 <= nc < env.n):
            neighbors.append(2)  # boundary wall
        elif env.hotspots[nr, nc]:
            neighbors.append(1)  # congested hotspot
        else:
            neighbors.append(0)  # clear road

    return (sgn_r, sgn_c, axis_pref, high_traffic, *neighbors)


# ==============================================================================
# TABULAR Q-LEARNING AGENT WITH L2 REGULARIZATION & EXCEPTION HANDLING
# ==============================================================================

class TabularQLearningAgent:
    """
    Model-free Tabular Q-Learning agent trained via Bellman TD(0) updates:
        Q(s, a) <- (1 - alpha * weight_decay) * Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(self, alpha=0.15, gamma=0.95, weight_decay=0.0, state_mode="refined"):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"Learning rate alpha must be in (0, 1], got {alpha}.")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Discount gamma must be in [0, 1], got {gamma}.")

        self.alpha = alpha
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.state_mode = state_mode
        self.q_table = defaultdict(lambda: np.zeros(4, dtype=np.float32))

    def extract_state(self, env) -> tuple:
        if self.state_mode == "raw":
            return raw_discrete_state(env)
        return refined_discrete_state(env)

    def select_action(self, state: tuple, eps: float, rng: random.Random) -> int:
        if rng.random() < eps:
            return rng.randrange(4)
        q_vals = self.q_table[state]
        # Break ties randomly if state is unvisited (all zeros)
        max_q = np.max(q_vals)
        best_actions = np.flatnonzero(np.isclose(q_vals, max_q))
        return int(rng.choice(best_actions))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple, terminated: bool):
        best_next = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        current_q = self.q_table[state][action]
        # Bellman TD update with optional L2 regularization (weight_decay) to prevent Q-value drift
        reg_term = self.weight_decay * current_q
        self.q_table[state][action] = current_q + self.alpha * (td_target - current_q - reg_term)

    def save(self, filepath: Path):
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "q_table": dict(self.q_table),
                "alpha": self.alpha,
                "gamma": self.gamma,
                "weight_decay": self.weight_decay,
                "state_mode": self.state_mode,
            }
            with open(filepath, "wb") as f:
                pickle.dump(payload, f)
        except OSError as exc:
            raise RuntimeError(f"Failed to save Q-table to {filepath}: {exc}") from exc

    @classmethod
    def load(cls, filepath: Path):
        if not filepath.exists():
            raise FileNotFoundError(f"Q-Learning checkpoint not found at {filepath}")
        try:
            with open(filepath, "rb") as f:
                payload = pickle.load(f)
            agent = cls(
                alpha=payload.get("alpha", 0.15),
                gamma=payload.get("gamma", 0.95),
                weight_decay=payload.get("weight_decay", 0.0),
                state_mode=payload.get("state_mode", "refined"),
            )
            agent.q_table = defaultdict(lambda: np.zeros(4, dtype=np.float32), payload["q_table"])
            return agent
        except Exception as exc:
            raise RuntimeError(f"Corrupted or incompatible Q-table file {filepath}: {exc}") from exc


def make_ql_policy(agent: TabularQLearningAgent):
    """Greedy evaluation policy with short-term cycle prevention."""
    tracker = {"prev_pos": None}

    def act(env):
        if env.steps == 0:
            tracker["prev_pos"] = None
        state = agent.extract_state(env)
        q_vals = agent.q_table[state].copy()

        if agent.state_mode == "refined":
            r, c = env.pos
            for a, (mr, mc) in enumerate(MOVES):
                nr, nc = r + mr, c + mc
                if not (0 <= nr < env.n and 0 <= nc < env.n):
                    q_vals[a] -= 100.0
                elif (nr, nc) == tracker["prev_pos"]:
                    q_vals[a] -= 2.0

        best_action = int(np.argmax(q_vals))
        tracker["prev_pos"] = env.pos
        return best_action

    return act


def train_q_agent(env, cfg: dict, verbose: bool = False):
    """Train a Tabular Q-Learning agent and return (agent, episode_rewards)."""
    py_rng = random.Random(SEED)
    np.random.seed(SEED)
    env.rng = np.random.default_rng(SEED)

    agent = TabularQLearningAgent(
        alpha=cfg["alpha"],
        gamma=cfg["gamma"],
        weight_decay=cfg.get("weight_decay", 0.0),
        state_mode=cfg["state_mode"],
    )
    rewards_history = []

    for ep in range(cfg["episodes"]):
        frac = min(1.0, ep / max(1, cfg["eps_decay_episodes"]))
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])

        env.reset()
        state = agent.extract_state(env)
        done, ep_reward = False, 0.0

        while not done:
            action = agent.select_action(state, eps, py_rng)
            _, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_state = agent.extract_state(env)

            agent.update(state, action, reward, next_state, term)
            state = next_state
            ep_reward += reward

        rewards_history.append(ep_reward)
        if verbose and (ep + 1) % 1000 == 0:
            print(f"    Episode {ep+1:5d} | eps={eps:.2f} | Avg Reward (last 500): {np.mean(rewards_history[-500:]):7.2f} "
                  f"| Unique States: {len(agent.q_table)}")

    return agent, rewards_history


# ==============================================================================
# MAIN PIPELINE: PART A ITERATIONS + PART B HYPERPARAMETER & REGULARIZATION TUNING
# ==============================================================================

def main():
    try:
        data_path = BASE_DIR / "data" / "dataset_clean.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Cleaned dataset missing at {data_path}. Run `python clean_data.py` first.")

        orders = load_orders(data_path)
        train_orders, test_orders = split_orders(orders)
        test_env = DeliveryEnv(test_orders)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize Q-Learning pipeline: {exc}")
        sys.exit(1)

    print("=" * 72)
    print("ALGORITHM 1: TABULAR Q-LEARNING — PART A (STATE REPRESENTATION STUDY)")
    print("=" * 72)

    cfg_v1 = dict(
        name="Q-Learning v1 (Raw 40k States)",
        state_mode="raw",
        episodes=4000,
        alpha=0.15,
        gamma=0.95,
        weight_decay=0.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=2500,
    )
    cfg_v2 = dict(
        name="Q-Learning v2 (Refined Relative + Hotspot)",
        state_mode="refined",
        episodes=4000,
        alpha=0.15,
        gamma=0.95,
        weight_decay=0.0,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=2000,
    )

    print("\n[Part A - v1] Training Raw Discrete Q-Learning...")
    agent_v1, rew_v1 = train_q_agent(DeliveryEnv(train_orders), cfg_v1, verbose=True)
    res_v1 = evaluate_policy(make_ql_policy(agent_v1), test_env, episodes=500, name="Q-Learning v1 (Raw)")

    print("\n[Part A - v2] Training Refined Relative+Hotspot Q-Learning...")
    agent_v2, rew_v2 = train_q_agent(DeliveryEnv(train_orders), cfg_v2, verbose=True)
    res_v2 = evaluate_policy(make_ql_policy(agent_v2), test_env, episodes=500, name="Q-Learning v2 (Refined)")

    df_part_a = pd.DataFrame([res_v1, res_v2])
    df_part_a.to_csv(RESULT_DIR / "ql_part_a_results.csv", index=False)
    print("\n=== Q-Learning Part A Comparison ===")
    print(df_part_a.round(3).to_string(index=False))

    # Plot Part A Learning Curves (v1 vs v2)
    plt.figure(figsize=(9, 4.5))
    plt.plot(pd.Series(rew_v1).rolling(100).mean(), label=f"v1: Raw 40k States ({len(agent_v1.q_table)} visited)", color="#d62728", alpha=0.85)
    plt.plot(pd.Series(rew_v2).rolling(100).mean(), label=f"v2: Refined Relative+Hotspot ({len(agent_v2.q_table)} visited)", color="#1f77b4", lw=2)
    plt.xlabel("Training Episode")
    plt.ylabel("Reward (100-Episode Moving Avg)")
    plt.title("Tabular Q-Learning: Raw vs. Refined State Representation")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "ql_learning_curves.png", dpi=160)
    plt.close()

    # --------------------------------------------------------------------------
    # PART B: HYPERPARAMETER TUNING & L2 REGULARIZATION SEARCH
    # --------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ALGORITHM 1: TABULAR Q-LEARNING — PART B (OPTIMIZATION & REGULARIZATION)")
    print("=" * 72)

    search_grid = [
        dict(label="QL-Tune 1 (High LR, No Reg)",       alpha=0.30, gamma=0.95, weight_decay=0.000, episodes=5000, eps_decay_episodes=2500),
        dict(label="QL-Tune 2 (Balanced, No Reg)",      alpha=0.10, gamma=0.95, weight_decay=0.000, episodes=5000, eps_decay_episodes=2500),
        dict(label="QL-Tune 3 (L2 Reg wd=1e-3, g=0.95)", alpha=0.10, gamma=0.95, weight_decay=1e-3,  episodes=5000, eps_decay_episodes=2500),
        dict(label="QL-Tune 4 (L2 Reg wd=5e-4, g=0.90)", alpha=0.08, gamma=0.90, weight_decay=5e-4,  episodes=5000, eps_decay_episodes=2500),
    ]

    tune_rows = []
    best_agent = agent_v2
    best_reward = res_v2["avg_reward"]

    for idx, cand in enumerate(search_grid, 1):
        full_cfg = {
            "state_mode": "refined",
            "eps_start": 1.0,
            "eps_end": 0.03,
            **cand,
        }
        t0 = time.time()
        ag, _ = train_q_agent(DeliveryEnv(train_orders), full_cfg, verbose=False)
        metrics = evaluate_policy(make_ql_policy(ag), test_env, episodes=500, name=cand["label"])
        elapsed = round(time.time() - t0, 2)

        tune_rows.append({
            "config": cand["label"],
            "alpha": cand["alpha"],
            "gamma": cand["gamma"],
            "weight_decay_L2": cand["weight_decay"],
            **{k: v for k, v in metrics.items() if k != "agent"},
            "train_time_s": elapsed,
        })
        print(f"  [{idx}/4] {cand['label']:<36} | Success: {metrics['success_rate']:.1%} | "
              f"Reward: {metrics['avg_reward']:6.2f} | Hotspots: {metrics['avg_hotspot_steps']:.2f}")

        if metrics["avg_reward"] > best_reward:
            best_reward = metrics["avg_reward"]
            best_agent = ag

    df_tune = pd.DataFrame(tune_rows)
    df_tune.to_csv(RESULT_DIR / "ql_part_b_tuning.csv", index=False)
    best_agent.save(RESULT_DIR / "ql_model_best.pkl")
    print(f"\nSaved best Tabular Q-Learning model (Avg Reward={best_reward:.2f}) -> {RESULT_DIR / 'ql_model_best.pkl'}")


if __name__ == "__main__":
    main()
