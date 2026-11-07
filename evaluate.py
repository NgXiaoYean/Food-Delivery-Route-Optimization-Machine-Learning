"""
Automated Master Evaluation & Cross-Algorithm Ensemble Pipeline
===============================================================
Evaluates all Baselines and all 3 Reinforcement Learning Algorithms on the
500 unseen Zomato delivery test episodes (seeds 0..499):
  - Non-Learning Baselines: Random, Greedy (Straight-Line), Planner (Dijkstra Oracle)
  - Algorithm 1 (Tabular RL):     Tabular Q-Learning (Raw v1 & Tuned Refined v2)
  - Algorithm 2 (Value Deep RL):  Deep Q-Network (DQN v1, v2, v3)
  - Algorithm 3 (Dual-Stream RL): Dueling Double DQN (D3QN)
  - Part B Combined System:       Cross-Algorithm Ensemble (Q-Learning + DQN + D3QN)

Outputs:
  - dqn/result/results.csv
  - result/master_all_models_results.csv
  - result/master_comparison_chart.png
  - dqn/result/route_comparison_map.png

Run:  python evaluate.py
"""
import heapq
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch

from env import DeliveryEnv, load_orders
from shared import evaluate_policy, split_orders
from dqn.dqn_v1 import QNetwork as QNetV1, dqn_policy as dqn_policy_v1
from dqn.dqn_v2 import QNetwork as QNetV2, dqn_policy as dqn_policy_v2, features as features_v2
from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3, features_v3

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
DQN_RESULT_DIR = BASE_DIR / "dqn" / "result"
QL_RESULT_DIR = BASE_DIR / "q_learning" / "result"
DDQN_RESULT_DIR = BASE_DIR / "ddqn" / "result"
MASTER_RESULT_DIR = BASE_DIR / "result"

DQN_RESULT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_RESULT_DIR.mkdir(parents=True, exist_ok=True)

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ==============================================================================
# BASELINE POLICIES
# ==============================================================================

def random_policy(env) -> int:
    return int(env.rng.integers(4))


def greedy_policy(env) -> int:
    """Walks straight toward destination (row first, then col), ignoring hotspots."""
    (r, c), (dr, dc) = env.pos, env.dest
    if r != dr:
        return 1 if dr > r else 0
    return 3 if dc > c else 2


def planner_policy(env) -> int:
    """Dijkstra shortest-path oracle with full map knowledge of hotspots and step costs."""
    base = 1.0 + 0.3 * env.traffic + 0.2 * env.weather
    dist = {env.pos: 0.0}
    first = {env.pos: None}
    heap = [(0.0, env.pos)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if (r, c) == env.dest:
            return 0 if first[(r, c)] is None else first[(r, c)]
        if d > dist[(r, c)]:
            continue
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.n and 0 <= nc < env.n):
                continue
            cost = base + (1.0 + env.traffic if env.hotspots[nr, nc] else 0.0)
            if d + cost < dist.get((nr, nc), 1e18):
                dist[(nr, nc)] = d + cost
                first[(nr, nc)] = a if first[(r, c)] is None else first[(r, c)]
                heapq.heappush(heap, (d + cost, (nr, nc)))
    return 0


# ==============================================================================
# PART B.2: CROSS-ALGORITHM ENSEMBLE (TABULAR Q-LEARNING + DQN + DUELING DDQN)
# ==============================================================================

def make_cross_algorithm_ensemble(ql_agent, dqn_v2_net, dqn_v3_net, ddqn_net):
    """
    Combines all 3 distinct RL algorithms via standardized Q-value fusion:
      - Tabular Q-Learning (fast discrete lookup preference)
      - DQN v2 & v3 (continuous MLP value estimation)
      - Dueling Double DQN (debiased Value + Advantage hotspot-aware stream)
    """
    tracker = {"prev_pos": None}

    def norm(q: np.ndarray) -> np.ndarray:
        std = float(np.std(q))
        return (q - float(np.mean(q))) / (std + 1e-6) if std > 1e-6 else np.zeros_like(q)

    def act(env) -> int:
        if env.steps == 0:
            tracker["prev_pos"] = None

        obs = env._obs()
        blended = np.zeros(4, dtype=np.float32)

        # 1. Tabular Q-Learning contribution
        if ql_agent is not None:
            q_tab = ql_agent.q_table[ql_agent.extract_state(env)]
            blended += 0.10 * norm(q_tab)

        # 2. DQN (v2 + v3) contribution
        with torch.no_grad():
            if dqn_v2_net is not None:
                q_v2 = dqn_v2_net(torch.tensor(features_v2(obs)).unsqueeze(0)).squeeze(0).numpy()
                blended += 0.15 * norm(q_v2)
            if dqn_v3_net is not None:
                q_v3 = dqn_v3_net(torch.tensor(features_v3(obs, env)).unsqueeze(0)).squeeze(0).numpy()
                blended += 0.05 * norm(q_v3)
            # 3. Dueling Double DQN contribution
            if ddqn_net is not None:
                q_ddqn = ddqn_net(torch.tensor(features_v3(obs, env)).unsqueeze(0)).squeeze(0).numpy()
                blended += 0.70 * norm(q_ddqn)

        # Boundary safety & short-term oscillation filter
        r, c = env.pos
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.n and 0 <= nc < env.n):
                blended[a] -= 10.0
            elif (nr, nc) == tracker["prev_pos"]:
                blended[a] -= 1.5

        best_a = int(np.argmax(blended))
        tracker["prev_pos"] = env.pos
        return best_a

    return act


# ==============================================================================
# VISUALIZATION GENERATORS
# ==============================================================================

def save_master_comparison_chart(df: pd.DataFrame, out_path: Path):
    """Plot 4 key metrics across Baselines, Q-Learning, DQN, Dueling DDQN, and Ensemble."""
    core_names = [
        "Greedy",
        "Q-Learning (Tuned)",
        "DQN v2",
        "DQN v3",
        "Dueling DDQN",
        "3-Model Ensemble",
        "Planner",
    ]
    sub = df[df["agent"].isin(core_names)].copy()
    if sub.empty:
        sub = df.copy()

    palette = ["#7f7f7f", "#ff7f0e", "#1f77b4", "#aec7e8", "#2ca02c", "#d62728", "#9467bd"][:len(sub)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1. Average Cumulative Reward
    b1 = axes[0, 0].bar(sub["agent"], sub["avg_reward"], color=palette)
    axes[0, 0].set_title("1. Mean Cumulative Reward (Higher = Better)", fontweight="bold")
    axes[0, 0].set_ylabel("Avg Reward")
    axes[0, 0].tick_params(axis="x", rotation=22)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(b1, sub["avg_reward"]):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.2f}", ha="center", fontsize=8.5, fontweight="bold")

    # 2. SLA On-Time Delivery Rate (%)
    on_time_pct = sub["on_time_rate"] * 100.0
    b2 = axes[0, 1].bar(sub["agent"], on_time_pct, color=palette)
    axes[0, 1].set_title("2. Zomato SLA On-Time Delivery Rate (%)", fontweight="bold")
    axes[0, 1].set_ylabel("On-Time Rate (%)")
    axes[0, 1].tick_params(axis="x", rotation=22)
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(b2, on_time_pct):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # 3. Average Delivery Time (min)
    b3 = axes[1, 0].bar(sub["agent"], sub["avg_delivery_time_min"], color=palette)
    axes[1, 0].set_title("3. Mean Delivery Time in Minutes (Lower = Faster)", fontweight="bold")
    axes[1, 0].set_ylabel("Minutes")
    axes[1, 0].tick_params(axis="x", rotation=22)
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(b3, sub["avg_delivery_time_min"]):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}m", ha="center", fontsize=8.5, fontweight="bold")

    # 4. Average Traffic Hotspot Steps (Lower = Better Congestion Avoidance)
    b4 = axes[1, 1].bar(sub["agent"], sub["avg_hotspot_steps"], color=palette)
    axes[1, 1].set_title("4. Mean Traffic Hotspots Entered (Lower = Better)", fontweight="bold")
    axes[1, 1].set_ylabel("Hotspot Steps / Trip")
    axes[1, 1].tick_params(axis="x", rotation=22)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(b4, sub["avg_hotspot_steps"]):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Food Delivery Route Optimization — Multi-Algorithm Benchmark (500 Unseen Test Orders)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_route_comparison_map(env, map_agents: dict, out_path: Path):
    """Save 10x10 city grid route comparison across the 3 RL families + Greedy."""
    sample_seed = 7
    fig, axes = plt.subplots(1, len(map_agents), figsize=(4.5 * len(map_agents), 4.6))
    if len(map_agents) == 1:
        axes = [axes]

    for ax, (title, pol) in zip(axes, map_agents.items()):
        env.reset(seed=sample_seed)
        start, dest = env.pos, env.dest
        path = [env.pos]
        rew, hot, done = 0.0, 0, False
        while not done:
            _, r, t, tr, info = env.step(pol(env))
            rew += r
            path.append(env.pos)
            hot += int(env.hotspots[env.pos])
            done = t or tr

        for r_i in range(env.n):
            for c_i in range(env.n):
                is_h = env.hotspots[r_i, c_i]
                ax.add_patch(patches.Rectangle(
                    (c_i - 0.5, r_i - 0.5), 1, 1,
                    facecolor="#ffd6d6" if is_h else "#f8f9fa",
                    edgecolor="#dee2e6"
                ))
                if is_h:
                    ax.text(c_i, r_i, "x", ha="center", va="center", color="#cc0000", fontsize=10, fontweight="bold")

        rows_p = [p[0] for p in path]
        cols_p = [p[1] for p in path]
        ax.plot(cols_p, rows_p, color="#1f77b4", lw=2.5, marker="o", markersize=4)
        ax.scatter([start[1]], [start[0]], c="#2ca02c", s=160, marker="s", zorder=5)
        ax.scatter([dest[1]], [dest[0]], c="#ff7f0e", s=200, marker="*", zorder=6)
        el_min = info["elapsed_min"]
        ax.set_title(f"{title}\nReward: {rew:.1f} | Time: {el_min:.1f}m | Hotspots: {hot}", fontsize=9.5, fontweight="bold")
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(9.5, -0.5)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

    plt.suptitle(f"10x10 City Grid Route Comparison Across Models (Test Order #{sample_seed})", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}. Run `python clean_data.py` first.")
        orders = load_orders(DATA_PATH)
        _, test_orders = split_orders(orders)
        env = DeliveryEnv(test_orders)
    except Exception as exc:
        print(f"[ERROR] Initialization failed: {exc}")
        sys.exit(1)

    results = [
        evaluate_policy(random_policy, env, name="Random"),
        evaluate_policy(greedy_policy, env, name="Greedy"),
        evaluate_policy(planner_policy, env, name="Planner"),
    ]

    # 1. Load Algorithm 1: Tabular Q-Learning
    ql_best_agent = None
    ql_path = QL_RESULT_DIR / "ql_model_best.pkl"
    if ql_path.exists():
        try:
            from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy
            ql_best_agent = TabularQLearningAgent.load(ql_path)
            results.append(evaluate_policy(make_ql_policy(ql_best_agent), env, name="Q-Learning (Tuned)"))
        except Exception as exc:
            print(f"[WARN] Could not load Q-Learning checkpoint: {exc}")

    # 2. Load Algorithm 2: DQN (v1, v2, v3)
    net_v1, net_v2, net_v3 = None, None, None
    v1_path = DQN_RESULT_DIR / "dqn_model_v1.pt"
    if v1_path.exists():
        net_v1 = QNetV1()
        net_v1.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True))
        net_v1.eval()
        results.append(evaluate_policy(dqn_policy_v1(net_v1), env, name="DQN v1"))

    v2_path = DQN_RESULT_DIR / "dqn_model_v2.pt"
    if v2_path.exists():
        net_v2 = QNetV2()
        net_v2.load_state_dict(torch.load(v2_path, map_location="cpu", weights_only=True))
        net_v2.eval()
        results.append(evaluate_policy(dqn_policy_v2(net_v2), env, name="DQN v2"))

    v3_path = DQN_RESULT_DIR / "dqn_model_v3.pt"
    if v3_path.exists():
        net_v3 = QNetV2(obs_dim=OBS_DIM_V3)
        net_v3.load_state_dict(torch.load(v3_path, map_location="cpu", weights_only=True))
        net_v3.eval()
        results.append(evaluate_policy(dqn_policy_v3(net_v3), env, name="DQN v3"))

    # 3. Load Algorithm 3: Dueling Double DQN (or fallback to tuned DDQN)
    ddqn_net = None
    ddqn_pol = None
    ddqn_path = DDQN_RESULT_DIR / "ddqn_model_best.pt"
    tuned_fallback = DQN_RESULT_DIR / "dqn_model_tuned.pt"
    if ddqn_path.exists():
        try:
            from ddqn.ddqn_pipeline import load_best_ddqn, make_ddqn_policy
            ddqn_net = load_best_ddqn(ddqn_path)
            ddqn_pol = make_ddqn_policy(ddqn_net)
            results.append(evaluate_policy(ddqn_pol, env, name="Dueling DDQN"))
        except Exception as exc:
            print(f"[WARN] Could not load Dueling DDQN checkpoint: {exc}")
    elif tuned_fallback.exists():
        ckpt = torch.load(tuned_fallback, map_location="cpu", weights_only=False)
        ddqn_net = QNetV2(obs_dim=OBS_DIM_V3, hidden=ckpt.get("hidden", 128))
        ddqn_net.load_state_dict(ckpt["state_dict"])
        ddqn_net.eval()
        ddqn_pol = dqn_policy_v3(ddqn_net)
        results.append(evaluate_policy(ddqn_pol, env, name="Dueling DDQN"))

    # 4. Evaluate Part B Cross-Algorithm Ensemble (Q-Learning + DQN + Dueling DDQN)
    ens_pol = None
    if net_v2 is not None and ddqn_net is not None:
        ens_pol = make_cross_algorithm_ensemble(ql_best_agent, net_v2, net_v3, ddqn_net)
        results.append(evaluate_policy(ens_pol, env, name="3-Model Ensemble"))

    table = pd.DataFrame(results)
    print("\n" + "=" * 95)
    print("MASTER MULTI-ALGORITHM BENCHMARK (500 UNSEEN ZOMATO TEST ORDERS)")
    print("=" * 95)
    print(table.round(3).to_string(index=False))

    table.to_csv(DQN_RESULT_DIR / "results.csv", index=False)
    table.to_csv(MASTER_RESULT_DIR / "master_all_models_results.csv", index=False)

    save_master_comparison_chart(table, MASTER_RESULT_DIR / "master_comparison_chart.png")

    map_dict = {"Greedy (Baseline)": greedy_policy}
    if ql_best_agent is not None:
        from q_learning.q_learning_pipeline import make_ql_policy
        map_dict["1. Q-Learning (Tuned)"] = make_ql_policy(ql_best_agent)
    if net_v2 is not None:
        map_dict["2. DQN v2"] = dqn_policy_v2(net_v2)
    if ddqn_pol is not None:
        map_dict["3. Dueling DDQN"] = ddqn_pol
    if ens_pol is not None:
        map_dict["Ensemble (QL+DQN+D3QN)"] = ens_pol
    save_route_comparison_map(env, map_dict, DQN_RESULT_DIR / "route_comparison_map.png")
    print(f"\nSaved master charts & CSVs to {MASTER_RESULT_DIR} and {DQN_RESULT_DIR}")
