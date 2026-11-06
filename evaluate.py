"""
Evaluate baselines (Random, Greedy, Dijkstra Planner), DQN iterations (v1, v2, v3),
Tuned Double DQN, and the Part B Combined Ensemble on the 500 unseen test orders.

Run:  python evaluate.py
"""
from pathlib import Path
import heapq

import pandas as pd
import torch

from dqn.dqn_v1 import QNetwork as QNetV1, dqn_policy as dqn_policy_v1
from dqn.dqn_v2 import QNetwork as QNetV2, dqn_policy as dqn_policy_v2
from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3
from env import DeliveryEnv, load_orders
from shared import evaluate_policy, split_orders

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
RESULT_DIR = BASE_DIR / "dqn" / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = RESULT_DIR / "results.csv"

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def random_policy(env):
    return int(env.rng.integers(4))


def greedy_policy(env):
    """Walks straight toward destination (row first, then col), ignoring hotspots."""
    (r, c), (dr, dc) = env.pos, env.dest
    if r != dr:
        return 1 if dr > r else 0
    return 3 if dc > c else 2


def planner_policy(env):
    """Dijkstra shortest-path oracle with full knowledge of the hotspot map and step costs."""
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


if __name__ == "__main__":
    orders = load_orders(DATA_PATH)
    _, test_orders = split_orders(orders)
    env = DeliveryEnv(test_orders)

    results = [
        evaluate_policy(random_policy, env, name="Random"),
        evaluate_policy(greedy_policy, env, name="Greedy"),
        evaluate_policy(planner_policy, env, name="Planner"),
    ]

    v1_path = RESULT_DIR / "dqn_model_v1.pt"
    if v1_path.exists():
        net_v1 = QNetV1()
        net_v1.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True))
        net_v1.eval()
        results.append(evaluate_policy(dqn_policy_v1(net_v1), env, name="DQN v1"))

    net_v2, net_v3, net_tuned = None, None, None

    v2_path = RESULT_DIR / "dqn_model_v2.pt"
    if v2_path.exists():
        net_v2 = QNetV2()
        net_v2.load_state_dict(torch.load(v2_path, map_location="cpu", weights_only=True))
        net_v2.eval()
        results.append(evaluate_policy(dqn_policy_v2(net_v2), env, name="DQN v2"))

    v3_path = RESULT_DIR / "dqn_model_v3.pt"
    if v3_path.exists():
        net_v3 = QNetV2(obs_dim=OBS_DIM_V3)
        net_v3.load_state_dict(torch.load(v3_path, map_location="cpu", weights_only=True))
        net_v3.eval()
        results.append(evaluate_policy(dqn_policy_v3(net_v3), env, name="DQN v3"))

    tuned_path = RESULT_DIR / "dqn_model_tuned.pt"
    if tuned_path.exists():
        ckpt = torch.load(tuned_path, map_location="cpu", weights_only=False)
        net_tuned = QNetV2(obs_dim=OBS_DIM_V3, hidden=ckpt.get("hidden", 128))
        net_tuned.load_state_dict(ckpt["state_dict"])
        net_tuned.eval()
        results.append(evaluate_policy(dqn_policy_v3(net_tuned), env, name="Tuned DDQN"))

    if net_v2 is not None and net_v3 is not None and net_tuned is not None:
        from dqn.part_b_refine import make_ensemble_policy
        ens_pol = make_ensemble_policy(net_v2, net_v3, net_tuned)
        results.append(evaluate_policy(ens_pol, env, name="Ensemble DQN"))

    table = pd.DataFrame(results)
    print(table.round(3).to_string(index=False))
    table.to_csv(RESULT_PATH, index=False)

    # Also save a visual 10x10 route comparison map for the report
    if net_v2 is not None and net_v3 is not None and net_tuned is not None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        map_agents = {
            "Greedy (Straight-Line)": greedy_policy,
            "DQN v2 (Direction Only)": dqn_policy_v2(net_v2),
            "Ensemble DDQN (Hotspot-Aware)": ens_pol,
            "Dijkstra Planner (Oracle)": planner_policy,
        }
        # Automatically pick a test order where Greedy hits >=2 hotspots and Ensemble DDQN avoids them
        sample_seed = 0
        for s_try in range(100):
            env.reset(seed=s_try)
            hg, done = 0, False
            while not done:
                _, _, t, tr, _ = env.step(greedy_policy(env))
                hg += int(env.hotspots[env.pos])
                done = t or tr
            env.reset(seed=s_try)
            hd, done = 0, False
            while not done:
                _, _, t, tr, _ = env.step(ens_pol(env))
                hd += int(env.hotspots[env.pos])
                done = t or tr
            if hg >= 2 and hd == 0:
                sample_seed = s_try
                break
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
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
            ax.set_title(f"{title}\nReward: {rew:.1f} | Time: {el_min:.1f}m | Hotspots: {hot}", fontsize=10, fontweight="bold")
            ax.set_xlim(-0.5, 9.5)
            ax.set_ylim(9.5, -0.5)
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))

        plt.suptitle(f"10x10 City Grid Route Comparison (Test Order #{sample_seed})", fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "route_comparison_map.png", dpi=180, bbox_inches="tight")
        plt.close()
        print("Saved visual route map to:", RESULT_DIR / "route_comparison_map.png")
