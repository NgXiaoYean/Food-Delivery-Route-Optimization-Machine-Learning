"""
Error and Pattern Analysis across Traffic, Weather, Distance Bands, and Hotspots.
Compares Greedy, Planner (Dijkstra Oracle), Q-Learning (Tuned), DQN v2,
Dueling Double DQN, and the 3-Model Ensemble across 500 unseen test orders.

Run:  python analyze_errors.py
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch

from dqn.dqn_v2 import QNetwork, dqn_policy as dqn_policy_v2
from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3
from env import DeliveryEnv, load_orders
from evaluate import greedy_policy, make_cross_algorithm_ensemble, planner_policy
from shared import split_orders

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
QL_PATH = BASE_DIR / "q_learning" / "result" / "ql_model_best.pkl"
DQN_V2_PATH = BASE_DIR / "dqn" / "result" / "dqn_model_v2.pt"
DQN_V3_PATH = BASE_DIR / "dqn" / "result" / "dqn_model_v3.pt"
DDQN_PATH = BASE_DIR / "ddqn" / "result" / "ddqn_model_best.pt"
TUNED_FALLBACK = BASE_DIR / "dqn" / "result" / "dqn_model_tuned.pt"

RESULT_DIR = BASE_DIR / "result" / "error_analysis"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAFFIC = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}
WEATHER = {0: "Clear", 1: "Windy", 2: "Fog/Sand", 3: "Stormy"}


def run_episodes(policy_fn, env, episodes: int = 500) -> pd.DataFrame:
    """Run a policy on seeds 0..episodes-1 and record episode-level error metrics."""
    rows = []
    for ep in range(episodes):
        env.reset(seed=ep)
        dist = abs(env.pos[0] - env.dest[0]) + abs(env.pos[1] - env.dest[1])
        traffic = env.traffic
        weather = env.weather
        done, total, hot = False, 0.0, 0

        while not done:
            _, r, term, trunc, info = env.step(int(policy_fn(env)))
            total += r
            hot += int(env.hotspots[env.pos])
            done = term or trunc

        rows.append(dict(
            episode=ep,
            traffic=traffic,
            weather=weather,
            distance=dist,
            reward=total,
            steps=info["steps"],
            minutes=info["elapsed_min"],
            on_time=int(info["delivered"] and info["elapsed_min"] <= env.benchmark_min),
            delivered=info["delivered"],
            hotspot_steps=hot,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    try:
        orders = load_orders(DATA_PATH)
        _, test_orders = split_orders(orders)
        env = DeliveryEnv(test_orders)
    except Exception as exc:
        print(f"[ERROR] Could not load test dataset: {exc}")
        sys.exit(1)

    agents = {
        "Greedy": greedy_policy,
        "Planner": planner_policy,
    }

    ql_agent = None
    if QL_PATH.exists():
        from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy
        ql_agent = TabularQLearningAgent.load(QL_PATH)
        agents["Q-Learning"] = make_ql_policy(ql_agent)

    net_v2, net_v3, ddqn_net = None, None, None
    if DQN_V2_PATH.exists():
        net_v2 = QNetwork()
        net_v2.load_state_dict(torch.load(DQN_V2_PATH, map_location="cpu", weights_only=True))
        net_v2.eval()
        agents["DQN v2"] = dqn_policy_v2(net_v2)

    if DQN_V3_PATH.exists():
        net_v3 = QNetwork(obs_dim=OBS_DIM_V3)
        net_v3.load_state_dict(torch.load(DQN_V3_PATH, map_location="cpu", weights_only=True))
        net_v3.eval()

    if DDQN_PATH.exists():
        from ddqn.ddqn_pipeline import load_best_ddqn, make_ddqn_policy
        ddqn_net = load_best_ddqn(DDQN_PATH)
        agents["Dueling DDQN"] = make_ddqn_policy(ddqn_net)
    elif TUNED_FALLBACK.exists():
        ckpt = torch.load(TUNED_FALLBACK, map_location="cpu", weights_only=False)
        ddqn_net = QNetwork(obs_dim=OBS_DIM_V3, hidden=ckpt.get("hidden", 128))
        ddqn_net.load_state_dict(ckpt["state_dict"])
        ddqn_net.eval()
        agents["Dueling DDQN"] = dqn_policy_v3(ddqn_net)

    if net_v2 is not None and ddqn_net is not None:
        agents["Ensemble"] = make_cross_algorithm_ensemble(ql_agent, net_v2, net_v3, ddqn_net)

    runs = pd.concat([
        run_episodes(fn, env).assign(agent=name)
        for name, fn in agents.items()
    ], ignore_index=True)

    runs["traffic_level"] = runs["traffic"].map(TRAFFIC)
    runs["weather_type"] = runs["weather"].map(WEATHER)
    runs["distance_band"] = pd.cut(
        runs["distance"],
        [0, 6, 12, 100],
        labels=["short (<=6)", "medium (7-12)", "long (13+)"],
    )
    runs.to_csv(RESULT_DIR / "episode_details.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    breakdowns = [
        ("traffic_level", list(TRAFFIC.values())),
        ("weather_type", list(WEATHER.values())),
        ("distance_band", ["short (<=6)", "medium (7-12)", "long (13+)"]),
    ]

    for ax, (key, order) in zip(axes, breakdowns):
        table = (
            runs.pivot_table(index=key, columns="agent", values="reward", aggfunc="mean", observed=True)
            .reindex(order)
        )
        gap_cols = []
        for col in ["Greedy", "Q-Learning", "DQN v2", "Dueling DDQN", "Ensemble"]:
            if col in table.columns:
                g_col = f"{col} gap"
                table[g_col] = table["Planner"] - table[col]
                gap_cols.append(g_col)

        table["episodes"] = runs[runs["agent"] == "Planner"].groupby(key, observed=True).size().reindex(order)
        print(f"\n=== Mean reward by {key} (gap = Planner - Agent; lower gap = better) ===")
        print(table.round(2).to_string())
        table.to_csv(RESULT_DIR / f"breakdown_{key}.csv")

        table[gap_cols].plot(kind="bar", ax=ax, title=f"Reward Gap vs Oracle Planner by {key}")
        ax.set_ylabel("Reward Lost vs Planner (Lower = Better)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(RESULT_DIR / "error_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("\n=== Average steps taken inside hotspot cells ===")
    hotspot_table = (
        runs.pivot_table(index="traffic_level", columns="agent", values="hotspot_steps", aggfunc="mean")
        .reindex(list(TRAFFIC.values()))
    )
    print(hotspot_table.round(2).to_string())
    hotspot_table.to_csv(RESULT_DIR / "hotspot_analysis.csv")
    print(f"\nError analysis saved to: {RESULT_DIR}")
