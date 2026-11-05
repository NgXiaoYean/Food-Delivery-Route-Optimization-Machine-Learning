
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from dqn.dqn_v2 import QNetwork, dqn_policy
from env import DeliveryEnv, load_orders
from evaluate import greedy_policy, planner_policy
from shared import split_orders


# ==========================================
# PATH CONFIGURATION
# ==========================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"

MODEL_PATH = BASE_DIR / "dqn" / "result" / "dqn_model_v2.pt"

RESULT_DIR = BASE_DIR / "result" / "error_analysis"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAFFIC = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Jam"
}

WEATHER = {
    0: "Clear",
    1: "Windy",
    2: "Fog/Sand",
    3: "Stormy"
}


# ==========================================
# EPISODE EVALUATION
# ==========================================

def run_episodes(policy_fn, env, episodes=500):
    """Run a policy on seeds 0..episodes-1 and record details."""

    rows = []

    for ep in range(episodes):

        env.reset(seed=ep)

        dist = (
            abs(env.pos[0] - env.dest[0])
            + abs(env.pos[1] - env.dest[1])
        )

        traffic = env.traffic
        weather = env.weather

        done, total, hot = False, 0.0, 0

        while not done:

            _, r, term, trunc, info = env.step(policy_fn(env))

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
            delivered=info["delivered"],
            hotspot_steps=hot
        ))

    return pd.DataFrame(rows)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    # Load dataset
    orders = load_orders(DATA_PATH)

    _, test_orders = split_orders(orders)

    env = DeliveryEnv(test_orders)

    # Load trained DQN v2
    net = QNetwork()

    net.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    net.eval()

    # Compare agents
    agents = {
        "Greedy": greedy_policy,
        "Planner": planner_policy,
        "DQN v2": dqn_policy(net)
    }

    runs = pd.concat([
        run_episodes(fn, env).assign(agent=name)
        for name, fn in agents.items()
    ], ignore_index=True)

    # Feature categories
    runs["traffic_level"] = runs["traffic"].map(TRAFFIC)

    runs["weather_type"] = runs["weather"].map(WEATHER)

    runs["distance_band"] = pd.cut(
        runs["distance"],
        [0, 6, 12, 100],
        labels=["short (<=6)", "medium (7-12)", "long (13+)"]
    )

    # Save episode details
    runs.to_csv(
        RESULT_DIR / "episode_details.csv",
        index=False
    )

    # ======================================
    # REWARD BREAKDOWN ANALYSIS
    # ======================================

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    breakdowns = [
        ("traffic_level", list(TRAFFIC.values())),
        ("weather_type", list(WEATHER.values())),
        ("distance_band", [
            "short (<=6)",
            "medium (7-12)",
            "long (13+)"
        ])
    ]

    for ax, (key, order) in zip(axes, breakdowns):

        table = (
            runs.pivot_table(
                index=key,
                columns="agent",
                values="reward",
                aggfunc="mean",
                observed=True
            )
            .reindex(order)
        )

        # Reward gap compared with planner
        table["Greedy gap"] = table["Planner"] - table["Greedy"]

        table["DQN gap"] = table["Planner"] - table["DQN v2"]

        table["episodes"] = (
            runs[runs["agent"] == "DQN v2"]
            .groupby(key, observed=True)
            .size()
            .reindex(order)
        )

        print(
            f"\n=== Mean reward by {key} "
            "(gap = planner - agent; bigger = worse) ==="
        )

        print(table.round(2).to_string())

        table.to_csv(
            RESULT_DIR / f"breakdown_{key}.csv"
        )

        table[["Greedy gap", "DQN gap"]].plot(
            kind="bar",
            ax=ax,
            title=f"Reward gap vs planner by {key}"
        )

        ax.set_ylabel("Reward lost vs planner")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "error_analysis.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # HOTSPOT ANALYSIS
    # ======================================

    print("\n=== Average steps taken inside hotspot cells ===")

    hotspot_table = (
        runs.pivot_table(
            index="traffic_level",
            columns="agent",
            values="hotspot_steps",
            aggfunc="mean"
        )
        .reindex(list(TRAFFIC.values()))
    )

    print(hotspot_table.round(2).to_string())

    hotspot_table.to_csv(
        RESULT_DIR / "hotspot_analysis.csv"
    )

    print(f"\nAnalysis completed. Results saved to: {RESULT_DIR}")
