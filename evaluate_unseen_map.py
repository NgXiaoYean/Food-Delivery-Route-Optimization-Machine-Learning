"""Evaluate saved 80/20 single-order checkpoints on Seen Map (map_seed=0) vs Unseen Map (map_seed=202).

Uses the training distance scale (train_max_km) and the exact same 500 test trips (seeds 0..499)
without retraining any model.

Run from the project root:
    python evaluate_unseen_map.py
"""
from pathlib import Path
import pandas as pd
import torch

from dqn.dqn_v1 import QNetwork as QNetV1, dqn_policy as dqn_policy_v1
from dqn.dqn_v2 import QNetwork as QNetV2, dqn_policy as dqn_policy_v2
from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3
from d3qn.d3qn_pipeline import load_best_d3qn, make_d3qn_policy
from env import DeliveryEnv, load_orders
from evaluate import greedy_policy, make_cross_algorithm_ensemble, planner_policy, random_policy
from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy
from shared import evaluate_policy, split_orders_single_3way

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
QL_PATH = BASE_DIR / "q_learning" / "result" / "ql_model_best.pkl"
V1_PATH = BASE_DIR / "dqn" / "result" / "dqn_model_v1.pt"
V2_PATH = BASE_DIR / "dqn" / "result" / "dqn_model_v2.pt"
V3_PATH = BASE_DIR / "dqn" / "result" / "dqn_model_v3.pt"
D3QN_PATH = BASE_DIR / "d3qn" / "result" / "d3qn_model_best.pt"
OUT_CSV = BASE_DIR / "result" / "seen_vs_unseen_map_results.csv"


def load_policies():
    policies = {
        "Random": random_policy,
        "Greedy": greedy_policy,
        "Planner": planner_policy,
    }
    ql_agent = None
    if QL_PATH.exists():
        ql_agent = TabularQLearningAgent.load(QL_PATH)
        policies["Q-Learning (Tuned)"] = make_ql_policy(ql_agent)

    net_v1 = net_v2 = net_v3 = d3qn_net = None
    if V1_PATH.exists():
        net_v1 = QNetV1()
        net_v1.load_state_dict(torch.load(V1_PATH, map_location="cpu", weights_only=True))
        net_v1.eval()
        policies["DQN v1"] = dqn_policy_v1(net_v1)
    if V2_PATH.exists():
        net_v2 = QNetV2()
        net_v2.load_state_dict(torch.load(V2_PATH, map_location="cpu", weights_only=True))
        net_v2.eval()
        policies["DQN v2"] = dqn_policy_v2(net_v2)
    if V3_PATH.exists():
        net_v3 = QNetV2(obs_dim=OBS_DIM_V3)
        net_v3.load_state_dict(torch.load(V3_PATH, map_location="cpu", weights_only=True))
        net_v3.eval()
        policies["DQN v3"] = dqn_policy_v3(net_v3)
    if D3QN_PATH.exists():
        d3qn_net = load_best_d3qn(D3QN_PATH)
        policies["D3QN"] = make_d3qn_policy(d3qn_net)
    if net_v2 is not None and d3qn_net is not None:
        policies["3-Model Ensemble"] = make_cross_algorithm_ensemble(ql_agent, net_v2, net_v3, d3qn_net)
    return policies


def main():
    orders = load_orders(DATA_PATH)
    _, _, test_orders, train_max_km = split_orders_single_3way(orders)
    policies = load_policies()

    rows = []
    for map_label, map_seed in [("Seen map (map_seed=0)", 0), ("Unseen map (map_seed=202)", 202)]:
        env = DeliveryEnv(test_orders, map_seed=map_seed, max_km=train_max_km)
        for name, pol in policies.items():
            res = evaluate_policy(pol, env, episodes=500, name=name)
            rows.append({"map": map_label, "map_seed": map_seed, **res})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nSaved comparison to {OUT_CSV}")


if __name__ == "__main__":
    main()
