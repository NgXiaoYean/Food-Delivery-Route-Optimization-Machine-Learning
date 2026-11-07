"""
Interactive Streamlit Prototype for Food Delivery Route Optimization (RL).
Integrates all 3 RL Algorithms (Tabular Q-Learning, DQN, Dueling Double DQN),
the Part B Cross-Algorithm Ensemble, Baselines, Error Analysis, and EDA.

Launch from project root:
    streamlit run app.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st
import torch

from env import DeliveryEnv, load_orders
from shared import split_orders
from evaluate import (
    greedy_policy,
    make_cross_algorithm_ensemble,
    planner_policy,
    random_policy,
)
from dqn.dqn_v1 import QNetwork as QNetV1, dqn_policy as dqn_policy_v1
from dqn.dqn_v2 import QNetwork as QNetV2, dqn_policy as dqn_policy_v2
from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
QL_RESULT_DIR = BASE_DIR / "q_learning" / "result"
DQN_RESULT_DIR = BASE_DIR / "dqn" / "result"
DDQN_RESULT_DIR = BASE_DIR / "ddqn" / "result"
MASTER_RESULT_DIR = BASE_DIR / "result"
ERR_RESULT_DIR = MASTER_RESULT_DIR / "error_analysis"
EDA_DIR = BASE_DIR / "exploration" / "zomato_exploration"

TRAFFIC_NAMES = {0: "Low 🟢", 1: "Medium 🟡", 2: "High 🟠", 3: "Jam 🔴"}
WEATHER_NAMES = {0: "Clear / Sunny ☀️", 1: "Windy 💨", 2: "Fog / Sandstorms 🌫️", 3: "Stormy ⛈️"}
ACTION_NAMES = {0: "↑ Up", 1: "↓ Down", 2: "← Left", 3: "→ Right"}

st.set_page_config(
    page_title="Food Delivery Route Optimization — 3-Algorithm RL System",
    page_icon="🛵",
    layout="wide",
)


@st.cache_resource
def load_all_data_and_policies():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset missing at {DATA_PATH}. Run `python clean_data.py` first.")
    orders = load_orders(DATA_PATH)
    train_orders, test_orders = split_orders(orders)

    policies = {
        "Random (Baseline)": random_policy,
        "Greedy (Straight-Line)": greedy_policy,
        "Planner (Dijkstra Oracle)": planner_policy,
    }

    # 1. Algorithm 1: Tabular Q-Learning
    ql_agent = None
    ql_ckpt = QL_RESULT_DIR / "ql_model_best.pkl"
    if ql_ckpt.exists():
        try:
            from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy
            ql_agent = TabularQLearningAgent.load(ql_ckpt)
            policies["1. Q-Learning (Tabular)"] = make_ql_policy(ql_agent)
        except Exception:
            pass

    # 2. Algorithm 2: Deep Q-Network (v1, v2, v3)
    m1, m2, m3 = None, None, None
    p1 = DQN_RESULT_DIR / "dqn_model_v1.pt"
    if p1.exists():
        m1 = QNetV1()
        m1.load_state_dict(torch.load(p1, map_location="cpu", weights_only=True))
        m1.eval()
        policies["2a. DQN v1 (Raw 8D)"] = dqn_policy_v1(m1)

    p2 = DQN_RESULT_DIR / "dqn_model_v2.pt"
    if p2.exists():
        m2 = QNetV2()
        m2.load_state_dict(torch.load(p2, map_location="cpu", weights_only=True))
        m2.eval()
        policies["2b. DQN v2 (+Direction)"] = dqn_policy_v2(m2)

    p3 = DQN_RESULT_DIR / "dqn_model_v3.pt"
    if p3.exists():
        m3 = QNetV2(obs_dim=OBS_DIM_V3)
        m3.load_state_dict(torch.load(p3, map_location="cpu", weights_only=True))
        m3.eval()
        policies["2c. DQN v3 (+Hotspot Radar)"] = dqn_policy_v3(m3)

    # 3. Algorithm 3: Dueling Double DQN (D3QN)
    ddqn_net = None
    ddqn_ckpt = DDQN_RESULT_DIR / "ddqn_model_best.pt"
    tuned_fallback = DQN_RESULT_DIR / "dqn_model_tuned.pt"
    if ddqn_ckpt.exists():
        try:
            from ddqn.ddqn_pipeline import load_best_ddqn, make_ddqn_policy
            ddqn_net = load_best_ddqn(ddqn_ckpt)
            policies["3. Dueling Double DQN (D3QN)"] = make_ddqn_policy(ddqn_net)
        except Exception:
            pass
    elif tuned_fallback.exists():
        ckpt = torch.load(tuned_fallback, map_location="cpu", weights_only=False)
        ddqn_net = QNetV2(obs_dim=OBS_DIM_V3, hidden=ckpt.get("hidden", 128))
        ddqn_net.load_state_dict(ckpt["state_dict"])
        ddqn_net.eval()
        policies["3. Dueling Double DQN (D3QN)"] = dqn_policy_v3(ddqn_net)

    # 4. Part B Combined 3-Algorithm Ensemble
    if m2 is not None and ddqn_net is not None:
        policies["★ 3-Model Ensemble (QL + DQN + D3QN)"] = make_cross_algorithm_ensemble(ql_agent, m2, m3, ddqn_net)

    return train_orders, test_orders, policies


def simulate_single_episode(policy_fn, test_orders, seed, custom_cfg=None):
    """Simulate a single delivery episode and capture trajectory & metrics."""
    hotspot_frac = custom_cfg["hotspot_frac"] if custom_cfg else 0.15
    env = DeliveryEnv(test_orders, hotspot_frac=hotspot_frac)
    env.reset(seed=seed)

    if custom_cfg and custom_cfg["use_custom"]:
        env.traffic = custom_cfg["traffic"]
        env.weather = custom_cfg["weather"]
        env.pos = custom_cfg["start"]
        env.dest = custom_cfg["dest"]
        env.steps = 0
        env.elapsed_min = 0.0

    start_pos, dest_pos = env.pos, env.dest
    manhattan_dist = abs(start_pos[0] - dest_pos[0]) + abs(start_pos[1] - dest_pos[1])
    path = [env.pos]
    actions_taken = []
    total_reward = 0.0
    hotspot_hits = 0
    done = False
    info = {"delivered": False, "steps": 0, "elapsed_min": 0.0}

    while not done:
        a = int(policy_fn(env))
        actions_taken.append(ACTION_NAMES.get(a, str(a)))
        _, r, term, trunc, info = env.step(a)
        total_reward += r
        path.append(env.pos)
        if env.hotspots[env.pos]:
            hotspot_hits += 1
        done = term or trunc

    on_time = info["delivered"] and (info["elapsed_min"] <= env.benchmark_min)
    eff = (manhattan_dist / max(1, info["steps"])) * 100.0 if info["delivered"] else 0.0

    return {
        "env": env,
        "start": start_pos,
        "dest": dest_pos,
        "path": path,
        "actions": actions_taken,
        "reward": round(total_reward, 2),
        "steps": info["steps"],
        "efficiency_pct": round(min(100.0, eff), 1),
        "minutes": round(info["elapsed_min"], 1),
        "benchmark_min": round(env.benchmark_min, 1),
        "delivered": info["delivered"],
        "on_time": on_time,
        "hotspot_hits": hotspot_hits,
    }


def draw_route_map(sim_data, title):
    """Render 10x10 city map with traffic hotspots, start/goal, and path."""
    env = sim_data["env"]
    n = env.n
    fig, ax = plt.subplots(figsize=(4.4, 4.4))

    for r in range(n):
        for c in range(n):
            is_hot = env.hotspots[r, c]
            face = "#ffd6d6" if is_hot else "#f8f9fa"
            edge = "#e0a0a0" if is_hot else "#dee2e6"
            ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face, edgecolor=edge, linewidth=1))
            if is_hot:
                ax.text(c, r, "×", ha="center", va="center", color="#cc0000", fontsize=11, fontweight="bold", alpha=0.65)

    path = sim_data["path"]
    if len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        color = "#1f77b4" if sim_data["delivered"] else "#d62728"
        ax.plot(cols, rows, color=color, linewidth=2.6, marker="o", markersize=4, alpha=0.85)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            if (r1, c1) != (r2, c2):
                ax.annotate("", xy=(c2, r2), xytext=(c1, r1), arrowprops=dict(arrowstyle="->", color=color, lw=1.7))

    sr, sc = sim_data["start"]
    dr, dc = sim_data["dest"]
    ax.scatter([sc], [sr], c="#2ca02c", s=180, marker="s", zorder=5, edgecolors="black")
    ax.scatter([dc], [dr], c="#ff7f0e", s=220, marker="*", zorder=6, edgecolors="black")
    ax.text(sc, sr, "R", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=7)
    ax.text(dc, dr, "C", ha="center", va="center", color="black", fontsize=7, fontweight="bold", zorder=7)

    status = "✅" if sim_data["delivered"] else "❌"
    ax.set_title(
        f"{title}\n{status} Rew: {sim_data['reward']} | {sim_data['minutes']}m | Hotspots: {sim_data['hotspot_hits']}",
        fontsize=9.2,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN UI
# ==============================================================================

st.title("🛵 Food Delivery Route Optimization — Multi-Algorithm RL Prototype")
st.markdown(
    """
    **3 Reinforcement Learning Algorithms + Cross-Model Ensemble grounded in 40,482 Kaggle Zomato Orders:**  
    Compare **1. Tabular Q-Learning**, **2. Deep Q-Network (DQN)**, **3. Dueling Double DQN (D3QN)**, and the **Part B Combined Ensemble** 
    navigating from **Restaurant (`R`)** to **Customer (`C`)** while avoiding congested **Traffic Hotspots (`×`)**.
    """
)

try:
    train_orders, test_orders, all_policies = load_all_data_and_policies()
except Exception as exc:
    st.error(f"Failed to load data or models: {exc}")
    st.stop()

tab_sim, tab_master, tab_part_a, tab_part_b, tab_eda = st.tabs([
    "🗺️ 1. Live Route Simulator",
    "🏆 2. 3-Algorithm Master Comparison",
    "📊 3. Part A: Iterations & Error Analysis",
    "🚀 4. Part B: Regularization, Missing Data, Ensemble & Stress Test",
    "🔍 5. Zomato Dataset & EDA",
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE SIMULATOR
# ------------------------------------------------------------------------------
with tab_sim:
    col_ctrl, col_map = st.columns([1, 3.2])
    with col_ctrl:
        st.subheader("⚙️ Order & Map Controls")
        seed = st.number_input("Test Order ID (Seed 0–499)", min_value=0, max_value=499, value=7, step=1)
        use_custom = st.checkbox("Custom Scenario Override", value=False)

        custom_cfg = {"use_custom": use_custom, "hotspot_frac": 0.15}
        if use_custom:
            custom_cfg["traffic"] = st.selectbox("Traffic Level", [0, 1, 2, 3], index=3, format_func=lambda x: TRAFFIC_NAMES[x])
            custom_cfg["weather"] = st.selectbox("Weather", [0, 1, 2, 3], index=3, format_func=lambda x: WEATHER_NAMES[x])
            custom_cfg["hotspot_frac"] = st.slider("Hotspot Density", 0.05, 0.40, 0.20, 0.05)
            c1, c2 = st.columns(2)
            sr = c1.number_input("Start Row", 0, 9, 6)
            sc = c2.number_input("Start Col", 0, 9, 6)
            dr = c1.number_input("Dest Row", 0, 9, 9)
            dc = c2.number_input("Dest Col", 0, 9, 1)
            custom_cfg["start"] = (int(sr), int(sc))
            custom_cfg["dest"] = (int(dr), int(dc))

        preferred_defaults = [
            k for k in [
                "Greedy (Straight-Line)",
                "1. Q-Learning (Tabular)",
                "2b. DQN v2 (+Direction)",
                "★ 3-Model Ensemble (QL + DQN + D3QN)",
            ] if k in all_policies
        ]
        selected_agents = st.multiselect(
            "Select Models to Compare on Map:",
            options=list(all_policies.keys()),
            default=preferred_defaults or list(all_policies.keys())[:4],
        )

    with col_map:
        if not selected_agents:
            st.warning("Select at least one model on the left panel.")
        else:
            sims = {name: simulate_single_episode(all_policies[name], test_orders, int(seed), custom_cfg) for name in selected_agents}
            first_sim = next(iter(sims.values()))
            env_ref = first_sim["env"]
            manhattan = abs(first_sim["start"][0] - first_sim["dest"][0]) + abs(first_sim["start"][1] - first_sim["dest"][1])

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Road Traffic", TRAFFIC_NAMES[env_ref.traffic])
            m2.metric("Weather", WEATHER_NAMES[env_ref.weather])
            m3.metric("Manhattan Dist", f"{manhattan} cells")
            m4.metric("Zomato SLA Target", f"{first_sim['benchmark_min']} min")
            m5.metric("Coordinates", f"{first_sim['start']} → {first_sim['dest']}")

            cols = st.columns(min(len(selected_agents), 4))
            for idx, (agent_name, sim) in enumerate(sims.items()):
                with cols[idx % len(cols)]:
                    fig = draw_route_map(sim, agent_name)
                    st.pyplot(fig)
                    plt.close(fig)

            st.markdown("#### 📋 Detailed Episode Performance Metrics")
            rows = []
            for agent_name, sim in sims.items():
                rows.append({
                    "Model": agent_name,
                    "Delivered": "✅ Yes" if sim["delivered"] else "❌ Timeout",
                    "SLA On-Time": "✅ On-Time" if sim["on_time"] else "⚠️ Late",
                    "Reward": sim["reward"],
                    "Steps": sim["steps"],
                    "Route Efficiency": f"{sim['efficiency_pct']}%",
                    "Delivery Time (min)": sim["minutes"],
                    "Zomato Target (min)": sim["benchmark_min"],
                    "Hotspots Entered": sim["hotspot_hits"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: MASTER 3-ALGORITHM LEADERBOARD
# ------------------------------------------------------------------------------
with tab_master:
    st.subheader("🏆 Master Benchmark: Baselines vs. Q-Learning vs. DQN vs. Dueling DDQN vs. Ensemble")
    master_csv = MASTER_RESULT_DIR / "master_all_models_results.csv"
    fallback_csv = DQN_RESULT_DIR / "results.csv"
    csv_to_show = master_csv if master_csv.exists() else fallback_csv

    if csv_to_show.exists():
        df_m = pd.read_csv(csv_to_show)
        fmt = {
            "success_rate": "{:.1%}",
            "avg_reward": "{:.2f}",
            "avg_steps": "{:.2f}",
            "avg_delivery_time_min": "{:.2f}",
        }
        if "on_time_rate" in df_m.columns:
            fmt["on_time_rate"] = "{:.1%}"
        if "std_reward" in df_m.columns:
            fmt["std_reward"] = "{:.2f}"
        if "route_efficiency" in df_m.columns:
            fmt["route_efficiency"] = "{:.1%}"
        if "avg_hotspot_steps" in df_m.columns:
            fmt["avg_hotspot_steps"] = "{:.2f}"
        st.dataframe(df_m.style.format(fmt), use_container_width=True, hide_index=True)

    master_img = MASTER_RESULT_DIR / "master_comparison_chart.png"
    if master_img.exists():
        st.image(str(master_img), caption="4-Metric Benchmark Across All 3 RL Families & Cross-Model Ensemble", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: PART A ITERATIONS & ERROR ANALYSIS
# ------------------------------------------------------------------------------
with tab_part_a:
    st.subheader("📊 Part A: Training Progression Across All 3 Algorithms")
    c_ql, c_dqn, c_ddqn = st.columns(3)

    if (QL_RESULT_DIR / "ql_learning_curves.png").exists():
        c_ql.image(str(QL_RESULT_DIR / "ql_learning_curves.png"), caption="Algorithm 1: Tabular Q-Learning (Raw 40k vs Refined 520 States)", use_container_width=True)
    if (DQN_RESULT_DIR / "dqn_learning_curve_v2.png").exists():
        c_dqn.image(str(DQN_RESULT_DIR / "dqn_learning_curve_v2.png"), caption="Algorithm 2: Deep Q-Network (DQN v2)", use_container_width=True)
    if (DDQN_RESULT_DIR / "ddqn_learning_curves.png").exists():
        c_ddqn.image(str(DDQN_RESULT_DIR / "ddqn_learning_curves.png"), caption="Algorithm 3: Vanilla DDQN vs Dueling DDQN", use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Error & Congestion Analysis Across Traffic, Weather & Distance")
    if (ERR_RESULT_DIR / "error_analysis.png").exists():
        st.image(str(ERR_RESULT_DIR / "error_analysis.png"), use_container_width=True)
    if (ERR_RESULT_DIR / "hotspot_analysis.csv").exists():
        st.markdown("**Mean Steps Inside Congested Hotspot Cells by Traffic Level:**")
        st.dataframe(pd.read_csv(ERR_RESULT_DIR / "hotspot_analysis.csv"), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 4: PART B OPTIMIZATION, REGULARIZATION, MISSING DATA & STRESS TESTS
# ------------------------------------------------------------------------------
with tab_part_b:
    st.subheader("🚀 Part B: Hyperparameter Tuning, Regularization, Missing-Data Study & Stress Testing")

    if (DQN_RESULT_DIR / "part_b_analysis.png").exists():
        st.image(str(DQN_RESULT_DIR / "part_b_analysis.png"), caption="Part B Optimization, Ensembling & Extreme Scenario Stress Testing", use_container_width=True)

    col1, col2 = st.columns(2)
    if (QL_RESULT_DIR / "ql_part_b_tuning.csv").exists():
        with col1:
            st.markdown("#### Algorithm 1 (Q-Learning): LR & L2 Regularization Tuning")
            st.dataframe(pd.read_csv(QL_RESULT_DIR / "ql_part_b_tuning.csv"), use_container_width=True, hide_index=True)

    if (DDQN_RESULT_DIR / "ddqn_part_b_tuning.csv").exists():
        with col2:
            st.markdown("#### Algorithm 3 (Dueling DDQN): Weight Decay & Dropout Regularization")
            st.dataframe(pd.read_csv(DDQN_RESULT_DIR / "ddqn_part_b_tuning.csv"), use_container_width=True, hide_index=True)

    if (MASTER_RESULT_DIR / "missing_data_comparison.csv").exists():
        st.markdown("---")
        st.markdown("#### Part B.1: Missing & Unclear Information Handling Study (`missing_data_study.py`)")
        m_c1, m_c2 = st.columns([1.3, 1])
        m_c1.dataframe(pd.read_csv(MASTER_RESULT_DIR / "missing_data_comparison.csv"), use_container_width=True, hide_index=True)
        if (MASTER_RESULT_DIR / "missing_data_comparison.png").exists():
            m_c2.image(str(MASTER_RESULT_DIR / "missing_data_comparison.png"), use_container_width=True)

    if (DQN_RESULT_DIR / "part_b_robustness_results.csv").exists():
        st.markdown("---")
        st.markdown("#### Part B.3: Reliability Stress Testing Across Extreme Scenarios")
        st.dataframe(pd.read_csv(DQN_RESULT_DIR / "part_b_robustness_results.csv"), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 5: DATASET & EDA
# ------------------------------------------------------------------------------
with tab_eda:
    st.subheader("🔍 Zomato Dataset Cleaning Log & Exploratory Data Analysis")
    if (BASE_DIR / "data" / "cleaning_log.csv").exists():
        st.dataframe(pd.read_csv(BASE_DIR / "data" / "cleaning_log.csv"), use_container_width=True, hide_index=True)
    if EDA_DIR.exists():
        png_files = sorted(EDA_DIR.glob("*.png"))
        cols = st.columns(3)
        for idx, p in enumerate(png_files):
            cols[idx % 3].image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)
