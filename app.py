"""
Interactive Streamlit Prototype for Food Delivery Route Optimization (RL).
Run in terminal:
    streamlit run app.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dqn.dqn_v1 import QNetwork as QNetV1, dqn_policy as dqn_policy_v1
from dqn.dqn_v2 import QNetwork as QNetV2, dqn_policy as dqn_policy_v2
from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3
from dqn.part_b_refine import make_ensemble_policy
from env import DeliveryEnv, load_orders
from evaluate import greedy_policy, planner_policy, random_policy
from shared import split_orders

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
DQN_RESULT_DIR = BASE_DIR / "dqn" / "result"
ERR_RESULT_DIR = BASE_DIR / "result" / "error_analysis"
EDA_DIR = BASE_DIR / "exploration" / "zomato_exploration"

TRAFFIC_NAMES = {0: "Low 🟢", 1: "Medium 🟡", 2: "High 🟠", 3: "Jam 🔴"}
WEATHER_NAMES = {0: "Clear / Sunny ☀️", 1: "Windy 💨", 2: "Fog / Sandstorms 🌫️", 3: "Stormy ⛈️"}
ACTION_NAMES = {0: "↑ Up", 1: "↓ Down", 2: "← Left", 3: "→ Right"}

st.set_page_config(
    page_title="Food Delivery Route Optimization (RL)",
    page_icon="🛵",
    layout="wide",
)


@st.cache_resource
def load_data_and_models():
    orders = load_orders(DATA_PATH)
    train_orders, test_orders = split_orders(orders)

    models = {}
    # Load DQN v1
    p1 = DQN_RESULT_DIR / "dqn_model_v1.pt"
    if p1.exists():
        m1 = QNetV1()
        m1.load_state_dict(torch.load(p1, map_location="cpu", weights_only=True))
        m1.eval()
        models["DQN v1 (Raw 8D)"] = (m1, dqn_policy_v1(m1))

    # Load DQN v2
    p2 = DQN_RESULT_DIR / "dqn_model_v2.pt"
    m2 = None
    if p2.exists():
        m2 = QNetV2()
        m2.load_state_dict(torch.load(p2, map_location="cpu", weights_only=True))
        m2.eval()
        models["DQN v2 (+Direction)"] = (m2, dqn_policy_v2(m2))

    # Load DQN v3
    p3 = DQN_RESULT_DIR / "dqn_model_v3.pt"
    m3 = None
    if p3.exists():
        m3 = QNetV2(obs_dim=OBS_DIM_V3)
        m3.load_state_dict(torch.load(p3, map_location="cpu", weights_only=True))
        m3.eval()
        models["DQN v3 (+Hotspot Radar)"] = (m3, dqn_policy_v3(m3))

    # Load Tuned Double DQN (Part B)
    pt = DQN_RESULT_DIR / "dqn_model_tuned.pt"
    mt = None
    if pt.exists():
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        hidden = ckpt.get("hidden", 128)
        mt = QNetV2(obs_dim=OBS_DIM_V3, hidden=hidden)
        mt.load_state_dict(ckpt["state_dict"])
        mt.eval()
        models["Tuned Double DQN"] = (mt, dqn_policy_v3(mt))

    # Build Ensemble (Part B.2)
    if m2 is not None and m3 is not None and mt is not None:
        models["Ensemble DQN (Part B)"] = (None, make_ensemble_policy(m2, m3, mt))

    return train_orders, test_orders, models


def simulate_single_episode(policy_fn, test_orders, seed, custom_cfg=None):
    """Run one episode and record every grid position visited."""
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

    start_pos = env.pos
    dest_pos = env.dest
    path = [env.pos]
    actions_taken = []
    total_reward = 0.0
    hotspot_hits = 0
    done = False
    info = {"delivered": False, "steps": 0, "elapsed_min": 0.0}

    while not done:
        a = policy_fn(env)
        actions_taken.append(ACTION_NAMES.get(a, str(a)))
        _, r, term, trunc, info = env.step(a)
        total_reward += r
        path.append(env.pos)
        if env.hotspots[env.pos]:
            hotspot_hits += 1
        done = term or trunc

    return {
        "env": env,
        "start": start_pos,
        "dest": dest_pos,
        "path": path,
        "actions": actions_taken,
        "reward": round(total_reward, 2),
        "steps": info["steps"],
        "minutes": round(info["elapsed_min"], 1),
        "benchmark_min": round(env.benchmark_min, 1),
        "delivered": info["delivered"],
        "hotspot_hits": hotspot_hits,
    }


def draw_route_map(sim_data, title):
    """Plot the 10x10 delivery grid, hotspots, start/destination, and agent trajectory."""
    env = sim_data["env"]
    n = env.n
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    # Draw grid cells and hotspots
    for r in range(n):
        for c in range(n):
            is_hot = env.hotspots[r, c]
            face = "#ffd6d6" if is_hot else "#f8f9fa"
            edge = "#e0a0a0" if is_hot else "#dee2e6"
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face, edgecolor=edge, linewidth=1)
            ax.add_patch(rect)
            if is_hot:
                ax.text(c, r, "×", ha="center", va="center", color="#cc0000", fontsize=11, fontweight="bold", alpha=0.6)

    # Plot trajectory path
    path = sim_data["path"]
    if len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        color = "#1f77b4" if sim_data["delivered"] else "#d62728"
        ax.plot(cols, rows, color=color, linewidth=2.6, marker="o", markersize=4, alpha=0.85, label="Route")

        # Draw direction arrows along path
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            if (r1, c1) != (r2, c2):
                ax.annotate(
                    "",
                    xy=(c2, r2),
                    xytext=(c1, r1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
                )

    # Start (Restaurant) and Destination (Customer)
    sr, sc = sim_data["start"]
    dr, dc = sim_data["dest"]
    ax.scatter([sc], [sr], c="#2ca02c", s=180, marker="s", zorder=5, edgecolors="black", label="Restaurant (Start)")
    ax.scatter([dc], [dr], c="#ff7f0e", s=220, marker="*", zorder=6, edgecolors="black", label="Customer (Goal)")
    ax.text(sc, sr, "R", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=7)
    ax.text(dc, dr, "C", ha="center", va="center", color="black", fontsize=7, fontweight="bold", zorder=7)

    status = "✅ Delivered" if sim_data["delivered"] else "❌ Timeout"
    ax.set_title(
        f"{title}\n{status} | Reward: {sim_data['reward']} | Steps: {sim_data['steps']} | Hotspots: {sim_data['hotspot_hits']}",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # row 0 at top
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig


# ==============================================================================
# MAIN UI LAYOUT
# ==============================================================================

st.title("🛵 Food Delivery Route Optimization — RL Prototype")
st.markdown(
    """
    **How this system works:** Every episode samples a real delivery order from the **Kaggle Zomato Dataset** 
    (distance, road traffic, weather, and benchmark delivery duration) and places a **Restaurant (`R`)** and **Customer (`C`)** 
    on a **10×10 city grid** with congested **Traffic Hotspots (`×`)**. Reinforcement Learning agents learn to navigate from `R` to `C` 
    while minimising travel time and avoiding hotspots.
    """
)

train_orders, test_orders, loaded_models = load_data_and_models()

all_policies = {
    "Random (Baseline)": random_policy,
    "Greedy (Straight-Line)": greedy_policy,
    "Planner (Dijkstra Oracle)": planner_policy,
}
for name, (_, pol) in loaded_models.items():
    all_policies[name] = pol

tab_sim, tab_part_a, tab_part_b, tab_eda = st.tabs([
    "🗺️ 1. Live Route Simulator (10×10 Map)",
    "📊 2. Part A: Model Evaluation & Error Analysis",
    "🚀 3. Part B: Tuning, DDQN, Ensemble & Stress Test",
    "🔍 4. Zomato Dataset & EDA",
])

# ------------------------------------------------------------------------------
# TAB 1: LIVE ROUTE SIMULATOR
# ------------------------------------------------------------------------------
with tab_sim:
    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.subheader("⚙️ Order & Map Settings")
        seed = st.number_input("Test Order ID (Seed 0–499)", min_value=0, max_value=499, value=7, step=1)
        use_custom = st.checkbox("Override with Custom Traffic / Weather / Positions", value=False)

        custom_cfg = {"use_custom": use_custom, "hotspot_frac": 0.15}
        if use_custom:
            custom_cfg["traffic"] = st.selectbox("Traffic Level", [0, 1, 2, 3], index=3, format_func=lambda x: TRAFFIC_NAMES[x])
            custom_cfg["weather"] = st.selectbox("Weather", [0, 1, 2, 3], index=3, format_func=lambda x: WEATHER_NAMES[x])
            custom_cfg["hotspot_frac"] = st.slider("Hotspot Density", 0.05, 0.40, 0.20, 0.05)
            c1, c2 = st.columns(2)
            sr = c1.number_input("Start Row", 0, 9, 1)
            sc = c2.number_input("Start Col", 0, 9, 1)
            dr = c1.number_input("Dest Row", 0, 9, 8)
            dc = c2.number_input("Dest Col", 0, 9, 8)
            custom_cfg["start"] = (int(sr), int(sc))
            custom_cfg["dest"] = (int(dr), int(dc))

        default_agents = [
            a for a in ["Greedy (Straight-Line)", "Planner (Dijkstra Oracle)", "DQN v2 (+Direction)", "Ensemble DQN (Part B)"]
            if a in all_policies
        ]
        selected_agents = st.multiselect(
            "Select Agents to Compare:",
            options=list(all_policies.keys()),
            default=default_agents or list(all_policies.keys())[:3],
        )

    with col_map:
        if not selected_agents:
            st.warning("Please select at least one agent on the left.")
        else:
            sims = {
                name: simulate_single_episode(all_policies[name], test_orders, int(seed), custom_cfg)
                for name in selected_agents
            }
            first_sim = next(iter(sims.values()))
            env_ref = first_sim["env"]
            manhattan = abs(first_sim["start"][0] - first_sim["dest"][0]) + abs(first_sim["start"][1] - first_sim["dest"][1])

            # Order info banner
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Traffic Condition", TRAFFIC_NAMES[env_ref.traffic])
            m2.metric("Weather Condition", WEATHER_NAMES[env_ref.weather])
            m3.metric("Grid Distance", f"{manhattan} cells")
            m4.metric("Zomato Target Time", f"{first_sim['benchmark_min']} min")
            m5.metric("Start → Customer", f"{first_sim['start']} → {first_sim['dest']}")

            # Plot maps side by side
            cols = st.columns(min(len(selected_agents), 4))
            for idx, (agent_name, sim) in enumerate(sims.items()):
                with cols[idx % len(cols)]:
                    fig = draw_route_map(sim, agent_name)
                    st.pyplot(fig)
                    plt.close(fig)

            # Summary table for this episode
            st.markdown("#### 📋 Episode Comparison Table")
            ep_rows = []
            for agent_name, sim in sims.items():
                ep_rows.append({
                    "Agent": agent_name,
                    "Delivered": "✅ Yes" if sim["delivered"] else "❌ Timeout",
                    "Cumulative Reward": sim["reward"],
                    "Steps Taken": sim["steps"],
                    "Simulated Time (min)": sim["minutes"],
                    "Zomato Benchmark (min)": sim["benchmark_min"],
                    "Hotspot Cells Entered": sim["hotspot_hits"],
                    "Route Actions": " ".join(sim["actions"][:18]) + (" ..." if len(sim["actions"]) > 18 else ""),
                })
            st.dataframe(pd.DataFrame(ep_rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: PART A EVALUATION & ERROR ANALYSIS
# ------------------------------------------------------------------------------
with tab_part_a:
    st.subheader("📊 Part A: Benchmark Evaluation on 500 Unseen Test Orders")
    res_csv = DQN_RESULT_DIR / "results.csv"
    if res_csv.exists():
        df_res = pd.read_csv(res_csv)
        st.dataframe(df_res.style.format({
            "success_rate": "{:.1%}",
            "avg_reward": "{:.2f}",
            "avg_steps": "{:.2f}",
            "avg_delivery_time_min": "{:.2f}",
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📈 DQN Iterative Progression (v1 → v2 → v3 Learning Curves)")
    c1, c2, c3 = st.columns(3)
    for col, ver, desc in [
        (c1, "v1", "DQN v1: Raw 8D State (7.2% Success — Ping-Pong Loop)"),
        (c2, "v2", "DQN v2: +Relative Direction & Reward Scaling (100% Success)"),
        (c3, "v3", "DQN v3: +4 Neighbor Hotspot Sensors (98.8% Success)"),
    ]:
        img_path = DQN_RESULT_DIR / f"dqn_learning_curve_{ver}.png"
        if img_path.exists():
            col.image(str(img_path), caption=desc, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Error & Hotspot Analysis Across Traffic, Weather & Distance")
    err_img = ERR_RESULT_DIR / "error_analysis.png"
    if err_img.exists():
        st.image(str(err_img), caption="Reward Gap Compared to Dijkstra Oracle Planner Across Conditions", use_container_width=True)

    hot_csv = ERR_RESULT_DIR / "hotspot_analysis.csv"
    if hot_csv.exists():
        st.markdown("**Average Steps Taken Inside Congested Traffic Hotspots:**")
        st.dataframe(pd.read_csv(hot_csv), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 3: PART B REFINEMENT, DDQN, ENSEMBLE & ROBUSTNESS
# ------------------------------------------------------------------------------
with tab_part_b:
    st.subheader("🚀 Part B: System Refinement, Double DQN, Ensembling & Stress Testing")
    part_b_img = DQN_RESULT_DIR / "part_b_analysis.png"
    if part_b_img.exists():
        st.image(str(part_b_img), caption="Part B Overview: Hyperparameter Tuning, Model Ensembling, and Stress Testing", use_container_width=True)

    c_left, c_right = st.columns(2)
    tune_csv = DQN_RESULT_DIR / "part_b_tuning_results.csv"
    if tune_csv.exists():
        with c_left:
            st.markdown("#### 1. Automated Hyperparameter & Double-DQN Search")
            st.dataframe(pd.read_csv(tune_csv), use_container_width=True, hide_index=True)

    ens_csv = DQN_RESULT_DIR / "part_b_ensemble_results.csv"
    if ens_csv.exists():
        with c_right:
            st.markdown("#### 2. Individual Models vs Combined Ensemble (500 Orders)")
            st.dataframe(pd.read_csv(ens_csv), use_container_width=True, hide_index=True)

    rob_csv = DQN_RESULT_DIR / "part_b_robustness_results.csv"
    if rob_csv.exists():
        st.markdown("#### 3. Robustness Testing Across Extreme Scenarios")
        st.dataframe(pd.read_csv(rob_csv), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 4: DATASET & EDA
# ------------------------------------------------------------------------------
with tab_eda:
    st.subheader("🔍 Zomato Delivery Operations Dataset & Cleaning Audit")
    log_csv = BASE_DIR / "data" / "cleaning_log.csv"
    if log_csv.exists():
        st.markdown("#### Data Cleaning Pipeline Log (`clean_data.py`)")
        st.dataframe(pd.read_csv(log_csv), use_container_width=True, hide_index=True)

    if EDA_DIR.exists():
        st.markdown("#### Exploratory Data Analysis Visualizations (`exploration/zomato_exploration/`)")
        png_files = sorted(EDA_DIR.glob("*.png"))
        cols = st.columns(3)
        for idx, p in enumerate(png_files):
            cols[idx % 3].image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)
