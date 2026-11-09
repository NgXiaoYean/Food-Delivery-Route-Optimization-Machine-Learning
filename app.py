"""
Food Delivery Route Dispatch & Analysis

Streamlit interface for evaluating reinforcement learning route models
(Q-Learning, DQN, Dueling Double DQN, and Ensemble) on Zomato delivery orders.

Run from project root:
    streamlit run app.py
"""
from pathlib import Path
import re
import folium
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

from env import DeliveryEnv, TRAFFIC_MAP, WEATHER_MAP, haversine_km, load_orders
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

CITY_PREFIX_MAP = {
    "MUM": "Mumbai",
    "BANG": "Bangalore",
    "PUNE": "Pune",
    "HYD": "Hyderabad",
    "KOL": "Kolkata",
    "CHEN": "Chennai",
    "JAI": "Jaipur",
    "COIMB": "Coimbatore",
    "IND": "Indore",
    "KOC": "Kochi",
    "LUDH": "Ludhiana",
    "DEH": "Dehradun",
    "MYS": "Mysore",
    "AGR": "Agra",
    "ALH": "Allahabad",
    "AURG": "Aurangabad",
    "BHP": "Bhopal",
    "GOA": "Goa",
    "KNPR": "Kanpur",
    "KNP": "Kanpur",
    "RANCHI": "Ranchi",
    "SUR": "Surat",
    "VAD": "Vadodara",
}

TRAFFIC_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}
WEATHER_LABELS = {0: "Clear", 1: "Windy", 2: "Fog", 3: "Stormy"}
ACTION_LABELS = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

ROUTE_STYLES = {
    "Direct Route (Baseline)": {"color": "#64748B", "weight": 3, "dash": "6, 6"},
    "Random Baseline": {"color": "#94A3B8", "weight": 2, "dash": "4, 6"},
    "Shortest-Time Planner (Dijkstra)": {"color": "#475569", "weight": 3, "dash": "4, 4"},
    "Ensemble (QL + DQN + DDQN)": {"color": "#2563EB", "weight": 4, "dash": None},
    "Dueling Double DQN": {"color": "#1D4ED8", "weight": 4, "dash": None},
    "Deep Q-Network (DQN v2)": {"color": "#0284C7", "weight": 4, "dash": None},
    "Deep Q-Network (DQN v3)": {"color": "#0D9488", "weight": 4, "dash": None},
    "Tabular Q-Learning": {"color": "#4F46E5", "weight": 4, "dash": None},
    "Deep Q-Network (DQN v1)": {"color": "#7C3AED", "weight": 3, "dash": "4, 4"},
}

st.set_page_config(
    page_title="Food Delivery Route Dispatch & Analysis",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    .kv-row {
        display: flex;
        justify-content: space-between;
        padding: 0.42rem 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.9rem;
    }
    .kv-row:last-child {
        border-bottom: none;
    }
    .kv-key {
        color: #64748B;
        font-weight: 500;
    }
    .kv-val {
        color: #0F172A;
        font-weight: 600;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_city_name(delivery_person_id: str) -> str:
    m = re.match(r"^([A-Z]+?)RES", str(delivery_person_id).strip())
    if m:
        prefix = m.group(1)
        return CITY_PREFIX_MAP.get(prefix, prefix.title())
    return "Urban"


def load_orders_with_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    df["traffic"] = df["Road_traffic_density"].map(TRAFFIC_MAP)
    df["weather"] = df["Weather_conditions"].str.split().str[-1].map(WEATHER_MAP)
    df["multi"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    tt_col = next(c for c in df.columns if c.lower().replace(" ", "").startswith("time_taken"))
    df["time_taken"] = pd.to_numeric(df[tt_col].astype(str).str.extract(r"(\d+)")[0])
    coords = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for c in coords:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["distance_km"] = haversine_km(df[coords[0]], df[coords[1]], df[coords[2]], df[coords[3]])
    df = df.dropna(subset=["traffic", "weather", "multi", "time_taken", "distance_km"])
    df = df[(df[coords].abs() > 1).all(axis=1)]
    df = df[df["distance_km"] < 50]
    df["city_name"] = df["Delivery_person_ID"].apply(extract_city_name)
    return df.reset_index(drop=True)


@st.cache_resource
def load_system_resources():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset missing at {DATA_PATH}. Run `python clean_data.py` first.")

    orders = load_orders(DATA_PATH)
    train_orders, test_orders = split_orders(orders)

    full_df = load_orders_with_metadata(DATA_PATH)
    _, test_meta = split_orders(full_df)

    catalog_rows = []
    for seed in range(500):
        rng = np.random.default_rng(seed)
        row_idx = int(rng.integers(len(test_meta)))
        row = test_meta.iloc[row_idx]
        catalog_rows.append({
            "seed": seed,
            "order_id": str(row.get("ID", f"ORD-{seed:03d}")),
            "city": str(row.get("city_name", "Urban")),
            "area_type": str(row.get("City", "Metropolitan")),
            "traffic_code": int(row["traffic"]),
            "traffic_label": TRAFFIC_LABELS.get(int(row["traffic"]), "Medium"),
            "weather_code": int(row["weather"]),
            "weather_label": WEATHER_LABELS.get(int(row["weather"]), "Clear"),
            "order_type": str(row.get("Type_of_order", "Meal")).strip(),
            "vehicle_type": str(row.get("Type_of_vehicle", "motorcycle")).strip().title(),
            "rider_rating": float(row.get("Delivery_person_Ratings", 4.6)),
            "distance_km": round(float(row["distance_km"]), 2),
            "sla_min": float(row["time_taken"]),
            "rest_lat": float(row["Restaurant_latitude"]),
            "rest_lon": float(row["Restaurant_longitude"]),
            "cust_lat": float(row["Delivery_location_latitude"]),
            "cust_lon": float(row["Delivery_location_longitude"]),
        })
    catalog_df = pd.DataFrame(catalog_rows)

    policies = {}

    ql_agent = None
    ql_ckpt = QL_RESULT_DIR / "ql_model_best.pkl"
    if ql_ckpt.exists():
        try:
            from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy
            ql_agent = TabularQLearningAgent.load(ql_ckpt)
        except Exception:
            pass

    m1, m2, m3 = None, None, None
    p1 = DQN_RESULT_DIR / "dqn_model_v1.pt"
    if p1.exists():
        m1 = QNetV1()
        m1.load_state_dict(torch.load(p1, map_location="cpu", weights_only=True))
        m1.eval()

    p2 = DQN_RESULT_DIR / "dqn_model_v2.pt"
    if p2.exists():
        m2 = QNetV2()
        m2.load_state_dict(torch.load(p2, map_location="cpu", weights_only=True))
        m2.eval()

    p3 = DQN_RESULT_DIR / "dqn_model_v3.pt"
    if p3.exists():
        m3 = QNetV2(obs_dim=OBS_DIM_V3)
        m3.load_state_dict(torch.load(p3, map_location="cpu", weights_only=True))
        m3.eval()

    ddqn_net = None
    ddqn_ckpt = DDQN_RESULT_DIR / "ddqn_model_best.pt"
    tuned_fallback = DQN_RESULT_DIR / "dqn_model_tuned.pt"
    if ddqn_ckpt.exists():
        try:
            from ddqn.ddqn_pipeline import load_best_ddqn, make_ddqn_policy
            ddqn_net = load_best_ddqn(ddqn_ckpt)
        except Exception:
            pass
    elif tuned_fallback.exists():
        ckpt = torch.load(tuned_fallback, map_location="cpu", weights_only=False)
        ddqn_net = QNetV2(obs_dim=OBS_DIM_V3, hidden=ckpt.get("hidden", 128))
        ddqn_net.load_state_dict(ckpt["state_dict"])
        ddqn_net.eval()

    if m2 is not None and ddqn_net is not None:
        policies["Ensemble (QL + DQN + DDQN)"] = make_cross_algorithm_ensemble(
            ql_agent, m2, m3, ddqn_net
        )
    if ddqn_net is not None:
        try:
            from ddqn.ddqn_pipeline import make_ddqn_policy
            policies["Dueling Double DQN"] = make_ddqn_policy(ddqn_net)
        except Exception:
            policies["Dueling Double DQN"] = dqn_policy_v3(ddqn_net)
    if m2 is not None:
        policies["Deep Q-Network (DQN v2)"] = dqn_policy_v2(m2)
    if m3 is not None:
        policies["Deep Q-Network (DQN v3)"] = dqn_policy_v3(m3)
    if ql_agent is not None:
        from q_learning.q_learning_pipeline import make_ql_policy
        policies["Tabular Q-Learning"] = make_ql_policy(ql_agent)
    if m1 is not None:
        policies["Deep Q-Network (DQN v1)"] = dqn_policy_v1(m1)

    policies["Direct Route (Baseline)"] = greedy_policy
    policies["Shortest-Time Planner (Dijkstra)"] = planner_policy
    policies["Random Baseline"] = random_policy

    return train_orders, test_orders, catalog_df, policies


def simulate_single_episode(policy_fn, test_orders, seed: int, custom_cfg=None):
    hotspot_frac = custom_cfg["hotspot_frac"] if custom_cfg else 0.15
    env = DeliveryEnv(test_orders, hotspot_frac=hotspot_frac)
    env.reset(seed=seed)

    if custom_cfg and custom_cfg.get("use_custom"):
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
        actions_taken.append(ACTION_LABELS.get(a, str(a)))
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


def grid_to_gps_mapper(start_rc, dest_rc, rest_lat, rest_lon, cust_lat, cust_lon):
    sr, sc = start_rc
    dr, dc = dest_rc
    d_row = dr - sr
    d_col = dc - sc

    lat_span = cust_lat - rest_lat
    lon_span = cust_lon - rest_lon

    step_lat = (lat_span / d_row) if d_row != 0 else -(abs(lon_span) / max(abs(d_col), 3))
    step_lon = (lon_span / d_col) if d_col != 0 else (abs(lat_span) / max(abs(d_row), 3))

    def to_latlon(r: float, c: float):
        return float(rest_lat + (r - sr) * step_lat), float(rest_lon + (c - sc) * step_lon)

    return to_latlon


def build_route_map(order_meta: dict, sims: dict, primary_model: str, tile_style: str, show_hotspots: bool):
    first_sim = sims[primary_model]
    env = first_sim["env"]
    sr, sc = first_sim["start"]
    dr, dc = first_sim["dest"]

    rest_lat, rest_lon = order_meta["rest_lat"], order_meta["rest_lon"]
    cust_lat, cust_lon = order_meta["cust_lat"], order_meta["cust_lon"]
    to_latlon = grid_to_gps_mapper((sr, sc), (dr, dc), rest_lat, rest_lon, cust_lat, cust_lon)

    m = folium.Map(
        location=[(rest_lat + cust_lat) / 2.0, (rest_lon + cust_lon) / 2.0],
        zoom_start=13,
        tiles=None,
        control_scale=True,
    )
    if tile_style == "Standard (OpenStreetMap)":
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    elif tile_style == "Clean Street (Esri)":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri World Street",
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Light Gray",
        ).add_to(m)

    if show_hotspots:
        min_r, max_r = max(0, min(sr, dr) - 1), min(env.n - 1, max(sr, dr) + 1)
        min_c, max_c = max(0, min(sc, dc) - 1), min(env.n - 1, max(sc, dc) + 1)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if env.hotspots[r, c] and (r, c) != (sr, sc) and (r, c) != (dr, dc):
                    lat1, lon1 = to_latlon(r - 0.38, c - 0.38)
                    lat2, lon2 = to_latlon(r + 0.38, c + 0.38)
                    folium.Rectangle(
                        bounds=[[min(lat1, lat2), min(lon1, lon2)], [max(lat1, lat2), max(lon1, lon2)]],
                        color="#DC2626",
                        weight=1,
                        fill=True,
                        fill_color="#EF4444",
                        fill_opacity=0.14,
                        tooltip=f"Congested Zone ({r}, {c})",
                    ).add_to(m)

    # Draw baseline / secondary routes first, then primary route on top
    ordered_models = [k for k in sims.keys() if k != primary_model] + [primary_model]
    for idx, model_name in enumerate(ordered_models):
        sim = sims[model_name]
        is_primary = model_name == primary_model
        style = ROUTE_STYLES.get(model_name, {"color": "#2563EB", "weight": 4, "dash": None})
        color = "#2563EB" if is_primary else style["color"]
        weight = 4.5 if is_primary else 3.0
        dash = None if is_primary else style["dash"]

        jitter = 0.0 if is_primary else ((idx + 1) * 0.00014)
        coords = []
        for step_i, (r, c) in enumerate(sim["path"]):
            lat, lon = to_latlon(r, c)
            if 0 < step_i < len(sim["path"]) - 1:
                lat += jitter
                lon -= jitter
            coords.append([lat, lon])

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=0.92 if is_primary else 0.75,
            dash_array=dash,
            tooltip=f"{model_name}: {sim['minutes']} min ({sim['steps']} steps)",
        ).add_to(m)

    folium.Marker(
        location=[rest_lat, rest_lon],
        tooltip=f"Pickup: Restaurant ({order_meta['city']})",
        icon=folium.Icon(color="green", icon="cutlery", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=[cust_lat, cust_lon],
        tooltip=f"Destination: Customer (Target: {order_meta['sla_min']:.0f} min)",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    all_lats = [rest_lat, cust_lat]
    all_lons = [rest_lon, cust_lon]
    for sim in sims.values():
        for r, c in sim["path"]:
            lat, lon = to_latlon(r, c)
            all_lats.append(lat)
            all_lons.append(lon)

    pad_lat = max((max(all_lats) - min(all_lats)) * 0.18, 0.004)
    pad_lon = max((max(all_lons) - min(all_lons)) * 0.18, 0.004)
    m.fit_bounds([
        [min(all_lats) - pad_lat, min(all_lons) - pad_lon],
        [max(all_lats) + pad_lat, max(all_lons) + pad_lon],
    ])
    return m


def draw_grid_diagnostic(sim_data, title: str):
    env = sim_data["env"]
    n = env.n
    fig, ax = plt.subplots(figsize=(3.8, 3.8), dpi=110)

    for r in range(n):
        for c in range(n):
            is_hot = env.hotspots[r, c]
            face = "#FEE2E2" if is_hot else "#F8FAFC"
            edge = "#FCA5A5" if is_hot else "#E2E8F0"
            ax.add_patch(
                patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.8)
            )
            if is_hot:
                ax.text(c, r, "×", ha="center", va="center", color="#B91C1C", fontsize=9, alpha=0.65)

    path = sim_data["path"]
    if len(path) > 1:
        rows = [p[0] for p in path]
        cols = [p[1] for p in path]
        color = "#2563EB" if sim_data["delivered"] else "#DC2626"
        ax.plot(cols, rows, color=color, linewidth=2.2, marker="o", markersize=3.5, alpha=0.9)

    sr, sc = sim_data["start"]
    dr, dc = sim_data["dest"]
    ax.scatter([sc], [sr], c="#15803D", s=140, marker="s", zorder=5)
    ax.scatter([dc], [dr], c="#1D4ED8", s=150, marker="D", zorder=6)
    ax.text(sc, sr, "R", ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=7)
    ax.text(dc, dr, "C", ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=7)

    ax.set_title(f"{title}\n{sim_data['minutes']} min | {sim_data['steps']} steps", fontsize=8.5, fontweight="bold")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


# ==============================================================================
# INITIALIZE DATA & MODELS
# ==============================================================================
try:
    train_orders, test_orders, catalog_df, all_policies = load_system_resources()
except Exception as exc:
    st.error(f"Failed to load dataset or model checkpoints: {exc}")
    st.stop()

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.subheader("Order Selection")
    source_mode = st.radio(
        "Order Source",
        options=["Dataset Order", "Custom Scenario"],
        index=0,
    )

    if source_mode == "Dataset Order":
        cities = ["All Cities"] + sorted(catalog_df["city"].unique().tolist())
        selected_city = st.selectbox("City", options=cities, index=0)

        traffic_opts = ["All", "Jam", "High", "Medium", "Low"]
        selected_traffic = st.selectbox("Traffic Condition", options=traffic_opts, index=0)

        filtered_cat = catalog_df.copy()
        if selected_city != "All Cities":
            filtered_cat = filtered_cat[filtered_cat["city"] == selected_city]
        if selected_traffic != "All":
            filtered_cat = filtered_cat[filtered_cat["traffic_label"] == selected_traffic]
        if filtered_cat.empty:
            filtered_cat = catalog_df.copy()

        order_options = filtered_cat["seed"].tolist()
        order_labels = {
            row["seed"]: f"{row['order_id']} ({row['city']}, {row['distance_km']} km)"
            for _, row in filtered_cat.iterrows()
        }
        default_idx = order_options.index(7) if 7 in order_options else 0

        selected_seed = st.selectbox(
            "Order ID",
            options=order_options,
            index=default_idx,
            format_func=lambda s: order_labels[s],
        )
        custom_cfg = {"use_custom": False, "hotspot_frac": 0.15}
    else:
        selected_seed = 7
        t_level = st.selectbox("Traffic Level", [0, 1, 2, 3], index=3, format_func=lambda x: TRAFFIC_LABELS[x])
        w_level = st.selectbox("Weather", [0, 1, 2, 3], index=0, format_func=lambda x: WEATHER_LABELS[x])
        h_frac = st.slider("Congestion Density", 0.05, 0.35, 0.15, 0.05)
        c1, c2 = st.columns(2)
        sr = c1.number_input("Start Row", 0, 9, 2)
        sc = c2.number_input("Start Col", 0, 9, 1)
        dr = c1.number_input("Dest Row", 0, 9, 8)
        dc = c2.number_input("Dest Col", 0, 9, 8)
        custom_cfg = {
            "use_custom": True,
            "traffic": int(t_level),
            "weather": int(w_level),
            "hotspot_frac": float(h_frac),
            "start": (int(sr), int(sc)),
            "dest": (int(dr), int(dc)),
        }

    st.divider()
    st.subheader("Model Settings")
    policy_names = list(all_policies.keys())
    primary_model = st.selectbox("Active Route Model", options=policy_names, index=0)
    show_baseline = st.checkbox("Compare with Direct Route (Baseline)", value=True)
    extra_models = st.multiselect(
        "Additional Models to Compare",
        options=[p for p in policy_names if p not in (primary_model, "Direct Route (Baseline)")],
        default=[],
    )

# ==============================================================================
# MAIN PAGE HEADER & TABS
# ==============================================================================
st.title("Food Delivery Route Dispatch & Analysis")

tab_map, tab_bench, tab_tuning, tab_data = st.tabs([
    "Route Dispatch",
    "Algorithm Benchmark",
    "Model Tuning & Validation",
    "Dataset & Error Analysis",
])

# ------------------------------------------------------------------------------
# TAB 1: ROUTE DISPATCH
# ------------------------------------------------------------------------------
with tab_map:
    order_meta = catalog_df.loc[catalog_df["seed"] == int(selected_seed)].iloc[0].to_dict()

    active_models = [primary_model]
    if show_baseline and "Direct Route (Baseline)" in all_policies and primary_model != "Direct Route (Baseline)":
        active_models.append("Direct Route (Baseline)")
    for m_name in extra_models:
        if m_name not in active_models:
            active_models.append(m_name)

    sims = {
        name: simulate_single_episode(all_policies[name], test_orders, int(selected_seed), custom_cfg)
        for name in active_models
    }
    baseline_sim = (
        sims["Direct Route (Baseline)"]
        if "Direct Route (Baseline)" in sims
        else simulate_single_episode(greedy_policy, test_orders, int(selected_seed), custom_cfg)
    )
    primary_sim = sims[primary_model]
    env_ref = primary_sim["env"]

    time_saved = round(baseline_sim["minutes"] - primary_sim["minutes"], 1)
    sla_diff = round(primary_sim["minutes"] - primary_sim["benchmark_min"], 1)
    zones_avoided = max(0, baseline_sim["hotspot_hits"] - primary_sim["hotspot_hits"])

    # 3 Clean Native KPI Cards
    k1, k2, k3 = st.columns(3)
    k1.metric(
        label="Estimated Duration",
        value=f"{primary_sim['minutes']} min",
        delta=f"{sla_diff:+.1f} min vs. {primary_sim['benchmark_min']:.0f} min target",
        delta_color="inverse",
        border=True,
    )
    k2.metric(
        label="Time Saved vs. Direct Route",
        value=f"{time_saved:+.1f} min" if time_saved != 0 else "0.0 min",
        delta=f"{zones_avoided} congestion zones avoided",
        delta_color="normal",
        border=True,
    )
    k3.metric(
        label="Route Distance",
        value=f"{order_meta['distance_km']} km",
        delta=f"{TRAFFIC_LABELS[env_ref.traffic]} traffic · {WEATHER_LABELS[env_ref.weather]}",
        delta_color="off",
        border=True,
    )

    col_map, col_summary = st.columns([2.3, 1.0])

    with col_map:
        ctrl_1, ctrl_2, _ = st.columns([1.2, 1.2, 1.6])
        tile_style = ctrl_1.selectbox(
            "Basemap Style",
            options=["Standard (OpenStreetMap)", "Clean Street (Esri)", "Minimal Light (Esri)"],
            index=0,
            label_visibility="collapsed",
        )
        show_hotspots = ctrl_2.checkbox("Show Congestion Zones", value=True)

        fmap = build_route_map(order_meta, sims, primary_model, tile_style, show_hotspots)
        components.html(fmap._repr_html_(), height=470)
        st.caption(
            "Solid blue line: Active optimized route · Dashed gray line: Direct baseline route · "
            "Green marker: Restaurant pickup · Blue marker: Customer destination."
        )

    with col_summary:
        with st.container(border=True):
            st.markdown("##### Order Summary")
            summary_items = [
                ("Order ID", order_meta["order_id"]),
                ("City", f"{order_meta['city']} ({order_meta['area_type']})"),
                ("Order Type", order_meta["order_type"]),
                ("Vehicle", order_meta["vehicle_type"]),
                ("Traffic", TRAFFIC_LABELS[env_ref.traffic]),
                ("Weather", WEATHER_LABELS[env_ref.weather]),
                ("Target Duration", f"{primary_sim['benchmark_min']:.0f} min"),
                ("Pickup Coords", f"{order_meta['rest_lat']:.4f}, {order_meta['rest_lon']:.4f}"),
                ("Destination Coords", f"{order_meta['cust_lat']:.4f}, {order_meta['cust_lon']:.4f}"),
            ]
            html_rows = "".join(
                f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v}</span></div>'
                for k, v in summary_items
            )
            st.markdown(html_rows, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("##### Route Comparison")
            comp_rows = []
            for m_name, sim in sims.items():
                comp_rows.append({
                    "Model": m_name,
                    "Time (min)": sim["minutes"],
                    "Congestion Hits": sim["hotspot_hits"],
                    "Status": "On Time" if sim["on_time"] else ("Delayed" if sim["delivered"] else "Timeout"),
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    with st.expander("Model Diagnostics (Grid State & Step Log)", expanded=False):
        d_col1, d_col2 = st.columns([2.0, 1.2])
        with d_col1:
            g_cols = st.columns(min(len(sims), 3))
            for idx, (m_name, sim) in enumerate(sims.items()):
                with g_cols[idx % len(g_cols)]:
                    fig = draw_grid_diagnostic(sim, m_name)
                    st.pyplot(fig)
                    plt.close(fig)
        with d_col2:
            st.markdown("**Step-by-Step Actions**")
            st.caption(f"Model: {primary_model} ({primary_sim['steps']} steps, Reward: {primary_sim['reward']})")
            st.code(" -> ".join(primary_sim["actions"]) or "Reached destination", language="text")

# ------------------------------------------------------------------------------
# TAB 2: ALGORITHM BENCHMARK
# ------------------------------------------------------------------------------
with tab_bench:
    st.subheader("Algorithm Benchmark (500 Test Orders)")
    st.caption("Evaluated across 500 held-out Zomato delivery orders (`seed = 0..499`).")

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
        st.image(str(master_img), use_container_width=True)

    st.divider()
    st.subheader("Training Curves (Part A)")
    c_ql, c_dqn, c_ddqn = st.columns(3)
    if (QL_RESULT_DIR / "ql_learning_curves.png").exists():
        c_ql.image(str(QL_RESULT_DIR / "ql_learning_curves.png"), caption="Tabular Q-Learning", use_container_width=True)
    if (DQN_RESULT_DIR / "dqn_learning_curve_v2.png").exists():
        c_dqn.image(str(DQN_RESULT_DIR / "dqn_learning_curve_v2.png"), caption="Deep Q-Network (DQN v2)", use_container_width=True)
    if (DDQN_RESULT_DIR / "ddqn_learning_curves.png").exists():
        c_ddqn.image(str(DDQN_RESULT_DIR / "ddqn_learning_curves.png"), caption="Double DQN vs. Dueling DDQN", use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: MODEL TUNING & VALIDATION (PART B)
# ------------------------------------------------------------------------------
with tab_tuning:
    st.subheader("Hyperparameter Tuning, Regularization & Stress Testing")

    if (DQN_RESULT_DIR / "part_b_analysis.png").exists():
        st.image(str(DQN_RESULT_DIR / "part_b_analysis.png"), use_container_width=True)

    col1, col2 = st.columns(2)
    if (QL_RESULT_DIR / "ql_part_b_tuning.csv").exists():
        with col1:
            st.markdown("##### Q-Learning Parameter Tuning")
            st.dataframe(pd.read_csv(QL_RESULT_DIR / "ql_part_b_tuning.csv"), use_container_width=True, hide_index=True)

    if (DDQN_RESULT_DIR / "ddqn_part_b_tuning.csv").exists():
        with col2:
            st.markdown("##### Dueling DDQN Regularization & Tuning")
            st.dataframe(pd.read_csv(DDQN_RESULT_DIR / "ddqn_part_b_tuning.csv"), use_container_width=True, hide_index=True)

    if (MASTER_RESULT_DIR / "missing_data_comparison.csv").exists():
        st.divider()
        st.markdown("##### Missing-Data Imputation Comparison")
        m_c1, m_c2 = st.columns([1.3, 1])
        m_c1.dataframe(pd.read_csv(MASTER_RESULT_DIR / "missing_data_comparison.csv"), use_container_width=True, hide_index=True)
        if (MASTER_RESULT_DIR / "missing_data_comparison.png").exists():
            m_c2.image(str(MASTER_RESULT_DIR / "missing_data_comparison.png"), use_container_width=True)

    if (DQN_RESULT_DIR / "part_b_robustness_results.csv").exists():
        st.divider()
        st.markdown("##### Scenario Stress Testing")
        st.dataframe(pd.read_csv(DQN_RESULT_DIR / "part_b_robustness_results.csv"), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 4: DATASET & ERROR ANALYSIS
# ------------------------------------------------------------------------------
with tab_data:
    st.subheader("Error & Congestion Analysis")
    if (ERR_RESULT_DIR / "error_analysis.png").exists():
        st.image(str(ERR_RESULT_DIR / "error_analysis.png"), use_container_width=True)
    if (ERR_RESULT_DIR / "hotspot_analysis.csv").exists():
        st.markdown("##### Mean Congestion Steps by Traffic Level")
        st.dataframe(pd.read_csv(ERR_RESULT_DIR / "hotspot_analysis.csv"), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Dataset Cleaning & Exploration")
    if (BASE_DIR / "data" / "cleaning_log.csv").exists():
        st.dataframe(pd.read_csv(BASE_DIR / "data" / "cleaning_log.csv"), use_container_width=True, hide_index=True)
    if EDA_DIR.exists():
        png_files = sorted(EDA_DIR.glob("*.png"))
        cols = st.columns(3)
        for idx, p in enumerate(png_files):
            cols[idx % 3].image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)
