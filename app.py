"""
Food Delivery Route Dispatch & Analysis

Streamlit dashboard for evaluating reinforcement learning route optimization models
(Tabular Q-Learning, Deep Q-Network, Dueling Double DQN, and Ensemble) on
real-world Zomato delivery orders with multi-order VRP batching and 45-minute SLA tracking.

Run from project root:
    streamlit run app.py
"""
from itertools import permutations
from pathlib import Path
import json
import re
import time
import urllib.request
import altair as alt
import folium
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from env import DeliveryEnv, TRAFFIC_MAP, WEATHER_MAP, haversine_km, load_orders
from shared import split_orders
from evaluate import (
    greedy_policy,
    make_cross_algorithm_ensemble,
    planner_policy,
    random_policy,
)

try:
    import torch
    from dqn.dqn_v1 import QNetwork as QNetV1, dqn_policy as dqn_policy_v1
    from dqn.dqn_v2 import QNetwork as QNetV2, dqn_policy as dqn_policy_v2
    from dqn.dqn_v3 import OBS_DIM_V3, dqn_policy_v3
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    QNetV1 = dqn_policy_v1 = QNetV2 = dqn_policy_v2 = OBS_DIM_V3 = dqn_policy_v3 = None
    TORCH_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
QL_RESULT_DIR = BASE_DIR / "q_learning" / "result"
DQN_RESULT_DIR = BASE_DIR / "dqn" / "result"
DDQN_RESULT_DIR = BASE_DIR / "ddqn" / "result"
MASTER_RESULT_DIR = BASE_DIR / "result"
ERR_RESULT_DIR = MASTER_RESULT_DIR / "error_analysis"
EDA_DIR = BASE_DIR / "exploration" / "zomato_exploration"

SLA_LIMIT_MIN = 45.0  # Maximum allowed minutes from order pickup to customer drop-off

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
    "Dueling Double DQN (DDQN)": {"color": "#2563EB", "weight": 4, "dash": None},
    "Deep Q-Network (DQN v2)": {"color": "#0284C7", "weight": 4, "dash": None},
    "Deep Q-Network (DQN v3)": {"color": "#0D9488", "weight": 4, "dash": None},
    "Tabular Q-Learning": {"color": "#4F46E5", "weight": 4, "dash": None},
    "Deep Q-Network (DQN v1)": {"color": "#7C3AED", "weight": 3, "dash": "4, 4"},
    "Ensemble (QL + DQN + DDQN)": {"color": "#0F766E", "weight": 4, "dash": None},
}

MODEL_INFO = {
    "Dueling Double DQN (DDQN)": ("Dueling Double DQN", "Best all-rounder. Learns to steer around traffic hotspots."),
    "Deep Q-Network (DQN v2)": ("Deep Q-Network v2", "Neural network that learned direction and distance."),
    "Deep Q-Network (DQN v3)": ("Deep Q-Network v3", "DQN v2 plus a short-range traffic radar."),
    "Tabular Q-Learning": ("Q-Learning", "A fast lookup table learned from thousands of trips."),
    "Deep Q-Network (DQN v1)": ("Deep Q-Network v1", "Early version using raw map coordinates."),
    "Ensemble (QL + DQN + DDQN)": ("Ensemble (all models)", "Combines the votes of all three AI model families."),
    "Direct Route (Baseline)": ("Direct route", "Heads straight for the customer and ignores traffic."),
    "Shortest-Time Planner (Dijkstra)": ("Shortest-path planner", "Classic map search that sees all traffic in advance."),
    "Random Baseline": ("Random moves", "Moves randomly. Used as a sanity check."),
}


def model_label(name: str) -> str:
    return MODEL_INFO.get(name, (name, ""))[0]


def model_desc(name: str) -> str:
    return MODEL_INFO.get(name, ("", ""))[1]


st.set_page_config(
    page_title="Food Delivery Route Planner",
    page_icon="🛵",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #111827;
        --ink-2: #374151;
        --muted: #6B7280;
        --line: #E5E7EB;
        --line-soft: #F1F2F5;
        --surface: #FFFFFF;
        --canvas: #F6F7FB;
        --brand: #EA580C;
        --brand-soft: #FFF1E7;
        --route: #2563EB;
        --good: #15803D;
        --good-soft: #DCFCE7;
        --bad: #B91C1C;
        --bad-soft: #FEE2E2;
        --warn-soft: #FEF3C7;
    }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    /* Compact option buttons so 3-4 choices fit on one line in the sidebar */
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
        padding-left: 9px !important; padding-right: 9px !important; min-width: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButtonGroup"] button p { font-size: 0.85rem !important; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* ---------- Sidebar ---------- */
    .brand { display:flex; align-items:center; gap:10px; margin-bottom: 0.4rem; }
    .brand-logo {
        width: 36px; height: 36px; border-radius: 10px; background: var(--brand);
        display:flex; align-items:center; justify-content:center; font-size: 19px;
    }
    .brand-name { font-weight: 800; font-size: 1.02rem; color: var(--ink); line-height: 1.1; }
    .brand-sub { font-size: 0.75rem; color: var(--muted); }
    .step-head {
        display:flex; align-items:center; gap:9px; margin: 1.1rem 0 0.35rem;
        font-weight: 700; font-size: 0.95rem; color: var(--ink);
    }
    .step-num {
        width: 22px; height: 22px; border-radius: 50%; background: var(--brand-soft); color: var(--brand);
        display:inline-flex; align-items:center; justify-content:center; font-size: 0.75rem; font-weight: 800;
    }
    .model-hint { font-size: 0.8rem; color: var(--muted); margin-top: -0.45rem; margin-bottom: 0.4rem; }

    /* ---------- Hero ---------- */
    .hero h1 {
        font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em; color: var(--ink);
        margin: 0; padding: 0; line-height: 1.15;
    }
    .hero p { color: var(--muted); font-size: 0.98rem; margin: 0.35rem 0 0; max-width: 760px; }
    .eyebrow {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
        color: var(--brand); margin-bottom: 0.3rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--surface); padding: 5px; border-radius: 12px;
        border: 1px solid var(--line); width: fit-content; max-width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px; padding: 0 18px; border-radius: 8px; background: transparent;
    }
    .stTabs [data-baseweb="tab"] p { font-weight: 600; font-size: 0.92rem; }
    .stTabs [aria-selected="true"] { background: var(--brand-soft); }
    .stTabs [aria-selected="true"] p { color: var(--brand); }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Chips ---------- */
    .chips { display:flex; flex-wrap:wrap; gap:8px; margin: 0.2rem 0 1rem; }
    .chip {
        display:inline-flex; align-items:center; gap:6px; background: var(--surface);
        border: 1px solid var(--line); border-radius: 999px; padding: 5px 12px;
        font-size: 0.83rem; color: var(--ink-2); font-weight: 500;
    }

    /* ---------- KPI cards ---------- */
    .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 1.1rem; }
    .kpi {
        background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
        padding: 16px 18px; box-shadow: 0 1px 2px rgba(17,24,39,0.04);
    }
    .kpi-label { font-size: 0.82rem; color: var(--muted); font-weight: 600; display:flex; align-items:center; gap:7px; }
    .kpi-value { font-size: 1.75rem; font-weight: 800; color: var(--ink); margin-top: 6px; letter-spacing: -0.01em; }
    .kpi-value.text { font-size: 1.2rem; line-height: 1.3; }
    .kpi-sub { font-size: 0.8rem; margin-top: 4px; color: var(--muted); }
    .kpi-sub.good { color: var(--good); font-weight: 600; }
    .kpi-sub.bad { color: var(--bad); font-weight: 600; }

    /* ---------- Cards / panels ---------- */
    .panel {
        background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
        padding: 18px 18px 8px;
    }
    .panel-title { font-weight: 700; font-size: 1rem; color: var(--ink); margin-bottom: 2px; }
    .panel-sub { font-size: 0.8rem; color: var(--muted); margin-bottom: 12px; }
    .note {
        display:flex; gap:10px; align-items:flex-start; border-radius: 12px; padding: 10px 14px;
        font-size: 0.84rem; line-height: 1.45; color: var(--ink-2); margin: 0 0 12px;
        background: #EFF6FF; border: 1px solid #BFDBFE;
    }
    .note.warm { background: var(--brand-soft); border-color: #FED7AA; }
    .note b { color: var(--ink); }
    .note-icon { font-size: 1rem; line-height: 1.3; }
    .real-sim { display:flex; flex-wrap:wrap; gap: 6px 16px; font-size: 0.8rem; color: var(--muted); margin: 8px 2px 0; }
    .real-sim .tag { font-weight: 700; padding: 1px 8px; border-radius: 999px; margin-right: 4px; font-size: 0.72rem; }
    .real-sim .tag.real { background: var(--good-soft); color: var(--good); }
    .real-sim .tag.sim { background: var(--warn-soft); color: #92400E; }
    .map-frame { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; background: var(--surface); }

    /* ---------- Trip timeline ---------- */
    .stops { position: relative; margin: 6px 0 10px; }
    .stop { display:flex; gap: 12px; position: relative; padding-bottom: 16px; }
    .stop:not(:last-child)::before {
        content: ""; position: absolute; left: 15px; top: 32px; bottom: 0; width: 2px; background: var(--line);
    }
    .stop-icon {
        flex: 0 0 32px; height: 32px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
        font-size: 15px; background: var(--line-soft); border: 1px solid var(--line);
    }
    .stop-icon.rider { background: #FFEDD5; border-color: #FED7AA; }
    .stop-icon.store { background: #DCFCE7; border-color: #BBF7D0; }
    .stop-icon.home { background: #DBEAFE; border-color: #BFDBFE; }
    .stop-body { flex: 1; min-width: 0; }
    .stop-row { display:flex; justify-content: space-between; align-items: baseline; gap: 8px; }
    .stop-title { font-weight: 650; font-size: 0.9rem; color: var(--ink); }
    .stop-time { font-weight: 700; font-size: 0.85rem; color: var(--ink-2); white-space: nowrap; font-variant-numeric: tabular-nums; }
    .stop-meta { font-size: 0.78rem; color: var(--muted); margin-top: 1px; overflow-wrap: anywhere; }
    .pill { display:inline-block; font-size: 0.72rem; font-weight: 700; padding: 1px 8px; border-radius: 999px; margin-left: 4px; }
    .pill.good { background: var(--good-soft); color: var(--good); }
    .pill.bad { background: var(--bad-soft); color: var(--bad); }

    .facts { border-top: 1px solid var(--line-soft); margin-top: 4px; padding-top: 8px; }
    .fact { display:flex; justify-content: space-between; padding: 6px 0; font-size: 0.84rem; border-bottom: 1px solid var(--line-soft); gap: 10px; }
    .fact:last-child { border-bottom: none; }
    .fact span:first-child { color: var(--muted); }
    .fact span:last-child { color: var(--ink); font-weight: 600; text-align: right; }

    .section-title { font-size: 1.15rem; font-weight: 750; color: var(--ink); margin: 1.4rem 0 0.15rem; }
    .section-sub { font-size: 0.88rem; color: var(--muted); margin-bottom: 0.8rem; }

    div[data-testid="stExpander"] details { background: var(--surface); border-radius: 12px; }
    .main iframe, section.stMain iframe { border-radius: 14px; border: 1px solid var(--line) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_kpi(icon: str, label: str, value: str, sub: str = "", tone: str = "", text: bool = False) -> str:
    return (
        f'<div class="kpi"><div class="kpi-label"><span>{icon}</span>{label}</div>'
        f'<div class="kpi-value{" text" if text else ""}">{value}</div><div class="kpi-sub {tone}">{sub}</div></div>'
    )


def bar_chart(df: pd.DataFrame, value_col: str, color: str, fmt: str, ascending: bool):
    order = df.sort_values(value_col, ascending=ascending)["Model"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12, ticks=False, domain=False)),
        x=alt.X(f"{value_col}:Q", title=None,
                scale=alt.Scale(domain=[0, float(df[value_col].max()) * 1.12 or 1], nice=False), axis=alt.Axis(grid=True, labelFontSize=11, domain=False, ticks=False)),
        tooltip=["Model", alt.Tooltip(f"{value_col}:Q", format=fmt)],
    )
    bars = base.mark_bar(color=color, cornerRadiusEnd=4, size=18)
    labels = base.mark_text(align="left", dx=4, fontSize=11, color="#374151").encode(
        text=alt.Text(f"{value_col}:Q", format=fmt)
    )
    return (bars + labels).properties(height=max(160, 34 * len(df))).configure_view(stroke=None)


def section(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True,
    )


def style_sla_status(val: str) -> str:
    """Pandas Styler callback to color-code SLA Status cells."""
    if str(val).startswith("Compliant"):
        return "color: #15803D; background-color: #DCFCE7; font-weight: 600;"
    if str(val).startswith("Breached") or str(val).startswith("Timeout"):
        return "color: #B91C1C; background-color: #FEE2E2; font-weight: 600;"
    return "color: #334155;"


def extract_city_name(delivery_person_id: str) -> str:
    m = re.match(r"^([A-Z]+?)RES", str(delivery_person_id).strip())
    if m:
        prefix = m.group(1)
        return CITY_PREFIX_MAP.get(prefix, prefix.title())
    return "Urban"


def parse_order_minutes(time_str: str) -> int:
    parts = str(time_str).strip().split(":")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]) * 60 + int(parts[1])
    return 19 * 60 + 30


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
    df["order_min"] = df["Time_Orderd"].apply(parse_order_minutes)
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
            "multi_count": max(1, int(row["multi"]) if int(row["multi"]) > 0 else 1),
            "raw_multi": int(row["multi"]),
            "order_type": str(row.get("Type_of_order", "Meal")).strip(),
            "vehicle_type": str(row.get("Type_of_vehicle", "motorcycle")).strip().title(),
            "rider_rating": float(row.get("Delivery_person_Ratings", 4.6)),
            "distance_km": round(float(row["distance_km"]), 2),
            "sla_min": float(row["time_taken"]),
            "rest_lat": float(row["Restaurant_latitude"]),
            "rest_lon": float(row["Restaurant_longitude"]),
            "cust_lat": float(row["Delivery_location_latitude"]),
            "cust_lon": float(row["Delivery_location_longitude"]),
            "time_ordered": str(row.get("Time_Orderd", "19:30")),
            "order_min": int(row.get("order_min", 1170)),
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
    ddqn_net = None
    if TORCH_AVAILABLE and torch is not None:
        p1 = DQN_RESULT_DIR / "dqn_model_v1.pt"
        if p1.exists() and QNetV1 is not None:
            m1 = QNetV1()
            m1.load_state_dict(torch.load(p1, map_location="cpu", weights_only=True))
            m1.eval()

        p2 = DQN_RESULT_DIR / "dqn_model_v2.pt"
        if p2.exists() and QNetV2 is not None:
            m2 = QNetV2()
            m2.load_state_dict(torch.load(p2, map_location="cpu", weights_only=True))
            m2.eval()

        p3 = DQN_RESULT_DIR / "dqn_model_v3.pt"
        if p3.exists() and QNetV2 is not None:
            m3 = QNetV2(obs_dim=OBS_DIM_V3)
            m3.load_state_dict(torch.load(p3, map_location="cpu", weights_only=True))
            m3.eval()

        ddqn_ckpt = DDQN_RESULT_DIR / "ddqn_model_best.pt"
        tuned_fallback = DQN_RESULT_DIR / "dqn_model_tuned.pt"
        if ddqn_ckpt.exists():
            try:
                from ddqn.ddqn_pipeline import load_best_ddqn, make_ddqn_policy
                ddqn_net = load_best_ddqn(ddqn_ckpt)
            except Exception:
                pass
        elif tuned_fallback.exists() and QNetV2 is not None:
            ckpt = torch.load(tuned_fallback, map_location="cpu", weights_only=False)
            ddqn_net = QNetV2(obs_dim=OBS_DIM_V3, hidden=ckpt.get("hidden", 128))
            ddqn_net.load_state_dict(ckpt["state_dict"])
            ddqn_net.eval()

    if ddqn_net is not None:
        try:
            from ddqn.ddqn_pipeline import make_ddqn_policy
            policies["Dueling Double DQN (DDQN)"] = make_ddqn_policy(ddqn_net)
        except Exception:
            policies["Dueling Double DQN (DDQN)"] = dqn_policy_v3(ddqn_net)
    if m2 is not None:
        policies["Deep Q-Network (DQN v2)"] = dqn_policy_v2(m2)
    if m3 is not None:
        policies["Deep Q-Network (DQN v3)"] = dqn_policy_v3(m3)
    if ql_agent is not None:
        from q_learning.q_learning_pipeline import make_ql_policy
        policies["Tabular Q-Learning"] = make_ql_policy(ql_agent)
    if m1 is not None:
        policies["Deep Q-Network (DQN v1)"] = dqn_policy_v1(m1)
    if m2 is not None and ddqn_net is not None:
        policies["Ensemble (QL + DQN + DDQN)"] = make_cross_algorithm_ensemble(
            ql_agent, m2, m3, ddqn_net
        )

    policies["Direct Route (Baseline)"] = greedy_policy
    policies["Shortest-Time Planner (Dijkstra)"] = planner_policy
    policies["Random Baseline"] = random_policy

    return train_orders, test_orders, test_meta, catalog_df, policies


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
    sla_compliant = info["delivered"] and (info["elapsed_min"] <= SLA_LIMIT_MIN)
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
        "sla_compliant": sla_compliant,
        "hotspot_hits": hotspot_hits,
    }


def build_real_order_batch(
    primary_order: dict,
    test_meta: pd.DataFrame,
    n_deliveries: int,
    diff_restaurants: bool,
    primary_sim: dict,
    policy_fn,
    test_orders: pd.DataFrame,
    include_pickup_leg: bool = True,
    radius_km: float = 2.5,
):
    """Group real Zomato orders from the same city within a ~2.5 km restaurant radius,
    optimize the drop-off sequence under the 45-minute SLA window, and compute total batch metrics."""
    letters = ["A", "B", "C"]
    batch_orders = [
        {
            "tag": "A",
            "order_id": primary_order["order_id"],
            "order_type": primary_order["order_type"],
            "rest_lat": float(primary_order["rest_lat"]),
            "rest_lon": float(primary_order["rest_lon"]),
            "cust_lat": float(primary_order["cust_lat"]),
            "cust_lon": float(primary_order["cust_lon"]),
            "sla_min": float(primary_order["sla_min"]),
            "leg_minutes": float(primary_sim["minutes"]),
        }
    ]

    if n_deliveries > 1:
        city_df = test_meta[
            (test_meta["city_name"] == primary_order["city"])
            & (test_meta["ID"] != primary_order["order_id"])
        ].copy()

        if not city_df.empty:
            city_df["rest_dist_km"] = haversine_km(
                primary_order["rest_lat"],
                primary_order["rest_lon"],
                city_df["Restaurant_latitude"],
                city_df["Restaurant_longitude"],
            )
            city_df["time_diff_min"] = (city_df["order_min"] - primary_order["order_min"]).abs()
            nearby = city_df[(city_df["rest_dist_km"] <= radius_km) & (city_df["rest_dist_km"] > 0.05)]
            if len(nearby) < (n_deliveries - 1):
                nearby = city_df[city_df["rest_dist_km"] > 0.05]
            nearby = nearby.sort_values(["rest_dist_km", "time_diff_min"]).head(n_deliveries - 1)
        else:
            nearby = pd.DataFrame()

        for idx in range(1, n_deliveries):
            tag = letters[idx]
            sub_sim = simulate_single_episode(policy_fn, test_orders, int(primary_order["seed"]) + idx * 19)
            if idx <= len(nearby):
                r_row = nearby.iloc[idx - 1]
                r_lat = float(r_row["Restaurant_latitude"]) if diff_restaurants else float(primary_order["rest_lat"])
                r_lon = float(r_row["Restaurant_longitude"]) if diff_restaurants else float(primary_order["rest_lon"])
                c_lat = float(r_row["Delivery_location_latitude"])
                c_lon = float(r_row["Delivery_location_longitude"])
                o_id = str(r_row["ID"])
                o_type = str(r_row["Type_of_order"]).strip()
                o_sla = float(r_row["time_taken"])
            else:
                d_lat = primary_order["cust_lat"] - primary_order["rest_lat"]
                d_lon = primary_order["cust_lon"] - primary_order["rest_lon"]
                r_lat = (
                    float(primary_order["rest_lat"] + d_lat * 0.18 * idx + 0.005)
                    if diff_restaurants
                    else float(primary_order["rest_lat"])
                )
                r_lon = (
                    float(primary_order["rest_lon"] + d_lon * 0.18 * idx - 0.005)
                    if diff_restaurants
                    else float(primary_order["rest_lon"])
                )
                c_lat = float(primary_order["cust_lat"] + d_lat * 0.25 * idx + 0.007)
                c_lon = float(primary_order["cust_lon"] + d_lon * 0.25 * idx - 0.007)
                o_id = f"{primary_order['order_id']}-{tag}"
                o_type = primary_order["order_type"]
                o_sla = float(primary_order["sla_min"])

            dist_to_cust_a = float(haversine_km(primary_order["cust_lat"], primary_order["cust_lon"], c_lat, c_lon))
            if dist_to_cust_a > 4.5:
                c_lat = float(primary_order["cust_lat"] + (c_lat - primary_order["cust_lat"]) * (2.5 / dist_to_cust_a))
                c_lon = float(primary_order["cust_lon"] + (c_lon - primary_order["cust_lon"]) * (2.5 / dist_to_cust_a))

            batch_orders.append({
                "tag": tag,
                "order_id": o_id,
                "order_type": o_type,
                "rest_lat": r_lat,
                "rest_lon": r_lon,
                "cust_lat": c_lat,
                "cust_lon": c_lon,
                "sla_min": o_sla,
                "leg_minutes": round(sub_sim["minutes"] * 0.42, 1),
            })

    # Compute pickup timestamps & pickup distance
    total_batch_km = float(primary_order["distance_km"])
    if include_pickup_leg:
        total_batch_km += round(float(primary_order["distance_km"]) * 0.28, 2)

    elapsed = 0.0
    pickup_times = {}
    for i, item in enumerate(batch_orders):
        if i > 0:
            if diff_restaurants:
                d_km = float(
                    haversine_km(
                        batch_orders[i - 1]["rest_lat"],
                        batch_orders[i - 1]["rest_lon"],
                        item["rest_lat"],
                        item["rest_lon"],
                    )
                )
                total_batch_km += d_km
                elapsed += round(max(2.0, d_km * 2.4), 1)
            else:
                elapsed += 1.0
        pickup_times[item["tag"]] = round(elapsed, 1)
        item["pickup_min"] = pickup_times[item["tag"]]

    last_pickup_lat = batch_orders[-1]["rest_lat"]
    last_pickup_lon = batch_orders[-1]["rest_lon"]

    # Optimize drop-off permutation to prevent >45m SLA breaches
    best_perm = tuple(range(len(batch_orders)))
    best_score = float("inf")
    for perm in permutations(range(len(batch_orders))):
        t_cur = elapsed
        cur_lat, cur_lon = last_pickup_lat, last_pickup_lon
        penalty = 0.0
        for step_k, idx in enumerate(perm):
            bo = batch_orders[idx]
            if step_k == 0 and idx == 0:
                leg_t = primary_sim["minutes"]
            else:
                d_km = float(haversine_km(cur_lat, cur_lon, bo["cust_lat"], bo["cust_lon"]))
                leg_t = max(3.5, round(d_km * 2.6, 1)) if step_k > 0 else primary_sim["minutes"]
            t_cur += leg_t
            wait_min = t_cur - pickup_times[bo["tag"]]
            if wait_min > SLA_LIMIT_MIN:
                penalty += (wait_min - SLA_LIMIT_MIN) * 50.0
            penalty += wait_min
            cur_lat, cur_lon = bo["cust_lat"], bo["cust_lon"]
        if penalty < best_score:
            best_score = penalty
            best_perm = perm

    t_cur = elapsed
    cur_lat, cur_lon = last_pickup_lat, last_pickup_lon
    dropoff_sequence = []
    sla_rows = []
    for step_k, idx in enumerate(best_perm):
        bo = batch_orders[idx]
        if step_k == 0:
            leg_t = primary_sim["minutes"]
        else:
            d_km = float(haversine_km(cur_lat, cur_lon, bo["cust_lat"], bo["cust_lon"]))
            total_batch_km += d_km
            leg_t = max(3.5, round(d_km * 2.5, 1))
        t_cur = round(t_cur + leg_t, 1)
        wait_from_pickup = round(t_cur - pickup_times[bo["tag"]], 1)
        is_compliant = wait_from_pickup <= SLA_LIMIT_MIN

        bo_copy = dict(bo)
        bo_copy["pickup_min"] = pickup_times[bo["tag"]]
        bo_copy["dropoff_min"] = t_cur
        bo_copy["wait_min"] = wait_from_pickup
        bo_copy["leg_min"] = leg_t
        bo_copy["sla_compliant"] = is_compliant
        dropoff_sequence.append(bo_copy)

        sla_rows.append({
            "Order ID": f"Order {bo['tag']} ({bo['order_id']})",
            "Pickup ETA": f"+{pickup_times[bo['tag']]:.1f}m",
            "Dropoff ETA": f"+{t_cur:.1f}m",
            "Transit Wait": f"{wait_from_pickup:.1f} min",
            "SLA Status": "Compliant" if is_compliant else "Breached",
        })
        cur_lat, cur_lon = bo["cust_lat"], bo["cust_lon"]

    batch_summary = {
        "total_km": round(total_batch_km, 2),
        "total_duration_min": round(t_cur, 1),
        "max_wait_min": max(d["wait_min"] for d in dropoff_sequence),
        "all_compliant": all(d["sla_compliant"] for d in dropoff_sequence),
    }
    return batch_orders, dropoff_sequence, pd.DataFrame(sla_rows), batch_summary


@st.cache_data(show_spinner=False)
def run_validation_benchmark(n_episodes: int = 100):
    _, test_orders, _, _, all_policies = load_system_resources()

    greedy_times = []
    for ep in range(n_episodes):
        sim_g = simulate_single_episode(greedy_policy, test_orders, ep)
        greedy_times.append(sim_g["minutes"])
    mean_greedy_time = float(np.mean(greedy_times))

    rows = []
    for name, policy_fn in all_policies.items():
        t0 = time.perf_counter()
        successes, sla_ok, zomato_sla_ok = 0, 0, 0
        min_list, rew_list, hot_hits = [], [], []
        for ep in range(n_episodes):
            s = simulate_single_episode(policy_fn, test_orders, ep)
            successes += int(s["delivered"])
            sla_ok += int(s["sla_compliant"])
            zomato_sla_ok += int(s["on_time"])
            min_list.append(s["minutes"])
            rew_list.append(s["reward"])
            hot_hits.append(s["hotspot_hits"])
        elapsed_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, n_episodes)

        mean_min = float(np.mean(min_list))
        time_saved_min = mean_greedy_time - mean_min
        dist_saved_pct = ((mean_greedy_time - mean_min) / max(1.0, mean_greedy_time)) * 100.0

        rows.append({
            "Algorithm": name,
            "Delivery Rate": f"{(successes / n_episodes) * 100:.1f}%",
            "SLA Adherence (<=45m)": f"{(sla_ok / n_episodes) * 100:.1f}%",
            "Zomato Target SLA": f"{(zomato_sla_ok / n_episodes) * 100:.1f}%",
            "Avg Duration (min)": round(mean_min, 2),
            "Time Saved vs. Direct (min)": f"{time_saved_min:+.2f}",
            "Distance/Cost Saved (%)": f"{dist_saved_pct:+.1f}%",
            "Heavy Traffic Hit": round(float(np.mean(hot_hits)), 2),
            "Compute Time (ms/order)": round(elapsed_ms, 2),
            "Avg RL Reward": round(float(np.mean(rew_list)), 2),
        })
    return pd.DataFrame(rows)


def grid_to_gps_mapper(start_rc, dest_rc, rest_lat, rest_lon, cust_lat, cust_lon):
    sr, sc = start_rc
    dr, dc = dest_rc

    z_G = complex(dc - sc, -(dr - sr))
    if abs(z_G) < 1e-6:
        z_G = complex(3.0, 0.0)
    z_gps = complex(cust_lon - rest_lon, cust_lat - rest_lat)
    scale_rot = z_gps / z_G

    def to_latlon(r: float, c: float):
        z_g = complex(c - sc, -(r - sr))
        offset = z_g * scale_rot
        return float(rest_lat + offset.imag), float(rest_lon + offset.real)

    return to_latlon


def extract_route_waypoints(path, to_latlon, rest_lat, rest_lon, cust_lat, cust_lon):
    if len(path) <= 2:
        return [[rest_lat, rest_lon], [cust_lat, cust_lon]]

    corners = []
    for i in range(1, len(path) - 1):
        r_prev, c_prev = path[i - 1]
        r_curr, c_curr = path[i]
        r_next, c_next = path[i + 1]
        if (r_curr - r_prev, c_curr - c_prev) != (r_next - r_curr, c_next - c_curr):
            corners.append(path[i])

    if len(corners) == 1:
        chosen = [corners[0]]
    elif len(corners) >= 2:
        chosen = [corners[0], corners[-1]]
    else:
        chosen = [path[len(path) // 2]]

    waypoints = [[rest_lat, rest_lon]]
    for r, c in chosen:
        lat, lon = to_latlon(r, c)
        waypoints.append([lat, lon])
    waypoints.append([cust_lat, cust_lon])
    return waypoints


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_osrm_road_geometry(waypoints_tuple):
    coord_str = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in waypoints_tuple)
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FoodDeliveryRouteApp/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("code") == "Ok" and data.get("routes"):
                geojson_coords = data["routes"][0]["geometry"]["coordinates"]
                return [[float(c[1]), float(c[0])] for c in geojson_coords]
    except Exception:
        pass
    return [list(pt) for pt in waypoints_tuple]


def make_pin_icon(symbol: str, label: str, bg_color: str, border_color: str) -> folium.DivIcon:
    """Create a crisp HTML badge marker with Store (🏪) or Home (🏠) icon and compact label."""
    html = f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: {bg_color};
        color: #FFFFFF;
        border: 2px solid {border_color};
        border-radius: 16px;
        padding: 3px 9px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 11.5px;
        font-weight: 700;
        white-space: nowrap;
        box-shadow: 0 2px 5px rgba(15,23,42,0.28);
        transform: translate(-50%, -50%);
    ">
        <span style="font-size: 13px;">{symbol}</span>
        <span>{label}</span>
    </div>
    """
    return folium.DivIcon(html=html, icon_size=(120, 28), icon_anchor=(60, 14))


def synthetic_rider_point(rest_lat, rest_lon, cust_lat, cust_lon):
    """Fallback point a little behind the restaurant (opposite the customer). Not checked against land."""
    d_lat = (cust_lat - rest_lat) * 0.32
    d_lon = (cust_lon - rest_lon) * 0.32
    return rest_lat - d_lat + (d_lon * 0.35), rest_lon - d_lon - (d_lat * 0.35)


def pick_rider_start(order_meta: dict, test_meta: pd.DataFrame, min_km: float = 0.8, max_km: float = 3.0):
    """Start the rider at a real past drop-off address 0.8-3 km from the restaurant in the same city
    (riders usually wait near their last delivery). Real addresses keep the pin on land instead of in a
    lake or river; among the candidates, pick the one nearest the old synthetic point so the choice is
    stable and the rider still approaches from the side away from the customer."""
    rest_lat, rest_lon = float(order_meta["rest_lat"]), float(order_meta["rest_lon"])
    target_lat, target_lon = synthetic_rider_point(
        rest_lat, rest_lon, float(order_meta["cust_lat"]), float(order_meta["cust_lon"])
    )
    city = test_meta[test_meta["city_name"] == order_meta["city"]]
    pts = city[["Delivery_location_latitude", "Delivery_location_longitude"]].drop_duplicates()
    if pts.empty:
        return None
    d_rest = haversine_km(rest_lat, rest_lon, pts.iloc[:, 0], pts.iloc[:, 1])
    cand = pts[(d_rest >= min_km) & (d_rest <= max_km)]
    if cand.empty:
        cand = pts[d_rest > 0.3]
    if cand.empty:
        return None
    d_target = haversine_km(target_lat, target_lon, cand.iloc[:, 0], cand.iloc[:, 1])
    best = cand.iloc[int(np.argmin(np.asarray(d_target)))]
    return float(best.iloc[0]), float(best.iloc[1])


def build_route_map(
    order_meta: dict,
    sims: dict,
    primary_model: str,
    tile_style: str,
    show_hotspots: bool,
    include_pickup_leg: bool = True,
    batch_orders: list = None,
    dropoff_sequence: list = None,
    diff_restaurants: bool = True,
    rider_start: tuple = None,
):
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

    # Distinct heavy traffic zone markers instead of generic floating circles
    if show_hotspots:
        min_r, max_r = max(0, min(sr, dr) - 1), min(env.n - 1, max(sr, dr) + 1)
        min_c, max_c = max(0, min(sc, dc) - 1), min(env.n - 1, max(sc, dc) + 1)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if env.hotspots[r, c] and (r, c) != (sr, sc) and (r, c) != (dr, dc):
                    h_lat, h_lon = to_latlon(r, c)
                    folium.Marker(
                        location=[h_lat, h_lon],
                        tooltip=f"Heavy Traffic Zone ({r}, {c}) — 2.0x Travel Delay",
                        icon=folium.DivIcon(
                            html="""
                            <div style="
                                width: 22px; height: 22px;
                                background: rgba(245, 158, 11, 0.92);
                                color: #FFFFFF;
                                border: 2px solid #B45309;
                                border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;
                                font-size: 11px; font-weight: 700;
                                box-shadow: 0 1px 4px rgba(0,0,0,0.25);
                                transform: translate(-50%, -50%);
                            ">!</div>
                            """,
                            icon_size=(22, 22),
                            icon_anchor=(11, 11),
                        ),
                    ).add_to(m)

    all_lats = [rest_lat, cust_lat]
    all_lons = [rest_lon, cust_lon]

    # Leg 0: Rider starting location -> Rest A
    if include_pickup_leg:
        if rider_start is not None:
            rider_lat, rider_lon = rider_start
        else:
            rider_lat, rider_lon = synthetic_rider_point(rest_lat, rest_lon, cust_lat, cust_lon)
        pickup_coords = fetch_osrm_road_geometry(((rider_lat, rider_lon), (rest_lat, rest_lon)))
        if len(pickup_coords) > 2:
            # OSRM's route starts on the nearest drivable road: put the rider pin there
            rider_lat, rider_lon = pickup_coords[0]
        for lat, lon in pickup_coords:
            all_lats.append(lat)
            all_lons.append(lon)

        folium.PolyLine(
            locations=pickup_coords,
            color="#16A34A",
            weight=3.8,
            opacity=0.88,
            dash_array="6, 6",
            tooltip=f"Rider Dispatch -> Pickup Rest A (Order #{order_meta['order_id']})",
        ).add_to(m)

        folium.Marker(
            location=[rider_lat, rider_lon],
            tooltip=f"Rider Origin | Dispatched for Order #{order_meta['order_id']} (t = 0.0 min)",
            icon=make_pin_icon("🛵", "Rider", "#EA580C", "#9A3412"),
        ).add_to(m)

    # Pickup sequence across batched restaurants (Rest A -> Rest B -> Rest C)
    last_rest_lat, last_rest_lon = rest_lat, rest_lon
    if batch_orders and len(batch_orders) > 1 and diff_restaurants:
        for b_idx in range(1, len(batch_orders)):
            bo = batch_orders[b_idx]
            r_lat, r_lon = bo["rest_lat"], bo["rest_lon"]
            p_coords = fetch_osrm_road_geometry(((last_rest_lat, last_rest_lon), (r_lat, r_lon)))
            for lat, lon in p_coords:
                all_lats.append(lat)
                all_lons.append(lon)

            folium.PolyLine(
                locations=p_coords,
                color="#16A34A",
                weight=3.8,
                opacity=0.88,
                dash_array="6, 6",
                tooltip=f"Pickup Leg: Rest {batch_orders[b_idx - 1]['tag']} -> Rest {bo['tag']} (ETA: +{bo['pickup_min']} min)",
            ).add_to(m)

            folium.Marker(
                location=[r_lat, r_lon],
                tooltip=(
                    f"Pickup Rest {bo['tag']} | Order #{bo['order_id']} ({bo['order_type']}) | "
                    f"Pickup ETA: +{bo['pickup_min']} min"
                ),
                icon=make_pin_icon("🏪", f"Pickup {bo['tag']}", "#15803D", "#14532D"),
            ).add_to(m)
            last_rest_lat, last_rest_lon = r_lat, r_lon

    first_drop = dropoff_sequence[0] if dropoff_sequence else {
        "cust_lat": cust_lat,
        "cust_lon": cust_lon,
        "tag": "A",
        "order_id": order_meta["order_id"],
        "dropoff_min": first_sim["minutes"],
    }
    first_cust_lat, first_cust_lon = first_drop["cust_lat"], first_drop["cust_lon"]

    ordered_models = [k for k in sims.keys() if k != primary_model] + [primary_model]
    for model_name in ordered_models:
        sim = sims[model_name]
        is_primary = model_name == primary_model
        style = ROUTE_STYLES.get(model_name, {"color": "#2563EB", "weight": 4, "dash": None})
        color = "#2563EB" if is_primary else style["color"]
        weight = 4.6 if is_primary else 3.2
        dash = None if is_primary else style["dash"]

        waypoints = extract_route_waypoints(
            sim["path"], to_latlon, last_rest_lat, last_rest_lon, first_cust_lat, first_cust_lon
        )
        road_coords = fetch_osrm_road_geometry(tuple(tuple(pt) for pt in waypoints))

        for lat, lon in road_coords:
            all_lats.append(lat)
            all_lons.append(lon)

        folium.PolyLine(
            locations=road_coords,
            color=color,
            weight=weight,
            opacity=0.92 if is_primary else 0.78,
            dash_array=dash,
            tooltip=f"{model_name} -> Cust {first_drop['tag']} (Order #{first_drop['order_id']}): {sim['minutes']} min",
        ).add_to(m)

    # Chained customer drop-offs (Dropoff Cust A -> Dropoff Cust B -> Dropoff Cust C)
    if dropoff_sequence and len(dropoff_sequence) > 1:
        prev_lat, prev_lon = first_cust_lat, first_cust_lon
        for d_idx in range(1, len(dropoff_sequence)):
            d_stop = dropoff_sequence[d_idx]
            s_lat, s_lon = d_stop["cust_lat"], d_stop["cust_lon"]
            leg_coords = fetch_osrm_road_geometry(((prev_lat, prev_lon), (s_lat, s_lon)))
            for lat, lon in leg_coords:
                all_lats.append(lat)
                all_lons.append(lon)

            folium.PolyLine(
                locations=leg_coords,
                color="#2563EB",
                weight=4.2,
                opacity=0.9,
                tooltip=(
                    f"Dropoff Leg: Cust {dropoff_sequence[d_idx - 1]['tag']} -> Cust {d_stop['tag']} "
                    f"(Order #{d_stop['order_id']}, ETA: +{d_stop['dropoff_min']}m)"
                ),
            ).add_to(m)
            prev_lat, prev_lon = s_lat, s_lon

    # Store Pickup Marker for Rest A
    rest_a_label = "Pickup A" if (batch_orders and len(batch_orders) > 1) else "Restaurant"
    folium.Marker(
        location=[rest_lat, rest_lon],
        tooltip=(
            f"Pickup Rest A | Order #{order_meta['order_id']} ({order_meta['order_type']}) | "
            f"City: {order_meta['city']} | Pickup ETA: +0.0 min"
        ),
        icon=make_pin_icon("🏪", rest_a_label, "#15803D", "#14532D"),
    ).add_to(m)

    # Customer Drop-off Home Markers with Order ID & ETA hover tooltips
    if dropoff_sequence:
        for d_stop in dropoff_sequence:
            sla_txt = "Compliant" if d_stop["sla_compliant"] else "Breached"
            bg_c = "#1D4ED8" if d_stop["sla_compliant"] else "#DC2626"
            bd_c = "#1E3A8A" if d_stop["sla_compliant"] else "#7F1D1D"
            folium.Marker(
                location=[d_stop["cust_lat"], d_stop["cust_lon"]],
                tooltip=(
                    f"Dropoff Cust {d_stop['tag']} | Order #{d_stop['order_id']} | "
                    f"ETA: +{d_stop['dropoff_min']} min (Wait: {d_stop['wait_min']}m — {sla_txt})"
                ),
                icon=make_pin_icon("🏠", f"Drop-off {d_stop['tag']} · {d_stop['dropoff_min']:.0f} min", bg_c, bd_c),
            ).add_to(m)

    pad_lat = max((max(all_lats) - min(all_lats)) * 0.16, 0.003)
    pad_lon = max((max(all_lons) - min(all_lons)) * 0.16, 0.003)
    fit_box = [
        [min(all_lats) - pad_lat, min(all_lons) - pad_lon],
        [max(all_lats) + pad_lat, max(all_lons) + pad_lon],
    ]
    m.fit_bounds(fit_box)

    # Mount visual legend as a native Leaflet control (bottom-right corner) + OSRM street snapper
    map_var = m.get_name()
    leaflet_legend_and_snap_js = f"""
    <script>
    window.addEventListener("load", function() {{
        var mapObj = window["{map_var}"];
        if (mapObj && typeof L !== "undefined") {{
            // Re-fit once the iframe has its real size (fitBounds at 0px size zooms far out).
            var fitBox = {json.dumps(fit_box)};
            var refit = function() {{ mapObj.invalidateSize(); mapObj.fitBounds(fitBox); }};
            refit();
            if (window.ResizeObserver) {{
                var lastW = 0;
                new ResizeObserver(function(entries) {{
                    var w = entries[0].contentRect.width;
                    if (w > 0 && Math.abs(w - lastW) > 40) {{ lastW = w; refit(); }}
                }}).observe(mapObj.getContainer());
            }}
            var legend = L.control({{position: "bottomright"}});
            legend.onAdd = function(map) {{
                var div = L.DomUtil.create("div", "info legend");
                div.style.background = "rgba(255, 255, 255, 0.96)";
                div.style.border = "1px solid #CBD5E1";
                div.style.borderRadius = "10px";
                div.style.padding = "8px 12px";
                div.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
                div.style.fontSize = "11.5px";
                div.style.color = "#0F172A";
                div.style.boxShadow = "0 2px 8px rgba(15, 23, 42, 0.16)";
                div.style.lineHeight = "1.6";
                div.innerHTML =
                    '<div style="font-weight:700;margin-bottom:3px;color:#1E293B;font-size:12px;">Legend</div>' +
                    '<div style="display:flex;align-items:center;gap:7px;">' +
                        '<span style="display:inline-block;width:18px;height:3.5px;background:#2563EB;border-radius:2px;"></span>' +
                        '<span>AI route</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:7px;">' +
                        '<span style="display:inline-block;width:18px;height:0;border-top:3px dashed #16A34A;"></span>' +
                        '<span>Ride to restaurant</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:7px;">' +
                        '<span style="display:inline-block;width:18px;height:0;border-top:2.5px dashed #64748B;"></span>' +
                        '<span>Direct route (for comparison)</span>' +
                    '</div>' +
                    '<div style="display:flex;align-items:center;gap:6px;margin-top:2px;">' +
                        '<span>🛵 Rider</span><span>&middot;</span>' +
                        '<span>🏪 Restaurant</span><span>&middot;</span>' +
                        '<span>🏠 Customer</span><span>&middot;</span>' +
                        '<span style="display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;background:#F59E0B;color:#fff;border-radius:50%;font-size:9px;font-weight:700;">!</span>' +
                        '<span>Traffic jam</span>' +
                    '</div>';
                return div;
            }};
            legend.addTo(mapObj);
        }}

        setTimeout(function() {{
            for (var key in window) {{
                if (key.indexOf("poly_line_") === 0 && window[key] && typeof window[key].getLatLngs === "function") {{
                    (function(poly) {{
                        var pts = poly.getLatLngs();
                        if (pts && pts.length >= 2 && pts.length <= 6) {{
                            var coordStr = pts.map(function(p) {{ return p.lng.toFixed(5) + "," + p.lat.toFixed(5); }}).join(";");
                            var url = "https://router.project-osrm.org/route/v1/driving/" + coordStr + "?overview=full&geometries=geojson";
                            fetch(url)
                                .then(function(r) {{ return r.json(); }})
                                .then(function(data) {{
                                    if (data && data.code === "Ok" && data.routes && data.routes.length > 0) {{
                                        var coords = data.routes[0].geometry.coordinates.map(function(c) {{ return [c[1], c[0]]; }});
                                        poly.setLatLngs(coords);
                                    }}
                                }})
                                .catch(function() {{}});
                        }}
                    }})(window[key]);
                }}
            }}
        }}, 120);
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(leaflet_legend_and_snap_js))
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
    with st.spinner("Loading delivery data and trained models..."):
        train_orders, test_orders, test_meta, catalog_df, all_policies = load_system_resources()
except Exception as exc:
    st.error(f"Failed to load dataset or model checkpoints: {exc}")
    st.stop()

WEATHER_ICONS = {0: "☀️", 1: "🌬️", 2: "🌫️", 3: "⛈️"}
TRAFFIC_ICONS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}

# Restaurant (start) and customer (dest) cells on the 10x10 grid for custom scenarios.
TRIP_PRESETS = {
    "Short": {"start": (4, 3), "dest": (6, 6), "desc": "A few blocks away (about 5 blocks)."},
    "Medium": {"start": (2, 1), "dest": (7, 6), "desc": "Across the neighbourhood (about 10 blocks)."},
    "Cross-town": {"start": (0, 0), "dest": (9, 9), "desc": "Corner to corner across the city (18 blocks)."},
}


def step_head(num: int, text: str) -> None:
    st.markdown(f'<div class="step-head"><span class="step-num">{num}</span>{text}</div>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: 3 SIMPLE STEPS (advanced options tucked into expanders)
# ==============================================================================
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">🛵</div><div>'
        '<div class="brand-name">Route Planner</div>'
        '<div class="brand-sub">AI routing for food delivery</div></div></div>',
        unsafe_allow_html=True,
    )

    # ---- Step 1: Order ----
    step_head(1, "Pick an order")
    use_custom = st.toggle(
        "Build my own scenario",
        value=False,
        help="Off: replay a real order from the Zomato dataset. On: set traffic and weather yourself.",
    )

    if not use_custom:
        city_list = sorted(catalog_df["city"].unique().tolist())
        selected_city = st.selectbox("City", options=["All cities"] + city_list, index=0)

        with st.expander("Filter by traffic"):
            selected_traffic = st.segmented_control(
                "Traffic at order time",
                options=["Low", "Medium", "High", "Jam"],
                selection_mode="single",
                default=None,
                help="Leave empty to show all orders.",
            )

        filtered_cat = catalog_df.copy()
        if selected_city != "All cities":
            filtered_cat = filtered_cat[filtered_cat["city"] == selected_city]
        if selected_traffic:
            filtered_cat = filtered_cat[filtered_cat["traffic_label"] == selected_traffic]
        if filtered_cat.empty:
            st.caption("No orders match those filters, so all orders are shown.")
            filtered_cat = catalog_df.copy()

        order_options = filtered_cat["seed"].tolist()
        order_labels = {
            row["seed"]: f"#{row['order_id']} · {row['city']} · {row['distance_km']} km"
            for _, row in filtered_cat.iterrows()
        }
        selected_seed = st.selectbox(
            f"Order ({len(order_options)} available)",
            options=order_options,
            index=order_options.index(7) if 7 in order_options else 0,
            format_func=lambda s: order_labels[s],
        )
        custom_cfg = {"use_custom": False, "hotspot_frac": 0.15}
    else:
        selected_seed = 7
        t_level = st.segmented_control(
            "Traffic", options=[0, 1, 2, 3], default=3, format_func=lambda x: TRAFFIC_LABELS[x]
        )
        w_level = st.segmented_control(
            "Weather", options=[0, 1, 2, 3], default=0, format_func=lambda x: WEATHER_LABELS[x]
        )
        h_pct = st.slider(
            "Traffic jams on the map",
            5, 35, 15, 5,
            format="%d%%",
            help="Share of city blocks with heavy traffic (travel there is 2x slower).",
        )
        h_frac = h_pct / 100.0
        trip_len = st.segmented_control(
            "Trip length",
            options=list(TRIP_PRESETS.keys()),
            default="Medium",
            help="How far the customer is from the restaurant on the AI's 10×10 city grid.",
        ) or "Medium"
        st.caption(TRIP_PRESETS[trip_len]["desc"])
        (sr, sc), (dr, dc) = TRIP_PRESETS[trip_len]["start"], TRIP_PRESETS[trip_len]["dest"]

        with st.expander("Advanced: exact grid cells"):
            st.caption(
                "The AI plans on a 10×10 grid of city blocks. Row 0 is the top edge and column 0 is the "
                "left edge. Start is the restaurant and destination is the customer."
            )
            use_exact = st.toggle("Use exact cells instead of trip length", value=False)
            if use_exact:
                c1, c2 = st.columns(2)
                sr = c1.number_input("Start row", 0, 9, int(sr))
                sc = c2.number_input("Start column", 0, 9, int(sc))
                dr = c1.number_input("Destination row", 0, 9, int(dr))
                dc = c2.number_input("Destination column", 0, 9, int(dc))
                if (sr, sc) == (dr, dc):
                    st.warning("Start and destination are the same cell, so there is nothing to route.")
        custom_cfg = {
            "use_custom": True,
            "traffic": int(t_level if t_level is not None else 3),
            "weather": int(w_level if w_level is not None else 0),
            "hotspot_frac": float(h_frac),
            "start": (int(sr), int(sc)),
            "dest": (int(dr), int(dc)),
        }

    # ---- Step 2: Trip ----
    step_head(2, "Set up the trip")
    n_choice = st.segmented_control(
        "Orders the rider carries",
        options=[1, 2, 3],
        default=2,
        format_func=lambda n: f"{n} order" if n == 1 else f"{n} orders",
        help="Batch nearby orders (restaurants within 2.5 km) into one trip.",
    )
    with st.expander("More trip options"):
        match_dataset = st.toggle(
            "Use the order's real batch size",
            value=False,
            help="Ignore the choice above and use the number of orders recorded in the dataset.",
        )
        diff_restaurants = st.toggle(
            "Pick up from different restaurants",
            value=True,
            help="On: each order comes from a different nearby restaurant. Off: all orders come from one restaurant.",
        )
        include_pickup_leg = st.toggle(
            "Show rider's trip to the restaurant",
            value=True,
            help="Draw the rider's ride from their starting point to the first restaurant.",
        )

    # ---- Step 3: Model ----
    step_head(3, "Choose the AI model")
    policy_names = list(all_policies.keys())
    primary_model = st.selectbox(
        "Routing model",
        options=policy_names,
        index=0,
        format_func=model_label,
        label_visibility="collapsed",
    )
    st.markdown(f'<div class="model-hint">{model_desc(primary_model)}</div>', unsafe_allow_html=True)
    show_baseline = st.toggle(
        "Compare with the direct route",
        value=True,
        help="Also draw the naive straight-to-customer route (grey dashed line) to see the time saved.",
    )
    with st.expander("Map & comparison options"):
        extra_models = st.multiselect(
            "Also draw these models",
            options=[p for p in policy_names if p not in (primary_model, "Direct Route (Baseline)")],
            default=[],
            format_func=model_label,
            placeholder="Choose models...",
        )
        tile_style = st.selectbox(
            "Map style",
            options=["Standard (OpenStreetMap)", "Clean Street (Esri)", "Minimal Light (Esri)"],
            index=2,
            format_func=lambda s: s.split(" (")[0],
        )
        show_hotspots = st.toggle("Show traffic jam zones", value=True)

    if not TORCH_AVAILABLE:
        st.caption("PyTorch is not installed, so only Q-Learning and the baseline models are available.")

# ==============================================================================
# HEADER & TABS
# ==============================================================================
st.markdown(
    '<div class="hero"><div class="eyebrow">Food delivery · Reinforcement learning</div>'
    "<h1>Delivery Route Planner</h1>"
    "<p>Pick an order in the sidebar. The AI plans the rider's route around traffic and checks that "
    f"every customer gets their food within {SLA_LIMIT_MIN:.0f} minutes of pickup.</p></div>",
    unsafe_allow_html=True,
)
st.write("")

tab_map, tab_compare, tab_research = st.tabs(["🗺️  Plan a route", "📊  Compare models", "🔬  Research details"])

# ------------------------------------------------------------------------------
# TAB 1: PLAN A ROUTE
# ------------------------------------------------------------------------------
with tab_map:
    order_meta = catalog_df.loc[catalog_df["seed"] == int(selected_seed)].iloc[0].to_dict()

    if match_dataset:
        n_deliveries = int(np.clip(order_meta.get("multi_count", 1), 1, 3))
    else:
        n_deliveries = int(n_choice) if n_choice else 1

    active_models = [primary_model]
    if show_baseline and "Direct Route (Baseline)" in all_policies and primary_model != "Direct Route (Baseline)":
        active_models.append("Direct Route (Baseline)")
    for m_name in extra_models:
        if m_name not in active_models:
            active_models.append(m_name)

    with st.spinner("Planning the route..."):
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

        batch_orders, dropoff_sequence, sla_df, batch_summary = build_real_order_batch(
            order_meta,
            test_meta,
            n_deliveries,
            diff_restaurants,
            primary_sim,
            all_policies[primary_model],
            test_orders,
            include_pickup_leg=include_pickup_leg,
        )

    time_saved = round(baseline_sim["minutes"] - primary_sim["minutes"], 1)
    zones_avoided = max(0, baseline_sim["hotspot_hits"] - primary_sim["hotspot_hits"])
    compliant_count = sum(1 for d in dropoff_sequence if d["sla_compliant"])
    traffic_lbl = TRAFFIC_LABELS[env_ref.traffic]
    weather_lbl = WEATHER_LABELS[env_ref.weather]

    # ---- Context chips ----
    place = "Custom scenario" if use_custom else f"{order_meta['city']} · {order_meta['area_type']}"
    chips = [
        f"📍 {place}",
        f"{TRAFFIC_ICONS[env_ref.traffic]} {traffic_lbl} traffic",
        f"{WEATHER_ICONS[env_ref.weather]} {weather_lbl}",
        f"🏍️ {order_meta['vehicle_type']}",
        f"🤖 {model_label(primary_model)}",
    ]
    st.markdown(
        '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
        unsafe_allow_html=True,
    )

    # ---- KPI cards ----
    grid_blocks = abs(primary_sim["start"][0] - primary_sim["dest"][0]) + abs(primary_sim["start"][1] - primary_sim["dest"][1])
    if primary_model == "Direct Route (Baseline)":
        saved_value, saved_sub, saved_tone = "—", "This is the direct route itself", ""
    elif time_saved > 0:
        saved_value = f"{time_saved:.1f} min"
        saved_sub = f"faster than the direct route · {zones_avoided} jam{'s' if zones_avoided != 1 else ''} avoided"
        saved_tone = "good"
    elif time_saved == 0:
        saved_value, saved_sub, saved_tone = "0 min", "Same time as the direct route", ""
    else:
        saved_value = f"{time_saved:.1f} min"
        saved_sub = "slower than the direct route"
        saved_tone = "bad"

    if batch_summary["all_compliant"]:
        ontime_value = f"{compliant_count}/{n_deliveries}"
        ontime_sub, ontime_tone = f"All delivered within {SLA_LIMIT_MIN:.0f} min", "good"
    else:
        ontime_value = f"{compliant_count}/{n_deliveries}"
        ontime_sub = f"Late — longest wait {batch_summary['max_wait_min']:.0f} min"
        ontime_tone = "bad"

    st.markdown(
        '<div class="kpi-grid">'
        + html_kpi("⏱️", "Total trip time", f"{batch_summary['total_duration_min']:.0f} min",
                   f"{n_deliveries} {'order' if n_deliveries == 1 else 'orders'} in this trip")
        + (
            html_kpi("📏", "Grid distance", f"{grid_blocks} blocks", "Shortest possible path on the AI's grid")
            if use_custom
            else html_kpi("📏", "Distance", f"{batch_summary['total_km']:.1f} km",
                          f"First delivery leg: {order_meta['distance_km']:.1f} km")
        )
        + html_kpi("⚡", "Time saved", saved_value, saved_sub, saved_tone)
        + html_kpi("✅" if batch_summary["all_compliant"] else "⚠️", "On time", ontime_value, ontime_sub, ontime_tone)
        + "</div>",
        unsafe_allow_html=True,
    )

    # ---- Map + trip plan ----
    col_map, col_plan = st.columns([2.15, 1], gap="medium")

    with col_map:
        if use_custom:
            st.markdown(
                '<div class="note warm"><span class="note-icon">🧪</span><div><b>Custom scenario.</b> '
                "The route, times and on-time result come from your traffic, weather and trip settings. "
                f"The street map borrows the locations of a sample real order (#{order_meta['order_id']}) "
                "only to illustrate the route, so the pins don't move when you change the trip length."
                "</div></div>",
                unsafe_allow_html=True,
            )
        fmap = build_route_map(
            order_meta,
            sims,
            primary_model,
            tile_style,
            show_hotspots,
            include_pickup_leg=include_pickup_leg,
            batch_orders=batch_orders,
            dropoff_sequence=dropoff_sequence,
            diff_restaurants=diff_restaurants,
            rider_start=pick_rider_start(order_meta, test_meta),
        )
        components.html(fmap.get_root().render(), height=540)
        st.markdown(
            '<div class="real-sim">'
            '<span><span class="tag real">Real</span>Streets, restaurant & customer locations, batched orders</span>'
            '<span><span class="tag sim">Simulated</span>Traffic jams, rider start point, trip times</span>'
            "<span>ℹ️ The AI plans on a 10×10 grid of city blocks, which is fitted onto the real roads to draw the route.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_plan:
        stops = []

        def add_stop(kind: str, icon: str, title: str, meta: str, t_min: float, pill: str = "") -> None:
            stops.append(
                f'<div class="stop"><div class="stop-icon {kind}">{icon}</div><div class="stop-body">'
                f'<div class="stop-row"><span class="stop-title">{title}{pill}</span>'
                f'<span class="stop-time">{t_min:.0f} min</span></div>'
                f'<div class="stop-meta">{meta}</div></div></div>'
            )

        if include_pickup_leg:
            add_stop("rider", "🛵", "Rider sets off", f"Starting in {order_meta['city']}", 0.0)

        if diff_restaurants or n_deliveries == 1:
            for bo in batch_orders:
                add_stop("store", "🏪", f"Pick up order {bo['tag']}",
                         f"#{bo['order_id']} · {bo['order_type']}", bo["pickup_min"])
        else:
            ids = ", ".join(b["tag"] for b in batch_orders)
            add_stop("store", "🏪", f"Pick up orders {ids}", "All from the same restaurant", 0.0)

        for d_stop in dropoff_sequence:
            pill = (
                '<span class="pill good">On time</span>'
                if d_stop["sla_compliant"]
                else '<span class="pill bad">Late</span>'
            )
            add_stop("home", "🏠", f"Deliver order {d_stop['tag']}",
                     f"Food in transit {d_stop['wait_min']:.0f} min", d_stop["dropoff_min"], pill)

        facts = [
            ("Order", f"#{order_meta['order_id']}"),
            ("Rider rating", f"⭐ {order_meta['rider_rating']:.1f}"),
            ("Delivery promise", f"≤ {SLA_LIMIT_MIN:.0f} min after pickup"),
        ]
        facts_html = "".join(f'<div class="fact"><span>{k}</span><span>{v}</span></div>' for k, v in facts)

        st.markdown(
            '<div class="panel"><div class="panel-title">Trip plan</div>'
            '<div class="panel-sub">Stops in order, with minutes since the trip started</div>'
            f'<div class="stops">{"".join(stops)}</div>'
            f'<div class="facts">{facts_html}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- Optional details ----
    st.write("")
    if len(sims) > 1:
        with st.expander(f"How do the {len(sims)} routes on the map compare?", expanded=False):
            comp_rows = []
            for m_name, sim in sims.items():
                status_label = (
                    "On time" if sim["sla_compliant"] else ("Late" if sim["delivered"] else "Not delivered")
                )
                comp_rows.append({
                    "Model": model_label(m_name),
                    "Time (min)": round(sim["minutes"], 1),
                    "Traffic jams hit": sim["hotspot_hits"],
                    "Result": status_label,
                })
            st.dataframe(
                pd.DataFrame(comp_rows).style.map(
                    lambda v: style_sla_status("Compliant" if v == "On time" else ("Breached" if v else "")),
                    subset=["Result"],
                ),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("How the AI sees the city (10×10 grid view)", expanded=False):
        st.caption(
            "The models learn on a simplified 10×10 grid of city blocks. Red cells are traffic jams, "
            "R is the restaurant and C is the customer."
        )
        g_cols = st.columns(min(len(sims), 3))
        for idx, (m_name, sim) in enumerate(sims.items()):
            with g_cols[idx % len(g_cols)]:
                fig = draw_grid_diagnostic(sim, model_label(m_name))
                st.pyplot(fig)
                plt.close(fig)
        st.caption(
            f"Moves taken by {model_label(primary_model)} ({primary_sim['steps']} steps, reward {primary_sim['reward']}):"
        )
        st.code(" → ".join(primary_sim["actions"]) or "Already at destination", language="text")

# ------------------------------------------------------------------------------
# TAB 2: COMPARE MODELS
# ------------------------------------------------------------------------------
with tab_compare:
    section(
        "How well does each model do?",
        "Every model routes the same set of real test orders it has never seen before. "
        "Higher on-time rate and lower trip time are better.",
    )

    n_val_eps = st.segmented_control(
        "Test orders",
        options=[100, 250, 500],
        default=100,
        format_func=lambda n: f"{n} orders",
        help="More orders give more reliable results but take longer to compute.",
    ) or 100

    with st.spinner(f"Testing every model on {n_val_eps} orders..."):
        val_df = run_validation_benchmark(int(n_val_eps))

    val_num = val_df.copy()
    val_num["On-time rate (%)"] = val_num["SLA Adherence (<=45m)"].str.rstrip("%").astype(float)
    val_num["Delivered (%)"] = val_num["Delivery Rate"].str.rstrip("%").astype(float)
    val_num["Model"] = val_num["Algorithm"].map(model_label)

    best_ontime = val_num.sort_values(["On-time rate (%)", "Avg Duration (min)"], ascending=[False, True]).iloc[0]
    delivered = val_num[val_num["Delivered (%)"] >= 99.0]
    fastest = (delivered if not delivered.empty else val_num).sort_values("Avg Duration (min)").iloc[0]
    quickest_cpu = val_num.sort_values("Compute Time (ms/order)").iloc[0]

    st.markdown(
        '<div class="kpi-grid">'
        + html_kpi("🏆", "Most on-time", best_ontime["Model"],
                   f"{best_ontime['On-time rate (%)']:.0f}% of orders within {SLA_LIMIT_MIN:.0f} min", "good", text=True)
        + html_kpi("⏱️", "Shortest trips", fastest["Model"],
                   f"{fastest['Avg Duration (min)']:.1f} min on average", text=True)
        + html_kpi("⚙️", "Fastest to compute", quickest_cpu["Model"],
                   f"{quickest_cpu['Compute Time (ms/order)']:.2f} ms per order", text=True)
        + "</div>",
        unsafe_allow_html=True,
    )

    ch1, ch2 = st.columns(2, gap="medium")
    with ch1:
        st.markdown("**On-time rate (%)**")
        st.altair_chart(bar_chart(val_num, "On-time rate (%)", "#EA580C", ".0f", False), use_container_width=True)
    with ch2:
        st.markdown("**Average trip time (min)**")
        st.altair_chart(bar_chart(val_num, "Avg Duration (min)", "#2563EB", ".1f", True), use_container_width=True)

    with st.expander("See the full results table"):
        st.dataframe(val_df.assign(Algorithm=val_df["Algorithm"].map(model_label)), use_container_width=True, hide_index=True)

    section("Offline benchmark & training", "Results saved by the training scripts (500 test orders).")
    master_csv = MASTER_RESULT_DIR / "master_all_models_results.csv"
    fallback_csv = DQN_RESULT_DIR / "results.csv"
    csv_to_show = master_csv if master_csv.exists() else fallback_csv
    master_img = MASTER_RESULT_DIR / "master_comparison_chart.png"
    curves = [
        (QL_RESULT_DIR / "ql_learning_curves.png", "Q-Learning"),
        (DQN_RESULT_DIR / "dqn_learning_curve_v2.png", "Deep Q-Network v2"),
        (DDQN_RESULT_DIR / "ddqn_learning_curves.png", "Double DQN vs. Dueling DDQN"),
    ]
    curves = [(p, c) for p, c in curves if p.exists()]

    if not csv_to_show.exists() and not master_img.exists() and not curves:
        st.info("No saved results yet. Run `python evaluate.py` and the training scripts to generate them.")

    if master_img.exists() or csv_to_show.exists():
        with st.expander("Benchmark chart and table", expanded=master_img.exists()):
            if master_img.exists():
                st.image(str(master_img), use_container_width=True)
            if csv_to_show.exists():
                df_m = pd.read_csv(csv_to_show)
                fmt = {
                    "success_rate": "{:.1%}", "avg_reward": "{:.2f}", "avg_steps": "{:.2f}",
                    "avg_delivery_time_min": "{:.2f}", "on_time_rate": "{:.1%}", "std_reward": "{:.2f}",
                    "route_efficiency": "{:.1%}", "avg_hotspot_steps": "{:.2f}",
                }
                fmt = {k: v for k, v in fmt.items() if k in df_m.columns}
                st.dataframe(df_m.style.format(fmt), use_container_width=True, hide_index=True)

    if curves:
        with st.expander("Training curves"):
            cols = st.columns(len(curves))
            for col, (p, cap) in zip(cols, curves):
                col.image(str(p), caption=cap, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: RESEARCH DETAILS
# ------------------------------------------------------------------------------
with tab_research:
    section(
        "Research details",
        "Tuning, robustness tests and data analysis behind the models. Open any section to explore.",
    )

    any_shown = False

    tuning_files = [
        (DQN_RESULT_DIR / "part_b_analysis.png", None),
        (QL_RESULT_DIR / "ql_part_b_tuning.csv", "Q-Learning settings"),
        (DDQN_RESULT_DIR / "ddqn_part_b_tuning.csv", "Dueling DDQN settings"),
    ]
    if any(p.exists() for p, _ in tuning_files):
        any_shown = True
        with st.expander("⚙️  Hyperparameter tuning & regularization", expanded=True):
            if tuning_files[0][0].exists():
                st.image(str(tuning_files[0][0]), use_container_width=True)
            for p, title in tuning_files[1:]:
                if p.exists():
                    st.markdown(f"**{title}**")
                    st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)

    md_csv = MASTER_RESULT_DIR / "missing_data_comparison.csv"
    md_png = MASTER_RESULT_DIR / "missing_data_comparison.png"
    if md_csv.exists():
        any_shown = True
        with st.expander("🧩  Handling missing data"):
            st.caption("Comparison of strategies for filling in missing values in the dataset.")
            m_c1, m_c2 = st.columns([1.3, 1])
            m_c1.dataframe(pd.read_csv(md_csv), use_container_width=True, hide_index=True)
            if md_png.exists():
                m_c2.image(str(md_png), use_container_width=True)

    rob_csv = DQN_RESULT_DIR / "part_b_robustness_results.csv"
    if rob_csv.exists():
        any_shown = True
        with st.expander("🌪️  Stress tests (extreme traffic & weather)"):
            st.dataframe(pd.read_csv(rob_csv), use_container_width=True, hide_index=True)

    err_png = ERR_RESULT_DIR / "error_analysis.png"
    hot_csv = ERR_RESULT_DIR / "hotspot_analysis.csv"
    if err_png.exists() or hot_csv.exists():
        any_shown = True
        with st.expander("🔍  Where the models go wrong"):
            if err_png.exists():
                st.image(str(err_png), use_container_width=True)
            if hot_csv.exists():
                st.markdown("**Average steps spent in traffic jams, by traffic level**")
                st.dataframe(pd.read_csv(hot_csv), use_container_width=True, hide_index=True)

    clean_log = BASE_DIR / "data" / "cleaning_log.csv"
    png_files = sorted(EDA_DIR.glob("*.png")) if EDA_DIR.exists() else []
    if clean_log.exists() or png_files:
        any_shown = True
        with st.expander("📦  The dataset (cleaning & exploration)"):
            if clean_log.exists():
                st.markdown("**Cleaning steps**")
                st.dataframe(pd.read_csv(clean_log), use_container_width=True, hide_index=True)
            if png_files:
                st.markdown("**Exploration charts**")
                cols = st.columns(3)
                for idx, p in enumerate(png_files):
                    cols[idx % 3].image(str(p), caption=p.stem.replace("_", " "), use_container_width=True)

    if not any_shown:
        st.info("No research outputs found yet. Run the training and analysis scripts to generate them.")
