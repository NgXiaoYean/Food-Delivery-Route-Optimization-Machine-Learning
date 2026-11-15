"""
Multi-order demo page: one rider, 1 to 3 customers, optional new order mid-trip.
Shows the route chosen by the trained Dueling Double DQN (DDQN) agents next to Greedy and the Planner.

Streamlit shows this file automatically in the sidebar next to app.py.
Run from the project root:  streamlit run app.py
Needs the saved models in ddqn/result_multi/  (made by:  python ddqn/ddqn_multi.py --stage C)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "ddqn"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

RESULT_DIR = BASE_DIR / "ddqn" / "result_multi"
TRAFFIC = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}
WEATHER = {0: "Clear/Cloudy", 1: "Windy", 2: "Fog/Sandstorm", 3: "Stormy"}
COLOURS = {"Greedy (nearest first)": "#9a9a9a", "Planner (oracle)": "#333333",
           "DDQN single model (seed 0)": "#7fb2e5", "DDQN ensemble (5 seeds)": "#d62728"}

st.set_page_config(page_title="Multi-order DDQN", layout="wide")
st.title("Multi-order delivery: Dueling Double DQN (DDQN)")
st.caption("One rider picks up 1 to 3 orders and must choose the delivery order and the route through a congested city grid. "
           "Optionally a new order appears during the trip.")


@st.cache_resource(show_spinner="Loading data and trained models ...")
def load_everything():
    import ddqn_multi as dm
    from ensemble_multi import make_ensemble_policy
    from env_multi import greedy_policy_multi, planner_policy_multi

    ctx = dm.make_context(False)
    paths = sorted(RESULT_DIR.glob("ddqn_multi_seed*.pt"))
    nets = [dm.load_model(p)[0] for p in paths]
    policies = {"Greedy (nearest first)": greedy_policy_multi, "Planner (oracle)": planner_policy_multi}
    if nets:
        policies["DDQN single model (seed 0)"] = dm.make_policy(nets[0])
    if len(nets) >= 2:
        policies["DDQN ensemble (5 seeds)" if len(nets) == 5 else f"DDQN ensemble ({len(nets)} seeds)"] = make_ensemble_policy(nets)
    scen = {"Normal trip": ctx["test_env"], "A new order arrives mid-trip": ctx["dynamic_env"],
            "Congestion map never seen in training": ctx["unseen_map_env"]}
    return scen, policies, len(nets)


def rollout(env, policy, seed):
    """Drive one trip and record the path and the delivery events."""
    env.reset(seed=seed)
    start, n0 = env.pos, len(env.dests)
    trail, events, new_order_step, total, hot = [env.pos], [], None, 0.0, 0
    done, info = False, {}
    while not done:
        before_flags, before_n = list(env.delivered_flags), len(env.dests)
        _, rew, term, trunc, info = env.step(int(policy(env)))
        total += rew
        hot += int(env.hotspots[env.pos])
        trail.append(env.pos)
        if len(env.dests) > before_n and new_order_step is None:
            new_order_step = env.steps
        for i, (b, a) in enumerate(zip(before_flags, env.delivered_flags)):
            if a and not b:
                events.append((env.steps, i))
        done = term or trunc
    return dict(trail=trail, start=start, dests=list(env.dests), deadlines=list(env.deadlines), events=events,
                new_order_step=new_order_step, n_initial=n0, reward=total, hot=hot, info=info,
                hotspots=env.hotspots.copy(), traffic=env.traffic, weather=env.weather)


def draw(runs, n):
    ref = next(iter(runs.values()))
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(ref["hotspots"], cmap="Reds", vmin=0, vmax=2.2, extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    for j, (name, r) in enumerate(runs.items()):
        ys, xs = zip(*r["trail"])
        off = (j - (len(runs) - 1) / 2) * 0.07
        ax.plot(np.array(xs) + off, np.array(ys) + off, color=COLOURS.get(name, "C0"), lw=2.2, alpha=0.9, label=name)
    sr, sc = ref["start"]
    ax.scatter([sc], [sr], marker="o", s=140, color="#2ca02c", edgecolor="black", zorder=5, label="Rider start")
    for i, (r, c) in enumerate(ref["dests"]):
        is_new = i >= ref["n_initial"]
        ax.scatter([c], [r], marker="*", s=420, color="#ff7f0e" if is_new else "#1f77b4", edgecolor="black", zorder=6)
        ax.annotate(f"{'new ' if is_new else ''}#{i + 1}", (c, r), ha="center", va="center", fontsize=8, zorder=7, color="white", weight="bold")
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_title("Pink-red cells = congested (hotspot). Stars = customers.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


try:
    scenarios, policies, n_models = load_everything()
except Exception as exc:  # missing data or models: tell the user exactly what to run
    st.error(f"Could not load the multi-order models: {exc}")
    st.info("Run `python clean_data.py` then `python ddqn/ddqn_multi.py --stage C` first, "
            "or copy the folder `ddqn/result_multi/` (with the ddqn_multi_seed*.pt files) from the teammate who trained it.")
    st.stop()

if n_models == 0:
    st.warning("No trained DDQN models found in ddqn/result_multi/. Showing Greedy and Planner only. "
               "Run `python ddqn/ddqn_multi.py --stage C` to create them.")

left, right = st.columns([1, 1.15])
with left:
    scen_name = st.selectbox("Scenario", list(scenarios))
    seed = st.slider("Test trip number", 0, 499, 7, help="Each number is one fixed unseen trip, so every method drives exactly the same trip.")
    chosen = st.multiselect("Methods to compare", list(policies), default=list(policies))
    env = scenarios[scen_name]
    runs = {name: rollout(env, policies[name], seed) for name in chosen}
    if runs:
        ref = next(iter(runs.values()))
        st.markdown(f"**Trip {seed}:** {len(ref['dests'])} customer(s) | traffic **{TRAFFIC[ref['traffic']]}** | weather **{WEATHER[ref['weather']]}**"
                    + (f" | new order appears at step **{ref['new_order_step']}**" if ref["new_order_step"] else ""))
        rows = []
        for name, r in runs.items():
            info = r["info"]
            rows.append({"Method": name, "Reward (higher is better)": round(r["reward"], 1),
                         "All orders delivered": "yes" if info.get("delivered") else f"no ({info.get('n_delivered', 0)}/{len(r['dests'])})",
                         "Steps": info.get("steps"), "Late minutes": round(info.get("late_total", 0.0), 1),
                         "Steps in congested cells": r["hot"]})
        st.dataframe(pd.DataFrame(rows).set_index("Method"))
        order_txt = []
        for name, r in runs.items():
            order = " then ".join(f"#{i + 1} (step {s})" for s, i in r["events"]) or "none delivered"
            order_txt.append(f"- **{name}:** {order}")
        st.markdown("**Delivery order**\n" + "\n".join(order_txt))
        st.caption("Reward = +50 per delivery, minus 0.5 per late minute and small costs for each move, congested cell and bad weather; "
                   "-20 if the trip times out. The Planner sees the full traffic map (an oracle ceiling); the DDQN agents only see the numbers described in the report.")
with right:
    if runs:
        st.pyplot(draw(runs, env.n))

st.divider()
st.subheader("Results over 500 unseen test trips")
c1, c2 = st.columns(2)
with c1:
    summary = RESULT_DIR / "stageC_summary_mean_std.csv"
    if summary.exists():
        st.markdown("**Tuned DDQN, mean and standard deviation over training seeds**")
        st.dataframe(pd.read_csv(summary, header=[0, 1], index_col=0).round(3))
    ens = RESULT_DIR / "ensemble_paired_differences.csv"
    if ens.exists():
        d = pd.read_csv(ens)
        d = d[d["metric"] == "reward"].drop(columns=["metric"])
        st.markdown("**Reward differences with 95% paired-bootstrap confidence intervals**")
        st.dataframe(d, hide_index=True)
with c2:
    for fname, cap in [("ensemble_comparison.png", "Mean reward per trip in three scenarios"),
                       ("error_regret_by_group.png", "Where the agent loses reward compared with the Planner"),
                       ("ablation.png", "Ablation: one change at a time (validation set)")]:
        if (RESULT_DIR / fname).exists():
            st.image(str(RESULT_DIR / fname), caption=cap)
