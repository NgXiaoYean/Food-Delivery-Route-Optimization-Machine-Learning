"""
"Several deliveries" page: one rider, 1 to 3 customers, optional new order mid-trip.
Shows the route chosen by the trained Dueling Double DQN (D3QN) agents next to two simple baselines.

Opened from the page switcher in the sidebar (ui_shared.sidebar_header).
Run from the project root:  streamlit run app.py
Needs the saved models in d3qn/result_multi/  (made by:  python d3qn/d3qn_multi.py --stage C)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "d3qn"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui_shared import apply_theme, page_hero, sidebar_header, step_head

RESULT_DIR = BASE_DIR / "d3qn" / "result_multi"
TRAFFIC = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}
WEATHER = {0: "Clear", 1: "Windy", 2: "Fog", 3: "Stormy"}
TRAFFIC_ICONS = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
WEATHER_ICONS = {0: "☀️", 1: "🌬️", 2: "🌫️", 3: "⛈️"}

# Internal method name -> (short label, colour, one-line explanation)
METHODS = {
    "Greedy (nearest first)": ("Nearest customer first", "#9a9a9a",
                               "Simple rule: always drive to the closest customer, ignoring traffic."),
    "Planner (oracle)": ("Perfect-knowledge planner", "#333333",
                         "Sees the whole traffic map in advance. A best-case reference, not a real rider."),
    "D3QN single model": ("AI: D3QN (1 model)", "#7fb2e5",
                          "One trained Dueling Double DQN. Only sees nearby traffic, like a real rider."),
    "D3QN ensemble": ("AI: D3QN (team of models)", "#d62728",
                      "Several trained D3QN models vote on each move. Usually the most reliable AI."),
}
SCENARIOS = {
    "Normal trip": "All orders are known when the rider leaves.",
    "A new order arrives mid-trip": "An extra customer is added while the rider is already driving.",
    "Unfamiliar traffic map": "Traffic jams are in places the AI never saw during training.",
}


def label(name):
    return METHODS.get(name, (name,))[0]


st.set_page_config(page_title="Several deliveries", page_icon="📦", layout="wide")
apply_theme()


@st.cache_resource(show_spinner="Loading data and trained models ...")
def load_everything():
    import d3qn_multi as dm
    from ensemble_multi import make_ensemble_policy
    from env_multi import greedy_policy_multi, planner_policy_multi

    ctx = dm.make_context(False)
    paths = sorted(RESULT_DIR.glob("d3qn_multi_seed*.pt"))
    nets = [dm.load_model(p)[0] for p in paths]
    policies = {"Greedy (nearest first)": greedy_policy_multi, "Planner (oracle)": planner_policy_multi}
    if nets:
        policies["D3QN single model"] = dm.make_policy(nets[0])
    if len(nets) >= 2:
        policies["D3QN ensemble"] = make_ensemble_policy(nets)
    envs = {"Normal trip": ctx["test_env"], "A new order arrives mid-trip": ctx["dynamic_env"],
            "Unfamiliar traffic map": ctx["unseen_map_env"]}
    return envs, policies, len(nets)


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
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.imshow(ref["hotspots"], cmap="Reds", vmin=0, vmax=2.2, extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for j, (name, r) in enumerate(runs.items()):
        ys, xs = zip(*r["trail"])
        off = (j - (len(runs) - 1) / 2) * 0.07
        ax.plot(np.array(xs) + off, np.array(ys) + off, color=METHODS.get(name, (0, "C0"))[1],
                lw=2.4, alpha=0.9, label=label(name))
    sr, sc = ref["start"]
    ax.scatter([sc], [sr], marker="o", s=160, color="#2ca02c", edgecolor="black", zorder=5, label="Rider starts here")
    for i, (r, c) in enumerate(ref["dests"]):
        is_new = i >= ref["n_initial"]
        ax.scatter([c], [r], marker="*", s=460, color="#ff7f0e" if is_new else "#1f77b4", edgecolor="black", zorder=6)
        ax.annotate(f"{i + 1}", (c, r), ha="center", va="center", fontsize=8, zorder=7, color="white", weight="bold")
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# ==============================================================================
# SIDEBAR: page switcher, then three numbered choices
# ==============================================================================
try:
    envs, policies, n_models = load_everything()
    load_error = None
except Exception as exc:  # missing data or models: tell the user exactly what to run
    envs, policies, n_models, load_error = {}, {}, 0, exc

with st.sidebar:
    sidebar_header("pages/2_Multi_Order_D3QN.py")
    if load_error is None:
        step_head(1, "Choose a situation")
        scen_name = st.radio("Situation", list(SCENARIOS), label_visibility="collapsed",
                             captions=list(SCENARIOS.values()))

        step_head(2, "Choose a trip")
        seed = st.number_input("Trip number (0-499)", 0, 499, 7, step=1,
                               help="Each number is one fixed test trip the AI never trained on. "
                                    "Every method drives exactly the same trip, so the comparison is fair.")

        step_head(3, "Choose who plans the route")
        chosen = st.multiselect("Route planners", list(policies), default=list(policies),
                                format_func=label, label_visibility="collapsed")
        for name in chosen:
            st.caption(f"**{label(name)}:** {METHODS.get(name, ('', '', ''))[2]}")

page_hero(
    "Several deliveries · Reinforcement learning",
    "Plan several deliveries",
    "One rider carries 1 to 3 orders. The planner must decide which customer to visit first "
    "and which streets to take, while traffic jams slow some blocks down.",
    ["Choose a situation", "Choose a trip", "Compare the routes on the map"],
)
st.write("")

if load_error is not None:
    st.error(f"Could not load the multi-order models: {load_error}")
    st.info("Run `python clean_data.py` then `python d3qn/d3qn_multi.py --stage C` first, "
            "or copy the folder `d3qn/result_multi/` (with the d3qn_multi_seed*.pt files) from the teammate who trained it.")
    st.stop()

if n_models == 0:
    st.markdown(
        '<div class="note warm"><span class="note-icon">ℹ️</span><div><b>The trained AI models are not on this computer.</b> '
        "Only the two simple planners are shown. Run <code>python d3qn/d3qn_multi.py --stage C</code>, "
        "or copy the <code>d3qn_multi_seed*.pt</code> files into <code>d3qn/result_multi/</code>.</div></div>",
        unsafe_allow_html=True,
    )

tab_trip, tab_results = st.tabs(["🗺️  Plan a trip", "📊  Overall results"])

with tab_trip:
    env = envs[scen_name]
    runs = {name: rollout(env, policies[name], int(seed)) for name in chosen}
    if not runs:
        st.info("Choose at least one route planner on the left (step 3).")
    else:
        ref = next(iter(runs.values()))
        chips = [f"📦 {len(ref['dests'])} customer{'s' if len(ref['dests']) != 1 else ''}",
                 f"{TRAFFIC_ICONS[ref['traffic']]} {TRAFFIC[ref['traffic']]} traffic",
                 f"{WEATHER_ICONS[ref['weather']]} {WEATHER[ref['weather']]}"]
        if ref["new_order_step"]:
            chips.append(f"🆕 New order appears at step {ref['new_order_step']}")
        st.markdown('<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
                    unsafe_allow_html=True)

        best = max(runs, key=lambda n: runs[n]["reward"])
        st.success(f"On this trip, **{label(best)}** scores best "
                   f"({runs[best]['reward']:.1f} points, {runs[best]['info'].get('late_total', 0.0):.0f} late minutes).")

        col_map, col_table = st.columns([1, 1.15], gap="medium")
        with col_map:
            st.pyplot(draw(runs, env.n))
            st.caption("Red blocks = traffic jams (slower). Blue stars = customers, orange star = order that "
                       "arrived mid-trip. Each line is one planner's route.")
        with col_table:
            rows = []
            for name, r in runs.items():
                info = r["info"]
                order = " → ".join(f"#{i + 1}" for _, i in r["events"]) or "none"
                rows.append({"Planner": label(name),
                             "Score": round(r["reward"], 1),
                             "Delivered": f"{info.get('n_delivered', len(r['events']))}/{len(r['dests'])}",
                             "Visit order": order,
                             "Late (min)": round(info.get("late_total", 0.0), 1),
                             "Jam blocks": r["hot"],
                             "Steps": info.get("steps")})
            st.dataframe(pd.DataFrame(rows).set_index("Planner"), width="stretch")
            with st.expander("What do these numbers mean?"):
                st.markdown(
                    "- **Score:** +50 per delivery, minus 0.5 per late minute and a small cost for every move, "
                    "jam block and bad weather; −20 if the trip runs out of time. Higher is better.\n"
                    "- **Visit order:** which customer was reached first, second, third.\n"
                    "- **Late (min):** total minutes after each customer's expected delivery time.\n"
                    "- **Jam blocks:** how many congested blocks the rider drove through. Lower is better.\n"
                    "- The **perfect-knowledge planner** sees the whole traffic map, so it is a best-case "
                    "reference. The AI only sees nearby traffic."
                )

with tab_results:
    st.markdown('<div class="section-title">How the planners do over many test trips</div>'
                '<div class="section-sub">Averages over unseen test trips, from the D3QN team\'s experiments.</div>',
                unsafe_allow_html=True)
    summary = RESULT_DIR / "2000trips_ensemble_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        df = df[df["model"].isin(["Greedy", "Planner", "D3QN ensemble"])]
        df = df.assign(model=df["model"].map({"Greedy": "Nearest customer first",
                                              "Planner": "Perfect-knowledge planner",
                                              "D3QN ensemble": "AI: D3QN (team of models)"}))
        table = df[["scenario", "model", "avg_reward", "on_time_order_rate", "avg_late_min", "avg_hotspot_steps"]]
        table = table.rename(columns={"scenario": "Situation", "model": "Planner", "avg_reward": "Avg score",
                                      "on_time_order_rate": "Orders on time", "avg_late_min": "Avg late (min)",
                                      "avg_hotspot_steps": "Avg jam blocks"})
        st.dataframe(table.round(2), hide_index=True, width="stretch",
                     column_config={"Orders on time": st.column_config.ProgressColumn(format="percent",
                                                                                      min_value=0, max_value=1)})
    images = [("2000trips_ensemble_comparison.png", "Average score per trip in each situation"),
              ("ensemble_comparison.png", "Average score per trip in each situation"),
              ("error_regret_by_group.png", "Where the AI loses points compared with the perfect-knowledge planner"),
              ("ablation.png", "Ablation: one change at a time (validation set)")]
    shown = [(f, cap) for f, cap in images if (RESULT_DIR / f).exists()]
    if shown:
        cols = st.columns(2)
        for k, (fname, cap) in enumerate(shown):
            cols[k % 2].image(str(RESULT_DIR / fname), caption=cap)
    if not summary.exists() and not shown:
        st.info("No saved results found in d3qn/result_multi/ yet.")
