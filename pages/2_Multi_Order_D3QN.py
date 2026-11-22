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
from env_multi import MultiDeliveryEnv
from pages.multi_trip_setup import prepare_trip

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


def rollout(env, policy, seed, setup):
    """Drive one trip and record the path and the delivery events."""
    prepare_trip(env, seed, setup)
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
# SIDEBAR: page switcher, then the two choices needed to plan a trip
# ==============================================================================
try:
    envs, policies, n_models = load_everything()
    load_error = None
except Exception as exc:  # missing data or models: tell the user exactly what to run
    envs, policies, n_models, load_error = {}, {}, 0, exc

if load_error is None:
    test_orders = envs["Normal trip"].orders
    order_labels = [
        (f"Test order #{i + 1} · {row.distance_km:.1f} km · "
         f"{TRAFFIC[int(row.traffic)]} traffic · {row.time_taken:.0f} historical min")
        for i, row in enumerate(test_orders.itertuples(index=False))
    ]

    def order_label(index):
        return order_labels[index]

with st.sidebar:
    sidebar_header("pages/2_Multi_Order_D3QN.py")
    if load_error is None:
        step_head(1, "Choose orders")
        source = st.segmented_control(
            "Where the orders come from", ["Dataset orders", "Make my own"],
            default="Dataset orders", label_visibility="collapsed",
            help="Pick held-out dataset rows, or set the traffic and customer locations yourself.",
        ) or "Dataset orders"
        with st.expander("More options: trip situation"):
            scen_name = st.selectbox("Situation", list(SCENARIOS))
            st.caption(SCENARIOS[scen_name])
        is_dynamic = scen_name == "A new order arrives mid-trip"
        customer_options = [1, 2] if is_dynamic else [1, 2, 3]

        if source == "Dataset orders":
            example_trip = st.toggle("Use a ready-made example trip", value=False,
                                     help="Reproduce one of the original numbered test trips.")
            if example_trip:
                seed = st.number_input("Example trip number (0–499)", 0, 499, 7, step=1,
                                       help="Every planner sees the same generated test trip.")
                setup = {"mode": "example"}
            else:
                n_customers = st.segmented_control("Customers at start", customer_options,
                                                   default=2, key=f"dataset_count_{is_dynamic}")
                available = list(range(len(test_orders)))
                selected = []
                for i in range(n_customers):
                    choice = st.selectbox(f"Customer {i + 1} · dataset order", available,
                                          index=min(7 + i, len(available) - 1),
                                          format_func=order_label)
                    selected.append(choice)
                    available.remove(choice)
                setup = {"mode": "dataset", "order_indices": selected}
                if is_dynamic:
                    setup["late_index"] = st.selectbox("Order arriving mid-trip", available,
                                                         index=min(9, len(available) - 1),
                                                         format_func=order_label)
                with st.expander("Change the map layout"):
                    seed = st.number_input("Layout number (0–499)", 0, 499, 7, step=1,
                                           help="Controls the starting cell and the placement of customers.")
                st.caption("These are separate test-set orders combined into a simulated trip, "
                           "not a recorded group delivery. Their distances become grid destinations; "
                           "historical times act as delivery targets. The first order sets traffic and weather.")
        else:
            n_customers = st.segmented_control("Customers at start", customer_options,
                                               default=2, key=f"custom_count_{is_dynamic}")
            traffic = st.segmented_control("Traffic", list(TRAFFIC), default=1,
                                           format_func=lambda x: TRAFFIC[x])
            weather = st.segmented_control("Weather", list(WEATHER), default=0,
                                           format_func=lambda x: WEATHER[x])
            jam_pct = st.slider("Traffic-jam blocks", 0, 50, 15, 5, format="%d%%")
            defaults = [(2, 6), (7, 3), (8, 8)]
            with st.expander("Set rider and customer locations", expanded=True):
                st.caption("Choose cells on the 10×10 grid. Row 0 is the top; column 0 is the left.")
                col1, col2 = st.columns(2)
                start = (col1.number_input("Rider row", 0, 9, 4),
                         col2.number_input("Rider column", 0, 9, 4))
                destinations = []
                for i in range(n_customers):
                    col1, col2 = st.columns(2)
                    destinations.append((col1.number_input(f"Customer {i + 1} row", 0, 9, defaults[i][0]),
                                         col2.number_input(f"Customer {i + 1} column", 0, 9, defaults[i][1])))
            deadlines = [st.number_input(f"Customer {i + 1} target (min)", 5, 180, 40, step=5)
                         for i in range(n_customers)]
            st.caption("Each customer's target is measured from the start of the trip.")
            setup = {"mode": "custom", "start": start, "destinations": destinations,
                     "deadlines": deadlines, "traffic": traffic, "weather": weather,
                     "hotspot_frac": jam_pct / 100}
            seed = 7
            if is_dynamic:
                setup["arrival_step"] = st.number_input("New order arrives after step", 3, 8, 5)
                setup["late_distance_km"] = st.number_input("New order distance (km)", 1.0, 30.0, 6.0,
                                                              step=1.0)
                setup["late_target_min"] = st.number_input("New order target after arrival (min)", 5, 180, 40,
                                                             step=5)
                st.caption("The new customer's grid cell is generated at roughly this distance "
                           "from the rider when the order arrives.")

        step_head(2, "Choose a route planner")
        preferred = next((name for name in ("D3QN ensemble", "D3QN single model", "Greedy (nearest first)")
                          if name in policies), next(iter(policies)))
        primary = st.selectbox("Route planner", list(policies), index=list(policies).index(preferred),
                               format_func=label)
        st.caption(METHODS[primary][2])
        with st.expander("More options: compare routes"):
            compare_all = st.toggle("Compare all planners on this trip", value=False)
        chosen = [primary] + ([name for name in policies if name != primary] if compare_all else [])

page_hero(
    "Several deliveries · Reinforcement learning",
    "Plan several deliveries",
    "One rider carries 1 to 3 orders. The planner must decide which customer to visit first "
    "and which streets to take, while traffic jams slow some blocks down.",
    ["Choose orders", "Choose a route planner", "See the route and results"],
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

tab_trip, tab_results = st.tabs(["🗺️  1 · Try a trip", "📊  2 · Results across trips"])

with tab_trip:
    env = envs[scen_name]
    if setup["mode"] == "custom":
        env = MultiDeliveryEnv(env.orders, max_km=env.max_km, dynamic=is_dynamic,
                               map_seed=123 if scen_name == "Unfamiliar traffic map" else 0,
                               hotspot_frac=setup["hotspot_frac"])
    try:
        runs = {name: rollout(env, policies[name], int(seed), setup) for name in chosen}
    except ValueError as exc:
        st.error(str(exc))
        runs = {}
    if not runs:
        if not chosen:
            st.info("Choose at least one route planner on the left (step 3).")
    else:
        if setup["mode"] == "custom":
            st.caption("Your scenario is simulated on the same grid for every planner. "
                       "The results below are for this trip only.")
        ref = runs[primary]
        chips = [f"📦 {len(ref['dests'])} customer{'s' if len(ref['dests']) != 1 else ''}",
                 f"{TRAFFIC_ICONS[ref['traffic']]} {TRAFFIC[ref['traffic']]} traffic",
                 f"{WEATHER_ICONS[ref['weather']]} {WEATHER[ref['weather']]}"]
        if ref["new_order_step"]:
            chips.append(f"🆕 New order appears at step {ref['new_order_step']}")
        st.markdown('<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
                    unsafe_allow_html=True)

        delivered = ref["info"].get("n_delivered", len(ref["events"]))
        total_orders = len(ref["dests"])
        elapsed = ref["info"].get("elapsed_min", 0.0)
        outcome = (f"**{label(primary)}** delivered {delivered}/{total_orders} orders "
                   f"in {elapsed:.1f} simulated minutes.")
        if delivered == total_orders:
            st.success(outcome)
        else:
            st.warning(outcome + " The trip ended before every order was delivered.")
        if compare_all:
            best = max(runs, key=lambda name: runs[name]["reward"])
            st.caption(f"On this trip, {label(best)} earned the highest score "
                       f"({runs[best]['reward']:.1f} points).")

        col_map, col_table = st.columns([1, 1.15], gap="medium")
        with col_map:
            fig = draw(runs, env.n)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Red blocks = traffic jams (slower). Blue stars = customers, orange star = order that "
                       "arrived mid-trip. Each line is one selected planner's route.")
        with col_table:
            rows = []
            for name, r in runs.items():
                info = r["info"]
                order = " → ".join(f"#{i + 1}" for _, i in r["events"]) or "none"
                rows.append({"Planner": label(name),
                             "Score": round(r["reward"], 1),
                             "Delivered": f"{info.get('n_delivered', len(r['events']))}/{len(r['dests'])}",
                             "Trip time (min)": round(info.get("elapsed_min", 0.0), 1),
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
                    "- **Trip time:** simulated minutes until the rider stops.\n"
                    "- **Late (min):** total minutes after each customer's expected delivery time.\n"
                    "- **Jam blocks:** how many congested blocks the rider drove through. Lower is better.\n"
                    "- The **perfect-knowledge planner** sees the whole traffic map, so it is a best-case "
                    "reference. The AI only sees nearby traffic."
                )

with tab_results:
    st.markdown('<div class="section-title">How the planners do over many test trips</div>'
                '<div class="section-sub">Saved averages over generated test trips from the D3QN team’s '
                'experiments. These results do not change when you choose orders or make a scenario above.</div>',
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
