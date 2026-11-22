"""Saved project evidence, separate from the one- and several-delivery demos."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ui_shared import apply_theme, page_hero, sidebar_header

RESULT_DIR = BASE_DIR / "result"
QL_RESULT_DIR = BASE_DIR / "q_learning" / "result"
DQN_RESULT_DIR = BASE_DIR / "dqn" / "result"
FIVE_MODEL_SUMMARY_PATH = BASE_DIR / "dqn" / "five_model_summary.json"
D3QN_RESULT_DIR = BASE_DIR / "d3qn" / "result"
MULTI_RESULT_DIR = BASE_DIR / "d3qn" / "result_multi"
EXPLORATION_DIR = BASE_DIR / "exploration" / "zomato_exploration"

st.set_page_config(page_title="Data & Experiments", page_icon="🔬", layout="wide")
apply_theme()

with st.sidebar:
    sidebar_header("pages/3_Data_and_Experiments.py")

page_hero(
    "Project evidence",
    "Data & experiments",
    "See how the dataset was prepared and how the route planners were tested. "
    "These are saved assignment results; changing an order on the planning pages does not change them.",
    ["Review shared data", "See one-delivery tests", "See several-delivery tests"],
)
st.write("")

st.subheader("1 · Shared dataset")
st.caption("These preparation steps support both delivery demos. They are not results for one particular route.")

clean_log = BASE_DIR / "data" / "cleaning_log.csv"
with st.expander("📦 How was the dataset cleaned?"):
    if clean_log.exists():
        st.caption("Rows kept after each cleaning step, such as removing invalid locations and missing fields.")
        st.dataframe(pd.read_csv(clean_log), width="stretch", hide_index=True)
    else:
        st.info("No cleaning log has been saved yet.")

missing_csv = RESULT_DIR / "missing_data_comparison.csv"
missing_png = RESULT_DIR / "missing_data_comparison.png"
with st.expander("🧩 What happened to missing values?"):
    st.caption(
        "The rows show three ways to handle incomplete records. Retention is how much data remains; "
        "R² and MAE come from a simple delivery-duration benchmark, not from the routing models."
    )
    if missing_csv.exists():
        st.dataframe(pd.read_csv(missing_csv), width="stretch", hide_index=True)
        if missing_png.exists():
            st.image(str(missing_png), width="stretch")
    else:
        st.info("No missing-data comparison has been saved yet.")

exploration_images = sorted(EXPLORATION_DIR.glob("*.png")) if EXPLORATION_DIR.exists() else []
with st.expander("📈 What does the dataset look like?"):
    if exploration_images:
        cols = st.columns(3)
        for idx, path in enumerate(exploration_images):
            cols[idx % 3].image(str(path), caption=path.stem.replace("_", " "), width="stretch")
    else:
        st.info("No exploration charts have been saved yet.")

st.subheader("2 · One-delivery model experiments")
st.caption(
    "These results explain model settings and difficult cases for one customer. "
    "They were produced by saved experiments, not by the delivery currently selected in the app."
)

with st.expander("🏁 How do the five learned models compare on the same deliveries?", expanded=True):
    st.caption(
        "Q-Learning is the tabular reference. DQN, Double DQN, Dueling DQN and D3QN "
        "form the four-way neural-network comparison. This experiment uses the same training "
        "step budget and held-out delivery scenarios in the 10×10 grid simulator for all five. "
        "Q-Learning keeps its team's "
        "refined state and cycle-avoidance policy; the four DQN models share one neural setup "
        "so the Double and Dueling changes can be compared directly."
    )
    if FIVE_MODEL_SUMMARY_PATH.exists():
        summary = json.loads(FIVE_MODEL_SUMMARY_PATH.read_text(encoding="utf-8"))
        results = pd.DataFrame(summary["comparison"])
        expected_models = {"Q-Learning", "DQN", "Double DQN", "Dueling DQN", "D3QN"}
        if expected_models <= set(results["model"]):
            comparison = (results.set_index("model")
                          .rename(columns={"delivered_pct": "Delivered (%)", "on_time_pct": "On time (%)",
                                           "completed_trip_min": "Completed trip (min)"})
                          .reindex(["Q-Learning", "DQN", "Double DQN", "Dueling DQN", "D3QN"]))
            chart_col, table_col = st.columns([1.15, 1])
            with chart_col:
                st.bar_chart(comparison[["Delivered (%)", "On time (%)"]], height=290)
            with table_col:
                st.dataframe(comparison.round(1), width="stretch")
            neural = comparison.loc[["DQN", "Double DQN", "Dueling DQN", "D3QN"]]
            st.info(
                f"The four DQN variants delivered {neural['Delivered (%)'].min():.1f}–"
                f"{neural['Delivered (%)'].max():.1f}% of test trips. Their on-time rates were "
                f"{neural['On time (%)'].min():.1f}–{neural['On time (%)'].max():.1f}%. "
                "These small gaps do not establish that one DQN architecture is consistently better."
            )
            with st.expander("See results for each training seed"):
                per_seed = pd.DataFrame(summary["per_seed"])
                per_seed["delivered"] = (per_seed["delivered"] * 100).round(1)
                per_seed["on_time"] = (per_seed["on_time"] * 100).round(1)
                st.dataframe(
                    per_seed[["model", "training_seed", "delivered", "on_time", "reward", "steps"]]
                    .rename(columns={"model": "Model", "training_seed": "Seed",
                                     "delivered": "Delivered (%)", "on_time": "On time (%)",
                                     "reward": "Mean reward", "steps": "Mean moves"}).round(1),
                    width="stretch", hide_index=True,
                )
            st.caption(
                f"{len(summary['seeds'])} training seeds · {summary['train_steps']:,} "
                f"training steps per model and seed · {summary['eval_episodes']} test trips per seed. "
                "On time means delivered within the order's recorded delivery duration. "
                "Trip minutes are averaged only over completed deliveries. "
                "The historical test split has been inspected before, so treat this as comparative evidence."
            )
        else:
            st.info("A completed run was found, but it does not contain results for all five models.")
    else:
        st.info("The five-model experiment has not completed yet. Its results will appear here after the run finishes.")

master_csv = RESULT_DIR / "master_all_models_results.csv"
fallback_csv = DQN_RESULT_DIR / "results.csv"
benchmark_csv = master_csv if master_csv.exists() else fallback_csv
benchmark_image = RESULT_DIR / "master_comparison_chart.png"
with st.expander("📊 What did the saved one-delivery benchmark find?"):
    st.caption(
        "These historical results were saved by the evaluation scripts. For a live comparison "
        "on the same selected delivery, use 'Compare all planners' on the One delivery page."
    )
    if benchmark_image.exists():
        st.image(str(benchmark_image), width="stretch")
    if benchmark_csv.exists():
        benchmark = pd.read_csv(benchmark_csv)
        formats = {
            "success_rate": "{:.1%}", "avg_reward": "{:.2f}", "avg_steps": "{:.2f}",
            "avg_delivery_time_min": "{:.2f}", "on_time_rate": "{:.1%}",
            "std_reward": "{:.2f}", "route_efficiency": "{:.1%}",
            "avg_hotspot_steps": "{:.2f}",
        }
        formats = {name: fmt for name, fmt in formats.items() if name in benchmark.columns}
        st.dataframe(benchmark.style.format(formats), width="stretch", hide_index=True)
    if not benchmark_image.exists() and not benchmark_csv.exists():
        st.info("No saved one-delivery benchmark has been found yet.")

training_curves = [
    (QL_RESULT_DIR / "ql_learning_curves.png", "Q-Learning"),
    (DQN_RESULT_DIR / "dqn_learning_curve_v1.png", "DQN v1"),
    (DQN_RESULT_DIR / "dqn_learning_curve_v2.png", "DQN v2"),
    (DQN_RESULT_DIR / "dqn_learning_curve_v3.png", "DQN v3"),
    (D3QN_RESULT_DIR / "d3qn_learning_curves.png", "Double DQN and Dueling Double DQN"),
]
training_curves = [(path, label) for path, label in training_curves if path.exists()]
with st.expander("📉 How did the models learn during training?"):
    st.caption("Learning curves show training progress; they are not a head-to-head test score.")
    if training_curves:
        cols = st.columns(3)
        for idx, (path, label) in enumerate(training_curves):
            cols[idx % 3].image(str(path), caption=label, width="stretch")
    else:
        st.info("No training curves have been saved yet.")

tuning_files = [
    (DQN_RESULT_DIR / "part_b_tuning_results.csv", "DQN settings and validation results"),
    (DQN_RESULT_DIR / "part_b_ensemble_results.csv", "Historical DQN and Double DQN variants"),
    (QL_RESULT_DIR / "ql_part_b_tuning.csv", "Q-Learning settings"),
    (D3QN_RESULT_DIR / "d3qn_part_b_tuning.csv", "Dueling Double DQN settings"),
]
with st.expander("⚙️ Which training settings were tried?"):
    st.caption("Tuning compares training choices. Tables from different runs may use different test protocols.")
    analysis_image = DQN_RESULT_DIR / "part_b_analysis.png"
    if analysis_image.exists():
        st.image(str(analysis_image), width="stretch")
    found_tuning = False
    for path, label in tuning_files:
        if path.exists():
            found_tuning = True
            st.markdown(f"**{label}**")
            st.dataframe(pd.read_csv(path), width="stretch", hide_index=True)
    if not found_tuning and not analysis_image.exists():
        st.info("No tuning results have been saved yet.")

robustness_csv = DQN_RESULT_DIR / "part_b_robustness_results.csv"
with st.expander("🌪️ How did DQN variants handle difficult traffic and weather?"):
    st.caption(
        "This historical study compares DQN variants and simple baselines. It does not include Q-Learning "
        "or the several-delivery D3QN. 'On time' uses each order's recorded duration as a proxy target, "
        "so it is different from the 45-minute target on the planning page."
    )
    if robustness_csv.exists():
        robustness = pd.read_csv(robustness_csv)
        required = {"agent", "scenario", "on_time_rate", "avg_delivery_time_min"}
        if required <= set(robustness.columns):
            robustness["agent"] = robustness["agent"].replace({"Tuned DDQN": "Tuned Double DQN"})
            robustness["scenario"] = robustness["scenario"].str.replace(r"^\d+\.\s*", "", regex=True)
            on_time_col, duration_col = st.columns(2)
            on_time_col.markdown("**On-time rate (%) — higher is better**")
            on_time_col.dataframe(
                (robustness.pivot(index="agent", columns="scenario", values="on_time_rate") * 100).round(1),
                width="stretch",
            )
            duration_col.markdown("**Average completed trip (min) — lower is better**")
            duration_col.dataframe(
                robustness.pivot(index="agent", columns="scenario", values="avg_delivery_time_min").round(1),
                width="stretch",
            )
            with st.popover("Full stress-test table"):
                st.dataframe(robustness, width="stretch", hide_index=True)
        else:
            st.dataframe(robustness, width="stretch", hide_index=True)
    else:
        st.info("No one-delivery stress-test results have been saved yet.")

error_image = RESULT_DIR / "error_analysis" / "error_analysis.png"
hotspot_csv = RESULT_DIR / "error_analysis" / "hotspot_analysis.csv"
with st.expander("🔍 Where did the one-delivery planners struggle?"):
    if error_image.exists():
        st.image(str(error_image), width="stretch")
    if hotspot_csv.exists():
        st.markdown("**Average moves through traffic jams by traffic level**")
        st.dataframe(pd.read_csv(hotspot_csv), width="stretch", hide_index=True)
    if not error_image.exists() and not hotspot_csv.exists():
        st.info("No one-delivery error analysis has been saved yet.")

st.subheader("3 · Several-delivery model experiments")
st.caption(
    "This is a different task: one rider serves several customers. Its order-level on-time rate "
    "should not be ranked directly against the one-delivery results above."
)
multi_summary = MULTI_RESULT_DIR / "2000trips_ensemble_summary.csv"
multi_chart = MULTI_RESULT_DIR / "2000trips_ensemble_comparison.png"
with st.expander("📦 How did the multi-order D3QN compare with its baselines?"):
    if multi_summary.exists():
        summary = pd.read_csv(multi_summary)
        shown = summary[summary["model"].isin(["Greedy", "Planner", "D3QN ensemble"])].copy()
        shown["model"] = shown["model"].replace({
            "Greedy": "Nearest customer first",
            "Planner": "Perfect-knowledge planner",
            "D3QN ensemble": "D3QN ensemble",
        })
        shown = shown[["scenario", "model", "success_rate", "on_time_order_rate", "avg_trip_time_min"]]
        shown.columns = ["Situation", "Planner", "Trips completed", "Orders on time", "Avg trip (min)"]
        st.dataframe(shown.round(2), width="stretch", hide_index=True)
    if multi_chart.exists():
        st.image(str(multi_chart), width="stretch")
    if not multi_summary.exists() and not multi_chart.exists():
        st.info("No several-delivery experiment results have been saved yet.")

st.page_link("pages/2_Multi_Order_D3QN.py", label="Try a several-delivery trip", icon="📦")
