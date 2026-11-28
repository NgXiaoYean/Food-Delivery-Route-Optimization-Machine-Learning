# Food Delivery Route Optimization using Reinforcement Learning

An end-to-end Reinforcement Learning system optimizing urban food delivery routes under dynamic traffic, weather, and congestion hotspots, grounded in **40,482 cleaned delivery orders** (from 45,584 raw records) from the [Kaggle Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset?select=Zomato+Dataset.csv).

---

## 👥 3-Algorithm Architecture (Mapped to Team Members & Assessment Rubric)

| Module / Folder | Algorithm | Methodology | Part A (Model Selection & Iteration) | Part B (Optimization, Regularization & Ensemble) |
| :--- | :--- | :--- | :--- | :--- |
| **`q_learning/`** | **1. Tabular Q-Learning** | Model-Free Tabular TD(0) | Raw 40k Discrete States (`v1`, 15.6% success) $\rightarrow$ Refined Relative + Hotspot States (`v2`, 73.8% success) | Validation search over LR ($\alpha$), Discount ($\gamma$), & L2 Q-Table Regularization (`weight_decay=1e-3`) $\rightarrow$ **92.8% delivered (same map, 64/16/20 split)** / **72.0% delivered (unseen map 202)** |
| **`dqn/`** | **2. Deep Q-Network (DQN)** | Value-Based Deep RL (MLP + Replay + Target Net) | `v1` Raw 8D (23.0%) $\rightarrow$ `v2` +Direction & Scaling (**100%**, 29.09 reward) $\rightarrow$ `v3` +Hotspot Radar (**100%**, 30.02 reward) | Validation hyperparameter search (Double-Q update), Missing-Data Handling Study, & Stress Testing across 4 extreme scenarios |
| **`d3qn/`** | **3. Dueling Double DQN (D3QN)** | Dual-Stream Value $V(s)$ + Advantage $A(s,a)$ & Double Q-Update | Controlled 4-way DQN-family comparison (`DQN`, `Double DQN`, `Dueling DQN`, `D3QN` in `compare/part_a_comparison.py` & `d3qn/ladder_multi.py`) | Multi-order (`env_multi.py`, 1–3 customers, 70/15/15 split): Validation Random Search, 5-Seed Ensemble with Paired Bootstrap 95% CIs, One-at-a-Time Ablation, 10-Rule Sensitivity, Calibrated Clock & Grid-Size Scalability |
| **`evaluate.py` & `app.py`** | **Ensembles & Streamlit Prototype** | Standardized Q-Value Fusion | Automated 8-metric single-order & multi-order benchmarks against `Random`, `Greedy`, and `Dijkstra Planner` | Single-order 3-family ensemble (`QL + DQN + D3QN`, **30.76 reward**), Multi-order 5-seed `D3QN` ensemble, & 3-Page **Streamlit Prototype UI** (`app.py`, `pages/2_Multi_Order_D3QN.py`, `pages/3_Data_and_Experiments.py`) |

> **Note on Tabular Q-Learning on Seen vs. Unseen Maps (`92.8%` vs. `72.0%` / `57.8%`):**
> - **`92.8%` (Seen Map `map_seed=0`, `q_learning/q_learning_pipeline.py`):** Trained on `map_seed=0` (64/16/20 split, validation-selected L2 `weight_decay=1e-3`) and evaluated on the same fixed hotspot map (`map_seed=0`).
> - **`72.0%` (`evaluate_unseen_map.py`) / `57.8% ± 14.0%` (`compare/part_a_comparison.py`, 5 seeds, 60k steps):** When evaluated on an unseen test hotspot map (`map_seed=202`), Tabular Q-Learning drops sharply because new hotspot coordinates produce discrete state tuples rarely visited during training, whereas neural `D3QN` maintains **100.0% delivery (`30.65` reward)** on unseen `map_seed=202`.
>
> **Note on Single-Order vs. Multi-Order DQN-Family Results:**
> - On **Single-Order (`compare/part_a_comparison.py`, 5 seeds)**, `DQN` (`99.8%` delivered, `57.5%` on-time proxy), `Double DQN` (`100.0%`, `56.6%`), `Dueling DQN` (`100.0%`, `57.7%`), and `D3QN` (`100.0%`, `57.6%`) are statistically tied (mean rewards `32.1–32.7`).
> - On **Multi-Order (`d3qn/`, 1–3 customers, 2,000 test trips)**, the 5-seed **D3QN Ensemble** beats `Greedy` by **`+4.35`** (`95% CI [4.09, 4.61]`, normal), **`+5.25`** (`[4.67, 5.81]`, new order mid-trip), and **`+3.92`** (`[3.69, 4.16]`, unseen map), closing **74%–84%** of the gap to the full-map Dijkstra Planner.

---

## 📂 Project Structure

```text
├── data/
│   ├── zomato_dataset.csv             # Raw Kaggle Zomato dataset (45,584 rows)
│   ├── dataset_clean.csv              # Cleaned delivery dataset (40,482 valid orders)
│   ├── cleaning_log.csv               # Step-by-step data cleaning audit log
│   └── data_checks.csv                # Empirical verification of negative-GPS & dropped-row assumptions
├── compare/                           # Official Part A Controlled 5-Model Comparison (64/16/20 split, 5 seeds)
│   ├── part_a_comparison.py           # Runs Q-Learning, DQN, Double DQN, Dueling DQN, D3QN on identical budget
│   └── part_a_summary.json            # Portable 5-seed summary loaded by pages/3_Data_and_Experiments.py
├── d3qn/                              # Algorithm 3: Dueling Double DQN (D3QN) — Multi-Order & Single-Order
│   ├── d3qn_multi.py                  # Multi-order Stage A (arch check), Stage B (random search), Stage C (5 seeds)
│   ├── ladder_multi.py                # 4-way multi-order architecture ladder (DQN, Double DQN, Dueling DQN, D3QN)
│   ├── ensemble_multi.py              # 5-seed D3QN ensemble + 5,000-resample paired bootstrap 95% CIs
│   ├── ablation_multi.py              # One-at-a-time hyperparameter & feature ablation on validation trips
│   ├── robustness_multi.py            # 10 reward-rule variations, input-group masking & safe_argmax filter tests
│   ├── retrain_study_multi.py         # Grid-size scalability (6x6 to 15x15) & data-fitted clock retraining
│   ├── error_analysis_multi.py        # Per-trip regret vs. Dijkstra Planner by customer count, traffic & weather
│   ├── timing_multi.py                # Per-decision CPU inference latency benchmark
│   ├── split_study_multi.py           # Sensitivity across 5 train/val/test split ratios & seeds
│   ├── d3qn_pipeline.py               # Historical single-order D3QN pipeline (80/20 split, loaded by app.py)
│   ├── result/                        # Saved single-order D3QN checkpoint (.pt), curves & tuning CSVs
│   └── result_multi/                  # Saved multi-order 5-seed models (.pt), best_config.json, CSVs & PNGs
├── q_learning/                        # Algorithm 1: Tabular Q-Learning
│   ├── q_learning_pipeline.py         # Single-order Part A (v1 vs v2 states) + Part B (L2 reg & tuning)
│   └── result/                        # Saved Q-table (.pkl), curves (.png) & tuning CSVs
├── dqn/                               # Algorithm 2: Deep Q-Network (Part A v1-v3 + Part B Refinement)
│   ├── dqn_v1.py                      # Iteration 1: Raw 8D coordinates
│   ├── dqn_v2.py                      # Iteration 2: +Relative displacement & reward scaling
│   ├── dqn_v3.py                      # Iteration 3: +4 neighboring hotspot sensors (15D)
│   ├── part_b_refine.py               # Part B: Double-Q search, DQN ensemble & 4-scenario stress test
│   ├── experiments.py                 # 64/16/20 3-way split, SHA-256 manifest & distance scale helpers
│   ├── learning_demo.py               # Interactive in-app teaching animation for Tab 1 of app.py
│   └── result/                        # Saved weights (.pt), curves (.png) & Part B CSVs
├── exploration/
│   └── zomato_exploration/            # 16 EDA distributions, correlations & dataset report
├── result/
│   ├── master_all_models_results.csv  # Historical single-order 8-metric scoreboard (80/20 split, same map)
│   ├── seen_vs_unseen_map_results.csv # Saved 80/20 models on seen map (map_seed=0) vs unseen map (map_seed=202)
│   ├── master_comparison_chart.png    # 4-panel multi-algorithm visual benchmark
│   ├── missing_data_comparison.csv    # Part B.1 missing-data strategy comparison
│   └── error_analysis/                # Single-order breakdown tables & charts by traffic, weather & distance
├── tests/
│   └── test_core.py                   # Core unit tests (splits, seed reproducibility, rewards, Double-Q, Planner)
├── pages/
│   ├── 2_Multi_Order_D3QN.py          # Page 2: Multi-order D3QN interactive simulator & 2,000-trip results
│   ├── 3_Data_and_Experiments.py      # Page 3: Saved project evidence (cleaning, 5-model Part A, benchmarks)
│   └── multi_trip_setup.py            # Trip preparation helper for dataset/example/custom multi-order demos
├── ui_shared.py                       # Shared CSS theme and 3-page sidebar navigation switcher
├── clean_data.py                      # 8-stage data cleaning pipeline
├── data_checks.py                     # Verification of negative-coordinate & removed-row distributions
├── missing_data_study.py              # Empirical comparison of 3 missing-value strategies
├── eda.py                             # Automated Exploratory Data Analysis
├── env.py                             # Shared 10x10 Single-Order Gymnasium Delivery Environment
├── env_multi.py                       # Extended 10x10 Multi-Order Delivery Environment (1-3 customers + dynamic arrival)
├── shared.py                          # Shared 80/20 split & 8-metric single-order evaluation protocol
├── evaluate.py                        # Historical single-order master benchmark & Cross-Algorithm Ensemble
├── evaluate_unseen_map.py             # Evaluates saved 80/20 checkpoints on map_seed=0 vs map_seed=202
├── analyze_errors.py                  # Historical single-order error & hotspot analysis
├── app.py                             # Page 1: Single-Order Interactive Route Planner Dashboard
└── requirements.txt                   # Dependencies
```

---

## 🚀 Execution Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Data Preparation, Sanity Checks, EDA & Missing-Data Strategy Study
python clean_data.py
python data_checks.py
python eda.py
python missing_data_study.py

# 3. Official Part A Controlled 5-Model Comparison (Single-Order, 64/16/20 split, 5 seeds)
python -B -m compare.part_a_comparison --smoke   # quick wiring check
python -B -m compare.part_a_comparison           # full 5-seed run -> compare/part_a_summary.json

# 4. Multi-Order D3QN Pipeline (Part A Ladder + Part B Search, 5-Seed Ensemble, Ablation & Robustness)
python d3qn/ladder_multi.py --seeds 5            # Part A 4-way ladder (DQN, Double DQN, Dueling DQN, D3QN)
python d3qn/d3qn_multi.py --stage A              # Architecture check on validation trips
python d3qn/d3qn_multi.py --stage B --n-configs 8  # Random search on validation trips -> best_config.json
python d3qn/d3qn_multi.py --stage C --seeds 5    # Train 5 seeds & score once on test scenarios
python d3qn/ensemble_multi.py --episodes 2000    # 5-seed ensemble + paired bootstrap 95% CIs
python d3qn/ablation_multi.py                    # One-at-a-time hyperparameter & feature ablation
python d3qn/robustness_multi.py                  # 10 rule sets, input masking & action-filter tests
python d3qn/retrain_study_multi.py --study grid  # Grid-size scalability (6x6 to 15x15)
python d3qn/retrain_study_multi.py --study calibrated  # Data-fitted clock retrain
python d3qn/error_analysis_multi.py              # Regret breakdown by customer count, traffic & weather
python d3qn/timing_multi.py                      # Per-decision CPU latency
python d3qn/split_study_multi.py --seeds 3       # Split-ratio & split-seed sensitivity

# 5. Single-Order Pipelines, Unseen-Map Check & Unit Tests
python q_learning/q_learning_pipeline.py
python dqn/dqn_v1.py
python dqn/dqn_v2.py
python dqn/dqn_v3.py
python dqn/part_b_refine.py
python d3qn/d3qn_pipeline.py
python evaluate.py
python evaluate_unseen_map.py
python analyze_errors.py
python -B -m unittest tests.test_core -v

# 6. Launch Interactive 3-Page Streamlit Dashboard
streamlit run app.py
```
