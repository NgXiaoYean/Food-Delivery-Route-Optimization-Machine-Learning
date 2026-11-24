# Food Delivery Route Optimization using Reinforcement Learning

An end-to-end Reinforcement Learning system optimizing urban food delivery routes under dynamic traffic, weather, and congestion hotspots, grounded in **40,482 cleaned delivery orders** (from 45,584 raw records) from the [Kaggle Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset?select=Zomato+Dataset.csv).

See **[`docs/PROTOCOL.md`](docs/PROTOCOL.md)** for the complete one-page experiment protocol mapping every report table to its script, data split, seed count, timeout target rule, and hotspot map seed.

---

## 👥 3-Algorithm Architecture (Mapped to Team Members & Assessment Rubric)

| Module / Folder | Algorithm | Methodology | Part A (Model Selection & Iteration) | Part B (Optimization, Regularization & Ensemble) |
| :--- | :--- | :--- | :--- | :--- |
| **`q_learning/`** | **1. Tabular Q-Learning** | Model-Free Tabular TD(0) | Raw 40k Discrete States (`v1`, 5.2% success) $\rightarrow$ Refined Relative + Hotspot States (`v2`, 86.4% success on same map) | LR ($\alpha$), Discount ($\gamma$), & L2 Q-Table Regularization (`weight_decay`) $\rightarrow$ **92.4% delivered (same map, 80/20 split, 1 seed)** / **57.8% ± 14.0% delivered (unseen map, 64/16/20 split, 5 seeds in `compare/part_a_comparison.py`)** |
| **`dqn/`** | **2. Deep Q-Network (DQN)** | Value-Based Deep RL (MLP + Replay + Target Net) | `v1` Raw 8D (7.2%) $\rightarrow$ `v2` +Direction & Scaling (**100%**) $\rightarrow$ `v3` +Hotspot Radar (98.8%) | Hyperparameter search (Double-Q update), Missing-Data Handling Study, & Stress Testing across 4 extreme scenarios (`dqn_dynamic/` adds a dynamic-traffic v4 extension) |
| **`d3qn/`** | **3. Dueling Double DQN (D3QN)** | Dual-Stream Value $V(s)$ + Advantage $A(s,a)$ & Double Q-Update | Controlled 4-way DQN-family comparison (`DQN`, `Double DQN`, `Dueling DQN`, `D3QN` in `compare/part_a_comparison.py` & `d3qn/ladder_multi.py`) | Multi-order (`env_multi.py`, 1–3 customers, 70/15/15 split): Validation Random Search, 5-Seed Ensemble with Paired Bootstrap 95% CIs, One-at-a-Time Ablation, 10-Rule Sensitivity, Calibrated Clock & Grid-Size Scalability |
| **`evaluate.py` & `app.py`** | **Ensembles & Streamlit Prototype** | Standardized Q-Value Fusion | Automated 8-metric single-order & multi-order benchmarks against `Random`, `Greedy`, and `Dijkstra Planner` | Single-order 3-family ensemble (`QL + DQN + D3QN`), Multi-order 5-seed `D3QN` ensemble, & 3-Page **Streamlit Prototype UI** (`app.py`, `pages/2_Multi_Order_D3QN.py`, `pages/3_Data_and_Experiments.py`) |

> **Note on the two Q-Learning delivery rates (`92.4%` vs. `57.8%`):**
> - **`92.4%` (Same Map, `q_learning/q_learning_pipeline.py`):** Trained and evaluated on the same fixed hotspot map (`map_seed=0`) using the historical 80/20 split (1 training seed).
> - **`57.8% ± 14.0%` (Unseen Map, `compare/part_a_comparison.py`):** Trained on `map_seed=0` (60,000 steps, 64/16/20 split, 5 seeds) and evaluated on an unseen test hotspot map (`map_seed=202`).
>
> **Note on Single-Order vs. Multi-Order DQN-Family Results:**
> - On **Single-Order (`compare/part_a_comparison.py`, 5 seeds)**, `DQN` (`99.8%` delivered, `57.5%` on-time proxy), `Double DQN` (`100.0%`, `56.6%`), `Dueling DQN` (`100.0%`, `57.6%`), and `D3QN` (`100.0%`, `57.2%`) are statistically tied (mean rewards `32.1–32.8`).
> - On **Multi-Order (`d3qn/`, 1–3 customers, 2,000 test trips)**, the 5-seed **D3QN Ensemble** beats `Greedy` by **`+4.35`** (`95% CI [4.09, 4.61]`, normal), **`+5.25`** (`[4.67, 5.81]`, new order mid-trip), and **`+3.92`** (`[3.69, 4.16]`, unseen map), closing **74%–84%** of the gap to the full-map Dijkstra Planner.

---

## 📂 Project Structure

```text
├── data/
│   ├── zomato_dataset.csv             # Raw Kaggle Zomato dataset (45,584 rows)
│   ├── dataset_clean.csv              # Cleaned delivery dataset (40,482 valid orders)
│   ├── cleaning_log.csv               # Step-by-step data cleaning audit log
│   └── data_checks.csv                # Empirical verification of negative-GPS & dropped-row assumptions
├── docs/
│   └── PROTOCOL.md                    # One-page protocol mapping every report table to its script & split
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
├── dqn/                               # Algorithm 2: Deep Q-Network (Historical Single-Order v1-v3 + Split Helpers)
│   ├── dqn_v1.py                      # Iteration 1: Raw 8D coordinates
│   ├── dqn_v2.py                      # Iteration 2: +Relative displacement & reward scaling
│   ├── dqn_v3.py                      # Iteration 3: +4 neighboring hotspot sensors (15D)
│   ├── experiments.py                 # 64/16/20 3-way split, SHA-256 manifest & distance scale helpers
│   ├── learning_demo.py               # Interactive in-app teaching animation for Tab 1 of app.py
│   └── result/                        # Saved weights (.pt), curves (.png) & Part B CSVs
├── dqn_dynamic/                       # Extension: Dynamic-Traffic Standard DQN (v4), successive halving & 24 unit tests
├── exploration/
│   └── zomato_exploration/            # 16 EDA distributions, correlations & dataset report
├── result/
│   ├── master_all_models_results.csv  # Historical single-order 8-metric scoreboard (80/20 split, same map)
│   ├── master_comparison_chart.png    # 4-panel multi-algorithm visual benchmark
│   ├── missing_data_comparison.csv    # Part B.1 missing-data strategy comparison
│   └── error_analysis/                # Single-order breakdown tables & charts by traffic, weather & distance
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
python d3qn/ladder_multi.py --seeds 3            # Part A 4-way ladder (DQN, Double DQN, Dueling DQN, D3QN)
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

# 5. Historical Single-Order Pipelines & Extension Unit Tests
python q_learning/q_learning_pipeline.py
python dqn/dqn_v1.py
python dqn/dqn_v2.py
python dqn/dqn_v3.py
python d3qn/d3qn_pipeline.py
python evaluate.py
python analyze_errors.py
python -B -m unittest dqn_dynamic.test_v4

# 6. Launch Interactive 3-Page Streamlit Dashboard
streamlit run app.py
```
