# Food Delivery Route Optimization using Reinforcement Learning

An end-to-end Reinforcement Learning system optimizing urban food delivery routes under dynamic traffic, weather, and congestion hotspots, grounded in **40,482 real delivery orders** from the [Kaggle Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset?select=Zomato+Dataset.csv).

---

## 👥 3-Algorithm Architecture (Mapped to Team Members & Assessment Rubric)

| Module / Folder | Algorithm | Methodology | Part A (Model Selection & Iteration) | Part B (Optimization, Regularization & Ensemble) |
| :--- | :--- | :--- | :--- | :--- |
| **`q_learning/`** | **1. Tabular Q-Learning** | Model-Free Tabular TD(0) | Raw 40k Discrete States (`v1`, 5.2% success) $\rightarrow$ Refined Relative + Hotspot States (`v2`, 86.4% success) | LR ($\alpha$), Discount ($\gamma$), & L2 Q-Table Regularization (`weight_decay`) $\rightarrow$ **92.4% success** |
| **`dqn/`** | **2. Deep Q-Network (DQN)** | Value-Based Deep RL (MLP + Replay + Target Net) | `v1` Raw 8D (7.2%) $\rightarrow$ `v2` +Direction & Scaling (**100%**) $\rightarrow$ `v3` +Hotspot Radar (98.8%) | Hyperparameter search, Missing-Data Handling Study, & Stress Testing across 4 extreme scenarios |
| **`ddqn/`** | **3. Dueling Double DQN (D3QN)** | Dual-Stream Value $V(s)$ + Advantage $A(s,a)$ & Double Q-Update | Vanilla Double DQN (MLP) vs. **Dueling Double DQN** (fixes Q-value overestimation around hotspots) | Decoupled L2 Weight Decay (`AdamW`), `Dropout`, & Polyak Soft Target Updates ($\tau$) $\rightarrow$ **100% success, halves hotspot steps** |
| **`evaluate.py` & `app.py`** | **Cross-Algorithm Ensemble** | Standardized Q-Value Fusion (`QL + DQN + D3QN`) | Automated 8-metric benchmark against `Random`, `Greedy`, and `Dijkstra Planner` | Combines all 3 RL families + Interactive 5-Tab **Streamlit Prototype UI** |

---

## 📂 Project Structure

```text
├── data/
│   ├── zomato_dataset.csv             # Raw Kaggle Zomato dataset (45,584 rows)
│   ├── dataset_clean.csv              # Cleaned delivery dataset (40,482 valid orders)
│   └── cleaning_log.csv               # Step-by-step data cleaning audit log
├── exploration/
│   └── zomato_exploration/            # 16 EDA distributions, correlations & dataset report
├── q_learning/                        # Algorithm 1: Tabular Q-Learning
│   ├── q_learning_pipeline.py         # Part A (v1 vs v2 states) + Part B (L2 reg & tuning)
│   └── result/                        # Saved Q-table (.pkl), curves (.png) & tuning CSVs
├── dqn/                               # Algorithm 2: Deep Q-Network (DQN)
│   ├── dqn_v1.py                      # Iteration 1: Raw 8D coordinates
│   ├── dqn_v2.py                      # Iteration 2: +Relative displacement & reward scaling
│   ├── dqn_v3.py                      # Iteration 3: +4 neighboring hotspot sensors
│   ├── part_b_refine.py               # Hyperparameter search, ensembling & 4-scenario stress test
│   └── result/                        # Saved weights (.pt), curves (.png) & Part B CSVs
├── ddqn/                              # Algorithm 3: Dueling Double DQN (D3QN)
│   ├── ddqn_pipeline.py               # Part A (MLP vs Dueling) + Part B (AdamW L2 & Dropout reg)
│   └── result/                        # Saved D3QN weights (.pt), curves (.png) & tuning CSVs
├── result/
│   ├── master_all_models_results.csv  # Consolidated 8-metric scoreboard across all models
│   ├── master_comparison_chart.png    # 4-panel multi-algorithm visual benchmark
│   ├── missing_data_comparison.csv    # Part B.1 missing-data imputation strategy comparison
│   └── error_analysis/                # Breakdown tables & charts by traffic, weather, distance & hotspots
├── clean_data.py                      # Data cleaning pipeline
├── missing_data_study.py              # Empirical comparison of 3 missing-value strategies
├── eda.py                             # Automated Exploratory Data Analysis
├── env.py                             # Shared 10x10 Gymnasium Delivery Environment
├── shared.py                          # Shared 80/20 split & 8-metric fair evaluation protocol
├── evaluate.py                        # Automated master benchmark & Cross-Algorithm Ensemble
├── analyze_errors.py                  # Error & hotspot analysis across all 3 RL families
├── app.py                             # Interactive 5-Tab Streamlit Visual Prototype
└── requirements.txt                   # Dependencies
```

---

## 🚀 Execution Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Data Preparation, EDA & Missing-Data Strategy Study
python clean_data.py
python eda.py
python missing_data_study.py

# 3. Train & Optimize All 3 RL Algorithms (Part A + Part B)
python q_learning/q_learning_pipeline.py
python dqn/dqn_v1.py
python dqn/dqn_v2.py
python dqn/dqn_v3.py
python dqn/part_b_refine.py
python ddqn/ddqn_pipeline.py

# 4. Run Automated Master Evaluation & Error Analysis
python evaluate.py
python analyze_errors.py

# 5. Launch Interactive Visual Prototype Dashboard
streamlit run app.py
```
