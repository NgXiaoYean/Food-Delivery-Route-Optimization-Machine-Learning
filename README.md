# Food Delivery Route Optimization using Reinforcement Learning

A Machine Learning project optimizing urban food delivery routes under real-world traffic and weather conditions using **Tabular Q-Learning**, **Deep Q-Networks (DQN)**, and **Double DQN / Ensemble Refinement**, grounded in the [Zomato Delivery Operations Analytics Dataset](https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset?select=Zomato+Dataset.csv).

---

## 📂 Project Structure

```text
├── data/
│   ├── zomato_dataset.csv         # Raw Kaggle dataset (45,584 delivery records)
│   ├── dataset_clean.csv          # Cleaned dataset (40,482 valid orders)
│   └── cleaning_log.csv           # Audit trail of data cleaning steps
├── exploration/
│   └── zomato_exploration/        # EDA visualizations, correlations, and dataset report
├── dqn/
│   ├── dqn_v1.py                  # Part A Iteration 1: Raw 8D state (7.2% success)
│   ├── dqn_v2.py                  # Part A Iteration 2: +Relative direction & reward scaling (100% success)
│   ├── dqn_v3.py                  # Part A Iteration 3: +4 local hotspot sensors (hotspot avoidance)
│   ├── part_b_refine.py           # Part B: Hyperparameter tuning, Double DQN, Ensembling & Stress Testing
│   └── result/                    # Saved weights (.pt), learning curves (.png), and metrics (.csv)
├── result/
│   └── error_analysis/            # Breakdown tables and charts by traffic, weather, distance, and hotspots
├── clean_data.py                  # Data cleaning and preprocessing pipeline
├── eda.py                         # Exploratory Data Analysis script
├── env.py                         # Shared Gymnasium-compatible Delivery Environment
├── shared.py                      # Shared 80/20 train-test split and fair evaluation protocol
├── test_baselines.py              # Quick sanity check for Random and Greedy baselines
├── evaluate.py                    # Benchmark evaluation across all baselines, DQN iterations, DDQN & Ensemble
├── analyze_errors.py              # Error and hotspot analysis across traffic, weather, and distance bands
├── app.py                         # Interactive Streamlit Visual Prototype (10x10 Route Simulator & Dashboard)
└── requirements.txt               # Python dependencies
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Clean raw dataset & generate EDA (already saved in data/ and exploration/)
python clean_data.py
python eda.py

# 3. Part A: Train DQN iterations & run error analysis
python dqn/dqn_v1.py
python dqn/dqn_v2.py
python dqn/dqn_v3.py
python analyze_errors.py

# 4. Part B: Run automated hyperparameter tuning, Double DQN, Ensembling & Stress Tests
python dqn/part_b_refine.py

# 5. Evaluate all baselines & trained models on 500 unseen test orders
python evaluate.py

# 6. Launch Interactive Visual Prototype UI
streamlit run app.py
```
