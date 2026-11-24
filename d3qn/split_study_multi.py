"""
Split-sensitivity study (does the result depend on how the orders were divided?)
===============================================================================
The main experiments use ONE split of the cleaned orders (70/15/15, seed 42). This script repeats the headline
test with other random splits and other ratios. For each scheme it
  1. splits the orders into train / validation / test,
  2. trains the tuned D3QN (best_config.json) with several training seeds on that scheme's TRAIN orders,
  3. scores Greedy, the Planner, each single model and the model ensemble on that scheme's TEST trips
     (500 trips, seeds 0..499),
  4. reports the ensemble's gain over Greedy with a paired-bootstrap 95% interval, and the share of the
     Greedy-to-Planner gap that it closes.
If the gain stays about the same in every scheme, the result is not an accident of one split.

    python d3qn/split_study_multi.py --seeds 3                      # all schemes
    python d3qn/split_study_multi.py --seeds 3 --only 70_15_15_s42  # one scheme (for running in parallel)
    python d3qn/split_study_multi.py --summary                      # rebuild the table from saved rows
    python d3qn/split_study_multi.py --quick                        # smoke test

Output: d3qn/result_multi/split_study.csv  (one row per scheme; re-running skips finished schemes)
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "d3qn"))

import numpy as np
import pandas as pd
import torch

import d3qn_multi as dm
from ensemble_multi import make_ensemble_policy
from env import load_orders
from env_multi import (MultiDeliveryEnv, evaluate_multi, greedy_policy_multi, planner_policy_multi,
                       split_orders_3way)

# name: (validation fraction, test fraction, split seed)
SCHEMES = {
    "70_15_15_s42": (0.15, 0.15, 42),   # the split used everywhere else in the report
    "70_15_15_s1": (0.15, 0.15, 1),     # same ratio, different random split
    "70_15_15_s2": (0.15, 0.15, 2),
    "80_10_10_s42": (0.10, 0.10, 42),   # more training data
    "60_20_20_s42": (0.20, 0.20, 42),   # less training data
}


def boot(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return diff.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def run_scheme(name, orders, best, seeds, base, n_test):
    vf, tf, ss = SCHEMES[name]
    train, val, test = split_orders_3way(orders, val_frac=vf, test_frac=tf, seed=ss)
    factory = lambda: MultiDeliveryEnv(train)
    max_km = factory().max_km
    test_env = MultiDeliveryEnv(test, max_km=max_km)
    nets = []
    t0 = time.time()
    for s in range(seeds):
        cfg = {**base, **best, "seed": s}
        net, _, _ = dm.train_agent(factory(), cfg, verbose=False)
        nets.append(net)
        print(f"  {name}: trained seed {s} ({time.time() - t0:.0f}s)", flush=True)
    _, g = evaluate_multi(greedy_policy_multi, test_env, n_test, "Greedy", 0, return_episodes=True)
    _, p = evaluate_multi(planner_policy_multi, test_env, n_test, "Planner", 0, return_episodes=True)
    singles = []
    for net in nets:
        r, _ = evaluate_multi(dm.make_policy(net), test_env, n_test, "single", 0, return_episodes=True)
        singles.append(r["avg_reward"])
    re, e = evaluate_multi(make_ensemble_policy(nets), test_env, n_test, "ensemble", 0, return_episodes=True)
    gm, glo, ghi = boot(e["reward"].values - g["reward"].values)
    gap = (p["reward"].mean() - g["reward"].mean())
    return {"scheme": name, "train_orders": len(train), "val_orders": len(val), "test_orders": len(test),
            "training_seeds": seeds, "greedy": round(g["reward"].mean(), 2), "planner": round(p["reward"].mean(), 2),
            "single_model_mean": round(float(np.mean(singles)), 2), "ensemble": round(e["reward"].mean(), 2),
            "gain_over_greedy": round(gm, 2), "ci_low": round(glo, 2), "ci_high": round(ghi, 2),
            "gap_closed_pct": round(100 * gm / gap, 0), "ensemble_all_delivered": round(re["success_rate"], 3),
            "train_time_s": round(time.time() - t0, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", choices=list(SCHEMES), default=None)
    ap.add_argument("--result-dir", default=str(BASE_DIR / "d3qn" / "result_multi"))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = Path(a.result_dir); out.mkdir(parents=True, exist_ok=True)
    csv = out / ("split_study_quick.csv" if a.quick else "split_study.csv")
    if a.summary:
        print(pd.read_csv(csv).to_string(index=False)); return

    torch.set_num_threads(1)
    orders = load_orders(BASE_DIR / "data" / "dataset_clean.csv")
    best = json.loads((BASE_DIR / "d3qn" / "result_multi" / "best_config.json").read_text())
    base = {**dm.BASE_CFG}
    n_test = 500
    if a.quick:
        base.update(episodes=300, eps_decay_episodes=150, warmup=200); n_test = 60
    done = set(pd.read_csv(csv)["scheme"]) if csv.exists() else set()
    for name in SCHEMES:
        if a.only and name != a.only:
            continue
        if name in done:
            print(f"skip {name} (saved)"); continue
        print(f"scheme {name}", flush=True)
        row = run_scheme(name, orders, best, a.seeds, base, n_test)
        pd.DataFrame([row]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        print(row, flush=True)
    print(pd.read_csv(csv).to_string(index=False))


if __name__ == "__main__":
    main()
