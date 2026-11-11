"""
Retraining study for the multi-order DDQN: grid size and a data-calibrated clock   (Part B: robustness / scalability)
====================================================================================================================
Everything here RETRAINS the tuned configuration (ddqn/result_multi/best_config.json), so it is slow; it saves
after every run and can be resumed by running the same command again.

  --study grid        grid sizes (default 6 8 10 12 15). The km-to-cell rule maps the same real distances onto the
                      new diagonal, so one cell stands for 19.92 / (2n-2) km: the table prints that number.
                      max_steps grows with the grid (10 * n) so a trip can still finish.
  --study calibrated  trains and tests with the clock fitted to the real data (CALIBRATED in robustness_multi.py)
                      next to the original rules, so the report can say whether the conclusions survive a more
                      realistic clock.

Each setting is scored against Greedy and the Planner on the SAME validation trips of its own environment
(use --test to score on the test trips instead, once, at the very end).

    python ddqn/retrain_study_multi.py --study grid --seeds 2
    python ddqn/retrain_study_multi.py --study calibrated --seeds 3
    python ddqn/retrain_study_multi.py --study grid --quick       # few-minute smoke test
Outputs (ddqn/result_multi/): retrain_<study>_runs.csv, retrain_<study>_summary.csv, retrain_<study>.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "ddqn"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ddqn_multi as dm
from env import load_orders
from env_multi import (MultiDeliveryEnv, evaluate_multi, greedy_policy_multi, planner_policy_multi, split_orders_3way)
from robustness_multi import CALIBRATED


def settings(study, sizes):
    if study == "grid":
        return {f"{n}x{n}": dict(grid_size=n, max_steps=10 * n, rules=None) for n in sizes}
    return {"Original rules": dict(grid_size=10, max_steps=100, rules=None),
            "Calibrated clock": dict(grid_size=10, max_steps=100, rules=CALIBRATED)}


def main():
    ap = argparse.ArgumentParser(description="Grid-size / calibrated-clock retraining study")
    ap.add_argument("--study", choices=["grid", "calibrated"], default="grid")
    ap.add_argument("--grid-sizes", type=int, nargs="+", default=[6, 8, 10, 12, 15])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=None, help="override training episodes")
    ap.add_argument("--test", action="store_true", help="score on TEST trips (do this once, at the end)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = BASE_DIR / "ddqn" / ("result_multi_quick" if args.quick else "result_multi")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = BASE_DIR / "ddqn" / "result_multi" / "best_config.json"
    best = json.loads(cfg_path.read_text()) if cfg_path.exists() else {**dm.DEFAULT_MODEL, "dueling": True, "double": True}
    if not cfg_path.exists():
        print("[note] best_config.json not found; using the default settings")
    base = dict(dm.BASE_CFG)
    n_eval, n_seeds = dm.EVAL_EPISODES, args.seeds
    if args.quick:
        base.update(episodes=60, warmup=200, eps_decay_episodes=30)
        n_eval, n_seeds = 20, 2
    if args.episodes:
        base["episodes"] = args.episodes

    orders = load_orders(BASE_DIR / "data" / "dataset_clean.csv")
    train, val, test = split_orders_3way(orders)
    eval_orders, offset = (test, 0) if args.test else (val, dm.VAL_SEED_OFFSET)
    max_km = MultiDeliveryEnv(train).max_km
    runs_path = out / f"retrain_{args.study}_runs.csv"
    done = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    sett = settings(args.study, args.grid_sizes)
    print(f"Study {args.study}: {list(sett)} | seeds {n_seeds} | train episodes {base['episodes']} | scored on {'TEST' if args.test else 'validation'}")

    refs = {}
    for name, kw in sett.items():
        n = kw["grid_size"]
        ev = MultiDeliveryEnv(eval_orders, max_km=max_km, **kw)
        refs[name] = {w: evaluate_multi(p, ev, n_eval, w, offset)["avg_reward"] for w, p in
                      {"Greedy": greedy_policy_multi, "Planner": planner_policy_multi}.items()}
        for s in range(n_seeds):
            if len(done) and ((done["setting"] == name) & (done["seed"] == s)).any():
                print(f"[skip] {name}, seed {s}")
                continue
            tr_env = MultiDeliveryEnv(train, max_km=max_km, **kw)
            t0 = time.time()
            print(f"\n[train] {name} | seed {s}")
            net, _, _ = dm.train_agent(tr_env, {**base, **best, "seed": s}, verbose=False)
            r = evaluate_multi(dm.make_policy(net), ev, n_eval, name, offset)
            rows.append({"setting": name, "seed": s, "km_per_cell": round(19.92 / (2 * n - 2), 2) if args.study == "grid" else round(19.92 / 18, 2),
                         "agent": r["avg_reward"], "greedy": refs[name]["Greedy"], "planner": refs[name]["Planner"],
                         "delivered_all": r["success_rate"], "hotspot_steps": r["avg_hotspot_steps"],
                         "train_time_s": round(time.time() - t0, 1)})
            pd.DataFrame(rows).to_csv(runs_path, index=False)
            print(f"    -> agent {r['avg_reward']:.2f} | greedy {refs[name]['Greedy']:.2f} | planner {refs[name]['Planner']:.2f} | {rows[-1]['train_time_s']:.0f}s")

    runs = pd.DataFrame(rows)
    g = runs.groupby("setting", sort=False).agg(runs=("seed", "count"), km_per_cell=("km_per_cell", "first"), agent=("agent", "mean"),
                                                sd=("agent", "std"), greedy=("greedy", "first"), planner=("planner", "first"),
                                                delivered_all=("delivered_all", "mean")).reset_index()
    g["gain_vs_greedy"] = g["agent"] - g["greedy"]
    g["gap_closed"] = g["gain_vs_greedy"] / (g["planner"] - g["greedy"]).clip(lower=1e-9)
    g.round(3).to_csv(out / f"retrain_{args.study}_summary.csv", index=False)
    print("\n=== summary (reward per trip; gap_closed = share of the Greedy-to-Planner gap the agent closes) ===")
    print(g.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(g))
    ax.plot(x, g["greedy"], "o--", color="grey", label="Greedy")
    ax.plot(x, g["planner"], "s:", color="black", label="Planner")
    ax.errorbar(x, g["agent"], yerr=g["sd"].fillna(0), fmt="o-", color="#1f6fd1", capsize=3, label="DDQN (mean of seeds)")
    ax.set_xticks(x); ax.set_xticklabels(g["setting"])
    ax.set_ylabel("Reward per trip"); ax.set_title(f"Retraining study: {args.study}")
    ax.legend(); ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out / f"retrain_{args.study}.png", dpi=160); plt.close(fig)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
