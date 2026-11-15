"""
Ablation study for the multi-order DDQN  (Part A "which factors matter" + Part B "which setting helped")
=======================================================================================================
Starting from the TUNED configuration (ddqn/result_multi/best_config.json) we change ONE thing at a time,
retrain, and score on the VALIDATION trips (never the test set):

  Hyperparameters (put one tuned setting back to its default)
    - gamma 0.90 -> 0.95
    - hidden 256 -> 128
    - tau   0.02 -> 0.005
  Input features (hide one group of inputs from the network by setting them to 0)
    - no distance feature : the Manhattan distance to each customer
    - no hotspot flags    : the 4 flags telling whether a neighbouring cell is congested
    - Euclidean distance  : the distance input is straight-line instead of Manhattan (tests the Manhattan choice;
                            the row/column offsets and the grid moves are unchanged)

Each variant is trained with several seeds so we can see whether a change is bigger than training noise.

Run from the project root (takes hours on a laptop; run it overnight, it saves after every run, and
re-running the same command continues where it stopped):
    python ddqn/ablation_multi.py                 # everything, 2 seeds per variant
    python ddqn/ablation_multi.py --group feature # only the feature ablation
    python ddqn/ablation_multi.py --seeds 3       # more seeds = more reliable
    python ddqn/ablation_multi.py --quick         # a few-minute smoke test

Outputs (ddqn/result_multi/): ablation_val_runs.csv, ablation_summary.csv, ablation.png
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
from env_multi import (K_MAX, MultiDeliveryEnv, evaluate_multi, greedy_policy_multi, planner_policy_multi)

DISTANCE_IDX = [2 + 4 * j + 2 for j in range(K_MAX)]   # Manhattan-distance slot of each customer in the observation
HOTSPOT_IDX = [18, 19, 20, 21]                          # 4 neighbour-hotspot flags appended by obs_v3()


class MaskedEnv(MultiDeliveryEnv):
    """Same environment, but chosen input numbers are replaced by 0 so the network cannot use them."""

    def __init__(self, orders, mask_idx=(), euclid=False, **kw):
        super().__init__(orders, **kw)
        self.mask_idx = list(mask_idx)
        self.euclid = euclid

    def obs_v3(self):
        v = super().obs_v3()
        if self.euclid:   # replace Manhattan distance by straight-line distance (same scale: divided by its maximum)
            n = self.n - 1
            for j, i in enumerate(DISTANCE_IDX):
                if v[2 + 4 * j + 3] > 0:
                    dr, dc = v[2 + 4 * j] * n, v[2 + 4 * j + 1] * n
                    v[i] = np.sqrt(dr * dr + dc * dc) / (np.sqrt(2.0) * n)
        v[self.mask_idx] = 0.0
        return v


def get_variants(best, group):
    """name -> (changes to the config, indices of inputs to hide, use Euclidean distance)."""
    default = dm.DEFAULT_MODEL
    v = {"Tuned (reference)": ({}, [], False)}
    if group in ("hyper", "all"):
        for key in ["gamma", "hidden", "tau"]:
            v[f"{key} back to default ({default[key]})"] = ({key: default[key]}, [], False)
    if group in ("feature", "all"):
        v["No distance feature"] = ({}, DISTANCE_IDX, False)
        v["No hotspot flags"] = ({}, HOTSPOT_IDX, False)
        v["Euclidean instead of Manhattan distance"] = ({}, [], True)
    return v


def main():
    ap = argparse.ArgumentParser(description="One-at-a-time ablation of the multi-order DDQN (validation set)")
    ap.add_argument("--group", choices=["hyper", "feature", "all"], default="all")
    ap.add_argument("--seeds", type=int, default=2, help="training seeds per variant")
    ap.add_argument("--episodes", type=int, default=None, help="override training episodes")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = BASE_DIR / "ddqn" / ("result_multi_quick" if args.quick else "result_multi")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = BASE_DIR / "ddqn" / "result_multi" / "best_config.json"
    if cfg_path.exists():
        best = json.loads(cfg_path.read_text())
    else:
        print("[note] best_config.json not found; using the default settings as the reference")
        best = {**dm.DEFAULT_MODEL, "dueling": True, "double": True}

    base = dict(dm.BASE_CFG)
    n_eval, n_seeds = dm.EVAL_EPISODES, args.seeds
    if args.quick:
        base.update(episodes=60, warmup=200, eps_decay_episodes=30)
        n_eval, n_seeds = 20, 2
    if args.episodes:
        base["episodes"] = args.episodes

    ctx = dm.make_context(args.quick)
    train_env0, val_env0 = ctx["factory"](), ctx["val_env"]
    runs_path = out / "ablation_val_runs.csv"
    done = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    variants = get_variants(best, args.group)
    print(f"Variants: {list(variants)} | seeds per variant: {n_seeds} | train episodes: {base['episodes']}")

    rows = done.to_dict("records") if len(done) else []
    for name, (changes, mask, euclid) in variants.items():
        for s in range(n_seeds):
            if len(done) and ((done["variant"] == name) & (done["seed"] == s)).any():
                print(f"[skip] {name}, seed {s} (already in {runs_path.name})")
                continue
            cfg = {**base, **best, **changes, "seed": s}
            train_env = MaskedEnv(train_env0.orders, mask_idx=mask, euclid=euclid, max_km=train_env0.max_km)
            val_env = MaskedEnv(val_env0.orders, mask_idx=mask, euclid=euclid, max_km=train_env0.max_km)
            t0 = time.time()
            print(f"\n[train] {name} | seed {s} | changes {changes or 'none'} | hidden inputs {mask or 'none'}")
            net, _, _ = dm.train_agent(train_env, cfg, verbose=False)
            res = evaluate_multi(dm.make_policy(net), val_env, n_eval, name, dm.VAL_SEED_OFFSET)
            row = {"variant": name, "seed": s, "val_reward": res["avg_reward"], "delivered_all": res["success_rate"],
                   "on_time_order_rate": res["on_time_order_rate"], "hotspot_steps": res["avg_hotspot_steps"],
                   "train_time_s": round(time.time() - t0, 1)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(runs_path, index=False)   # saved after every run
            print(f"    -> val reward {row['val_reward']:.2f} | all delivered {row['delivered_all']:.1%} | {row['train_time_s']:.0f}s")

    runs = pd.DataFrame(rows)
    ref = runs[runs["variant"] == "Tuned (reference)"]["val_reward"]
    g = runs.groupby("variant").agg(runs=("seed", "count"), val_reward=("val_reward", "mean"), sd=("val_reward", "std"),
                                    delivered_all=("delivered_all", "mean"), hotspot_steps=("hotspot_steps", "mean")).reset_index()
    if len(ref):
        g["change_vs_tuned"] = g["val_reward"] - ref.mean()
        # "bigger than noise" only if the two groups of seeds do not overlap (a deliberately strict rule)
        lo_hi = runs.groupby("variant")["val_reward"].agg(["min", "max"])
        g["clearly_different"] = [bool(v != "Tuned (reference)" and (lo_hi.loc[v, "max"] < ref.min() or lo_hi.loc[v, "min"] > ref.max()))
                                  for v in g["variant"]]
    g = g.round(3)
    g.to_csv(out / "ablation_summary.csv", index=False)
    print("\n=== ablation summary (validation set; reward per trip) ===")
    print(g.to_string(index=False))

    gr = evaluate_multi(greedy_policy_multi, val_env0, n_eval, "Greedy", dm.VAL_SEED_OFFSET)["avg_reward"]
    pl = evaluate_multi(planner_policy_multi, val_env0, n_eval, "Planner", dm.VAL_SEED_OFFSET)["avg_reward"]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(g) + 2))
    order = g.sort_values("val_reward")["variant"].tolist()
    for i, name in enumerate(order):
        pts = runs[runs["variant"] == name]["val_reward"]
        ax.barh(i, pts.mean(), color="#1f6fd1" if name == "Tuned (reference)" else "#9db8d9", xerr=pts.std() if len(pts) > 1 else 0, capsize=3)
        ax.scatter(pts, [i] * len(pts), color="black", s=14, zorder=3)
    ax.axvline(gr, color="grey", linestyle="--", label=f"Greedy ({gr:.1f})")
    ax.axvline(pl, color="black", linestyle=":", label=f"Planner ({pl:.1f})")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    lo = min(gr, runs["val_reward"].min()) - 3
    ax.set_xlim(lo, pl + 2)
    ax.set_xlabel("Validation reward per trip (bars: mean over seeds, dots: single runs)")
    ax.set_title("Ablation: change one thing at a time")
    ax.legend(loc="lower right"); ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out / "ablation.png", dpi=160); plt.close(fig)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
