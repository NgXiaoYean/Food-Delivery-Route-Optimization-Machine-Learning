"""
Architecture ladder (Part A evidence for "why D3QN?")
=====================================================
Four networks that differ ONLY in two switches, trained with identical settings and the same seeds, then scored on
the same VALIDATION trips (never the test set):

    DQN              double=False  dueling=False
    Double DQN       double=True   dueling=False
    Dueling DQN      double=False  dueling=True
    D3QN             double=True   dueling=True

Comparing the rungs shows what each idea adds on its own:
    Double effect   = Double DQN  - DQN        and  D3QN - Dueling DQN
    Dueling effect  = Dueling DQN - DQN        and  D3QN - Double DQN

Run from the project root. Each (variant, seed) run is saved as soon as it finishes, and re-running the same
command skips finished runs, so it can be stopped and continued.
    python d3qn/ladder_multi.py --seeds 3                 # all four variants, seeds 0..2, default settings
    python d3qn/ladder_multi.py --seeds 3 --only D3QN     # one variant only
    python d3qn/ladder_multi.py --seeds 3 --settings tuned  # use best_config.json for every variant (favours D3QN)
    python d3qn/ladder_multi.py --summary                 # only rebuild the tables from the saved runs
    python d3qn/ladder_multi.py --quick                   # tiny smoke test (a few minutes)

Outputs (d3qn/result_multi/ or --result-dir): ladder_<settings>_runs.csv, ladder_<settings>_episodes.csv,
ladder_<settings>_summary.csv, ladder_<settings>_paired.csv
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
from env_multi import evaluate_multi

VARIANTS = {
    "DQN": dict(double=False, dueling=False),
    "Double DQN": dict(double=True, dueling=False),
    "Dueling DQN": dict(double=False, dueling=True),
    "D3QN": dict(double=True, dueling=True),
}


def model_settings(kind, out_dir):
    if kind == "tuned":
        best = json.loads((out_dir / "best_config.json").read_text())
        return {k: best[k] for k in ["lr", "gamma", "hidden", "weight_decay", "dropout", "tau"]}
    return dict(dm.DEFAULT_MODEL)


def paired_bootstrap(a, b, n=5000, seed=0):
    """Mean of (a-b) over matched trips with a 95% percentile interval."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def build_tables(runs, eps, out_dir, tag):
    runs = pd.read_csv(runs)
    summ = (runs.groupby("variant")[["avg_reward", "success_rate", "avg_hotspot_steps", "train_time_s"]]
            .agg(["mean", "std", "count"]).round(3))
    summ.to_csv(out_dir / f"ladder_{tag}_summary.csv")
    print("\n=== validation, mean and std over seeds ===\n" + summ.to_string())

    e = pd.read_csv(eps)
    # average each trip over the seeds of that variant, so the bootstrap resamples TRIPS (same trips for every variant)
    per_trip = e.groupby(["variant", "seed"])["reward"].mean().unstack(0)
    rows = []
    pairs = [("Double DQN", "DQN", "Double effect (no dueling)"),
             ("D3QN", "Dueling DQN", "Double effect (with dueling)"),
             ("Dueling DQN", "DQN", "Dueling effect (no double)"),
             ("D3QN", "Double DQN", "Dueling effect (with double)"),
             ("D3QN", "DQN", "Both together")]
    for a, b, label in pairs:
        if a in per_trip and b in per_trip:
            m, lo, hi = paired_bootstrap(per_trip[a].values, per_trip[b].values)
            rows.append({"comparison": f"{a} minus {b}", "what_it_tests": label, "mean_diff": round(m, 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                         "interval_excludes_zero": bool(lo > 0 or hi < 0)})
    paired = pd.DataFrame(rows)
    paired.to_csv(out_dir / f"ladder_{tag}_paired.csv", index=False)
    print("\n=== paired bootstrap over validation trips (5,000 resamples) ===\n" + paired.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", choices=list(VARIANTS), default=None)
    ap.add_argument("--settings", choices=["default", "tuned"], default="default")
    ap.add_argument("--result-dir", default=str(BASE_DIR / "d3qn" / "result_multi"))
    ap.add_argument("--episodes", type=int, default=None, help="override training episodes (testing only)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    out = Path(a.result_dir); out.mkdir(parents=True, exist_ok=True)
    tag = a.settings + ("_quick" if a.quick else "")
    runs_p, eps_p = out / f"ladder_{tag}_runs.csv", out / f"ladder_{tag}_episodes.csv"
    if a.summary:
        build_tables(runs_p, eps_p, out, tag); return

    torch.set_num_threads(1)
    ctx = dm.make_context(a.quick)
    base = {**dm.BASE_CFG}
    if a.quick:
        base.update(episodes=300, eps_decay_episodes=150, warmup=200)
    if a.episodes:
        base["episodes"] = a.episodes
    n_eval = 60 if a.quick else dm.EVAL_EPISODES
    settings = model_settings(a.settings, out)
    print(f"settings = {a.settings}: {settings}\nepisodes = {base['episodes']}, eval trips = {n_eval}", flush=True)

    done = set()
    if runs_p.exists():
        r = pd.read_csv(runs_p); done = set(zip(r["variant"], r["seed"]))
    for seed in range(a.seeds):
        for name, flags in VARIANTS.items():
            if a.only and name != a.only:
                continue
            if (name, seed) in done:
                print(f"skip {name} seed {seed} (already saved)"); continue
            cfg = {**base, **settings, **flags, "seed": seed}
            t0 = time.time()
            net, _, _ = dm.train_agent(ctx["factory"](), cfg, verbose=False)
            res, ep_df = evaluate_multi(dm.make_policy(net), ctx["val_env"], n_eval, name, dm.VAL_SEED_OFFSET, return_episodes=True)
            row = {"variant": name, "seed": seed, "train_time_s": round(time.time() - t0, 1),
                   **{k: v for k, v in res.items() if k != "agent"}}
            pd.DataFrame([row]).to_csv(runs_p, mode="a", header=not runs_p.exists(), index=False)
            ep_df.assign(variant=name, train_seed=seed).to_csv(eps_p, mode="a", header=not eps_p.exists(), index=False)
            print(f"{name:<12} seed {seed}: val reward {res['avg_reward']:.2f} | all delivered {res['success_rate']:.1%} "
                  f"| {row['train_time_s']:.0f}s", flush=True)
    if runs_p.exists():
        build_tables(runs_p, eps_p, out, tag)


if __name__ == "__main__":
    main()
