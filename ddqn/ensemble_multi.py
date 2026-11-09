"""
Ensemble of the saved DDQN seed models + PAIRED BOOTSTRAP confidence intervals  (multi-order environment)
=========================================================================================================
Part B asks you to combine models and to check the combination fairly. This script:
  1. loads the models saved by Stage C (ddqn_multi_seed0.pt, seed1, ...),
  2. builds an ensemble: every model's 4 Q-values are z-scored, then averaged (equal weights, no tuning),
  3. scores on the SAME 500 test trips: Greedy, Planner, each single seed, and the ensemble, in all three
     scenarios (normal / new order mid-trip / unseen hotspot map),
  4. puts a 95% confidence interval on every gap using a paired bootstrap over trips
     (paired = both models drive the same trips, so trip difficulty cancels out).

Run from the project root (after Stage C):
    python ddqn/ensemble_multi.py
    python ddqn/ensemble_multi.py --tag _default      # ensemble of the models saved with --tag _default

Outputs (in ddqn/result_multi/): ensemble_summary.csv, ensemble_paired_differences.csv, ensemble_comparison.png
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "ddqn"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import ddqn_multi as dm
from env_multi import evaluate_multi, greedy_policy_multi, planner_policy_multi, safe_argmax

METRICS = ["reward", "hotspot_steps", "late_min"]   # per-trip columns returned by evaluate_multi(return_episodes=True)


def make_ensemble_policy(nets):
    """Average of z-scored Q-values over all models, then the team's shared safe_argmax rule."""
    tracker = {"prev": None}

    def act(env):
        if env.steps == 0:
            tracker["prev"] = None
        x = torch.tensor(env.obs_v3()).unsqueeze(0)
        zs = []
        with torch.no_grad():
            for net in nets:
                net.eval()
                q = net(x).squeeze(0).numpy().astype(np.float64)
                zs.append((q - q.mean()) / (q.std() + 1e-8))
        action = safe_argmax(np.mean(zs, axis=0), env, tracker["prev"])
        tracker["prev"] = env.pos
        return action

    return act


def paired_bootstrap(diff, n_boot, rng):
    """Mean of per-trip differences with a 95% bootstrap CI over trips."""
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description="Ensemble of saved DDQN seeds + paired bootstrap CIs")
    ap.add_argument("--result-dir", default=str(BASE_DIR / "ddqn" / "result_multi"))
    ap.add_argument("--tag", default="", help="file suffix used when the models were saved (e.g. _default)")
    ap.add_argument("--episodes", type=int, default=500, help="test trips per scenario")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    out_dir = Path(args.result_dir)
    paths = sorted(out_dir.glob(f"ddqn_multi{args.tag}_seed*.pt"))
    if len(paths) < 2:
        print(f"[ERROR] need at least 2 saved models like ddqn_multi{args.tag}_seed0.pt in {out_dir}; found {len(paths)}. "
              "Run Stage C first.")
        sys.exit(1)
    nets = [dm.load_model(p)[0] for p in paths]
    print(f"Loaded {len(nets)} models: {[p.name for p in paths]}")

    ctx = dm.make_context(False)
    scenarios = {"normal test": ctx["test_env"], "new order mid-trip": ctx["dynamic_env"],
                 "unseen hotspot map": ctx["unseen_map_env"]}
    rng = np.random.default_rng(0)
    summary_rows, diff_rows, plot_data = [], [], {}

    for scen, env in scenarios.items():
        print(f"\n--- scenario: {scen} ---")
        per_trip = {}   # model name -> DataFrame of per-trip results, sorted by trip seed
        policies = {"Greedy": greedy_policy_multi, "Planner": planner_policy_multi,
                    "DDQN ensemble": make_ensemble_policy(nets)}
        for i, net in enumerate(nets):
            policies[f"DDQN single seed {i}"] = dm.make_policy(net)
        for name, pol in policies.items():
            res, df = evaluate_multi(pol, env, args.episodes, name, 0, return_episodes=True)
            per_trip[name] = df.sort_values("seed").reset_index(drop=True)
            summary_rows.append({"scenario": scen, "model": name, **{k: v for k, v in res.items() if k != "agent"}})
            print(f"  {name:<22} reward {res['avg_reward']:7.2f} | delivered {res['success_rate']:.1%} | hotspots {res['avg_hotspot_steps']:.2f}")

        single_names = [n for n in per_trip if n.startswith("DDQN single")]
        singles = {m: np.mean([per_trip[n][m].to_numpy() for n in single_names], axis=0) for m in METRICS}  # average single seed, per trip
        ens = {m: per_trip["DDQN ensemble"][m].to_numpy() for m in METRICS}
        greedy = {m: per_trip["Greedy"][m].to_numpy() for m in METRICS}
        planner = {m: per_trip["Planner"][m].to_numpy() for m in METRICS}

        comparisons = [
            ("Ensemble minus Greedy", ens, greedy),
            ("Planner minus Ensemble", planner, ens),
            ("Ensemble minus average single seed", ens, singles),
            ("Average single seed minus Greedy", singles, greedy),
            ("Planner minus average single seed", planner, singles),
        ]
        for label, a, b in comparisons:
            for m in METRICS:
                mean, lo, hi = paired_bootstrap(a[m] - b[m], args.n_bootstrap, rng)
                diff_rows.append({"scenario": scen, "comparison": label, "metric": m,
                                  "mean_difference": round(mean, 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                                  "ci_excludes_zero": bool(lo > 0 or hi < 0)})
        # data for the figure: mean reward with a bootstrap CI for each headline model
        plot_data[scen] = {}
        for label, arr in [("Greedy", greedy["reward"]), ("DDQN (avg single seed)", singles["reward"]),
                           ("DDQN ensemble", ens["reward"]), ("Planner", planner["reward"])]:
            mean, lo, hi = paired_bootstrap(arr, args.n_bootstrap, rng)
            plot_data[scen][label] = (mean, mean - lo, hi - mean)

    summary = pd.DataFrame(summary_rows)
    diffs = pd.DataFrame(diff_rows)
    summary.to_csv(out_dir / f"ensemble{args.tag}_summary.csv", index=False)
    diffs.to_csv(out_dir / f"ensemble{args.tag}_paired_differences.csv", index=False)
    print("\n=== paired differences with 95% bootstrap CI (reward; higher is better) ===")
    print(diffs[diffs["metric"] == "reward"].drop(columns="metric").to_string(index=False))
    print("\n=== paired differences, congested-cell steps per trip (lower is better) ===")
    print(diffs[diffs["metric"] == "hotspot_steps"].drop(columns="metric").to_string(index=False))

    # figure: reward per scenario, with 95% CI
    labels = ["Greedy", "DDQN (avg single seed)", "DDQN ensemble", "Planner"]
    colours = ["#9a9a9a", "#7fb2e5", "#1f6fd1", "#555555"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.2
    for j, lab in enumerate(labels):
        xs = np.arange(len(scenarios)) + (j - 1.5) * width
        vals = [plot_data[s][lab][0] for s in scenarios]
        errs = [[plot_data[s][lab][1] for s in scenarios], [plot_data[s][lab][2] for s in scenarios]]
        ax.bar(xs, vals, width, yerr=errs, capsize=3, label=lab, color=colours[j])
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(list(scenarios))
    ax.set_ylabel("Average reward per trip (95% bootstrap CI)")
    ax.set_title("Multi-order delivery: learned agents vs baselines on the same test trips")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"ensemble{args.tag}_comparison.png", dpi=160)
    plt.close(fig)
    print(f"\nSaved tables and figure to {out_dir}")


if __name__ == "__main__":
    main()
