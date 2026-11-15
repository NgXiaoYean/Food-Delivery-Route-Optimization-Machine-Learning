"""
Error analysis for the multi-order DDQN (Part A item 4: "examine types of errors or inefficiencies")
===================================================================================================
No training needed: it loads the models saved by Stage C and answers
  1. WHERE does the learned agent lose reward compared with the oracle Planner?
     (by number of customers, traffic level, weather level)
  2. Which individual trips are the worst, and what do they have in common?
  3. Which trips were NOT fully delivered (single seeds), and why might that be?
  4. Is the loss mostly lateness or detours through congested cells?

"Regret" = Planner reward minus agent reward on the SAME trip (higher = agent did worse).

Run from the project root (after Stage C):
    python ddqn/error_analysis_multi.py
Outputs (ddqn/result_multi/): error_regret_by_group.csv, error_worst_trips.csv, error_failed_trips.csv,
                              error_summary.txt, error_regret_by_group.png
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

import ddqn_multi as dm
from ensemble_multi import make_ensemble_policy
from env_multi import evaluate_multi, greedy_policy_multi, planner_policy_multi

TRAFFIC = {0: "Low", 1: "Medium", 2: "High", 3: "Jam"}
WEATHER = {0: "Clear/Cloudy", 1: "Windy", 2: "Fog/Sandstorm", 3: "Stormy"}


def trip_features(env, episodes):
    """Static description of each trip (read right after reset, before any move)."""
    rows = []
    for s in range(episodes):
        env.reset(seed=s)
        rows.append({"seed": s, "traffic": TRAFFIC[env.traffic], "weather": WEATHER[env.weather],
                     "k": env.k, "arrival_step": env.arrival_step if env.arrival_step is not None else -1})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Error analysis of the multi-order DDQN ensemble")
    ap.add_argument("--result-dir", default=str(BASE_DIR / "ddqn" / "result_multi"))
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--min-group", type=int, default=15, help="hide groups with fewer trips than this")
    args = ap.parse_args()

    out = Path(args.result_dir)
    paths = sorted(out.glob("ddqn_multi_seed*.pt"))
    if len(paths) < 2:
        print(f"[ERROR] need at least 2 models like ddqn_multi_seed0.pt in {out}. Run Stage C first.")
        sys.exit(1)
    nets = [dm.load_model(p)[0] for p in paths]
    ctx = dm.make_context(False)
    scenarios = {"normal test": ctx["test_env"], "new order mid-trip": ctx["dynamic_env"]}

    group_rows, worst_rows, fail_rows, lines = [], [], [], []
    for scen, env in scenarios.items():
        feats = trip_features(env, args.episodes)
        per = {}
        pols = {"Ensemble": make_ensemble_policy(nets), "Planner": planner_policy_multi, "Greedy": greedy_policy_multi}
        for i, n in enumerate(nets):
            pols[f"seed{i}"] = dm.make_policy(n)
        for name, pol in pols.items():
            _, df = evaluate_multi(pol, env, args.episodes, name, 0, return_episodes=True)
            per[name] = df.sort_values("seed").reset_index(drop=True)

        t = feats.copy()
        for name in ["Ensemble", "Planner", "Greedy"]:
            t[f"{name}_reward"] = per[name]["reward"].to_numpy()
            t[f"{name}_late"] = per[name]["late_min"].to_numpy()
            t[f"{name}_hot"] = per[name]["hotspot_steps"].to_numpy()
        t["regret"] = t["Planner_reward"] - t["Ensemble_reward"]
        t["gain_vs_greedy"] = t["Ensemble_reward"] - t["Greedy_reward"]

        # 1. where does the agent lose?
        for col in ["k", "traffic", "weather"]:
            g = t.groupby(col).agg(trips=("seed", "count"), regret=("regret", "mean"),
                                   gain_vs_greedy=("gain_vs_greedy", "mean"),
                                   ensemble_late=("Ensemble_late", "mean"), planner_late=("Planner_late", "mean"),
                                   ensemble_hot=("Ensemble_hot", "mean"), planner_hot=("Planner_hot", "mean")).reset_index()
            g = g[g["trips"] >= args.min_group].rename(columns={col: "group"})
            g.insert(0, "factor", col)
            g.insert(0, "scenario", scen)
            group_rows.append(g)

        # 2. worst trips
        worst = t.sort_values("regret", ascending=False).head(15).copy()
        worst.insert(0, "scenario", scen)
        worst_rows.append(worst[["scenario", "seed", "k", "traffic", "weather", "arrival_step",
                                 "Ensemble_reward", "Planner_reward", "regret", "Ensemble_late", "Planner_late"]])
        top10 = t.sort_values("regret", ascending=False).head(int(0.1 * len(t)))
        share = top10["regret"].sum() / t["regret"].clip(lower=0).sum()
        lines.append(f"[{scen}] mean regret vs Planner {t['regret'].mean():.2f}; the worst 10% of trips cause "
                     f"{share:.0%} of the total regret; ensemble beats the Planner outright on "
                     f"{(t['regret'] < 0).mean():.0%} of trips.")
        lines.append(f"[{scen}] extra lateness vs Planner: {t['Ensemble_late'].mean() - t['Planner_late'].mean():+.2f} min/trip; "
                     f"extra congested-cell steps: {t['Ensemble_hot'].mean() - t['Planner_hot'].mean():+.2f}/trip.")

        # 3. failed trips (single seeds)
        fails = []
        for i in range(len(nets)):
            f = per[f"seed{i}"]
            bad = f[f["delivered_all"] == 0]["seed"]
            fails += [{"scenario": scen, "model_seed": i, "trip_seed": int(s)} for s in bad]
        fdf = pd.DataFrame(fails)
        if len(fdf):
            fdf = fdf.merge(feats, left_on="trip_seed", right_on="seed").drop(columns="seed")
            fail_rows.append(fdf)
            u = fdf.drop_duplicates("trip_seed")
            lines.append(f"[{scen}] {len(fdf)} failed trip-runs over {len(nets)} seeds ({u['trip_seed'].nunique()} distinct trips). "
                         f"k of failed trips: {u['k'].value_counts().sort_index().to_dict()}; "
                         f"traffic: {u['traffic'].value_counts().to_dict()}")
            rep = fdf.groupby("trip_seed").size()
            lines.append(f"[{scen}] trips failed by more than one seed (systematic, not luck): {int((rep > 1).sum())} of {len(rep)}")
        else:
            lines.append(f"[{scen}] no failed trips in any seed.")

    groups = pd.concat(group_rows, ignore_index=True)
    groups.round(3).to_csv(out / "error_regret_by_group.csv", index=False)
    pd.concat(worst_rows).round(2).to_csv(out / "error_worst_trips.csv", index=False)
    (pd.concat(fail_rows) if fail_rows else pd.DataFrame()).to_csv(out / "error_failed_trips.csv", index=False)
    (out / "error_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== regret vs Planner by group (reward points per trip; higher = agent worse) ===")
    print(groups[["scenario", "factor", "group", "trips", "regret", "gain_vs_greedy", "ensemble_hot", "planner_hot"]].round(2).to_string(index=False))
    print("\n=== summary ===")
    print("\n".join(lines))

    # figure: regret by factor, normal vs mid-trip
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, factor in zip(axes, ["k", "traffic", "weather"]):
        sub = groups[groups["factor"] == factor]
        order = sorted(sub["group"].unique(), key=lambda x: (str(x)))
        if factor == "traffic":
            order = [v for v in TRAFFIC.values() if v in set(sub["group"])]
        if factor == "weather":
            order = [v for v in WEATHER.values() if v in set(sub["group"])]
        xs = np.arange(len(order))
        for j, (scen, c) in enumerate(zip(scenarios, ["#1f6fd1", "#e08a1e"])):
            vals = [sub[(sub["scenario"] == scen) & (sub["group"] == g)]["regret"].mean() for g in order]
            ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, label=scen, color=c)
        ax.set_xticks(xs); ax.set_xticklabels([str(o) for o in order], rotation=15)
        ax.set_title(f"By {'number of customers' if factor == 'k' else factor}")
        ax.set_ylabel("Regret vs Planner (reward / trip)"); ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    fig.suptitle("Where the DDQN ensemble loses reward compared with the oracle Planner")
    fig.tight_layout(); fig.savefig(out / "error_regret_by_group.png", dpi=160); plt.close(fig)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
