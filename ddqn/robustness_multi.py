"""
Robustness and interpretability checks for the multi-order DDQN  (NO training: uses the saved Stage C models)
============================================================================================================
Answers, with numbers, the questions "is safe_argmax necessary?", "which inputs does the network use?",
"does the result depend on our hand-set reward and time rules?" and "does it survive a more realistic clock?".

  1. filter   : wall/backtrack penalties of safe_argmax. The back-step penalty is SWEPT on VALIDATION trips,
                then everything is reported once on TEST trips (never tune on test).
  2. inputs   : input importance. One group of network inputs at a time is replaced by its average value;
                the reward drop shows how much the agent relies on that group.
  3. rules    : the agent, Greedy and Planner are re-scored when one scoring/timing rule is changed
                (late penalty, congestion cost, weather weight, ...). Policies are NOT retrained, so this tests
                whether the RANKING depends on our hand-set numbers, not whether another agent would do better.
  4. calibrated: a clock fitted to the real data (see CALIBRATED below), same idea as 3.
  5. self-test: with default rules the parameterised environment must give exactly the original rewards.

Run from the project root (a few minutes on a laptop):
    python ddqn/robustness_multi.py --result-dir ddqn/result_multi
    python ddqn/robustness_multi.py --quick
Outputs (result dir): robust_filter.csv, robust_inputs.csv, robust_rules.csv, robust_summary.txt, robust_*.png
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
from env_multi import (DEFAULT_RULES, K_MAX, MultiDeliveryEnv, evaluate_multi, greedy_policy_multi,
                       planner_policy_multi, safe_argmax)

# Fitted on data/dataset_clean.csv (40,482 orders): Jam vs Low trips take 1.46x as long (one-level factor about
# 1.11), and weather changes delivery time by only about 2% per level. The original rules use 1.75x for Jam and
# +15% per weather level, so this preset keeps the structure but uses the fitted strengths.
CALIBRATED = dict(m_traffic=0.15, m_weather=0.02, c_weather=0.03)

RULE_CASES = {
    "Original rules": {},
    "Late penalty 0.25 per min": dict(late_w=0.25),
    "Late penalty 1.0 per min": dict(late_w=1.0),
    "Congestion cost x0.5": dict(hot_extra=0.5, hot_traffic=0.5),
    "Congestion cost x2": dict(hot_extra=2.0, hot_traffic=2.0),
    "Time in congested cell x1.5": dict(hot_minutes=1.5),
    "Time in congested cell x3": dict(hot_minutes=3.0),
    "Weather weight 0 (cost and time)": dict(c_weather=0.0, m_weather=0.0),
    "Delivery bonus 30": dict(deliver_bonus=30.0),
    "Calibrated clock (fitted to data)": CALIBRATED,
}

# input groups of the 22-number observation (see env_multi._obs and obs_v3)
GROUPS = {
    "Rider position": [0, 1],
    "Direction to customers (row, col offsets)": [2 + 4 * j + o for j in range(K_MAX) for o in (0, 1)],
    "Distance to customers": [2 + 4 * j + 2 for j in range(K_MAX)],
    "Which customers are still active": [2 + 4 * j + 3 for j in range(K_MAX)],
    "Traffic level": [14],
    "Weather level": [15],
    "Share of orders left": [16],
    "Time left": [17],
    "Neighbour-congestion flags": [18, 19, 20, 21],
}


def ensemble_act(nets, wall_pen=10.0, back_pen=1.0, mean_fill=None, hide=()):
    """Ensemble policy (z-scored Q average) with adjustable filter and optional hidden inputs."""
    tr = {"p": None}
    hide = list(hide)

    def act(env):
        if env.steps == 0:
            tr["p"] = None
        obs = env.obs_v3()
        if hide:
            obs = obs.copy()
            obs[hide] = mean_fill[hide]
        x = torch.tensor(obs).unsqueeze(0)
        zs = []
        with torch.no_grad():
            for n in nets:
                n.eval()
                q = n(x).squeeze(0).numpy().astype(np.float64)
                zs.append((q - q.mean()) / (q.std() + 1e-8) if len(nets) > 1 else q)
        a = safe_argmax(np.mean(zs, axis=0), env, tr["p"], wall_pen=wall_pen, back_pen=back_pen)
        tr["p"] = env.pos
        return a
    return act


def mean_observation(env, policy, trips=60):
    """Average value of every input along real trips (used to 'hide' an input group without leaving the data range)."""
    obs = []
    for s in range(trips):
        env.reset(seed=5000 + s)
        done = False
        while not done:
            obs.append(env.obs_v3())
            _, _, t, tr_, _ = env.step(policy(env))
            done = t or tr_
    return np.mean(obs, axis=0).astype(np.float32)


def rules_env(base_env, rules, **kw):
    return MultiDeliveryEnv(base_env.orders, max_km=base_env.max_km, rules=rules, **kw)


def main():
    ap = argparse.ArgumentParser(description="Robustness and interpretability checks (no training)")
    ap.add_argument("--result-dir", default=str(BASE_DIR / "ddqn" / "result_multi"))
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = Path(args.result_dir)
    paths = sorted(out.glob("ddqn_multi_seed*.pt"))
    if len(paths) < 2:
        print(f"[ERROR] need models like ddqn_multi_seed0.pt in {out}. Run Stage C first.")
        sys.exit(1)
    nets = [dm.load_model(p)[0] for p in paths]
    n_ep = 20 if args.quick else args.episodes
    ctx = dm.make_context(args.quick)
    lines = []

    # 5. self-test -------------------------------------------------------------------------------------------
    e_old, e_new = MultiDeliveryEnv(ctx["test_env"].orders, max_km=ctx["test_env"].max_km), rules_env(ctx["test_env"], dict(DEFAULT_RULES))
    a = evaluate_multi(planner_policy_multi, e_old, 30, "a")["avg_reward"]
    b = evaluate_multi(planner_policy_multi, e_new, 30, "b")["avg_reward"]
    lines.append(f"[self-test] default rules reproduce the original environment: {abs(a - b) < 1e-9} ({a:.4f} vs {b:.4f})")
    assert abs(a - b) < 1e-9

    # 1. safe_argmax ----------------------------------------------------------------------------------------
    rows = []
    sweep = [0.0, 0.25, 0.5, 1.0, 2.0]
    val_env = ctx["val_env"]
    for bp in sweep:
        r = evaluate_multi(ensemble_act(nets, back_pen=bp), val_env, n_ep, "ens", dm.VAL_SEED_OFFSET)
        rows.append({"stage": "validation sweep", "scenario": "normal", "who": "Ensemble", "wall_pen": 10, "back_pen": bp,
                     "reward": r["avg_reward"], "delivered_all": r["success_rate"]})
    sweep_df = pd.DataFrame(rows)
    best_bp = float(sweep_df.sort_values("reward", ascending=False).iloc[0]["back_pen"])
    lines.append("[filter] back-step penalty sweep on VALIDATION trips (ensemble): " +
                 ", ".join(f"{r.back_pen}: {r.reward:.2f}" for r in sweep_df.itertuples()) + f" -> best {best_bp}")
    scen = {"normal": ctx["test_env"], "new order mid-trip": ctx["dynamic_env"], "unseen map": ctx["unseen_map_env"]}
    cfgs = {"No filter": (0.0, 0.0), "Wall rule only": (10.0, 0.0), "Wall + backtrack 1.0 (ours)": (10.0, 1.0),
            f"Wall + backtrack {best_bp} (validation-best)": (10.0, best_bp)}
    who = {"Single seed 0": nets[:1], "Single seed 1": nets[1:2], "Ensemble": nets}
    for sname, env in scen.items():
        for cname, (w, bp) in cfgs.items():
            for wname, nl in who.items():
                r = evaluate_multi(ensemble_act(nl, w, bp), env, n_ep, wname, 0)
                rows.append({"stage": "test", "scenario": sname, "who": wname, "filter": cname, "wall_pen": w, "back_pen": bp,
                             "reward": r["avg_reward"], "delivered_all": r["success_rate"], "hotspot_steps": r["avg_hotspot_steps"]})
        for gname, g in {"Greedy": greedy_policy_multi, "Planner": planner_policy_multi}.items():
            r = evaluate_multi(g, env, n_ep, gname, 0)
            rows.append({"stage": "test", "scenario": sname, "who": gname, "filter": "-", "wall_pen": np.nan, "back_pen": np.nan,
                         "reward": r["avg_reward"], "delivered_all": r["success_rate"], "hotspot_steps": r["avg_hotspot_steps"]})
    fdf = pd.DataFrame(rows)
    fdf.round(3).to_csv(out / "robust_filter.csv", index=False)
    print("\n=== safe_argmax filter (TEST trips, reward per trip) ===")
    piv = fdf[fdf["stage"] == "test"].pivot_table(index=["scenario", "who"], columns="filter", values="reward")
    print(piv.round(2).to_string())

    # 2. input importance ----------------------------------------------------------------------------------
    base_pol = ensemble_act(nets)
    fill = mean_observation(ctx["val_env"], base_pol, 20 if args.quick else 60)
    irows = []
    for sname in ["normal", "new order mid-trip"]:
        env = scen[sname]
        ref = evaluate_multi(base_pol, env, n_ep, "ref", 0)["avg_reward"]
        for gname, idx in GROUPS.items():
            r = evaluate_multi(ensemble_act(nets, mean_fill=fill, hide=idx), env, n_ep, gname, 0)
            irows.append({"scenario": sname, "input_group": gname, "reward": r["avg_reward"], "reward_drop": ref - r["avg_reward"],
                          "delivered_all": r["success_rate"]})
    idf = pd.DataFrame(irows)
    idf.round(3).to_csv(out / "robust_inputs.csv", index=False)
    print("\n=== input importance (reward drop when a group is replaced by its average; bigger = relied on more) ===")
    print(idf.pivot(index="input_group", columns="scenario", values="reward_drop").round(2).sort_values("normal", ascending=False).to_string())
    top = idf[idf.scenario == "normal"].sort_values("reward_drop", ascending=False).head(3)
    lines.append("[inputs] most relied-on inputs (normal): " + "; ".join(f"{r.input_group} ({r.reward_drop:.1f})" for r in top.itertuples()))

    # 3 and 4. rule sensitivity ----------------------------------------------------------------------------------
    rrows = []
    for cname, rules in RULE_CASES.items():
        env = rules_env(ctx["test_env"], rules)
        res = {n: evaluate_multi(p, env, n_ep, n, 0) for n, p in {"Ensemble": base_pol, "Greedy": greedy_policy_multi, "Planner": planner_policy_multi}.items()}
        rrows.append({"rule_change": cname, **{f"{n}_reward": r["avg_reward"] for n, r in res.items()},
                      **{f"{n}_late_min": r["avg_late_min"] for n, r in res.items()},
                      "gain_vs_greedy": res["Ensemble"]["avg_reward"] - res["Greedy"]["avg_reward"],
                      "gap_closed": (res["Ensemble"]["avg_reward"] - res["Greedy"]["avg_reward"]) / max(1e-9, res["Planner"]["avg_reward"] - res["Greedy"]["avg_reward"]),
                      "ensemble_beats_greedy": res["Ensemble"]["avg_reward"] > res["Greedy"]["avg_reward"]})
    rdf = pd.DataFrame(rrows)
    rdf.round(3).to_csv(out / "robust_rules.csv", index=False)
    print("\n=== rule sensitivity (models NOT retrained; test trips) ===")
    print(rdf[["rule_change", "Ensemble_reward", "Greedy_reward", "Planner_reward", "gain_vs_greedy", "gap_closed", "ensemble_beats_greedy"]].round(2).to_string(index=False))
    lines.append(f"[rules] ensemble beats Greedy under {int(rdf.ensemble_beats_greedy.sum())} of {len(rdf)} rule sets; "
                 f"gain over Greedy ranges {rdf.gain_vs_greedy.min():.2f} to {rdf.gain_vs_greedy.max():.2f}")
    (out / "robust_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = idf[idf.scenario == "normal"].sort_values("reward_drop")
    ax.barh(s["input_group"], s["reward_drop"], color="#1f6fd1")
    ax.set_xlabel("Reward lost when this input group is hidden (normal trips)")
    ax.set_title("What the DDQN ensemble relies on"); ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out / "robust_inputs.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(rdf["rule_change"], rdf["gain_vs_greedy"], color="#e08a1e")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Ensemble reward minus Greedy reward (same rules for both)")
    ax.set_title("Does the advantage survive other scoring rules?"); ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(out / "robust_rules.png", dpi=160); plt.close(fig)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
