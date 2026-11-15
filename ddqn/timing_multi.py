"""
Decision-time measurement (Part B "efficiency": inference speed)
================================================================
Time per decision of Greedy, the Planner, one D3QN model and the 5-model ensemble on the same states, measured over
real trips. Also answers whether a learned agent is "faster than Dijkstra": that depends on the map size and on
the hardware, so it is measured, not assumed.

    python ddqn/timing_multi.py --result-dir ddqn/result_multi
Output: <result-dir>/timing.csv
"""
import argparse, sys, time
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR)); sys.path.insert(0, str(BASE_DIR / "ddqn"))
import numpy as np, pandas as pd, torch
import ddqn_multi as dm
from ensemble_multi import make_ensemble_policy
from env_multi import greedy_policy_multi, planner_policy_multi


def collect_states(env, policy, trips):
    """Replay trips with `policy` and keep a deep copy of the environment at each step."""
    import copy
    snaps = []
    for s in range(trips):
        env.reset(seed=7000 + s)
        done = False
        while not done:
            snaps.append(copy.deepcopy(env))
            _, _, t, tr, _ = env.step(policy(env))
            done = t or tr
    return snaps


def time_policy(policy, snaps, repeats=3):
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter()
        for e in snaps:
            policy(e)
        best = min(best, (time.perf_counter() - t0) / len(snaps))
    return best * 1000.0   # ms per decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default=str(BASE_DIR / "ddqn" / "result_multi"))
    ap.add_argument("--trips", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.result_dir)
    paths = sorted(out.glob("ddqn_multi_seed*.pt"))
    if not paths:
        print("[ERROR] no models found"); sys.exit(1)
    nets = [dm.load_model(p)[0] for p in paths]
    torch.set_num_threads(1)
    ctx = dm.make_context(False)
    snaps = collect_states(ctx["test_env"], greedy_policy_multi, a.trips)
    rows = []
    for name, pol in {"Greedy": greedy_policy_multi, "Planner (Dijkstra, all orders)": planner_policy_multi,
                      "D3QN single model": dm.make_policy(nets[0]),
                      f"D3QN ensemble ({len(nets)} models)": make_ensemble_policy(nets)}.items():
        rows.append({"method": name, "ms_per_decision": round(time_policy(pol, snaps), 3)})
    df = pd.DataFrame(rows); df["decisions_timed"] = len(snaps)
    df.to_csv(out / "timing.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
