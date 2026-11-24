"""Reproducible Part A ablations and Part B validation-only model selection.

python -B -m dqn_dynamic.run_experiments --smoke
python -B -m dqn_dynamic.run_experiments --steps 60000 --seeds 11 22 33
python -B -m dqn_dynamic.run_experiments --search random --trials 12 --finalists 3
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from env import load_orders
from dqn_dynamic.dqn_v4 import Config, policy, save_checkpoint, train
from dqn_dynamic.hparam_search import (SEARCH_SPACE, grid_run_count, rung_budgets, sample_configs,
                               successive_halving)
from dqn_dynamic.dynamic_env import DynamicDeliveryEnv, MOVES, features
from dqn_dynamic.evaluate_dqn import evaluate, greedy_policy, paired_interval, planner_policy, summarize
from dqn.experiments import (MAP_BANKS, dataset_hash, split_manifest,
                             split_orders_three_way, training_scale, write_json)

ROOT = Path(__file__).resolve().parents[1]
# Factorial state/reward comparison plus a static-training control. No shared
# benchmark model is silently reused: all variants use the new training split.
VARIANTS = {
    "baseline": ("baseline", "legacy", "dynamic"),
    "state_only": ("deadline", "legacy", "dynamic"),
    "reward_only": ("baseline", "time", "dynamic"),
    "full_dynamic": ("deadline", "time", "dynamic"),
    "full_static": ("deadline", "time", "static"),
}


def make_env(orders, scale, partition, *, scenario="dynamic", reward_mode="time"):
    return DynamicDeliveryEnv(orders, max_km=scale, map_seeds=MAP_BANKS[partition],
                              scenario=scenario, reward_mode=reward_mode)


def ensemble_policy(networks, backtracking_penalty=0.0):
    """Predeclared equal-weight seed ensemble; heuristic tested separately."""
    previous = None

    def act(env):
        nonlocal previous
        if env.steps == 0:
            previous = None
        with torch.no_grad():
            state = torch.from_numpy(features(env))
            qs = np.stack([net(state).numpy() for net in networks])
        normalized = (qs - qs.mean(axis=1, keepdims=True)) / (qs.std(axis=1, keepdims=True) + 1e-6)
        q = normalized.mean(axis=0)
        for a, (dr, dc) in enumerate(MOVES):
            if (env.pos[0] + dr, env.pos[1] + dc) == previous:
                q[a] -= backtracking_penalty
        previous = env.pos
        return int(q.argmax())
    return act


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/dataset_clean.csv")
    parser.add_argument("--output", type=Path, help="New directory inside dqn_dynamic/result; refuses overwrite")
    parser.add_argument("--steps", type=int, default=60_000, help="Same interaction budget for every run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--validation-only", action="store_true", help="Do not open final-test evaluation")
    parser.add_argument("--smoke", action="store_true", help="Tiny validation-only integration check")
    parser.add_argument("--search", choices=("manual", "random"), default="manual",
                        help="manual = the 4 hand-picked settings; random = random search + successive halving")
    parser.add_argument("--trials", type=int, default=12, help="Random-search trials (with --search random)")
    parser.add_argument("--finalists", type=int, default=3,
                        help="Random-search trials that reach full multi-seed selection")
    parser.add_argument("--search-seed", type=int, default=0, help="Seed that draws the random trials")
    parser.add_argument("--rungs", type=int, default=2, help="Successive-halving rounds before the finalists")
    parser.add_argument("--eta", type=int, default=2, help="Keep 1/eta of the trials after each round")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.eval_episodes < 1 or args.threads < 1:
        parser.error("steps, eval-episodes and threads must be positive")
    if args.trials < 1 or not 1 <= args.finalists <= args.trials or args.rungs < 1 or args.eta < 2:
        parser.error("Need trials >= 1, 1 <= finalists <= trials, rungs >= 1 and eta >= 2")
    if min(args.seeds) < 0 or max(args.seeds) > 1000 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct training seeds in 0..1000")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        args.steps, args.seeds, args.eval_episodes, args.validation_only = 256, [11], 4, True
        args.trials, args.finalists = min(args.trials, 4), 1
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    base_dir = ROOT / "dqn_dynamic/result"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or base_dir / run_id).resolve()
    if not output.is_relative_to(base_dir.resolve()) or output == base_dir.resolve():
        raise ValueError("Output must be a new subdirectory of dqn_dynamic/result")
    # Fail before creating output if input cannot be loaded.
    orders = load_orders(args.data)
    partitions = split_orders_three_way(orders)
    train_orders, validation_orders, test_orders = partitions
    scale = training_scale(train_orders)
    output.mkdir(parents=True, exist_ok=False)
    cfg = Config(total_steps=args.steps)
    if args.smoke:
        cfg = replace(cfg, hidden=32, warmup=32, batch_size=16, target_update=64)
    candidates = {
        "baseline_config": cfg,
        "lower_lr": replace(cfg, lr=0.0005),
        "lower_gamma": replace(cfg, gamma=0.95),
        "faster_target": replace(cfg, target_update=250 if not args.smoke else 32),
    }
    if args.smoke or args.search == "random":
        # Random search replaces the hand-picked alternatives but keeps the
        # default settings as the reference point.
        candidates = dict(list(candidates.items())[:1 if args.search == "random" else 2])
    manifest = dict(run_id=run_id, status="running", smoke=args.smoke,
                    validation_only=args.validation_only, algorithm="standard_dqn",
                    data_sha256=dataset_hash(args.data), split_seed=42,
                    partitions=split_manifest(partitions), max_km=scale,
                    training_seeds=args.seeds, map_banks=MAP_BANKS,
                    evaluation_episodes=args.eval_episodes, python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, torch=str(torch.__version__),
                    torch_threads=args.threads,
                    source_sha256={p.name: dataset_hash(p) for p in Path(__file__).parent.glob("*.py")},
                    environment=dict(grid_size=10, max_steps=60, dynamic_interval_min=8,
                                     stress_interval_min=4, deadline="historical duration proxy",
                                     move_cost_timing="departure snapshot"),
                    candidates={name: asdict(c) for name, c in candidates.items()},
                    search=dict(method=args.search, trials=args.trials, finalists=args.finalists,
                                search_seed=args.search_seed, rungs=args.rungs, eta=args.eta,
                                search_training_seed=args.seeds[0], space=SEARCH_SPACE,
                                rung_steps=rung_budgets(args.steps, args.rungs, args.eta),
                                equivalent_grid_runs=grid_run_count())
                    if args.search == "random" else dict(method="manual"),
                    variants=VARIANTS,
                    test_caveat="Preserves historical team test rows, previously inspected; new map/event seeds are held out.")
    write_json(output / "manifest.json", manifest)
    # The project path can contain Thai characters on a cp1252 Windows console.
    print(f"Output (relative to project): {output.relative_to(ROOT)}", flush=True)
    tuning, models = [], {}

    def train_one(label, config, seed, variant):
        feature_mode, reward_mode, scenario = VARIANTS[variant]
        env = make_env(train_orders, scale, "train", scenario=scenario, reward_mode=reward_mode)
        started = time.perf_counter()
        net, history, stats = train(env, config, seed=seed, feature_mode=feature_mode)
        metadata = dict(max_km=scale, variant=variant, reward_mode=reward_mode,
                        scenario=scenario, data_sha256=manifest["data_sha256"], split_seed=42,
                        training_map_seeds=list(MAP_BANKS["train"]))
        save_checkpoint(output / f"{label}_seed{seed}.pt", net, config,
                        seed=seed, feature_mode=feature_mode, metadata=metadata)
        pd.DataFrame(history).to_csv(output / f"{label}_seed{seed}_learning.csv", index=False)
        print(f"Trained {label} seed={seed}: {stats['updates']} updates in {time.perf_counter()-started:.1f}s", flush=True)
        return net

    if args.search == "random":
        trials = sample_configs(cfg, args.trials, args.search_seed)
        search_seed = args.seeds[0]

        def score(name, config):
            started = time.perf_counter()
            env = make_env(train_orders, scale, "train")
            net, _, _ = train(env, config, seed=search_seed, feature_mode="deadline")
            frame = evaluate(policy(net), make_env(validation_orders, scale, "validation"),
                             episodes=args.eval_episodes, agent=name, training_seed=search_seed)
            result = summarize(frame)
            print(f"Search {name} at {config.total_steps} steps: success={result['success_rate']:.2f} "
                  f"common return={result['mean_common_return']:.1f} "
                  f"({time.perf_counter()-started:.1f}s)", flush=True)
            return dict(success_rate=result["success_rate"],
                        mean_common_return=result["mean_common_return"])

        finalists, search_rows = successive_halving(
            trials, score, total_steps=args.steps, rungs=args.rungs, eta=args.eta,
            keep=args.finalists)
        pd.DataFrame(search_rows).to_csv(output / "random_search.csv", index=False)
        candidates.update({name: trials[name] for name in finalists})
        manifest["candidates"] = {name: asdict(c) for name, c in candidates.items()}
        manifest["search"]["finalists"] = finalists
        manifest["search"]["search_steps"] = int(sum(r["steps"] for r in search_rows))
        write_json(output / "manifest.json", manifest)
        print(f"Random-search finalists: {', '.join(finalists)} "
              f"(a full grid over the same space would need {grid_run_count()} runs)", flush=True)

    # Selection sees validation orders and validation maps/events only.
    validation_details = []
    for name, config in candidates.items():
        for seed in args.seeds:
            net = train_one(name, config, seed, "full_dynamic")
            models[(name, seed)] = net
            frame = evaluate(policy(net), make_env(validation_orders, scale, "validation"),
                             episodes=args.eval_episodes, agent=name, training_seed=seed)
            validation_details.append(frame)
            tuning.append(dict(candidate=name, training_seed=seed, **summarize(frame)))
    tuning_frame = pd.DataFrame(tuning)
    tuning_frame.to_csv(output / "validation_tuning.csv", index=False)
    pd.concat(validation_details, ignore_index=True).to_csv(output / "validation_tuning_episodes.csv", index=False)
    ranked = tuning_frame.groupby("candidate").mean(numeric_only=True).sort_values(
        ["success_rate", "mean_common_return"], ascending=False, kind="stable")
    winner = str(ranked.index[0])
    selected = candidates[winner]
    # Written BEFORE any final-test simulation. No best-seed cherry-picking.
    write_json(output / "selection.json", dict(candidate=winner, config=asdict(selected),
               criterion="mean validation success rate, then mean common return across all training seeds",
               seeds=args.seeds, ensemble_weights="equal", backtracking_penalty=1.5))
    print(f"Frozen validation selection: {winner}", flush=True)
    policies = {}
    selected_networks = []
    for variant, (feature_mode, _, _) in VARIANTS.items():
        for seed in args.seeds:
            net = models[(winner, seed)] if variant == "full_dynamic" else train_one(variant, selected, seed, variant)
            policies[(variant, seed)] = policy(net, feature_mode)
            if variant == "full_dynamic":
                selected_networks.append(net)
    policies[("Ensemble", -1)] = ensemble_policy(selected_networks)
    policies[("Ensemble+backtracking", -1)] = ensemble_policy(selected_networks, 1.5)
    policies[("Greedy", -1)] = greedy_policy
    policies[("Current-map time planner", -1)] = planner_policy
    partition = "validation" if args.validation_only else "test"
    evaluation_orders = validation_orders if args.validation_only else test_orders
    frames = []
    for scenario in ("static", "dynamic", "stress"):
        for (name, seed), act in policies.items():
            frame = evaluate(act, make_env(evaluation_orders, scale, partition, scenario=scenario),
                             episodes=args.eval_episodes, partition=partition,
                             agent=name, training_seed=seed)
            frames.append(frame)
        print(f"Evaluated {scenario} ({partition})", flush=True)
    episodes = pd.concat(frames, ignore_index=True)
    episodes.to_csv(output / f"{partition}_episodes.csv", index=False)
    summary = []
    for (scenario, agent, seed), frame in episodes.groupby(["scenario", "agent", "training_seed"]):
        summary.append(dict(scenario=scenario, agent=agent, training_seed=seed, **summarize(frame)))
    pd.DataFrame(summary).to_csv(output / f"{partition}_summary.csv", index=False)
    intervals = []
    for scenario, frame in episodes.groupby("scenario"):
        for agent in VARIANTS:
            for metric in ("common_return", "delivered", "on_time"):
                intervals.append(dict(scenario=scenario, **paired_interval(frame, agent, metric=metric)))
        for control in ("baseline", "state_only", "reward_only", "full_static"):
            for metric in ("common_return", "delivered", "on_time"):
                intervals.append(dict(scenario=scenario, **paired_interval(
                    frame, "full_dynamic", baseline=control, metric=metric)))
    pd.DataFrame(intervals).to_csv(output / f"{partition}_paired_intervals.csv", index=False)
    manifest["status"] = "complete"
    manifest["selected_candidate"] = winner
    write_json(output / "manifest.json", manifest)
    print(f"Completed. {'Smoke check only; no performance claim.' if args.smoke else 'Inspect paired intervals before claiming improvement.'}", flush=True)
    return output


if __name__ == "__main__":
    main()
