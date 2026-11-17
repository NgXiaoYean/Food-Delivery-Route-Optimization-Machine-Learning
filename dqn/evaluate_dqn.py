"""Paired DQN-only evaluation; common return is independent of training reward."""
import heapq

import numpy as np
import pandas as pd

from dqn.dynamic_env import MOVES
from dqn.experiments import EPISODE_OFFSETS


def greedy_policy(env):
    r, c = env.pos
    return (1 if r < env.dest[0] else 0) if r != env.dest[0] else (3 if c < env.dest[1] else 2)


def planner_policy(env):
    """Replan for current travel minutes; full current map, NO future knowledge."""
    distances, heap = {env.pos: 0.0}, [(0.0, env.pos, -1)]
    while heap:
        cost, cell, first = heapq.heappop(heap)
        if cell == env.dest:
            return max(first, 0)
        if cost > distances[cell]:
            continue
        for action, (dr, dc) in enumerate(MOVES):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < env.n and 0 <= nxt[1] < env.n):
                continue
            new_cost = cost + env.cell_minutes(nxt)
            if new_cost < distances.get(nxt, float("inf")):
                distances[nxt] = new_cost
                heapq.heappush(heap, (new_cost, nxt, action if first < 0 else first))
    raise RuntimeError("No path to delivery")


def evaluate(policy_fn, env, *, episodes=200, partition="validation", agent="DQN", training_seed=-1):
    if episodes < 1 or partition not in EPISODE_OFFSETS:
        raise ValueError("Need positive episode count and a known partition")
    rows = []
    for ep in range(episodes):
        episode_seed = EPISODE_OFFSETS[partition] + ep
        env.reset(seed=episode_seed)
        start, destination = env.pos, env.dest
        common, reward, hotspots, bumps, reversals, events = 0.0, 0.0, 0, 0, 0, 0
        previous = None
        while True:
            old_pos = env.pos
            _, rew, term, trunc, info = env.step(policy_fn(env))
            reversals += int(previous is not None and env.pos == previous and env.pos != old_pos)
            previous = old_pos
            reward += rew
            common += info["common_reward"]
            hotspots += int(info["hotspot_entered"])
            bumps += int(info["wall_bump"])
            events += info["events_crossed"]
            if term or trunc:
                break
        delivered = bool(info["delivered"])
        rows.append(dict(agent=agent, training_seed=training_seed, partition=partition,
                         scenario=env.scenario, episode_seed=episode_seed, map_seed=env.map_seed,
                         start=str(start), destination=str(destination),
                         deadline_proxy_min=env.benchmark_min, delivered=delivered,
                         on_time=delivered and info["elapsed_min"] <= env.benchmark_min,
                         elapsed_min=info["elapsed_min"], steps=info["steps"],
                         lateness_min=info["lateness_min"] if delivered else np.nan,
                         common_return=common, training_reward=reward, hotspot_steps=hotspots,
                         wall_bumps=bumps, backtracks=reversals, events_crossed=events))
    return pd.DataFrame(rows)


def summarize(frame):
    completed = frame.loc[frame.delivered]
    return dict(success_rate=float(frame.delivered.mean()), on_time_rate=float(frame.on_time.mean()),
                mean_common_return=float(frame.common_return.mean()),
                mean_elapsed_all_min=float(frame.elapsed_min.mean()),
                mean_delivery_min=float(completed.elapsed_min.mean()),
                p95_delivery_min=float(completed.elapsed_min.quantile(0.95)),
                mean_lateness_min=float(completed.lateness_min.mean()),
                p95_lateness_min=float(completed.lateness_min.quantile(0.95)),
                mean_steps=float(frame.steps.mean()), mean_hotspots=float(frame.hotspot_steps.mean()),
                mean_backtracks=float(frame.backtracks.mean()), mean_bumps=float(frame.wall_bumps.mean()))


def paired_interval(frame, agent, baseline="Greedy", metric="common_return", samples=2000):
    """Crossed bootstrap over training seeds AND paired evaluation episodes.

    Input must contain one scenario/partition. Non-learning baselines have
    training_seed=-1; learned controls are additionally paired by training seed.
    The interval is exploratory with few training seeds; episodes alone do not
    constitute independent model-training replicates.
    """
    if frame.scenario.nunique() != 1 or frame.partition.nunique() != 1:
        raise ValueError("Pair within one scenario and partition")
    keys = ["episode_seed", "map_seed", "start", "destination", "deadline_proxy_min"]
    baseline_frame = frame[frame.agent == baseline]
    if not (baseline_frame.training_seed == -1).all():
        keys = keys + ["training_seed"]
    reference = baseline_frame[keys + [metric]]
    candidate = frame[frame.agent == agent]
    joined = candidate.merge(reference, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
    if len(joined) != len(candidate) or joined.empty:
        raise ValueError("Candidate and baseline episodes must match exactly")
    joined = joined.assign(delta=joined[metric].astype(float) - joined[metric + "_baseline"].astype(float))
    matrix = joined.pivot(index="training_seed", columns="episode_seed", values="delta").to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("Incomplete paired seed/episode matrix")
    rng = np.random.default_rng(42)
    boot = np.empty(samples)
    for i in range(samples):
        seed_ids = rng.integers(matrix.shape[0], size=matrix.shape[0])
        episode_ids = rng.integers(matrix.shape[1], size=matrix.shape[1])
        boot[i] = matrix[np.ix_(seed_ids, episode_ids)].mean()
    return dict(agent=agent, baseline=baseline, metric=metric, delta=float(matrix.mean()),
                ci95_low=float(np.quantile(boot, 0.025)), ci95_high=float(np.quantile(boot, 0.975)),
                training_seeds=matrix.shape[0], paired_episodes=matrix.shape[1])
