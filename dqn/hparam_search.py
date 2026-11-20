"""Part B hyperparameter search for the v4 standard DQN.

Two ideas from the assignment brief:

1. Random search instead of a full grid. Each trial draws every setting at
   random from SEARCH_SPACE, so N trials try N different values of every
   setting, while a grid of the same size would repeat the same few values.
2. Successive halving for a model that is slow to train. Every trial first
   gets a small step budget; only the best fraction is trained further. Weak
   settings are dropped early instead of costing a full 60k-step run each.

This module only proposes and ranks settings. The finalists then go through
the normal validation selection in dqn/run_experiments.py (all training seeds,
full budget), so the final choice is made exactly like the hand-picked one.
"""
from dataclasses import asdict, replace
import math
import random

from dqn.dqn_v4 import Config

# (kind, ...) per Config field. "log_uniform" suits the learning rate, whose
# useful values span orders of magnitude.
SEARCH_SPACE = {
    "lr": ("log_uniform", 1e-4, 3e-3),
    "gamma": ("choice", (0.9, 0.95, 0.97, 0.99)),
    "batch_size": ("choice", (32, 64, 128)),
    "hidden": ("choice", (64, 128, 256)),
    "target_update": ("choice", (250, 500, 1000)),
    "train_every": ("choice", (1, 2, 4)),
    "epsilon_end": ("choice", (0.01, 0.05, 0.1)),
    "epsilon_decay_fraction": ("uniform", 0.3, 0.8),
}
# How many values a grid would need for the continuous settings, used only to
# report how many runs a full grid over the same space would cost.
GRID_POINTS_PER_CONTINUOUS = 4


def sample_config(base: Config, rng: random.Random) -> Config:
    values = {}
    for name, (kind, *spec) in SEARCH_SPACE.items():
        if kind == "choice":
            values[name] = rng.choice(spec[0])
        elif kind == "uniform":
            values[name] = round(rng.uniform(spec[0], spec[1]), 3)
        elif kind == "log_uniform":
            values[name] = float(f"{math.exp(rng.uniform(math.log(spec[0]), math.log(spec[1]))):.2e}")
        else:
            raise ValueError(f"Unknown search kind {kind!r} for {name}")
    return replace(base, **values)


def sample_configs(base: Config, trials: int, seed: int) -> dict:
    if trials < 1:
        raise ValueError("Need at least one search trial")
    rng = random.Random(seed)
    return {f"random_{i:02d}": sample_config(base, rng) for i in range(trials)}


def grid_run_count() -> int:
    count = 1
    for kind, *spec in SEARCH_SPACE.values():
        count *= len(spec[0]) if kind == "choice" else GRID_POINTS_PER_CONTINUOUS
    return count


def rung_budgets(total_steps: int, rungs: int, eta: int) -> list:
    """Step budgets that grow by `eta` per rung and stop one rung below the full budget."""
    if rungs < 1 or eta < 2:
        raise ValueError("Need rungs >= 1 and eta >= 2")
    return [max(1, total_steps // eta ** (rungs - i)) for i in range(rungs)]


def successive_halving(configs: dict, score_fn, *, total_steps: int, rungs: int = 2,
                       eta: int = 2, keep: int = 3):
    """Rank `configs` with growing budgets, keeping the best 1/eta after each rung.

    score_fn(name, config) -> dict with "success_rate" and "mean_common_return".
    Returns (finalist names best-first, list of per-trial result rows).
    """
    if keep < 1:
        raise ValueError("Need keep >= 1")
    alive, rows = list(configs), []
    for rung, budget in enumerate(rung_budgets(total_steps, rungs, eta)):
        scored = []
        for name in alive:
            config = replace(configs[name], total_steps=budget)
            result = score_fn(name, config)
            scored.append((result["success_rate"], result["mean_common_return"], name))
            rows.append(dict(rung=rung, steps=budget, trial=name, **result,
                             **{k: v for k, v in asdict(config).items() if k in SEARCH_SPACE}))
        # Sort best first; the name breaks exact ties so the order is reproducible.
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        survivors = max(keep, math.ceil(len(scored) / eta)) if rung < rungs - 1 else keep
        alive = [name for *_, name in scored[:survivors]]
        for row in rows:
            if row["rung"] == rung:
                row["promoted"] = row["trial"] in alive
    return alive, rows
