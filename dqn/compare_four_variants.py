"""Controlled single-order comparison of Q-Learning and four DQN architectures.

Run from the project root:
    python -B -m dqn.compare_four_variants --smoke
    python -B -m dqn.compare_four_variants

The DQN comparison isolates Double targets and Dueling heads. All five models
use the same training step budget, data partitions, maps, and evaluation trips.
The DQN architectures also share features, reward, replay, optimizer, and
target-copy schedule. Q-Learning uses its own tabular state and update rule.
These are new controlled runs, not the historical app checkpoints.
"""
import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from dqn.dqn_v2 import ReplayBuffer
from dqn.dqn_v3 import OBS_DIM_V3, features_v3
from dqn.experiments import dataset_hash, split_manifest, split_orders_three_way, training_scale, write_json
from env import DeliveryEnv, load_orders
from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "dqn" / "result" / "v4"
PORTABLE_SUMMARY = ROOT / "dqn" / "five_model_summary.json"
VARIANTS = {
    "DQN": (False, False),
    "Double DQN": (True, False),
    "Dueling DQN": (False, True),
    "D3QN": (True, True),
}
TABULAR_NAME = "Q-Learning"
MODEL_NAMES = (TABULAR_NAME, *VARIANTS)
METRICS = ("delivered", "on_time", "reward", "elapsed_min", "steps", "hotspot_steps")


@dataclass(frozen=True)
class Settings:
    train_steps: int = 60_000
    hidden: int = 128
    lr: float = 0.001
    gamma: float = 0.95
    batch_size: int = 64
    replay_size: int = 50_000
    warmup: int = 1_000
    reward_scale: float = 10.0
    target_update: int = 500
    update_every: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000


class ComparableQNetwork(nn.Module):
    """One shared encoder; only the head differs between plain and dueling."""

    def __init__(self, *, hidden: int, dueling: bool):
        super().__init__()
        self.dueling = dueling
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM_V3, hidden), nn.ReLU())
        if dueling:
            self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
            self.advantage = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 4))
        else:
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 4))

    def forward(self, state):
        encoded = self.encoder(state)
        if not self.dueling:
            return self.head(encoded)
        advantage = self.advantage(encoded)
        return self.value(encoded) + advantage - advantage.mean(dim=-1, keepdim=True)


def make_env(orders, scale, map_seed):
    env = DeliveryEnv(orders.reset_index(drop=True), map_seed=map_seed)
    env.max_km = scale  # keep the train-set distance scale on validation and test
    return env


def train_one(orders, scale, name, seed, settings):
    double, dueling = VARIANTS[name]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_env(orders, scale, map_seed=0)
    online = ComparableQNetwork(hidden=settings.hidden, dueling=dueling)
    target = ComparableQNetwork(hidden=settings.hidden, dueling=dueling)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.AdamW(online.parameters(), lr=settings.lr, weight_decay=0.0)
    replay = ReplayBuffer(settings.replay_size)
    episode = 0
    observation, _ = env.reset(seed=seed * 1_000_000 + episode)
    state = features_v3(observation, env)
    recent_rewards, recent_success, history = [], [], []
    episode_reward = 0.0
    updates = 0

    for step in range(1, settings.train_steps + 1):
        fraction = min(1.0, (step - 1) / settings.epsilon_decay_steps)
        epsilon = settings.epsilon_start + fraction * (settings.epsilon_end - settings.epsilon_start)
        if random.random() < epsilon:
            action = random.randrange(4)
        else:
            with torch.no_grad():
                action = int(online(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = features_v3(next_observation, env)
        done = terminated or truncated
        # Timeout ends this finite delivery episode, so it is terminal for the target.
        replay.add(state, action, reward / settings.reward_scale, next_state, float(done))
        state = next_state
        episode_reward += reward

        if step % settings.update_every == 0 and len(replay) >= max(settings.warmup, settings.batch_size):
            states, actions, rewards, next_states, terminals = replay.sample(settings.batch_size)
            predicted = online(states).gather(1, actions)
            with torch.no_grad():
                if double:
                    # Double: online selects the action; frozen target evaluates it.
                    next_actions = online(next_states).argmax(dim=1, keepdim=True)
                    next_values = target(next_states).gather(1, next_actions)
                else:
                    # Standard: frozen target both selects and evaluates the action.
                    next_values = target(next_states).max(dim=1, keepdim=True).values
                expected = rewards + settings.gamma * next_values * (1.0 - terminals)
            loss = nn.functional.smooth_l1_loss(predicted, expected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 5.0)
            optimizer.step()
            updates += 1

        if step % settings.target_update == 0:
            target.load_state_dict(online.state_dict())
        if done:
            recent_rewards.append(episode_reward)
            recent_success.append(float(info["delivered"]))
            episode += 1
            episode_reward = 0.0
            observation, _ = env.reset(seed=seed * 1_000_000 + episode)
            state = features_v3(observation, env)
        if step % 1_000 == 0 or step == settings.train_steps:
            history.append(dict(model=name, training_seed=seed, step=step, episodes=episode,
                                mean_recent_reward=float(np.mean(recent_rewards[-100:])) if recent_rewards else np.nan,
                                recent_success=float(np.mean(recent_success[-100:])) if recent_success else np.nan,
                                updates=updates))

    online.eval()
    return online, history


def train_tabular(orders, scale, seed, settings):
    """Give the team's refined Q-Learning implementation the same step budget."""
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    env = make_env(orders, scale, map_seed=0)
    agent = TabularQLearningAgent(alpha=0.10, gamma=settings.gamma,
                                  weight_decay=0.0, state_mode="refined")
    episode = 0
    env.reset(seed=seed * 1_000_000 + episode)
    recent_rewards, recent_success, history = [], [], []
    episode_reward = 0.0

    for step in range(1, settings.train_steps + 1):
        fraction = min(1.0, (step - 1) / settings.epsilon_decay_steps)
        epsilon = settings.epsilon_start + fraction * (settings.epsilon_end - settings.epsilon_start)
        state = agent.extract_state(env)
        action = agent.select_action(state, epsilon, rng)
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = agent.extract_state(env)
        agent.update(state, action, reward, next_state, done)
        episode_reward += reward
        if done:
            recent_rewards.append(episode_reward)
            recent_success.append(float(info["delivered"]))
            episode += 1
            episode_reward = 0.0
            env.reset(seed=seed * 1_000_000 + episode)
        if step % 1_000 == 0 or step == settings.train_steps:
            history.append(dict(model=TABULAR_NAME, training_seed=seed, step=step,
                                episodes=episode,
                                mean_recent_reward=float(np.mean(recent_rewards[-100:])) if recent_rewards else np.nan,
                                recent_success=float(np.mean(recent_success[-100:])) if recent_success else np.nan,
                                updates=step))
    return agent, history


def evaluate_one(network, orders, scale, *, name, training_seed, partition, episodes):
    """Use identical orders, map and episode seeds for every trained variant."""
    map_seed, offset = (101, 100_000) if partition == "validation" else (202, 200_000)
    env = make_env(orders, scale, map_seed)
    records = []
    if name == TABULAR_NAME:
        tabular_policy = make_ql_policy(network)
    else:
        network.eval()
    for index in range(episodes):
        episode_seed = offset + index
        observation, _ = env.reset(seed=episode_seed)
        reward_total, hotspot_steps = 0.0, 0
        while True:
            if name == TABULAR_NAME:
                action = tabular_policy(env)
            else:
                state = features_v3(observation, env)
                with torch.no_grad():
                    action = int(network(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            hotspot_steps += int(env.hotspots[env.pos])
            if terminated or truncated:
                break
        delivered = bool(info["delivered"])
        records.append(dict(model=name, training_seed=training_seed, partition=partition,
                            episode_seed=episode_seed, delivered=int(delivered),
                            on_time=int(delivered and info["elapsed_min"] <= env.benchmark_min),
                            reward=reward_total, elapsed_min=info["elapsed_min"],
                            delivery_min=info["elapsed_min"] if delivered else np.nan,
                            steps=info["steps"], hotspot_steps=hotspot_steps))
    return records


def write_results(output, partition, records):
    episodes = pd.DataFrame(records)
    episodes.to_csv(output / f"{partition}_episodes.csv", index=False)
    by_seed = episodes.groupby(["model", "training_seed"], as_index=False)[list(METRICS)].mean()
    by_seed.to_csv(output / f"{partition}_per_seed.csv", index=False)
    summary = by_seed.groupby("model")[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / f"{partition}_summary.csv", index=False)
    return by_seed


def write_portable_summary(output, manifest):
    """Keep a small, Git-reviewable result while detailed run artifacts stay local."""
    episodes = pd.read_csv(output / "test_episodes.csv")
    rates = episodes.groupby("model")[["delivered", "on_time"]].mean() * 100
    completed_times = episodes[episodes["delivered"] == 1].groupby("model")["delivery_min"].mean()
    comparison = rates.rename(columns={"delivered": "delivered_pct", "on_time": "on_time_pct"})
    comparison["completed_trip_min"] = completed_times
    comparison = comparison.reindex(MODEL_NAMES).reset_index()
    per_seed = pd.read_csv(output / "test_per_seed.csv")
    payload = dict(
        source_run=output.name,
        data_sha256=manifest["data_sha256"],
        seeds=manifest["seeds"],
        train_steps=manifest["settings"]["train_steps"],
        eval_episodes=manifest["eval_episodes"],
        test_caveat=manifest["test_caveat"],
        validation_winner=manifest["validation_winner"],
        comparison=json.loads(comparison.to_json(orient="records")),
        per_seed=json.loads(per_seed.to_json(orient="records")),
    )
    write_json(PORTABLE_SUMMARY, payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dataset_clean.csv")
    parser.add_argument("--output", type=Path, help="New run directory under dqn/result/v4")
    parser.add_argument("--train-steps", type=int, default=60_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny validation-only wiring check, not evidence")
    args = parser.parse_args(argv)
    if args.train_steps < 1 or args.eval_episodes < 1 or args.threads < 1:
        parser.error("train-steps, eval-episodes and threads must be positive")
    if not args.seeds or min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct non-negative integers")
    return args


def main(argv=None):
    args = parse_args(argv)
    settings = Settings(train_steps=args.train_steps)
    if args.smoke:
        settings = replace(settings, train_steps=128, hidden=32, batch_size=8,
                           warmup=16, target_update=32, epsilon_decay_steps=80)
        args.seeds, args.eval_episodes, args.validation_only = [11], 4, True
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    orders = load_orders(args.data)
    train_orders, validation_orders, test_orders = split_orders_three_way(orders)
    scale = training_scale(train_orders)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or OUTPUT_ROOT / f"four_way_{run_id}").resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("Output must be a new directory inside dqn/result/v4")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="running", run_id=run_id, smoke=args.smoke,
                    validation_only=args.validation_only, data_sha256=dataset_hash(args.data),
                    partitions=split_manifest((train_orders, validation_orders, test_orders)),
                    distance_scale=scale, model_factors={name: dict(double=flags[0], dueling=flags[1])
                                                   for name, flags in VARIANTS.items()},
                    parameter_counts={name: sum(p.numel() for p in ComparableQNetwork(
                        hidden=settings.hidden, dueling=flags[1]).parameters())
                        for name, flags in VARIANTS.items()},
                    settings=asdict(settings), seeds=args.seeds, eval_episodes=args.eval_episodes,
                    q_learning=dict(state_mode="refined", alpha=0.10, gamma=settings.gamma,
                                    weight_decay=0.0, evaluation_policy="team greedy policy with cycle prevention"),
                    train_map_seed=0, validation_map_seed=101, test_map_seed=202,
                    test_caveat="The team has inspected this historical test split before; it is not pristine.")
    write_json(output / "manifest.json", manifest)
    print(f"Results: {output.relative_to(ROOT)}", flush=True)

    networks, learning, validation = {}, [], []
    for name in MODEL_NAMES:
        for seed in args.seeds:
            started = time.perf_counter()
            if name == TABULAR_NAME:
                network, history = train_tabular(train_orders, scale, seed, settings)
                network.save(output / f"Q-Learning_seed{seed}.pkl")
            else:
                network, history = train_one(train_orders, scale, name, seed, settings)
                torch.save(dict(state_dict=network.state_dict(), model=name, training_seed=seed,
                                settings=asdict(settings), data_sha256=manifest["data_sha256"]),
                           output / f"{name.replace(' ', '_')}_seed{seed}.pt")
            networks[(name, seed)] = network
            learning.extend(history)
            validation.extend(evaluate_one(network, validation_orders, scale, name=name,
                                           training_seed=seed, partition="validation",
                                           episodes=args.eval_episodes))
            print(f"{name}, seed {seed}: {time.perf_counter() - started:.1f}s", flush=True)
    pd.DataFrame(learning).to_csv(output / "learning.csv", index=False)
    by_seed = write_results(output, "validation", validation)
    ranking = by_seed.groupby("model")[["delivered", "on_time", "reward"]].mean().sort_values(
        ["delivered", "on_time", "reward"], ascending=False, kind="stable")
    winner = str(ranking.index[0]) if not args.smoke and ranking.iloc[0]["delivered"] > 0 else None
    write_json(output / "selection.json", dict(validation_winner=winner, smoke_only=args.smoke,
               criterion="mean delivery success, then on-time rate, then reward across seeds; fixed before test",
               warning="This is a controlled architecture comparison, not the app's saved model selection."))
    print(f"Validation leader: {winner}" if winner else "No model selected from this run.", flush=True)

    if not args.validation_only:
        test = []
        for name in MODEL_NAMES:
            for seed in args.seeds:
                test.extend(evaluate_one(networks[(name, seed)], test_orders, scale, name=name,
                                         training_seed=seed, partition="test", episodes=args.eval_episodes))
        write_results(output, "test", test)
    manifest["status"] = "complete"
    manifest["validation_winner"] = winner
    write_json(output / "manifest.json", manifest)
    if not args.smoke and len(args.seeds) >= 5 and settings.train_steps >= 60_000:
        write_portable_summary(output, manifest)
    print("Completed. Compare per-seed and held-out results before making a performance claim.", flush=True)
    return output


if __name__ == "__main__":
    main()
