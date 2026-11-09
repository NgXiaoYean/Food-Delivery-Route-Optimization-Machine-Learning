"""
D3QN (Dueling Double DQN) on the MULTI-ORDER environment  --  Part A + Part B + final evidence
================================================================================================
A rider carries 1-3 orders; the agent chooses WHICH customer to reach next and WHICH cells to cross.
This file is self-contained (it does not import from dqn/), so the old single-order pipeline keeps working.

Three stages (run from the project root):
  Stage A  Part A  architecture comparison on the VALIDATION set:
                   standard DQN  vs  Double DQN (MLP)  vs  Dueling Double DQN       (same settings)
  Stage B  Part B  random search over lr, gamma, width, L2 weight decay, dropout, Polyak tau
                   scored on the VALIDATION set only  (the test set is never used for choosing)
  Stage C  Final   retrain the chosen config over several seeds, score ONCE on the TEST set, plus
                   two harder scenarios: (1) a new order arrives mid-trip, (2) a hotspot map never seen in training.
                   Reports mean +/- std over seeds and a per-number-of-customers breakdown.

Usage:
    python ddqn/ddqn_multi.py --quick            # ~2 min smoke test (tiny numbers, writes to result_multi_quick/)
    python ddqn/ddqn_multi.py --stage A
    python ddqn/ddqn_multi.py --stage B --n-configs 8
    python ddqn/ddqn_multi.py --stage C --seeds 5
    add --resume to continue an interrupted Stage B / Stage C instead of starting over
    python ddqn/ddqn_multi.py                    # all three stages, in order

Everything is saved as CSV / PNG / .pt in ddqn/result_multi/ so you can drop the numbers straight into the report.
"""
import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")  # no window needed; we only save figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from env import load_orders
from env_multi import (OBS_DIM_MULTI_V3, MultiDeliveryEnv, evaluate_multi, greedy_policy_multi,
                       planner_policy_multi, safe_argmax, split_orders_3way)

# ----------------------------------------------------------------------------- settings
BASE_CFG = dict(
    episodes=4000,            # training trips (multi-order trips are longer than single-order ones)
    eps_start=1.0, eps_end=0.05, eps_decay_episodes=2000,   # epsilon-greedy exploration, linear decay
    batch_size=64, buffer_size=50_000, warmup=1000,         # experience replay
    reward_scale=10.0,        # learn from reward/10 so Q-values stay small and stable
    grad_clip=5.0,
    seed=42,
)
# Default model settings used in Stage A (identical for all three architectures = fair comparison)
DEFAULT_MODEL = dict(lr=1e-3, gamma=0.95, hidden=128, weight_decay=0.0, dropout=0.0, tau=0.005)

# Stage B search space (random search samples from this; cheaper than trying every combination)
SEARCH_SPACE = dict(
    lr=[5e-4, 1e-3, 2e-3],
    gamma=[0.90, 0.95, 0.99],
    hidden=[64, 128, 256],
    weight_decay=[0.0, 1e-4, 1e-3],   # decoupled L2 regularisation (AdamW)
    dropout=[0.0, 0.05, 0.1],         # only used by the dueling network
    tau=[0.005, 0.01, 0.02],          # Polyak soft target update rate
)

VAL_SEED_OFFSET = 1000   # validation trips use seeds 1000.. ; test trips use seeds 0..  (never overlap)
EVAL_EPISODES = 500


# ----------------------------------------------------------------------------- networks
class MLPQNetwork(nn.Module):
    """Plain two-hidden-layer Q-network: state -> 4 Q-values."""

    def __init__(self, obs_dim=OBS_DIM_MULTI_V3, n_actions=4, hidden=128):
        super().__init__()
        if obs_dim <= 0 or hidden <= 0:
            raise ValueError("`obs_dim` and `hidden` must be positive.")
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: Q(s,a) = V(s) + ( A(s,a) - mean_a' A(s,a') ).
    V(s)   = how good is this situation (customers left, traffic, weather)?
    A(s,a) = how much better is each move than the average move (e.g. avoid the congested cell)?"""

    def __init__(self, obs_dim=OBS_DIM_MULTI_V3, n_actions=4, hidden=128, dropout=0.0):
        super().__init__()
        if obs_dim <= 0 or hidden <= 0:
            raise ValueError("`obs_dim` and `hidden` must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("`dropout` must be in [0, 1).")
        self.feature = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                                     nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_actions))

    def forward(self, x):
        f = self.feature(x)
        adv = self.advantage(f)
        return self.value(f) + (adv - adv.mean(dim=1, keepdim=True))   # mean-subtracted => identifiable


def build_net(cfg, training_net=True):
    if cfg["dueling"]:
        return DuelingQNetwork(hidden=cfg["hidden"], dropout=cfg["dropout"] if training_net else 0.0)
    return MLPQNetwork(hidden=cfg["hidden"])


class ReplayBuffer:
    """Stores past transitions; random sampling breaks the correlation between consecutive steps."""

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buf)


# ----------------------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agent(env, cfg, verbose=False):
    """Train one agent. cfg needs: dueling (bool), double (bool), lr, gamma, hidden, weight_decay,
    dropout, tau (None = hard update every target_update_steps), plus BASE_CFG keys.
    Returns (network in eval mode, list of episode rewards, list of episode success flags)."""
    set_seed(cfg["seed"])
    env.rng = np.random.default_rng(cfg["seed"])

    online, target = build_net(cfg, True), build_net(cfg, False)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimiser = optim.AdamW(online.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])  # decoupled L2
    buffer = ReplayBuffer(cfg["buffer_size"])
    total_steps, ep_rewards, ep_success = 0, [], []

    for ep in range(cfg["episodes"]):
        frac = min(1.0, ep / max(1, cfg["eps_decay_episodes"]))
        eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])

        env.reset()
        state = env.obs_v3()
        done, ep_r, delivered = False, 0.0, False
        while not done:
            if random.random() < eps:                       # explore
                action = random.randrange(4)
            else:                                           # exploit
                online.eval()
                with torch.no_grad():
                    action = int(online(torch.tensor(state).unsqueeze(0)).argmax(1))
                online.train()

            _, reward, terminated, truncated, info = env.step(action)
            next_state = env.obs_v3()
            done = terminated or truncated
            delivered = info["delivered"]
            # Only a real delivery of ALL orders stops bootstrapping; a timeout is just a cut-off.
            buffer.add(state, action, reward / cfg["reward_scale"], next_state, float(terminated))
            state, ep_r = next_state, ep_r + reward
            total_steps += 1

            if len(buffer) >= max(cfg["warmup"], cfg["batch_size"]):
                online.train()
                s, a, r, s2, d = buffer.sample(cfg["batch_size"])
                q = online(s).gather(1, a)                  # Q of the action actually taken
                with torch.no_grad():
                    if cfg["double"]:
                        # Double Q-learning: ONLINE net picks the next action, TARGET net scores it
                        online.eval()
                        best_a = online(s2).argmax(1, keepdim=True)
                        online.train()
                        best_next = target(s2).gather(1, best_a)
                    else:
                        # Standard DQN: target net both picks and scores (prone to over-estimation)
                        best_next = target(s2).max(1, keepdim=True)[0]
                    y = r + cfg["gamma"] * best_next * (1.0 - d)
                loss = nn.functional.smooth_l1_loss(q, y)   # Huber loss
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), cfg["grad_clip"])
                optimiser.step()

                if cfg.get("tau") is not None:              # Polyak soft update
                    for p_on, p_tg in zip(online.parameters(), target.parameters()):
                        p_tg.data.copy_(cfg["tau"] * p_on.data + (1.0 - cfg["tau"]) * p_tg.data)
                elif total_steps % cfg["target_update_steps"] == 0:   # periodic hard update
                    target.load_state_dict(online.state_dict())

        ep_rewards.append(ep_r)
        ep_success.append(float(delivered))
        # Stage B only: stop a config that is clearly not learning (it would run ~100 steps per episode for hours)
        if cfg.get("abort_if_failing") and ep + 1 == cfg["eps_decay_episodes"] + 500 and np.mean(ep_success[-250:]) < 0.5:
            print(f"    [ABORT] only {np.mean(ep_success[-250:]):.0%} of recent trips fully delivered at episode {ep + 1}; "
                  "stopping this config early (counted as a poor config).", flush=True)
            break
        if verbose and (ep + 1) % max(1, cfg["episodes"] // 8) == 0:
            print(f"    ep {ep + 1:5d} | eps={eps:.2f} | avg reward (last 250)={np.mean(ep_rewards[-250:]):7.1f} "
                  f"| all-delivered={np.mean(ep_success[-250:]):.0%}")
    online.eval()
    return online, ep_rewards, ep_success


def make_policy(net):
    """Greedy policy for evaluation. Uses the team's ONE safe_argmax rule (no wall bumps, no immediate backtrack)."""
    tracker = {"prev": None}

    def act(env):
        if env.steps == 0:
            tracker["prev"] = None
        net.eval()
        with torch.no_grad():
            q = net(torch.tensor(env.obs_v3()).unsqueeze(0)).squeeze(0).numpy()
        action = safe_argmax(q, env, tracker["prev"])
        tracker["prev"] = env.pos
        return action

    return act


def save_model(net, cfg, path):
    torch.save({"state_dict": net.state_dict(), "config": cfg}, path)


def load_model(path):
    """Reload a saved agent (use this when building the ensemble)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(ckpt["config"], training_net=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["config"]


# ----------------------------------------------------------------------------- data
def make_context(quick, shaping=0.0):
    data_path = BASE_DIR / "data" / "dataset_clean.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset missing at {data_path}. Run `python clean_data.py` first.")
    orders = load_orders(data_path)
    train, val, test = split_orders_3way(orders)
    # shaping > 0 adds a small potential-based distance bonus to the TRAINING env only (val/test stay unshaped)
    train_env_factory = lambda: MultiDeliveryEnv(train, shaping=shaping)
    max_km = train_env_factory().max_km        # every env uses the TRAIN distance scale
    return dict(
        factory=train_env_factory,
        val_env=MultiDeliveryEnv(val, max_km=max_km),
        test_env=MultiDeliveryEnv(test, max_km=max_km),
        dynamic_env=MultiDeliveryEnv(test, max_km=max_km, dynamic=True),            # new order arrives mid-trip
        unseen_map_env=MultiDeliveryEnv(test, max_km=max_km, map_seed=123),         # congestion map never seen in training
        sizes=(len(train), len(val), len(test)),
    )


def moving_avg(x, w=100):
    return pd.Series(x).rolling(w, min_periods=1).mean()


# ----------------------------------------------------------------------------- Stage A
def stage_a(ctx, out_dir, base):
    print("\n" + "=" * 78 + "\nSTAGE A (Part A): standard DQN vs Double DQN vs Dueling Double DQN  [VALIDATION set]\n" + "=" * 78)
    archs = [
        ("Standard DQN (MLP)", dict(dueling=False, double=False)),
        ("Double DQN (MLP)", dict(dueling=False, double=True)),
        ("Dueling Double DQN (D3QN)", dict(dueling=True, double=True)),
    ]
    rows, curves = [], {}
    for name, arch in archs:
        cfg = {**base, **DEFAULT_MODEL, **arch}
        t0 = time.time()
        print(f"\n[A] Training {name} ...")
        net, rewards, _ = train_agent(ctx["factory"](), cfg, verbose=True)
        res = evaluate_multi(make_policy(net), ctx["val_env"], EVAL_EPISODES, name, VAL_SEED_OFFSET)
        res["train_time_s"] = round(time.time() - t0, 1)
        rows.append(res)
        curves[name] = rewards
        print(f"    -> val reward {res['avg_reward']:.2f} | all delivered {res['success_rate']:.1%} | hotspots {res['avg_hotspot_steps']:.2f}")
    for name, pol in [("Greedy (nearest first)", greedy_policy_multi), ("Planner (oracle)", planner_policy_multi)]:
        rows.append(evaluate_multi(pol, ctx["val_env"], EVAL_EPISODES, name, VAL_SEED_OFFSET))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stageA_architecture_val.csv", index=False)
    print("\n" + df.round(3).to_string(index=False))

    plt.figure(figsize=(9, 4.5))
    for name, rewards in curves.items():
        plt.plot(moving_avg(rewards), label=name, lw=2 if "Dueling" in name else 1.3)
    plt.xlabel("Training episode"); plt.ylabel("Reward (100-episode moving avg)")
    plt.title("Multi-order: standard DQN vs Double DQN vs Dueling Double DQN")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.4); plt.tight_layout()
    plt.savefig(out_dir / "stageA_learning_curves.png", dpi=160); plt.close()
    return df


# ----------------------------------------------------------------------------- Stage B
def sample_configs(n, seed=0):
    rng, seen, out = random.Random(seed), set(), []
    n = min(n, 3 ** len(SEARCH_SPACE))
    while len(out) < n:
        c = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        key = tuple(c.items())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def stage_b(ctx, out_dir, base, n_configs, arch, resume=False):
    print("\n" + "=" * 78 + f"\nSTAGE B (Part B): random search, {n_configs} configs, arch={arch}  [VALIDATION set only]\n" + "=" * 78)
    arch_flags = dict(dueling=(arch == "dueling"), double=True)
    rows, hp_by_id = [], {}
    csv_path = out_dir / "stageB_search_val.csv"
    done_ids = set()
    if resume and csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        done_ids = {int(r["config_id"]) for r in rows}
        print(f"[resume] {len(done_ids)} config(s) already finished: {sorted(done_ids)}")
    for i, hp in enumerate(sample_configs(n_configs), 1):
        hp_by_id[i] = hp   # keep the ORIGINAL python values (pandas would turn ints into floats)
        if i in done_ids:
            continue
        cfg = {**base, **hp, **arch_flags, "abort_if_failing": True}
        t0 = time.time()
        print(f"\n[B {i}/{n_configs}] {hp}")
        try:
            net, _, _ = train_agent(ctx["factory"](), cfg, verbose=True)   # live progress lines
            res = evaluate_multi(make_policy(net), ctx["val_env"], EVAL_EPISODES, f"cfg{i}", VAL_SEED_OFFSET)
        except Exception as exc:   # one bad config must not kill a multi-hour search
            print(f"    [WARN] config {i} failed: {exc}")
            continue
        row = {"config_id": i, **hp, **{k: v for k, v in res.items() if k != "agent"},
               "train_time_s": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"    -> val reward {res['avg_reward']:.2f} | all delivered {res['success_rate']:.1%} | "
              f"hotspots {res['avg_hotspot_steps']:.2f} | {row['train_time_s']}s")
        pd.DataFrame(rows).to_csv(csv_path, index=False)   # saved after every config

    if not rows:
        raise RuntimeError("Every search configuration failed; nothing to choose from.")
    df = pd.DataFrame(rows).sort_values("avg_reward", ascending=False)
    df.to_csv(out_dir / "stageB_search_val.csv", index=False)
    best_cfg = {**hp_by_id[int(df.iloc[0]["config_id"])], **arch_flags}
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nBest config (by VALIDATION reward) saved to {out_dir / 'best_config.json'}:\n{best_cfg}")
    return best_cfg


# ----------------------------------------------------------------------------- Stage C
def stage_c(ctx, out_dir, base, n_seeds, arch, resume=False, tag="", use_default=False):
    print("\n" + "=" * 78 + f"\nSTAGE C (final): {n_seeds} seeds, scored ONCE on the TEST set\n" + "=" * 78)
    cfg_path = out_dir / "best_config.json"
    if use_default:
        best = {**DEFAULT_MODEL, "dueling": arch == "dueling", "double": True}
        print(f"Using the DEFAULT settings (no search): {best}")
    elif cfg_path.exists():
        best = json.loads(cfg_path.read_text())
        print(f"Using the Stage B winner: {best}")
    else:
        best = {**DEFAULT_MODEL, "dueling": arch == "dueling", "double": True}
        print(f"[WARN] {cfg_path.name} not found; using the default settings instead: {best}")

    scenarios = {"normal test": ctx["test_env"],
                 "new order mid-trip": ctx["dynamic_env"],
                 "unseen hotspot map": ctx["unseen_map_env"]}
    seed_rows, episode_frames = [], []
    for s in range(n_seeds):
        cfg = {**base, **best, "seed": s}
        model_path = out_dir / f"ddqn_multi{tag}_seed{s}.pt"
        if resume and model_path.exists():
            print(f"\n[C] seed {s}: reusing saved model {model_path.name}")
            net, _ = load_model(model_path)
        else:
            print(f"\n[C] seed {s}: training ...", flush=True)
            net, _, _ = train_agent(ctx["factory"](), cfg, verbose=True)
            save_model(net, cfg, model_path)
        for scen, env in scenarios.items():
            res, eps_df = evaluate_multi(make_policy(net), env, EVAL_EPISODES, f"D3QN seed {s}", 0, return_episodes=True)
            seed_rows.append({"scenario": scen, "train_seed": s, **{k: v for k, v in res.items() if k != "agent"}})
            episode_frames.append(eps_df.assign(scenario=scen, train_seed=s))
            print(f"    {scen:<20} reward {res['avg_reward']:7.2f} | all delivered {res['success_rate']:.1%} | hotspots {res['avg_hotspot_steps']:.2f}")

    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(out_dir / f"stageC{tag}_per_seed_test.csv", index=False)
    metrics = ["success_rate", "on_time_order_rate", "avg_reward", "avg_late_min", "avg_hotspot_steps"]
    summary = per_seed.groupby("scenario")[metrics].agg(["mean", "std"]).round(3)
    summary.to_csv(out_dir / f"stageC{tag}_summary_mean_std.csv")
    print("\n=== mean and std over seeds (TEST set) ===\n" + summary.to_string())

    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes.to_csv(out_dir / f"stageC{tag}_episode_details.csv", index=False)   # use for bootstrap CIs / error analysis
    by_k = (episodes[episodes["scenario"] == "normal test"].groupby("k")[["reward", "late_min", "hotspot_steps"]]
            .mean().round(2).assign(trips=episodes[episodes["scenario"] == "normal test"].groupby("k").size()))
    by_k.to_csv(out_dir / f"stageC{tag}_by_number_of_customers.csv")
    print("\n=== normal test, by number of customers on the trip (mean over seeds) ===\n" + by_k.to_string())

    base_rows = []   # reference lines for every scenario
    for scen, env in scenarios.items():
        for name, pol in [("Greedy (nearest first)", greedy_policy_multi), ("Planner (oracle)", planner_policy_multi)]:
            base_rows.append({"scenario": scen, **evaluate_multi(pol, env, EVAL_EPISODES, name, 0)})
    pd.DataFrame(base_rows).to_csv(out_dir / f"stageC{tag}_baselines_test.csv", index=False)
    print("\n=== baselines on the same test trips ===\n" + pd.DataFrame(base_rows).round(3).to_string(index=False))


# ----------------------------------------------------------------------------- main
def main():
    global EVAL_EPISODES
    ap = argparse.ArgumentParser(description="D3QN on the multi-order delivery environment")
    ap.add_argument("--stage", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (a few minutes)")
    ap.add_argument("--n-configs", type=int, default=8, help="Stage B: number of random configs")
    ap.add_argument("--seeds", type=int, default=5, help="Stage C: number of training seeds")
    ap.add_argument("--arch", choices=["dueling", "vanilla"], default="dueling",
                    help="architecture used in Stage B/C (pick the winner of Stage A)")
    ap.add_argument("--episodes", type=int, default=None, help="override training episodes")
    ap.add_argument("--tag", default="", help="Stage C: suffix for output files, e.g. _default")
    ap.add_argument("--use-default-config", action="store_true",
                    help="Stage C: use the untuned default settings instead of the Stage B winner (before/after comparison)")
    ap.add_argument("--resume", action="store_true",
                    help="Stage B: skip configs already in the CSV. Stage C: reuse saved seed models.")
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="training-only distance bonus (try 1.0); 0 = off. Val/test are never shaped.")
    args = ap.parse_args()

    base = dict(BASE_CFG)
    n_configs, n_seeds = args.n_configs, args.seeds
    out_dir = BASE_DIR / "ddqn" / "result_multi"
    if args.quick:
        base.update(episodes=60, warmup=200, eps_decay_episodes=30)
        EVAL_EPISODES, n_configs, n_seeds = 20, 2, 2
        out_dir = BASE_DIR / "ddqn" / "result_multi_quick"
    if args.episodes:
        base["episodes"] = args.episodes
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ctx = make_context(args.quick, args.shaping)
    except Exception as exc:
        print(f"[ERROR] could not set up data and environments: {exc}")
        sys.exit(1)
    print(f"Orders: train {ctx['sizes'][0]}, validation {ctx['sizes'][1]}, test {ctx['sizes'][2]}  | outputs -> {out_dir}")

    if args.stage in ("A", "all"):
        stage_a(ctx, out_dir, base)
    if args.stage in ("B", "all"):
        stage_b(ctx, out_dir, base, n_configs, args.arch, args.resume)
    if args.stage in ("C", "all"):
        stage_c(ctx, out_dir, base, n_seeds, args.arch, args.resume, args.tag, args.use_default_config)


if __name__ == "__main__":
    main()
