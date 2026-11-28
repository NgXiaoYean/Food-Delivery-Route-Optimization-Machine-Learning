"""Read-only, shared scenario evaluation for the AI grid and route player."""

import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from env import DeliveryEnv, MINUTES_PER_STEP


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "compare" / "checkpoints"
ACTION_NAMES = ("Up", "Down", "Left", "Right")
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def catalog():
    path = ARTIFACTS / "catalog.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def learning_history():
    path = ARTIFACTS / "learning.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def validation_progress():
    path = ARTIFACTS / "validation_progress.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def load_policy(model, stage):
    """Use the original inference interfaces, with no new teaching model."""
    path = ARTIFACTS / stage["file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    if model == "Q-Learning":
        from q_learning.q_learning_pipeline import TabularQLearningAgent, make_ql_policy
        return make_ql_policy(TabularQLearningAgent.load(path))
    import torch
    from compare.part_a_comparison import ComparableQNetwork

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model") != model:
        raise ValueError(f"Checkpoint model mismatch: {path}")
    settings = checkpoint["settings"]
    net = ComparableQNetwork(hidden=int(settings["hidden"]),
                             dueling=model in ("Dueling DQN", "D3QN"))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    def act(env):
        from dqn.dqn_v3 import features_v3
        with torch.no_grad():
            features = torch.from_numpy(features_v3(env._obs(), env)).unsqueeze(0)
            return int(net(features).argmax(dim=1).item())

    return act


def _copy_scenario(orders, scenario):
    source = scenario["env"]
    env = DeliveryEnv(orders, grid_size=source.n, max_steps=source.max_steps,
                      max_km=source.max_km)
    env.reset(seed=0)
    env.hotspots = source.hotspots.copy()
    for field in ("traffic", "weather", "multi", "benchmark_min"):
        setattr(env, field, getattr(source, field))
    env.pos = tuple(scenario["start"])
    env.dest = tuple(scenario["dest"])
    env.steps = 0
    env.elapsed_min = 0.0
    return env


def replay(policy, orders, scenario):
    """Return the exact episode used by both the static grid and player."""
    env = _copy_scenario(orders, scenario)
    start, dest = tuple(env.pos), tuple(env.dest)
    points, moves = [list(start)], []
    total = 0.0
    while True:
        before = tuple(env.pos)
        old_distance = abs(before[0] - dest[0]) + abs(before[1] - dest[1])
        action = int(policy(env))
        if action not in range(4):
            raise ValueError(f"Invalid routing action: {action}")
        _, reward, terminal, truncated, info = env.step(action)
        total += reward
        new_distance = abs(env.pos[0] - dest[0]) + abs(env.pos[1] - dest[1])
        feedback = ("Delivered" if info["delivered"] else "Time limit" if truncated else
                    "Map edge" if env.pos == before else
                    "Traffic jam" if env.hotspots[env.pos] else
                    "Closer to customer" if new_distance < old_distance else "Moved away")
        moves.append({"action": ACTION_NAMES[action], "reward": round(reward, 2),
                      "total": round(total, 2), "feedback": feedback,
                      "minutes": round(info["elapsed_min"], 1)})
        points.append(list(env.pos))
        if terminal or truncated:
            break
    minimum = abs(start[0] - dest[0]) + abs(start[1] - dest[1])
    delivered = bool(info["delivered"])
    return {"start": list(start), "dest": list(dest), "points": points, "moves": moves,
            "hotspots": np.argwhere(env.hotspots).tolist(), "traffic": env.traffic,
            "weather": env.weather, "min_steps": minimum, "steps": info["steps"],
            "minutes": round(info["elapsed_min"], 1), "reward": round(total, 2),
            "delivered": delivered, "efficiency_pct": round(
                100 * minimum / max(1, info["steps"]), 1) if delivered else 0.0}


def shortest_distance_policy(env):
    """Manhattan minimum-step route; congestion does not block cells."""
    r, c = env.pos
    dr, dc = env.dest
    if r != dr:
        return 1 if dr > r else 0
    return 3 if dc > c else 2


def fastest_time_policy(env):
    """Dijkstra with the simulator's actual per-move minute formula."""
    base = MINUTES_PER_STEP * (1 + 0.25 * env.traffic + 0.15 * env.weather)
    dist = {env.pos: 0.0}
    first = {env.pos: None}
    heap = [(0.0, env.pos)]
    while heap:
        elapsed, cell = heapq.heappop(heap)
        if elapsed > dist[cell]:
            continue
        if cell == env.dest:
            return first[cell] if first[cell] is not None else 0
        for action, (dr, dc) in enumerate(MOVES):
            nr, nc = cell[0] + dr, cell[1] + dc
            if not (0 <= nr < env.n and 0 <= nc < env.n):
                continue
            cost = base * (2 if env.hotspots[nr, nc] else 1)
            new_cost = elapsed + cost
            if new_cost < dist.get((nr, nc), float("inf")):
                dist[(nr, nc)] = new_cost
                first[(nr, nc)] = action if first[cell] is None else first[cell]
                heapq.heappush(heap, (new_cost, (nr, nc)))
    return 0


def grid_svg(result):
    """Identical responsive 10x10 geometry for every planner card."""
    hot = {tuple(cell) for cell in result["hotspots"]}
    cells = []
    for r in range(10):
        for c in range(10):
            x, y = c * 24, r * 24
            cells.append(f'<rect x="{x}" y="{y}" width="24" height="24" '
                         f'fill="{"#fee2e2" if (r,c) in hot else "#f8fafc"}" stroke="#dce3ed"/>')
            if (r, c) in hot:
                cells.append(f'<text x="{x+12}" y="{y+17}" text-anchor="middle" '
                             'fill="#ef4444" font-size="15">×</text>')
    polyline = " ".join(f"{p[1]*24+12},{p[0]*24+12}" for p in result["points"])
    cells.append(f'<polyline points="{polyline}" fill="none" stroke="#2563eb" '
                 'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    for point, color, label in ((result["start"], "#15803d", "R"),
                                (result["dest"], "#1d4ed8", "C")):
        x, y = point[1] * 24 + 12, point[0] * 24 + 12
        cells.append(f'<circle cx="{x}" cy="{y}" r="10" fill="{color}"/>')
        cells.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" fill="white" '
                     f'font-size="10" font-weight="700">{label}</text>')
    return ('<svg viewBox="0 0 240 240" role="img" aria-label="Delivery route grid" '
            'style="display:block;width:100%;max-width:260px;height:auto;margin:0 auto">'
            + "".join(cells) + '</svg>')
