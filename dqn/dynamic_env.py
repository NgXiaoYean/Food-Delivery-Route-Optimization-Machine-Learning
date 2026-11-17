"""DQN-only, single-order simulator. Shared DeliveryEnv is never modified.

Events are synthetic, not measured traffic forecasts. A move is charged using
conditions at departure; events crossed during that move are visible on arrival.
Historical Time_taken is a benchmark proxy, not a promised customer deadline.
"""
from dataclasses import dataclass

import numpy as np

from env import DeliveryEnv

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
FEATURE_DIMS = {"baseline": 15, "deadline": 30}


@dataclass(frozen=True)
class Conditions:
    minute: float
    traffic: int
    weather: int
    hotspots: np.ndarray


class DynamicDeliveryEnv(DeliveryEnv):
    """Finite delivery task: delivery AND exhausted step budget terminate it."""

    def __init__(self, orders, *, max_km, scenario="dynamic", reward_mode="time",
                 map_seeds=tuple(range(64)), grid_size=10, max_steps=60):
        if scenario not in ("static", "dynamic", "stress"):
            raise ValueError("scenario must be static, dynamic or stress")
        if reward_mode not in ("legacy", "time"):
            raise ValueError("reward_mode must be legacy or time")
        if not map_seeds or grid_size < 2 or max_steps < 1:
            raise ValueError("Need map seeds, grid_size >= 2 and max_steps >= 1")
        if not np.isfinite(max_km) or max_km <= 0 or orders.empty:
            raise ValueError("Need nonempty orders and a positive training distance scale")
        super().__init__(orders, grid_size=grid_size, max_steps=max_steps)
        self.max_km = float(max_km)
        self.scenario, self.reward_mode = scenario, reward_mode
        self.map_seeds = tuple(int(s) for s in map_seeds)
        self.episode_rng = np.random.default_rng(42)
        self.time_scale = max_steps * 8.8  # maximum possible per-move duration

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = int(self.episode_rng.integers(0, 2**31))
        self.episode_seed = int(seed)
        # Base reset controls only the order and endpoints, independent of events.
        super().reset(seed=self.episode_seed)
        chooser = np.random.default_rng(np.random.SeedSequence([seed, 17]))
        self.map_seed = int(chooser.choice(self.map_seeds))
        map_rng = np.random.default_rng(self.map_seed)
        initial = map_rng.random((self.n, self.n)) < (0.30 if self.scenario == "stress" else 0.15)
        event_rng = np.random.default_rng(np.random.SeedSequence([seed, 29]))
        self.timeline = [Conditions(0.0, self.traffic, self.weather, initial.copy())]
        if self.scenario != "static":
            interval = 4.0 if self.scenario == "stress" else 8.0
            traffic, weather, hot = self.traffic, self.weather, initial.copy()
            # Fixed incident cell chosen from the initial direct corridor, never
            # from the agent's actual route. It appears then clears on the clock.
            r, c = self.pos
            corridor = []
            while (r, c) != self.dest:
                if r != self.dest[0]:
                    r += 1 if r < self.dest[0] else -1
                else:
                    c += 1 if c < self.dest[1] else -1
                corridor.append((r, c))
            incident = corridor[len(corridor) // 2]
            for k, minute in enumerate(np.arange(interval, self.time_scale + interval, interval), 1):
                traffic = int(np.clip(traffic + event_rng.choice([-1, 0, 1]), 0, 3))
                if k % 3 == 0:  # weather evolves more slowly than traffic
                    weather = int(np.clip(weather + event_rng.choice([-1, 1]), 0, 3))
                toggle = event_rng.random(hot.shape) < (0.08 if self.scenario == "stress" else 0.03)
                hot = np.logical_xor(hot, toggle)
                if k == 1:
                    hot[incident] = True
                elif k == 3:
                    hot[incident] = False
                self.timeline.append(Conditions(float(minute), traffic, weather, hot.copy()))
        self.event_index = 0
        self._finished = False
        self._apply_conditions()
        self.previous_neighbor_minutes = self.neighbor_minutes()
        return self._obs(), {"map_seed": self.map_seed, "episode_seed": seed}

    def conditions_at(self, minute):
        """Read-only lookup: no RNG draws and no dependence on policy actions."""
        index = max(0, int(np.searchsorted([x.minute for x in self.timeline], minute, side="right")) - 1)
        return self.timeline[min(index, len(self.timeline) - 1)]

    def _apply_conditions(self):
        item = self.conditions_at(self.elapsed_min)
        self.traffic, self.weather = item.traffic, item.weather
        self.hotspots = item.hotspots.copy()
        self.event_index = sum(item.minute >= x.minute for x in self.timeline) - 1

    def valid_actions(self):
        r, c = self.pos
        return np.array([0 <= r + dr < self.n and 0 <= c + dc < self.n
                         for dr, dc in MOVES], dtype=bool)

    def cell_minutes(self, cell):
        return 2.0 * (1 + 0.25 * self.traffic + 0.15 * self.weather) * (2 if self.hotspots[cell] else 1)

    def neighbor_minutes(self):
        r, c = self.pos
        return np.array([self.cell_minutes((r + dr, c + dc)) if valid else 0.0
                         for (dr, dc), valid in zip(MOVES, self.valid_actions())], dtype=np.float32)

    def step(self, action):
        if self._finished:
            raise RuntimeError("reset() is required after the episode ends")
        if not isinstance(action, (int, np.integer)) or action not in range(4):
            raise ValueError("action must be an integer in 0..3")
        dr, dc = MOVES[action]
        bumped = not self.valid_actions()[action]
        if not bumped:
            self.pos = (self.pos[0] + dr, self.pos[1] + dc)
        self.steps += 1
        hot = bool(self.hotspots[self.pos])
        minutes = self.cell_minutes(self.pos)
        legacy_cost = 1 + 0.3 * self.traffic + 0.2 * self.weather + ((1 + self.traffic) if hot else 0)
        self.elapsed_min += minutes
        delivered = self.pos == self.dest
        failed = not delivered and self.steps >= self.max_steps
        lateness = max(0.0, self.elapsed_min - self.benchmark_min)
        terminal_reward = (50.0 - 0.5 * lateness) if delivered else (-20.0 if failed else 0.0)
        common_reward = -minutes - 2 * bumped + terminal_reward
        reward = common_reward if self.reward_mode == "time" else -legacy_cost - 2 * bumped + terminal_reward
        previous_event = self.event_index
        # Compare costs at the SAME new position before/after clock updates.
        self.previous_neighbor_minutes = self.neighbor_minutes()
        self._apply_conditions()
        self._finished = delivered or failed
        info = dict(delivered=delivered, failed=failed, elapsed_min=self.elapsed_min,
                    steps=self.steps, minutes=minutes, hotspot_entered=hot,
                    wall_bump=bumped, lateness_min=lateness, common_reward=common_reward,
                    events_crossed=self.event_index - previous_event)
        return self._obs(), float(reward), self._finished, False, info


def features(env, mode="deadline"):
    """15D v3-style baseline or 30D deadline/local-cost state (no future events)."""
    if mode not in FEATURE_DIMS:
        raise ValueError(f"Unknown feature mode: {mode}")
    obs = env._obs()
    dr, dc = obs[2] - obs[0], obs[3] - obs[1]
    r, c = env.pos
    valid = env.valid_actions()
    hot = [float(ok and env.hotspots[r + dy, c + dx])
           for (dy, dx), ok in zip(MOVES, valid)]
    base = np.concatenate([obs, [dr, dc, abs(dr) + abs(dc)], hot])
    if mode == "baseline":
        return base.astype(np.float32)
    costs = env.neighbor_minutes()
    extra = [env.elapsed_min / env.time_scale,
             env.benchmark_min / env.time_scale,
             (env.benchmark_min - env.elapsed_min) / env.time_scale]
    return np.concatenate([base, extra, valid.astype(float), costs / 8.8,
                           (costs - env.previous_neighbor_minutes) / 8.8]).astype(np.float32)
