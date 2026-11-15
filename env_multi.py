"""
Multi-order delivery environment (extension of env.DeliveryEnv).

Real-world idea: a rider carries 1-3 orders at once and must decide BOTH which customer to
visit next AND which cells to cross, while traffic/weather and congested cells change the cost.

Design choices (state these as assumptions in the report):
  * Number of orders on the trip  k = 1 + min(multiple_deliveries, 2)   -> 1, 2 or 3 orders.
    (multiple_deliveries is a real column of the Zomato data; it was unused before.)
  * Extra orders take their own distance_km and Time_taken from other random dataset rows, so
    each customer has its own destination and its own deadline (minutes from trip start).
  * All orders are already picked up (pickup is not modelled). Traffic/weather are shared.
  * Reward: same step costs as DeliveryEnv; +50 per delivered order minus 0.5 per late minute
    for THAT order; -20 on timeout. Episode ends when every order is delivered.
  * Slots in the observation are sorted by current Manhattan distance (slot 0 = nearest), so the
    network does not have to learn a fixed customer order.

Nothing in env.py is changed, so all single-order results stay valid.
"""
import heapq
from itertools import permutations

import numpy as np

from env import DeliveryEnv, MINUTES_PER_STEP

K_MAX = 3
# Scoring / timing rules. The DEFAULTS reproduce the original environment exactly; the sensitivity and
# calibration experiments (ddqn/sensitivity_multi.py) change them one at a time.
DEFAULT_RULES = dict(
    c_traffic=0.3,    # move cost per traffic level
    c_weather=0.2,    # move cost per weather level
    m_traffic=0.25,   # extra minutes factor per traffic level
    m_weather=0.15,   # extra minutes factor per weather level
    hot_extra=1.0,    # fixed extra cost of entering a congested cell
    hot_traffic=1.0,  # extra cost per traffic level inside a congested cell
    hot_minutes=2.0,  # minutes multiplier inside a congested cell
    late_w=0.5,       # penalty per late minute
    deliver_bonus=50.0,
    wall_pen=2.0,
    timeout_pen=20.0,
)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
OBS_DIM_MULTI = 2 + 4 * K_MAX + 4            # 18 numbers: position, 3 slots x (d_row, d_col, distance, active), 4 globals
OBS_DIM_MULTI_V3 = OBS_DIM_MULTI + 4         # + 4 neighbour-hotspot flags = 22


class MultiDeliveryEnv(DeliveryEnv):
    def __init__(self, orders, grid_size=10, max_steps=100, map_seed=0, hotspot_frac=0.15, max_km=None, dynamic=False, shaping=0.0, shaping_gamma=0.95, rules=None):
        super().__init__(orders, grid_size, max_steps, map_seed, hotspot_frac)
        self.rules = {**DEFAULT_RULES, **(rules or {})}
        self.dynamic = dynamic  # True = a new order can appear MID-TRIP (use for testing re-planning)
        self.shaping, self.shaping_gamma = shaping, shaping_gamma  # >0: potential-based shaping (training only)
        if max_km is not None:  # reuse the TRAIN distance scale for val/test envs (avoids a tiny scale mismatch)
            self.max_km = float(max_km)
        self.n_actions = 4
        if hasattr(self, "observation_space"):  # only when gymnasium is installed
            from gymnasium import spaces
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM_MULTI,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _target_cells(self, distance_km):
        """Same km -> Manhattan-cells rule as DeliveryEnv.reset."""
        return int(np.clip(round(distance_km / self.max_km * (2 * self.n - 2)), 3, 2 * self.n - 2))

    def _place_dest(self, distance_km, taken):
        target = self._target_cells(distance_km)
        cells = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) != self.pos and (r, c) not in taken]
        dist = np.array([abs(r - self.pos[0]) + abs(c - self.pos[1]) for r, c in cells])
        best = np.where(np.abs(dist - target) == np.abs(dist - target).min())[0]
        return cells[int(self.rng.choice(best))]

    @property
    def active(self):
        """Indices of orders not yet delivered."""
        return [i for i, d in enumerate(self.delivered_flags) if not d]

    @property
    def dest(self):
        """Nearest remaining customer (keeps old code that reads env.dest working)."""
        act = self.active
        if not act:
            return self.pos
        return min((self.dests[i] for i in act), key=lambda d: abs(d[0] - self.pos[0]) + abs(d[1] - self.pos[1]))

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = self.orders.iloc[int(self.rng.integers(len(self.orders)))]
        self.traffic, self.weather = int(o["traffic"]), int(o["weather"])
        self.multi = int(o["multi"])
        self.k = 1 + min(self.multi, K_MAX - 1)

        self.pos = (int(self.rng.integers(self.n)), int(self.rng.integers(self.n)))
        self.dests, self.deadlines = [], []
        rows = [o] + [self.orders.iloc[int(self.rng.integers(len(self.orders)))] for _ in range(self.k - 1)]
        for row in rows:
            self.dests.append(self._place_dest(row["distance_km"], set(self.dests)))
            self.deadlines.append(float(row["time_taken"]))
        self.benchmark_min = max(self.deadlines)  # kept so old metrics that read it do not crash
        self.delivered_flags = [False] * self.k
        self.steps, self.elapsed_min, self.late_total = 0, 0.0, 0.0
        # Dynamic mode: one extra order is placed on the rider's phone at a random early step (3..8)
        self.arrival_step, self._late_row = None, None
        if self.dynamic and self.k < K_MAX:
            self.arrival_step = int(self.rng.integers(3, 9))
            self._late_row = self.orders.iloc[int(self.rng.integers(len(self.orders)))]
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def _phi(self):
        """Potential = minus distance (in cells) to the nearest remaining customer; 0 when none are left."""
        if not self.active:
            return 0.0
        return -self.shaping * (abs(self.dest[0] - self.pos[0]) + abs(self.dest[1] - self.pos[1]))

    def step(self, action):
        R = self.rules
        phi_old = self._phi() if self.shaping else 0.0
        r, c = self.pos
        dr, dc = MOVES[int(action)]
        nr, nc = r + dr, c + dc
        reward = 0.0
        if not (0 <= nr < self.n and 0 <= nc < self.n):
            nr, nc = r, c
            reward -= R["wall_pen"]
        self.pos = (nr, nc)
        self.steps += 1

        move_cost = 1.0 + R["c_traffic"] * self.traffic + R["c_weather"] * self.weather
        minutes = MINUTES_PER_STEP * (1 + R["m_traffic"] * self.traffic + R["m_weather"] * self.weather)
        if self.hotspots[nr, nc]:
            move_cost += R["hot_extra"] + R["hot_traffic"] * self.traffic
            minutes *= R["hot_minutes"]
        reward -= move_cost
        self.elapsed_min += minutes

        newly = 0
        for i in self.active:
            if self.dests[i] == self.pos:
                self.delivered_flags[i] = True
                newly += 1
                late = max(0.0, self.elapsed_min - self.deadlines[i])
                self.late_total += late
                reward += R["deliver_bonus"] - R["late_w"] * late

        if self.arrival_step == self.steps and self._late_row is not None:  # new order arrives
            self.dests.append(self._place_dest(self._late_row["distance_km"], set(self.dests)))
            self.deadlines.append(self.elapsed_min + float(self._late_row["time_taken"]))  # clock starts on arrival
            self.delivered_flags.append(False)
            self.k += 1
            self._late_row = None

        terminated = all(self.delivered_flags)
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            reward -= R["timeout_pen"]
        if self.shaping:  # F = gamma*phi(s') - phi(s): helps learning, does not change the best policy
            reward += self.shaping_gamma * (0.0 if terminated else self._phi()) - phi_old

        info = {"delivered": terminated, "n_delivered": sum(self.delivered_flags), "k": self.k,
                "elapsed_min": self.elapsed_min, "steps": self.steps, "late_total": self.late_total,
                "newly_delivered": newly}
        return self._obs(), reward, terminated, truncated, info

    # ---------------------------------------------------------- observations
    def _slots(self):
        """Remaining customers sorted nearest-first, padded to K_MAX."""
        act = sorted(self.active, key=lambda i: abs(self.dests[i][0] - self.pos[0]) + abs(self.dests[i][1] - self.pos[1]))
        return [self.dests[i] for i in act]

    def _obs(self):
        n = self.n - 1
        v = [self.pos[0] / n, self.pos[1] / n]
        slots = self._slots()
        for j in range(K_MAX):
            if j < len(slots):
                drow, dcol = slots[j][0] - self.pos[0], slots[j][1] - self.pos[1]
                v += [drow / n, dcol / n, (abs(drow) + abs(dcol)) / (2 * n), 1.0]
            else:
                v += [0.0, 0.0, 0.0, 0.0]
        v += [self.traffic / 3, self.weather / 3, len(slots) / K_MAX, 1 - self.steps / self.max_steps]
        return np.array(v, dtype=np.float32)

    def hotspot_flags(self):
        r, c = self.pos
        return np.array([float(0 <= r + dr < self.n and 0 <= c + dc < self.n and self.hotspots[r + dr, c + dc])
                         for dr, dc in MOVES], dtype=np.float32)

    def obs_v3(self):
        """Observation + 4 neighbour-hotspot flags (same idea as dqn_v3.features_v3)."""
        return np.concatenate([self._obs(), self.hotspot_flags()]).astype(np.float32)

    def discrete_state(self):
        """Small state for tabular Q-learning: direction to the nearest customer + local hazards."""
        (r, c), (dr, dc) = self.pos, self.dest
        sgn = lambda x: 0 if x == 0 else (1 if x > 0 else -1)
        nb = []
        for mr, mc in MOVES:
            nr, nc = r + mr, c + mc
            nb.append(2 if not (0 <= nr < self.n and 0 <= nc < self.n) else int(self.hotspots[nr, nc]))
        return (sgn(dr - r), sgn(dc - c), int(abs(dr - r) >= abs(dc - c)), int(self.traffic >= 2),
                len(self.active), *nb)



# ============================================================ shared helpers (whole team)
def split_orders_3way(orders, val_frac=0.15, test_frac=0.15, seed=42):
    """Train / validation / test split of the cleaned orders (default 70/15/15, fixed seed).
    Tune on VALIDATION (seed_offset=1000); touch TEST once at the end (seed_offset=0)."""
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("val_frac and test_frac must be in (0,1) and sum to less than 1.")
    test = orders.sample(frac=test_frac, random_state=seed)
    rest = orders.drop(test.index)
    val = rest.sample(frac=val_frac / (1.0 - test_frac), random_state=seed)
    train = rest.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def safe_argmax(q_values, env, prev_pos, wall_pen=10.0, back_pen=1.0):
    """ONE action-selection rule for EVERY model: best Q-value, but never bump the map edge
    and discourage immediate backtracking. Use it for all agents so the comparison is fair."""
    q = np.array(q_values, dtype=np.float64).copy()
    r, c = env.pos
    for a, (dr, dc) in enumerate(MOVES):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < env.n and 0 <= nc < env.n):
            q[a] -= wall_pen
        elif (nr, nc) == prev_pos:
            q[a] -= back_pen
    return int(np.argmax(q))

# ============================================================ baselines (multi-order)
def greedy_policy_multi(env):
    """Visit the nearest customer, straight line (row first), ignoring congestion."""
    (r, c), (dr, dc) = env.pos, env.dest
    if r != dr:
        return 1 if dr > r else 0
    return 3 if dc > c else 2


def _dijkstra_all(env, src):
    """Cheapest cost from src to every cell, plus the first action on that path."""
    R = getattr(env, "rules", DEFAULT_RULES)
    base = 1.0 + R["c_traffic"] * env.traffic + R["c_weather"] * env.weather
    dist, first = {src: 0.0}, {src: None}
    heap = [(0.0, src)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > dist[(r, c)]:
            continue
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.n and 0 <= nc < env.n):
                continue
            nd = d + base + (R["hot_extra"] + R["hot_traffic"] * env.traffic if env.hotspots[nr, nc] else 0.0)
            if nd < dist.get((nr, nc), 1e18):
                dist[(nr, nc)] = nd
                first[(nr, nc)] = a if first[(r, c)] is None else first[(r, c)]
                heapq.heappush(heap, (nd, (nr, nc)))
    return dist, first


def planner_policy_multi(env):
    """Oracle: try every visiting order of the remaining customers (<= 3! = 6), follow the cheapest."""
    act = env.active
    if not act:
        return 0
    nodes = {"pos": env.pos, **{i: env.dests[i] for i in act}}
    runs = {key: _dijkstra_all(env, cell) for key, cell in nodes.items()}
    best_cost, best_first = 1e18, None
    for order in permutations(act):
        cost, here = 0.0, "pos"
        for i in order:
            cost += runs[here][0][nodes[i]]
            here = i
        if cost < best_cost:
            best_cost, best_first = cost, order[0]
    a = runs["pos"][1][nodes[best_first]]
    return 0 if a is None else a


# ============================================================ evaluation (multi-order)
def evaluate_multi(policy_fn, env, episodes=500, name="agent", seed_offset=0, return_episodes=False):
    """Same seeds for every model (seed_offset+0 .. seed_offset+episodes-1).
    Use seed_offset=1000 on the VALIDATION env while tuning, seed_offset=0 on the TEST env once at the end.
    Success = ALL orders delivered before timeout."""
    wins, orders_total, orders_done, on_time = 0, 0, 0, 0
    rewards, steps, minutes, hot_counts, late = [], [], [], [], []
    ep_rows = []
    for ep in range(episodes):
        env.reset(seed=seed_offset + ep)
        done, ep_r, hot, info = False, 0.0, 0, {}
        delivered_at = {}
        while not done:
            action = int(policy_fn(env))
            if action not in (0, 1, 2, 3):
                raise ValueError(f"Policy '{name}' returned invalid action {action}")
            before = list(env.delivered_flags)
            _, rew, term, trunc, info = env.step(action)
            ep_r += rew
            hot += int(env.hotspots[env.pos])
            for i, (b, a) in enumerate(zip(before, env.delivered_flags)):
                if a and not b:
                    delivered_at[i] = env.elapsed_min
            done = term or trunc
        deadlines = env.deadlines  # read after the episode: dynamic mode may have added an order
        wins += int(info["delivered"])
        orders_total += env.k
        orders_done += info["n_delivered"]
        on_time += sum(1 for i, t in delivered_at.items() if t <= deadlines[i])
        rewards.append(ep_r); steps.append(info["steps"]); hot_counts.append(hot); late.append(info["late_total"])
        ep_rows.append({"seed": seed_offset + ep, "k": env.k, "reward": ep_r, "steps": info["steps"],
                        "delivered_all": int(info["delivered"]), "late_min": info["late_total"], "hotspot_steps": hot})
        if info["delivered"]:
            minutes.append(info["elapsed_min"])
    summary = {
        "agent": name,
        "success_rate": wins / episodes,
        "orders_delivered_rate": orders_done / orders_total,
        "on_time_order_rate": on_time / orders_total,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps)),
        "avg_trip_time_min": float(np.mean(minutes)) if minutes else float("nan"),
        "avg_late_min": float(np.mean(late)),
        "avg_hotspot_steps": float(np.mean(hot_counts)),
    }
    if return_episodes:
        import pandas as pd
        return summary, pd.DataFrame(ep_rows)
    return summary


if __name__ == "__main__":
    import pandas as pd
    from env import load_orders
    from shared import split_orders

    orders = load_orders("./data/dataset_clean.csv")
    _, test_orders = split_orders(orders)
    env = MultiDeliveryEnv(test_orders)
    ks = np.bincount([1 + min(int(m), K_MAX - 1) for m in test_orders["multi"]], minlength=K_MAX + 1)[1:]
    print("test orders with k=1,2,3 customers:", ks.tolist())
    rows = [
        evaluate_multi(lambda e: int(e.rng.integers(4)), env, name="Random"),
        evaluate_multi(greedy_policy_multi, env, name="Greedy (nearest first)"),
        evaluate_multi(planner_policy_multi, env, name="Planner (best order + Dijkstra)"),
    ]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nDYNAMIC: a new order arrives mid-trip (agent must re-plan)")
    denv = MultiDeliveryEnv(test_orders, dynamic=True)
    rows = [
        evaluate_multi(greedy_policy_multi, denv, name="Greedy (nearest first)"),
        evaluate_multi(planner_policy_multi, denv, name="Planner (re-plans every step)"),
    ]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
