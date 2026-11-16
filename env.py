# SHARED FILE, environment file for fair comparison

"""
0 = up, 1 = down, 2 = left, 3 = right 

Reward: //can change if want
1. Step cost = - (1 + traffic_penalty + weather_penalty) ( + hostpot extra)
2. wall bump = -2 
3. delivered = +50 
4. late arrival = -0.5 per minute over the dataset's Time_taken
5. timeout = -20
"""

import numpy as np
import pandas as pd
 
try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # still works without gymnasium (Q-learning doesn't need it)
    gym, spaces, _Base = None, None, object
 
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
WEATHER_MAP = {"Sunny": 0, "Cloudy": 0, "Windy": 1, "Fog": 2, "Sandstorms": 2, "Stormy": 3}
MINUTES_PER_STEP = 2.0  # one grid move is treated as ~2 minutes of base travel time
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
 
 
def load_orders(csv_path):
    """Load + clean the dataset into the columns the environment needs."""
    df = pd.read_csv(csv_path)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    df["traffic"] = df["Road_traffic_density"].map(TRAFFIC_MAP)
    df["weather"] = df["Weather_conditions"].str.split().str[-1].map(WEATHER_MAP)
    df["multi"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
    tt_col = next(c for c in df.columns if c.lower().replace(" ", "").startswith("time_taken"))
    df["time_taken"] = pd.to_numeric(df[tt_col].astype(str).str.extract(r"(\d+)")[0])
    coords = ["Restaurant_latitude", "Restaurant_longitude",
              "Delivery_location_latitude", "Delivery_location_longitude"]
    for c in coords:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["distance_km"] = haversine_km(df[coords[0]], df[coords[1]], df[coords[2]], df[coords[3]])
    df = df.dropna(subset=["traffic", "weather", "multi", "time_taken", "distance_km"])
    df = df[(df[coords].abs() > 1).all(axis=1)]
    df = df[df["distance_km"] < 50]
    return df[["traffic", "weather", "multi", "time_taken", "distance_km"]].reset_index(drop=True)
 
 
class DeliveryEnv(_Base):
    def __init__(self, orders: pd.DataFrame, grid_size=10, max_steps=60, map_seed=0, hotspot_frac=0.15):
        self.orders = orders
        self.n = grid_size
        self.max_steps = max_steps
        # Fixed congested cells (same for everyone because of map_seed)
        rng = np.random.default_rng(map_seed)
        self.hotspots = rng.random((self.n, self.n)) < hotspot_frac
        self.max_km = float(orders["distance_km"].quantile(0.95))
        self.rng = np.random.default_rng()
        self.n_actions = 4
        if spaces is not None:
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)
 
    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = self.orders.iloc[int(self.rng.integers(len(self.orders)))]
        self.traffic, self.weather = int(o["traffic"]), int(o["weather"])
        self.multi, self.benchmark_min = int(o["multi"]), float(o["time_taken"])
 
        # distance in km -> manhattan distance in grid cells (between 3 and the grid diagonal)
        target = int(np.clip(round(o["distance_km"] / self.max_km * (2 * self.n - 2)), 3, 2 * self.n - 2))
        # random start cell, then pick a destination cell at that manhattan distance (or the closest)
        self.pos = (int(self.rng.integers(self.n)), int(self.rng.integers(self.n)))
        cells = [(r, c) for r in range(self.n) for c in range(self.n)]
        dist = np.array([abs(r - self.pos[0]) + abs(c - self.pos[1]) for r, c in cells])
        best = np.where(np.abs(dist - target) == np.abs(dist - target).min())[0]
        self.dest = cells[int(self.rng.choice(best))]
 
        self.steps, self.elapsed_min = 0, 0.0
        return self._obs(), {}
 
    # ------------------------------------------------------------------- step
    def step(self, action):
        r, c = self.pos
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][int(action)]
        nr, nc = r + dr, c + dc
        reward = 0.0
        if not (0 <= nr < self.n and 0 <= nc < self.n):  # bumped into the map edge
            nr, nc = r, c
            reward -= 2.0
        self.pos = (nr, nc)
        self.steps += 1
 
        # cost of the move: traffic + weather make every minute more expensive
        traffic_pen, weather_pen = 0.3 * self.traffic, 0.2 * self.weather
        move_cost = 1.0 + traffic_pen + weather_pen
        minutes = MINUTES_PER_STEP * (1 + 0.25 * self.traffic + 0.15 * self.weather)
        if self.hotspots[nr, nc]:  # congested cell: extra cost scales with traffic level
            move_cost += 1.0 + self.traffic
            minutes *= 2.0
        reward -= move_cost
        self.elapsed_min += minutes
 
        terminated, truncated, delivered = False, False, False
        if self.pos == self.dest:
            terminated, delivered = True, True
            reward += 50.0
            late = max(0.0, self.elapsed_min - self.benchmark_min)
            reward -= 0.5 * late
        elif self.steps >= self.max_steps:
            truncated = True
            reward -= 20.0
 
        info = {"delivered": delivered, "elapsed_min": self.elapsed_min, "steps": self.steps}
        return self._obs(), reward, terminated, truncated, info
 
    # ---------------------------------------------------------- observations
    def _obs(self):
        n = self.n - 1
        return np.array([self.pos[0] / n, self.pos[1] / n, self.dest[0] / n, self.dest[1] / n,
                         self.traffic / 3, self.weather / 3, min(self.multi, 3) / 3,
                         1 - self.steps / self.max_steps], dtype=np.float32)
 
    def discrete_state(self):
        """Small hashable state for tabular Q-learning (no steps_left, no multi)."""
        return (self.pos[0], self.pos[1], self.dest[0], self.dest[1], self.traffic)
 
 
if __name__ == "__main__":
    orders = load_orders("./data/dataset_clean.csv")
    print("Usable orders:", len(orders))
    env = DeliveryEnv(orders)
    wins, total = 0, 0.0
    for ep in range(200):
        env.reset(seed=ep)
        done, ep_r = False, 0.0
        while not done:
            _, rew, term, trunc, info = env.step(env.rng.integers(4))
            ep_r += rew
            done = term or trunc
        wins += info["delivered"]
        total += ep_r
    print(f"Random agent: success={wins/200:.0%}, avg reward={total/200:.1f}  (your RL agents must beat this)")

    df = pd.read_csv("./data/dataset_clean.csv")
    print(len(df))
    print(df["Weather_conditions"].str.split().str[-1].value_counts(dropna=False))
    print(df["Road_traffic_density"].value_counts(dropna=False))