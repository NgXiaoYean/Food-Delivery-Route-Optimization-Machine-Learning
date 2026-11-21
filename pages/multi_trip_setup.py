"""Set up a multi-order demo trip without changing the trained environment rules."""

from env_multi import K_MAX


def prepare_trip(env, seed, setup):
    """Reset a trip, then apply the same selected inputs for every route planner.

    The example mode keeps the original seeded test trip. Dataset mode combines
    separate held-out rows into a simulated trip; custom mode uses grid inputs.
    """
    env.reset(seed=int(seed))
    mode = setup["mode"]
    if mode == "example":
        return

    if mode == "dataset":
        indices = tuple(int(i) for i in setup["order_indices"])
        if len(set(indices)) != len(indices) or not all(0 <= i < len(env.orders) for i in indices):
            raise ValueError("Choose different valid dataset orders for each customer.")
        rows = [env.orders.iloc[i] for i in indices]
        env.traffic, env.weather = int(rows[0]["traffic"]), int(rows[0]["weather"])
        env.dests = []
        for row in rows:
            env.dests.append(env._place_dest(float(row["distance_km"]), set(env.dests)))
        env.deadlines = [float(row["time_taken"]) for row in rows]
        if env.dynamic:
            late_index = int(setup["late_index"])
            if late_index in indices or not 0 <= late_index < len(env.orders):
                raise ValueError("The arriving order must be a different dataset row.")
            env._late_row = env.orders.iloc[late_index]
            env.arrival_step = int(env.rng.integers(3, 9))
    elif mode == "custom":
        start = tuple(int(x) for x in setup["start"])
        dests = [tuple(int(x) for x in pos) for pos in setup["destinations"]]
        if not all(0 <= x < env.n for pos in [start, *dests] for x in pos):
            raise ValueError("Every grid row and column must be inside the map.")
        if len(set([start, *dests])) != len(dests) + 1:
            raise ValueError("The rider and each customer need different grid cells.")
        env.pos = start
        env.dests = dests
        env.deadlines = [float(t) for t in setup["deadlines"]]
        env.traffic, env.weather = int(setup["traffic"]), int(setup["weather"])
        if env.dynamic:
            env._late_row = {"distance_km": float(setup["late_distance_km"]),
                             "time_taken": float(setup["late_target_min"])}
            env.arrival_step = int(setup["arrival_step"])
    else:
        raise ValueError(f"Unknown trip setup: {mode}")

    if not 1 <= len(env.dests) <= K_MAX - int(env.dynamic):
        raise ValueError("Choose fewer starting customers when a new order will arrive.")
    if len(env.deadlines) != len(env.dests) or any(t <= 0 for t in env.deadlines):
        raise ValueError("Give every customer a positive delivery target.")
    env.k = len(env.dests)
    env.multi = env.k - 1
    env.delivered_flags = [False] * env.k
    env.benchmark_min = max(env.deadlines)
    env.steps, env.elapsed_min, env.late_total = 0, 0.0, 0.0
    if not env.dynamic:
        env.arrival_step, env._late_row = None, None
