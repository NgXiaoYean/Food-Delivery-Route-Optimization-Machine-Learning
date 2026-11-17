"""DQN-owned data partitions, scenario banks and reproducibility helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np

MAP_BANKS = {"train": tuple(range(64)), "validation": tuple(range(1000, 1016)),
             "test": tuple(range(2000, 2032))}
EPISODE_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


def split_orders_three_way(orders, seed=42):
    """64/16/20 split; preserve the team's 20% test rows; retain original indices.

    Historical experiments already inspected this test partition. It is held out
    from NEW tuning, but cannot honestly be described as never previously used.
    """
    if len(orders) < 10 or not orders.index.is_unique:
        raise ValueError("Need at least ten orders with unique row indices")
    test = orders.sample(frac=0.2, random_state=seed)
    development = orders.drop(test.index)
    validation = development.sample(frac=0.2, random_state=seed + 1)
    train = development.drop(validation.index)
    return train.copy(), validation.copy(), test.copy()


def training_scale(train):
    value = float(train.distance_km.quantile(0.95))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Invalid training distance scale")
    return value


def dataset_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_manifest(partitions):
    return {name: [int(i) for i in frame.index]
            for name, frame in zip(("train", "validation", "test"), partitions)}
