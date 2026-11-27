"""Regression checks for the shared grid/player scenario and time benchmark."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from compare.learning_view import (catalog, fastest_time_policy, grid_svg, replay,
                                   shortest_distance_policy)
from dqn.learning_demo import animation_html
from env import DeliveryEnv


class LearningViewTests(unittest.TestCase):
    def setUp(self):
        orders = pd.DataFrame({"traffic": [3], "weather": [0], "multi": [1],
                               "time_taken": [45.0], "distance_km": [5.0]})
        env = DeliveryEnv(orders, max_steps=60, hotspot_frac=0)
        env.reset(seed=5)
        env.hotspots = np.zeros((10, 10), dtype=bool)
        env.hotspots[0, 1:4] = True
        env.traffic, env.weather, env.benchmark_min = 3, 0, 45.0
        self.orders = orders
        self.scenario = {"env": env, "start": (0, 0), "dest": (0, 5)}

    def test_shortest_distance_and_fastest_time_are_distinct(self):
        shortest = replay(shortest_distance_policy, self.orders, self.scenario)
        fastest = replay(fastest_time_policy, self.orders, self.scenario)
        self.assertEqual(shortest["steps"], 5)
        self.assertEqual(shortest["min_steps"], 5)
        self.assertGreater(fastest["steps"], shortest["steps"])
        self.assertLess(fastest["minutes"], shortest["minutes"])
        self.assertTrue(fastest["delivered"])

    def test_shared_replay_is_deterministic_and_does_not_change_source(self):
        first = replay(fastest_time_policy, self.orders, self.scenario)
        second = replay(fastest_time_policy, self.orders, self.scenario)
        self.assertEqual(first, second)
        self.assertEqual(self.scenario["env"].steps, 0)
        self.assertEqual(first["steps"], len(first["moves"]))
        self.assertEqual(first["steps"] + 1, len(first["points"]))
        self.assertIn('viewBox="0 0 240 240"', grid_svg(first))
        player = animation_html("DQN", {"label": "Final training checkpoint",
                                        "episode": 100, "training_seed": 11}, first)
        for control in ('id="play"', 'id="step"', 'id="reset"', 'id="action"', 'id="reward"'):
            self.assertIn(control, player)

    def test_exported_stages_have_real_files_and_training_counts(self):
        record = catalog()
        self.assertIsNotNone(record)
        for model in ("Q-Learning", "DQN", "Double DQN", "Dueling DQN", "D3QN"):
            details = record["models"][model]
            self.assertGreater(details["total_completed_episodes"], 0)
            for stage in details["stages"]:
                self.assertTrue((Path(__file__).resolve().parents[1] /
                                 "compare" / "checkpoints" / stage["file"]).is_file())


if __name__ == "__main__":
    unittest.main()
