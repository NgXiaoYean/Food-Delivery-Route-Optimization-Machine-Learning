"""Core unit tests for data splits, environment step logic, Double-Q target, and Planner optimality.

Run from the project root:
    python -B -m unittest tests.test_core -v
"""
from pathlib import Path
import unittest

import numpy as np
import torch

from dqn.experiments import split_orders_three_way
from env import DeliveryEnv, load_orders
from env_multi import MultiDeliveryEnv, split_orders_3way
from evaluate import greedy_policy, planner_policy
from shared import split_orders

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset_clean.csv"


class CoreProtocolAndEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orders = load_orders(DATA_PATH)

    def test_splits_are_disjoint_and_sum_to_40482(self):
        """All three data splits must be disjoint and sum to the 40,482 cleaned orders."""
        total = len(self.orders)
        self.assertEqual(total, 40482)

        # 1. 80/20 split (shared.split_orders)
        test_80_20 = self.orders.sample(frac=0.2, random_state=42)
        train_80_20 = self.orders.drop(test_80_20.index)
        self.assertEqual((len(train_80_20), len(test_80_20)), (32386, 8096))
        self.assertFalse(set(train_80_20.index) & set(test_80_20.index))
        self.assertEqual(len(train_80_20) + len(test_80_20), total)

        tr, te = split_orders(self.orders, test_frac=0.2, seed=42)
        self.assertEqual((len(tr), len(te)), (32386, 8096))

        # 2. 64/16/20 split (dqn.experiments.split_orders_three_way)
        tr3, val3, te3 = split_orders_three_way(self.orders, seed=42)
        self.assertEqual((len(tr3), len(val3), len(te3)), (25909, 6477, 8096))
        self.assertFalse(set(tr3.index) & set(val3.index))
        self.assertFalse(set(tr3.index) & set(te3.index))
        self.assertFalse(set(val3.index) & set(te3.index))
        self.assertEqual(len(tr3) + len(val3) + len(te3), total)

        # 3. 70/15/15 split (env_multi.split_orders_3way)
        te_m = self.orders.sample(frac=0.15, random_state=42)
        rest_m = self.orders.drop(te_m.index)
        val_m = rest_m.sample(frac=0.15 / 0.85, random_state=42)
        tr_m = rest_m.drop(val_m.index)
        self.assertFalse(set(tr_m.index) & set(val_m.index))
        self.assertFalse(set(tr_m.index) & set(te_m.index))
        self.assertFalse(set(val_m.index) & set(te_m.index))

        tr_m2, val_m2, te_m2 = split_orders_3way(self.orders, val_frac=0.15, test_frac=0.15, seed=42)
        self.assertEqual((len(tr_m2), len(val_m2), len(te_m2)), (28338, 6072, 6072))
        self.assertEqual(len(tr_m2) + len(val_m2) + len(te_m2), total)

    def test_same_trip_seed_reproduces_trip(self):
        """Same trip seed must give the exact same trip; different seeds must differ."""
        env = DeliveryEnv(self.orders.iloc[:200].reset_index(drop=True))
        obs1, _ = env.reset(seed=7)
        snap1 = (env.pos, env.dest, env.traffic, env.weather, env.benchmark_min)

        obs2, _ = env.reset(seed=7)
        snap2 = (env.pos, env.dest, env.traffic, env.weather, env.benchmark_min)
        np.testing.assert_array_equal(obs1, obs2)
        self.assertEqual(snap1, snap2)

        _, _ = env.reset(seed=8)
        snap3 = (env.pos, env.dest, env.traffic, env.weather, env.benchmark_min)
        self.assertNotEqual(snap1, snap3)

        menv = MultiDeliveryEnv(self.orders.iloc[:200].reset_index(drop=True))
        mobs1, _ = menv.reset(seed=15)
        msnap1 = (menv.pos, list(menv.dests), list(menv.deadlines), menv.traffic, menv.weather)
        mobs2, _ = menv.reset(seed=15)
        msnap2 = (menv.pos, list(menv.dests), list(menv.deadlines), menv.traffic, menv.weather)
        np.testing.assert_array_equal(mobs1, mobs2)
        self.assertEqual(msnap1, msnap2)

    def test_reward_and_termination_math(self):
        """Verify wall bump, move cost, hotspot penalty, delivery bonus, and timeout truncation."""
        env = DeliveryEnv(self.orders.iloc[:50].reset_index(drop=True), max_steps=2)
        env.reset(seed=0)
        env.pos = (0, 0)
        env.dest = (0, 1)
        env.traffic = 2
        env.weather = 1
        env.benchmark_min = 10.0
        env.hotspots[:] = False
        env.hotspots[0, 0] = False
        env.hotspots[0, 1] = True

        # Step 1: Bump into top wall (action 0 = Up from row 0)
        _, r_wall, term, trunc, info = env.step(0)
        expected_move = 1.0 + 0.3 * 2 + 0.2 * 1  # 1.8
        self.assertEqual(env.pos, (0, 0))
        self.assertAlmostEqual(r_wall, -2.0 - expected_move, places=5)
        self.assertFalse(term)
        self.assertFalse(trunc)

        # Step 2: Move right into destination (0, 1), which is a hotspot
        _, r_deliv, term, trunc, info = env.step(3)
        expected_hot_move = expected_move + (1.0 + 2)  # 1.8 + 3.0 = 4.8
        step1_min = 2.0 * (1 + 0.25 * 2 + 0.15 * 1)    # 3.3 min
        step2_min = step1_min * 2.0                     # 6.6 min (hotspot doubles step minutes)
        total_min = step1_min + step2_min               # 9.9 min <= 10.0 benchmark (0 late)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertTrue(info["delivered"])
        self.assertAlmostEqual(info["elapsed_min"], total_min, places=5)
        self.assertAlmostEqual(r_deliv, 50.0 - expected_hot_move, places=5)

        # Timeout check
        env.reset(seed=0)
        env.pos, env.dest = (0, 0), (9, 9)
        env.hotspots[:] = False
        env.step(1)
        _, r_to, term, trunc, info = env.step(1)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertFalse(info["delivered"])
        self.assertLess(r_to, -20.0)

    def test_double_q_target_uses_online_argmax_and_target_gather(self):
        """Double-Q target must select argmax action via online net and evaluate via target net."""
        online_q = torch.tensor([
            [1.0, 9.0, 2.0, 0.0],  # online argmax = action 1
            [8.0, 3.0, 4.0, 1.0],  # online argmax = action 0
        ])
        target_q = torch.tensor([
            [100.0, 15.0, 50.0, 20.0],  # target max is 100 (action 0), but action 1 score is 15.0
            [12.0, 90.0, 30.0, 40.0],   # target max is 90 (action 1), but action 0 score is 12.0
        ])
        rewards = torch.tensor([[2.0], [5.0]])
        dones = torch.tensor([[0.0], [1.0]])  # second state is terminal
        gamma = 0.9

        best_actions = online_q.argmax(dim=1, keepdim=True)
        self.assertEqual(best_actions.squeeze(1).tolist(), [1, 0])
        best_next = target_q.gather(1, best_actions)
        self.assertEqual(best_next.squeeze(1).tolist(), [15.0, 12.0])

        y_double = rewards + gamma * best_next * (1.0 - dones)
        # Row 0: 2.0 + 0.9 * 15.0 = 15.5 (whereas standard DQN max would give 2.0 + 0.9 * 100.0 = 92.0)
        # Row 1: terminal (done=1.0) -> 5.0 (no bootstrap)
        torch.testing.assert_close(y_double, torch.tensor([[15.5], [5.0]]))

    def test_planner_never_loses_to_greedy_on_same_trip(self):
        """On single-order static grid, Dijkstra Planner must achieve >= Greedy reward on every trip."""
        env = DeliveryEnv(self.orders.iloc[:500].reset_index(drop=True))
        for seed in range(50):
            rewards = {}
            for name, pol in (("greedy", greedy_policy), ("planner", planner_policy)):
                env.reset(seed=seed)
                done, total = False, 0.0
                while not done:
                    _, r, term, trunc, _ = env.step(pol(env))
                    total += r
                    done = term or trunc
                rewards[name] = total
            self.assertGreaterEqual(
                rewards["planner"] + 1e-6,
                rewards["greedy"],
                msg=f"Planner lost to Greedy on trip seed {seed}: {rewards}",
            )


if __name__ == "__main__":
    unittest.main()
