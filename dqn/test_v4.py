"""Behavioral checks for the new simulator, learning targets and experiment protocol."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from shared import split_orders
from dqn.dqn_v4 import Config, bellman_targets, load_checkpoint, policy, save_checkpoint, train
from dqn.dynamic_env import Conditions, DynamicDeliveryEnv, features
from dqn.evaluate_dqn import evaluate, greedy_policy, paired_interval, planner_policy, summarize
from dqn.experiments import MAP_BANKS, split_orders_three_way, training_scale


def orders():
    return pd.DataFrame(dict(traffic=[1] * 100, weather=[0] * 100, multi=[0] * 100,
                             time_taken=[30.0] * 100, distance_km=np.linspace(1, 10, 100)))


def environment(**kwargs):
    return DynamicDeliveryEnv(orders(), max_km=10.0, **kwargs)


class EnvironmentTests(unittest.TestCase):
    def test_reset_reproduces_entire_timeline(self):
        env = environment()
        first, _ = env.reset(seed=12)
        snapshots = [(x.minute, x.traffic, x.weather, x.hotspots.copy()) for x in env.timeline]
        env.step(0)
        second, _ = env.reset(seed=12)
        np.testing.assert_equal(first, second)
        for a, b in zip(snapshots, env.timeline):
            self.assertEqual(a[:3], (b.minute, b.traffic, b.weather))
            np.testing.assert_equal(a[3], b.hotspots)

    def test_event_schedule_independent_of_actions(self):
        a, b = environment(), environment()
        a.reset(seed=19)
        b.reset(seed=19)
        a.step(0)
        b.step(3)
        for minute in (0, 8, 24, 50, 200):
            x, y = a.conditions_at(minute), b.conditions_at(minute)
            self.assertEqual((x.traffic, x.weather), (y.traffic, y.weather))
            np.testing.assert_equal(x.hotspots, y.hotspots)

    def test_static_conditions_stay_fixed(self):
        env = environment(scenario="static")
        env.reset(seed=5)
        self.assertEqual(len(env.timeline), 1)
        before = (env.traffic, env.weather, env.hotspots.copy())
        for _ in range(10):
            _, _, term, _, _ = env.step(0)
            if term:
                break
        self.assertEqual(before[:2], (env.traffic, env.weather))
        np.testing.assert_equal(before[2], env.hotspots)

    def test_dynamic_has_weather_and_map_changes(self):
        env = environment()
        env.reset(seed=12)
        self.assertTrue(any(x.weather != env.timeline[0].weather for x in env.timeline[1:]))
        self.assertTrue(any(not np.array_equal(x.hotspots, env.timeline[0].hotspots) for x in env.timeline[1:]))

    def test_move_cost_uses_departure_snapshot(self):
        env = environment()
        env.reset(seed=3)
        env.pos, env.dest = (0, 0), (9, 9)
        clear = np.zeros((10, 10), dtype=bool)
        congested = np.ones((10, 10), dtype=bool)
        env.timeline = [Conditions(0, 0, 0, clear), Conditions(1, 3, 3, congested)]
        env._apply_conditions()
        _, _, _, _, info = env.step(3)
        self.assertEqual(info["minutes"], 2.0)
        self.assertFalse(info["hotspot_entered"])
        self.assertEqual(info["events_crossed"], 1)
        self.assertEqual(env.traffic, 3)
        self.assertAlmostEqual(env.neighbor_minutes()[3], 8.8, places=5)

    def test_deadline_changes_observation(self):
        env = environment()
        env.reset(seed=2)
        baseline, aware = features(env, "baseline"), features(env)
        env.benchmark_min -= 10
        np.testing.assert_equal(baseline, features(env, "baseline"))
        self.assertFalse(np.array_equal(aware, features(env)))
        self.assertEqual(features(env).shape, (30,))
        self.assertEqual(features(env, "baseline").shape, (15,))
        self.assertTrue(np.isfinite(features(env)).all())

    def test_no_future_information_in_features(self):
        env = environment()
        env.reset(seed=2)
        before = features(env)
        env.timeline[-1] = Conditions(1000, 3, 3, np.ones((10, 10), dtype=bool))
        np.testing.assert_equal(before, features(env))

    def test_horizon_is_terminal_failure(self):
        env = environment(max_steps=1)
        env.reset(seed=2)
        env.pos, env.dest = (0, 0), (9, 9)
        _, _, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["failed"])
        self.assertFalse(info["delivered"])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_delivery_on_final_step_is_success(self):
        env = environment(max_steps=1)
        env.reset(seed=2)
        env.pos, env.dest = (0, 0), (0, 1)
        _, _, term, trunc, info = env.step(3)
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertTrue(info["delivered"])
        self.assertFalse(info["failed"])

    def test_reward_ablation_keeps_common_metrics_identical(self):
        a = environment(reward_mode="time")
        b = environment(reward_mode="legacy")
        a.reset(seed=11)
        b.reset(seed=11)
        ra = a.step(0)
        rb = b.step(0)
        self.assertNotEqual(ra[1], rb[1])
        self.assertEqual(ra[4], rb[4])

    def test_planner_chooses_longer_but_faster_route(self):
        env = environment(scenario="static")
        env.reset(seed=1)
        env.pos, env.dest = (0, 0), (0, 4)
        env.traffic, env.weather = 0, 0
        env.hotspots[:] = False
        env.hotspots[0, 1:4] = True
        self.assertEqual(greedy_policy(env), 3)
        self.assertEqual(planner_policy(env), 1)
        env.hotspots[:] = False
        self.assertEqual(planner_policy(env), 3)

    def test_invalid_action_rejected(self):
        env = environment()
        env.reset(seed=1)
        for action in (-1, 4, 1.5):
            with self.assertRaises(ValueError):
                env.step(action)


class ProtocolTests(unittest.TestCase):
    def test_split_disjoint_reproducible_and_preserves_team_test(self):
        data = orders()
        a, b, c = split_orders_three_way(data)
        self.assertEqual((len(a), len(b), len(c)), (64, 16, 20))
        self.assertFalse(set(a.index) & set(b.index) or set(a.index) & set(c.index) or set(b.index) & set(c.index))
        self.assertEqual(set(a.index) | set(b.index) | set(c.index), set(data.index))
        _, team_test = split_orders(data)
        pd.testing.assert_frame_equal(c.reset_index(drop=True), team_test)
        for first, second in zip((a, b, c), split_orders_three_way(data)):
            pd.testing.assert_frame_equal(first, second)

    def test_map_banks_are_disjoint(self):
        self.assertFalse(set(MAP_BANKS["train"]) & set(MAP_BANKS["validation"]))
        self.assertFalse(set(MAP_BANKS["train"]) & set(MAP_BANKS["test"]))
        self.assertFalse(set(MAP_BANKS["validation"]) & set(MAP_BANKS["test"]))

    def test_test_outlier_does_not_change_training_scale(self):
        train_orders, _, test_orders = split_orders_three_way(orders())
        value = training_scale(train_orders)
        test_orders.loc[:, "distance_km"] = 999
        env = DynamicDeliveryEnv(test_orders, max_km=value)
        self.assertEqual(env.max_km, value)

    def test_same_policy_paired_interval_is_zero(self):
        env = environment()
        baseline = evaluate(greedy_policy, env, episodes=8, agent="Greedy")
        candidate = evaluate(greedy_policy, env, episodes=8, agent="copy", training_seed=11)
        interval = paired_interval(pd.concat([baseline, candidate]), "copy", samples=50)
        self.assertEqual((interval["delta"], interval["ci95_low"], interval["ci95_high"]), (0, 0, 0))
        candidate.loc[0, "map_seed"] = -1
        with self.assertRaises(ValueError):
            paired_interval(pd.concat([baseline, candidate]), "copy", samples=10)

    def test_failure_is_not_counted_as_fast_delivery(self):
        frame = evaluate(lambda env: 0, environment(max_steps=1), episodes=8)
        stats = summarize(frame)
        self.assertEqual(stats["success_rate"], 0)
        self.assertTrue(np.isnan(stats["mean_delivery_min"]))
        self.assertGreater(stats["mean_elapsed_all_min"], 0)

    def test_learned_control_pairs_by_training_seed(self):
        frame = evaluate(greedy_policy, environment(), episodes=4, agent="control", training_seed=11)
        second = frame.assign(training_seed=22)
        reference = pd.concat([frame, second], ignore_index=True)
        candidate = reference.assign(agent="candidate", common_return=reference.common_return + 2.0)
        interval = paired_interval(pd.concat([reference, candidate]), "candidate", baseline="control", samples=30)
        self.assertAlmostEqual(interval["delta"], 2.0)
        self.assertAlmostEqual(interval["ci95_low"], 2.0)
        self.assertEqual(interval["training_seeds"], 2)


class LearningTests(unittest.TestCase):
    def test_terminal_target_has_no_bootstrap(self):
        result = bellman_targets(torch.tensor([-20., 2.]), torch.tensor([1., 0.]),
                                 torch.tensor([[100., 200.], [10., 20.]]), 0.9)
        torch.testing.assert_close(result, torch.tensor([-20., 20.]))

    def test_training_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        cfg = Config(total_steps=96, hidden=16, warmup=16, batch_size=8, target_update=16)
        env = environment(max_steps=5)
        net, history, stats = train(env, cfg, seed=3)
        self.assertGreater(stats["updates"], 0)
        self.assertGreater(len(history), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, net, cfg, seed=3, feature_mode="deadline", metadata={"max_km": 10.0})
            restored, item = load_checkpoint(path)
            self.assertEqual(item["algorithm"], "standard_dqn")
            env.reset(seed=10)
            self.assertEqual(policy(net)(env), policy(restored)(env))
            for a, b in zip(net.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
