"""Check real exported policies against the shared scenario/player contract."""

import unittest

from compare.learning_view import catalog, load_policy, replay
from env import DeliveryEnv, load_orders


class ExportedModelTests(unittest.TestCase):
    def test_each_saved_best_checkpoint_replays_deterministically(self):
        orders = load_orders("data/dataset_clean.csv")
        env = DeliveryEnv(orders)
        env.reset(seed=17)
        scenario = {"env": env, "start": env.pos, "dest": env.dest}
        models = catalog()["models"]
        self.assertEqual(set(models), {"Q-Learning", "DQN", "Double DQN",
                                       "Dueling DQN", "D3QN"})
        for name, details in models.items():
            with self.subTest(model=name):
                best, latest = details["stages"][:2]
                policy = load_policy(name, best)
                route = replay(policy, orders, scenario)
                self.assertEqual(route, replay(policy, orders, scenario))
                self.assertEqual(route["steps"], len(route["moves"]))
                if best["file"] == latest["file"]:
                    final_route = replay(load_policy(name, latest), orders, scenario)
                    self.assertEqual(route["points"], final_route["points"])

    def test_untrained_and_early_saved_stages_are_replayable(self):
        orders = load_orders("data/dataset_clean.csv")
        env = DeliveryEnv(orders)
        env.reset(seed=23)
        scenario = {"env": env, "start": env.pos, "dest": env.dest}
        for name, details in catalog()["models"].items():
            stages = {stage["episode"]: stage for stage in details["stages"]
                      if stage["kind"] == "milestone"}
            self.assertEqual(set(stages), {0, 50, 100, 150, 200, 300})
            for episode in (0, 50, 300):
                with self.subTest(model=name, episode=episode):
                    route = replay(load_policy(name, stages[episode]), orders, scenario)
                    self.assertLessEqual(route["steps"], 60)
                    self.assertEqual(route["steps"], len(route["moves"]))


if __name__ == "__main__":
    unittest.main()
