"""
Reward parts (PacmanGym.reward_parts, for split-reward training as in the Hybrid Reward
Architecture): each step's reward split into pellets, super pellets, ghosts and fruit, death,
clear and shaping, adding up to the step's reward, in one-life and full games.

Run from src/: python -m unittest discover ../tests
"""

import random
import unittest

from pacbot_rs import PacmanGym
from test_full_game import config, greedy_action


class RewardPartTests(unittest.TestCase):
    def test_parts_add_up_and_each_holds_its_own_points(self):
        self.assertEqual(list(PacmanGym.REWARD_PART_NAMES),
                         ["pellets", "super_pellets", "ghosts_fruit", "death", "clear", "shaping"])
        totals = [0] * 6
        clears = deaths = 0
        for seed in range(40):
            for full in (False, True):
                env = PacmanGym(config(clearable=seed % 2 == 0), seed=seed)
                env.full_game = full
                rng = random.Random(seed)
                for _ in range(3_000):
                    reward, done = env.step(greedy_action(env, rng, explore=0.05))
                    parts = env.reward_parts()
                    with self.subTest(seed=seed, full=full):
                        self.assertEqual(sum(parts), reward)
                        pellets, supers, ghosts_fruit, death, clear, _ = parts
                        self.assertIn(death, (0, -200))
                        self.assertIn(clear, (0, 3_000))
                        self.assertIn(supers, (0, 50))
                        if clear == 0:
                            self.assertIn(pellets + supers, (0, 10, 50))
                        self.assertGreaterEqual(ghosts_fruit, 0)
                        self.assertEqual(ghosts_fruit % 100, 0)  # ghosts 200-1600, fruit 100
                    totals = [t + p for t, p in zip(totals, parts)]
                    clears += clear > 0
                    deaths += death < 0
                    if done:
                        break
        # Every part shows up somewhere
        self.assertGreater(totals[0], 0)
        self.assertGreater(totals[1], 0)
        self.assertGreater(totals[2], 0)
        self.assertGreater(clears, 0)
        self.assertGreater(deaths, 0)
        self.assertLess(totals[5], 0)

    def test_a_new_game_starts_with_no_parts(self):
        env = PacmanGym(config(), seed=3)
        self.assertEqual(list(env.reward_parts()), [0] * 6)


if __name__ == "__main__":
    unittest.main()
