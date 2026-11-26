"""
Latency (PacmanGym.latency_max, utils.set_latency_max): before each move the game runs a random
0 to latency_max extra ticks with Pacman still on his square while the ghosts move, as when the
robot carries a move out late. Off (0, the default), games are exactly as without it.

Run from src/: python -m unittest discover ../tests
"""

import os
import random
import unittest

import numpy as np

import utils
from pacbot_rs import GumbelConfig, PacmanGym, PacmanGymConfiguration, gumbel_search
from utils import ENV_VERSION

OBS = PacmanGymConfiguration({})
SPEED = 8


def ticks(env: PacmanGym) -> int:
    """The game's tick counter (the first two bytes of the server's game state message)."""
    b = env.game_state_bytes()
    return b[0] << 8 | b[1]


def game(seed: int, latency: int = 0) -> PacmanGym:
    env = PacmanGym(utils.speed_configuration(SPEED), seed=seed)
    env.env_version = ENV_VERSION
    env.latency_max = latency
    return env


def play(env: PacmanGym, moves: int, move_seed: int) -> list:
    """Random legal moves; each step's (reward, done, ticks so far, state bytes)."""
    rng = random.Random(move_seed)
    out = []
    for _ in range(moves):
        legal = [a for a, ok in enumerate(env.action_mask()) if ok]
        reward, done = env.step(rng.choice(legal))
        out.append((reward, done, ticks(env), bytes(env.game_state_bytes())))
        if done:
            break
    return out


class LatencyTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("PACBOT_LATENCY_MAX")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("PACBOT_LATENCY_MAX", None)
        else:
            os.environ["PACBOT_LATENCY_MAX"] = self._saved

    def test_off_by_default(self):
        os.environ.pop("PACBOT_LATENCY_MAX", None)
        self.assertEqual(utils.latency_max(), 0)
        self.assertEqual(PacmanGym(OBS).latency_max, 0)
        self.assertEqual(utils.make_gym(OBS, ENV_VERSION, seed=1).latency_max, 0)
        utils.set_latency_max(3)
        self.assertEqual(utils.make_gym(OBS, ENV_VERSION, seed=1).latency_max, 3)
        with self.assertRaises(ValueError):
            utils.set_latency_max(-1)

    def test_off_the_latency_seed_changes_nothing(self):
        # With no latency nothing is drawn, so its seed can't affect the game
        a, b = game(5), game(5)
        b.latency_seed = 12345
        self.assertEqual(play(a, 300, 1), play(b, 300, 1))
        for steps in (play(game(5), 300, 1),):
            gaps = np.diff([0] + [s[2] for s in steps if not s[1]])
            self.assertTrue((gaps == SPEED).all())  # every move takes exactly the speed's ticks

    def test_each_move_comes_0_to_latency_max_ticks_late(self):
        for latency in (1, 3):
            with self.subTest(latency=latency):
                steps = play(game(7, latency), 400, 2)
                live = [s for s in steps if not s[1]]
                gaps = np.diff([0] + [s[2] for s in live])
                self.assertTrue(((gaps >= SPEED) & (gaps <= SPEED + latency)).all(), gaps)
                # Every delay from 0 to latency_max turns up
                self.assertEqual(set(gaps.tolist()), set(range(SPEED, SPEED + latency + 1)))

    def test_seeded_games_repeat_with_their_delays(self):
        self.assertEqual(play(game(9, 2), 300, 3), play(game(9, 2), 300, 3))
        # Setting the seed after a reset sets the delays' too
        a, b = game(1, 2), game(2, 2)
        a.seed = b.seed = 77
        a.set_game_state_bytes(b.game_state_bytes(), SPEED)
        self.assertEqual(a.latency_seed, b.latency_seed)
        # Another latency seed: other delays, the same start
        c = game(9, 2)
        c.latency_seed ^= 1
        self.assertNotEqual(play(game(9, 2), 300, 3), play(c, 300, 3))

    def test_search_copies_draw_their_own_delays(self):
        env = game(4, 2)
        self.assertNotEqual(env.copy().latency_seed, env.latency_seed)
        self.assertEqual(env.copy(reseed=False).latency_seed, env.latency_seed)
        # The search runs with latency and leaves the real game's coming delays alone
        before = env.latency_seed

        def evaluator(obs, legal):
            return np.zeros(len(obs), dtype=np.float32), np.zeros((len(obs), 5), dtype=np.float32)

        action = int(gumbel_search([env], evaluator, GumbelConfig(num_simulations=16, gumbel_scale=0.0), OBS, 1)[0][0])
        self.assertTrue(env.action_mask()[action])
        self.assertEqual(env.latency_seed, before)

    def test_latency_costs_lives(self):
        # Moving late near the ghosts: over the same seeds and moves, games with latency lose
        # lives sooner on average (the same moves, but the ghosts get extra ticks)
        def steps_alive(latency):
            return np.mean([len(play(game(s, latency), 3000, s)) for s in range(30)])
        self.assertLess(steps_alive(6), steps_alive(0))


if __name__ == "__main__":
    unittest.main()
