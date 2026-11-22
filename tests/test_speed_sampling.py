"""
Training speeds (utils.training_configuration, max_speed) and speed_sampling.SpeedSampler,
alone and in the DQN replay buffer.

Run from src/: python -m unittest discover -s ../tests -p test_speed_sampling.py
"""

import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402
from speed_sampling import SpeedSampler  # noqa: E402

CONFIG = utils.SINGLE_MODEL_REPLAY_BUFFER_OBS_GYM_CONFIGURATION


class TrainingConfigurationTests(unittest.TestCase):
    def test_games_run_at_the_speed_asked_for_up_to_18(self):
        for speed in (4, 8, 14, 15, 18):
            for random_start in (False, True):
                env = utils.make_gym(utils.training_configuration(speed, random_start), utils.ENV_VERSION, seed=speed)
                # Channel 15: Pacman's ticks per move over the ghosts' (12)
                self.assertAlmostEqual(float(env.obs_numpy(CONFIG)[15, 0, 0]), speed / 12, places=5)
                rng = random.Random(speed)
                for _ in range(40):
                    mask = env.action_mask()
                    _, done = env.step(rng.choice([a for a in range(utils.NUM_ACTIONS) if mask[a]]))
                    if done:
                        break

    def test_max_speed_sets_the_training_speeds(self):
        with mock.patch.dict(os.environ):
            self.assertEqual(utils.training_speeds(), tuple(range(4, 15)))  # the default
            utils.set_max_speed(18)
            self.assertEqual(utils.training_speeds(), tuple(range(4, 19)))
            with self.assertRaises(ValueError):
                utils.set_max_speed(3)
            with self.assertRaises(ValueError):
                utils.set_speed_sampling("by_feel")


class SpeedSamplerTests(unittest.TestCase):
    def play(self, sampler: SpeedSampler, games: int, length) -> dict[int, int]:
        moves = {s: 0 for s in sampler.speeds}
        for _ in range(games):
            speed = sampler.draw()
            n = length(speed)
            moves[speed] += n
            sampler.record(speed, n)
        return moves

    def test_uniform_draws_every_speed_alike(self):
        sampler = SpeedSampler("uniform", tuple(range(4, 19)), random.Random(0))
        self.assertTrue(all(abs(w - 1 / 15) < 1e-9 for w in sampler.weights().values()))
        moves = self.play(sampler, 30000, lambda s: 3000 // s)
        # Uniform games: the slow speeds get most of the moves
        self.assertGreater(moves[4] / moves[18], 3.5)

    def test_equal_moves_gives_every_speed_about_the_same_moves(self):
        sampler = SpeedSampler("equal_moves", tuple(range(4, 19)), random.Random(0))
        self.play(sampler, 5000, lambda s: 3000 // s)  # let the averages settle
        moves = self.play(sampler, 30000, lambda s: 3000 // s)
        share = {s: m / sum(moves.values()) for s, m in moves.items()}
        self.assertLess(max(share.values()) / min(share.values()), 1.25, share)
        weights = sampler.weights()
        self.assertGreater(weights[18] / weights[4], 3.5)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_short_games_cant_take_every_draw(self):
        sampler = SpeedSampler("equal_moves", (4, 18), random.Random(0), min_moves=10)
        for _ in range(500):
            sampler.record(18, 1)
            sampler.record(4, 1000)
        self.assertLess(sampler.weights()[18], 0.991)  # 1/10 against 1/1000 at most


class ReplayBufferSpeedTests(unittest.TestCase):
    def test_collects_at_sampled_speeds(self):
        with mock.patch.dict(os.environ):
            utils.set_single_model(True)
            utils.set_max_speed(18)
            utils.set_speed_sampling("equal_moves")

            def random_policy(obs, masks):
                return masks.float().multinomial(1).squeeze(1)

            buffer = ReplayBuffer(maxlen=2000, policy=random_policy, num_parallel_envs=16, random_start_proportion=0.25)
            self.assertEqual(buffer.speed_sampler.speeds, tuple(range(4, 19)))
            seen = set()
            for _ in range(300):
                seen.update(buffer._speeds)
                buffer.generate_experience_step()
                for env, speed in zip(buffer._envs, buffer._speeds):
                    self.assertAlmostEqual(float(env.obs_numpy(CONFIG)[15, 0, 0]), speed / 12, places=5)
            self.assertTrue(all(4 <= s <= 18 for s in seen))
            self.assertGreater(len(seen), 8)
            self.assertNotEqual(set(buffer.speed_sampler.mean_moves.values()), {300.0})  # games ended and were recorded


if __name__ == "__main__":
    unittest.main()
