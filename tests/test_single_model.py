"""
Tests for single-model mode (utils.set_single_model): the trained network plays every state,
with the normal action mask and super pellets in its observations, and spawned processes
inherit the mode. Run from the repository root with PYTHONPATH=src.
"""

import multiprocessing as mp
import os
import unittest

import numpy as np
import torch

import utils
from pacbot_rs import PacmanGym
from utils import DEFAULT_GYM_CONFIGURATION, NUM_ACTIONS, OBS_SHAPE


def _child_mode(queue) -> None:
    queue.put(utils.single_model())


class SingleModelTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("PACBOT_SINGLE_MODEL")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("PACBOT_SINGLE_MODEL", None)
        else:
            os.environ["PACBOT_SINGLE_MODEL"] = self._saved

    def test_off_by_default_and_toggles(self):
        os.environ.pop("PACBOT_SINGLE_MODEL", None)
        self.assertFalse(utils.single_model())
        utils.set_single_model(True)
        self.assertTrue(utils.single_model())
        utils.set_single_model(False)
        self.assertFalse(utils.single_model())

    def test_network_plays_every_state_with_the_normal_mask(self):
        env = PacmanGym(DEFAULT_GYM_CONFIGURATION)
        rng = np.random.default_rng(0)
        seen_outside_purgatory = False
        for _ in range(600):
            utils.set_single_model(False)
            two_model = (utils.in_purgatory(env), utils.replay_buffer_action_mask(env))
            seen_outside_purgatory |= not two_model[0]
            utils.set_single_model(True)
            self.assertTrue(utils.in_purgatory(env))
            self.assertEqual(list(utils.replay_buffer_action_mask(env)), list(env.action_mask()))
            self.assertEqual(two_model[1], env.purgatory_action_mask())
            actions = [a for a, ok in enumerate(env.action_mask()) if ok]
            _, done = env.step(int(rng.choice(actions)))
            if done:
                env.reset(DEFAULT_GYM_CONFIGURATION)
        self.assertTrue(seen_outside_purgatory, "the walk never left purgatory; test proves nothing")

    def test_observations_keep_super_pellets(self):
        env = PacmanGym(DEFAULT_GYM_CONFIGURATION)
        utils.set_single_model(True)
        single = env.obs_numpy(utils.replay_buffer_obs_config())
        utils.set_single_model(False)
        two_model = env.obs_numpy(utils.replay_buffer_obs_config())
        np.testing.assert_array_equal(single, env.obs_numpy(utils.SUPER_PELLET_MODE_CONFIGUARTION))
        self.assertFalse(np.array_equal(single, two_model))

    def test_super_pellet_model_never_steps(self):
        utils.set_single_model(True)
        env = PacmanGym(DEFAULT_GYM_CONFIGURATION)
        before = env.game_state_bytes() if hasattr(env, "game_state_bytes") else None
        utils.step_envs_while_not_policy([env], torch.device("cpu"))
        if before is not None:
            self.assertEqual(env.game_state_bytes(), before)

    def test_random_starts_for_a_share_of_the_games(self):
        from replay_buffer import ReplayBuffer
        from policies import EpsilonGreedy, MaxQPolicy
        import models
        policy = EpsilonGreedy(MaxQPolicy(models.QNetV2(OBS_SHAPE, NUM_ACTIONS)), NUM_ACTIONS, 1.0)
        utils.set_single_model(False)
        with self.assertRaises(ValueError):
            ReplayBuffer(maxlen=8, policy=policy, num_parallel_envs=4, random_start_proportion=0.5)
        utils.set_single_model(True)
        buffer = ReplayBuffer(maxlen=8, policy=policy, num_parallel_envs=4, random_start_proportion=0.5)
        self.assertEqual([c.random_start for c in buffer._reset_configs], [True, True, False, False])
        self.assertTrue(all(c.random_ticks for c in buffer._reset_configs))
        full = PacmanGym(DEFAULT_GYM_CONFIGURATION).remaining_pellets()
        env = PacmanGym(DEFAULT_GYM_CONFIGURATION)
        pellets = []
        for _ in range(40):
            env.reset(utils.random_start_configuration())
            pellets.append(env.remaining_pellets())
        self.assertLess(min(pellets), full * 0.7, "random starts should often remove half the board")

    def test_spawned_processes_inherit_the_mode(self):
        ctx = mp.get_context("spawn")
        for enabled in (True, False):
            utils.set_single_model(enabled)
            queue = ctx.Queue()
            proc = ctx.Process(target=_child_mode, args=(queue,))
            proc.start()
            self.assertEqual(queue.get(timeout=60), enabled)
            proc.join(timeout=30)


if __name__ == "__main__":
    unittest.main()
