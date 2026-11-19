"""
Ghost-next observations (PacmanGym.ghost_next_obs, utils.set_ghost_next_obs): channels 8-11
hold each ghost's next square instead of a copy of its current square.

Run from src/: python -m unittest discover ../tests
"""

import os
import random
import unittest

import numpy as np

import utils
from utils import DEFAULT_GYM_CONFIGURATION, ENV_VERSION

GHOSTS = range(4, 8)
NEXT = range(8, 12)


def _square(channel: np.ndarray):
    found = np.argwhere(channel > 0)
    return tuple(found[0]) if len(found) else None


class GhostNextObsTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("PACBOT_GHOST_NEXT_OBS")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("PACBOT_GHOST_NEXT_OBS", None)
        else:
            os.environ["PACBOT_GHOST_NEXT_OBS"] = self._saved

    def test_off_by_default_and_copies_the_current_squares(self):
        os.environ.pop("PACBOT_GHOST_NEXT_OBS", None)
        self.assertFalse(utils.ghost_next_obs())
        gym = utils.make_gym(DEFAULT_GYM_CONFIGURATION, ENV_VERSION)
        self.assertFalse(gym.ghost_next_obs)
        for _ in range(30):
            obs = gym.obs_numpy(DEFAULT_GYM_CONFIGURATION)
            np.testing.assert_array_equal(obs[8:12], obs[4:8])
            if gym.step(random.choice([i for i, ok in enumerate(gym.action_mask()) if ok]))[1]:
                break

    def test_make_gym_follows_the_mode_and_resets_keep_it(self):
        utils.set_ghost_next_obs(True)
        gym = utils.make_gym(DEFAULT_GYM_CONFIGURATION, ENV_VERSION)
        self.assertTrue(gym.ghost_next_obs)
        gym.reset(DEFAULT_GYM_CONFIGURATION)
        self.assertTrue(gym.ghost_next_obs)

    def test_predicts_where_each_ghost_moves(self):
        """When a ghost's square changes, it moves to the square the observation showed."""
        utils.set_ghost_next_obs(True)
        rng = random.Random(0)
        checked = matched = 0
        for _ in range(5):
            gym = utils.make_gym(DEFAULT_GYM_CONFIGURATION, ENV_VERSION)
            obs = gym.obs_numpy(DEFAULT_GYM_CONFIGURATION)
            for _ in range(300):
                lives = gym.lives()
                _, done = gym.step(rng.choice([i for i, ok in enumerate(gym.action_mask()) if ok]))
                if done:
                    break
                new = gym.obs_numpy(DEFAULT_GYM_CONFIGURATION)
                if gym.lives() == lives:  # a lost life resets the ghosts
                    for g, n in zip(GHOSTS, NEXT):
                        before, after = _square(obs[g]), _square(new[g])
                        if before is not None and after is not None and before != after:
                            checked += 1
                            matched += after == _square(obs[n])
                obs = new
        self.assertGreater(checked, 200)
        # Eating a ghost sends it home and ghosts can update more than once between
        # Pacman's moves at fast speeds, so a few changes aren't one square ahead
        self.assertGreater(matched / checked, 0.95)

    def test_one_square_per_ghost_on_the_board(self):
        utils.set_ghost_next_obs(True)
        gym = utils.make_gym(DEFAULT_GYM_CONFIGURATION, ENV_VERSION)
        for _ in range(50):
            obs = gym.obs_numpy(DEFAULT_GYM_CONFIGURATION)
            for g, n in zip(GHOSTS, NEXT):
                self.assertEqual(obs[n].sum(), obs[g].sum())
                cur, nxt = _square(obs[g]), _square(obs[n])
                if cur is not None:
                    self.assertLessEqual(abs(cur[0] - nxt[0]) + abs(cur[1] - nxt[1]), 1)
            if gym.step(random.choice([i for i, ok in enumerate(gym.action_mask()) if ok]))[1]:
                break

    def test_checkpoint_flag(self):
        self.assertTrue(utils.checkpoint_ghost_next_obs({"ghost_next_obs": True}))
        self.assertFalse(utils.checkpoint_ghost_next_obs({"single_model": True}))
        self.assertFalse(utils.checkpoint_ghost_next_obs(object()))


if __name__ == "__main__":
    unittest.main()
