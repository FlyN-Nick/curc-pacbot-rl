"""
The Pacbot game as a LightZero environment (lightzero_pacbot.pacbot_env). Needs LightZero,
which lives in its own environment (docs/muzero.md); skipped without it.

Run from src/: python -m unittest discover ../tests
"""

import importlib.util
import unittest

import numpy as np

HAVE_LIGHTZERO = importlib.util.find_spec("lzero") is not None


@unittest.skipUnless(HAVE_LIGHTZERO, "needs LightZero (see docs/muzero.md)")
class PacbotLightZeroEnvTests(unittest.TestCase):
    def setUp(self):
        from lightzero_pacbot.pacbot_env import PacbotLightZeroEnv
        self.Env = PacbotLightZeroEnv

    def play(self, env, rng):
        obs, steps = env.reset(), 0
        while True:
            legal = np.flatnonzero(obs["action_mask"])
            step = env.step(np.array([rng.choice(legal)]))
            obs, steps = step.obs, steps + 1
            if step.done:
                return step, steps

    def test_observations_masks_and_game_ends(self):
        import utils
        env = self.Env({})
        env.seed(0)
        obs = env.reset()
        self.assertEqual(obs["observation"].shape, utils.OBS_SHAPE)
        self.assertEqual(obs["observation"].dtype, np.float32)
        self.assertEqual(obs["action_mask"].tolist(), [int(x) for x in env._env.action_mask()])
        self.assertEqual(obs["to_play"], -1)
        step, steps = self.play(env, np.random.default_rng(0))
        self.assertEqual(step.info["steps"], steps)
        self.assertEqual(step.info["score"], env._env.score())
        self.assertIn(step.info["speed"], range(4, 19))  # equal_moves up to 18 by default

    def test_evaluation_games_are_the_fixed_seed_games(self):
        from utils import ENV_VERSION, make_gym, speed_configuration
        cfgs = self.Env.create_evaluator_env_cfg({"evaluator_env_num": 2, "collector_env_num": 4,
                                                  "eval_games": [(4, 1), (8, 1), (12, 2)]})
        self.assertEqual([c["eval_games"] for c in cfgs], [[(4, 1), (12, 2)], [(8, 1)]])
        env = self.Env(cfgs[0])
        env.seed(5)
        env.reset()
        self.assertEqual(env._speed, 4)
        reference = make_gym(speed_configuration(4), ENV_VERSION)
        reference.seed = 1
        self.assertEqual(env._env.game_state_bytes(), reference.game_state_bytes())
        env.reset()
        self.assertEqual(env._speed, 12)

    def test_games_are_cut_off_at_the_step_limit(self):
        env = self.Env({"max_episode_steps": 5, "random_start_proportion": 0.0})
        env.seed(1)
        env.reset()
        dones = [env.step(np.array([0])).done for _ in range(5)]
        self.assertTrue(dones[-1])


@unittest.skipUnless(HAVE_LIGHTZERO, "needs LightZero (see docs/muzero.md)")
class PolicyTests(unittest.TestCase):
    def test_every_algorithm_searches_as_the_evaluators_call_it(self):
        # LightZero's evaluator passes timestep=; Stochastic MuZero needed a patch to accept it
        # and to search with a policy set up only for evaluation (lightzero_pacbot.stochastic_muzero)
        import torch

        from eval_lightzero_seeds import make_policy
        from lightzero_pacbot.configs import ALGORITHMS
        from lightzero_pacbot.pacbot_env import PacbotLightZeroEnv

        obs = PacbotLightZeroEnv({"eval_games": [(8, 1)]}).reset()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = torch.from_numpy(obs["observation"][None]).to(device)
        legal = np.flatnonzero(obs["action_mask"])
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                policy = make_policy(algorithm, 2, 1, 8).eval_mode
                for extra in ({}, {"timestep": [0]}):
                    out = policy.forward(data, action_mask=[obs["action_mask"]], to_play=[-1],
                                         ready_env_id=np.arange(1), **extra)
                    self.assertIn(out[0]["action"], legal)


if __name__ == "__main__":
    unittest.main()
