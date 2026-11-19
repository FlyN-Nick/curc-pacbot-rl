"""
Search at decision time (search.LookaheadPolicy) and PacmanGym.copy.

Run from src/: python -m unittest discover ../tests
"""

import random
import unittest

import torch
from torch import nn

import models
from pacbot_rs import PacmanGym, PacmanGymConfiguration
from search import FULL_OBS, LookaheadPolicy, play_game
from utils import NUM_ACTIONS, OBS_SHAPE

GAMMA, SCALE = 0.99, 0.02


class ConstantQ(nn.Module):
    """The same Q values everywhere."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, obs):
        return self.values.expand(obs.shape[0], -1)


def played_env(steps: int, seed: int = 0) -> PacmanGym:
    rng = random.Random(seed)
    env = PacmanGym(PacmanGymConfiguration({}))
    for _ in range(steps):
        env.step(rng.choice([a for a, ok in enumerate(env.action_mask()) if ok]))
    return env


class CopyTests(unittest.TestCase):
    def test_copy_is_independent(self):
        env = played_env(20)
        before = (env.score(), env.remaining_pellets(), env.game_state_bytes())
        copy = env.copy()
        self.assertEqual(copy.game_state_bytes(), env.game_state_bytes())
        for _ in range(30):
            copy.step(random.choice([a for a, ok in enumerate(copy.action_mask()) if ok]))
        self.assertEqual((env.score(), env.remaining_pellets(), env.game_state_bytes()), before)

    def test_copy_without_reseeding_repeats_the_game(self):
        env = played_env(15)
        a, b = env.copy(reseed=False), env.copy(reseed=False)
        actions = [random.Random(3).choice([0, 1, 2, 3, 4]) for _ in range(200)]
        for action in actions:
            self.assertEqual(a.step(action), b.step(action))
        self.assertEqual(a.game_state_bytes(), b.game_state_bytes())


class LookaheadTests(unittest.TestCase):
    def test_depth_zero_is_the_greedy_policy(self):
        torch.manual_seed(0)
        net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        policy = LookaheadPolicy(net, 0, SCALE, GAMMA)
        env = played_env(10)
        obs = torch.from_numpy(env.obs_numpy(FULL_OBS)).unsqueeze(0)
        q = net(obs)[0]
        legal = [a for a, ok in enumerate(env.action_mask()) if ok]
        self.assertEqual(policy(env), max(legal, key=lambda a: q[a].item()))

    def test_values_are_discounted_rewards_plus_the_leaf_value(self):
        """With constant Q values V, a depth-1 value is reward * scale + gamma * V."""
        v = 3.0
        policy = LookaheadPolicy(ConstantQ([v] * NUM_ACTIONS), 1, SCALE, GAMMA)
        env = played_env(12, seed=4)
        values = policy.action_values(env)
        for action, value in values.items():
            reward, done = env.copy(reseed=False).step(action)
            self.assertAlmostEqual(value, reward * SCALE + (0.0 if done else GAMMA * v), places=5)

    def test_search_leaves_the_game_unchanged(self):
        policy = LookaheadPolicy(ConstantQ([0.0] * NUM_ACTIONS), 3, SCALE, GAMMA)
        env = played_env(25, seed=1)
        before = env.game_state_bytes()
        policy(env)
        self.assertEqual(env.game_state_bytes(), before)
        self.assertGreater(policy.states_searched, 8)

    def test_search_finds_pellets_a_flat_network_cant(self):
        """With no information from the network, depth 3 still eats pellets by looking ahead."""
        flat = ConstantQ([0.0] * NUM_ACTIONS)
        results = {}
        for depth in (0, 3):
            env = PacmanGym(PacmanGymConfiguration({}))
            results[depth] = play_game(env, LookaheadPolicy(flat, depth, SCALE, GAMMA), max_steps=150)
        self.assertGreater(results[3]["pellets_eaten"], results[0]["pellets_eaten"] + 10)


if __name__ == "__main__":
    unittest.main()
