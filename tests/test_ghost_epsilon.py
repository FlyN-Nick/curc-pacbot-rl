"""
Exploration that depends on how close the dangerous ghosts are (--epsilon_near_ghost):
PacmanGym.nearest_danger_distance, policies.ghost_aware_epsilons, and their use in ReplayBuffer.

Run from src/: python -m unittest discover ../tests
"""

import collections
import random
import unittest

import torch

import replay_buffer
import utils
from pacbot_rs import PacmanGym, PacmanGymConfiguration
from policies import EpsilonGreedy, ghost_aware_epsilons
from utils import NUM_ACTIONS

H, W = 28, 31  # observation layout: [channel, col, 31 - row - 1]


def bfs_nearest_danger(obs) -> int | None:
    """The nearest non-frightened ghost by breadth-first search over the observation's walls."""
    wall = obs[0]
    pacman = tuple(int(i) for i in (obs[3] == 1).nonzero()[0])
    frightened = obs[14] > 0
    targets = set()
    for g in range(4):
        found = (obs[4 + g] == 1).nonzero()
        if len(found):
            cell = tuple(int(i) for i in found[0])
            if not frightened[cell]:
                targets.add(cell)
    if not targets:
        return None
    seen, queue = {pacman}, collections.deque([(pacman, 0)])
    while queue:
        cell, d = queue.popleft()
        if cell in targets:
            return d
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cell[0] + dc, cell[1] + dr)
            if 0 <= nxt[0] < H and 0 <= nxt[1] < W and nxt not in seen and not wall[nxt]:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return None  # e.g. a ghost still in the pen


def shares_a_square(obs) -> bool:
    """Whether two ghosts are on one square (then which of them is frightened is ambiguous)."""
    return bool(((obs[4:8] == 1).sum(dim=0) > 1).any())


class DistanceTests(unittest.TestCase):
    def test_matches_a_search_over_the_maze(self):
        rng = random.Random(0)
        C = PacmanGymConfiguration
        configs = [C({}), C({"random_start": True, "randomize_ghosts": True})]
        checked = frightened_seen = 0
        # 60 seeded random games (the same in every run): frightened ghosts must show up often
        # enough to check
        for game in range(60):
            env = PacmanGym(configs[game % 2], seed=rng.randrange(2**63))
            for _ in range(400):
                obs = torch.from_numpy(env.obs_numpy(C({})))
                if not shares_a_square(obs):
                    self.assertEqual(env.nearest_danger_distance(), bfs_nearest_danger(obs), (game, checked))
                    checked += 1
                    frightened_seen += bool((obs[14] > 0).any())
                _, done = env.step(rng.choice([a for a, ok in enumerate(env.action_mask()) if ok]))
                if done:
                    break
        self.assertGreater(checked, 4000)
        self.assertGreater(frightened_seen, 20)


class EpsilonTests(unittest.TestCase):
    def test_near_and_far_rates(self):
        eps = ghost_aware_epsilons(0.2, [0, 4, 6, 8, 20, None], near_epsilon=0.02, radius=4)
        torch.testing.assert_close(eps, torch.tensor([0.02, 0.02, 0.11, 0.2, 0.2, 0.2]))

    def test_rates_can_also_rise_near_ghosts(self):
        eps = ghost_aware_epsilons(0.1, [2, 6], near_epsilon=0.3, radius=4)
        torch.testing.assert_close(eps, torch.tensor([0.3, 0.2]))

    def test_policy_uses_each_rows_rate(self):
        greedy_action = 2

        def greedy(obs, masks):
            return torch.full((obs.shape[0],), greedy_action)

        policy = EpsilonGreedy(greedy, NUM_ACTIONS, epsilon=0.5)
        obs = torch.zeros(2, 1)
        masks = torch.ones(2, NUM_ACTIONS, dtype=torch.bool)
        random.seed(0)
        torch.manual_seed(0)
        rows = [policy(obs, masks, torch.tensor([0.0, 1.0])).tolist() for _ in range(300)]
        self.assertTrue(all(r[0] == greedy_action for r in rows))
        self.assertEqual(set(r[1] for r in rows), set(range(NUM_ACTIONS)))


class RecordingPolicy:
    """Stand-in policy: records the rates it's given and plays random legal moves."""

    def __init__(self):
        self.epsilon = 0.25
        self.calls = []

    def __call__(self, obs, masks, epsilons=None):
        self.calls.append(epsilons)
        return torch.tensor([random.choice([a for a, ok in enumerate(m) if ok]) for m in masks.tolist()])


class ReplayBufferTests(unittest.TestCase):
    def setUp(self):
        utils.set_single_model(True)
        self.addCleanup(utils.set_single_model, False)

    def test_passes_rates_from_each_games_distance(self):
        policy = RecordingPolicy()
        buffer = replay_buffer.ReplayBuffer(
            maxlen=1000, policy=policy, num_parallel_envs=4, random_start_proportion=0.5,
            near_ghost_epsilon=(0.01, 3),
        )
        random.seed(1)
        for _ in range(300):
            expected = ghost_aware_epsilons(
                policy.epsilon, [env.nearest_danger_distance() for env in buffer._envs], 0.01, 3)
            buffer.generate_experience_step()
            torch.testing.assert_close(policy.calls[-1], expected)
        rates = torch.stack(policy.calls)
        self.assertTrue((rates < policy.epsilon).any())  # some games came near a ghost

    def test_off_by_default(self):
        policy = RecordingPolicy()
        buffer = replay_buffer.ReplayBuffer(
            maxlen=1000, policy=policy, num_parallel_envs=2, random_start_proportion=0.0)
        buffer.generate_experience_step()
        self.assertIsNone(policy.calls[-1])


if __name__ == "__main__":
    unittest.main()
