"""
n-step returns (ReplayStorage.set_n_step): sampled transitions follow the same environment for
n steps, or until its game ends.

Run from src/: python -m unittest discover ../tests
"""

import random
import unittest

import torch

from replay_storage import ReplayStorage
from utils import NUM_ACTIONS, OBS_SHAPE

DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
ENVS, CAPACITY, STEPS = 3, 60, 50
GAMMA = 0.9


def marked_obs(markers: list[int]) -> torch.Tensor:
    """Observations told apart by channel 1 (stored exactly), the rest valid 0/1 values."""
    obs = torch.zeros((len(markers), *OBS_SHAPE))
    obs[:, 1, 0, 0] = torch.tensor(markers, dtype=torch.float32)
    return obs


def fill(device, n: int):
    """A storage with STEPS random steps. Observation O(t) is marked t + ENVS + 1."""
    rng = random.Random(n)
    storage = ReplayStorage(CAPACITY, ENVS, device)
    storage.set_n_step(n, GAMMA)
    storage.set_initial_obs(marked_obs(list(range(1, ENVS + 1))))
    log = []  # (action, reward, done, next_mask) of each transition t
    for step in range(STEPS):
        first = step * ENVS
        rows = [(rng.randrange(NUM_ACTIONS), rng.randrange(-5, 50), rng.random() < 0.15,
                 [rng.random() < 0.7 for _ in range(NUM_ACTIONS)]) for _ in range(ENVS)]
        log += rows
        storage.add_step(marked_obs([first + e + ENVS + 1 for e in range(ENVS)]), *zip(*rows))
    return storage, log


class NStepTests(unittest.TestCase):
    def check(self, n: int):
        for device in DEVICES:
            storage, log = fill(device, n)
            total = len(log)
            batch = storage.sample(storage.capacity - (n - 1) * ENVS)
            starts = (batch.obs[:, 1, 0, 0].round().long() - 1).tolist()  # O(t - E) is marked t + 1
            # Every transition that has n steps after it stored, once each
            self.assertEqual(sorted(starts), list(range(total - CAPACITY, total - (n - 1) * ENVS)))
            for i, t in enumerate(starts):
                expected_reward, done, k = 0.0, False, 0
                for k in range(n):
                    _, reward, done, _ = log[t + k * ENVS]
                    expected_reward += GAMMA ** k * reward
                    if done:
                        break
                last = t + (n - 1) * ENVS
                self.assertAlmostEqual(batch.reward[i].item(), expected_reward, places=9, msg=(device, t))
                self.assertEqual(batch.done[i].item(), done, (device, t))
                self.assertEqual(batch.action[i].item(), log[t][0], (device, t))
                if not done:
                    self.assertEqual(batch.next_action_mask[i].tolist(), log[last][3], (device, t))
                    self.assertEqual(batch.next_obs[i, 1, 0, 0].item(), last + ENVS + 1, (device, t))

    def test_one_step_is_unchanged(self):
        self.check(1)

    def test_three_steps(self):
        self.check(3)

    def test_five_steps(self):
        self.check(5)

    def test_rejects_too_long_a_window(self):
        storage = ReplayStorage(CAPACITY, ENVS, torch.device("cpu"))
        with self.assertRaises(ValueError):
            storage.set_n_step(CAPACITY // ENVS + 1, GAMMA)
        with self.assertRaises(ValueError):
            storage.set_n_step(0, GAMMA)


if __name__ == "__main__":
    unittest.main()
