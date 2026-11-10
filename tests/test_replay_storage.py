"""
Tests for replay_storage: observations decode back exactly, and ReplayBuffer with a
ReplayStorage holds exactly the transitions it holds without one. Run from the repository
root with PYTHONPATH=src.
"""

import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import replay_buffer
import utils
from policies import EpsilonGreedy, MaxQPolicy
from replay_storage import ObsCodec, ReplayStorage
sys.path.insert(0, str(Path(__file__).parent))
from test_replay_buffer_batching import FakeEnv, FakeQNet  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE, PacmanGymConfiguration

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
if torch.backends.mps.is_available():  # Apple Silicon GPU
    DEVICES.append(torch.device("mps"))


def real_observations(games: int = 40, steps: int = 300) -> torch.Tensor:
    """Observations from real games: varied starts, speeds, frightened ghosts, both configs."""
    from pacbot_rs import PacmanGym

    C = PacmanGymConfiguration
    starts = [C({}), C({"random_start": True}), C({"random_start": True, "randomize_ghosts": True}),
              C({"random_ticks": True})]
    rng = random.Random(0)
    out = []
    for game in range(games):
        env = PacmanGym(starts[game % len(starts)])
        for step in range(steps):
            for config in (C({}), C({"obs_ignore_super_pellets": True})):
                out.append(env.obs_numpy(config))
            if step % 5 == 0:
                action = rng.choice([a for a, ok in enumerate(env.action_mask()) if ok])
            _, done = env.step(action)
            if done:
                break
    return torch.from_numpy(np.stack(out))


class ObsCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = real_observations()

    def test_real_observations_decode_exactly(self):
        for device in DEVICES:
            codec = ObsCodec(device)
            obs = self.obs.to(device)
            for chunk in obs.split(997):
                decoded = codec.decode(*codec.encode(chunk))
                self.assertTrue(torch.equal(decoded, chunk), device)
            self.assertEqual(codec.mismatches(), 0, device)
        self.assertGreater(len(self.obs), 5_000)

    def test_values_outside_the_format_are_counted(self):
        codec = ObsCodec(DEVICES[-1])
        obs = self.obs[:4].to(DEVICES[-1]).clone()
        obs[0, 5, 3, 3] = 0.5  # a 0/1 channel
        obs[1, 12, 2, 2] = 0.123  # not a whole number of steps
        codec.encode(obs)
        self.assertEqual(codec.mismatches(), 2)


class ReplayStorageTests(unittest.TestCase):
    def collect(self, storage_device, capacity: int, steps: int):
        """ReplayBuffer with deterministic stand-ins, with or without storage."""
        FakeEnv._count = 0
        random.seed(99)
        torch.manual_seed(99)
        super_policy = MaxQPolicy(FakeQNet(offset=7))
        with mock.patch.object(replay_buffer.pacbot_rs, "PacmanGym", FakeEnv), mock.patch.object(
            utils, "safetensors_qnet", lambda *a, **k: super_policy
        ):
            storage = None if storage_device is None else ReplayStorage(capacity, 8, storage_device)
            buffer = replay_buffer.ReplayBuffer(
                maxlen=capacity,
                policy=EpsilonGreedy(MaxQPolicy(FakeQNet(offset=3)), NUM_ACTIONS, 0.5),
                num_parallel_envs=8,
                random_start_proportion=0.0,
                storage=storage,
            )
            for _ in range(steps):
                buffer.generate_experience_step()
        return buffer

    def test_holds_the_same_transitions_as_without_storage(self):
        capacity, steps = 50, 40  # 320 transitions through a ring of 50: wraps mid-step
        expected = list(self.collect(None, capacity, steps)._buffer)
        for device in DEVICES:
            storage = self.collect(device, capacity, steps).storage
            self.assertEqual(len(storage), len(expected))
            first = storage.total - len(storage)
            dones = 0
            for i, item in enumerate(expected):
                got = storage.transition(first + i)
                self.assertEqual(got["action"], item.action, (device, i))
                self.assertEqual(got["reward"], item.reward, (device, i))
                self.assertEqual(got["next_action_mask"], list(item.next_action_mask), (device, i))
                self.assertTrue(torch.equal(got["obs"].cpu(), item.obs), (device, i))
                self.assertEqual(got["next_obs"] is None, item.next_obs is None, (device, i))
                if item.next_obs is None:
                    dones += 1
                else:
                    self.assertTrue(torch.equal(got["next_obs"].cpu(), item.next_obs), (device, i))
            self.assertGreater(dones, 0)
            self.assertEqual(storage.codec.mismatches(), 0)

    def test_samples_distinct_stored_transitions(self):
        for device in DEVICES:
            storage = self.collect(device, 50, 40).storage
            idx = storage.sample_indices(32)
            positions = idx.position.tolist()
            self.assertEqual(len(set(positions)), 32)
            batch = storage.gather(idx)
            self.assertEqual(tuple(batch.obs.shape), (32, *OBS_SHAPE))
            self.assertEqual(batch.obs.dtype, torch.float32)
            with self.assertRaises(ValueError):
                storage.sample_indices(51)

    def test_memory_is_about_a_third(self):
        storage = ReplayStorage(1000, 32, torch.device("cpu"))
        float32_bytes = 1000 * int(np.prod(OBS_SHAPE)) * 4
        self.assertLess(storage.memory_bytes(), 0.32 * float32_bytes)


if __name__ == "__main__":
    unittest.main()
