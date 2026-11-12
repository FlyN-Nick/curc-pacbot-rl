"""
Checks that the batched experience collection (ReplayBuffer.generate_experience_step with
utils.step_envs_while_not_policy) produces exactly the replay data of the original
one-environment-at-a-time code, which is frozen below as the reference.

The real game engine seeds each game from OS randomness, so the comparison uses
deterministic stand-in environments and stand-in models whose outputs don't depend on
batch size. Run from the repository root with PYTHONPATH=src.
"""

import random
import unittest
from unittest import mock

import numpy as np
import torch

import policies
import replay_buffer
import utils
from policies import MaxQPolicy
from utils import NUM_ACTIONS, OBS_SHAPE, PacmanGymConfiguration


class FakeEnv:
    """Deterministic stand-in for pacbot_rs.PacmanGym with the methods training uses."""

    _count = 0

    def __init__(self, config: PacmanGymConfiguration) -> None:
        FakeEnv._count += 1
        self.id = FakeEnv._count
        self.resets = 0
        self.reset(config)

    def reset(self, config: PacmanGymConfiguration) -> None:
        self.resets += 1
        self.state = (self.id * 97 + self.resets * 13) % 1009
        self.steps = 0
        self.limit = 15 + (self.id * 7 + self.resets) % 20

    def step(self, action: int) -> tuple[int, bool]:
        self.state = (self.state * 31 + action * 7 + 3) % 1009
        self.steps += 1
        return self.state % 11 - 5, self.steps >= self.limit

    def are_ghosts_close(self) -> bool:
        return self.state % 5 == 0

    def all_ghosts_not_frightened(self) -> bool:
        return self.state % 7 != 0

    def action_mask(self) -> list[bool]:
        return [True] + [(self.state >> k) & 1 == 0 for k in range(NUM_ACTIONS - 1)]

    def purgatory_action_mask(self) -> list[bool]:
        mask = self.action_mask()
        mask[1 + self.state % (NUM_ACTIONS - 1)] = False
        return mask

    def obs_numpy(self, config: PacmanGymConfiguration) -> np.ndarray:
        obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        s = self.state
        obs[0, s % 28, s % 31] = 1.0
        obs[1, (s // 3) % 28, (s // 5) % 31] = (s % 13) / 13
        obs[15] = (s % 9) / 12
        if not config.obs_ignore_super_pellets:
            obs[16, s % 2, s % 3] = 1.0
        return obs


class FakeQNet(torch.nn.Module):
    """Q values read straight from fixed observation entries, so each row is batch-independent."""

    def __init__(self, offset: int) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        flat = obs.flatten(1)
        idx = [self.offset + 31 * k for k in range(NUM_ACTIONS)]
        return flat[:, idx] + obs[:, 15, 0, 0].unsqueeze(1) * torch.arange(NUM_ACTIONS)


# ── Reference: the original implementation, frozen ──────────────────────────────


class ReferenceEpsilonGreedy:
    def __init__(self, original_policy, num_actions: int, epsilon: float) -> None:
        self.original_policy = original_policy
        self.num_actions = num_actions
        self.epsilon = epsilon

    def __call__(self, obs, action_masks):
        batch_size = obs.shape[0]
        device = obs.device
        actions = [
            random.choice([i for i, valid in enumerate(action_mask.tolist()) if valid])
            for action_mask in action_masks
        ]
        actions = torch.tensor(actions, device=device, dtype=torch.int64)
        greedy_mask = torch.rand(batch_size, device=device) > self.epsilon
        if greedy_mask.any():
            greedy_actions = self.original_policy(obs[greedy_mask], action_masks[greedy_mask])
            greedy_actions = greedy_actions.to(device=device)
            actions[greedy_mask] = greedy_actions
        return actions


def reference_step_while_not_policy(env, device, reset_if_necessary: bool = True):
    while env.are_ghosts_close() or not env.all_ghosts_not_frightened():
        _, done = utils._step_always_with(
            env,
            utils.safetensors_qnet("checkpoints_known/superpelletmode.safetensors", device=device),
            utils.SUPER_PELLET_MODE_CONFIGUARTION,
            env.action_mask(),
            device,
        )
        if done:
            if reset_if_necessary:
                utils.reset_env(env, utils.DEFAULT_GYM_CONFIGURATION)
            else:
                return


class ReferenceReplayBuffer(replay_buffer.ReplayBuffer):
    @torch.no_grad()
    def generate_experience_step(self) -> None:
        for env in self._envs:
            reference_step_while_not_policy(env, self.device, reset_if_necessary=True)
        action_masks = [utils.replay_buffer_action_mask(env) for env in self._envs]
        action_masks = torch.from_numpy(np.stack(action_masks)).to(self.device)
        actions = self.policy(self._last_obs, action_masks)
        next_obs_stack = []
        for env, last_obs, action in zip(self._envs, self._last_obs, actions.tolist()):
            reward, done = env.step(action)
            if done:
                next_obs = None
                next_action_mask = [False] * NUM_ACTIONS
            else:
                next_obs = torch.from_numpy(
                    env.obs_numpy(utils.REPLAY_BUFFER_OBS_GYM_CONFIGURATION)
                ).to(self.device)
                next_action_mask = utils.replay_buffer_action_mask(env)
            self._buffer.append(
                replay_buffer.ReplayItem(last_obs, action, reward, next_obs, next_action_mask)
            )
            if next_obs is None:
                utils.reset_env(env, utils.REPLAY_BUFFER_OBS_GYM_CONFIGURATION)
                next_obs_stack.append(
                    torch.from_numpy(env.obs_numpy(utils.REPLAY_BUFFER_OBS_GYM_CONFIGURATION)).to(
                        self.device
                    )
                )
            else:
                next_obs_stack.append(next_obs)
        self._last_obs = torch.stack(next_obs_stack)


# ── Test ─────────────────────────────────────────────────────────────────────────


class ReplayBufferBatchingTests(unittest.TestCase):
    def collect(self, buffer_class, policy_class, steps: int):
        FakeEnv._count = 0
        random.seed(1234)
        torch.manual_seed(1234)
        super_pellet_policy = MaxQPolicy(FakeQNet(offset=7))
        with mock.patch.object(replay_buffer.pacbot_rs, "PacmanGym", FakeEnv), mock.patch.object(
            utils, "safetensors_qnet", lambda *args, **kwargs: super_pellet_policy
        ):
            policy = policy_class(MaxQPolicy(FakeQNet(offset=3)), NUM_ACTIONS, 0.5)
            buffer = buffer_class(
                maxlen=10_000, policy=policy, num_parallel_envs=8, random_start_proportion=0.0
            )
            for _ in range(steps):
                buffer.generate_experience_step()
        return buffer

    def test_matches_reference_implementation(self):
        steps = 60
        expected = self.collect(ReferenceReplayBuffer, ReferenceEpsilonGreedy, steps)
        actual = self.collect(replay_buffer.ReplayBuffer, policies.EpsilonGreedy, steps)

        self.assertEqual(len(actual._buffer), len(expected._buffer))
        self.assertEqual(len(actual._buffer), steps * 8)
        dones = super_steps = 0
        for i, (a, e) in enumerate(zip(actual._buffer, expected._buffer)):
            self.assertEqual(a.action, e.action, f"item {i}")
            self.assertEqual(a.reward, e.reward, f"item {i}")
            self.assertEqual(list(a.next_action_mask), list(e.next_action_mask), f"item {i}")
            self.assertTrue(torch.equal(a.obs, e.obs), f"item {i}")
            self.assertEqual(a.next_obs is None, e.next_obs is None, f"item {i}")
            if e.next_obs is None:
                dones += 1
            else:
                self.assertTrue(torch.equal(a.next_obs, e.next_obs), f"item {i}")
        self.assertTrue(torch.equal(actual._last_obs, expected._last_obs))
        # The stand-ins must exercise episode ends and the super pellet model
        self.assertGreater(dones, 5)
        for env in actual._envs:
            super_steps += env.resets
        self.assertGreater(super_steps, len(actual._envs) * 2)

    def test_super_pellet_model_runs_in_reference_too(self):
        # Guard against a stand-in that never leaves purgatory (which would make the
        # comparison above skip the super pellet model entirely)
        calls = []
        policy = MaxQPolicy(FakeQNet(offset=7))

        def counting_policy(obs, masks):
            calls.append(obs.shape[0])
            return policy(obs, masks)

        FakeEnv._count = 0
        with mock.patch.object(utils, "safetensors_qnet", lambda *a, **k: counting_policy):
            envs = [FakeEnv(utils.DEFAULT_GYM_CONFIGURATION) for _ in range(8)]
            for _ in range(30):
                utils.step_envs_while_not_policy(envs, torch.device("cpu"))
                for env in envs:
                    env.step(1)
        self.assertTrue(any(n > 1 for n in calls), calls)


if __name__ == "__main__":
    unittest.main()
