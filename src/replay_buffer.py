from collections import deque
import random
from typing import Generic, NamedTuple, Optional

import torch
import numpy as np

import pacbot_rs

from utils import reset_env, NUM_ACTIONS, P, replay_buffer_obs_config, DEFAULT_GYM_CONFIGURATION, replay_buffer_action_mask, step_envs_while_not_policy

class ReplayItem(NamedTuple):
    obs: torch.Tensor
    action: int
    reward: int
    next_obs: Optional[torch.Tensor]
    next_action_mask: list[bool]

class ReplayBuffer(Generic[P]):
    """
    Handles gathering experience from an environment instance and storing it in a replay buffer.
    """

    def __init__(
        self,
        maxlen: int,
        policy: P,
        num_parallel_envs: int,
        random_start_proportion: float,
        device: torch.device = torch.device("cpu"),
        storage: "ReplayStorage | None" = None,
    ) -> None:
        """
        With `storage` (a replay_storage.ReplayStorage), transitions go there, and are sampled
        with sample(); otherwise they're kept as ReplayItems, sampled with sample_batch().
        """
        self._buffer = deque[ReplayItem](maxlen=maxlen)
        self._maxlen = maxlen
        self.storage = storage
        self.policy = policy
        self.device = device

        # Initialize the environments.
        self._envs = [
            pacbot_rs.PacmanGym(
                # pacbot_rs.PacmanGymConfiguration({"random_start":i < num_parallel_envs * random_start_proportion, "random_ticks": True})
                DEFAULT_GYM_CONFIGURATION
            )
            for i in range(num_parallel_envs)
        ]
        self._last_obs = self._make_current_obs()
        if storage is not None:
            storage.set_initial_obs(self._last_obs)

    def __len__(self) -> int:
        return len(self._buffer) if self.storage is None else len(self.storage)

    def _make_current_obs(self) -> torch.Tensor:
        obs = [env.obs_numpy(replay_buffer_obs_config()) for env in self._envs]
        return torch.from_numpy(np.stack(obs)).to(self.device)

    @property
    def obs_shape(self) -> torch.Size:
        return self._last_obs.shape[1:]

    def fill(self) -> None:
        """Generates experience until the buffer is filled to capacity."""
        while len(self) < self._maxlen:
            self.generate_experience_step()

    @torch.no_grad()
    def generate_experience_step(self) -> None:
        """Generates one step of experience for each parallel env and adds them to the buffer."""

        # if other models should take actions in the current state of the games, let them go first
        step_envs_while_not_policy(self._envs, self.device)

        # Choose an action using the provided policy.
        action_masks = [replay_buffer_action_mask(env) for env in self._envs]
        action_masks = torch.from_numpy(np.stack(action_masks)).to(self.device)
        actions = self.policy(self._last_obs, action_masks).tolist()

        # Step every environment (resetting finished ones, in order, as before), then move all
        # the new observations to the device in one transfer instead of one per environment.
        transitions = []
        new_obs = []
        for env, action in zip(self._envs, actions):
            reward, done = env.step(action)
            if done:
                next_action_mask = [False] * NUM_ACTIONS
                reset_env(env, replay_buffer_obs_config())
            else:
                next_action_mask = replay_buffer_action_mask(env)
            transitions.append((action, reward, done, next_action_mask))
            new_obs.append(env.obs_numpy(replay_buffer_obs_config()))
        new_obs = torch.from_numpy(np.stack(new_obs)).to(self.device)

        if self.storage is not None:
            self.storage.add_step(new_obs, *zip(*transitions))
            self._last_obs = new_obs
            return

        for last_obs, obs, (action, reward, done, next_action_mask) in zip(
            self._last_obs, new_obs, transitions
        ):
            # A finished game has no next observation; its row of new_obs is the reset game.
            next_obs = None if done else obs

            # # Subsample to focus training on end-game states.
            # keep_prob = 1.0 if env.remaining_pellets() < 140 else 0.1
            # if random.random() < keep_prob:
            #     print(f"{env.remaining_pellets()=}")
            #     # Add the transition to the replay buffer.
            #     item = ReplayItem(last_obs, action, reward, next_obs, next_action_mask)
            #     self._buffer.append(item)
            # Add the transition to the replay buffer.
            self._buffer.append(ReplayItem(last_obs, action, reward, next_obs, next_action_mask))

        self._last_obs = new_obs

    def sample_batch(self, batch_size: int) -> list[ReplayItem]:
        """Samples a batch of transitions from the buffer (without storage)."""
        return random.sample(self._buffer, k=batch_size)

    def sample(self, batch_size: int):
        """Samples a replay_storage.Batch (with storage)."""
        return self.storage.sample(batch_size)
