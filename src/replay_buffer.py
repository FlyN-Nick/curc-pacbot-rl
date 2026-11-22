from collections import deque
import random
from typing import Generic, NamedTuple, Optional

import torch
import numpy as np

import pacbot_rs

from utils import reset_env, NUM_ACTIONS, P, REPLAY_BUFFER_OBS_GYM_CONFIGURATION, DEFAULT_GYM_CONFIGURATION, replay_buffer_action_mask, step_while_not_policy

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
    ) -> None:
        self._buffer = deque[ReplayItem](maxlen=maxlen)
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

    def _make_current_obs(self) -> torch.Tensor:
        obs = [env.obs_numpy(REPLAY_BUFFER_OBS_GYM_CONFIGURATION) for env in self._envs]
        return torch.from_numpy(np.stack(obs)).to(self.device)

    @property
    def obs_shape(self) -> torch.Size:
        return self._last_obs.shape[1:]

    def fill(self) -> None:
        """Generates experience until the buffer is filled to capacity."""
        while len(self._buffer) < self._buffer.maxlen:
            self.generate_experience_step()

    @torch.no_grad()
    def generate_experience_step(self) -> None:
        """Generates one step of experience for each parallel env and adds them to the buffer."""

        # if other models should take actions in the current state of the games, let them go first
        for env in self._envs:
            step_while_not_policy(env, self.device, reset_if_necessary=True)

        # The other policy may have moved or reset the game.
        self._last_obs = self._make_current_obs()

        # Choose an action using the provided policy.
        action_masks = [replay_buffer_action_mask(env) for env in self._envs]
        action_masks = torch.from_numpy(np.stack(action_masks)).to(self.device)
        actions = self.policy(self._last_obs, action_masks)

        next_obs_stack = []
        for env, last_obs, action in zip(self._envs, self._last_obs, actions.tolist()):
            # Perform the action and observe the transition.
            reward, done = env.step(action)
            if done:
                next_obs = None
                next_action_mask = [False] * NUM_ACTIONS
            else:
                next_obs = torch.from_numpy(env.obs_numpy(REPLAY_BUFFER_OBS_GYM_CONFIGURATION)).to(self.device)
                next_action_mask = replay_buffer_action_mask(env)

            # # Subsample to focus training on end-game states.
            # keep_prob = 1.0 if env.remaining_pellets() < 140 else 0.1
            # if random.random() < keep_prob:
            #     print(f"{env.remaining_pellets()=}")
            #     # Add the transition to the replay buffer.
            #     item = ReplayItem(last_obs, action, reward, next_obs, next_action_mask)
            #     self._buffer.append(item)
            # Add the transition to the replay buffer.
            self._buffer.append(ReplayItem(last_obs, action, reward, next_obs, next_action_mask))

            # Reset the environment if necessary and update last_obs.
            if next_obs is None:
                reset_env(env, REPLAY_BUFFER_OBS_GYM_CONFIGURATION)
                next_obs_stack.append(torch.from_numpy(env.obs_numpy(REPLAY_BUFFER_OBS_GYM_CONFIGURATION)).to(self.device))
            else:
                next_obs_stack.append(next_obs)

        self._last_obs = torch.stack(next_obs_stack)

    def sample_batch(self, batch_size: int) -> list[ReplayItem]:
        """Samples a batch of transitions from the buffer."""
        return random.sample(self._buffer, k=batch_size)
