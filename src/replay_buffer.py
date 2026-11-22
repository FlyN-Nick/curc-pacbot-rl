from collections import deque
import random
from typing import Generic, NamedTuple, Optional

import torch
import numpy as np

import pacbot_rs

from policies import ghost_aware_epsilons

from speed_sampling import SpeedSampler
from utils import ghost_next_obs, training_configuration, reset_env, single_model, NUM_ACTIONS, P, replay_buffer_obs_config, DEFAULT_GYM_CONFIGURATION, replay_buffer_action_mask, step_envs_while_not_policy

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
        random_pellet_removal: bool = False,
        near_ghost_epsilon: "tuple[float, int] | None" = None,
    ) -> None:
        """
        With `near_ghost_epsilon` (exploration rate, radius), the policy (an EpsilonGreedy)
        explores at that rate near dangerous ghosts (see policies.ghost_aware_epsilons).
        With `storage` (a replay_storage.ReplayStorage), transitions go there, and are sampled
        with sample(); otherwise they're kept as ReplayItems, sampled with sample_batch().
        """
        self._buffer = deque[ReplayItem](maxlen=maxlen)
        self._maxlen = maxlen
        self.storage = storage
        self.policy = policy
        self.device = device
        self.near_ghost_epsilon = near_ghost_epsilon

        # Initialize the environments. The first `random_start_proportion` of them start each
        # game after the first from a random Pacman position, on a board with half its pellets
        # removed 4 times in 5 (PacmanGymConfiguration.random_start), or with
        # `random_pellet_removal` a random share of them, for practice at clearing the board;
        # the rest start normally.
        if random_start_proportion > 0 and not single_model():
            raise ValueError("random_start_proportion needs --single-model: the super pellet "
                             "model resets the games it finishes with the normal start")
        num_random = round(num_parallel_envs * random_start_proportion)
        self._random_start = [i < num_random for i in range(num_parallel_envs)]
        self._random_pellet_removal = random_pellet_removal
        # Each game's speed (utils.speed_sampling: uniform, or equal moves per speed)
        self.speed_sampler = SpeedSampler()
        self._speeds = [self.speed_sampler.draw() for _ in range(num_parallel_envs)]
        self._moves = [0] * num_parallel_envs
        self._envs = [pacbot_rs.PacmanGym(DEFAULT_GYM_CONFIGURATION) for _ in range(num_parallel_envs)]
        for env, speed in zip(self._envs, self._speeds):
            env.ghost_next_obs = ghost_next_obs()
            # The first game too starts normally, at its drawn speed
            reset_env(env, training_configuration(speed))

        self._last_obs = self._make_current_obs()
        if storage is not None:
            storage.set_initial_obs(self._last_obs)

    def _reset_configuration(self, i: int):
        """The configuration environment i's next game starts with: its drawn speed, and a random
        start (with random_pellet_removal, if set) for the first random_start_proportion."""
        random_start = self._random_start[i]
        return training_configuration(self._speeds[i], random_start, random_start and self._random_pellet_removal)

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

        # The other policy may have moved or reset the game.
        self._last_obs = self._make_current_obs()

        # Choose an action using the provided policy.
        action_masks = [replay_buffer_action_mask(env) for env in self._envs]
        action_masks = torch.from_numpy(np.stack(action_masks)).to(self.device)
        if self.near_ghost_epsilon is None:
            actions = self.policy(self._last_obs, action_masks).tolist()
        else:
            epsilons = ghost_aware_epsilons(
                self.policy.epsilon, [env.nearest_danger_distance() for env in self._envs],
                *self.near_ghost_epsilon,
            )
            actions = self.policy(self._last_obs, action_masks, epsilons).tolist()

        # Step every environment (resetting finished ones, in order, as before), then move all
        # the new observations to the device in one transfer instead of one per environment.
        transitions = []
        new_obs = []
        for i, (env, action) in enumerate(zip(self._envs, actions)):
            reward, done = env.step(action)
            self._moves[i] += 1
            if done:
                next_action_mask = [False] * NUM_ACTIONS
                self.speed_sampler.record(self._speeds[i], self._moves[i])
                self._speeds[i], self._moves[i] = self.speed_sampler.draw(), 0
                reset_env(env, self._reset_configuration(i))
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
