"""
Replay storage for DQN training in preallocated tensors on the training device, instead of a
deque of Python objects, so sampling a batch is a few indexing operations.

Transitions arrive one experience step at a time: one per environment, in the same
environment order every step (ReplayBuffer and ActorReplayBuffer both add them that way). So
a transition starts from the observation that the same environment's previous transition
ended with, and each observation is stored once. Counting transitions from 0, transition t
ends with observation O(t) and starts from O(t - E), with E environments; O(-E..-1) are the
environments' starting observations. O(t) is kept in row (t + E) mod (capacity + E), which
holds every observation the stored transitions need.

Observations are stored compactly and restored exactly (see ObsCodec).
"""

from typing import NamedTuple

import numpy as np
import torch

from utils import NUM_ACTIONS, OBS_SHAPE

_CHANNELS, _H, _W = OBS_SHAPE
# Channel 1 (rewards) stays float32 and channel 15 (Pacman's speed) is one value per
# observation. The others are 0/1, except 12-14, which are mode or fright steps over a
# fixed duration (see PacmanGym::obs in pacbot_rs/src/env.rs).
_REWARD_CHANNEL, _SPEED_CHANNEL = 1, 15
_BYTE_CHANNELS = [c for c in range(_CHANNELS) if c not in (_REWARD_CHANNEL, _SPEED_CHANNEL)]
_SCALES = {12: 180.0, 13: 180.0, 14: 40.0}


class ObsCodec:
    """
    Encodes observations into about 30% of their float32 size and decodes them exactly: the
    0/1 channels and the step counts (as integers) are stored in bytes, and decoded with the
    same float32 division the environment uses. CUDA divides with exact IEEE rounding, so it
    divides directly; other devices (e.g. MPS) may divide less exactly, so they look each byte
    up in a table computed on the CPU, which is a little slower. Decoding is checked on
    everything encoded, and mismatches are counted (read with mismatches(); the check makes
    the CPU wait for nothing).
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._byte_channels = torch.tensor(_BYTE_CHANNELS, device=device)
        scales = [_SCALES.get(c, 1.0) for c in _BYTE_CHANNELS]
        self._scales = torch.tensor(scales, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        # Decoded value of each byte for each channel: n / scale in float32, as env.rs computes it
        tables = np.arange(256, dtype=np.float32)[None, :] / np.array(scales, dtype=np.float32)[:, None]
        self._tables = torch.from_numpy(tables).to(device)
        self._table_rows = torch.arange(len(scales), device=device).view(1, -1, 1, 1)
        self._divide = device.type == "cuda"
        self._mismatches = torch.zeros((), dtype=torch.long, device=device)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        byte_part = torch.round(obs.index_select(1, self._byte_channels) * self._scales).to(torch.uint8)
        encoded = (byte_part, obs[:, _REWARD_CHANNEL].clone(), obs[:, _SPEED_CHANNEL, 0, 0].clone())
        self._mismatches += (self.decode(*encoded) != obs).sum()
        return encoded

    def decode(self, byte_part, reward_part, speed_part) -> torch.Tensor:
        n = byte_part.shape[0]
        obs = torch.empty((n, *OBS_SHAPE), dtype=torch.float32, device=byte_part.device)
        if self._divide:
            values = byte_part.to(torch.float32) / self._scales
        else:
            values = self._tables[self._table_rows, byte_part.long()]
        # index_copy_ with a device index (not a Python list), so this can run in a CUDA graph
        obs.index_copy_(1, self._byte_channels, values)
        obs[:, _REWARD_CHANNEL] = reward_part
        obs[:, _SPEED_CHANNEL] = speed_part.view(n, 1, 1)
        return obs

    def mismatches(self) -> int:
        """Values encoded so far that didn't decode exactly (waits for the device)."""
        return int(self._mismatches.item())

    @staticmethod
    def bytes_per_obs() -> int:
        return len(_BYTE_CHANNELS) * _H * _W + _H * _W * 4 + 4


class Batch(NamedTuple):
    obs: torch.Tensor
    next_obs: torch.Tensor  # for finished games, the reset game's observation (masked out)
    done: torch.Tensor
    next_action_mask: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor  # unscaled; float64, or float32 on MPS (see ReplayStorage)


class BatchIndices(NamedTuple):
    obs_row: torch.Tensor
    next_row: torch.Tensor
    position: torch.Tensor


class ReplayStorage:
    """
    Holds the most recent `capacity` transitions from `num_envs` environments. Call
    set_initial_obs() once, then add_step() for each experience step; sample_indices() and
    gather() give a batch drawn uniformly without replacement, like random.sample().
    """

    def __init__(self, capacity: int, num_envs: int, device: torch.device) -> None:
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.codec = ObsCodec(device)
        self.total = 0  # transitions added so far
        self._rows = capacity + num_envs
        self._obs_bytes = torch.zeros(
            (self._rows, len(_BYTE_CHANNELS), _H, _W), dtype=torch.uint8, device=device
        )
        self._obs_reward = torch.zeros((self._rows, _H, _W), dtype=torch.float32, device=device)
        self._obs_speed = torch.zeros(self._rows, dtype=torch.float32, device=device)
        self._action = torch.zeros(capacity, dtype=torch.long, device=device)
        # Rewards are kept (and scaled) in float64 like the original Python code did, except on
        # MPS, which has no float64; rewards are integers, so float32 holds them exactly too.
        self._float = torch.float32 if device.type == "mps" else torch.float64
        self._reward = torch.zeros(capacity, dtype=self._float, device=device)
        self._done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._next_mask = torch.zeros((capacity, NUM_ACTIONS), dtype=torch.bool, device=device)
        self._initialized = False

    def __len__(self) -> int:
        return min(self.total, self.capacity)

    def memory_bytes(self) -> int:
        tensors = (self._obs_bytes, self._obs_reward, self._obs_speed, self._action,
                   self._reward, self._done, self._next_mask)
        return sum(t.numel() * t.element_size() for t in tensors)

    @staticmethod
    def _ranges(start: int, count: int, size: int) -> list[tuple[int, int, int]]:
        """(destination start, destination end, source start) for writing `count` items
        from position `start` of a ring of `size`."""
        start %= size
        first = min(count, size - start)
        ranges = [(start, start + first, 0)]
        if first < count:
            ranges.append((0, count - first, first))
        return ranges

    def _write_obs(self, first_row: int, obs: torch.Tensor) -> None:
        byte_part, reward_part, speed_part = self.codec.encode(obs.to(self.device))
        for dst, end, src in self._ranges(first_row, obs.shape[0], self._rows):
            n = end - dst
            self._obs_bytes[dst:end] = byte_part[src:src + n]
            self._obs_reward[dst:end] = reward_part[src:src + n]
            self._obs_speed[dst:end] = speed_part[src:src + n]

    def set_initial_obs(self, obs: torch.Tensor) -> None:
        """The environments' observations before their first step, shape (num_envs, ...)."""
        if self._initialized:
            raise RuntimeError("set_initial_obs() was already called")
        if obs.shape[0] != self.num_envs:
            raise ValueError(f"expected {self.num_envs} observations, got {obs.shape[0]}")
        self._write_obs(0, obs)
        self._initialized = True

    def add_step(self, new_obs, actions, rewards, dones, next_action_masks) -> None:
        """
        One transition per environment, in environment order: the observation each ended
        with (for a finished game, the reset game's), and its action, reward, whether the
        game ended, and the next state's action mask. The values are lists or CPU tensors.
        """
        self.add_steps(new_obs, actions, rewards, dones, next_action_masks)

    def add_steps(self, new_obs, actions, rewards, dones, next_action_masks) -> None:
        """Like add_step() for several consecutive steps at once, steps one after another."""
        if not self._initialized:
            raise RuntimeError("call set_initial_obs() first")
        n = new_obs.shape[0]
        if n % self.num_envs:
            raise ValueError(f"expected whole steps of {self.num_envs} transitions, got {n}")
        self._write_obs(self.total + self.num_envs, new_obs)

        # One copy to the device, from pinned memory: a copy from ordinary memory would make
        # the CPU wait for the device's queued work first.
        packed = torch.cat([
            torch.as_tensor(actions, dtype=self._float).view(n, 1),
            torch.as_tensor(rewards, dtype=self._float).view(n, 1),
            torch.as_tensor(dones, dtype=self._float).view(n, 1),
            torch.as_tensor(next_action_masks, dtype=self._float).view(n, NUM_ACTIONS),
        ], dim=1)
        if self.device.type == "cuda":
            # Pinned, so the copy can run in the background (PyTorch keeps the pinned source
            # alive until it's done)
            packed = packed.pin_memory().to(self.device, non_blocking=True)
        else:
            # Elsewhere (e.g. MPS) a background copy could read this temporary after it's freed
            packed = packed.to(self.device)
        columns = (
            (self._action, packed[:, 0].long()),
            (self._reward, packed[:, 1]),
            (self._done, packed[:, 2].bool()),
            (self._next_mask, packed[:, 3:].bool()),
        )
        for dst, end, src in self._ranges(self.total, n, self.capacity):
            for column, values in columns:
                column[dst:end] = values[src:src + end - dst]
        self.total += n

    def sample_indices(self, batch_size: int) -> BatchIndices:
        """Indices of `batch_size` distinct stored transitions, chosen uniformly."""
        size = len(self)
        if batch_size > size:
            raise ValueError(f"can't sample {batch_size} of {size} transitions")
        # randperm on the device where it's supported well (CUDA), otherwise on the CPU
        where = self.device if self.device.type == "cuda" else "cpu"
        perm = torch.randperm(size, device=where)[:batch_size].to(self.device)
        t = (self.total - size) + perm
        return BatchIndices(t % self._rows, (t + self.num_envs) % self._rows, t % self.capacity)

    def gather(self, idx: BatchIndices) -> Batch:
        def obs(rows):
            return self.codec.decode(self._obs_bytes[rows], self._obs_reward[rows], self._obs_speed[rows])

        return Batch(
            obs=obs(idx.obs_row),
            next_obs=obs(idx.next_row),
            done=self._done[idx.position],
            next_action_mask=self._next_mask[idx.position],
            action=self._action[idx.position],
            reward=self._reward[idx.position],
        )

    def sample(self, batch_size: int) -> Batch:
        return self.gather(self.sample_indices(batch_size))

    def transition(self, t: int) -> dict:
        """Transition t (counted from 0) as Python values, for tests and debugging."""
        if not self.total - len(self) <= t < self.total:
            raise IndexError(f"transition {t} isn't stored")
        idx = BatchIndices(*(torch.tensor([v], device=self.device) for v in (
            t % self._rows, (t + self.num_envs) % self._rows, t % self.capacity)))
        b = self.gather(idx)
        return dict(
            obs=b.obs[0], next_obs=None if b.done.item() else b.next_obs[0],
            done=b.done.item(), next_action_mask=b.next_action_mask[0].tolist(),
            action=b.action.item(), reward=b.reward.item(),
        )
