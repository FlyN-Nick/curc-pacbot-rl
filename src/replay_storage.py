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

Observations are stored compactly and restored exactly (see ObsCodec), except on MPS, where
decoding a batch costs a large share of each update and unified memory has room for plain
float32 (about 1.7 GB for 30,000 transitions).
"""

from typing import NamedTuple

import numpy as np
import torch

from utils import NUM_ACTIONS, OBS_SHAPE

_CHANNELS, _H, _W = OBS_SHAPE
_CELLS = _H * _W
# Observation channels (see PacmanGym::obs in pacbot_rs/src/env.rs):
#   1      pellet / fruit / frightened-ghost values: a small set of float32 values
#   12-14  scatter, chase and fright timers: at the ghosts' squares, a step count over a fixed
#          duration, the same for every ghost (the game frightens all ghosts at once and counts
#          their fright steps down together; the mode timer is the game's)
#   15     Pacman's speed: one value for the whole board
#   others 0/1
_VALUE_CHANNEL, _SPEED_CHANNEL = 1, 15
_TIMER_CHANNELS = [12, 13, 14]
_TIMER_SCALES = [180.0, 180.0, 40.0]  # GameMode::CHASE.duration(), GHOST_FRIGHT_STEPS
_BIT_CHANNELS = [c for c in range(_CHANNELS) if c not in (_VALUE_CHANNEL, _SPEED_CHANNEL, *_TIMER_CHANNELS)]
# Stored as bit planes: the 0/1 channels, then where each timer channel is nonzero
_PLANES = len(_BIT_CHANNELS) + len(_TIMER_CHANNELS)
_PACKED_BYTES = (_PLANES * _CELLS + 7) // 8

# Game values (pacbot-rs variables.rs) and env.rs's LAST_REWARD
_PELLET, _SUPER_PELLET, _FRUIT, _COMBO, _LAST = 10, 50, 100, 200, 3000


def square_values() -> np.ndarray:
    """
    Every float32 value channel 1 can hold, sorted: a square's points (a pellet, super pellet
    or fruit, plus the last-pellet bonus), over 200, plus 2**ghost_combo for each frightened
    ghost on it, added one at a time in float32 as env.rs adds them.
    """
    bases = [0, _PELLET, _SUPER_PELLET, _FRUIT, _PELLET + _LAST, _SUPER_PELLET + _LAST]
    values = set()
    for base in bases:
        start = np.float32(base) / np.float32(_COMBO)
        for combo in range(8):
            v = start
            values.add(float(v))
            for _ in range(4):
                v = np.float32(v + np.float32(2 ** combo))
                values.add(float(v))
    out = np.array(sorted(values), dtype=np.float32)
    assert len(out) <= 256
    return out


class ObsCodec:
    """
    Encodes observations into about 15% of their float32 size and decodes them exactly: the 0/1
    channels as bits, channel 1 as an index into square_values(), each timer channel as where it
    is nonzero (bits) and its value (a byte of steps, decoded with the same float32 division the
    environment uses), and the speed as one float. Decoding is checked on everything encoded,
    and mismatches are counted (read with mismatches(); the check makes the CPU wait for
    nothing). CUDA divides with exact IEEE rounding, so it divides directly; other devices may
    divide less exactly, so they look the timer bytes up in a table computed on the CPU.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._bit_channels = torch.tensor(_BIT_CHANNELS, device=device)
        self._timer_channels = torch.tensor(_TIMER_CHANNELS, device=device)
        self._timer_scales = torch.tensor(_TIMER_SCALES, dtype=torch.float32, device=device).view(1, -1)
        tables = np.arange(256, dtype=np.float32)[None, :] / np.array(_TIMER_SCALES, dtype=np.float32)[:, None]
        self._timer_tables = torch.from_numpy(tables).to(device)
        self._timer_rows = torch.arange(len(_TIMER_SCALES), device=device).view(1, -1)
        self._divide = device.type == "cuda"
        self._values = torch.from_numpy(square_values()).to(device)
        self._shifts = torch.arange(8, dtype=torch.uint8, device=device)
        self._mismatches = torch.zeros((), dtype=torch.long, device=device)

    def _pack(self, bits: torch.Tensor) -> torch.Tensor:
        """(n, k) 0/1 uint8 -> (n, ceil(k / 8)) bytes, least significant bit first."""
        n, k = bits.shape
        pad = _PACKED_BYTES * 8 - k
        if pad:
            bits = torch.cat([bits, bits.new_zeros(n, pad)], dim=1)
        return (bits.view(n, -1, 8) << self._shifts).sum(dim=2, dtype=torch.uint8)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        n = packed.shape[0]
        bits = (packed.unsqueeze(2) >> self._shifts) & 1
        return bits.view(n, -1)[:, :_PLANES * _CELLS].view(n, _PLANES, _H, _W)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        n = obs.shape[0]
        timers = obs.index_select(1, self._timer_channels)
        planes = torch.cat([obs.index_select(1, self._bit_channels), timers != 0], dim=1)
        packed = self._pack(planes.to(torch.uint8).view(n, -1))
        steps = torch.round(timers.amax(dim=(2, 3)) * self._timer_scales).clamp(0, 255).to(torch.uint8)
        values = obs[:, _VALUE_CHANNEL]
        index = torch.searchsorted(self._values, values.contiguous()).clamp(max=len(self._values) - 1)
        encoded = (packed, steps, index.to(torch.uint8), obs[:, _SPEED_CHANNEL, 0, 0].clone())
        self._mismatches += (self.decode(*encoded) != obs).sum()
        return encoded

    def decode(self, packed, steps, index, speed) -> torch.Tensor:
        n = packed.shape[0]
        planes = self._unpack(packed).to(torch.float32)
        if self._divide:
            timer_values = steps.to(torch.float32) / self._timer_scales
        else:
            timer_values = self._timer_tables[self._timer_rows, steps.long()]
        bits = len(_BIT_CHANNELS)
        obs = torch.empty((n, *OBS_SHAPE), dtype=torch.float32, device=packed.device)
        # index_copy_ with a device index (not a Python list), so this can run in a CUDA graph
        obs.index_copy_(1, self._bit_channels, planes[:, :bits])
        obs.index_copy_(1, self._timer_channels, planes[:, bits:] * timer_values.view(n, -1, 1, 1))
        obs[:, _VALUE_CHANNEL] = self._values[index.long()]
        obs[:, _SPEED_CHANNEL] = speed.view(n, 1, 1)
        return obs

    def mismatches(self) -> int:
        """Values encoded so far that didn't decode exactly (waits for the device)."""
        return int(self._mismatches.item())

    @staticmethod
    def stores(rows: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Empty storage for `rows` encoded observations (the parts encode() returns)."""
        return (
            torch.zeros((rows, _PACKED_BYTES), dtype=torch.uint8, device=device),
            torch.zeros((rows, len(_TIMER_CHANNELS)), dtype=torch.uint8, device=device),
            torch.zeros((rows, _H, _W), dtype=torch.uint8, device=device),
            torch.zeros(rows, dtype=torch.float32, device=device),
        )

    @staticmethod
    def bytes_per_obs() -> int:
        return _PACKED_BYTES + len(_TIMER_CHANNELS) + _CELLS + 4


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
    position: torch.Tensor  # (batch,), or (batch, n) with n-step returns


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
        self._plain_obs = device.type == "mps"
        if self._plain_obs:
            self._obs = torch.zeros((self._rows, *OBS_SHAPE), dtype=torch.float32, device=device)
            self._obs_stores = (self._obs,)
        else:
            self._obs_stores = ObsCodec.stores(self._rows, device)
        self._action = torch.zeros(capacity, dtype=torch.long, device=device)
        # Rewards are kept (and scaled) in float64 like the original Python code did, except on
        # MPS, which has no float64; rewards are integers, so float32 holds them exactly too.
        self._float = torch.float32 if device.type == "mps" else torch.float64
        self._reward = torch.zeros(capacity, dtype=self._float, device=device)
        self._done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._next_mask = torch.zeros((capacity, NUM_ACTIONS), dtype=torch.bool, device=device)
        self._initialized = False
        self.n_step = 1
        self._step_discounts = None

    def set_n_step(self, n: int, discount: float) -> None:
        """
        Sample n-step transitions: each starts from a stored transition and follows the same
        environment for n steps, or until its game ends. The batch's reward is then the
        discounted sum of those rewards (sum of discount**k * reward_k), done says whether the
        game ended within them, and next_obs and next_action_mask are the state after the
        n-th; with done False, bootstrap with discount**n.
        """
        if n < 1:
            raise ValueError(f"n_step must be at least 1, got {n}")
        if (n - 1) * self.num_envs >= self.capacity:
            raise ValueError(f"n_step {n} needs a capacity above {(n - 1) * self.num_envs}")
        self.n_step = n
        self._step_discounts = torch.tensor(
            [discount ** k for k in range(n)], dtype=self._float, device=self.device
        )

    def __len__(self) -> int:
        return min(self.total, self.capacity)

    def memory_bytes(self) -> int:
        tensors = (*self._obs_stores, self._action, self._reward, self._done, self._next_mask)
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
        obs = obs.to(self.device)
        self._write_parts(first_row, (obs,) if self._plain_obs else self.codec.encode(obs))

    def _write_parts(self, first_row: int, parts) -> None:
        """Writes stored observations (one tensor per store) from row `first_row` on."""
        for dst, end, src in self._ranges(first_row, parts[0].shape[0], self._rows):
            for store, part in zip(self._obs_stores, parts):
                store[dst:end] = part[src:src + end - dst]

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

    def state_dict(self) -> dict:
        """
        Everything stored, as CPU tensors, for saving a paused run's replay memory to disk
        (see load_state_dict).
        """
        return {
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "plain_obs": self._plain_obs,
            "total": self.total,
            "obs": [t.cpu() for t in self._obs_stores],
            "action": self._action.cpu(),
            "reward": self._reward.cpu(),
            "done": self._done.cpu(),
            "next_mask": self._next_mask.cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restores the transitions of state_dict() into storage that has only its environments'
        starting observations so far (set_initial_obs(), then nothing added). Those games
        continue from here; the saved games were cut off, so each one's last stored
        transition is marked as the end of its game, as nothing will follow it.
        """
        for key, mine in (("capacity", self.capacity), ("num_envs", self.num_envs),
                          ("plain_obs", self._plain_obs)):
            if state[key] != mine:
                raise ValueError(f"saved replay memory has {key} {state[key]}, this run {mine}")
        if not self._initialized or self.total:
            raise RuntimeError("load_state_dict() needs fresh storage with its starting observations")
        start = [store[:self.num_envs].clone() for store in self._obs_stores]
        for store, saved in zip(self._obs_stores, state["obs"]):
            store.copy_(saved)
        for column, key in ((self._action, "action"), (self._reward, "reward"),
                            (self._done, "done"), (self._next_mask, "next_mask")):
            column.copy_(state[key].to(column.dtype))
        self.total = state["total"]
        if self.total:
            last = (torch.arange(self.total - self.num_envs, self.total) % self.capacity).to(self.device)
            self._done[last] = True
        # The current games' observations follow the last stored step
        self._write_parts(self.total, start)

    def sample_indices(self, batch_size: int) -> BatchIndices:
        """Indices of `batch_size` distinct stored transitions, chosen uniformly."""
        size = len(self)
        # With n-step returns, the last n - 1 steps' transitions can't start one yet
        starts = size - (self.n_step - 1) * self.num_envs
        if batch_size > starts:
            raise ValueError(f"can't sample {batch_size} of {max(starts, 0)} transitions")
        # randperm on the device where it's supported well (CUDA), otherwise on the CPU
        where = self.device if self.device.type == "cuda" else "cpu"
        perm = torch.randperm(starts, device=where)[:batch_size].to(self.device)
        t = (self.total - size) + perm
        if self.n_step == 1:
            return BatchIndices(t % self._rows, (t + self.num_envs) % self._rows, t % self.capacity)
        steps = torch.arange(self.n_step, device=self.device) * self.num_envs
        return BatchIndices(
            t % self._rows,
            (t + self.n_step * self.num_envs) % self._rows,
            (t.unsqueeze(1) + steps) % self.capacity,
        )

    def gather(self, idx: BatchIndices) -> Batch:
        def obs(rows):
            if self._plain_obs:
                return self._obs[rows]
            return self.codec.decode(*(store[rows] for store in self._obs_stores))

        if idx.position.dim() == 1:
            return Batch(
                obs=obs(idx.obs_row),
                next_obs=obs(idx.next_row),
                done=self._done[idx.position],
                next_action_mask=self._next_mask[idx.position],
                action=self._action[idx.position],
                reward=self._reward[idx.position],
            )
        # n-step: rewards count until the game ends (a later step belongs to the next game)
        dones = self._done[idx.position]
        playing = torch.cumprod((~dones).to(self._float), dim=1)
        playing = torch.cat([torch.ones_like(playing[:, :1]), playing[:, :-1]], dim=1)
        return Batch(
            obs=obs(idx.obs_row),
            next_obs=obs(idx.next_row),
            done=dones.any(dim=1),
            next_action_mask=self._next_mask[idx.position[:, -1]],
            action=self._action[idx.position[:, 0]],
            reward=(self._reward[idx.position] * playing * self._step_discounts).sum(dim=1),
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
