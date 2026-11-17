"""
Search distillation (--search_workers): training the Q network toward what search at decision
time (search.LookaheadPolicy) finds, so the network itself plays more like the search does.

Searcher processes play their own games on the CPU with the newest weights the actors use
(ActorReplayBuffer shares them). Before each move a searcher searches every legal move
`depth` moves deep; the searched value of each move, its rewards plus the network's value of
where it ends (see search.py), is a target for the network's Q value of that move in that
state: a depth-step Bellman optimality backup, through the real game rather than through one
sampled transition. The searcher then plays the best move (or, with probability `epsilon`, a
random one), so its states are the ones a searching agent reaches, like the robot with search.

SearchTargets keeps the newest `capacity` searched states on the training device; each update
adds search_weight times the mean squared error between the network's Q values and the
searched values, over the legal moves of search_batch_size of them (dqn_update's `search`).
"""

from __future__ import annotations

import random
import signal
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import Connection, wait
from typing import NamedTuple

import numpy as np
import torch

import parallel_training
from replay_storage import ObsCodec
from utils import NUM_ACTIONS, OBS_SHAPE

# Records a searcher sends at once
_CHUNK = 8


class SearchBatch(NamedTuple):
    obs: torch.Tensor     # (batch, *OBS_SHAPE) float32
    values: torch.Tensor  # (batch, NUM_ACTIONS) float32, searched values; 0 for illegal moves
    legal: torch.Tensor   # (batch, NUM_ACTIONS) bool


def _searcher(
    conn: Connection,
    weights_shm_name: str,
    weights_lock,
    num_params: int,
    model_name: str,
    seed: int,
    depth: int,
    reward_scale: float,
    discount: float,
    epsilon: float,
    random_start_proportion: float,
    random_pellet_removal: bool,
) -> None:
    weights_shm = shared = None
    try:
        parallel_training._init_worker(1, seed)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the training process stops searchers itself
        from search import LookaheadPolicy
        from utils import ENV_VERSION, make_gym, random_start_configuration, replay_buffer_obs_config, reset_env

        weights_shm = shared_memory.SharedMemory(name=weights_shm_name)
        shared = np.ndarray((parallel_training._HEADER + num_params,), np.float64, buffer=weights_shm.buf)
        q_net = parallel_training._cpu_model(model_name)
        obs_config = replay_buffer_obs_config()
        policy = LookaheadPolicy(q_net, depth, reward_scale, discount, obs_config=obs_config)

        def reset_config():
            if random.random() < random_start_proportion:
                return random_start_configuration(random_pellet_removal)
            return obs_config

        env = make_gym(reset_config(), ENV_VERSION)
        loaded_version = None
        obs_out, values_out, legal_out = [], [], []
        conn.send(("ready",))
        while True:
            if conn.poll() and conn.recv()[0] == "stop":
                return
            if shared[0] != loaded_version:
                with weights_lock:
                    loaded_version = shared[0]
                    params = torch.from_numpy(shared[parallel_training._HEADER:].astype(np.float32))
                parallel_training._load_vector(params, q_net)

            values = policy.action_values(env)
            row = np.zeros(NUM_ACTIONS, np.float32)
            legal = np.zeros(NUM_ACTIONS, bool)
            for action, value in values.items():
                row[action] = value
                legal[action] = True
            obs_out.append(env.obs_numpy(obs_config))
            values_out.append(row)
            legal_out.append(legal)
            if len(obs_out) == _CHUNK:
                conn.send(("records", np.stack(obs_out), np.stack(values_out), np.stack(legal_out)))
                obs_out, values_out, legal_out = [], [], []

            if random.random() < epsilon:
                action = random.choice(list(values))
            else:
                action = max(values, key=values.get)
            _, done = env.step(action)
            if done:
                reset_env(env, reset_config())
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        shared = None  # views into the shared memory must go before closing it
        if weights_shm is not None:
            weights_shm.close()


class SearchStore:
    """
    The newest `capacity` searched states (observation, searched values, legal moves) on
    `device`, observations stored as bits like replay_storage. sample_indices() and gather()
    give a batch, uniformly with replacement.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.codec = ObsCodec(device)
        self._plain_obs = device.type == "mps"
        if self._plain_obs:
            self._obs_stores = (torch.zeros((capacity, *OBS_SHAPE), dtype=torch.float32, device=device),)
        else:
            self._obs_stores = ObsCodec.stores(capacity, device)
        self._values = torch.zeros((capacity, NUM_ACTIONS), dtype=torch.float32, device=device)
        self._legal = torch.zeros((capacity, NUM_ACTIONS), dtype=torch.bool, device=device)
        self.total = 0  # states added so far

    def __len__(self) -> int:
        return min(self.total, self.capacity)

    def add(self, obs: np.ndarray, values: np.ndarray, legal: np.ndarray) -> None:
        n = obs.shape[0]
        obs_t = torch.from_numpy(obs).to(self.device)
        parts = (obs_t,) if self._plain_obs else self.codec.encode(obs_t)
        columns = list(zip(self._obs_stores, parts)) + [
            (self._values, torch.from_numpy(values).to(self.device)),
            (self._legal, torch.from_numpy(legal).to(self.device)),
        ]
        start = self.total % self.capacity
        first = min(n, self.capacity - start)
        for store, part in columns:
            store[start:start + first] = part[:first]
            if first < n:
                store[:n - first] = part[first:]
        self.total += n

    def sample_indices(self, batch_size: int) -> torch.Tensor:
        if len(self) == 0:
            raise ValueError("no searched states yet")
        return torch.randint(len(self), (batch_size,), device=self.device)

    def gather(self, idx: torch.Tensor) -> SearchBatch:
        if self._plain_obs:
            obs = self._obs_stores[0][idx]
        else:
            obs = self.codec.decode(*(store[idx] for store in self._obs_stores))
        return SearchBatch(obs, self._values[idx], self._legal[idx])


class SearchTargets(SearchStore):
    """
    Searcher processes (see the module docstring), and a SearchStore of the states they've
    searched. drain() takes in what they've sent so far.
    """

    def __init__(
        self,
        weights: tuple[str, object, int],
        num_workers: int,
        capacity: int,
        device: torch.device,
        model_name: str,
        depth: int,
        reward_scale: float,
        discount: float,
        epsilon: float,
        random_start_proportion: float = 0.0,
        random_pellet_removal: bool = False,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be at least 1, got {num_workers}")
        if depth < 1:
            raise ValueError(f"search depth must be at least 1, got {depth}")
        super().__init__(capacity, device)
        self._procs = []
        self._conns: list[Connection] = []
        weights_name, weights_lock, num_params = weights
        try:
            for _ in range(num_workers):
                parent, child = parallel_training._CTX.Pipe()
                proc = parallel_training._CTX.Process(
                    target=_searcher,
                    args=(child, weights_name, weights_lock, num_params, model_name,
                          parallel_training._fresh_seed(), depth, reward_scale, discount, epsilon,
                          random_start_proportion, random_pellet_removal),
                    daemon=True,
                )
                proc.start()
                self._procs.append(proc)
                self._conns.append(parent)
            for conn in self._conns:
                self._receive(conn)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(conn: Connection):
        message = conn.recv()
        if message[0] == "error":
            raise RuntimeError(f"Searcher process failed:\n{message[1]}")
        return message

    def drain(self, at_least: int = 0) -> int:
        """Stores what the searchers have sent, waiting until at least `at_least` states are
        stored in all. Returns how many states were added."""
        added = 0
        while True:
            for conn in self._conns:
                while conn.poll():
                    _, obs, values, legal = self._receive(conn)
                    self.add(obs, values, legal)
                    added += obs.shape[0]
            if len(self) >= at_least:
                return added
            wait(self._conns, timeout=1.0)

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        self._procs, self._conns = [], []
