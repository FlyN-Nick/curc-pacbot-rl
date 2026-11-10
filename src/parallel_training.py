"""
Background processes for train_dqn: evaluation workers and experience-collecting actors.

Both play games with the Q network on the CPU in their own processes. Playing a game is
thousands of tiny inference calls, and on a GPU each one waits on a round trip, so moving
them off the training process leaves it free to do the gradient updates on the GPU.

- BackgroundEvaluator plays the periodic evaluation games and reports their results later.
- ActorReplayBuffer is a replay buffer whose experience is collected by actor processes, each
  playing a share of the parallel environments while the training process does its update.
"""

from collections import deque
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
import os
import queue
import random
import signal
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

import models
from policies import EpsilonGreedy, MaxQPolicy
from replay_buffer import ReplayBuffer, ReplayItem
from utils import (
    DETERMINISTIC_START_CONFIGURATION,
    NUM_ACTIONS,
    OBS_SHAPE,
    make_gym,
    step_env_until_done,
)

# Spawn (not fork): forking a process that has initialized CUDA is unsafe.
_CTX = mp.get_context("spawn")
_CPU = torch.device("cpu")


def _fresh_seed() -> int:
    return int.from_bytes(os.urandom(8), "little")


def _init_worker(threads: int, seed: int) -> None:
    # The training process handles Ctrl-C and shuts the workers down itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(threads)
    # Every process would otherwise share torch's fixed default seed.
    random.seed(seed)
    torch.manual_seed(seed)


def state_dict_to_numpy(module: torch.nn.Module) -> dict[str, np.ndarray]:
    """A CPU copy of a module's weights that pickles without torch's shared-memory handles."""
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}


@torch.no_grad()
def evaluate_episode(
    q_net: torch.nn.Module, device: torch.device, env_version: int, max_steps: int = 1000
) -> tuple[int, int, bool, int, int, int, float]:
    """
    Plays one evaluation game with the greedy policy.

    Returns (score, total_steps, is_board_cleared, pellets_start, pellets_end,
    purgatory_pellets, ghost_proximities).
    """
    gym = make_gym(DETERMINISTIC_START_CONFIGURATION, env_version)
    pellets_start = gym.remaining_pellets()
    q_net.eval()
    done, step_num = step_env_until_done(gym, MaxQPolicy(q_net), device, max_steps=max_steps)
    is_board_cleared = done and gym.lives() == 3
    pellets_end = 0 if is_board_cleared else gym.remaining_pellets()
    return (
        gym.score(),
        step_num,
        is_board_cleared,
        pellets_start,
        pellets_end,
        gym.purgatory_pellets,
        gym.ghost_proximities,
    )


# ── Evaluation workers ──────────────────────────────────────────────────────────


def _eval_worker(tasks, results, model_name: str, env_version: int, threads: int, seed: int):
    try:
        _init_worker(threads, seed)
        q_net = getattr(models, model_name)(OBS_SHAPE, NUM_ACTIONS)
        while (task := tasks.get()) is not None:
            iter_num, weights = task
            q_net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            results.put((iter_num, evaluate_episode(q_net, _CPU, env_version)))
    except Exception:
        results.put(("error", traceback.format_exc()))


class BackgroundEvaluator:
    """
    Plays evaluation games in worker processes. submit() hands one over if a worker is free
    (otherwise that evaluation is skipped); poll() returns the games finished since the last
    call, each with the payload given to submit().
    """

    def __init__(self, num_workers: int, model_name: str, env_version: int, threads: int = 1):
        self.num_workers = num_workers
        self.skipped = 0
        self._tasks = _CTX.Queue()
        self._results = _CTX.Queue()
        self._in_flight: dict[int, Any] = {}
        self._procs = [
            _CTX.Process(
                target=_eval_worker,
                args=(self._tasks, self._results, model_name, env_version, threads, _fresh_seed()),
                daemon=True,
            )
            for _ in range(num_workers)
        ]
        for proc in self._procs:
            proc.start()

    def has_capacity(self) -> bool:
        return len(self._in_flight) < self.num_workers

    def submit(self, iter_num: int, q_net: torch.nn.Module, make_payload: Callable[[], Any]) -> bool:
        if not self.has_capacity():
            self.skipped += 1
            return False
        self._in_flight[iter_num] = make_payload()
        self._tasks.put((iter_num, state_dict_to_numpy(q_net)))
        return True

    def poll(self) -> list[tuple[int, tuple, Any]]:
        finished = []
        while True:
            try:
                iter_num, result = self._results.get_nowait()
            except queue.Empty:
                break
            if iter_num == "error":
                raise RuntimeError(f"Evaluation worker failed:\n{result}")
            finished.append((iter_num, result, self._in_flight.pop(iter_num)))
        return finished

    def close(self) -> None:
        for _ in self._procs:
            self._tasks.put(None)
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


# ── Actors ───────────────────────────────────────────────────────────────────────


def _actor(
    conn: Connection,
    num_envs: int,
    max_steps: int,
    obs_shm_name: str,
    weights_shm_name: str,
    num_params: int,
    model_name: str,
    epsilon: float,
    threads: int,
    seed: int,
) -> None:
    obs_shm = weights_shm = obs_out = weights = None
    try:
        _init_worker(threads, seed)
        obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
        weights_shm = shared_memory.SharedMemory(name=weights_shm_name)
        obs_out = np.ndarray((max_steps, num_envs, *OBS_SHAPE), np.float32, buffer=obs_shm.buf)
        weights = np.ndarray((num_params,), np.float32, buffer=weights_shm.buf)

        q_net = getattr(models, model_name)(OBS_SHAPE, NUM_ACTIONS).eval()
        policy = EpsilonGreedy(MaxQPolicy(q_net), NUM_ACTIONS, epsilon)
        # The same collection code as single-process training, on this actor's environments
        buffer = ReplayBuffer(
            maxlen=num_envs * max_steps,
            policy=policy,
            num_parallel_envs=num_envs,
            random_start_proportion=0.0,
            device=_CPU,
        )
        obs_out[0] = buffer._last_obs.numpy()
        conn.send(("ready",))

        loaded_version = None
        while True:
            message = conn.recv()
            if message[0] == "stop":
                break
            _, version, epsilon, steps = message
            if version != loaded_version:
                torch.nn.utils.vector_to_parameters(
                    torch.from_numpy(weights.copy()), q_net.parameters()
                )
                loaded_version = version
            policy.epsilon = epsilon
            transitions = []
            for step in range(steps):
                buffer.generate_experience_step()
                obs_out[step] = buffer._last_obs.numpy()
                transitions.append(
                    [
                        (item.action, item.reward, item.next_obs is None, item.next_action_mask)
                        for item in buffer._buffer
                    ]
                )
                buffer._buffer.clear()
            conn.send(("done", transitions))
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        obs_out = weights = None  # views into the shared memory must go before closing it
        for shm in (obs_shm, weights_shm):
            if shm is not None:
                shm.close()


class ActorReplayBuffer:
    """
    A replay buffer filled by actor processes that play the parallel environments with a CPU
    copy of the Q network.

    Each training iteration calls start_collection() before its update and
    finish_collection() after it, so the actors play while the training process updates the
    network. They play with the weights published at start_collection(), one update older
    than single-process training uses; otherwise the transitions and the order they enter the
    buffer match ReplayBuffer's.

    Offers the parts of ReplayBuffer's interface train_dqn uses: policy.epsilon, fill(),
    generate_experience_step(), and sample_batch().
    """

    def __init__(
        self,
        maxlen: int,
        q_net: torch.nn.Module,
        model_name: str,
        epsilon: float,
        num_parallel_envs: int,
        num_actors: int,
        max_steps: int,
        device: torch.device,
        threads_per_actor: int = 2,
    ) -> None:
        if not 1 <= num_actors <= num_parallel_envs:
            raise ValueError(f"num_actors must be 1..{num_parallel_envs}, got {num_actors}")
        if any(True for _ in q_net.buffers()):
            raise ValueError("ActorReplayBuffer only shares parameters; this model has buffers")

        self._buffer = deque[ReplayItem](maxlen=maxlen)
        # train_dqn reads and sets the exploration rate through policy.epsilon
        self.policy = SimpleNamespace(epsilon=epsilon)
        self.device = device
        self._q_net = q_net
        self._max_steps = max_steps
        self._pending_steps = 0
        self._version = 0

        base, extra = divmod(num_parallel_envs, num_actors)
        env_counts = [base + (i < extra) for i in range(num_actors)]
        num_params = sum(p.numel() for p in q_net.parameters())
        self._shms: list[shared_memory.SharedMemory] = []
        self._procs = []
        self._conns: list[Connection] = []
        self._obs_views: list[np.ndarray] = []
        try:
            self._weights_shm = self._new_shm(num_params * 4)
            self._weights = np.ndarray((num_params,), np.float32, buffer=self._weights_shm.buf)
            self._publish_weights()
            for count in env_counts:
                obs_shm = self._new_shm(max_steps * count * int(np.prod(OBS_SHAPE)) * 4)
                self._obs_views.append(
                    np.ndarray((max_steps, count, *OBS_SHAPE), np.float32, buffer=obs_shm.buf)
                )
                parent, child = _CTX.Pipe()
                proc = _CTX.Process(
                    target=_actor,
                    args=(
                        child, count, max_steps, obs_shm.name, self._weights_shm.name,
                        num_params, model_name, epsilon, threads_per_actor, _fresh_seed(),
                    ),
                    daemon=True,
                )
                proc.start()
                self._procs.append(proc)
                self._conns.append(parent)
            self._last_obs = []
            for conn, view in zip(self._conns, self._obs_views):
                self._receive(conn)
                self._last_obs.append(self._to_device(view[0]))
        except BaseException:
            self.close()
            raise

    def _new_shm(self, size: int) -> shared_memory.SharedMemory:
        shm = shared_memory.SharedMemory(create=True, size=size)
        self._shms.append(shm)
        return shm

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        # Always copy: the shared memory is overwritten by the next collection.
        return torch.from_numpy(array).to(self.device, copy=True)

    @staticmethod
    def _receive(conn: Connection):
        message = conn.recv()
        if message[0] == "error":
            raise RuntimeError(f"Actor process failed:\n{message[1]}")
        return message

    def _publish_weights(self) -> None:
        with torch.no_grad():
            vector = torch.nn.utils.parameters_to_vector(self._q_net.parameters())
            self._weights[:] = vector.detach().cpu().numpy()
        self._version += 1

    def start_collection(self, steps: int, epsilon: float) -> None:
        """Publishes the current weights and has every actor play `steps` steps."""
        if self._pending_steps:
            raise RuntimeError("finish_collection() was not called for the previous collection")
        if not 1 <= steps <= self._max_steps:
            raise ValueError(f"steps must be 1..{self._max_steps}, got {steps}")
        self._publish_weights()
        for conn in self._conns:
            conn.send(("collect", self._version, epsilon, steps))
        self._pending_steps = steps

    def finish_collection(self) -> None:
        """Waits for the actors and adds their transitions to the buffer."""
        steps, self._pending_steps = self._pending_steps, 0
        if not steps:
            return
        per_actor = []
        for conn, view in zip(self._conns, self._obs_views):
            _, transitions = self._receive(conn)
            per_actor.append((self._to_device(view[:steps]), transitions))
        # Step by step, environments in order, as ReplayBuffer adds them
        for step in range(steps):
            for i, (new_obs, transitions) in enumerate(per_actor):
                last_obs = self._last_obs[i]
                for j, (action, reward, done, next_action_mask) in enumerate(transitions[step]):
                    next_obs = None if done else new_obs[step, j]
                    self._buffer.append(
                        ReplayItem(last_obs[j], action, reward, next_obs, next_action_mask)
                    )
                self._last_obs[i] = new_obs[step]

    def generate_experience_step(self) -> None:
        self.start_collection(1, self.policy.epsilon)
        self.finish_collection()

    def fill(self) -> None:
        """Collects experience until the buffer is filled to capacity."""
        while len(self._buffer) < self._buffer.maxlen:
            self.start_collection(self._max_steps, self.policy.epsilon)
            self.finish_collection()

    def sample_batch(self, batch_size: int) -> list[ReplayItem]:
        return random.sample(self._buffer, k=batch_size)

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
        # Views into the shared memory must go before closing it
        self._obs_views = []
        self._weights = None
        for shm in self._shms:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shms = []
