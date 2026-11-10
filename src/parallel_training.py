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
    SUPER_PELLET_MODEL_PATH,
    make_gym,
    safetensors_qnet,
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


def _cpu_model(model_name: str) -> torch.nn.Module:
    """
    A model for CPU inference in channels-last layout, which makes these small convolutions
    up to ~3x faster on the CPU with identical results. Also converts the super pellet model.
    """
    safetensors_qnet(SUPER_PELLET_MODEL_PATH, device=_CPU).q_net.to(memory_format=torch.channels_last)
    model = getattr(models, model_name)(OBS_SHAPE, NUM_ACTIONS).eval()
    return model.to(memory_format=torch.channels_last)


def _load_vector(vector: torch.Tensor, module: torch.nn.Module) -> None:
    """Like vector_to_parameters, but copies into the parameters so they keep their layout."""
    offset = 0
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(vector[offset:offset + param.numel()].view_as(param))
            offset += param.numel()


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
        q_net = _cpu_model(model_name)
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

# Header of the shared weights block: version, then epsilon, then the parameters
_HEADER = 2


def _actor(
    conn: Connection,
    num_envs: int,
    num_slots: int,
    obs_shm_name: str,
    weights_shm_name: str,
    weights_lock,
    num_params: int,
    model_name: str,
    threads: int,
    seed: int,
) -> None:
    """
    Plays this actor's environments continuously. Each experience step goes into a free slot of
    the shared observation ring and is announced as ("step", slot, transitions); the learner
    hands slots back with ("free", slot). With every slot waiting on the learner, it pauses.
    """
    obs_shm = weights_shm = obs_out = shared = None
    try:
        _init_worker(threads, seed)
        obs_shm = shared_memory.SharedMemory(name=obs_shm_name)
        weights_shm = shared_memory.SharedMemory(name=weights_shm_name)
        obs_out = np.ndarray((num_slots, num_envs, *OBS_SHAPE), np.float32, buffer=obs_shm.buf)
        shared = np.ndarray((_HEADER + num_params,), np.float64, buffer=weights_shm.buf)

        q_net = _cpu_model(model_name)
        policy = EpsilonGreedy(MaxQPolicy(q_net), NUM_ACTIONS, 1.0)
        # The same collection code as single-process training, on this actor's environments
        buffer = ReplayBuffer(
            maxlen=num_envs, policy=policy, num_parallel_envs=num_envs,
            random_start_proportion=0.0, device=_CPU,
        )
        obs_out[0] = buffer._last_obs.numpy()
        conn.send(("ready",))
        # Slot 0 holds the starting observations until the learner hands it back
        free_slots = deque(range(1, num_slots))
        loaded_version = None

        while True:
            # Take back slots the learner is done with; wait for one if none are free.
            while conn.poll() or not free_slots:
                message = conn.recv()
                if message[0] == "stop":
                    return
                free_slots.append(message[1])
            if shared[0] != loaded_version:
                with weights_lock:
                    loaded_version = shared[0]
                    policy.epsilon = float(shared[1])
                    params = torch.from_numpy(shared[_HEADER:].astype(np.float32))
                _load_vector(params, q_net)

            buffer.generate_experience_step()
            slot = free_slots.popleft()
            obs_out[slot] = buffer._last_obs.numpy()
            conn.send((
                "step",
                slot,
                [(i.action, i.reward, i.next_obs is None, i.next_action_mask) for i in buffer._buffer],
            ))
            buffer._buffer.clear()
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        obs_out = shared = None  # views into the shared memory must go before closing it
        for shm in (obs_shm, weights_shm):
            if shm is not None:
                shm.close()


class ActorReplayBuffer:
    """
    A replay buffer filled by actor processes that play the parallel environments with a CPU
    copy of the Q network.

    The actors play continuously, each up to `ahead_steps` experience steps ahead of what the
    training process has taken, using the most recently published weights. Each training
    iteration calls start_collection() before its update, which publishes the weights and
    exploration rate, and finish_collection() after it, which takes `steps` experience steps
    from every actor. So every environment still contributes the same number of transitions
    per iteration, in the same buffer order as ReplayBuffer, from the same collection code;
    they were just played with weights up to a few updates older. Running ahead keeps an
    actor stuck in a long super pellet phase from holding up every iteration.

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
        ahead_steps: int | None = None,
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
        # Room for the steps of the iteration being taken plus `ahead_steps` more
        ahead_steps = 2 * max_steps if ahead_steps is None else ahead_steps
        num_slots = max_steps + ahead_steps

        base, extra = divmod(num_parallel_envs, num_actors)
        env_counts = [base + (i < extra) for i in range(num_actors)]
        num_params = sum(p.numel() for p in q_net.parameters())
        self._shms: list[shared_memory.SharedMemory] = []
        self._procs = []
        self._conns: list[Connection] = []
        self._obs_views: list[np.ndarray] = []
        self._registered: list[int] = []
        self._weights_lock = _CTX.Lock()
        # On a GPU, observations are copied on their own stream (so waiting for them doesn't
        # wait for the update), and weights are copied out in the background (see _publish).
        self._cuda = device.type == "cuda"
        if self._cuda:
            self._copy_stream = torch.cuda.Stream(device)
            self._params_host = torch.empty(num_params, pin_memory=True)
        self._params_copied = None  # event for the background copy into _params_host
        try:
            weights_shm = self._new_shm((_HEADER + num_params) * 8)
            self._shared = np.ndarray((_HEADER + num_params,), np.float64, buffer=weights_shm.buf)
            self._shared[0] = 0
            self._publish(epsilon)
            for count in env_counts:
                obs_shm = self._new_shm(num_slots * count * int(np.prod(OBS_SHAPE)) * 4)
                view = np.ndarray((num_slots, count, *OBS_SHAPE), np.float32, buffer=obs_shm.buf)
                self._obs_views.append(view)
                self._pin(view)
                parent, child = _CTX.Pipe()
                proc = _CTX.Process(
                    target=_actor,
                    args=(
                        child, count, num_slots, obs_shm.name, weights_shm.name,
                        self._weights_lock, num_params, model_name, threads_per_actor,
                        _fresh_seed(),
                    ),
                    daemon=True,
                )
                proc.start()
                self._procs.append(proc)
                self._conns.append(parent)
            # Each actor writes its starting observations to slot 0 before saying it's ready
            self._last_obs = []
            for conn, view in zip(self._conns, self._obs_views):
                self._receive(conn)
                self._last_obs.append(self._to_device(view[0]))
            self._sync()
            for conn in self._conns:
                conn.send(("free", 0))
        except BaseException:
            self.close()
            raise

    def _new_shm(self, size: int) -> shared_memory.SharedMemory:
        shm = shared_memory.SharedMemory(create=True, size=size)
        self._shms.append(shm)
        return shm

    def _pin(self, view: np.ndarray) -> None:
        """Registers shared memory with CUDA, making copies to the GPU several times faster."""
        if self.device.type != "cuda":
            return
        ptr = view.ctypes.data
        if torch.cuda.cudart().cudaHostRegister(ptr, view.nbytes, 0) == 0:  # cudaSuccess
            self._registered.append(ptr)

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        # Always copy: the slot is reused once handed back. On a GPU the copy may still be
        # running when this returns, so slots go back only after _sync().
        if not self._cuda:
            return torch.from_numpy(array).to(self.device, copy=True)
        with torch.cuda.stream(self._copy_stream):
            tensor = torch.from_numpy(array).to(self.device, copy=True, non_blocking=True)
        # Made on the copy stream, used on the main one: keep its memory until that's done too
        tensor.record_stream(torch.cuda.current_stream(self.device))
        return tensor

    def _sync(self) -> None:
        """Waits for the observation copies (not for the training work on the main stream)."""
        if self._cuda:
            self._copy_stream.synchronize()

    @staticmethod
    def _receive(conn: Connection):
        message = conn.recv()
        if message[0] == "error":
            raise RuntimeError(f"Actor process failed:\n{message[1]}")
        return message

    def _params_vector(self) -> torch.Tensor:
        # reshape, not view: parameters may be in channels-last layout
        return torch.cat([p.detach().reshape(-1) for p in self._q_net.parameters()])

    def _write_shared(self, epsilon: float, params: np.ndarray | None) -> None:
        with self._weights_lock:
            self._shared[1] = epsilon
            if params is not None:
                self._shared[_HEADER:] = params
            self._shared[0] += 1

    def _publish(self, epsilon: float) -> None:
        """
        Shares the exploration rate and the network's weights with the actors. On a GPU, the
        weights are copied out in the background and shared at the next call, so the CPU
        doesn't wait for the update in progress; actors then play with weights one update older.
        """
        with torch.no_grad():
            if not self._cuda:
                self._write_shared(epsilon, self._params_vector().numpy())
                return
            params = None
            if self._params_copied is not None:
                self._params_copied.synchronize()
                params = self._params_host.numpy()
            elif self._shared[0] == 0:  # first publish: wait for the starting weights
                params = self._params_vector().cpu().numpy()
            self._write_shared(epsilon, params)
            self._params_host.copy_(self._params_vector(), non_blocking=True)
            self._params_copied = torch.cuda.Event()
            self._params_copied.record()

    def start_collection(self, steps: int, epsilon: float) -> None:
        """Publishes the current weights and exploration rate for the actors to play with."""
        if self._pending_steps:
            raise RuntimeError("finish_collection() was not called for the previous collection")
        if not 1 <= steps <= self._max_steps:
            raise ValueError(f"steps must be 1..{self._max_steps}, got {steps}")
        self._publish(epsilon)
        self._pending_steps = steps

    def finish_collection(self) -> None:
        """Takes the next `steps` experience steps from every actor into the buffer."""
        steps, self._pending_steps = self._pending_steps, 0
        taken = []
        for _ in range(steps):
            for i, (conn, view) in enumerate(zip(self._conns, self._obs_views)):
                _, slot, transitions = self._receive(conn)
                new_obs = self._to_device(view[slot])
                taken.append((conn, slot))
                last_obs = self._last_obs[i]
                for j, (action, reward, done, next_action_mask) in enumerate(transitions):
                    next_obs = None if done else new_obs[j]
                    self._buffer.append(
                        ReplayItem(last_obs[j], action, reward, next_obs, next_action_mask)
                    )
                self._last_obs[i] = new_obs
        # Hand the slots back once their copies are done
        self._sync()
        for conn, slot in taken:
            conn.send(("free", slot))

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
        for ptr in self._registered:
            torch.cuda.cudart().cudaHostUnregister(ptr)
        self._registered = []
        # Views into the shared memory must go before closing it
        self._obs_views = []
        self._shared = None
        for shm in self._shms:
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._shms = []
