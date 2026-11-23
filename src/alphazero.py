"""
Pieces of AlphaZero-style training (algorithms/train_alphazero.py) with Gumbel search
(pacbot_rs.gumbel_search; see pacbot_rs/src/gumbel.rs):

- NetworkEvaluator: the policy and value network, as the callable the search evaluates
  batches of leaves with.
- Targets: each game's moves, turned into training examples once their value target is known:
  the td_steps-step return (rewards in the network's units, discounted) bootstrapped from the
  search's value td_steps moves later, or the rewards up to the end of the game.
- TrainingStore: the newest training examples (observation, improved policy, legal moves,
  value target) on the training device, observations stored as bits like replay_storage.
- evaluate_games: fixed-seed games at each speed, played by the network alone (greedy policy)
  or with search, several games at once.
- search_config: the search a run trains or plays with, from its settings: Gumbel's rules (the
  default) or classic AlphaZero / MuZero PUCT ("search": "puct").
"""

from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from pacbot_rs import GumbelConfig, PacmanGym, gumbel_search
from replay_storage import ObsCodec
from utils import NUM_ACTIONS, OBS_SHAPE, full_game, make_gym, speed_configuration


# Search settings added after the first runs, with the values those runs used (Gumbel search)
SEARCH_DEFAULTS = {
    "search": "gumbel",  # or "puct": classic AlphaZero / MuZero search (pacbot_rs/src/gumbel.rs)
    "c_puct": 1.25,  # PUCT's exploration constant and its growth with visits (MuZero's c1, c2)
    "c_puct_base": 19652.0,
    "dirichlet_alpha": 10.0,  # root noise concentration, divided among the legal moves
    "dirichlet_fraction": 0.25,  # share of the root prior that is noise, in training
    "puct_temperature": "muzero",  # training moves ~ visits^(1/T); "muzero": 1, 0.5, 0.25 by progress
}
SEARCH_RULES = ("gumbel", "puct")


def puct_temperature(setting, progress: float) -> float:
    """
    The temperature of PUCT's training moves: a number, or "muzero" for MuZero's schedule
    (1 for the first half of training, 0.5 for the next quarter, 0.25 after).
    """
    if setting == "muzero":
        return 1.0 if progress < 0.5 else 0.5 if progress < 0.75 else 0.25
    return float(setting)


def search_config(cfg, num_simulations: int, explore: bool, threads: int = 1, progress: float = 0.0) -> GumbelConfig:
    """
    The search an AlphaZero run trains with (explore: Gumbel or Dirichlet noise and, for PUCT,
    sampled moves) or plays with (no exploration), from its settings (a checkpoint's config or
    wandb.config). Runs from before the PUCT option get Gumbel search, as they trained with.
    """
    def get(key):
        try:
            value = cfg[key]
        except KeyError:
            return SEARCH_DEFAULTS[key]
        return SEARCH_DEFAULTS[key] if value is None else value

    rule = get("search")
    if rule not in SEARCH_RULES:
        raise ValueError(f"search must be one of {SEARCH_RULES}, got {rule!r}")
    return GumbelConfig(num_simulations=num_simulations, discount=cfg["discount"], reward_scale=cfg["reward_scale"],
                        c_visit=cfg["c_visit"], c_scale=cfg["c_scale"], gumbel_scale=1.0 if explore else 0.0,
                        threads=threads, puct=rule == "puct", c_puct=float(get("c_puct")),
                        c_puct_base=float(get("c_puct_base")), dirichlet_alpha=float(get("dirichlet_alpha")),
                        dirichlet_fraction=float(get("dirichlet_fraction")),
                        temperature=puct_temperature(get("puct_temperature"), progress))


class NetworkEvaluator:
    """
    (obs float32 (B, 17, 28, 31), legal bool (B, 5)) -> (values (B,), logits (B, 5)) as NumPy.

    With `max_batch` on CUDA, every call runs one CUDA graph of the network on max_batch
    observations (the rest of the batch padding): a search makes dozens of small calls per move,
    and replaying a graph skips launching each kernel (~2.5 ms per call instead of ~4-6 ms for
    256 states on a busy GPU). The graph reads the network's parameters where they are, so it
    follows the training updates made in place (optimizer steps, load_state_dict); the network
    must not have batch statistics (batch norm) or other state that differs in training mode.
    """

    def __init__(self, net: torch.nn.Module, device: torch.device, max_batch: int | None = None) -> None:
        self.net = net
        self.device = device
        self.calls = 0
        self.states = 0
        self._graph = None
        if max_batch is not None and device.type == "cuda":
            self._capture(max_batch)

    @torch.inference_mode()
    def _capture(self, max_batch: int) -> None:
        self._input = torch.zeros((max_batch, *OBS_SHAPE), device=self.device).contiguous(
            memory_format=torch.channels_last)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up (and cuDNN's choice of algorithms) before capturing
                self.net(self._input)
        torch.cuda.current_stream(self.device).wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._logits, self._value = self.net(self._input)

    @torch.inference_mode()
    def __call__(self, obs: np.ndarray, legal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        n = obs.shape[0]
        self.states += n
        if self._graph is not None and n <= self._input.shape[0]:
            self._input[:n].copy_(torch.from_numpy(obs))
            self._graph.replay()
            logits, value = self._logits[:n], self._value[:n]
        else:
            logits, value = self.net(torch.from_numpy(obs).to(self.device, non_blocking=True))
        return value.float().cpu().numpy(), logits.float().cpu().numpy()


class QNetworkEvaluator:
    """
    A DQN Q network as the search's evaluator, so DQN agents can play with the same search:
    the value of a state is its best legal Q value, and the policy's logits are the legal Q
    values divided by `temperature`. DQN training uses the same reward scale and discount as
    the search (0.02 and 0.99), so its Q values are already in the search's units.
    """

    def __init__(self, q_net: torch.nn.Module, device: torch.device, temperature: float) -> None:
        self.q_net = q_net
        self.device = device
        self.temperature = temperature

    @torch.inference_mode()
    def __call__(self, obs: np.ndarray, legal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_net(torch.from_numpy(obs).to(self.device)).float()
        q = q.masked_fill(~torch.from_numpy(legal).to(self.device), -torch.inf)
        value = q.amax(dim=1)
        logits = (q / self.temperature).masked_fill(q == -torch.inf, 0.0)
        return value.cpu().numpy(), logits.cpu().numpy()


class Example(NamedTuple):
    obs: np.ndarray     # (17, 28, 31) float32
    legal: np.ndarray   # (5,) bool
    policy: np.ndarray  # (5,) float32, the search's improved policy
    value: float        # the value target, in the network's units


class Targets:
    """
    Each game's recent moves, until their td_steps-step value targets are known. add() takes a
    move (the state's observation, legal moves, the search's policy and value, and the scaled
    reward the move got) and returns the examples it completes; end() completes a game's
    remaining moves when it ends (no bootstrap after the end).
    """

    def __init__(self, num_games: int, td_steps: int, discount: float) -> None:
        if td_steps < 1:
            raise ValueError(f"td_steps must be at least 1, got {td_steps}")
        self.td_steps = td_steps
        self.discount = discount
        self._pending: list[deque] = [deque() for _ in range(num_games)]

    def _complete(self, moves: list, bootstrap: float) -> Example:
        """The example for moves[0]: its rewards, then `bootstrap` discounted past them all."""
        ret = bootstrap
        for move in reversed(moves):
            ret = move[4] + self.discount * ret
        obs, legal, policy, _, _ = moves[0]
        return Example(obs, legal, policy, float(ret))

    def add(self, game: int, obs: np.ndarray, legal: np.ndarray, policy: np.ndarray,
            search_value: float, reward: float) -> list[Example]:
        pending = self._pending[game]
        pending.append((obs, legal, policy, search_value, reward))
        done = []
        while len(pending) > self.td_steps:
            # The move td_steps after the oldest has its search value: the oldest is complete
            moves = [pending[i] for i in range(self.td_steps)]
            done.append(self._complete(moves, pending[self.td_steps][3]))
            pending.popleft()
        return done

    def end(self, game: int) -> list[Example]:
        pending = list(self._pending[game])
        self._pending[game].clear()
        return [self._complete(pending[i:], 0.0) for i in range(len(pending))]


class Batch(NamedTuple):
    obs: torch.Tensor     # (B, 17, 28, 31) float32
    legal: torch.Tensor   # (B, 5) bool
    policy: torch.Tensor  # (B, 5) float32
    value: torch.Tensor   # (B,) float32


class TrainingStore:
    """The newest `capacity` examples on `device`; sample() draws uniformly with replacement."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.codec = ObsCodec(device)
        self._plain_obs = device.type == "mps"
        if self._plain_obs:
            self._obs_stores = (torch.zeros((capacity, *OBS_SHAPE), dtype=torch.float32, device=device),)
        else:
            self._obs_stores = ObsCodec.stores(capacity, device)
        self._legal = torch.zeros((capacity, NUM_ACTIONS), dtype=torch.bool, device=device)
        self._policy = torch.zeros((capacity, NUM_ACTIONS), dtype=torch.float32, device=device)
        self._value = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.total = 0

    def __len__(self) -> int:
        return min(self.total, self.capacity)

    def add(self, examples: list[Example]) -> None:
        if not examples:
            return
        for start in range(0, len(examples), self.capacity):
            self._add(examples[start:start + self.capacity])

    def _add(self, examples: list[Example]) -> None:
        n = len(examples)
        obs = torch.from_numpy(np.stack([e.obs for e in examples])).to(self.device)
        parts = (obs,) if self._plain_obs else self.codec.encode(obs)
        columns = list(zip(self._obs_stores, parts)) + [
            (self._legal, torch.from_numpy(np.stack([e.legal for e in examples])).to(self.device)),
            (self._policy, torch.from_numpy(np.stack([e.policy for e in examples])).to(self.device)),
            (self._value, torch.tensor([e.value for e in examples], dtype=torch.float32, device=self.device)),
        ]
        start = self.total % self.capacity
        first = min(n, self.capacity - start)
        for store, part in columns:
            store[start:start + first] = part[:first]
            if first < n:
                store[:n - first] = part[first:]
        self.total += n

    def sample(self, batch_size: int) -> Batch:
        if len(self) == 0:
            raise ValueError("the store is empty")
        idx = torch.randint(len(self), (batch_size,), device=self.device)
        if self._plain_obs:
            obs = self._obs_stores[0][idx]
        else:
            obs = self.codec.decode(*(store[idx] for store in self._obs_stores))
        return Batch(obs, self._legal[idx], self._policy[idx], self._value[idx])


def losses(net: torch.nn.Module, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(policy loss: cross entropy with the improved policy over the legal moves, value loss:
    mean squared error, policy entropy)."""
    logits, value = net(batch.obs)
    log_policy = F.log_softmax(logits.masked_fill(~batch.legal, -torch.inf), dim=1).masked_fill(~batch.legal, 0.0)
    policy_loss = -(batch.policy * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(value, batch.value)
    entropy = -(log_policy.exp() * log_policy).sum(dim=1).mean()
    return policy_loss, value_loss, entropy


@torch.inference_mode()
def evaluate_games(net: torch.nn.Module, device: torch.device, env_version: int, obs_config,
                   speeds, seeds, search: GumbelConfig | None = None, max_steps: int | None = None,
                   search_seed: int = 0, evaluator=None) -> list[dict]:
    """
    Plays one game per (speed, seed) from the normal start, all at once, until each loses a
    life, clears the board or reaches max_steps (10,000), or with utils.full_game() until it
    loses its last life (or reaches max_steps, 40,000): greedily by the network's policy, or
    with `search` (use gumbel_scale=0 for no exploration). The search evaluates states with
    `evaluator` if given (e.g. QNetworkEvaluator), else with the network. Returns one dict per
    game: speed, seed, score, steps, cleared, pellets_left, boards (boards cleared) and lives
    (lives left).
    """
    full = full_game()
    if max_steps is None:
        max_steps = 40_000 if full else 10_000
    net.eval()
    games = []
    for speed in speeds:
        for seed in seeds:
            env = make_gym(speed_configuration(speed), env_version)
            env.seed = seed
            games.append({"speed": speed, "seed": seed, "env": env, "steps": 0, "done": False})
    if search is not None and evaluator is None:
        evaluator = NetworkEvaluator(net, device, max_batch=len(games))
    step = 0
    while True:
        live = [g for g in games if not g["done"]]
        if not live:
            break
        envs: list[PacmanGym] = [g["env"] for g in live]
        if search is None:
            obs = np.stack([e.obs_numpy(obs_config) for e in envs])
            legal = torch.tensor([e.action_mask() for e in envs], device=device)
            logits, _ = net(torch.from_numpy(obs).to(device))
            actions = logits.masked_fill(~legal, -torch.inf).argmax(dim=1).tolist()
        else:
            actions = gumbel_search(envs, evaluator, search, obs_config, search_seed + step)[0].tolist()
        step += 1
        for g, action in zip(live, actions):
            _, done = g["env"].step(int(action))
            g["steps"] += 1
            if done or g["steps"] >= max_steps:
                g["done"] = True
    results = []
    for g in games:
        env = g["env"]
        boards = env.level() - 1
        # A full game's "cleared": at least one board cleared
        cleared = boards > 0 if full else env.is_done() and env.lives() == 3
        results.append({"speed": g["speed"], "seed": g["seed"], "score": env.score(), "steps": g["steps"],
                        "cleared": cleared, "pellets_left": 0 if cleared and not full else env.remaining_pellets(),
                        "boards": boards, "lives": env.lives()})
    return results
