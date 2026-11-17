"""
Search at decision time with a trained Q network (single-model checkpoints).

Before each move, every sequence of legal moves `depth` moves deep is played out in copies of
the game (PacmanGym.copy), and each end state is scored by the network's best Q value. A
sequence's value is its discounted rewards plus the discounted score of where it ends (zero if
the game ended); the first move of the best sequence is played. Depth 0 is the plain greedy
policy. Random choices in the copies (frightened ghosts' moves) are drawn afresh, so the search
doesn't know how the real game's will turn out; it sees one possible outcome per sequence.

Rewards and Q values are compared in the network's units: game rewards times reward_scale,
discounted by discount_factor per move, as in training.
"""

from dataclasses import dataclass

import numpy as np
import torch

from pacbot_rs import PacmanGym, PacmanGymConfiguration

# Observation used when one network plays the whole game (utils.SUPER_PELLET_MODE_CONFIGUARTION)
FULL_OBS = PacmanGymConfiguration({})


@dataclass
class _Node:
    """A searched state: its children as (action, scaled reward, child or None if done)."""
    children: list
    leaf: int = -1  # index into the leaf batch, for states at the search depth


class LookaheadPolicy:
    def __init__(self, q_net: torch.nn.Module, depth: int, reward_scale: float, discount: float,
                 device: torch.device = torch.device("cpu"),
                 obs_config: PacmanGymConfiguration = FULL_OBS) -> None:
        if depth < 0:
            raise ValueError(f"depth must be at least 0, got {depth}")
        self.q_net = q_net.eval()
        self.depth = depth
        self.reward_scale = reward_scale
        self.discount = discount
        self.device = device
        self.obs_config = obs_config  # the observations the network was trained on
        self.states_searched = 0  # end states scored so far, for measuring cost

    @staticmethod
    def legal_actions(env: PacmanGym) -> list[int]:
        return [a for a, ok in enumerate(env.action_mask()) if ok]

    @torch.no_grad()
    def _best_q(self, envs: list[PacmanGym]) -> np.ndarray:
        """The best legal action's Q value in each state."""
        obs = torch.from_numpy(np.stack([e.obs_numpy(self.obs_config) for e in envs])).to(self.device)
        masks = torch.tensor([e.action_mask() for e in envs], device=self.device)
        q = self.q_net(obs).masked_fill(~masks, -torch.inf)
        return q.amax(dim=1).float().cpu().numpy()

    def action_values(self, env: PacmanGym) -> dict[int, float]:
        """The searched value of each legal first move, in the network's units."""
        if self.depth == 0:
            with torch.no_grad():
                obs = torch.from_numpy(env.obs_numpy(self.obs_config)).to(self.device).unsqueeze(0)
                q = self.q_net(obs)[0].float().cpu().numpy()
            return {a: float(q[a]) for a in self.legal_actions(env)}

        leaves: list[PacmanGym] = []

        def expand(state: PacmanGym, depth: int) -> _Node:
            if depth == 0:
                leaves.append(state)
                return _Node([], leaf=len(leaves) - 1)
            children = []
            for action in self.legal_actions(state):
                child = state.copy()
                reward, done = child.step(action)
                children.append((action, reward * self.reward_scale,
                                 None if done else expand(child, depth - 1)))
            return _Node(children)

        root = expand(env, self.depth)
        leaf_values = self._best_q(leaves) if leaves else np.zeros(0)
        self.states_searched += len(leaves)

        def value(node: _Node) -> float:
            if node.leaf >= 0:
                return float(leaf_values[node.leaf])
            return max(r + self.discount * (0.0 if c is None else value(c)) for _, r, c in node.children)

        return {a: r + self.discount * (0.0 if c is None else value(c)) for a, r, c in root.children}

    def __call__(self, env: PacmanGym) -> int:
        values = self.action_values(env)
        return max(values, key=values.get)


def play_game(env: PacmanGym, policy: LookaheadPolicy, max_steps: int = 100_000) -> dict:
    """Plays until the game ends (training's evaluation games end at the first lost life)."""
    pellets_start = env.remaining_pellets()
    steps = 0
    done = False
    while not done and steps < max_steps:
        _, done = env.step(policy(env))
        steps += 1
    cleared = done and env.lives() == 3
    return {
        "score": env.score(),
        "steps": steps,
        "cleared": cleared,
        "pellets_left": 0 if cleared else env.remaining_pellets(),
        "pellets_eaten": pellets_start - (0 if cleared else env.remaining_pellets()),
    }
