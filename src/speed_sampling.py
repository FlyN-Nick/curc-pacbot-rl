"""
SpeedSampler: the speed (game ticks per Pacman move) of each training game.

With uniform games, the slow speeds get most of the training data: a game at 4 ticks per
move lasts several times longer than one at 14 (in round-7 evaluation games, ~1500 moves
against ~170), so speeds 4-8 gave ~70% of the moves and 12-14 ~9%. "equal_moves" draws each
speed with weight 1 / (its mean moves per game), from a running average of the games actually
played, so every speed gets about the same number of training moves as the policy and its
game lengths change.
"""

from __future__ import annotations

import random

import utils


class SpeedSampler:
    def __init__(self, mode: str | None = None, speeds: tuple[int, ...] | None = None,
                 rng: random.Random | None = None, prior_moves: float = 300.0, smoothing: float = 0.02,
                 min_moves: float = 10.0) -> None:
        """
        mode and speeds default to utils.speed_sampling() and utils.training_speeds().
        prior_moves: the assumed game length before any game at a speed has ended (the same for
        every speed, so sampling starts uniform). smoothing: how much each finished game moves
        its speed's average. min_moves: a floor on the averages, so a speed whose games end at
        once can't take every draw.
        """
        self.mode = mode or utils.speed_sampling()
        if self.mode not in utils.SPEED_SAMPLING_MODES:
            raise ValueError(f"speed sampling must be one of {utils.SPEED_SAMPLING_MODES}, got {self.mode!r}")
        self.speeds = tuple(speeds or utils.training_speeds())
        self.rng = rng or random.Random()
        self.smoothing = smoothing
        self.min_moves = min_moves
        self.mean_moves = {s: float(prior_moves) for s in self.speeds}

    def weights(self) -> dict[int, float]:
        """Each speed's chance of being drawn (summing to 1)."""
        if self.mode == "uniform":
            raw = {s: 1.0 for s in self.speeds}
        else:
            raw = {s: 1.0 / max(self.min_moves, m) for s, m in self.mean_moves.items()}
        total = sum(raw.values())
        return {s: w / total for s, w in raw.items()}

    def draw(self) -> int:
        weights = self.weights()
        return self.rng.choices(self.speeds, weights=[weights[s] for s in self.speeds])[0]

    def record(self, speed: int, moves: int) -> None:
        """A game at `speed` ended after `moves` moves."""
        if speed in self.mean_moves:
            self.mean_moves[speed] += self.smoothing * (moves - self.mean_moves[speed])
