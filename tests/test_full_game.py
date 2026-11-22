"""
Full games (PacmanGym.full_game, utils.set_full_game, train_alphazero --full_game true): three
lives and board after board, as in competition. A lost life costs the death penalty and the game
goes on from the respawn; a cleared board earns the clear bonus and the next level starts; the
game ends with the last life. Off, the game ends at the first lost life or cleared board, as
before.

Run from src/: python -m unittest discover ../tests
"""

import os
import random
import unittest
from collections import deque

import numpy as np
import torch

import models
import utils
from alphazero import evaluate_games
from pacbot_rs import GumbelConfig, PacmanGym, PacmanGymConfiguration, gumbel_search
from utils import ENV_VERSION

OBS = PacmanGymConfiguration({})
STEP = {1: (1, 0), 2: (-1, 0), 3: (0, -1), 4: (0, 1)}  # action: (row, col) change
DEATH_PENALTY = -200
CLEAR_BONUS = 3_000


def greedy_action(env: PacmanGym, rng: random.Random, explore: float = 0.1) -> int:
    """Walks to the nearest pellet (breadth-first over the observation), with some random moves."""
    legal = [a for a, ok in enumerate(env.action_mask()) if ok]
    if rng.random() < explore:
        return rng.choice(legal)
    obs = env.obs_numpy(OBS)
    wall, pellet, pacman = (obs[c].T[::-1] for c in (0, 1, 3))  # [row, col]
    found = np.argwhere(pacman > 0)
    if not len(found):
        return rng.choice(legal)
    start = tuple(found[0])
    came_from = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell != start and pellet[cell] > 0:
            while came_from[cell][0] != start:
                cell = came_from[cell][0]
            action = came_from[cell][1]
            return action if action in legal else rng.choice(legal)
        for action, (dr, dc) in STEP.items():
            nxt = (cell[0] + dr, cell[1] + dc)
            if 0 <= nxt[0] < 31 and 0 <= nxt[1] < 28 and not wall[nxt] and nxt not in came_from:
                came_from[nxt] = (cell, action)
                queue.append(nxt)
    return rng.choice(legal)


def config(clearable: bool = False) -> PacmanGymConfiguration:
    # With most pellets removed, the greedy walk clears some boards
    return PacmanGymConfiguration({"random_start": clearable, "random_pellet_removal": clearable,
                                   "random_ticks": True, "random_ticks_per_step_min": 6,
                                   "random_ticks_per_step_max": 10})


def game(seed: int, full: bool, clearable: bool = False) -> PacmanGym:
    env = PacmanGym(config(clearable), seed=seed)
    env.full_game = full
    return env


def play_pair(seed: int, clearable: bool):
    """
    Plays a one-life game and a full game from the same seed with the same moves, until the
    one-life game ends. Returns both games, the full game's (reward, done) and the one-life
    game's (reward, done) for the last move.
    """
    rng = random.Random(seed)
    short, full = game(seed, False, clearable), game(seed, True, clearable)
    for _ in range(5_000):
        action = greedy_action(short, rng)
        short_step, full_step = short.step(action), full.step(action)
        if short_step[1]:
            return short, full, full_step, short_step
    return short, full, None, None


class FullGameTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("PACBOT_FULL_GAME")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("PACBOT_FULL_GAME", None)
        else:
            os.environ["PACBOT_FULL_GAME"] = self._saved

    def test_off_by_default(self):
        os.environ.pop("PACBOT_FULL_GAME", None)
        self.assertFalse(utils.full_game())
        self.assertFalse(utils.make_gym(config(), ENV_VERSION, seed=1).full_game)
        utils.set_full_game(True)
        self.assertTrue(utils.make_gym(config(), ENV_VERSION, seed=1).full_game)

    def test_a_lost_life_costs_the_penalty_and_the_game_goes_on(self):
        for seed in range(6):
            short, full, (reward, done), (short_reward, short_done) = play_pair(seed, False)
            with self.subTest(seed=seed):
                self.assertEqual(short.lives(), 2)  # the one-life game ended at a death
                self.assertEqual(full.lives(), 2)
                self.assertFalse(done)
                self.assertEqual(reward, short_reward)  # the same move and the same penalty
                self.assertLessEqual(reward, DEATH_PENALTY + 20)
                self.assertFalse(full.is_done())
                self.assertNotEqual(full.game_state_bytes()[3], 0)  # not paused (0 is paused)
                # Pacman respawns and the game plays on to the last life
                rng = random.Random(seed)
                moves = 0
                while not full.is_done() and moves < 20_000:
                    full.step(greedy_action(full, rng))
                    moves += 1
                self.assertEqual(full.lives(), 0)
                self.assertGreater(moves, 0)

    def test_a_cleared_board_earns_the_bonus_and_the_next_level_starts(self):
        cleared = 0
        for seed in range(200):
            short, full, full_step, short_step = play_pair(seed, True)
            if short_step is None or short.lives() != 3:
                continue  # it didn't clear the board
            cleared += 1
            reward, done = full_step
            with self.subTest(seed=seed):
                self.assertEqual(short.level(), 2)
                self.assertTrue(short_step[1])
                self.assertFalse(done)
                self.assertEqual(reward, short_step[0])
                self.assertGreaterEqual(reward, CLEAR_BONUS - 200)
                self.assertEqual(full.level(), 2)
                self.assertEqual(full.lives(), 3)
                self.assertGreater(full.remaining_pellets(), 200)  # a fresh board
            if cleared == 3:
                break
        self.assertEqual(cleared, 3)

    def test_the_search_plays_on_after_a_lost_life(self):
        _, full, _, _ = play_pair(0, False)
        self.assertEqual(full.lives(), 2)
        copy = full.copy()
        self.assertTrue(copy.full_game)
        net = models.PVNetV2(utils.OBS_SHAPE, utils.NUM_ACTIONS).eval()

        def evaluator(obs, legal):
            with torch.inference_mode():
                logits, value = net(torch.from_numpy(obs))
            return value.numpy(), logits.numpy()

        actions = gumbel_search([full], evaluator, GumbelConfig(num_simulations=8, gumbel_scale=0.0), OBS, 1)[0]
        self.assertTrue(full.action_mask()[int(actions[0])])

    def test_checkpoints_and_evaluations_record_full_games(self):
        self.assertFalse(utils.checkpoint_full_game({"config": {"model": "PVNetV2"}}))
        self.assertTrue(utils.checkpoint_full_game({"full_game": True}))
        self.assertTrue(utils.checkpoint_full_game({"config": {"full_game": True}}))
        self.assertTrue(utils.parse_bool("true"))
        self.assertFalse(utils.parse_bool("False"))
        with self.assertRaises(ValueError):
            utils.parse_bool("maybe")
        utils.set_full_game(True)
        torch.manual_seed(0)
        net = models.PVNetV2(utils.OBS_SHAPE, utils.NUM_ACTIONS)
        results = evaluate_games(net, torch.device("cpu"), ENV_VERSION, OBS, [8], [1, 2], max_steps=300)
        for r in results:
            self.assertIn("boards", r)
            self.assertIn("lives", r)
            self.assertEqual(r["boards"], 0)
            self.assertTrue(r["lives"] == 0 or r["steps"] == 300)


if __name__ == "__main__":
    unittest.main()
