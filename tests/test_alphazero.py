"""
AlphaZero-style training: the Gumbel search (pacbot_rs.gumbel_search) on the real game, value
targets, the training store, the losses and evaluation (alphazero.py).

Run from src/: python -m unittest discover -s ../tests -p test_alphazero.py
"""

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
import utils  # noqa: E402
from alphazero import Example, NetworkEvaluator, Targets, TrainingStore, evaluate_games, losses  # noqa: E402
from pacbot_rs import GumbelConfig, gumbel_search  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402

OBS_CONFIG = utils.SINGLE_MODEL_REPLAY_BUFFER_OBS_GYM_CONFIGURATION


def uniform(obs, legal):
    """Value 0 and a uniform policy: the search sees only the game's rewards."""
    return np.zeros(obs.shape[0], np.float32), np.zeros((obs.shape[0], NUM_ACTIONS), np.float32)


def random_states(n, seed=0, max_moves=300):
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        env = utils.make_gym(utils.speed_configuration(rng.randint(4, 14)), utils.ENV_VERSION)
        for _ in range(rng.randint(0, max_moves)):
            mask = env.action_mask()
            _, done = env.step(rng.choice([a for a in range(NUM_ACTIONS) if mask[a]]))
            if done:
                break
        if not env.is_done():
            states.append(env)
    return states


def deadly_states(n, seed=0):
    """States where some legal move always loses a life right away and another never does."""
    rng = random.Random(seed)
    found = []
    attempts = 0
    while len(found) < n and attempts < 20000:
        attempts += 1
        env = random_states(1, seed=rng.randrange(10**9), max_moves=400)[0]
        deadly, safe = [], []
        for action in range(NUM_ACTIONS):
            if not env.action_mask()[action]:
                continue
            outcomes = []
            for _ in range(6):
                copy = env.copy()
                outcomes.append(copy.step(action)[1])
            if all(outcomes):
                deadly.append(action)
            elif not any(outcomes):
                safe.append(action)
        if deadly and safe:
            found.append((env, deadly))
    return found


def lookahead_value(env, action, depth, samples=4, reward_scale=0.02, discount=0.99):
    """The mean over random outcomes of the action's reward plus the best lookahead after it,
    `depth` moves deep in all, with value 0 at the end (in the network's units)."""
    total = 0.0
    for _ in range(samples):
        copy = env.copy()
        reward, done = copy.step(action)
        value = reward * reward_scale
        if not done and depth > 1:
            value += discount * max(lookahead_value(copy, a, depth - 1, samples=1)
                                    for a in range(NUM_ACTIONS) if copy.action_mask()[a])
        total += value
    return total / samples


class GumbelSearchTests(unittest.TestCase):
    def test_outputs_are_well_formed_and_games_unchanged(self):
        envs = random_states(12)
        before = [e.game_state_bytes() for e in envs]
        config = GumbelConfig(num_simulations=16)
        actions, policy, value, network_value, q = gumbel_search(envs, uniform, config, OBS_CONFIG, 7)
        self.assertEqual([e.game_state_bytes() for e in envs], before)
        self.assertEqual((actions.shape, policy.shape, value.shape, network_value.shape, q.shape),
                         ((12,), (12, 5), (12,), (12,), (12, 5)))
        for i, env in enumerate(envs):
            legal = np.array(env.action_mask())
            self.assertTrue(legal[actions[i]])
            self.assertAlmostEqual(float(policy[i].sum()), 1.0, places=5)
            self.assertTrue((policy[i][~legal] == 0).all())
            self.assertTrue((q[i][~legal] == 0).all())
            self.assertEqual(network_value[i], 0.0)
        self.assertTrue(np.isfinite(value).all())

    def test_same_seed_same_search(self):
        envs = random_states(6, seed=3)
        config = GumbelConfig(num_simulations=8)
        a = gumbel_search(envs, uniform, config, OBS_CONFIG, 11)
        b = gumbel_search(envs, uniform, config, OBS_CONFIG, 11)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_batches_leaves_into_one_call_per_simulation(self):
        calls = []

        def counting(obs, legal):
            calls.append(obs.shape[0])
            return uniform(obs, legal)

        envs = random_states(10, seed=4)
        gumbel_search(envs, counting, GumbelConfig(num_simulations=5), OBS_CONFIG, 0)
        # One call for the roots, then at most one per simulation, each with at most 10 states
        self.assertEqual(calls[0], 10)
        self.assertLessEqual(len(calls), 6)
        self.assertTrue(all(n <= 10 for n in calls))

    def test_finds_the_best_move_when_one_loses_a_life(self):
        # Against exhaustive 3-move lookahead through the real game (value 0 after): the
        # search's move is (nearly) the best, and a move that loses a life at once is avoided
        # whenever another is clearly better
        cases = deadly_states(8, seed=5)
        self.assertGreaterEqual(len(cases), 4, "not enough deadly states found")
        envs = [env for env, _ in cases]
        config = GumbelConfig(num_simulations=32, gumbel_scale=0.0)
        actions, policy, _, _, _ = gumbel_search(envs, uniform, config, OBS_CONFIG, 1)
        avoided = 0
        for i, (env, deadly) in enumerate(cases):
            true = {a: lookahead_value(env, a, depth=3) for a in range(NUM_ACTIONS) if env.action_mask()[a]}
            best = max(true.values())
            self.assertGreaterEqual(true[int(actions[i])], best - 0.3, (true, int(actions[i])))
            if max(v for a, v in true.items() if a not in deadly) > max(true[a] for a in deadly) + 1.0:
                self.assertNotIn(int(actions[i]), deadly)
                for a in deadly:
                    self.assertLess(policy[i][a], 1.0 / len(true))
                avoided += 1
        self.assertGreaterEqual(avoided, 2, "too few cases where avoiding death is clearly better")

    def test_rejects_ended_games(self):
        env = deadly_states(1, seed=6)[0]
        game, deadly = env
        game = game.copy()
        game.step(deadly[0])
        with self.assertRaises(ValueError):
            gumbel_search([game], uniform, GumbelConfig(num_simulations=2), OBS_CONFIG, 0)


class TargetTests(unittest.TestCase):
    def test_td_targets_bootstrap_from_the_search_value(self):
        targets = Targets(num_games=2, td_steps=2, discount=0.5)
        obs, legal, policy = np.zeros(OBS_SHAPE, np.float32), np.ones(5, bool), np.full(5, 0.2, np.float32)
        out = []
        # Moves with search values 10, 20, 30, 40 and rewards 1, 2, 3, 4
        for v, r in zip((10, 20, 30, 40), (1, 2, 3, 4)):
            out += targets.add(0, obs, legal, policy, float(v), float(r))
        # Move 0: 1 + 0.5 * 2 + 0.25 * 30; move 1: 2 + 0.5 * 3 + 0.25 * 40
        self.assertEqual([e.value for e in out], [1 + 1 + 7.5, 2 + 1.5 + 10])
        # At the end: move 2: 3 + 0.5 * 4; move 3: 4
        self.assertEqual([e.value for e in targets.end(0)], [3 + 2, 4])
        self.assertEqual(targets.end(1), [])


class StoreAndLossTests(unittest.TestCase):
    def examples(self, n):
        envs = random_states(n, seed=8)
        rng = np.random.default_rng(0)
        out = []
        for env in envs:
            legal = np.array(env.action_mask())
            p = rng.random(5).astype(np.float32) * legal
            out.append(Example(env.obs_numpy(OBS_CONFIG), legal, p / p.sum(), float(rng.normal())))
        return out

    def test_store_keeps_the_newest_examples_exactly(self):
        examples = self.examples(40)
        store = TrainingStore(32, torch.device("cpu"))
        store.add(examples[:25])
        store.add(examples[25:])
        self.assertEqual((store.total, len(store)), (40, 32))
        batch = store.sample(200)
        newest = {float(np.float32(e.value)) for e in examples[8:]}
        self.assertTrue(set(batch.value.tolist()) <= newest)
        for row in range(5):
            example = next(e for e in examples[8:] if np.float32(e.value) == batch.value[row].item())
            self.assertTrue(np.array_equal(batch.obs[row].numpy(), example.obs))
            self.assertTrue(np.array_equal(batch.policy[row].numpy(), example.policy))
        self.assertEqual(store.codec.mismatches(), 0)

    def test_losses_fall_when_training(self):
        store = TrainingStore(64, torch.device("cpu"))
        store.add(self.examples(64))
        torch.manual_seed(0)
        net = models.PVNetV2(OBS_SHAPE, NUM_ACTIONS)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        batch = store.sample(64)
        first = None
        for _ in range(60):
            policy_loss, value_loss, _ = losses(net, batch)
            first = first or (policy_loss.item(), value_loss.item())
            opt.zero_grad()
            (policy_loss + value_loss).backward()
            opt.step()
        self.assertLess(policy_loss.item(), first[0])
        self.assertLess(value_loss.item(), 0.5 * first[1])


class EvaluationTests(unittest.TestCase):
    def test_greedy_and_search_games(self):
        torch.manual_seed(0)
        net = models.PVNetV2(OBS_SHAPE, NUM_ACTIONS)
        device = torch.device("cpu")
        greedy = evaluate_games(net, device, utils.ENV_VERSION, OBS_CONFIG, [8, 12], [1, 2], max_steps=300)
        again = evaluate_games(net, device, utils.ENV_VERSION, OBS_CONFIG, [8, 12], [1, 2], max_steps=300)
        self.assertEqual(greedy, again)
        self.assertEqual([(g["speed"], g["seed"]) for g in greedy], [(8, 1), (8, 2), (12, 1), (12, 2)])
        searched = evaluate_games(net, device, utils.ENV_VERSION, OBS_CONFIG, [8], [1], max_steps=60,
                                  search=GumbelConfig(num_simulations=4, gumbel_scale=0.0))
        self.assertEqual(len(searched), 1)
        self.assertGreater(searched[0]["steps"], 0)

    def test_network_evaluator_shapes(self):
        net = models.PVNetV2(OBS_SHAPE, NUM_ACTIONS)
        ev = NetworkEvaluator(net, torch.device("cpu"))
        obs = np.stack([e.obs_numpy(OBS_CONFIG) for e in random_states(3, seed=9)])
        values, logits = ev(obs, np.ones((3, 5), bool))
        self.assertEqual((values.shape, logits.shape, values.dtype, logits.dtype),
                         ((3,), (3, 5), np.float32, np.float32))


if __name__ == "__main__":
    unittest.main()
