"""
PPO training (algorithms/train_ppo.py): generalized advantage estimates, the clipped losses, and
short training runs that save checkpoints the evaluation scripts and the bot read like
AlphaZero's, and resume.

Run from src/: python -m unittest discover ../tests
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import models
import utils
from algorithms.train_ppo import gae, ppo_losses

SRC = Path(__file__).resolve().parents[1] / "src"


def brute_force_gae(rewards, values, dones, last_value, discount, lam):
    """GAE from its definition: the lambda-weighted sum of k-step advantages, per game."""
    steps, games = len(rewards), len(rewards[0])
    advantages = [[0.0] * games for _ in range(steps)]
    for g in range(games):
        for t in range(steps):
            total, weight = 0.0, 1.0
            for k in range(t, steps):
                next_value = 0.0 if dones[k][g] else (values[k + 1][g] if k + 1 < steps else last_value[g])
                delta = rewards[k][g] + discount * next_value - values[k][g]
                total += weight * delta
                if dones[k][g]:
                    break
                weight *= discount * lam
            advantages[t][g] = total
    return advantages


class GaeTests(unittest.TestCase):
    def test_matches_the_definition_across_game_ends(self):
        torch.manual_seed(0)
        rewards = torch.randn(7, 3)
        values = torch.randn(7, 3)
        dones = torch.zeros(7, 3)
        dones[2, 0] = dones[5, 1] = dones[6, 2] = 1.0
        last_value = torch.randn(3)
        advantages, returns = gae(rewards, values, dones, last_value, 0.99, 0.95)
        expected = brute_force_gae(rewards.tolist(), values.tolist(), dones.tolist(), last_value.tolist(), 0.99, 0.95)
        torch.testing.assert_close(advantages, torch.tensor(expected), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(returns, advantages + values)

    def test_lambda_one_gives_discounted_returns(self):
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.zeros(3, 1)
        dones = torch.tensor([[0.0], [0.0], [1.0]])
        _, returns = gae(rewards, values, dones, torch.tensor([100.0]), 0.5, 1.0)
        # The game ends with the last move, so the value after it isn't counted
        torch.testing.assert_close(returns.squeeze(1), torch.tensor([1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.0]))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = models.PVNetV2(utils.OBS_SHAPE, utils.NUM_ACTIONS)
        with torch.no_grad():  # a non-uniform policy
            self.net.policy_head.weight.normal_(0, 0.1)
        self.obs = torch.rand(6, *utils.OBS_SHAPE)
        self.legal = torch.tensor([[True, True, False, True, False]] * 6)
        self.actions = torch.tensor([0, 1, 3, 0, 1, 3])
        with torch.no_grad():
            logits, self.value = self.net(self.obs)
            log_policy = torch.log_softmax(logits.masked_fill(~self.legal, -torch.inf), dim=1)
            self.log_prob = log_policy.gather(1, self.actions[:, None]).squeeze(1)

    def losses(self, advantages, old_log_prob=None, returns=None, value_clip=0.0):
        return ppo_losses(self.net, self.obs, self.legal, self.actions,
                          self.log_prob if old_log_prob is None else old_log_prob, self.value, advantages,
                          self.value if returns is None else returns, 0.2, value_clip)

    def test_on_policy_the_loss_is_the_advantage_and_nothing_is_clipped(self):
        advantages = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 0.0])
        out = self.losses(advantages)
        self.assertAlmostEqual(out["policy_loss"].item(), -advantages.mean().item(), places=5)
        self.assertAlmostEqual(out["approx_kl"].item(), 0.0, places=6)
        self.assertEqual(out["clip_fraction"].item(), 0.0)
        self.assertAlmostEqual(out["value_loss"].item(), 0.0, places=6)
        self.assertGreater(out["entropy"].item(), 0.0)
        self.assertLessEqual(out["entropy"].item(), torch.log(torch.tensor(3.0)).item() + 1e-5)  # 3 legal moves

    def test_the_ratio_is_clipped_where_it_would_go_too_far(self):
        # Old probabilities half the current ones: ratio 2, clipped to 1.2 for positive advantages
        advantages = torch.ones(6)
        out = self.losses(advantages, old_log_prob=self.log_prob - torch.log(torch.tensor(2.0)))
        self.assertAlmostEqual(out["policy_loss"].item(), -1.2, places=4)
        self.assertEqual(out["clip_fraction"].item(), 1.0)
        out["policy_loss"].backward()
        grads = [p.grad for p in self.net.policy_head.parameters()]
        self.assertTrue(all(g is None or torch.count_nonzero(g) == 0 for g in grads))

    def test_illegal_moves_get_no_probability(self):
        out = self.losses(torch.zeros(6))
        self.assertTrue(torch.isfinite(out["entropy"]))
        logits, _ = self.net(self.obs)
        probs = torch.softmax(logits.masked_fill(~self.legal, -torch.inf), dim=1)
        self.assertTrue(torch.all(probs[~self.legal] == 0))

    def test_value_clipping_keeps_the_larger_error(self):
        returns = self.value + 1.0
        unclipped = self.losses(torch.zeros(6), returns=returns)["value_loss"].item()
        clipped = self.losses(torch.zeros(6), returns=returns, value_clip=0.1)["value_loss"].item()
        self.assertAlmostEqual(unclipped, 0.5, places=4)
        self.assertGreaterEqual(clipped, unclipped - 1e-6)


class TrainingRunTests(unittest.TestCase):
    def run_ppo(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "algorithms.train_ppo", "--no-wandb", "--device", "cpu",
             "--num_games", "8", "--rollout_steps", "8", "--minibatches", "2", "--max_speed", "8",
             "--eval_every", "128", "--eval_seeds", "1", "--search_eval_every", "128",
             "--search_eval_seeds", "1", "--search_eval_simulations", "2", "--checkpoint_every", "64", *args],
            cwd=SRC, capture_output=True, text=True, timeout=600)

    def test_trains_saves_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.run_ppo("--checkpoint-dir", tmp, "--num_moves", "64")
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            latest = Path(tmp) / "model-latest.ckpt.pt"
            ck = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual((ck["rollout"], ck["moves"], ck["algorithm"]), (1, 64, "ppo"))
            # What eval_az_seeds.py and the bot client read from an AlphaZero checkpoint
            for key in ("discount", "reward_scale", "c_visit", "c_scale", "model"):
                self.assertIn(key, ck["config"])
            self.assertTrue(ck["single_model"])
            net = getattr(models, ck["config"]["model"])(utils.OBS_SHAPE, utils.NUM_ACTIONS)
            net.load_state_dict(ck["state_dict"])
            self.assertIsInstance(net, models.PVNetV2)

            resumed = self.run_ppo("--checkpoint-dir", tmp, "--num_moves", "128", "--resume", str(latest))
            self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:])
            ck = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual((ck["rollout"], ck["moves"]), (2, 128))
            self.assertTrue((Path(tmp) / "model-best.ckpt.pt").exists())
            self.assertTrue((Path(tmp) / "model-best-search.ckpt.pt").exists())


if __name__ == "__main__":
    unittest.main()
