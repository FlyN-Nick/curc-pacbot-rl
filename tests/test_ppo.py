"""
PPO training (algorithms/train_ppo.py): generalized advantage estimates, the clipped losses, and
short training runs that save checkpoints the evaluation scripts and the bot read like
AlphaZero's, and resume.

Run from src/: python -m unittest discover ../tests
"""

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

import models
import utils
from algorithms.train_ppo import chunked_backward, gae, minibatch_update, ppo_losses

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


class MinibatchFixture(unittest.TestCase):
    """A minibatch whose update clips some ratios and has nonzero value errors."""

    def setUp(self):
        torch.manual_seed(0)
        self.net = models.PVNetV2(utils.OBS_SHAPE, utils.NUM_ACTIONS).double()  # float64: rounding far below the tolerance
        with torch.no_grad():
            self.net.policy_head.weight.normal_(0, 0.1)
        n = 10
        self.obs = torch.rand(n, *utils.OBS_SHAPE, dtype=torch.float64)
        self.legal = torch.rand(n, utils.NUM_ACTIONS) < 0.7
        self.legal[:, 0] = True
        self.actions = torch.tensor([0] * n)
        with torch.no_grad():
            logits, value = self.net(self.obs)
            log_policy = torch.log_softmax(logits.masked_fill(~self.legal, -torch.inf), dim=1)
        # Off-policy, so some ratios are clipped; returns away from the values
        self.old_log_prob = log_policy.gather(1, self.actions[:, None]).squeeze(1) + torch.linspace(-0.5, 0.5, n, dtype=torch.float64)
        self.old_value = value
        self.returns = value + torch.linspace(-1, 2, n, dtype=torch.float64)
        adv = torch.randn(n, dtype=torch.float64)
        self.adv = (adv - adv.mean()) / (adv.std() + 1e-8)


class ChunkTests(MinibatchFixture):
    """chunked_backward gives the same gradients and statistics as the whole minibatch at once."""

    def whole(self, value_clip):
        self.net.zero_grad(set_to_none=True)
        out = ppo_losses(self.net, self.obs, self.legal, self.actions, self.old_log_prob, self.old_value,
                         self.adv, self.returns, 0.2, value_clip)
        (out["policy_loss"] + 0.5 * out["value_loss"] - 0.01 * out["entropy"]).backward()
        return {k: v.detach() for k, v in out.items()}, [p.grad.clone() for p in self.net.parameters()]

    def chunked(self, max_chunk, value_clip):
        self.net.zero_grad(set_to_none=True)
        out = chunked_backward(self.net, lambda rows: self.obs[rows], self.legal, self.actions, self.old_log_prob,
                               self.old_value, self.adv, self.returns, 0.2, value_clip, 0.5, 0.01, max_chunk)
        return out, [p.grad.clone() for p in self.net.parameters()]

    def test_pieces_give_the_whole_minibatchs_gradients_and_statistics(self):
        for value_clip in (0.0, 0.1):
            want_out, want_grads = self.whole(value_clip)
            self.assertGreater(want_out["clip_fraction"].item(), 0)  # the clipped branch is exercised
            for max_chunk in (1, 3, 5, 10, 64):  # one position each, uneven, even, the whole, more than the whole
                with self.subTest(max_chunk=max_chunk, value_clip=value_clip):
                    out, grads = self.chunked(max_chunk, value_clip)
                    for k in want_out:
                        torch.testing.assert_close(out[k], want_out[k], rtol=1e-10, atol=1e-12)
                    for g, w in zip(grads, want_grads):
                        torch.testing.assert_close(g, w, rtol=1e-9, atol=1e-12)


class MinibatchUpdateTests(MinibatchFixture):
    """minibatch_update: off, exactly the step training took before --max_chunk; on, the same step."""

    def step(self, max_chunk, net=None):
        net = net or copy.deepcopy(self.net)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, eps=1e-5)
        out = minibatch_update(net, optimizer, lambda rows: self.obs[rows], self.legal, self.actions, self.old_log_prob,
                               self.old_value, self.adv, self.returns, 0.2, 0.0, 0.5, 0.01, 0.1, max_chunk)
        return out, net

    def test_off_is_the_step_training_took_before_chunking(self):
        # The minibatch step as the training loop wrote it before --max_chunk
        before = copy.deepcopy(self.net)
        optimizer = torch.optim.AdamW(before.parameters(), lr=3e-4, eps=1e-5)
        out = ppo_losses(before, self.obs, self.legal, self.actions, self.old_log_prob, self.old_value, self.adv,
                         self.returns, 0.2, 0.0)
        loss = out["policy_loss"] + 0.5 * out["value_loss"] - 0.01 * out["entropy"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(before.parameters(), 0.1)
        optimizer.step()
        for max_chunk in (0, 10, 64):  # off, and on but no minibatch bigger than it
            with self.subTest(max_chunk=max_chunk):
                got, net = self.step(max_chunk)
                self.assertTrue(torch.equal(got["grad_norm"], norm))
                for p, q in zip(net.parameters(), before.parameters()):
                    self.assertTrue(torch.equal(p, q))  # bit for bit

    def test_on_takes_the_same_step(self):
        want, want_net = self.step(0)
        self.assertGreater(want["grad_norm"].item(), 0.1)  # clipping is exercised
        for max_chunk in (1, 3, 5):
            with self.subTest(max_chunk=max_chunk):
                got, net = self.step(max_chunk)
                for k in want:
                    torch.testing.assert_close(got[k], want[k], rtol=1e-9, atol=1e-12)
                for p, q in zip(net.parameters(), want_net.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-9, atol=1e-12)


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
            self.assertEqual(ck["config"]["max_chunk"], 0)  # off unless asked for

    def one_rollout(self, tmp, name, *args):
        out = self.run_ppo("--checkpoint-dir", str(Path(tmp) / name), "--num_moves", "64", *args)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        return torch.load(Path(tmp) / name / "model-latest.ckpt.pt", map_location="cpu", weights_only=False)

    def test_a_resumed_run_keeps_its_chunking_unless_told_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = self.one_rollout(tmp, "run", "--max_chunk", "5")
            latest = Path(tmp) / "run" / "model-latest.ckpt.pt"
            kept = self.run_ppo("--checkpoint-dir", str(Path(tmp) / "run"), "--num_moves", "128", "--resume", str(latest))
            self.assertEqual(kept.returncode, 0, kept.stderr[-2000:])
            self.assertEqual(torch.load(latest, map_location="cpu", weights_only=False)["config"]["max_chunk"], 5)
            changed = self.run_ppo("--checkpoint-dir", str(Path(tmp) / "run"), "--num_moves", "192",
                                   "--resume", str(latest), "--max_chunk", "0")
            self.assertEqual(changed.returncode, 0, changed.stderr[-2000:])
            self.assertEqual(torch.load(latest, map_location="cpu", weights_only=False)["config"]["max_chunk"], 0)
            # A checkpoint from before --max_chunk existed resumes with it off
            del ck["config"]["max_chunk"]
            old = Path(tmp) / "old.ckpt.pt"
            torch.save(ck, old)
            resumed = self.run_ppo("--checkpoint-dir", str(Path(tmp) / "old"), "--num_moves", "128", "--resume", str(old))
            self.assertEqual(resumed.returncode, 0, resumed.stderr[-2000:])
            self.assertEqual(torch.load(Path(tmp) / "old" / "model-latest.ckpt.pt", map_location="cpu",
                                        weights_only=False)["config"]["max_chunk"], 0)


if __name__ == "__main__":
    unittest.main()
