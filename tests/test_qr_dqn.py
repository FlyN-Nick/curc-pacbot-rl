"""
QR-DQN (models.QRNetV2 and dqn_update's quantile loss).

Run from src/: python -m unittest discover ../tests
"""

import copy
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
from dqn_update import EagerUpdate, GraphedUpdate, dqn_update  # noqa: E402
from replay_storage import Batch  # noqa: E402
from test_dqn_update import KW, filled_storage  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402

N = models.QRNetV2.NUM_QUANTILES


class FreeQuantiles(nn.Module):
    """The same quantiles for every observation: one learnable value per action and quantile."""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(NUM_ACTIONS, N))

    def quantiles(self, obs):
        return self.theta.expand(obs.shape[0], -1, -1)

    def forward(self, obs):
        return self.quantiles(obs).mean(dim=2)


def reference_loss(q_net, target_q_net, batch, reward_scale, discount_factor, kappa):
    """The QR-DQN loss written out with loops."""
    b = len(batch.action)
    tau = [(i + 0.5) / N for i in range(N)]
    with torch.no_grad():
        q_next = q_net(batch.next_obs).masked_fill(~batch.next_action_mask, -torch.inf)
        a_next = q_next.argmax(dim=1)
        z_next = target_q_net.quantiles(batch.next_obs)
    z = q_net.quantiles(batch.obs)
    total = 0.0
    for k in range(b):
        r = batch.reward[k].item() * reward_scale
        target = [r + (0.0 if batch.done[k] else discount_factor * z_next[k, a_next[k], j].item()) for j in range(N)]
        for i in range(N):
            pred = z[k, batch.action[k], i]
            for j in range(N):
                u = target[j] - pred
                h = 0.5 * u * u if abs(u.item()) <= kappa else kappa * (abs(u) - 0.5 * kappa)
                total = total + abs(tau[i] - float(u.item() < 0)) * h / kappa / N
    return total / b


class QrDqnTests(unittest.TestCase):
    def test_forward_is_the_mean_of_the_quantiles(self):
        torch.manual_seed(0)
        net = models.QRNetV2(OBS_SHAPE, NUM_ACTIONS)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        obs = torch.rand(4, *OBS_SHAPE)
        z = net.quantiles(obs)
        self.assertEqual(tuple(z.shape), (4, NUM_ACTIONS, N))
        torch.testing.assert_close(net(obs), z.mean(dim=2))

    def test_loss_matches_the_written_out_definition(self):
        device = torch.device("cpu")
        storage = filled_storage(device, capacity=200, steps=30)
        batch = storage.sample(6)
        torch.manual_seed(1)
        q_net = models.QRNetV2(OBS_SHAPE, NUM_ACTIONS)
        with torch.no_grad():
            for p in q_net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        target = copy.deepcopy(q_net)
        with torch.no_grad():
            for p in target.parameters():
                p.add_(torch.randn_like(p) * 0.05)
        for kappa in (1.0, 0.3):
            net = copy.deepcopy(q_net)
            opt = torch.optim.SGD(net.parameters(), lr=0.0)
            metrics = dqn_update(net, target, opt, batch, **KW, huber_delta=kappa if kappa != 1.0 else 0.0)
            expected = reference_loss(net, target, batch, KW["reward_scale"], KW["discount_factor"], kappa)
            # float32 sums against a float64 reference: the order of the sums differs by platform
            # (1.06e-5 on a Windows CI runner), so allow 1e-4
            self.assertLess(abs(metrics[0].item() - expected.item()) / expected.item(), 1e-4)

    def test_learns_the_quantiles_of_the_reward(self):
        """
        With games that end at once and a small kappa (close to plain quantile regression; a
        large kappa pulls the estimates towards expectiles), the quantiles converge to the
        reward's quantiles.
        """
        torch.manual_seed(2)
        net, target = FreeQuantiles(), FreeQuantiles()
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        b = 256
        obs = torch.zeros(b, 1)
        for _ in range(3000):
            reward = torch.randint(0, 100, (b,)).double()  # uniform on 0..99, scaled by 0.02
            batch = Batch(obs, obs, torch.ones(b, dtype=torch.bool), torch.zeros(b, NUM_ACTIONS, dtype=torch.bool),
                          torch.zeros(b, dtype=torch.long), reward)
            dqn_update(net, target, opt, batch, **KW, huber_delta=0.01)
        learned = net.theta[0].detach() / KW["reward_scale"]
        expected = torch.tensor([(i + 0.5) / N * 100 - 0.5 for i in range(N)])
        self.assertLess((learned - expected).abs().max().item(), 3.0)
        self.assertTrue(torch.all(learned[1:] >= learned[:-1] - 1.0))  # increasing

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_graph_replays_the_same_updates(self):
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", False)
        storage = filled_storage(device)
        storage.set_n_step(3, KW["discount_factor"])
        runs = []
        for cls in (EagerUpdate, GraphedUpdate):
            torch.manual_seed(0)
            q_net = models.QRNetV2(OBS_SHAPE, NUM_ACTIONS).to(device)
            with torch.no_grad():
                for p in q_net.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            target = copy.deepcopy(q_net)
            opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
            runs.append((q_net, cls(q_net, target, opt, storage, 64, **KW, huber_delta=1.0)))
        for _ in range(5):
            idx = storage.sample_indices(64)
            self.assertEqual(runs[0][1].run(idx).tolist(), runs[1][1].run(idx).tolist())
        for pe, pg in zip(runs[0][0].parameters(), runs[1][0].parameters()):
            self.assertTrue(torch.equal(pe, pg))


if __name__ == "__main__":
    unittest.main()
