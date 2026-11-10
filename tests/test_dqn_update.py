"""
Tests for dqn_update: the update matches the training loop's original code, the CUDA graph
replays the same updates as running them eagerly, and capturing leaves no trace. Run from the
repository root with PYTHONPATH=src.
"""

import copy
import random
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
from dqn_update import EagerUpdate, GraphedUpdate, dqn_update, set_capturable  # noqa: E402
from replay_storage import ReplayStorage  # noqa: E402
from test_replay_storage import real_observations  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402

KW = dict(reward_scale=0.02, discount_factor=0.99, grad_clip_norm=10.0)
ENVS = 8


def filled_storage(device, capacity=600, steps=90, seed=0) -> ReplayStorage:
    """A storage holding real observations with random transitions (with some game ends)."""
    rng = random.Random(seed)
    obs = real_observations(games=6, steps=120)
    storage = ReplayStorage(capacity, ENVS, device)
    storage.set_initial_obs(obs[:ENVS].to(device))
    for step in range(steps):
        rows = [rng.randrange(len(obs)) for _ in range(ENVS)]
        dones = [rng.random() < 0.05 for _ in range(ENVS)]
        masks = [[False] * NUM_ACTIONS if d else [True] + [rng.random() < 0.6 for _ in range(4)]
                 for d in dones]
        storage.add_step(
            obs[rows].to(device),
            [rng.randrange(NUM_ACTIONS) for _ in range(ENVS)],
            [rng.choice([0, 10, 50, -2, -20, -200]) for _ in range(ENVS)],
            dones,
            masks,
        )
    return storage


def nets(device, seed=0):
    torch.manual_seed(seed)
    q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS).to(device)
    # Non-zero heads, so the update isn't trivially symmetric
    with torch.no_grad():
        for p in q_net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    return q_net, copy.deepcopy(q_net)


def reference_update(q_net, target_q_net, optimizer, batch, reward_scale, discount_factor, grad_clip_norm):
    """The training loop's update before dqn_update, frozen (with its zero next observations)."""
    next_obs = torch.where(batch.done.view(-1, 1, 1, 1), torch.zeros_like(batch.next_obs), batch.next_obs)
    reward = torch.tensor([r * reward_scale for r in batch.reward.tolist()], device=batch.obs.device)
    n = len(batch.action)
    with torch.no_grad():
        next_q_values = target_q_net(next_obs)
        next_q_values[~batch.next_action_mask] = -torch.inf
        online = q_net(next_obs)
        online[~batch.next_action_mask] = -torch.inf
        next_actions = online.argmax(dim=1)
        returns = next_q_values[range(n), next_actions]
        discounted = discount_factor * returns
        discounted[batch.done] = 0.0
        target = reward + discounted
    q_net.train()
    optimizer.zero_grad()
    pred = q_net(batch.obs)[range(n), batch.action]
    loss = F.mse_loss(pred, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=grad_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return loss


def params(module):
    return [p.detach().clone() for p in module.parameters()]


class DqnUpdateTests(unittest.TestCase):
    def test_matches_the_original_update(self):
        device = torch.device("cpu")
        storage = filled_storage(device)
        a, a_target = nets(device)
        b, b_target = nets(device)
        opt_a = torch.optim.Adam(a.parameters(), lr=1e-3)
        opt_b = torch.optim.Adam(b.parameters(), lr=1e-3)
        dones = 0
        for _ in range(5):
            batch = storage.sample(64)
            dones += int(batch.done.sum())
            expected = reference_update(a, a_target, opt_a, batch, **KW)
            metrics = dqn_update(b, b_target, opt_b, batch, **KW)
            self.assertEqual(metrics[0].item(), expected.item())
            for pa, pb in zip(a.parameters(), b.parameters()):
                self.assertTrue(torch.equal(pa, pb))
        self.assertGreater(dones, 0)  # finished games were exercised

    def test_capturable_optimizer_state_loads_either_way(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        storage = filled_storage(device)
        q_net, target = nets(device)
        opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=device.type == "cuda")
        dqn_update(q_net, target, opt, storage.sample(32), **KW)
        state = opt.state_dict()
        plain = torch.optim.Adam(q_net.parameters(), lr=1e-3)
        plain.load_state_dict(state)
        set_capturable(plain, False)
        dqn_update(q_net, target, plain, storage.sample(32), **KW)  # runs without errors
        self.assertTrue(all(not g["capturable"] for g in plain.param_groups))


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA GPU")
class GraphedUpdateTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True  # so eager and graphed runs can match exactly
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", False)

    def make(self, cls, storage, preload_steps=0):
        q_net, target = nets(self.device)
        opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
        for _ in range(preload_steps):  # an optimizer with state, as when fine-tuning
            dqn_update(q_net, target, opt, storage.sample(32), **KW)
        return q_net, target, opt, cls(q_net, target, opt, storage, 64, **KW)

    def test_graph_replays_the_same_updates(self):
        storage = filled_storage(self.device)
        eager = self.make(EagerUpdate, storage)
        graphed = self.make(GraphedUpdate, storage)
        for _ in range(6):
            idx = storage.sample_indices(64)
            m_eager = eager[3].run(idx).tolist()
            m_graph = graphed[3].run(idx).tolist()
            self.assertEqual(m_eager, m_graph)
        for pe, pg in zip(eager[0].parameters(), graphed[0].parameters()):
            self.assertTrue(torch.equal(pe, pg))
        # In-place target updates are seen by the graph
        for q_net, target, _, _ in (eager, graphed):
            target.load_state_dict(q_net.state_dict())
        idx = storage.sample_indices(64)
        self.assertEqual(eager[3].run(idx).tolist(), graphed[3].run(idx).tolist())

    def test_capture_leaves_no_trace(self):
        storage = filled_storage(self.device)
        for preload in (0, 2):
            q_net, target = nets(self.device)
            opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
            for _ in range(preload):
                dqn_update(q_net, target, opt, storage.sample(32), **KW)
            before = params(q_net)
            state_before = copy.deepcopy(opt.state_dict()["state"])
            GraphedUpdate(q_net, target, opt, storage, 64, **KW)
            for p, b in zip(q_net.parameters(), before):
                self.assertTrue(torch.equal(p, b), preload)
            state_after = opt.state_dict()["state"]
            for key in state_before:
                for name, value in state_before[key].items():
                    self.assertTrue(torch.equal(state_after[key][name], value), (preload, name))
            if preload == 0:  # the warm-up's state was reset to Adam's fresh state
                self.assertTrue(all(float(s["step"]) == 0 for s in state_after.values()))


if __name__ == "__main__":
    unittest.main()
