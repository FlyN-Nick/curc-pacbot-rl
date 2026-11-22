"""
Split rewards (Hybrid Reward Architecture): models.HRANetV2, reward parts in the replay storage
and the collection code, and dqn_update's split-reward loss.

Run from src/: python -m unittest discover ../tests
"""

import copy
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
from dqn_update import EagerUpdate, GraphedUpdate, dqn_update  # noqa: E402
from policies import EpsilonGreedy, MaxQPolicy  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402
from replay_storage import ReplayStorage  # noqa: E402
from test_dqn_update import KW, ENVS  # noqa: E402
from test_replay_storage import real_observations  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402

K = models.HRANetV2.REWARD_PARTS


def parts_storage(device, capacity=600, steps=90, seed=0) -> ReplayStorage:
    """Storage with real observations and random transitions whose rewards are split in K parts."""
    rng = random.Random(seed)
    obs = real_observations(games=6, steps=120)
    storage = ReplayStorage(capacity, ENVS, device, reward_parts=K)
    storage.set_initial_obs(obs[:ENVS].to(device))
    for _ in range(steps):
        rows = [rng.randrange(len(obs)) for _ in range(ENVS)]
        dones = [rng.random() < 0.05 for _ in range(ENVS)]
        masks = [[False] * NUM_ACTIONS if d else [True] + [rng.random() < 0.6 for _ in range(4)] for d in dones]
        rewards = [[rng.choice([0, 10]), rng.choice([0, 0, 50]), rng.choice([0, 0, 200]),
                    rng.choice([0, 0, -200]), rng.choice([0, 0, 0, 3000]), rng.choice([0, -2, -20])]
                   for _ in range(ENVS)]
        storage.add_step(obs[rows].to(device), [rng.randrange(NUM_ACTIONS) for _ in range(ENVS)],
                         rewards, dones, masks)
    return storage


def perturbed(net, scale, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * scale)
    return net


def reference_loss(q_net, target_q_net, batch, reward_scale, discount_factor):
    """The split-reward loss written out with loops."""
    with torch.no_grad():
        q_next = q_net(batch.next_obs).masked_fill(~batch.next_action_mask, -torch.inf)
        parts_next = target_q_net.components(batch.next_obs)
    parts = q_net.components(batch.obs)
    total = 0.0
    for b in range(len(batch.action)):
        a_next = q_next[b].argmax().item()
        for k in range(K):
            target = batch.reward[b, k].item() * reward_scale
            if not batch.done[b]:
                target += discount_factor * parts_next[b, k, a_next].item()
            total = total + (parts[b, k, batch.action[b]] - target) ** 2
    return total / len(batch.action)


class HraTests(unittest.TestCase):
    def test_forward_is_the_sum_of_the_parts(self):
        for cls in (models.HRANetV2, models.HRANetV2Wide):
            net = perturbed(cls(OBS_SHAPE, NUM_ACTIONS), 0.05, 0)
            obs = torch.rand(4, *OBS_SHAPE)
            parts = net.components(obs)
            self.assertEqual(tuple(parts.shape), (4, K, NUM_ACTIONS))
            torch.testing.assert_close(net(obs), parts.sum(dim=1))
        # Fresh networks predict 0 for every part, like the other Q networks
        self.assertEqual(models.HRANetV2(OBS_SHAPE, NUM_ACTIONS)(obs).abs().max().item(), 0.0)

    def test_loss_matches_the_written_out_definition(self):
        storage = parts_storage(torch.device("cpu"), capacity=200, steps=30)
        batch = storage.sample(6)
        self.assertEqual(tuple(batch.reward.shape), (6, K))
        q_net = perturbed(models.HRANetV2(OBS_SHAPE, NUM_ACTIONS), 0.05, 1)
        target = perturbed(copy.deepcopy(q_net), 0.05, 2)
        opt = torch.optim.SGD(q_net.parameters(), lr=0.0)
        metrics = dqn_update(q_net, target, opt, batch, **KW)
        expected = reference_loss(q_net, target, batch, KW["reward_scale"], KW["discount_factor"])
        self.assertLess(abs(metrics[0].item() - expected.item()) / expected.item(), 1e-4)

    def test_n_step_rewards_add_up_part_by_part(self):
        storage = parts_storage(torch.device("cpu"))
        storage.set_n_step(3, 0.9)
        idx = storage.sample_indices(32)
        batch = storage.gather(idx)
        for b in range(32):
            expected = torch.zeros(K, dtype=torch.float64)
            for j, pos in enumerate(idx.position[b].tolist()):
                expected += 0.9 ** j * storage._reward[pos]
                if storage._done[pos]:
                    break
            torch.testing.assert_close(batch.reward[b], expected)

    def test_collection_stores_the_games_reward_parts(self):
        device = torch.device("cpu")
        net = models.HRANetV2(OBS_SHAPE, NUM_ACTIONS)
        storage = ReplayStorage(400, 4, device, reward_parts=K)
        buffer = ReplayBuffer(maxlen=400, policy=EpsilonGreedy(MaxQPolicy(net), NUM_ACTIONS, 1.0),
                              num_parallel_envs=4, random_start_proportion=0.0, device=device,
                              storage=storage, reward_parts=True)
        for _ in range(60):
            buffer.generate_experience_step()
        rewards = storage._reward[:len(storage)]
        self.assertEqual(tuple(rewards.shape), (240, K))
        self.assertGreater(rewards[:, 0].sum().item(), 0)  # pellets eaten
        self.assertTrue(set(rewards[:, 0].tolist()) <= {0.0, 10.0})  # one pellet a move
        self.assertTrue(set(rewards[:, 3].tolist()) <= {0.0, -200.0})  # the death penalty

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_graph_replays_the_same_updates(self):
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", False)
        storage = parts_storage(device)
        storage.set_n_step(3, KW["discount_factor"])
        runs = []
        for cls in (EagerUpdate, GraphedUpdate):
            q_net = perturbed(models.HRANetV2(OBS_SHAPE, NUM_ACTIONS).to(device), 0.01, 0)
            target = copy.deepcopy(q_net)
            opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
            runs.append((q_net, cls(q_net, target, opt, storage, 64, **KW)))
        for _ in range(5):
            idx = storage.sample_indices(64)
            self.assertEqual(runs[0][1].run(idx).tolist(), runs[1][1].run(idx).tolist())
        for pe, pg in zip(runs[0][0].parameters(), runs[1][0].parameters()):
            self.assertTrue(torch.equal(pe, pg))


if __name__ == "__main__":
    unittest.main()
