"""
Search distillation (search_distill): the store of searched states, the search loss in
dqn_update (eagerly and as a CUDA graph), and searcher processes playing with the weights
published to the actors.

Run from src/: python -m unittest discover -s ../tests -p test_search_distill.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
import utils  # noqa: E402
from dqn_update import EagerUpdate, GraphedUpdate, dqn_update  # noqa: E402
from search_distill import SearchBatch, SearchStore, SearchTargets  # noqa: E402
from test_dqn_update import KW, filled_storage, nets  # noqa: E402
from test_replay_storage import real_observations  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402

DEVICES = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])


def searched_states(n: int, seed: int = 0):
    """Real observations with made-up searched values and legal moves."""
    rng = np.random.default_rng(seed)
    obs = real_observations(games=4, steps=60)[:n].numpy()
    legal = rng.random((n, NUM_ACTIONS)) < 0.6
    legal[:, 4] = True  # staying is always legal
    values = (rng.normal(size=(n, NUM_ACTIONS)) * legal).astype(np.float32)
    return obs, values, legal


def filled_store(device, capacity=100, n=130) -> tuple[SearchStore, tuple]:
    obs, values, legal = searched_states(n)
    store = SearchStore(capacity, device)
    for start in range(0, n, 8):  # in chunks, as the searchers send them; wraps around
        store.add(obs[start:start + 8], values[start:start + 8], legal[start:start + 8])
    return store, (obs, values, legal)


class SearchStoreTests(unittest.TestCase):
    def test_keeps_the_newest_states_exactly(self):
        for device in DEVICES:
            store, (obs, values, legal) = filled_store(device)
            self.assertEqual((store.total, len(store)), (130, 100))
            # The newest 100 (30..129) are kept; state i is in row i % 100
            rows = torch.arange(100, device=device)
            batch = store.gather(rows)
            for row in range(100):
                i = row + 100 if row < 30 else row
                self.assertTrue(np.array_equal(batch.obs[row].cpu().numpy(), obs[i]), (device, row))
                self.assertTrue(np.array_equal(batch.values[row].cpu().numpy(), values[i]), (device, row))
                self.assertTrue(np.array_equal(batch.legal[row].cpu().numpy(), legal[i]), (device, row))
            self.assertEqual(store.codec.mismatches(), 0)

    def test_samples_only_stored_states(self):
        store = SearchStore(50, torch.device("cpu"))
        with self.assertRaises(ValueError):
            store.sample_indices(4)
        obs, values, legal = searched_states(8)
        store.add(obs, values, legal)
        self.assertLess(int(store.sample_indices(1000).max()), 8)


class SearchLossTests(unittest.TestCase):
    def test_loss_is_the_mean_squared_error_over_legal_moves(self):
        device = torch.device("cpu")
        storage = filled_storage(device)
        store, _ = filled_store(device)
        search = store.gather(torch.arange(32))
        q_net, target = nets(device)
        with torch.no_grad():
            q = q_net(search.obs)
        expected = ((q - search.values)[search.legal] ** 2).mean().item()
        opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
        metrics = dqn_update(q_net, target, opt, storage.sample(32), search=search, search_weight=0.5, **KW)
        self.assertEqual(len(metrics), 5)
        self.assertAlmostEqual(metrics[4].item(), expected, places=5)
        without = dqn_update(q_net, target, opt, storage.sample(32), **KW)
        self.assertEqual(len(without), 4)

    def test_search_loss_pulls_q_values_to_the_searched_values(self):
        device = torch.device("cpu")
        storage = filled_storage(device)
        store, _ = filled_store(device, n=64, capacity=64)
        search = store.gather(torch.arange(64))
        # Learnable targets: each move has the same value in every state
        search = search._replace(values=torch.tensor([1.0, -1.0, 0.5, 2.0, 0.0]) * search.legal)
        q_net, target = nets(device)
        opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
        losses = [dqn_update(q_net, target, opt, storage.sample(16), search=search, search_weight=50.0,
                             **KW)[4].item() for _ in range(60)]
        self.assertLess(losses[-1], 0.5 * losses[0])

    def test_quantile_network_uses_the_mean_q_values(self):
        device = torch.device("cpu")
        storage = filled_storage(device)
        store, _ = filled_store(device)
        search = store.gather(torch.arange(16))
        torch.manual_seed(0)
        q_net = models.QRNetV2(OBS_SHAPE, NUM_ACTIONS)
        target = models.QRNetV2(OBS_SHAPE, NUM_ACTIONS)
        target.load_state_dict(q_net.state_dict())
        with torch.no_grad():
            expected = ((q_net(search.obs) - search.values)[search.legal] ** 2).mean().item()
        opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
        metrics = dqn_update(q_net, target, opt, storage.sample(16), search=search, search_weight=1.0, **KW)
        self.assertAlmostEqual(metrics[4].item(), expected, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_graph_replays_the_same_updates_with_search(self):
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", False)
        storage = filled_storage(device)
        store, _ = filled_store(device)
        runs = []
        for cls in (EagerUpdate, GraphedUpdate):
            q_net, target = nets(device)
            opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
            runs.append((q_net, cls(q_net, target, opt, storage, 64, search_targets=store,
                                    search_batch_size=32, search_weight=2.0, **KW)))
        for _ in range(5):
            idx, sidx = storage.sample_indices(64), store.sample_indices(32)
            m_eager = runs[0][1].run(idx, sidx).tolist()
            m_graph = runs[1][1].run(idx, sidx).tolist()
            self.assertEqual(len(m_eager), 5)
            self.assertEqual(m_eager, m_graph)
        for pe, pg in zip(runs[0][0].parameters(), runs[1][0].parameters()):
            self.assertTrue(torch.equal(pe, pg))


class SearcherTests(unittest.TestCase):
    def setUp(self):
        utils.set_single_model(True)
        self.addCleanup(utils.set_single_model, False)

    def test_searchers_send_searched_states(self):
        import parallel_training

        device = torch.device("cpu")
        torch.manual_seed(0)
        q_net = models.QNetV2(OBS_SHAPE, NUM_ACTIONS)
        actors = parallel_training.ActorReplayBuffer(
            maxlen=64, q_net=q_net, model_name="QNetV2", epsilon=1.0, num_parallel_envs=2,
            num_actors=1, max_steps=1, device=device, threads_per_actor=1,
        )
        self.addCleanup(actors.close)
        targets = SearchTargets(actors.weights_source(), num_workers=2, capacity=200, device=device,
                                model_name="QNetV2", depth=1, reward_scale=0.02, discount=0.99,
                                epsilon=0.1, random_start_proportion=0.5)
        self.addCleanup(targets.close)
        targets.drain(at_least=48)
        self.assertGreaterEqual(len(targets), 48)
        batch = targets.gather(torch.arange(len(targets)))
        self.assertTrue(batch.legal.any(dim=1).all())
        self.assertTrue(torch.isfinite(batch.values).all())
        self.assertTrue((batch.values[~batch.legal] == 0).all())
        self.assertEqual(targets.codec.mismatches(), 0)
        # A searched value is the move's scaled reward plus the discounted best Q value after it:
        # with the published (fresh, zero-headed) network the Q values are 0, so values are
        # scaled rewards, multiples of 0.02 * (pellet values)
        self.assertTrue(torch.allclose(batch.values * 50, (batch.values * 50).round(), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
