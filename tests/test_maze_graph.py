"""
The maze graph models' shared pieces (pacbot_rl_models.maze_graph) checked against the real
game, and MazeGraphQNet in training updates.

Run from src/: python -m unittest discover -s ../tests -p test_maze_graph.py
"""

import copy
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: E402
import utils  # noqa: E402
from dqn_update import EagerUpdate, GraphedUpdate, dqn_update  # noqa: E402
from pacbot_rl_models import maze_graph as mg  # noqa: E402
from test_dqn_update import KW, filled_storage  # noqa: E402
from utils import NUM_ACTIONS, OBS_SHAPE  # noqa: E402

CONFIG = utils.SINGLE_MODEL_REPLAY_BUFFER_OBS_GYM_CONFIGURATION
GRAPH_MODELS = [models.MazeGraphQNet]


def random_states(n: int, seed: int = 0):
    """Games played a random number of random legal moves from the start, at random speeds."""
    rng = random.Random(seed)
    states = []
    while len(states) < n:
        env = utils.make_gym(utils.speed_configuration(rng.randint(4, 14)), utils.ENV_VERSION)
        for _ in range(rng.randint(0, 300)):
            mask = env.action_mask()
            _, done = env.step(rng.choice([a for a in range(NUM_ACTIONS) if mask[a]]))
            if done:
                break
        if not env.is_done():
            states.append(env)
    return states


def pacman(env) -> tuple[int, int]:
    b = env.game_state_bytes()
    return b[29] & 0x3F, b[30] & 0x3F


class MazeGraphTests(unittest.TestCase):
    def test_nodes_are_the_open_squares(self):
        env = utils.make_gym(utils.speed_configuration(8), utils.ENV_VERSION)
        walls = env.obs_numpy(CONFIG)[0]
        open_squares = {(30 - r, c) for c, r in zip(*np.nonzero(walls == 0))}
        self.assertEqual(open_squares, set(mg.NODE_COORDS))

    def test_moves_lead_where_the_game_goes(self):
        index = {rc: i for i, rc in enumerate(mg.NODE_COORDS)}
        targets = mg.move_targets()
        for env in random_states(40):
            node = index[pacman(env)]
            mask = env.action_mask()
            for action in range(NUM_ACTIONS):
                # Legal exactly when the move leaves the square (or stays)
                self.assertEqual(mask[action], action == 0 or targets[node, action] != node)
                if mask[action]:
                    moved = env.copy()
                    moved.step(action)
                    if not moved.is_done():
                        self.assertEqual(index[pacman(moved)], targets[node, action].item())

    def test_reader_reads_the_observation_at_the_nodes(self):
        reader = mg.MazeReader()
        index = {rc: i for i, rc in enumerate(mg.NODE_COORDS)}
        envs = random_states(16, seed=1)
        obs = torch.from_numpy(np.stack([e.obs_numpy(CONFIG) for e in envs]))
        for layout in (obs, obs.to(memory_format=torch.channels_last)):
            nodes, whole, at = reader(layout)
            self.assertEqual(tuple(nodes.shape), (16, mg.NUM_NODES, mg.NUM_NODE_FEATURES))
            for b, env in enumerate(envs):
                self.assertEqual(at[b].item(), index[pacman(env)])
                for i, (r, c) in enumerate(mg.NODE_COORDS[::17]):
                    node = index[(r, c)]
                    expected = obs[b, list(mg.NODE_CHANNELS), c, 30 - r]
                    self.assertTrue(torch.equal(nodes[b, node], expected))
                # Ghost mass off the graph plus on it is all of it; the speed is channel 15
                ghost_total = obs[b, list(mg.GHOST_CHANNELS)].sum(dim=(1, 2))
                on_graph = nodes[b, :, 2:10].sum(0)
                self.assertTrue(torch.allclose(whole[b, :8] + on_graph, ghost_total))
                self.assertAlmostEqual(whole[b, -1].item(), obs[b, 15, 0, 0].item())

    def test_distances_are_maze_distances(self):
        dist = mg.distances()
        self.assertTrue(torch.equal(dist, dist.T))
        targets = mg.move_targets()
        for node in range(0, mg.NUM_NODES, 7):
            for action in range(1, NUM_ACTIONS):
                if targets[node, action] != node:
                    self.assertEqual(dist[node, targets[node, action]].item(), 1)


class GraphModelTests(unittest.TestCase):
    def test_outputs_zero_at_the_start_and_trains_every_parameter(self):
        obs = torch.from_numpy(np.stack([e.obs_numpy(CONFIG) for e in random_states(8, seed=2)]))
        for cls in GRAPH_MODELS:
            torch.manual_seed(0)
            q_net = cls(OBS_SHAPE, NUM_ACTIONS)
            out = q_net(obs)
            self.assertEqual(tuple(out.shape), (8, NUM_ACTIONS))
            self.assertEqual(out.abs().max().item(), 0.0)
            with torch.no_grad():
                q_net.head.mlp[-1].weight.normal_()
            q_net(obs).square().sum().backward()
            no_grad = [n for n, p in q_net.named_parameters() if p.grad is None or not p.grad.any()]
            self.assertEqual(no_grad, [], cls.__name__)

    def test_learns_in_dqn_updates(self):
        device = torch.device("cpu")
        storage = filled_storage(device)
        for cls in GRAPH_MODELS:
            torch.manual_seed(0)
            q_net = cls(OBS_SHAPE, NUM_ACTIONS)
            target = copy.deepcopy(q_net)
            opt = torch.optim.Adam(q_net.parameters(), lr=1e-3)
            batch = storage.sample(64)
            losses = [dqn_update(q_net, target, opt, batch, **KW)[0].item() for _ in range(80)]
            self.assertLess(losses[-1], 0.5 * losses[0], cls.__name__)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_graph_replays_the_same_updates(self):
        device = torch.device("cuda")
        storage = filled_storage(device)
        for cls in GRAPH_MODELS:
            runs = []
            for update_cls in (EagerUpdate, GraphedUpdate):
                torch.manual_seed(0)
                q_net = cls(OBS_SHAPE, NUM_ACTIONS).to(device)
                with torch.no_grad():
                    q_net.head.mlp[-1].weight.normal_(std=0.01)
                target = copy.deepcopy(q_net)
                opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
                runs.append((q_net, update_cls(q_net, target, opt, storage, 64, **KW)))
            for _ in range(4):
                idx = storage.sample_indices(64)
                m_eager = runs[0][1].run(idx).tolist()
                m_graph = runs[1][1].run(idx).tolist()
                for a, b in zip(m_eager, m_graph):
                    self.assertAlmostEqual(a, b, places=4, msg=cls.__name__)


def dense_reference(q_net: "mg.MazeGraphQNet", obs: torch.Tensor) -> torch.Tensor:
    """MazeGraphQNet's output computed the plain way: batched dense products with `within`."""
    nodes, whole, pacman = q_net.reader(obs)
    whole = q_net.whole(whole)
    h = q_net.node_in(nodes) + q_net.position + whole[:, None]
    for within, layer in zip(q_net.within, q_net.layers):
        h = h + layer(torch.cat([h, torch.matmul(within, h)], dim=2))
    return q_net.head(h, whole, pacman)


class FastPathTests(unittest.TestCase):
    """MazeGraphQNet's fast paths (sparse on the CPU, compiled on CUDA) compute the same thing."""

    def setUp(self):
        self.obs = torch.from_numpy(np.stack([e.obs_numpy(CONFIG) for e in random_states(16, seed=11)]))
        torch.manual_seed(0)
        self.q_net = models.MazeGraphQNet(OBS_SHAPE, NUM_ACTIONS).eval()
        with torch.no_grad():
            self.q_net.head.mlp[-1].weight.normal_()

    def test_cpu_sparse_path_matches_dense(self):
        self.assertEqual(self.q_net.whole_maze, [False] * 5 + [True])
        with torch.no_grad():
            ref = dense_reference(self.q_net, self.obs)
            out = self.q_net(self.obs)
            channels_last = models.MazeGraphQNet(OBS_SHAPE, NUM_ACTIONS).eval()
            channels_last.load_state_dict(self.q_net.state_dict())
            channels_last = channels_last.to(memory_format=torch.channels_last)
            out_cl = channels_last(self.obs.contiguous(memory_format=torch.channels_last))
        self.assertGreater(ref.abs().max().item(), 1.0)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(out_cl, ref, rtol=1e-5, atol=1e-4)
        # Gradients flow through the sparse products too
        q_net = copy.deepcopy(self.q_net).train()
        q_net(self.obs).square().sum().backward()
        self.assertTrue(all(p.grad is not None for p in q_net.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_and_compiled_match_dense(self):
        device = torch.device("cuda")
        with torch.no_grad():
            ref = dense_reference(self.q_net, self.obs)
            q_net = copy.deepcopy(self.q_net).to(device)
            out = q_net(self.obs.to(device)).cpu()
            q_net.compile()
            compiled = q_net(self.obs.to(device)).cpu()
            back = q_net.cpu()(self.obs)  # moved back, the sparse matrices are rebuilt on the CPU
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(compiled, ref, rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(back, ref, rtol=1e-5, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_compiled_updates_replay_as_a_cuda_graph(self):
        device = torch.device("cuda")
        storage = filled_storage(device)
        runs = []
        for compiled in (False, True):
            torch.manual_seed(0)
            q_net = models.MazeGraphQNet(OBS_SHAPE, NUM_ACTIONS).to(device)
            with torch.no_grad():
                q_net.head.mlp[-1].weight.normal_(std=0.01)
            target = copy.deepcopy(q_net)
            if compiled:
                q_net.compile()
                target.compile()
            opt = torch.optim.Adam(q_net.parameters(), lr=1e-3, capturable=True)
            update_cls = GraphedUpdate if compiled else EagerUpdate
            runs.append((q_net, update_cls(q_net, target, opt, storage, 64, **KW)))
        for _ in range(4):
            idx = storage.sample_indices(64)
            m_eager = runs[0][1].run(idx).tolist()
            m_graph = runs[1][1].run(idx).tolist()
            for a, b in zip(m_eager, m_graph):
                self.assertAlmostEqual(a, b, places=3)
        for (name, a), b in zip(runs[0][0].state_dict().items(), runs[1][0].state_dict().values()):
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5, msg=name)


if __name__ == "__main__":
    unittest.main()
