"""
The maze as a graph, and the pieces the graph models (MazeGraphQNet, MazeTransformerQNet)
share: reading the observation at the maze's squares, and scoring each move from the square it
leads to.

The 288 squares Pacman can be on are the graph's nodes (computed_data/node_coords.json, as
(row, col) in game coordinates); neighbouring squares are joined. Observations are
(17, 28, 31) grids indexed [channel, col, 30 - row] (pacbot_rs env.rs obs()):
  0 walls, 1 rewards (pellets, fruit, frightened ghosts), 2-3 Pacman, 4-7 ghosts, 8-11 ghosts'
  next squares (or their current ones), 12-14 ghost mode timers (scatter, chase, frightened)
  at the ghosts' squares, 15 speed (everywhere), 16 power pellets.
Ghosts waiting in the ghost house are on wall squares, off the graph, so whole-board features
say where the ghosts' mass is not on the graph, the largest timer values, and the speed.

Actions: 0 stay, 1 down (row + 1), 2 up (row - 1), 3 left (col - 1), 4 right (col + 1); a move
is legal exactly when the square in its direction is on the graph.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import nn

NODE_COORDS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12),
    (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (1, 24), (1, 25), (1, 26),
    (2, 1), (2, 6), (2, 12), (2, 15), (2, 21), (2, 26), (3, 1), (3, 6), (3, 12), (3, 15), (3, 21), (3, 26),
    (4, 1), (4, 6), (4, 12), (4, 15), (4, 21), (4, 26), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
    (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18),
    (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (6, 1), (6, 6), (6, 9), (6, 18),
    (6, 21), (6, 26), (7, 1), (7, 6), (7, 9), (7, 18), (7, 21), (7, 26), (8, 1), (8, 2), (8, 3), (8, 4),
    (8, 5), (8, 6), (8, 9), (8, 10), (8, 11), (8, 12), (8, 15), (8, 16), (8, 17), (8, 18), (8, 21), (8, 22),
    (8, 23), (8, 24), (8, 25), (8, 26), (9, 6), (9, 12), (9, 15), (9, 21), (10, 6), (10, 12), (10, 15), (10, 21),
    (11, 6), (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 21),
    (12, 6), (12, 9), (12, 18), (12, 21), (13, 6), (13, 9), (13, 18), (13, 21), (14, 6), (14, 7), (14, 8), (14, 9),
    (14, 18), (14, 19), (14, 20), (14, 21), (15, 6), (15, 9), (15, 18), (15, 21), (16, 6), (16, 9), (16, 18), (16, 21),
    (17, 6), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17), (17, 18), (17, 21),
    (18, 6), (18, 9), (18, 18), (18, 21), (19, 6), (19, 9), (19, 18), (19, 21), (20, 1), (20, 2), (20, 3), (20, 4),
    (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (20, 10), (20, 11), (20, 12), (20, 15), (20, 16), (20, 17), (20, 18),
    (20, 19), (20, 20), (20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (21, 1), (21, 6), (21, 12), (21, 15),
    (21, 21), (21, 26), (22, 1), (22, 6), (22, 12), (22, 15), (22, 21), (22, 26), (23, 1), (23, 2), (23, 3), (23, 6),
    (23, 7), (23, 8), (23, 9), (23, 10), (23, 11), (23, 12), (23, 13), (23, 14), (23, 15), (23, 16), (23, 17), (23, 18),
    (23, 19), (23, 20), (23, 21), (23, 24), (23, 25), (23, 26), (24, 3), (24, 6), (24, 9), (24, 18), (24, 21), (24, 24),
    (25, 3), (25, 6), (25, 9), (25, 18), (25, 21), (25, 24), (26, 1), (26, 2), (26, 3), (26, 4), (26, 5), (26, 6),
    (26, 9), (26, 10), (26, 11), (26, 12), (26, 15), (26, 16), (26, 17), (26, 18), (26, 21), (26, 22), (26, 23), (26, 24),
    (26, 25), (26, 26), (27, 1), (27, 12), (27, 15), (27, 26), (28, 1), (28, 12), (28, 15), (28, 26), (29, 1), (29, 2),
    (29, 3), (29, 4), (29, 5), (29, 6), (29, 7), (29, 8), (29, 9), (29, 10), (29, 11), (29, 12), (29, 13), (29, 14),
    (29, 15), (29, 16), (29, 17), (29, 18), (29, 19), (29, 20), (29, 21), (29, 22), (29, 23), (29, 24), (29, 25), (29, 26),
)
NUM_NODES = len(NODE_COORDS)
OBS_ROWS, OBS_COLS = 31, 28

# Directions of actions 0-4 as (d_row, d_col)
ACTION_DIRECTIONS = ((0, 0), (1, 0), (-1, 0), (0, -1), (0, 1))

# Observation channels read at each node, and the ghost channels whose mass may be off the graph
NODE_CHANNELS = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16)
GHOST_CHANNELS = (4, 5, 6, 7, 8, 9, 10, 11)
TIMER_CHANNELS = (12, 13, 14)
SPEED_CHANNEL = 15
NUM_NODE_FEATURES = len(NODE_CHANNELS)
NUM_GLOBAL_FEATURES = len(GHOST_CHANNELS) + len(TIMER_CHANNELS) + 1


def _index() -> dict[tuple[int, int], int]:
    return {rc: i for i, rc in enumerate(NODE_COORDS)}


def obs_flat_indices() -> torch.Tensor:
    """Each node's position in an observation flattened to (channel, col * 31 + obs_row)."""
    return torch.tensor([c * OBS_ROWS + (OBS_ROWS - 1 - r) for r, c in NODE_COORDS])


def move_targets() -> torch.Tensor:
    """(NUM_NODES, 5): the node each action leads to from each node (itself if illegal)."""
    index = _index()
    return torch.tensor([[index.get((r + dr, c + dc), i) for dr, dc in ACTION_DIRECTIONS]
                         for i, (r, c) in enumerate(NODE_COORDS)])


def distances() -> torch.Tensor:
    """(NUM_NODES, NUM_NODES) shortest path lengths through the maze, in squares."""
    index = _index()
    neighbours = [[index[(r + dr, c + dc)] for dr, dc in ACTION_DIRECTIONS[1:] if (r + dr, c + dc) in index]
                  for r, c in NODE_COORDS]
    dist = torch.full((NUM_NODES, NUM_NODES), -1, dtype=torch.long)
    for start in range(NUM_NODES):
        dist[start, start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in neighbours[node]:
                if dist[start, nb] < 0:
                    dist[start, nb] = dist[start, node] + 1
                    queue.append(nb)
    assert (dist >= 0).all(), "the maze graph is not connected"
    return dist


class MazeReader(nn.Module):
    """
    Reads an observation batch at the maze's nodes: returns (node features (B, NUM_NODES,
    NUM_NODE_FEATURES), whole-board features (B, NUM_GLOBAL_FEATURES), Pacman's node (B,)).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("flat_index", obs_flat_indices(), persistent=False)
        self.register_buffer("node_channels", torch.tensor(NODE_CHANNELS), persistent=False)
        self.register_buffer("ghost_channels", torch.tensor(GHOST_CHANNELS), persistent=False)
        self.register_buffer("timer_channels", torch.tensor(TIMER_CHANNELS), persistent=False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = obs.reshape(obs.shape[0], obs.shape[1], -1)  # (B, 17, 868); copies channels-last input
        at_nodes = flat.index_select(2, self.flat_index)  # (B, 17, NUM_NODES)
        nodes = at_nodes.index_select(1, self.node_channels).transpose(1, 2)
        ghosts_off_graph = (flat.index_select(1, self.ghost_channels).sum(2)
                            - at_nodes.index_select(1, self.ghost_channels).sum(2))
        timers = flat.index_select(1, self.timer_channels).amax(2)
        speed = flat[:, SPEED_CHANNEL, :1]
        whole = torch.cat([ghosts_off_graph, timers, speed], dim=1)
        pacman = at_nodes[:, 3].argmax(dim=1)
        return nodes, whole, pacman


class MoveHead(nn.Module):
    """
    Q values from node embeddings: each action's value from the embedding of the node it leads
    to, Pacman's node, the mean over all nodes, the whole-board features and the action's own
    embedding (staying differs from moving into a wall), through one shared MLP.
    """

    def __init__(self, dim: int, hidden: int, action_count: int = 5) -> None:
        super().__init__()
        if action_count != len(ACTION_DIRECTIONS):
            raise ValueError(f"the maze graph models have {len(ACTION_DIRECTIONS)} actions, got {action_count}")
        self.register_buffer("targets", move_targets(), persistent=False)
        self.action_embedding = nn.Parameter(torch.zeros(action_count, dim))
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            # Output zeros at the start of training, as the other models do
            self.mlp[-1].weight.fill_(0)
            self.mlp[-1].bias.fill_(0)

    def forward(self, h: torch.Tensor, whole: torch.Tensor, pacman: torch.Tensor) -> torch.Tensor:
        batch, _, dim = h.shape
        rows = torch.arange(batch, device=h.device)
        at_pacman = h[rows, pacman]  # (B, dim)
        target_nodes = self.targets[pacman]  # (B, 5)
        at_targets = h[rows[:, None], target_nodes]  # (B, 5, dim)
        context = torch.cat([at_pacman, h.mean(dim=1), whole], dim=1)  # (B, 2 dim + dim)
        features = torch.cat([at_targets + self.action_embedding, context[:, None].expand(-1, 5, -1)], dim=2)
        return self.mlp(features).squeeze(2)


class WholeBoard(nn.Module):
    """The whole-board features as a dim-sized vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(NUM_GLOBAL_FEATURES, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, whole: torch.Tensor) -> torch.Tensor:
        return self.proj(whole)


class MazeGraphQNet(nn.Module):
    """
    A graph network over the maze: each node starts from its observation features, a learned
    embedding of where it is, and the whole-board features; then each layer adds to every node
    an MLP of itself and the mean of the nodes within a maze distance (1, 2, 4, 8, 16, then 64
    squares: the whole maze). Q values come from MoveHead. Sized to cost about what QNetV2Wide
    does on the actors' CPUs (~2.5x QNetV2).
    """

    DIM = 48
    RADII = (1, 2, 4, 8, 16, 64)
    # Train with TF32 matrix products and torch.compile on CUDA (see train_dqn)
    FAST_CUDA = True

    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()
        dim = self.DIM
        self.reader = MazeReader()
        self.node_in = nn.Linear(NUM_NODE_FEATURES, dim)
        self.position = nn.Parameter(torch.randn(NUM_NODES, dim) * 0.1)
        self.whole = WholeBoard(dim)
        dist = distances()
        # Row-normalized "within radius" matrices, one per layer. A radius at least the maze's
        # diameter takes in every node: that layer's mean is the plain mean over the nodes.
        self.whole_maze = [r >= int(dist.max()) for r in self.RADII]
        within = torch.stack([(dist <= r).float() for r in self.RADII])
        self.register_buffer("within", within / within.sum(dim=2, keepdim=True), persistent=False)
        # On the CPU (actors, evaluation) they multiply as sparse matrices: radii 1-16 keep only
        # ~1-34% of the entries, and the dense products were ~45% of the arithmetic. Kept as
        # their CSR parts (plain tensors, so they move and copy like the other buffers).
        for i, w in enumerate(self.within):
            csr = w.to_sparse_csr()
            self.register_buffer(f"_csr{i}_rows", csr.crow_indices(), persistent=False)
            self.register_buffer(f"_csr{i}_cols", csr.col_indices(), persistent=False)
            self.register_buffer(f"_csr{i}_values", csr.values(), persistent=False)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))
            for _ in self.RADII
        )
        self.head = MoveHead(dim, 2 * dim, action_count)

    def _mean_within(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """The mean of h (NUM_NODES, B, dim) over the nodes within the layer's radius of each node."""
        if self.whole_maze[layer]:
            return h.mean(dim=0, keepdim=True).expand_as(h)
        flat = h.reshape(NUM_NODES, -1)  # the whole batch as one (N, N) @ (N, B * dim) product
        if h.device.type == "cpu":
            return self._sparse_mm(layer, flat).view_as(h)
        return torch.mm(self.within[layer], flat).view_as(h)

    @torch.compiler.disable  # sparse tensors can't be compiled; only the CPU takes this path
    def _sparse_mm(self, layer: int, flat: torch.Tensor) -> torch.Tensor:
        within = torch.sparse_csr_tensor(
            getattr(self, f"_csr{layer}_rows"), getattr(self, f"_csr{layer}_cols"),
            getattr(self, f"_csr{layer}_values"), (NUM_NODES, NUM_NODES),
        )
        return torch.sparse.mm(within, flat)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        nodes, whole, pacman = self.reader(obs)
        whole = self.whole(whole)
        h = self.node_in(nodes) + self.position + whole[:, None]  # (B, NUM_NODES, dim)
        # Node-major (NUM_NODES, B, dim) through the layers, so each mean is one 2-D product
        # (a batched product would copy the (N, N) matrix for every observation)
        h = h.transpose(0, 1).contiguous()
        for i, layer in enumerate(self.layers):
            h = h + layer(torch.cat([h, self._mean_within(i, h)], dim=2))
        return self.head(h.transpose(0, 1), whole, pacman)
