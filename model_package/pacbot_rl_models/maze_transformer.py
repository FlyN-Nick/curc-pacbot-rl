"""
MazeTransformerQNet: a transformer over the maze's squares (see maze_graph for the graph, the
observation reader and the move head it shares with MazeGraphQNet), in the Perceiver IO form:

  1. Each of the 288 squares Pacman can be on is a token: the observation at its square, a
     learned embedding of where it is, and an embedding of its maze distance from Pacman
     (bucketed), so where things are relative to Pacman, through the maze, is explicit.
     A board token holds the whole-board features (ghosts in the ghost house, timers, speed).
  2. LATENTS learned tokens read the squares and the board token (cross-attention), then attend
     to each other for LAYERS blocks: global, content-based attention (a latent can follow a
     ghost anywhere on the board).
  3. Every square reads back from the latents (cross-attention), and MoveHead scores each move
     from the square it leads to.

Full self-attention over the 289 tokens costs ~15-20 ms per layer for a batch of 16 on one CPU
thread, too slow for the actors, which play with the network on the CPU; going through 32
latents makes attention ~9x cheaper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .maze_graph import NUM_GLOBAL_FEATURES, NUM_NODE_FEATURES, NUM_NODES, MazeReader, MoveHead, distances

# Upper ends of the maze distance buckets (the maze's diameter is 53)
DISTANCE_BUCKETS = (0, 1, 2, 3, 4, 6, 9, 13, 19, 27, 39)


def distance_buckets() -> torch.Tensor:
    """(NUM_NODES, NUM_NODES) bucket (0..len(DISTANCE_BUCKETS)) of the maze distance between squares."""
    return torch.bucketize(distances(), torch.tensor(DISTANCE_BUCKETS))


class _Attend(nn.Module):
    """A pre-norm block: queries attend to a context (themselves, for self-attention), then an MLP."""

    def __init__(self, dim: int, heads: int, ffn: int, cross: bool) -> None:
        super().__init__()
        self.heads = heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim) if cross else None
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.SiLU(), nn.Linear(ffn, dim))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        return x.view(batch, tokens, self.heads, dim // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        q_in = self.norm_q(x)
        kv_in = q_in if self.norm_kv is None else self.norm_kv(context)
        k, v = self.kv(kv_in).chunk(2, dim=2)
        attended = F.scaled_dot_product_attention(self._split(self.q(q_in)), self._split(k), self._split(v))
        x = x + self.out(attended.transpose(1, 2).reshape(x.shape))
        return x + self.ffn(self.norm_ffn(x))


class MazeTransformerQNet(nn.Module):
    """See the module docstring. Sized to cost about what MazeGraphQNet and QNetV2Wide do on
    the actors' CPUs."""

    DIM = 48
    HEADS = 4
    LATENTS = 32
    LAYERS = 3
    FFN = 96

    def __init__(self, obs_shape: torch.Size, action_count: int) -> None:
        super().__init__()
        dim = self.DIM
        self.reader = MazeReader()
        self.node_in = nn.Linear(NUM_NODE_FEATURES, dim)
        self.position = nn.Parameter(torch.randn(NUM_NODES, dim) * 0.1)
        self.register_buffer("buckets", distance_buckets(), persistent=False)
        self.from_pacman = nn.Embedding(len(DISTANCE_BUCKETS) + 1, dim)
        self.board_in = nn.Sequential(nn.Linear(NUM_GLOBAL_FEATURES, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.latents = nn.Parameter(torch.randn(self.LATENTS, dim) * 0.1)
        self.read = _Attend(dim, self.HEADS, self.FFN, cross=True)
        self.think = nn.ModuleList(_Attend(dim, self.HEADS, self.FFN, cross=False) for _ in range(self.LAYERS))
        self.write = _Attend(dim, self.HEADS, self.FFN, cross=True)
        self.norm = nn.LayerNorm(dim)
        self.head = MoveHead(dim, 2 * dim, action_count)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        nodes, whole, pacman = self.reader(obs)
        board = self.board_in(whole)  # (B, dim)
        squares = self.node_in(nodes) + self.position + self.from_pacman(self.buckets[pacman])  # (B, 288, dim)
        latents = self.latents.expand(obs.shape[0], -1, -1) + board[:, None]
        latents = self.read(latents, torch.cat([squares, board[:, None]], dim=1))
        for block in self.think:
            latents = block(latents)
        squares = self.norm(self.write(squares, latents))
        return self.head(squares, latents.mean(dim=1), pacman)
