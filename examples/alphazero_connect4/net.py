# SPDX-License-Identifier: MIT OR Apache-2.0
"""The policy and value network of the AlphaZero Connect Four example, and
the evaluator that serves it to ``tessera_rl.search.Search``.

The input is the mover's two planes (own stones, opponent stones), shape
``(n, 2, 6, 7)``. The policy head gives one logit per column; the value
head a scalar in ``[-1, 1]`` from the mover's view, as the search expects.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

# Named sizes: (residual blocks, channels). "tiny" is for tests only.
SIZES: dict[str, tuple[int, int]] = {
    "tiny": (1, 8),
    "small": (4, 32),
    "medium": (6, 64),
    "large": (10, 128),
}

ROWS, COLS = 6, 7


class Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(x + self.b2(self.c2(y)))


class Net(nn.Module):
    """A residual tower with a policy head and a value head."""

    def __init__(self, blocks: int, channels: int):
        super().__init__()
        self.blocks = blocks
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU()
        )
        self.tower = nn.Sequential(*(Block(channels) for _ in range(blocks)))
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * ROWS * COLS, COLS),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(ROWS * COLS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.tower(self.stem(x))
        return self.policy(h), self.value(h).squeeze(1)


def build(size: str, seed: int) -> Net:
    """A freshly initialized network of a named size, seeded explicitly."""
    blocks, channels = SIZES[size]
    gen_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        net = Net(blocks, channels)
    finally:
        torch.random.set_rng_state(gen_state)
    return net.eval()


def parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def masked_priors(logits: torch.Tensor, legal: np.ndarray) -> np.ndarray:
    """Softmax over the legal columns; illegal columns get exactly 0."""
    mask = torch.from_numpy(np.array(legal, dtype=np.bool_))
    logits = logits.double().masked_fill(~mask, float("-inf"))
    return np.ascontiguousarray(torch.softmax(logits, dim=1).numpy(), dtype=np.float64)


class Evaluator:
    """Serves ``net`` to the search, timing the network section and
    recording the size of every leaf batch.

    ``net_ns`` covers tensor conversion, the forward pass and the masked
    softmax; it is disjoint from the search and boundary time outside the
    call, so ``run time - net_ns`` is the search plus the boundary."""

    def __init__(self, net: nn.Module):
        self.net = net
        self.net_ns = 0
        self.batches: list[int] = []

    def __call__(self, obs: np.ndarray, legal: np.ndarray, hashes: np.ndarray):
        start = time.perf_counter_ns()
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
            logits, values = self.net(x)
            priors = masked_priors(logits, legal)
            v = np.ascontiguousarray(values.double().numpy(), dtype=np.float64)
        self.net_ns += time.perf_counter_ns() - start
        self.batches.append(len(hashes))
        return priors, v


class Uniform:
    """A canned evaluator: uniform priors, zero values. Timing with it is
    the end-to-end canned search (engine plus the Python boundary), not the
    pure engine cost."""

    def __init__(self):
        self.net_ns = 0
        self.batches: list[int] = []

    def __call__(self, obs, legal, hashes):
        n = len(hashes)
        self.batches.append(n)
        return np.full((n, COLS), 1.0 / COLS), np.zeros(n)
