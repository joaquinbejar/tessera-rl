# SPDX-License-Identifier: MIT OR Apache-2.0
"""Games against the ADR-0042 bot, shared by the held-out look (PR 3)
and the gate (PR 4), so both play the same protocol by construction.

Each opening ``j`` is played twice, the candidate moving first and then
second, with:
- a fresh single-tree ``Search`` per game, greedy, no noise, search seed
  ``SEARCH_SEED``;
- the bot's tie seed ``TIE_BASE + j``.

Wins are wins; draws are non-wins.
"""

from __future__ import annotations

import torch

import game
from tessera_rl.search import Search

TIE_BASE = 0x7E55E2A000000000
SEARCH_SEED = 20261102


def play(evaluator, bot, openings: list[str], sims: int, batch_size: int, capacity: int) -> dict:
    """Every opening, both colours. Returns the per-game results (the
    candidate's view), the canonical log entries ``[j, first, result,
    log]``, the per-pair win counts and the W/D/L per colour."""
    results, logs, pairs = [], [], []
    by_colour = {"candidate_first": [0, 0, 0], "bot_first": [0, 0, 0]}
    with torch.inference_mode():
        for j, opening in enumerate(openings):
            pair = 0
            for first in (True, False):
                search = Search(simulations=sims, batch_size=batch_size, capacity=capacity, seed=SEARCH_SEED)
                g = game.versus_bot(search, evaluator, bot, j, opening, first, TIE_BASE + j, seed=SEARCH_SEED)
                key = "candidate_first" if first else "bot_first"
                by_colour[key][0 if g.result > 0 else 1 if g.result == 0 else 2] += 1
                pair += 1 if g.result > 0 else 0
                results.append(g.result)
                logs.append([j, first, g.result, g.log])
            pairs.append(pair)
    return {"results": results, "logs": logs, "pairs": pairs, "by_colour": by_colour}
