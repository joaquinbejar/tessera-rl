# SPDX-License-Identifier: MIT OR Apache-2.0
"""The shared bot-match loop of the look and the gate (examples/alphazero_connect4)."""

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import botmatch  # noqa: E402
import ckpt  # noqa: E402
import game  # noqa: E402
import net as netmod  # noqa: E402
import train_step  # noqa: E402
from tessera_rl.search import Search  # noqa: E402
from test_loop import bot_binary  # noqa: E402


def reference(evaluator, bot, openings, sims, batch, capacity):
    """The look's loop as it was before the refactor (PR 3, 28b9828)."""
    results, logs, pairs = [], [], []
    by_colour = {"candidate_first": [0, 0, 0], "bot_first": [0, 0, 0]}
    with torch.inference_mode():
        for j, opening in enumerate(openings):
            pair = 0
            for first in (True, False):
                search = Search(simulations=sims, batch_size=batch, capacity=capacity, seed=20261102)
                g = game.versus_bot(search, evaluator, bot, j, opening, first, 0x7E55E2A000000000 + j, seed=20261102)
                key = "candidate_first" if first else "bot_first"
                by_colour[key][0 if g.result > 0 else 1 if g.result == 0 else 2] += 1
                win = 1 if g.result > 0 else 0
                pair += win
                results.append(g.result)
                logs.append([j, first, g.result, g.log])
            pairs.append(pair)
    return {"results": results, "logs": logs, "pairs": pairs, "by_colour": by_colour}


def test_botmatch_play_matches_the_pre_refactor_look_loop(tmp_path):
    c = tmp_path / "C.pt"
    assert train_step.main(["init", "--size", "tiny", "--seed", "1", "--out", str(c), "--source-sha", "t"]) == 0
    model, _ = ckpt.load(str(c))
    openings = game.read_openings(str(EXAMPLE.parents[1] / "compat" / "connect4" / "openings-heldout.txt"))[:3]
    out = []
    for fn in (reference, botmatch.play):
        bot = game.BotClient(bot_binary())
        try:
            out.append(fn(netmod.Evaluator(model), bot, openings, 4, 2, 256))
        finally:
            bot.close()
    assert out[0] == out[1]
    assert len(out[1]["logs"]) == 6
