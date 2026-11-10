# SPDX-License-Identifier: MIT OR Apache-2.0
"""The shared bot-match loop of the look and the gate (examples/alphazero_connect4)."""

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
# The example locks its ledger with fcntl, which exists only on POSIX.
pytest.importorskip("fcntl", reason="POSIX only: the example uses fcntl")
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


def test_botmatch_opening_index_is_its_index_in_the_file(monkeypatch):
    """For the 500 evaluation openings j runs 0..499, and the tie seed is
    TIE_BASE + j for both colours."""
    seen = []

    def fake(search, evaluator, bot, j, opening, first, tie_seed, seed):
        seen.append((j, opening, first, tie_seed, seed))
        return type("G", (), {"result": 1.0, "log": []})()

    monkeypatch.setattr(game, "versus_bot", fake)
    openings = game.read_openings(str(EXAMPLE.parents[1] / "compat" / "connect4" / "openings-heldout.txt")) * 3
    out = botmatch.play(None, None, openings[:500], 1, 1, 64)
    assert [s[0] for s in seen] == [j for j in range(500) for _ in (0, 1)]
    assert all(t == botmatch.TIE_BASE + j and o == openings[j] and seed == 20261102 for j, o, _, t, seed in seen)
    assert [s[2] for s in seen[:2]] == [True, False] and len(out["logs"]) == 1000


def test_botmatch_look_report_fields_are_unchanged(tmp_path):
    """The look's report keeps its fields and its eligibility rule."""
    import look

    c = tmp_path / "C.pt"
    assert train_step.main(["init", "--size", "tiny", "--seed", "1", "--out", str(c), "--source-sha", "t"]) == 0
    out = tmp_path / "o"
    out.mkdir()
    bot = bot_binary()
    import artifacts

    args = look.parse(["--checkpoint", str(c), "--bot", bot, "--bot-sha256", artifacts.sha256_file(bot),
                       "--out", str(out), "--source-sha", "t", "--ledger", str(tmp_path / "l"),
                       "--budget-core-hours", "1", "--wall-cap-s", "1", "--limit", "1", "--sims", "2",
                       "--batch-size", "2", "--capacity", "64"])  # fmt: skip
    assert look.work(args, None) == 0
    import json

    report = json.loads((out / "look.json").read_text())
    assert sorted(report) == sorted([
        "kind", "checkpoint", "generation", "games", "wins", "wdl", "by_colour", "pair_wins", "log_sha256",
        "runtime", "protocol", "eligible", "note", "provenance", "complete",
    ])  # fmt: skip
    assert sorted(report["protocol"]) == sorted([
        "checkpoint", "openings_sha256", "sims", "batch_size", "capacity", "openings", "search_seed", "tie_base",
        "bot_sha256", "size",
    ])  # fmt: skip
    assert report["eligible"] is False and report["games"] == 2
