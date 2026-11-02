# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-M3 pilot harness (examples/alphazero_connect4): the mirror
augmentation, the network evaluator's wire format, the ledger's stop
rules, the bot line protocol and a tiny end-to-end pilot run."""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import augment  # noqa: E402
import game  # noqa: E402
import net as netmod  # noqa: E402
import pilot  # noqa: E402
from ledger import BudgetExceeded, Ledger  # noqa: E402

from tessera_rl.search import Search  # noqa: E402


def random_position(rng, plies):
    board = game.Board(seed=1)
    try:
        for _ in range(plies):
            legal = np.flatnonzero(board.step.masks[0, board.mover])
            board.play(int(rng.choice(legal)))
            if board.over:
                break
        return board.planes(), board.step.masks[0, board.mover].copy(), board.over
    finally:
        board.close()


def gravity_ok(planes):
    """No stone floats above an empty cell (row 0 is the top)."""
    filled = planes.sum(axis=0) > 0
    for col in range(7):
        column = filled[:, col]
        seen = False
        for row in range(6):
            if column[row]:
                seen = True
            elif seen:
                return False
    return True


def test_pilot_mirror_augmentation_is_legal_and_mirrors_policy():
    rng = np.random.default_rng(3)
    for plies in range(0, 20, 3):
        planes, legal, over = random_position(rng, plies)
        if over:
            continue
        policy = rng.dirichlet(np.ones(7))[None, :] * legal
        policy /= policy.sum()
        mp, mpol, mv = augment.mirror(planes[None], policy, np.array([0.25]))
        assert gravity_ok(mp[0])
        assert (mp[0] == planes[:, :, ::-1]).all()
        assert (mpol[0] == policy[0, ::-1]).all()
        # A mirrored legal column is legal in the mirrored position.
        assert ((mpol[0] > 0) <= (mp[0].sum(axis=0)[0] == 0)).all()
        assert mv.tolist() == [0.25]
    p, pol, v = augment.augment(np.zeros((3, 2, 6, 7)), np.full((3, 7), 1 / 7), np.zeros(3))
    assert p.shape == (6, 2, 6, 7) and pol.shape == (6, 7) and v.shape == (6,)


def test_pilot_evaluator_returns_the_search_wire_format():
    pilot.configure_torch(1)
    ev = netmod.Evaluator(netmod.build("tiny", seed=1))
    legal = np.ones((5, 7), dtype=bool)
    legal[:, 3] = False
    priors, values = ev(np.zeros((5, 2, 6, 7), dtype=np.uint8), legal, np.zeros(5, dtype=np.uint64))
    assert priors.dtype == np.float64 and priors.flags.c_contiguous and priors.shape == (5, 7)
    assert values.dtype == np.float64 and values.shape == (5,)
    assert (priors[:, 3] == 0).all() and np.allclose(priors.sum(axis=1), 1)
    assert (np.abs(values) <= 1).all() and ev.batches == [5]


def test_pilot_build_is_seeded_and_leaves_the_global_rng_alone():
    state = torch.random.get_rng_state()
    a, b = netmod.build("tiny", seed=9), netmod.build("tiny", seed=9)
    assert torch.equal(torch.random.get_rng_state(), state)
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)


def test_pilot_ledger_refuses_work_beyond_the_budget(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = Ledger(path, budget_core_hours=1.0, wall_cap_s=10_000, cores=4)
    led.check(600, "fits")  # 600 s x 4 cores = 0.667 core-hours
    with pytest.raises(BudgetExceeded):
        led.check(1200, "core-hours")  # 1.333 core-hours
    with pytest.raises(BudgetExceeded):
        Ledger(path, 100, wall_cap_s=5, cores=1).check(6, "wall")
    entry = led.close("run", "host", "failed: test")
    assert entry["status"] == "failed: test" and entry["cores"] == 4
    again = Ledger(path, budget_core_hours=1.0, wall_cap_s=10_000, cores=4)
    assert again.spent_core_hours() == entry["core_hours"]
    assert json.loads(pathlib.Path(path).read_text())["entries"][0]["run"] == "run"


def test_pilot_self_play_game_gives_mover_relative_targets():
    pilot.configure_torch(1)
    s = Search(simulations=8, batch_size=4, seed=5, noise_alpha=1.0, noise_epsilon=0.25)
    g = game.self_play(s, netmod.Uniform(), seed=5, temperature_moves=42)
    n = len(g.moves)
    assert g.planes.shape == (n, 2, 6, 7) and g.policy.shape == (n, 7)
    assert np.allclose(g.policy.sum(axis=1), 1)
    # Outcomes alternate in sign by mover, or are all 0 on a draw.
    if g.value[-1] != 0:
        assert all(g.value[i] == -g.value[i + 1] for i in range(n - 1))
        assert g.value[-1] == 1  # the last mover won
    assert g.timing.moves == n and sum(g.timing.batches) > 0


def bot_binary():
    path = os.environ.get("TESSERA_CONNECT4_REF")
    if path:
        return path
    root = pathlib.Path(__file__).parents[2] / "target"
    for p in (root / "release" / "tessera-rl-connect4-ref", root / "debug" / "tessera-rl-connect4-ref"):
        if p.exists():
            return str(p)
    return None


@pytest.fixture(scope="module")
def bot():
    path = bot_binary()
    if path is None:
        if shutil.which("cargo") is None:
            pytest.skip("no connect4-ref binary and no cargo")
        root = pathlib.Path(__file__).parents[2]
        subprocess.run(["cargo", "build", "-q", "-p", "tessera-rl-connect4-ref"], cwd=root, check=True)
        path = bot_binary()
    return path


def test_pilot_bot_serve_replies_a_legal_column_and_rejects_finished_games(bot):
    client = game.BotClient(bot)
    try:
        col = client.choose(pilot.TIE_BASE, "-")
        assert 0 <= col < 7
        assert client.choose(pilot.TIE_BASE, "4453") == client.choose(pilot.TIE_BASE, "4453")
        with pytest.raises(RuntimeError):
            client.choose(pilot.TIE_BASE, "1212121")
    finally:
        client.close()


def test_pilot_tiny_run_is_repeatable_and_writes_the_report(bot, tmp_path):
    out = tmp_path / "out"
    args = [
        "all",
        "--out", str(out),
        "--ledger", str(tmp_path / "ledger.json"),
        "--cores", "1",
        "--budget-core-hours", "1",
        "--wall-cap-s", "600",
        "--source-sha", "test",
        "--bot", bot,
        "--sizes", "tiny",
        "--sims", "4",
        "--batch-size", "4",
        "--capacity", "256",
        "--infer-batches", "1", "8",
        "--infer-inner", "2",
        "--train-batches", "16",
        "--warmup", "1",
        "--repeats", "2",
        "--games-per-worker", "1",
        "--selfplay-cell-s", "30",
        "--eval-size", "tiny",
        "--eval-sims", "4",
        "--eval-openings", "2",
        "--eval-cell-s", "30",
    ]  # fmt: skip
    assert pilot.main(args) == 0
    (report_path,) = out.glob("pilot-*.json")
    report = json.loads(report_path.read_text())
    assert report["provenance"]["threads"]["deterministic_algorithms"] is True
    assert report["provenance"]["threads"]["torch_intra_op"] == 1
    assert report["ledger"]["status"] == "ok" and report["ledger"]["cores"] == 1
    for name in ("heldout", "eval"):
        assert report["eval"]["passes"][name]["repeatable"] is True
        assert report["eval"]["passes"][name]["runs"][0]["games"] == 4
    assert {c["evaluator"] for c in report["selfplay"]["cells"]} == {"uniform", "net"}
    assert all(c["samples_s"] for c in report["infer"]["cells"])
