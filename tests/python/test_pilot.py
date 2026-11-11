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
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# The example locks its ledger with fcntl, which exists only on POSIX.
pytest.importorskip("fcntl", reason="POSIX only: the example uses fcntl")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import augment  # noqa: E402
import game  # noqa: E402
import net as netmod  # noqa: E402
import pilot  # noqa: E402
from ledger import BudgetExceeded, Ledger, LedgerBusy  # noqa: E402
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
    for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        assert torch.equal(x, y)


def test_pilot_ledger_reserves_before_work_and_settles_to_the_time_used(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = Ledger(path, budget_core_hours=1.0)
    i, granted = led.reserve("a", cores=4, deadline_s=600)  # 0.667 core-hours
    assert granted == 600
    assert json.loads(pathlib.Path(path).read_text())["entries"][0]["status"] == "running"
    # One run at a time per ledger file.
    with pytest.raises(LedgerBusy):
        led.reserve("b", cores=4, deadline_s=600)
    entry = led.settle(i, wall_s=600, status="ok")
    assert entry["core_hours"] == pytest.approx(600 * 4 / 3600, abs=1e-5)
    # The settled spend counts: only 0.333 core-hours are left.
    j, granted = led.reserve("b", cores=4, deadline_s=600)
    assert granted == pytest.approx(300)
    led.settle(j, wall_s=300, status="ok")
    with pytest.raises(BudgetExceeded):
        led.reserve("c", cores=4, deadline_s=10)


def test_pilot_ledger_settles_a_dead_supervisor_at_its_full_reservation(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = Ledger(path, budget_core_hours=10.0)
    led.reserve("lost", cores=2, deadline_s=1800)
    data = json.loads(pathlib.Path(path).read_text())
    data["entries"][0]["pid"] = 2**22 + 12345  # no such process
    pathlib.Path(path).write_text(json.dumps(data))
    (e,) = led.entries()
    assert e["status"].startswith("interrupted") and e["core_hours"] == pytest.approx(1.0)
    assert led.spent_core_hours() == pytest.approx(1.0)


def test_pilot_ledger_leaves_another_hosts_open_entry_and_refuses_a_second_run(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = Ledger(path, budget_core_hours=10.0)
    led.reserve("elsewhere", cores=2, deadline_s=600)
    data = json.loads(pathlib.Path(path).read_text())
    data["entries"][0].update(host="another-container", pid=2**22 + 12345)
    pathlib.Path(path).write_text(json.dumps(data))
    # Its pid cannot be checked from here: it stays open and blocks.
    assert led.entries()[0]["status"] == "running"
    with pytest.raises(LedgerBusy):
        led.reserve("second", cores=2, deadline_s=600)


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


def test_pilot_bot_serve_replies_a_legal_column_and_fails_on_a_finished_game(bot):
    client = game.BotClient(bot)
    try:
        col = client.choose(pilot.TIE_BASE, "-")
        assert 0 <= col < 7
        assert client.choose(pilot.TIE_BASE, "4453") == client.choose(pilot.TIE_BASE, "4453")
        with pytest.raises(RuntimeError, match="error finished"):
            client.choose(pilot.TIE_BASE, "1212121")
        # The server exits with a failure after an error line.
        assert client.proc.returncode == 1
    finally:
        client.close()


def tiny_args(tmp_path, bot, command="all", **over):
    opts = {
        "--out": str(tmp_path / "out"),
        "--ledger": str(tmp_path / "ledger.json"),
        "--cores": "1",
        "--budget-core-hours": "1",
        "--wall-cap-s": "120",
        "--source-sha": "test",
        "--bot": bot,
        "--sizes": "tiny",
        "--sims": "4",
        "--batch-size": "4",
        "--capacity": "256",
        "--infer-batches": "1",
        "--infer-inner": "1",
        "--train-batches": "8",
        "--warmup": "0",
        "--repeats": "1",
        "--games-per-worker": "1",
        "--cell-allowance-s": "1",
        "--eval-size": "tiny",
        "--eval-sims": "2",
        "--eval-openings": "1",
        "--eval-allowance-s": "1",
    }
    opts.update(over)
    return [command] + [x for k, v in opts.items() for x in (k, *str(v).split())]


def run_pilot(args, env=None):
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE / "pilot.py"), *args],
        capture_output=True,
        text=True,
        env={k: v for k, v in {**os.environ, **(env or {})}.items() if k != "PILOT_INHERITED_THREAD_VARS"},
        timeout=120,
    )
    return proc


def only_report(tmp_path):
    (path,) = (tmp_path / "out").glob("pilot-*.json")
    return json.loads(path.read_text())


def test_pilot_tiny_run_is_repeatable_and_writes_the_report(bot, tmp_path):
    proc = run_pilot(tiny_args(tmp_path, bot))
    assert proc.returncode == 0, proc.stderr
    report = only_report(tmp_path)
    assert report["status"] == "ok" and report["missing_sections"] == []
    threads = report["provenance"]["threads"]
    assert threads["deterministic_algorithms"] is True and threads["torch_intra_op"] == 1
    assert set(threads["env"].values()) == {"1"}
    assert report["provenance"]["bot_sha256"]
    assert report["ledger"]["status"] == "ok" and report["ledger"]["cores"] == 1
    for name in ("heldout", "eval"):
        verdict = report["eval"]["passes"][name]
        assert verdict["repeatable"] is True
        assert verdict["openings_played"] == 1 and verdict["games_per_pass"] == 2
        assert verdict["openings_in_file"] == {"heldout": 200, "eval": 500}[name]
    assert {c["evaluator"] for c in report["selfplay"]["cells"]} == {"uniform", "net"}
    assert all(c["samples_s"] for c in report["infer"]["cells"])


def test_pilot_inherited_thread_variables_are_overridden(bot, tmp_path):
    proc = run_pilot(tiny_args(tmp_path, bot, "train"), env={"OMP_NUM_THREADS": "8"})
    assert proc.returncode == 0, proc.stderr
    threads = only_report(tmp_path)["provenance"]["threads"]
    assert threads["env"]["OMP_NUM_THREADS"] == "1"
    assert threads["env_inherited"]["OMP_NUM_THREADS"] == "8"


def test_pilot_deadline_kills_running_work_and_charges_the_ledger(bot, tmp_path):
    # Far more self-play than any machine does in 4 s, and the child never
    # skips a cell, however long a loaded machine took to start it (exit 0 under
    # load otherwise): only the supervisor's kill can stop it.
    args = tiny_args(
        tmp_path, bot, "selfplay", **{"--wall-cap-s": "6", "--games-per-worker": "100000", "--sims": "64"}
    )
    start = time.monotonic()
    proc = run_pilot(args, env={"PILOT_TEST_NO_SKIP": "1"})
    wall = time.monotonic() - start
    assert proc.returncode == pilot.EXIT_DEADLINE, proc.stderr
    report = only_report(tmp_path)
    assert report["status"].startswith("killed at the deadline")
    assert report["missing_sections"] == ["selfplay"]
    (entry,) = json.loads((tmp_path / "ledger.json").read_text())["entries"]
    assert entry["status"].startswith("killed") and 0 < entry["wall_s"] <= 6
    # Loose on purpose: a late kill still shows, on however loaded a machine.
    assert wall < 60


def test_pilot_supervisor_kills_native_work_that_holds_the_gil_and_charges_it_unclamped(bot, tmp_path):
    # The child sleeps HOLD_S in C with the GIL held: its watchdog thread
    # never runs, so only the supervisor's own deadline can stop it.
    hold_s = 30
    ledger = tmp_path / "ledger.json"
    proc = subprocess.Popen(
        [sys.executable, str(EXAMPLE / "pilot.py"), *tiny_args(tmp_path, bot, "train", **{"--wall-cap-s": "4"})],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PILOT_TEST_HOLD_GIL_S": str(hold_s)},
    )
    try:
        # The clock starts when the supervisor reserves (the ledger appears
        # right before its deadline is armed), not at the interpreter's
        # start: torch's import time under load is not what is under test.
        armed = _wait_for(ledger.exists, 60)
        assert armed is not None, "the supervisor reserved"
        _, stderr = proc.communicate(timeout=hold_s + 60)
        wall = time.monotonic() - armed
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == pilot.EXIT_DEADLINE, stderr
    # A supervisor that never kills waits out the native work (hold_s);
    # one that does is done long before, however loaded the machine.
    assert wall < hold_s
    (entry,) = json.loads(ledger.read_text())["entries"]
    assert entry["status"].startswith("killed")
    # The charge is the measured wall, not a clamped value: the work
    # stopped at 4 - grace s, inside the 4 s reservation.
    assert 4 - pilot.CLEANUP_GRACE_S <= entry["wall_s"] <= 4
    assert "overran" not in entry["status"]


def test_pilot_ledger_charges_an_overrun_in_full(tmp_path):
    led = Ledger(str(tmp_path / "ledger.json"), budget_core_hours=1.0)
    i, _ = led.reserve("a", cores=1, deadline_s=10)
    entry = led.settle(i, wall_s=12.5, status="killed")
    assert entry["wall_s"] == 12.5 and "overran" in entry["status"]


def test_pilot_sigkilled_supervisor_takes_its_workers_and_its_reservation(bot, tmp_path):
    args = tiny_args(tmp_path, bot, "selfplay", **{"--wall-cap-s": "60", "--games-per-worker": "100000", "--sims": "64"})
    sup = subprocess.Popen([sys.executable, str(EXAMPLE / "pilot.py"), *args], stdout=subprocess.PIPE, text=True)
    ledger = tmp_path / "ledger.json"
    try:
        # Ready when the child runs its self-play workers: then there are
        # workers to take, however long a loaded machine took to start them.
        assert _wait_for(lambda: _children(sup.pid), 60), "the supervisor started its child"
        (child_pid,) = _children(sup.pid)
        assert _wait_for(lambda: _children(child_pid), 60), "the child started its workers"
        pids = [child_pid, *_children(child_pid)]
    finally:
        sup.kill()
        sup.wait()
    _assert_gone_long_before_the_deadline(pids)
    _assert_charged_once_at_the_full_reservation(ledger, 60)


def test_pilot_supervisor_killed_while_its_child_starts_takes_the_child_at_once(bot, tmp_path):
    # The test-only start gate (start_gate/sitecustomize.py) holds the child
    # before any line of pilot.py runs, so before it can check its
    # supervisor: the supervisor dies inside that window deterministically.
    gate = tmp_path / "gate"
    env = start_gate_env(gate)
    args = tiny_args(tmp_path, bot, "selfplay", **{"--wall-cap-s": "60", "--games-per-worker": "100000", "--sims": "64"})
    sup = subprocess.Popen([sys.executable, str(EXAMPLE / "pilot.py"), *args], env=env)
    try:
        held = gate.with_suffix(".held")
        assert _wait_for(held.exists, 60), "the child reached the start gate"
        child_pid = int(held.read_text())
    finally:
        sup.kill()
        sup.wait()
    gate.write_text("")
    # Without the check the child runs on to the work deadline (58 s).
    _assert_gone_long_before_the_deadline([child_pid])
    _assert_charged_once_at_the_full_reservation(tmp_path / "ledger.json", 60)


START_GATE = pathlib.Path(__file__).with_name("start_gate")


def start_gate_env(gate):
    path = os.pathsep.join(filter(None, [str(START_GATE), os.environ.get("PYTHONPATH")]))
    env = {**os.environ, "PYTHONPATH": path, "TESSERA_TEST_START_GATE": str(gate)}
    env.pop("PILOT_INHERITED_THREAD_VARS", None)
    return env


# The child sees its supervisor gone within one watchdog poll; the bound is
# many polls on purpose (a loaded machine delays the SIGKILLed processes'
# exit), and still far below the work deadline at which an unguarded child
# would stop (wall cap 60 s minus the cleanup grace).
GONE_BOUND_S = 50 * 0.2  # 50 polls of the watchdog (orphan.POLL_S)


def _assert_gone_long_before_the_deadline(pids):
    assert GONE_BOUND_S < 60 - pilot.CLEANUP_GRACE_S
    gone = _wait_for(lambda: not any(_exists(p) for p in pids), GONE_BOUND_S)
    alive = [p for p in pids if _exists(p)]
    for p in alive:  # never leave them running into the next test
        try:
            os.killpg(p, 9) if p == pids[0] else os.kill(p, 9)
        except (ProcessLookupError, PermissionError):
            pass
    assert gone is not None, f"still alive {GONE_BOUND_S} s after the supervisor died: {alive}"


def _assert_charged_once_at_the_full_reservation(ledger, wall_cap_s):
    # The dead supervisor never settled: its entry is charged once, at the
    # full reservation, exactly as on the watchdog's normal path.
    led = Ledger(str(ledger), budget_core_hours=1)
    (entry,) = led.entries()
    assert entry["status"].startswith("interrupted")
    assert entry["core_hours"] == pytest.approx(wall_cap_s / 3600, abs=1e-5)
    assert led.spent_core_hours() == pytest.approx(wall_cap_s / 3600, abs=1e-5)


def _wait_for(condition, bound_s):
    """Polls ``condition`` until it holds or ``bound_s`` passes; returns the
    monotonic time it first held, or None."""
    end = time.monotonic() + bound_s
    while True:
        if condition():
            return time.monotonic()
        if time.monotonic() >= end:
            return None
        time.sleep(0.05)


def _children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def _exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); it holds no resources.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_pilot_divergent_evaluation_logs_fail_the_run(bot, tmp_path):
    proc = run_pilot(tiny_args(tmp_path, bot, "eval"), env={"PILOT_TEST_PERTURB_EVAL": "1"})
    assert proc.returncode == pilot.EXIT_NOT_REPEATABLE, proc.stderr
    report = only_report(tmp_path)
    assert report["status"] == "failed: evaluation not repeatable"
    verdict = report["eval"]["passes"]["heldout"]
    assert verdict["repeatable"] is False and verdict["first_differing_game"] == 2
    assert (tmp_path / "out" / verdict["runs"][0]["log_file"]).exists()
    (entry,) = json.loads((tmp_path / "ledger.json").read_text())["entries"]
    assert entry["status"] == "failed: evaluation not repeatable"


def test_pilot_per_allocated_core_rate_divides_by_allocated_cores(bot, tmp_path):
    if (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()) < 2:
        pytest.skip("needs 2 cores")
    proc = run_pilot(tiny_args(tmp_path, bot, "selfplay", **{"--cores": "2", "--sizes": "tiny"}))
    assert proc.returncode == 0, proc.stderr
    for c in only_report(tmp_path)["selfplay"]["cells"]:
        assert c["allocated_cores"] == 2
        per_core = c["steady_games_per_hour_per_allocated_core"]["median"]
        assert per_core == pytest.approx(c["steady_games_per_hour"]["median"] / 2)
        assert c["workers_seen_ready"] == c["workers"]


def test_pilot_tie_base_matches_the_rust_constant():
    text = (pathlib.Path(__file__).parents[2] / "tools" / "connect4-ref" / "src" / "calibrate.rs").read_text()
    (line,) = [x for x in text.splitlines() if x.startswith("pub const TIE_BASE")]
    value = int(line.split("=")[1].strip().rstrip(";").replace("_", ""), 16)
    assert value == pilot.TIE_BASE


def test_pilot_rejects_non_positive_budgets_and_too_many_cores(tmp_path):
    base = ["train", "--out", str(tmp_path), "--ledger", str(tmp_path / "l.json"), "--source-sha", "x"]
    for bad in (
        ["--cores", "1", "--budget-core-hours", "0", "--wall-cap-s", "10"],
        ["--cores", "1", "--budget-core-hours", "nan", "--wall-cap-s", "10"],
        ["--cores", "1", "--budget-core-hours", "1", "--wall-cap-s", "-1"],
        ["--cores", "100000", "--budget-core-hours", "1", "--wall-cap-s", "10"],
    ):
        with pytest.raises(SystemExit):
            pilot.parse(base + bad)


def test_pilot_ledger_settle_charges_a_lost_runs_full_reservation(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = Ledger(path, budget_core_hours=10.0)
    led.reserve("lost container", cores=4, deadline_s=900)
    data = json.loads(pathlib.Path(path).read_text())
    data["entries"][0].update(host="lost-container")
    pathlib.Path(path).write_text(json.dumps(data))
    with pytest.raises(LedgerBusy):
        led.reserve("next", cores=4, deadline_s=60)
    proc = run_pilot(["ledger-settle", "--ledger", path, "--index", "0", "--interrupted"])
    assert proc.returncode == 0, proc.stderr
    (e,) = led.entries()
    assert e["status"].startswith("interrupted (settled by hand") and e["core_hours"] == pytest.approx(1.0)
    # Unblocked, and settling a closed entry again is refused.
    led.reserve("next", cores=4, deadline_s=60)
    assert run_pilot(["ledger-settle", "--ledger", path, "--index", "0", "--interrupted"]).returncode != 0
    assert run_pilot(["ledger-settle", "--ledger", path, "--index", "0"]).returncode != 0
