# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run safety for the P2-M3 loop and its transport (#154).

Covered:
- the STOP file: no attempt is allocated and nothing is dispatched once
  it exists; reconcile stops a running container by its recorded name;
- recorded identities: `loop.py stop` signals only a process whose PID,
  PGID and command match the record;
- `max_attempts` per profile in config.json, default 3;
- the phase container: --rm, run as the host user, HOME and the uv dirs
  under the mount, no chown, and the exit in a status record;
- a self-play pool worker that aborts fails the attempt at once (no
  respawn, no hang);
- M3 and single-host adr-0052 roots keep their argv and config.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fcntl", reason="POSIX only: the example uses fcntl")
pytestmark = pytest.mark.torch

HERE = pathlib.Path(__file__).parent
EXAMPLE = HERE.parents[1] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(HERE))

import artifacts  # noqa: E402
import loop  # noqa: E402
import remote  # noqa: E402
import remote_fake  # noqa: E402
import selfplay  # noqa: E402
import stopping  # noqa: E402
from test_loop import bot_binary, make_root, run_loop  # noqa: E402
from test_remote import cname, dispatches, drive, env_for, remote_root, runs, wait_for  # noqa: E402
from test_selfplay_baseline import make_c0, run, selfplay_args  # noqa: E402


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


# ---------------------------------------------------------------- STOP


def test_loop_stop_file_refuses_a_new_attempt(tmp_path, bot):
    root = make_root(tmp_path, bot)
    stopping.request(str(root), "test")
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["stop"] == "stopped"
    assert loop.Loop(str(root)).ledger.entries() == []
    assert not list(root.glob("gen-*/*.command.json")), "nothing allocated"


def test_loop_stop_file_written_during_a_step_refuses_the_phase(tmp_path, bot):
    """The STOP appears after the step's own check: run_phase still refuses
    before writing the command or starting anything."""
    root = make_root(tmp_path, bot)
    lp = loop.Loop(str(root))
    gens = lp.chain(lp.ledger.entries())
    stopping.request(str(root), "test")
    with pytest.raises(loop.Stop) as stop:
        lp.run_phase(gens, 0, "selfplay", 0)
    assert stop.value.reason == "stopped" and not (root / "gen-000" / "selfplay-0.command.json").exists()


def test_remote_stop_file_refuses_dispatch_before_any_record(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    stopping.request(str(root), "test")

    class Args:
        pass

    args = Args()
    args.root, args.host, args.cpuset, args.source_tree_sha256 = str(root), "h", "0", "x"
    with pytest.raises(remote.TransportError, match="STOP"):
        remote.dispatch(args, ["selfplay.py", "--out", str(root / "gen-000" / "selfplay-0")])
    assert not list(root.rglob("*.dispatch.json"))


def test_remote_stop_during_push_never_starts_the_container(tmp_path, bot, monkeypatch):
    root = remote_root(tmp_path, bot)
    calls = []
    real_push = remote_fake.FakeHost.push

    def push(self, local, rel):
        real_push(self, local, rel)
        stopping.request(str(root), "test")  # requested mid-push

    monkeypatch.setattr(remote_fake.FakeHost, "push", push)
    monkeypatch.setattr(remote_fake.FakeHost, "docker_run", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("REMOTE_TEST_TRANSPORT", "remote_fake:factory")
    monkeypatch.setenv("REMOTE_TEST_PATH", str(HERE))
    monkeypatch.setenv("REMOTE_TEST_HOSTS", str(tmp_path / "hosts"))
    lp = loop.Loop(str(root))
    argv = lp.phase_argv(lp.chain(lp.ledger.entries()), 0, "selfplay", 0)
    (root / "gen-000").mkdir()
    tree = json.loads((root / "config.json").read_text())["runners"]["selfplay"].split("--source-tree-sha256 ")[1].split()[0]

    class Args:
        pass

    args = Args()
    args.root, args.host, args.cpuset, args.source_tree_sha256 = str(root), "swarmer02", "0", tree
    with pytest.raises(remote.TransportError, match="before the container started"):
        remote.dispatch(args, [str(EXAMPLE / "selfplay.py"), *argv])
    assert calls == []
    (d,) = dispatches(root)
    assert d["started"] is False and d["runner"]["pid"] == os.getpid()


def test_remote_reconcile_stops_a_running_container_by_recorded_name(tmp_path, bot):
    """The driver dies with self-play running; after STOP, reconcile stops
    that container by its recorded name, never by pattern, and settles it."""
    root = remote_root(tmp_path, bot, **{"--games": 400})
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    name = cname(root, "g000-selfplay-0")
    os.environ["REMOTE_TEST_HOSTS"] = str(tmp_path / "hosts")
    try:
        host = remote_fake.FakeHost("swarmer02")
        rl = tmp_path / "hosts" / "swarmer02" / "root" / "ledger.json"
        wait_for(lambda: rl.exists() and any(e["status"] == "running" for e in json.loads(rl.read_text())["entries"]))
        assert host.docker_state(name) == ("running", None)
        stopping.request(str(root), "test")
        p = subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "reconcile", "--root", str(root)],
                           env=env_for(tmp_path), capture_output=True, text=True, timeout=120)  # fmt: skip
        assert p.returncode == 0, p.stderr
        assert host.docker_state(name) is None and host.docker_ps_by_name(name) == ""
    finally:
        del os.environ["REMOTE_TEST_HOSTS"]
    (d,) = [d for d in dispatches(root) if d["attempt"] == "g000-selfplay-0"]
    assert d["stopped_by_stop_file"] and d["closed"] and d["container"] == name and d["exit_code"] == 137
    (a, s) = runs(root)[0]
    assert a == "g000-selfplay-0" and s.startswith("interrupted")


# ---------------------------------------------------------------- identities


def sleeper(argv_tail, new_group):
    """A process whose command line ends with ``argv_tail``."""
    code = "import time; time.sleep(60)"
    return subprocess.Popen([sys.executable, "-c", code, *argv_tail], start_new_session=new_group)


def record_of(p, argv_tail):
    return {"pid": p.pid, "pgid": os.getpgid(p.pid), "argv": ["-c", "import time; time.sleep(60)", *argv_tail]}


def test_stop_checks_recorded_pid_pgid_and_command():
    p = sleeper(["loop.py", "run", "--root", "/r"], new_group=True)
    try:
        rec = record_of(p, ["loop.py", "run", "--root", "/r"])
        time.sleep(0.3)
        assert stopping.matches(rec, stopping.observed(p.pid))
        assert stopping.kill_recorded({**rec, "pgid": rec["pgid"] + 1}, group=True) == "refused"
        assert stopping.kill_recorded({**rec, "argv": [*rec["argv"][:-1], "/other"]}, group=True) == "refused"
        assert p.poll() is None, "a mismatch signals nothing"
        assert stopping.kill_recorded(rec, group=True) == "signalled"
        assert p.wait(timeout=10) == -signal.SIGTERM
        assert stopping.kill_recorded(rec, group=True) == "gone"
    finally:
        p.kill()


def test_stop_never_signals_a_group_the_recorded_process_does_not_lead():
    p = sleeper(["remote.py", "run"], new_group=False)  # in pytest's group
    try:
        rec = record_of(p, ["remote.py", "run"])
        time.sleep(0.3)
        assert stopping.kill_recorded(rec, group=True) == "refused"
        assert p.poll() is None
    finally:
        p.kill()


def test_loop_stop_writes_stop_then_signals_recorded_driver_only(tmp_path, bot):
    root = make_root(tmp_path, bot)
    p = sleeper(["loop.py", "run", "--root", str(root)], new_group=True)
    other = sleeper(["loop.py", "run", "--root", str(root)], new_group=True)  # same command, not recorded
    try:
        time.sleep(0.3)
        rec = record_of(p, ["loop.py", "run", "--root", str(root)])
        (root / stopping.DRIVER).write_text(json.dumps(rec))
        assert loop.main(["stop", "--root", str(root)]) == 0
        assert stopping.requested(str(root))
        assert p.wait(timeout=10) == -signal.SIGTERM
        assert other.poll() is None, "an unrecorded process with the same command is untouched"
    finally:
        p.kill()
        other.kill()


def test_loop_run_records_its_identity_and_leads_its_group(tmp_path, bot):
    root = make_root(tmp_path, bot)
    stopping.request(str(root), "test")  # returns at once
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    rec = json.loads((root / stopping.DRIVER).read_text())
    assert rec["pid"] == rec["pgid"] and rec["argv"][-3:] == ["run", "--root", str(root)]


# ---------------------------------------------------------------- attempts


def test_loop_max_attempts_default_three_and_absent_from_config(tmp_path, bot):
    root = make_root(tmp_path, bot)
    cfg = json.loads((root / "config.json").read_text())
    assert "max_attempts" not in cfg and loop.max_attempts(cfg) == loop.MAX_ATTEMPTS == 3


def test_loop_max_attempts_one_stops_failed_after_one_attempt(tmp_path, bot):
    root = make_root(tmp_path, bot, **{"--max-attempts": 1})
    assert json.loads((root / "config.json").read_text())["max_attempts"] == 1
    lp = loop.Loop(str(root))
    # One failed self-play attempt on record: its directory and command.
    d = lp.attempt_dir(0, "selfplay", 0)
    os.makedirs(d)
    artifacts.write_atomic(d + ".command.json", b"[]")
    with pytest.raises(loop.Stop) as stop:
        lp.step()
    assert stop.value.reason == "failed" and stop.value.inputs["attempts"] == 1


def test_loop_max_attempts_out_of_range_is_refused(tmp_path, bot):
    with pytest.raises(SystemExit):
        make_root(tmp_path, bot, **{"--max-attempts": 4})


# ---------------------------------------------------------------- container


def run_words():
    return remote.docker_run_argv("/home/u/b", "tessera-m3-x-g000-selfplay-0", "0-31", "swarmer02",
                                  ["/work/tessera/examples/alphazero_connect4/selfplay.py", "--x"], 1800)  # fmt: skip


def test_remote_container_is_rm_and_runs_as_the_host_user():
    words = run_words()
    cmd = " ".join(words)
    assert words[:4] == ["docker", "run", "-d", "--rm"]
    assert words[words.index("--user") + 1] == "$(id -u):$(id -g)", "expanded by the host shell"
    for env in ("HOME=/work/tessera/.home", "UV_CACHE_DIR=/work/tessera/.uvcache",
                "UV_PYTHON_INSTALL_DIR=/work/tessera/.uvpython", "CARGO_HOME=/work/tessera/.cargo"):  # fmt: skip
        assert env in words, env
    assert "chown" not in cmd and "HOST_UID" not in cmd
    inner = shlex.split(cmd.replace("$(id -u):$(id -g)", "U"))[-1]
    assert "/m3root/_status/tessera-m3-x-g000-selfplay-0.json" in inner and "exit $rc" in inner


def test_remote_status_record_shell_writes_the_exit_code(tmp_path):
    words = run_words()
    inner = shlex.split(" ".join(words).replace("$(id -u):$(id -g)", "U"))[-1]
    phase = "uv run --locked --python 3.11 --group test --group torch python /work/tessera/examples/alphazero_connect4/selfplay.py --x"
    inner = inner.replace("/m3root", str(tmp_path)).replace(f"timeout --kill-after 30 1800 sh -c {shlex.quote(phase)}",
                                                             "sh -c 'echo out; exit 3'")  # fmt: skip
    (tmp_path / remote.STATUS).mkdir()
    r = subprocess.run(["sh", "-c", inner])
    assert r.returncode == 3
    assert json.loads((tmp_path / remote.STATUS / "tessera-m3-x-g000-selfplay-0.json").read_text()) == {"exit_code": 3}
    assert (tmp_path / remote.STATUS / "tessera-m3-x-g000-selfplay-0.log").read_text() == "out\n"


def test_remote_ssh_reads_exit_from_status_when_container_is_gone(monkeypatch):
    def fake(self, cmd, check=True):
        if cmd.startswith("docker inspect"):
            return subprocess.CompletedProcess(cmd, 1, "", "Error: No such object: n")
        if cmd.startswith("cat ") and cmd.endswith("n.json"):
            return subprocess.CompletedProcess(cmd, 0, '{"exit_code": 6}\n', "")
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(remote.SSH, "_ssh", fake)
    assert remote.SSH("h").docker_state("n") == ("exited", 6)


def test_remote_phase_killed_mid_run_leaves_tree_removable_by_host_user(tmp_path, bot):
    """Mid-phase kill on the stand-in host: every file the phase and the
    transport left is owned by this user, and removing the trees as this
    user succeeds. With a real container, `--user` is what guarantees it
    (test_remote_container_is_rm_and_runs_as_the_host_user)."""
    root = remote_root(tmp_path, bot, **{"--games": 400})
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    name = cname(root, "g000-selfplay-0")
    os.environ["REMOTE_TEST_HOSTS"] = str(tmp_path / "hosts")
    try:
        host = remote_fake.FakeHost("swarmer02")
        wait_for(lambda: host.docker_state(name) == ("running", None))
        time.sleep(1.0)
        host.docker_kill(name)
    finally:
        del os.environ["REMOTE_TEST_HOSTS"]
    base = tmp_path / "hosts" / "swarmer02"
    uid = os.getuid()
    assert all(p.lstat().st_uid == uid for p in base.rglob("*"))
    import shutil

    shutil.rmtree(base)
    assert not base.exists()


# ---------------------------------------------------------------- worker death


def test_selfplay_worker_abort_fails_fast_without_manifest(tmp_path):
    c0 = make_c0(tmp_path)
    out = tmp_path / "died"
    t0 = time.monotonic()
    r = run("selfplay.py", *selfplay_args(tmp_path, c0, out, 2, games=8), env={"SELFPLAY_TEST_ABORT_GAME": "3"})
    assert time.monotonic() - t0 < 100, "no hang"
    assert r.returncode == selfplay.EXIT_WORKER_DIED, (r.returncode, r.stderr)
    with pytest.raises(artifacts.Incomplete):
        artifacts.load_manifest(str(out))
    (entry,) = json.loads((tmp_path / "ledger-died.json").read_text())["entries"]
    assert entry["status"] == "failed: a self-play worker died"
    death = json.loads((out / "worker-death.json").read_text())
    assert -signal.SIGABRT in death["died"].values(), death
    assert f"{-signal.SIGABRT}" in r.stderr


def test_selfplay_without_abort_hook_is_unchanged(tmp_path):
    c0 = make_c0(tmp_path)
    r = run("selfplay.py", *selfplay_args(tmp_path, c0, tmp_path / "ok", 2, games=8))
    assert r.returncode == 0, r.stderr
    manifest, _ = artifacts.load_manifest(str(tmp_path / "ok"))
    assert manifest["games"] == 8


# ---------------------------------------------------------------- regression


def test_loop_m3_root_argv_and_config_unchanged(tmp_path, bot):
    root = make_root(tmp_path, bot)
    cfg = json.loads((root / "config.json").read_text())
    for key in ("max_attempts",):
        assert key not in cfg
    lp = loop.Loop(str(root))
    argv = lp.phase_argv(lp.chain(lp.ledger.entries()), 0, "selfplay", 0)
    assert argv[argv.index("--sims") + 1] == "2" and "--max-attempts" not in argv
    assert loop.max_attempts(dict(loop.FORMAL)) == 3 and loop.is_formal(dict(loop.FORMAL))
