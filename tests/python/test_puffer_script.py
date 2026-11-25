# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/puffer_test.py leaves nothing behind when it is killed."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "puffer_test.py"
RUNNER = """
import importlib.util, sys, tempfile
spec = importlib.util.spec_from_file_location("puffer_test", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.exit_on_signals()
child = "import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); print('PIDS', c.pid, flush=True); time.sleep(600)"
sys.exit(m.run_group([sys.executable, "-c", child], tempfile.gettempdir(), 600))
"""


@pytest.fixture
def spawned():
    """Kills, on teardown, every process group and pid a test registers, so a
    failing test leaves nothing behind (#655). Runners get their own session,
    so their group is theirs alone."""
    groups, pids = [], []
    yield groups, pids
    for g in groups:
        try:
            os.killpg(g.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        g.wait()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGINT", "SIGHUP"])
def test_a_killed_runner_leaves_no_child(name, spawned):
    sig = getattr(signal, name)
    p = subprocess.Popen(
        [sys.executable, "-c", RUNNER, str(SCRIPT)], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    spawned[0].append(p)
    pid = int(p.stdout.readline().split()[1])
    spawned[1].append(pid)
    assert alive(pid)
    p.send_signal(sig)
    assert p.wait(timeout=30) == 128 + sig
    deadline = time.monotonic() + 30
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


def fake_uvx(tmp_path, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uvx = bin_dir / "uvx"
    uvx.write_text("#!/bin/sh\n" + body)
    uvx.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_a_killed_runner_leaves_no_child_of_the_build(tmp_path, spawned):
    pid_file = tmp_path / "pid"
    env = fake_uvx(tmp_path, f"sleep 600 &\necho $! > {pid_file}\nwait\n")
    p = subprocess.Popen([sys.executable, str(SCRIPT)], env=env, start_new_session=True)
    spawned[0].append(p)
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, "the fake build never started"
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    spawned[1].append(pid)
    assert alive(pid)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 128 + signal.SIGTERM
    deadline = time.monotonic() + 30
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


@pytest.mark.skipif(os.name == "nt", reason="POSIX only: the fake uvx is a sh script")
def test_a_failing_build_fails_the_run(tmp_path):
    env = fake_uvx(tmp_path, "exit 7\n")
    assert subprocess.run([sys.executable, str(SCRIPT)], env=env, timeout=60, capture_output=True).returncode == 7


# The #655 race, made deterministic: the first Popen.wait takes the internal
# lock and raises SystemExit before releasing it, as a signal handler landing
# between CPython's acquire and its try does. run_group must still reap its
# child and exit; before the fix its final Popen.wait blocked forever.
LOCK_RACE = """
import importlib.util, os, subprocess, sys
spec = importlib.util.spec_from_file_location("puffer_test", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
real_wait = subprocess.Popen.wait
def racy_wait(self, timeout=None):
    if timeout is not None:
        print("PIDS", self.pid, flush=True)
        self._waitpid_lock.acquire(False)
        raise SystemExit(143)
    return real_wait(self, timeout)
subprocess.Popen.wait = racy_wait
try:
    m.run_group([sys.executable, "-c", "import time; time.sleep(600)"], ".", 600)
except SystemExit as e:
    print("EXIT", e.code, flush=True)
    raise
"""


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_run_group_lock_held_by_interrupted_wait_still_reaps_child(spawned):
    p = subprocess.Popen(
        [sys.executable, "-c", LOCK_RACE, str(SCRIPT)], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    spawned[0].append(p)
    child = int(p.stdout.readline().split()[1])
    spawned[1].append(child)
    assert p.wait(timeout=30) == 143, "run_group hung on the held _waitpid_lock"
    assert p.stdout.read().strip() == "EXIT 143"
    assert not alive(child)
