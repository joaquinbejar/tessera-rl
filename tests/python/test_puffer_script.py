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


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
@pytest.mark.parametrize("name", ["SIGTERM", "SIGINT", "SIGHUP"])
def test_a_killed_runner_leaves_no_child(name):
    sig = getattr(signal, name)
    p = subprocess.Popen([sys.executable, "-c", RUNNER, str(SCRIPT)], stdout=subprocess.PIPE, text=True)
    pid = int(p.stdout.readline().split()[1])
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
def test_a_killed_runner_leaves_no_child_of_the_build(tmp_path):
    pid_file = tmp_path / "pid"
    env = fake_uvx(tmp_path, f"sleep 600 &\necho $! > {pid_file}\nwait\n")
    p = subprocess.Popen([sys.executable, str(SCRIPT)], env=env)
    deadline = time.monotonic() + 30
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, "the fake build never started"
        time.sleep(0.05)
    pid = int(pid_file.read_text())
    assert alive(pid)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 128 + signal.SIGTERM
    deadline = time.monotonic() + 30
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


def test_a_failing_build_fails_the_run(tmp_path):
    env = fake_uvx(tmp_path, "exit 7\n")
    assert subprocess.run([sys.executable, str(SCRIPT)], env=env, timeout=60, capture_output=True).returncode == 7
