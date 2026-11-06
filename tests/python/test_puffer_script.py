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


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_killed_runner_leaves_no_child(sig):
    p = subprocess.Popen([sys.executable, "-c", RUNNER, str(SCRIPT)], stdout=subprocess.PIPE, text=True)
    pid = int(p.stdout.readline().split()[1])
    assert alive(pid)
    p.send_signal(sig)
    assert p.wait(timeout=30) == 128 + sig
    deadline = time.monotonic() + 30
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)
