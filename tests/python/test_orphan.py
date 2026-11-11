# SPDX-License-Identifier: MIT OR Apache-2.0
"""The supervised child's orphan check and the shared group kill
(examples/alphazero_connect4/orphan.py; issues #207 and #202)."""

import ast
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import orphan  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")

# The scripts a supervisor runs as its child, and the modules whose import
# is heavy: the early check must come before the first of them.
TOOLS = ("gate.py", "group_bench.py", "h2h.py", "look.py", "pilot.py", "selfplay.py", "train_step.py")
STDLIB_OK = {"__future__", "os", "sys", "signal", "orphan"}


def _is_early_check(node) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and ast.unparse(s.value.func) == "orphan.check_argv"
            for s in node.body
        )
    )


@pytest.mark.parametrize("tool", TOOLS)
def test_orphan_early_check_comes_before_every_heavy_import(tool):
    """#207, part 1, statically: in each supervised tool the
    ``if __name__ == "__main__": orphan.check_argv()`` block precedes every
    module-level import other than a few standard modules, so a supervisor
    that died during start-up is seen before torch or the native module
    load."""
    tree = ast.parse((EXAMPLE / tool).read_text(encoding="utf-8"))
    seen_check = False
    for node in tree.body:
        if _is_early_check(node):
            seen_check = True
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            heavy = [n for n in names if n.split(".")[0] not in STDLIB_OK]
            assert seen_check or not heavy, f"{tool}: {heavy} imported before the early orphan check"
    assert seen_check, f"{tool} has no early orphan check"


EARLY_CHILD = """
import pathlib, sys
sys.path.insert(0, {example!r})
import orphan
if __name__ == "__main__":
    orphan.check_argv()
# Stands for the heavy imports: reached only by a child whose supervisor lives.
pathlib.Path({marker!r}).write_text("imported")
"""


def _wait(condition, bound_s):
    end = time.monotonic() + bound_s
    while not condition():
        if time.monotonic() >= end:
            return False
        time.sleep(0.02)
    return True


def test_orphan_early_check_kills_the_child_before_its_imports(tmp_path):
    """#207, part 1, dynamically: a child started with a supervisor pid that
    is not its parent (as after the supervisor's death) dies at the early
    check and never reaches the line after it."""
    marker = tmp_path / "imported"
    script = tmp_path / "child.py"
    script.write_text(EARLY_CHILD.format(example=str(EXAMPLE), marker=str(marker)))
    # pid 1 is never this child's parent here: the check sees an orphan.
    child = subprocess.Popen([sys.executable, str(script), orphan.FLAG, "1"], start_new_session=True)
    assert child.wait(timeout=60) == -signal.SIGKILL
    assert not marker.exists(), "the orphaned child ran past its early check"
    # Control: with its real parent as the supervisor pid it runs on.
    ok = subprocess.Popen([sys.executable, str(script), orphan.FLAG, str(os.getpid())], start_new_session=True)
    assert ok.wait(timeout=60) == 0 and marker.exists()


HOST = """
import os, subprocess, sys, time
sys.path.insert(0, {example!r})
sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
# The checker shares this host's group (no new session) and is not its leader.
checker = subprocess.Popen([sys.executable, "-c",
    "import sys; sys.path.insert(0, {example!r}); import orphan; orphan.check(1)"])
rc = checker.wait()
time.sleep(0.3)
alive = sibling.poll() is None
sibling.kill()
print(rc, alive)
"""


def test_orphan_check_outside_a_led_group_kills_only_itself(tmp_path):
    """#207, part 2: a process that does not lead its group (started with
    the flag by hand, or without a new session) kills only itself on an
    orphan check, never its caller's group. The host runs in its own
    session, so a broken check would take only the host and its sibling."""
    script = tmp_path / "host.py"
    script.write_text(HOST.format(example=str(EXAMPLE)))
    host = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60,
                          start_new_session=True)  # fmt: skip
    assert host.returncode == 0, "the host survived the checker"
    assert host.stdout.split() == [str(-signal.SIGKILL), "True"], host.stdout


def _group(tmp_path):
    """A leader in its own session with one descendant, as a supervisor's
    child and its worker."""
    pidfile = tmp_path / "descendant.pid"
    code = (
        "import pathlib, subprocess, sys, time; "
        "d = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(d.pid)); time.sleep(60)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    assert _wait(pidfile.exists, 30)
    return proc, int(pidfile.read_text())


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return not stat or stat.startswith("Z")


@pytest.mark.parametrize("refusals", [0, 3])
def test_orphan_kill_group_survives_transient_eperm(tmp_path, monkeypatch, refusals):
    """#202: macOS's EPERM for an exiting group is retried until ESRCH; the
    leader and its descendant are both gone and the kill is confirmed."""
    proc, descendant = _group(tmp_path)
    real, left = os.killpg, [refusals]

    def killpg(pgid, sig):
        if left[0] > 0:
            left[0] -= 1
            raise PermissionError(1, "Operation not permitted")
        return real(pgid, sig)

    monkeypatch.setattr(orphan.os, "killpg", killpg)
    try:
        assert orphan.kill_group(proc, confirm_s=10.0)
        proc.wait(timeout=30)
        assert _wait(lambda: _gone(descendant), 10)
    finally:
        monkeypatch.undo()
        for pid in (proc.pid, descendant):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_orphan_kill_group_reports_a_persistent_eperm(tmp_path, monkeypatch):
    """A group kill refused for the whole confirmation window returns False,
    so the caller fails closed; the leader is still killed directly."""
    proc, descendant = _group(tmp_path)

    def killpg(pgid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(orphan.os, "killpg", killpg)
    try:
        assert not orphan.kill_group(proc, confirm_s=0.5)
        assert proc.wait(timeout=30) == -signal.SIGKILL
    finally:
        monkeypatch.undo()
        for pid in (proc.pid, descendant):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_orphan_pilot_uses_the_shared_group_kill():
    """#202: pilot.py kills its child's group through orphan.kill_group (the
    EPERM path above) and fails closed when it returns False, like
    supervisor.py."""
    src = (EXAMPLE / "pilot.py").read_text(encoding="utf-8")
    assert "orphan.kill_group(proc)" in src
    assert "os.killpg(proc.pid" not in src
    assert 'f"failed: {orphan.NOT_CONFIRMED}"' in src


# pilot.py's supervisor, driven with a trivial child: pilot.__file__ points at
# a script that exits at once, so the supervise path runs without the bot or
# a model, and os.killpg refuses every group kill (#202, TR on #210).
PILOT_SUPERVISED = """
import argparse, os, sys, time
sys.path.insert(0, {example!r})
import pilot

def eperm(pgid, sig):
    raise PermissionError(1, "Operation not permitted")

os.killpg = eperm
pilot.__file__ = {child!r}
args = argparse.Namespace(out={out!r}, ledger={ledger!r}, budget_core_hours=1.0, cores=1, wall_cap_s=30.0,
                          command="infer")
sys.exit(pilot.supervise(args, []))
"""

QUICK_CHILD = "import sys; sys.exit(0)\n"
SLEEPY_CHILD = "import os, time; os.killpg = lambda *a: None; time.sleep(60)\n"


def _pilot_supervised(tmp_path, child_code):
    child = tmp_path / "child.py"
    child.write_text(child_code)
    sup = tmp_path / "supervised.py"
    ledger = tmp_path / "ledger.json"
    sup.write_text(PILOT_SUPERVISED.format(example=str(EXAMPLE), child=str(child), out=str(tmp_path / "out"),
                                           ledger=str(ledger)))  # fmt: skip
    return sup, ledger


def _status(ledger):
    import json

    (entry,) = json.loads(ledger.read_text())["entries"]
    return entry["status"]


@pytest.mark.torch
def test_orphan_pilot_fails_closed_when_the_group_kill_is_refused(tmp_path):
    """#202: a child that returned 0, then a group kill refused for the whole
    confirmation window: pilot exits 1 and settles "failed: ... not
    confirmed gone", never "ok"."""
    pytest.importorskip("torch")  # pilot.py imports it at module level
    sup, ledger = _pilot_supervised(tmp_path, QUICK_CHILD)
    p = subprocess.run([sys.executable, str(sup)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 1, p.stderr
    assert _status(ledger) == f"failed: {orphan.NOT_CONFIRMED}"


@pytest.mark.torch
def test_orphan_pilot_sigterm_with_a_refused_kill_says_so_in_the_status(tmp_path):
    """#202: SIGTERM while the child runs, with every group kill refused: the
    settled status keeps "interrupted by signal" and names the refusal."""
    pytest.importorskip("torch")  # pilot.py imports it at module level
    sup, ledger = _pilot_supervised(tmp_path, SLEEPY_CHILD)
    p = subprocess.Popen([sys.executable, str(sup)])
    try:
        # The reservation is written before the child starts.
        assert _wait(lambda: ledger.exists() and '"entries"' in ledger.read_text(), 60)
        time.sleep(1.0)
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=60) == 128 + signal.SIGTERM
        assert _status(ledger) == f"interrupted by signal {signal.SIGTERM} ({orphan.NOT_CONFIRMED})"
    finally:
        if p.poll() is None:
            p.kill()
