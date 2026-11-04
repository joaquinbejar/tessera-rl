# SPDX-License-Identifier: MIT OR Apache-2.0
"""The supervisor's group kill when the OS answers EPERM.

On macOS, ``killpg`` returns EPERM for a process group whose remaining
members are all exiting (zombies), which happens under load right after the
child returns. The supervisor must still settle its ledger entry, never
leave it open. ``os.killpg`` is replaced with one that raises EPERM, so the
regression runs on every platform.
"""

import json
import pathlib
import sys

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import supervisor  # noqa: E402

CHILD = """
import sys, time
sys.path.insert(0, {example!r})
import supervisor
sys.exit(supervisor.run(__file__, sys.argv[1:], lambda clock: {body}, ledger="", budget_core_hours=1.0,
                        cores=1, wall_cap_s=1.0, out="", name="child"))
"""


def _run(tmp_path, monkeypatch, body, wall_cap_s):
    script = tmp_path / "child.py"
    script.write_text(CHILD.format(example=str(EXAMPLE), body=body))

    def eperm(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(supervisor.os, "killpg", eperm)
    ledger = tmp_path / "ledger.json"
    rc = supervisor.run(str(script), [], None, ledger=str(ledger), budget_core_hours=1.0, cores=1,
                        wall_cap_s=wall_cap_s, out=str(tmp_path / "out"), name="t")  # fmt: skip
    return rc, json.loads(ledger.read_text())["entries"]


def test_supervisor_eperm_after_the_child_returns_still_settles_ok(tmp_path, monkeypatch):
    rc, entries = _run(tmp_path, monkeypatch, "0", wall_cap_s=30.0)
    assert rc == 0
    assert [e["status"] for e in entries] == ["ok"]


def test_supervisor_eperm_at_the_deadline_still_kills_the_child(tmp_path, monkeypatch):
    # The leader is killed directly when the group kill is refused.
    rc, entries = _run(tmp_path, monkeypatch, "time.sleep(60) or 0", wall_cap_s=3.0)
    assert rc == supervisor.EXIT_DEADLINE
    assert entries[0]["status"].startswith("killed at the deadline")
    assert entries[0]["wall_s"] < 30
