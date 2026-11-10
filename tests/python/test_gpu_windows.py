# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native Windows split harness, locally on CPU: no GPU, no Windows."""

import json
import pathlib
import subprocess
import sys
import time

import pytest

psutil = pytest.importorskip("psutil")
pytest.importorskip("torch")

HARNESS = pathlib.Path(__file__).resolve().parents[2] / "compat" / "gpu_windows"
sys.path.insert(0, str(HARNESS))

import run_native as driver  # noqa: E402
import split_native as split  # noqa: E402

pytestmark = pytest.mark.torch
SMALL = ["--epochs", "3", "--scale", "small", "--env-steps", "50"]


def _run(*args):
    p = subprocess.run(
        [sys.executable, str(HARNESS / "split_native.py"), *args],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    lines = [x for x in p.stdout.splitlines() if x.startswith('{"case"')]
    return p.returncode, (json.loads(lines[-1]) if lines else None), p.stderr[-2000:]


@pytest.mark.parametrize("case", ["tessera-w1", "tessera-w2"])
def test_split_e2e_counts_only_the_timed_epochs(case):
    rc, r, err = _run(case, "e2e", "cpu", *SMALL)
    assert rc == 0, err
    assert r["steps"] == 2 * split.BPTT * split.AGENTS["small"] and r["epochs_timed"] == 2
    assert "exclusive_s" not in r and len(r["native_sha256"]) == 64


def test_split_sync_phases_are_exclusive_and_reconcile():
    rc, r, err = _run("tessera-w1", "sync", "cpu", *SMALL)
    assert rc == 0, err
    assert set(r["exclusive_s"]) == set(split.PHASES) and all(v > 0 for v in r["exclusive_s"].values())
    assert r["unattributed_s"] >= 0
    assert sum(r["exclusive_s"].values()) + r["unattributed_s"] == pytest.approx(r["wall_s"])


def test_split_env_mode_runs_without_a_policy():
    rc, r, err = _run("tessera-w2", "env", "cpu", *SMALL)
    assert rc == 0, err
    assert r["steps"] == 50 * split.AGENTS["small"] and "device_name" not in r


def test_split_full_scale_is_the_planned_shape():
    assert split.AGENTS["full"] * split.BPTT == 524_288 and split.MINIBATCH["full"] == 32_768
    assert 10 * split.BPTT * split.AGENTS["full"] == 5_242_880


def test_split_missing_cuda_stops_with_its_own_exit_code():
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    rc, r, _ = _run("tessera-w1", "e2e", "cuda", *SMALL)
    assert rc == split.EXIT_NO_CUDA and "error" in r


def test_phases_disabled_never_synchronizes():
    calls = []
    phases = split.Phases(False, lambda: calls.append(1))
    phases.mark("env")
    phases.mark(None)
    assert not calls and phases.totals["env"] == 0.0


def test_phases_synchronize_at_every_boundary_and_charge_one_phase():
    calls = []
    phases = split.Phases(True, lambda: calls.append(1))
    phases.mark("env")
    time.sleep(0.01)
    phases.mark("copy")
    phases.mark(None)
    assert len(calls) == 3 and phases.totals["env"] >= 0.009 and phases.totals["learn"] == 0.0


def test_driver_plan_is_eighteen_runs_in_a_fixed_shuffle():
    runs = driver.plan()
    assert len(runs) == 18 and runs == driver.plan()
    for r in range(3):
        assert len({(c, m) for q, _, c, m in runs if q == r}) == 6
    assert [x[2:] for x in runs[:6]] != [x[2:] for x in runs[6:12]]


def test_driver_timeout_kills_the_run_and_its_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(c.pid));"
        "time.sleep(60)"
    )
    out = driver.one_run([sys.executable, "-c", code], timeout=1.5)
    assert out["timed_out"] and out["left"] == [] and out["result"] is None
    child = int(pid_file.read_text())
    time.sleep(0.2)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_driver_keeps_only_the_result_line():
    code = "print('noise');print('x {\"case\": \"a\", \"sps\": 1}')"
    out = driver.one_run([sys.executable, "-c", code], timeout=30)
    assert out["rc"] == 0 and out["result"] == {"case": "a", "sps": 1} and out["tail"] is None


def test_driver_first_cuda_trainer_gets_the_long_timeout(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(driver, "one_run", lambda c, t, w: seen.append((c, t)) or {"rc": 0, "timed_out": False, "left": [], "result": None, "tail": ""})
    assert driver.main(["--out", str(tmp_path / "o"), "--device", "cuda", "--work-dir", str(tmp_path), "--rounds", "1", "--timeout", "1", "--first-timeout", "9"]) == 0
    long = [(c, t) for c, t in seen if t == 9]
    assert len(long) == 1 and "env" not in long[0][0]
    assert seen.index(long[0]) == min(i for i, (c, _) in enumerate(seen) if "env" not in c)


def test_driver_leftover_process_stops_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "one_run", lambda c, t, w: {"rc": 0, "timed_out": False, "left": [123], "result": None, "tail": ""})
    assert driver.main(["--out", str(tmp_path / "o"), "--device", "cpu", "--work-dir", str(tmp_path)]) == driver.EXIT_LEFTOVER
    assert len((tmp_path / "o").read_text().splitlines()) == 1


class _Proc:
    def __init__(self, ppid, created, error=False):
        self._ppid, self._created, self._error = ppid, created, error

    def ppid(self):
        if self._error:
            raise psutil.NoSuchProcess(1)
        return self._ppid

    def create_time(self):
        return self._created


def test_driver_orphans_are_found_by_parent_and_start_time():
    child, old_pid_reuse, stranger = _Proc(10, 5.0), _Proc(10, 1.0), _Proc(99, 6.0)
    gone = _Proc(10, 5.0, error=True)
    assert driver.orphans([child, old_pid_reuse, stranger, gone], {10}, 2.0) == [child]


def test_driver_records_a_child_killed_after_a_normal_exit(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(c.pid));"
        "time.sleep(1)"
    )
    out = driver.one_run([sys.executable, "-c", code], timeout=30)
    assert out["rc"] == 0 and out["killed_after_exit"] == [int(pid_file.read_text())] and out["left"] == []


def test_driver_runs_inside_the_work_dir_and_removes_it(tmp_path):
    code = "import json,os;print(json.dumps({'case': os.path.basename(os.path.dirname(os.getcwd()))}))"
    out = driver.one_run([sys.executable, "-c", code], timeout=30, work_dir=str(tmp_path))
    assert out["result"] == {"case": tmp_path.name} and list(tmp_path.iterdir()) == []


def test_driver_one_run_kills_an_orphan_of_its_root(monkeypatch):
    real_popen = subprocess.Popen
    roots = []

    def popen(*args, **kwargs):
        roots.append(real_popen(*args, **kwargs))
        return roots[-1]

    class Orphan(_Proc):
        pid = 424242
        killed = 0

        def kill(self):
            Orphan.killed += 1

        def wait(self, timeout=None):
            return 0

        def is_running(self):
            return False

    orphan = Orphan(None, time.time() + 3600)
    orphan.ppid = lambda: roots[-1].pid
    stranger = Orphan(1, time.time() + 3600)
    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    monkeypatch.setattr(driver.psutil, "process_iter", lambda: [orphan, stranger])
    out = driver.one_run([sys.executable, "-c", "pass"], timeout=30)
    assert out["killed_after_exit"] == [424242] and Orphan.killed == 1 and out["left"] == []
