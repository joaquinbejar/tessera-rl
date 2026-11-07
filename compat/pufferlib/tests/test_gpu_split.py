# SPDX-License-Identifier: MIT OR Apache-2.0
"""The GPU split harness (compat/pufferlib/gpu_split, docs/reviews/gpu-split-plan.md r5),
locally on CPU: no GPU, no host, nothing installed.

Covered:
- the synchronised pufferl.py copy is exactly the published patch, two blocks
  enabled and nothing else, and an unexpected source is refused;
- every case runs in every mode at the small scale, closes its vector
  environment, stops its monitor and leaves no child process;
- the sync mode's attribution is exclusive: the categories add up to at most
  the wall, with a nonnegative remainder;
- CUDA requested without CUDA stops with its own exit code, no CUDA call
  (is_available included) happens before the vector environment exists, and
  CUDA initialised before it is refused;
- the env probe drains its warm-up and times each send with its matching
  recv, so no transition is left pending;
- the driver's per-round shuffle is fixed, only the first CUDA trainer run
  gets the long timeout, and a timed-out run's whole process group is
  killed, with nothing left.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

HARNESS = pathlib.Path(__file__).parents[1] / "gpu_split"
sys.path.insert(0, str(HARNESS))

import run as driver  # noqa: E402
import split  # noqa: E402
import sync  # noqa: E402

SMALL = ["--epochs", "3", "--scale", "small", "--env-steps", "200"]


def run_split(tmp_path, *args):
    p = subprocess.run(
        [sys.executable, "-I", str(HARNESS / "split.py"), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [x[x.index('{"case"') :] for x in p.stdout.splitlines() if '{"case"' in x]
    return p.returncode, (json.loads(lines[-1]) if lines else None), p.stdout[-3000:] + p.stderr[-3000:]


# ---------------------------------------------------------------- sync patch


def installed_source():
    with open(sync.original_path(), encoding="utf-8") as f:
        return f.read()


def test_gpu_split_sync_copy_is_exactly_the_published_patch():
    source = installed_source()
    with open(sync.PATCH, encoding="utf-8") as f:
        assert sync.diff(source) == f.read()
    patched = sync.patched_source(source)
    assert patched.count(sync.ENABLED) == 2 and sync.COMMENTED not in patched
    changed = [(a, b) for a, b in zip(source.splitlines(), patched.splitlines()) if a != b]
    assert len(changed) == 4 and len(source.splitlines()) == len(patched.splitlines())
    assert all("torch.cuda" in b and b.lstrip()[0] != "#" for _, b in changed)


@pytest.mark.parametrize("blocks", [0, 1, 3])
def test_gpu_split_sync_refuses_an_unexpected_source(blocks):
    with pytest.raises(sync.PatchError):
        sync.patched_source("x = 1\n" + sync.COMMENTED * blocks)


def test_gpu_split_sync_loads_a_scratch_copy_without_touching_the_installed_file(tmp_path):
    before = installed_source()
    module, digests = sync.load(str(tmp_path))
    assert module.__file__ == str(tmp_path / "pufferl_sync.py")
    assert installed_source() == before
    assert digests["pufferl_sha256"] != digests["pufferl_sync_sha256"]


# ------------------------------------------------------------------- runs


@pytest.mark.parametrize("case", split.CASES)
@pytest.mark.parametrize("mode", split.MODES)
def test_gpu_split_every_case_and_mode_runs_and_cleans_up(tmp_path, case, mode):
    rc, record, log = run_split(tmp_path, case, mode, "cpu", *SMALL)
    assert rc == 0, log
    assert record["case"] == case and record["mode"] == mode
    assert record["cuda_initialized_before_vecenv"] is False
    assert record["children_left"] == [] and record["monitors_left"] == []
    assert record["advantage_backend"] in ("cuda", "cpu-fallback")
    assert record["sps"] > 0
    if case == "ocean-mp":
        assert len(record["workers"]) >= 2
    assert sorted(os.listdir(tmp_path)) in ([], ["resources"])


def test_gpu_split_sync_attribution_is_exclusive(tmp_path):
    rc, record, log = run_split(tmp_path, "ocean-serial", "sync", "cpu", *SMALL)
    assert rc == 0, log
    exclusive = record["exclusive_s"]
    overhead = record["parent_overhead_s"]
    assert set(exclusive) == set(split.CHILDREN) and set(overhead) == set(split.PARENTS)
    assert all(v >= 0 for v in [*exclusive.values(), *overhead.values()])
    assert record["unattributed_s"] >= 0
    total = sum(exclusive.values()) + sum(overhead.values()) + record["unattributed_s"]
    assert abs(total - record["wall_s"]) < 1e-9
    assert record["pufferl_sha256"] and record["pufferl_sync_sha256"]
    assert record["profile_frequency"] == 1


def test_gpu_split_e2e_keeps_the_profiler_out_of_the_window(tmp_path):
    rc, record, log = run_split(tmp_path, "ocean-serial", "e2e", "cpu", *SMALL)
    assert rc == 0, log
    assert "exclusive_s" not in record and record["epochs_timed"] == 2
    assert record["profile_frequency"] == 10**9


@pytest.mark.skipif(__import__("torch").cuda.is_available(), reason="needs a host without CUDA")
def test_gpu_split_cuda_without_cuda_stops_with_its_exit_code(tmp_path):
    rc, record, _ = run_split(tmp_path, "ocean-mp", "e2e", "cuda", *SMALL)
    assert rc == split.EXIT_NO_CUDA
    assert "is_available" in record["error"]
    assert record["children_left"] == [] and record["monitors_left"] == []


def test_gpu_split_no_cuda_call_before_the_vecenv(monkeypatch, capsys):
    import torch

    calls = []
    real_make = split.make_vecenv

    def make(*a, **k):
        calls.append("make_vecenv")
        return real_make(*a, **k)

    def available():
        calls.append("is_available")
        return False

    monkeypatch.setattr(split, "make_vecenv", make)
    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: calls.append("device_count") or 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: calls.append("get_device_name") or "")
    assert split.main(["ocean-serial", "e2e", "cuda", *SMALL]) == split.EXIT_NO_CUDA
    assert calls == ["make_vecenv", "is_available"]
    capsys.readouterr()


class AsyncFakeVec:
    """recv blocks until the last send's transition completes; counts what is
    still pending."""

    num_agents = 4

    class single_action_space:
        n = 2

    def __init__(self, delay):
        self.delay, self.pending, self.sends, self.recvs = delay, 0, 0, 0

    def async_reset(self, seed=None):
        self.pending = 1

    def send(self, actions):
        assert self.pending == 0, "send while a transition is pending"
        self.pending, self.sends = 1, self.sends + 1

    def recv(self):
        assert self.pending == 1, "recv with nothing pending"
        time.sleep(self.delay)
        self.pending, self.recvs = 0, self.recvs + 1


def test_gpu_split_env_probe_completes_every_timed_transition():
    fake = AsyncFakeVec(delay=0.002)
    out = split.run_env(fake, steps=50, warmup=5)
    assert fake.pending == 0
    assert fake.sends == 55 and fake.recvs == 56
    assert out["steps"] == 50 * fake.num_agents
    assert out["wall_s"] >= 50 * 0.002


def test_gpu_split_refuses_cuda_initialised_before_the_vecenv(monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    assert split.main(["ocean-serial", "e2e", "cpu", *SMALL]) == 4
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["cuda_initialized_before_vecenv"] is True


# ----------------------------------------------------------------- driver


def test_gpu_split_driver_plan_is_a_fixed_shuffle_per_round():
    a = driver.plan(split.CASES, split.MODES, 3)
    assert a == driver.plan(split.CASES, split.MODES, 3)
    assert len(a) == 3 * len(split.CASES) * len(split.MODES)
    for r in range(3):
        pairs = [(c, m) for rr, _, c, m in a if rr == r]
        assert sorted(pairs) == sorted((c, m) for c in split.CASES for m in split.MODES)
    orders = [[(c, m) for rr, _, c, m in a if rr == r] for r in range(3)]
    assert len({tuple(o) for o in orders}) > 1


def test_gpu_split_driver_kills_the_whole_group_on_timeout(tmp_path):
    out = tmp_path / "r.jsonl"
    # A "run" that leaves a background child in its group and never finishes.
    fake = "sh -c 'sleep 60 & sleep 60' fake"
    rc = driver.main(["--python", fake, "--out", str(out), "--device", "cpu", "--rounds", "1",
                      "--cases", "ocean-serial", "--modes", "e2e", "--timeout", "2"])  # fmt: skip
    row = json.loads(out.read_text().splitlines()[0])
    assert row["timed_out"] is True and row["group_left"] == [] and row["result"] is None
    assert rc == 1


def test_gpu_split_driver_gives_the_long_timeout_to_the_first_cuda_trainer_run_only(tmp_path):
    out = tmp_path / "r.jsonl"
    fake = "sh -c 'sleep 3' fake"
    driver.main(["--python", fake, "--out", str(out), "--device", "cuda", "--rounds", "1",
                 "--cases", "ocean-serial,ocean-mp", "--modes", "env,e2e,sync",
                 "--timeout", "1", "--first-timeout", "10"])  # fmt: skip
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    long = [r for r in rows if r["timeout_s"] == 10]
    assert len(long) == 1 and long[0]["mode"] != "env" and not long[0]["timed_out"]
    first_trainer = next(r for r in rows if r["mode"] != "env")
    assert long[0] is first_trainer
    assert all(r["timed_out"] and r["timeout_s"] == 1 for r in rows if r is not first_trainer)


def test_gpu_split_driver_records_a_real_run(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = driver.main(["--python", f"{sys.executable} -I", "--out", str(out), "--device", "cpu", "--rounds", "1",
                      "--cases", "ocean-mp", "--modes", "env", "--split-args", " ".join(SMALL)])  # fmt: skip
    assert rc == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert row["rc"] == 0 and row["group_left"] == [] and row["result"]["children_left"] == []
