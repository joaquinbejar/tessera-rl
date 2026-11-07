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
- CUDA requested without CUDA stops with its own exit code, and CUDA
  initialised before the vector environment is refused;
- the driver's per-round shuffle is fixed, and a timed-out run's whole process
  group is killed, with nothing left.
"""

import json
import os
import pathlib
import subprocess
import sys

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
    assert set(exclusive) == set(split.CATEGORIES)
    assert all(v >= 0 for v in exclusive.values())
    assert record["unattributed_s"] >= 0
    assert abs(sum(exclusive.values()) + record["unattributed_s"] - record["wall_s"]) < 1e-9
    assert record["pufferl_sha256"] and record["pufferl_sync_sha256"]


def test_gpu_split_e2e_keeps_the_profiler_out_of_the_window(tmp_path):
    rc, record, log = run_split(tmp_path, "ocean-serial", "e2e", "cpu", *SMALL)
    assert rc == 0, log
    assert "exclusive_s" not in record and record["epochs_timed"] == 2


@pytest.mark.skipif(__import__("torch").cuda.is_available(), reason="needs a host without CUDA")
def test_gpu_split_cuda_without_cuda_stops_with_its_exit_code(tmp_path):
    rc, record, _ = run_split(tmp_path, "ocean-serial", "e2e", "cuda", *SMALL)
    assert rc == split.EXIT_NO_CUDA
    assert "is_available" in record["error"]


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
                      "--cases", "ocean-serial", "--modes", "e2e", "--first-timeout", "2"])  # fmt: skip
    row = json.loads(out.read_text().splitlines()[0])
    assert row["timed_out"] is True and row["group_left"] == [] and row["result"] is None
    assert rc == 1


def test_gpu_split_driver_records_a_real_run(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = driver.main(["--python", f"{sys.executable} -I", "--out", str(out), "--device", "cpu", "--rounds", "1",
                      "--cases", "ocean-mp", "--modes", "env", "--split-args", " ".join(SMALL)])  # fmt: skip
    assert rc == 0
    row = json.loads(out.read_text().splitlines()[0])
    assert row["rc"] == 0 and row["group_left"] == [] and row["result"]["children_left"] == []
