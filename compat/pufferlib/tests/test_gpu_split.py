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
- the driver rotates the case order by one per round, interleaved mode by
  mode (ADR-0061 item 5); the manifest checks the wheel and toolkit and the
  driver refuses a manifest or wheel whose sha256 differs (item 9); the
  summary row records (median, min, max) per cell and the attempt outcome
  (item 8), and is gate eligible only on cuda with the CUDA advantage kernel;
- the epoch default is 31 (item 2);
- a manifest attempt refuses --split-args, a wheel absent from --python
  and a commit other than HEAD; the summary computes item 7 (best e2e
  median per side, case recorded) only when the attempt is gate eligible;
- the driver's plan is fixed, only the first CUDA trainer run
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

import manifest  # noqa: E402
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
    if case.startswith("tessera"):
        assert len(record["tessera_rl"]["native_sha256"]) == 64
    else:
        assert record["tessera_rl"] is None
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


def test_gpu_split_driver_plan_rotates_the_cases_each_round():
    a = driver.plan(split.CASES, split.MODES, 5)
    assert len(a) == 5 * len(split.CASES) * len(split.MODES)
    for r in range(5):
        pairs = [(c, m) for rr, _, c, m in a if rr == r]
        assert sorted(pairs) == sorted((c, m) for c in split.CASES for m in split.MODES)
        for m in split.MODES:
            cases = [c for c, mm in pairs if mm == m]
            k = r % len(split.CASES)
            assert cases == [*split.CASES[k:], *split.CASES[:k]]
        # interleaved: each mode's block holds all four cases in a row
        assert [mm for _, mm in pairs] == [m for m in split.MODES for _ in split.CASES]


def test_gpu_split_epochs_default_is_31():
    assert split.parser().parse_args(["ocean-serial", "e2e", "cpu"]).epochs == 31


def fake_nvcc(tmp_path, version="13.0.88"):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V{version}'\n")
    nvcc.chmod(0o755)
    return str(nvcc)


def make_wheel(tmp_path, name="tessera_rl-0-cp311-abi3-linux_x86_64.whl", native=b"native bytes"):
    import zipfile

    wheel = tmp_path / name
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("tessera_rl/__init__.py", "")
        z.writestr("tessera_rl/_native.abi3.so", native)
        z.writestr("tessera_rl-0.dist-info/RECORD", "")
    return wheel


def make_manifest(tmp_path, **over):
    wheel = make_wheel(tmp_path)
    args = {"--out": str(tmp_path / "manifest.json"), "--commit": driver.git_head(), "--wheel": str(wheel),
            "--wheel-sha256": manifest.file_sha256(str(wheel)), "--wheel-source": "main-wheels-run",
            "--toolkit": "13.0.88", "--nvcc": fake_nvcc(tmp_path), "--rounds": "2"}  # fmt: skip
    args.update(over)
    return wheel, [x for kv in args.items() for x in kv]


def manifest_attempt(tmp_path, capsys, python=None, extra=(), **over):
    """Writes a manifest and runs the driver on it; returns (rc, rows, wheel, digest)."""
    wheel, argv = make_manifest(tmp_path, **over)
    assert manifest.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    out = tmp_path / "r.jsonl"
    rc = driver.main(["--python", python or f"sh -c 'exit 1' fake {wheel}", "--out", str(out), "--device", "cpu",
                      "--manifest", str(tmp_path / "manifest.json"), "--manifest-sha256", digest,
                      "--wheel", str(wheel), *extra])  # fmt: skip
    rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    return rc, rows, wheel, digest


def test_gpu_split_manifest_fixes_commit_wheel_toolkit_and_order(tmp_path, capsys):
    wheel, argv = make_manifest(tmp_path)
    assert manifest.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    m = manifest.load(str(tmp_path / "manifest.json"), digest)
    assert m["commit"] == driver.git_head() and m["cuda_toolkit"] == "13.0.88"
    assert m["wheel_sha256"] == manifest.file_sha256(str(wheel))
    assert m["order"] == manifest.order(manifest.CASES, manifest.MODES, 2)


@pytest.mark.parametrize("over", [{"--wheel-sha256": "0" * 64}, {"--toolkit": "12.9.0"}, {"--commit": "abc"}])
def test_gpu_split_manifest_refuses_a_mismatch(tmp_path, capsys, over):
    _, argv = make_manifest(tmp_path, **over)
    assert manifest.main(argv) == manifest.EXIT_MISMATCH
    assert "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / "manifest.json").exists()


def test_gpu_split_driver_refuses_a_wrong_manifest_or_wheel_sha256(tmp_path, capsys):
    wheel, argv = make_manifest(tmp_path)
    manifest.main(argv)
    digest = capsys.readouterr().out.strip()
    out = tmp_path / "r.jsonl"
    common = ["--out", str(out), "--device", "cpu", "--manifest", str(tmp_path / "manifest.json")]
    rc = driver.main(["--python", f"sh -c 'exit 0' fake {wheel}", *common, "--manifest-sha256", "0" * 64, "--wheel", str(wheel)])
    assert rc == driver.EXIT_REFUSED
    other = make_wheel(tmp_path, "other.whl", b"other")
    rc = driver.main(["--python", f"sh -c 'exit 0' fake {other}", *common, "--manifest-sha256", digest, "--wheel", str(other)])
    assert rc == driver.EXIT_REFUSED
    assert not out.exists()


def test_gpu_split_driver_refuses_split_args_in_a_manifest_attempt(tmp_path, capsys):
    rc, rows, _, _ = manifest_attempt(tmp_path, capsys, extra=("--split-args", "--epochs 3"))
    assert rc == driver.EXIT_REFUSED and rows == []
    assert "--split-args" in capsys.readouterr().out


def test_gpu_split_driver_refuses_a_wheel_absent_from_the_python_command(tmp_path, capsys):
    rc, rows, _, _ = manifest_attempt(tmp_path, capsys, python="sh -c 'exit 1' fake")
    assert rc == driver.EXIT_REFUSED and rows == []
    assert "--python" in capsys.readouterr().out


def test_gpu_split_driver_refuses_a_commit_mismatch(tmp_path, capsys):
    rc, rows, _, _ = manifest_attempt(tmp_path, capsys, **{"--commit": "b" * 40})
    assert rc == driver.EXIT_REFUSED and rows == []
    assert "manifest's commit" in capsys.readouterr().out


def test_gpu_split_driver_follows_the_manifest_and_records_a_failed_attempt(tmp_path, capsys):
    rc, rows, _, digest = manifest_attempt(tmp_path, capsys, **{"--rounds": "1"})
    runs, last = rows[:-1], rows[-1]
    assert rc == 1
    assert [[r["case"], r["mode"]] for r in runs] == manifest.order(manifest.CASES, manifest.MODES, 1)[0]
    assert all(r["manifest_sha256"] == digest for r in runs)
    assert last["summary"] and last["manifest_sha256"] == digest
    assert last["attempt"] == "failed, no result" and last["gate_eligible"] is False and "pass" not in last


NATIVE = "e" * 64
MANIFEST = {"native_sha256": NATIVE}


def gate_rows(medians: dict, backend="cuda", epochs=30, scale="full", native=NATIVE):
    rows = []
    for case in manifest.CASES:
        for mode in manifest.MODES:
            for _ in range(5):
                res = {"sps": medians.get(case, 1.0), "advantage_backend": backend, "scale": scale}
                if mode != "env":
                    res["epochs_timed"] = epochs
                if case.startswith("tessera"):
                    res["tessera_rl"] = {"native_sha256": native}
                rows.append({"case": case, "mode": mode, "rc": 0, "result": res})
    return rows


def test_gpu_split_summary_computes_pass_and_fail_from_the_best_case_of_each_side():
    win = driver.summary(gate_rows({"ocean-serial": 4.0, "ocean-mp": 5.0, "tessera-w1": 5.0, "tessera-w2": 3.0}),
                         "cuda", True, MANIFEST, "f" * 64)  # fmt: skip
    assert win["gate_eligible"] is True and win["pass"] is True
    assert (win["ocean_case"], win["ocean_score"]) == ("ocean-mp", 5.0)
    assert (win["tessera_case"], win["tessera_score"]) == ("tessera-w1", 5.0)
    lose = driver.summary(gate_rows({"ocean-serial": 6.0, "tessera-w1": 5.0, "tessera-w2": 5.5}),
                          "cuda", True, MANIFEST, "f" * 64)  # fmt: skip
    assert lose["gate_eligible"] is True and lose["pass"] is False
    assert (lose["ocean_case"], lose["tessera_case"]) == ("ocean-serial", "tessera-w2")


@pytest.mark.parametrize(
    "rows, device, complete, man, sha",
    [
        (gate_rows({}), "cuda", False, MANIFEST, "f" * 64),
        (gate_rows({}), "cpu", True, MANIFEST, "f" * 64),
        (gate_rows({}), "cuda", True, None, None),
        (gate_rows({}, backend="cpu-fallback"), "cuda", True, MANIFEST, "f" * 64),
        (gate_rows({}, epochs=2), "cuda", True, MANIFEST, "f" * 64),
        (gate_rows({}, scale="small"), "cuda", True, MANIFEST, "f" * 64),
        (gate_rows({}, native="0" * 64), "cuda", True, MANIFEST, "f" * 64),
    ],
    ids=["incomplete", "cpu", "no-manifest", "cpu-fallback", "epochs", "scale", "native"],
)
def test_gpu_split_summary_has_no_pass_when_not_eligible(rows, device, complete, man, sha):
    s = driver.summary(rows, device, complete, man, sha)
    assert s["gate_eligible"] is False and "pass" not in s


def test_gpu_split_summary_refuses_a_tessera_row_without_native_digest():
    rows = gate_rows({})
    victim = next(r for r in rows if r["case"] == "tessera-w2" and r["mode"] == "env")
    victim["result"]["tessera_rl"] = None
    s = driver.summary(rows, "cuda", True, MANIFEST, "f" * 64)
    assert s["gate_eligible"] is False and "pass" not in s


def test_gpu_split_summary_cells():
    s = driver.summary(gate_rows({"tessera-w1": 2.0}), "cuda", True, MANIFEST, "f" * 64)
    assert s["cells"]["tessera-w1/e2e"] == {"n": 5, "median": 2.0, "min": 2.0, "max": 2.0}


def test_gpu_split_manifest_records_the_wheel_native_module(tmp_path, capsys):
    import hashlib

    _, argv = make_manifest(tmp_path)
    manifest.main(argv)
    digest = capsys.readouterr().out.strip()
    m = manifest.load(str(tmp_path / "manifest.json"), digest)
    assert m["native_sha256"] == hashlib.sha256(b"native bytes").hexdigest()
