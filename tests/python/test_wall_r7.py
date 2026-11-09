# SPDX-License-Identifier: MIT OR Apache-2.0
"""The r7 driver against a fake host: no M4, sudo or powermetrics."""

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "bench" / "wall_r7.py"
_spec = importlib.util.spec_from_file_location("wall_r7", _TOOL)
w = importlib.util.module_from_spec(_spec)
sys.modules["wall_r7"] = w
_spec.loader.exec_module(w)

SAMPLE = "**** Thermal pressure ****\nCurrent pressure level: {level}\n"


def script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class Fake(w.Host):
    def __init__(self, tmp, **kw):
        level = kw.pop("level", "Nominal")
        lines = kw.pop("samples", "while :; do printf '{}'; sleep 0.05; done".format(SAMPLE.format(level=level).replace("\n", "\\n")))
        ps = script(tmp / "ps", kw.pop("ps", "echo '1 0 0:00.00'"))
        pmset = script(tmp / "pmset", kw.pop("pmset", "echo 'No thermal warning level has been recorded'"))
        super().__init__(
            powermetrics=[script(tmp / "pm", lines)],
            ps=[ps],
            pmset=[pmset],
            sample_s=0.05,
            quiet_gap_s=0.01,
            stop_timeout_s=5.0,
            first_sample_timeout_s=5.0,
        )
        self.fake_load = kw.pop("load", 0.1)
        self.fake_runners = kw.pop("runners", [])

    def load(self):
        return self.fake_load

    def runners(self):
        return self.fake_runners


def harness(tmp, name="bin", ns=1000, digest="d"):
    return script(tmp / name, f"sleep 0.01; echo '{{\"ns\": {ns}, \"digest\": \"{digest}\"}}'\n")


def case(host, tmp, base=None, head=None):
    b = base or harness(tmp)
    return w.run_case(host, b, head or b, "cartpole_inline", str(tmp))


def test_wall_r7_frozen_command_matches_the_protocol():
    assert w.POWERMETRICS == ["/usr/bin/powermetrics", "-i", "2000", "-n", "10800", "--samplers", "cpu_power,thermal"]
    assert w.Host().powermetrics[:2] == ["sudo", "-n"]


@pytest.mark.parametrize(("text", "seconds"), [("0:00.03", 0.03), ("1:02.50", 62.5), ("1:00:01", 3601.0), ("2-00:00:01", 172801.0)])
def test_wall_r7_parses_cputime(text, seconds):
    assert w.parse_cputime(text) == pytest.approx(seconds)


def test_wall_r7_own_tree_follows_descendants_only():
    rows = {10: (1, 0.0), 11: (10, 0.0), 12: (11, 0.0), 20: (1, 0.0)}
    assert w.own_tree(rows, 10) == {10, 11, 12}


def test_wall_r7_foreign_cpu_new_process_from_zero_and_reused_pid():
    prev = {5: (1, 10.0), 6: (1, 50.0), 7: (1, 1.0)}
    cur = {5: (1, 12.0), 6: (1, 3.0), 8: (1, 4.0), 7: (1, 9.0)}
    got = w.foreign_per_process(prev, cur, 2.0, {7})
    assert got == {5: 1.0, 6: 1.5, 8: 2.0}


@pytest.mark.parametrize(("values", "bad"), [([0.5, 0.5], False), ([0.6, 0.6], True), ([0.0, 0.0, 0.0, 1.5], False), ([0.0, 0.0, 1.51], True), ([], True)])
def test_wall_r7_foreign_limits(values, bad):
    assert bool(w.foreign_invalid(values)) is bad


def test_wall_r7_pmset_only_a_low_numeric_limit_invalidates():
    assert w.pmset_invalid("No thermal warning level has been recorded") is None
    assert w.pmset_invalid("CPU_Speed_Limit \t= 100") is None
    assert w.pmset_invalid("CPU_Speed_Limit = 87")


def test_wall_r7_thermal_needs_a_level_and_nominal():
    assert w.thermal_invalid(["Nominal", "Nominal"]) is None
    assert w.thermal_invalid(["Nominal", "Moderate"])
    assert w.thermal_invalid([])


def test_wall_r7_aa_uses_both_levels():
    tight = [(-1) ** i * 0.001 for i in range(20)]
    assert w.aa_verdict(tight)["passed"]
    assert not w.aa_verdict([0.02 + i * 1e-4 for i in range(20)])["contains_zero"]
    wide = [(-1) ** i * (0.01 + i * 0.01) for i in range(20)]
    v = w.aa_verdict(wide)
    assert v["contains_zero"] and v["half_width"] > w.AA_HALF_WIDTH_LIMIT and not v["passed"]
    assert v["ci98_75"][0] <= v["ci95"][0] and v["ci95"][1] <= v["ci98_75"][1]


def test_wall_r7_interval_matches_the_pairs_tool_quantiles():
    import random as r
    import statistics as s

    x = [0.01 * i - 0.1 for i in range(20)]
    b = r.Random(w.BOOT_SEED)
    m = sorted(s.median(b.choices(x, k=20)) for _ in range(w.RESAMPLES))
    assert w.interval(x, 0.95) == (m[int(0.025 * w.RESAMPLES)], m[int(0.975 * w.RESAMPLES) - 1])


def test_wall_r7_measures_a_quiet_case(tmp_path):
    rec = case(Fake(tmp_path), tmp_path)
    assert rec["status"] == "measured", rec
    assert len(rec["changes"]) == 20 and rec["thermal_levels"] == ["Nominal"] and rec["pmset_samples"] > 0
    assert len((tmp_path / "cartpole_inline.jsonl").read_text().splitlines()) == 40


@pytest.mark.parametrize("kw", [{"load": 1.0}, {"runners": ["Runner.Listener"]}])
def test_wall_r7_refuses_a_noisy_machine(tmp_path, kw):
    rec = case(Fake(tmp_path, **kw), tmp_path)
    assert rec["status"] == "refused" and not (tmp_path / "cartpole_inline.jsonl").exists()


def test_wall_r7_refuses_a_busy_foreign_process(tmp_path):
    state = tmp_path / "n"
    ps = f'n=$(cat {state} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {state}; echo "99 1 0:$((10+n)).00"\n'
    rec = case(Fake(tmp_path, ps=ps), tmp_path)
    assert rec["status"] == "refused" and "busy" in rec["reason"]


def test_wall_r7_hot_machine_invalidates(tmp_path):
    rec = case(Fake(tmp_path, level="Heavy"), tmp_path)
    assert rec["status"] == "invalid" and any("thermal" in p for p in rec["problems"])


def test_wall_r7_sampler_ending_during_the_case_invalidates(tmp_path):
    samples = "printf '{}'; exit 0".format(SAMPLE.format(level="Nominal").replace("\n", "\\n"))
    slow = script(tmp_path / "slow", "sleep 0.05; echo '{\"ns\": 1, \"digest\": \"d\"}'\n")
    rec = case(Fake(tmp_path, samples=samples), tmp_path, base=slow)
    assert rec["status"] == "invalid" and any("ended during" in p for p in rec["problems"])


def test_wall_r7_sampler_without_a_first_sample_is_refused(tmp_path):
    rec = case(Fake(tmp_path, samples="exit 0"), tmp_path)
    assert rec["status"] == "refused" and "first sample" in rec["reason"]


def test_wall_r7_ps_failure_invalidates(tmp_path):
    host = Fake(tmp_path)
    original = host.read
    calls = {"n": 0}

    def flaky(command):
        if command == host.ps:
            calls["n"] += 1
            if calls["n"] > 6:
                raise OSError("no ps")
        return original(command)

    host.read = flaky
    slow = script(tmp_path / "slow", "sleep 0.05; echo '{\"ns\": 1, \"digest\": \"d\"}'\n")
    rec = w.run_case(host, slow, slow, "cartpole_inline", str(tmp_path))
    assert rec["status"] == "invalid" and any("read failure" in p for p in rec["problems"])


def test_wall_r7_digest_mismatch_invalidates(tmp_path):
    a, b = harness(tmp_path, "a", digest="x"), harness(tmp_path, "b", digest="y")
    rec = case(Fake(tmp_path), tmp_path, base=a, head=b)
    assert rec["status"] == "invalid" and any("digests differ" in p for p in rec["problems"])


def test_wall_r7_closing_the_pipe_ends_the_sampler(tmp_path):
    host = Fake(tmp_path)
    s = w.Sampler(host.powermetrics)
    assert s.first.wait(5)
    assert s.stop(5) and not s.ended_early and s.process.poll() is not None


def test_wall_r7_aa_needs_every_case(tmp_path, monkeypatch):
    def fake(_host, _b, _h, c, _o):
        if c == "tag_inline":
            return {"case": c, "status": "invalid", "problems": ["x"]}
        return {"case": c, "status": "measured", "changes": [(-1) ** i * 0.001 for i in range(20)]}

    monkeypatch.setattr(w, "run_case", fake)
    assert w.aa(object(), "b", str(tmp_path))["host_usable"] is False
    monkeypatch.setattr(w, "run_case", lambda _h, _b, _a, c, _o: {"case": c, "status": "measured", "changes": [(-1) ** i * 0.001 for i in range(20)]})
    r = w.aa(object(), "b", str(tmp_path))
    assert r["host_usable"] is True and len(r["cases"]) == 4


def test_wall_r7_failing_binary_invalidates_and_stops_the_sampler(tmp_path):
    bad = script(tmp_path / "bad", "exit 3\n")
    rec = case(Fake(tmp_path), tmp_path, base=bad)
    assert rec["status"] == "invalid" and any("harness failed" in p for p in rec["problems"])
