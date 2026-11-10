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
        caffeinate = script(tmp / "caff", "exec sleep 60\n")
        super().__init__(
            powermetrics=[script(tmp / "pm", lines)],
            ps=[ps],
            pmset=[pmset],
            pmset_power=[pmset],
            caffeinate=[caffeinate],
            sample_s=0.05,
            quiet_gap_s=0.01,
            stop_timeout_s=5.0,
            first_sample_timeout_s=5.0,
            settle_timeout_s=0.2,
            settle_poll_s=0.01,
        )
        self.fake_load = kw.pop("load", 0.1)
        self.fake_runners = kw.pop("runners", [])
        self.tmp = str(tmp)

    def read(self, command):
        # Hermetic: only the fakes under tmp may run, never a real host command.
        if not str(command[0]).startswith(self.tmp):
            raise AssertionError(f"real host command leaked into a test: {command}")
        return super().read(command)

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


def test_wall_r7_interval_takes_the_frozen_order_statistics(monkeypatch):
    # Every resampled "median" distinct, so both indices are pinned exactly.
    counter = iter(range(w.RESAMPLES))
    monkeypatch.setattr(w.statistics, "median", lambda _x: next(counter))
    assert w.interval([0.0, 1.0], 0.95) == (250, 9749)


def test_wall_r7_interval_resamples_with_the_frozen_seed():
    import random as r
    import statistics as s

    x = [0.01 * i for i in range(20)]
    b = r.Random(w.BOOT_SEED)
    m = sorted(s.median(b.choices(x, k=20)) for _ in range(w.RESAMPLES))
    assert w.interval(x, 0.95) == (m[250], m[9749])


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
    # The sampler ends only once the harness has run, so its end falls inside
    # the case on any runner, never before the first-sample check.
    started = tmp_path / "started"
    samples = "printf '{}'; while [ ! -e {} ]; do sleep 0.01; done; exit 0".format(SAMPLE.format(level="Nominal").replace("\n", "\\n"), started)
    slow = script(tmp_path / "slow", f"touch {started}; sleep 0.05; echo '{{\"ns\": 1, \"digest\": \"d\"}}'\n")
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
    assert w.aa(Fake(tmp_path), "b", str(tmp_path))["host_usable"] is False
    monkeypatch.setattr(w, "run_case", lambda _h, _b, _a, c, _o: {"case": c, "status": "measured", "changes": [(-1) ** i * 0.001 for i in range(20)]})
    r = w.aa(Fake(tmp_path), "b", str(tmp_path))
    assert r["host_usable"] is True and len(r["cases"]) == 4


def test_wall_r7_failing_binary_invalidates_and_stops_the_sampler(tmp_path):
    bad = script(tmp_path / "bad", "exit 3\n")
    rec = case(Fake(tmp_path), tmp_path, base=bad)
    assert rec["status"] == "invalid" and any("harness failed" in p for p in rec["problems"])


def test_wall_r7_contains_zero_is_tested_at_98_75_not_95():
    x = [-0.005] * 5 + [0.001 * (i + 1) for i in range(15)]
    assert w.interval(x, 0.95)[0] > 0.0
    assert w.aa_verdict(x)["contains_zero"]


def test_wall_r7_quiet_check_takes_three_readings(tmp_path):
    host = Fake(tmp_path)
    calls = []
    original = host.read
    host.read = lambda c: calls.append(c) or original(c)
    assert w.quiet_refusal(host) is None
    assert len(calls) == 1 + w.QUIET_READINGS == 4


def test_wall_r7_two_busy_processes_add_up():
    prev = {5: (1, 0.0), 6: (1, 0.0)}
    high = {5: (1, 1.8), 6: (1, 1.8)}
    assert w.interval_foreign(prev, high, 2.0, set()) == pytest.approx(1.8)
    assert w.foreign_invalid([0.0, 0.0, 0.0, 1.8])
    low = {5: (1, 0.6), 6: (1, 0.6)}
    assert w.foreign_invalid([w.interval_foreign(prev, low, 2.0, set())] * 3)


def test_wall_r7_pair_order_is_the_seeded_one(tmp_path):
    import json
    import random

    case(Fake(tmp_path), tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "cartpole_inline.jsonl").read_text().splitlines()]
    order = random.Random(w.ORDER_SEED)
    expected = [b for _ in range(w.PAIRS) for b in order.sample(["base", "head"], 2)]
    assert [x["build"] for x in rows] == expected and len({tuple(expected[i : i + 2]) for i in range(0, 40, 2)}) == 2


def test_wall_r7_compare_verdict_applies_both_limits():
    def rows(target, other):
        return [{"case": c, "status": "measured", "ci95": target if c == "cartpole_inline" else other} for c in w.CASES]

    good = w.compare_verdict(rows([-0.08, -0.04], [-0.01, 0.01]))
    assert good == {"complete": True, "target_met": True, "no_regression": True}
    assert w.compare_verdict(rows([-0.08, -0.03], [-0.01, 0.01]))["target_met"] is False
    assert w.compare_verdict(rows([-0.08, -0.04], [0.04, 0.05]))["no_regression"] is False
    assert w.compare_verdict([{"case": "x", "status": "invalid"}])["complete"] is False


def test_wall_r7_records_pmset_before_and_after(tmp_path):
    rec = case(Fake(tmp_path), tmp_path)
    assert len(rec["pmset_before"]) == 2 and len(rec["pmset_after"]) == 2


def test_wall_r7_awake_ends_its_child(tmp_path):
    host = Fake(tmp_path)
    seen = []
    real = w.subprocess.Popen

    def popen(*a, **k):
        seen.append(real(*a, **k))
        return seen[-1]

    w.subprocess.Popen = popen
    try:
        with w.awake(host):
            assert seen[0].poll() is None
    finally:
        w.subprocess.Popen = real
    assert seen[0].poll() is not None


def test_wall_r7_half_width_limit_is_one_and_a_half_percent():
    x = [(-1) ** i * 0.01 * (1 + i / 10) for i in range(20)]
    v = w.aa_verdict(x)
    assert w.AA_HALF_WIDTH_LIMIT == 0.015 < v["half_width"] < 0.03
    assert v["contains_zero"] and not v["passed"]


def test_wall_r7_frozen_constants():
    assert (w.PAIRS, w.ORDER_SEED, w.BOOT_SEED, w.RESAMPLES) == (20, 20261106, 20261107, 10_000)
    assert (w.LOAD_LIMIT, w.QUIET_PROCESS_LIMIT, w.MEAN_FOREIGN_LIMIT, w.INTERVAL_FOREIGN_LIMIT) == (1.0, 0.25, 0.5, 1.5)
    assert w.CASES == ("cartpole_inline", "cartpole_w2", "tag_inline", "connect4_inline")


def test_wall_r7_quiet_readings_are_spaced(tmp_path):
    import time as t

    host = Fake(tmp_path)
    host.quiet_gap_s = 0.05
    stamps = []
    original = host.read

    def read(command):
        stamps.append(t.monotonic())
        return original(command)

    host.read = read
    assert w.quiet_refusal(host) is None
    assert all(b - a >= host.quiet_gap_s for a, b in zip(stamps, stamps[1:]))


def test_wall_r7_session_runs_the_configured_caffeinate(tmp_path, monkeypatch):
    import os
    import time as t

    pid_file = tmp_path / "caff.pid"
    host = Fake(tmp_path)
    host.caffeinate = [script(tmp_path / "caff2", f"echo $$ > {pid_file}\nexec sleep 60\n")]
    alive = []

    def fake(_host, _b, _h, c, _o):
        for _ in range(200):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            t.sleep(0.01)
        os.kill(int(pid_file.read_text()), 0)
        alive.append(c)
        return {"case": c, "status": "invalid", "problems": ["x"]}

    monkeypatch.setattr(w, "run_case", fake)
    w.aa(host, "b", str(tmp_path))
    assert len(alive) == 4
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_wall_r7_second_case_waits_for_the_first_cases_load(tmp_path):
    host = Fake(tmp_path)
    load = {"n": 0}

    def decaying():
        load["n"] += 1
        return 4.0 if load["n"] < 5 else 0.1

    host.load = decaying
    rec = case(host, tmp_path)
    assert rec["status"] == "measured", rec
    assert rec["settle_s"] > 0


def test_wall_r7_load_that_never_settles_is_refused_with_its_wait(tmp_path):
    rec = case(Fake(tmp_path, load=2.0), tmp_path)
    assert rec["status"] == "refused" and "load" in rec["reason"] and rec["settle_s"] >= 0.2


def test_wall_r7_settle_timeout_is_frozen():
    assert w.SETTLE_TIMEOUT_S == 600.0


@pytest.mark.parametrize("body", ["", "echo '[1]'", "echo '{\"ns\": 0, \"digest\": \"d\"}'"])
def test_wall_r7_malformed_harness_output_invalidates(tmp_path, body):
    bad = script(tmp_path / "bad", body + "\n")
    rec = case(Fake(tmp_path), tmp_path, base=bad)
    assert rec["status"] == "invalid" and any("harness failed" in p for p in rec["problems"])


def test_wall_r7_never_overwrites_raw_evidence(tmp_path):
    (tmp_path / "cartpole_inline.jsonl").write_text("old")
    rec = case(Fake(tmp_path), tmp_path)
    assert rec["status"] == "refused" and "exists" in rec["reason"]
    assert (tmp_path / "cartpole_inline.jsonl").read_text() == "old"


def test_wall_r7_each_case_is_kept_when_it_ends(tmp_path, monkeypatch):
    def fake(_host, _b, _h, c, _o):
        if c == "tag_inline":
            raise RuntimeError("crash")
        return {"case": c, "status": "invalid", "problems": ["x"]}

    monkeypatch.setattr(w, "run_case", fake)
    with pytest.raises(RuntimeError):
        w.aa(Fake(tmp_path), "b", str(tmp_path))
    assert (tmp_path / "cartpole_w2.result.json").exists()


def test_wall_r7_rerun_leaves_the_first_results_intact(tmp_path):
    first = {"case": "tag_inline", "status": "measured"}
    w.keep(first, str(tmp_path))
    saved = (tmp_path / "tag_inline.result.json").read_bytes()
    w.keep({"case": "tag_inline", "status": "refused"}, str(tmp_path))
    assert (tmp_path / "tag_inline.result.json").read_bytes() == saved


def test_wall_r7_settle_is_bounded(tmp_path):
    host = Fake(tmp_path)
    calls = {"n": 0}

    def high():
        # An unbounded loop fails here instead of hanging the suite.
        calls["n"] += 1
        assert calls["n"] < 5000, "settle did not stop at its timeout"
        return 2.0

    host.load = high
    waited = w.settle(host)
    assert host.settle_timeout_s <= waited <= host.settle_timeout_s + 10 * host.settle_poll_s + 0.5


def test_wall_r7_fake_host_never_reads_the_real_machine(tmp_path, monkeypatch):
    monkeypatch.setattr(w.os, "getloadavg", lambda: pytest.fail("real load read"))
    host = Fake(tmp_path)
    with pytest.raises(AssertionError, match="leaked"):
        host.read(["ps", "-A", "-o", "comm="])
    assert host.load() == 0.1 and host.runners() == []
    for command in (host.powermetrics, host.ps, host.pmset, host.pmset_power, host.caffeinate):
        assert command[0].startswith(str(tmp_path))
    rec = case(host, tmp_path)
    assert rec["status"] == "measured"
