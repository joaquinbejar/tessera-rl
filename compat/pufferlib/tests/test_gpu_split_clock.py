# SPDX-License-Identifier: MIT OR Apache-2.0
"""Clock validation of a gate session (ADR-0063, docs/reviews/152-adr0063-harness-plan.md r3),
on synthetic streams and local fakes: no GPU, no Windows, no ssh.

Covered:
- the sealed driver: rows only in --rows, no progress on stdout, a status
  file with four fields and the timed span on guest monotonic;
- summarise.py: refusals, ADR-0061 A1 first, the clock outcomes, rerun_of,
  and a self-hash mismatch written as a row;
- decide: each condition of ADR-0063 item 4, a shared fault that only the
  outside reference sees, a fault winning over missing coverage, read
  merging, the first-sample exclusion, the pre-check gate, the QPC axis and
  the sampler's CPU share;
- the sampler with a fake host loop (end markers, context rows, a dead host
  reader), run_quiet leaving the parent's stdin alone, and the reference
  client against a fake loop (monotonic stamps, a slow reply, a lost session).
"""

import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import textwrap
import time

import pytest

HARNESS = pathlib.Path(__file__).parents[1] / "gpu_split"
sys.path.insert(0, str(HARNESS))

import clock  # noqa: E402
import clock_ref  # noqa: E402
import manifest  # noqa: E402
import run as driver  # noqa: E402
import summarise  # noqa: E402

FREQ = 10_000_000
CMD = "python3 /r/clock_ref.py --out r.jsonl"
MANIFEST = {
    "clock_py_sha256": "a" * 64,
    "clock_ref_py_sha256": "b" * 64,
    "clock_ref_host": "ref.example",
    "clock_ref_command": CMD,
    "clock_cpu": 3,
}


def scaled(t0, f):
    """A clock equal to true time until t0, then running at rate f."""
    return lambda t: t if t < t0 else t0 + (t - t0) * f


def ident(t):
    return t


def session(pre=320.0, timed=600.0, post=320.0, step=2.0, qpc=ident, perf=None, ref_rtt=0.02, ref_every=5.0,
            drop=(), ref_drop=(), sources=("tsc", "tsc", "tsc", "tsc"), speeds=None, end=True, host_end=True,
            ref_end=True, events=(), first_long=False, cpu_rate=0.001, scope="session"):  # fmt: skip
    """Synthetic sampler and reference streams on true time t (the
    reference's monotonic clock). ``qpc`` maps t to host QPC seconds and
    ``perf`` maps QPC seconds to guest perf seconds (default: equal)."""
    perf = perf or ident
    total = pre + timed + post if scope == "session" else pre
    rows = [{"kind": "header", "clock_py_sha256": MANIFEST["clock_py_sha256"], "cpu": 3}]
    speeds = speeds or ["3000"] * 4
    ctx_at = [0.0, pre, pre + timed, total]

    def ctx(k):
        rows.append({"kind": "context", "at": str(k), "mono": perf(qpc(ctx_at[k])), "current_clocksource": sources[k],
                     "cpu_mhz_current_max": [speeds[k], "3000"], "host_cpu_s": 0.0005 * ctx_at[k]})  # fmt: skip

    ctx(0)
    i, t = 0, 0.0
    while t <= total + 1e-9:
        tag = "pre" if t < pre else "session" if t < pre + timed else "post"
        if scope == "pre-check" and tag != "pre":
            break
        if t == pre and len(rows) and scope == "session":
            ctx(1)
        if t == pre + timed and scope == "session":
            ctx(2)
        if not any(a <= t < b for a, b in drop):
            q = qpc(t)
            g = perf(q)
            w = 0.25 if (first_long and i == 0) else 0.001
            before = {k: g - w for k in ("perf", "mono", "raw", "real")}
            after = {k: g + w for k in ("perf", "mono", "raw", "real")}
            smp = {"kind": "sample", "i": i, "tag": tag, "guest_before": before, "guest_after": after,
                   "host": {"qpc": round(q * FREQ), "qpc_freq": FREQ, "utc_ticks": round(t * 1e7)},
                   "host_bracket_perf": [g - 0.0005, g + 0.0005], "rusage": [cpu_rate * t, 0.0]}  # fmt: skip
            rows.append(smp)
            i += 1
        t = round(t + step, 6)
    if scope == "session":
        ctx(3)
        if host_end:
            rows.append({"kind": "event", "event": "host_end"})
        if end:
            rows.append({"kind": "event", "event": "end"})
    for e in events:
        rows.append({"kind": "event", "event": e})
    ref = [{"kind": "header", "clock_ref_py_sha256": MANIFEST["clock_ref_py_sha256"], "host": "ref.example",
            "command": ["clock_ref.py", "--out", "r.jsonl"]}]  # fmt: skip
    t = 0.0
    while t <= total + 1e-9:
        if not any(a <= t < b for a, b in ref_drop):
            ref.append({"kind": "read", "t0": t - ref_rtt / 2, "t1": t + ref_rtt / 2, "qpc": round(qpc(t) * FREQ),
                        "qpc_freq": FREQ, "utc_ticks": 0, "usable": ref_rtt <= 0.25})  # fmt: skip
        if t % 60 == 0:
            ref.append({"kind": "cpu", "t": t, "cpu_s": 0.0002 * t})
        t = round(t + ref_every, 6)
    if ref_end and scope == "session":
        ref.append({"kind": "event", "event": "end"})
    status = {"timed_start_mono": perf(qpc(pre)) + 0.5, "timed_end_mono": perf(qpc(pre + timed)) - 0.5}
    return rows, ref, status


def decide(scope="session", manifest=MANIFEST, **kw):
    rows, ref, status = session(scope=scope, **kw)
    return clock.decide(rows, ref, manifest, "f" * 64, scope, status if scope == "session" else None)


# ------------------------------------------------------------- decide


def test_clock_decide_nominal_everywhere_is_clock_valid():
    d = decide()
    assert d["outcome"] == "clock-valid" and d["conditions"] == {}
    assert d["coverage"]["external"]["longest_gap_s"] <= clock.COVER_MAX_GAP_S


def test_clock_decide_perf_at_13_12_in_the_timed_span_is_b():
    d = decide(perf=scaled(500.0, 13 / 12))
    assert d["outcome"] == "clock-invalid" and "b" in d["conditions"] and "c" not in d["conditions"]


def test_clock_decide_a_shared_fault_is_seen_only_by_the_outside_reference():
    # guest perf and QPC both run at 13/12 of the reference: (b) stays nominal.
    d = decide(qpc=scaled(500.0, 13 / 12))
    assert "b" not in d["conditions"]
    assert d["outcome"] == "clock-invalid" and "c" in d["conditions"]


def test_clock_decide_a_clocksource_change_is_e():
    d = decide(sources=("tsc", "tsc", "hyperv_clocksource_tsc_page", "tsc"))
    assert d["outcome"] == "clock-invalid" and "e" in d["conditions"]


def test_clock_decide_a_processor_speed_change_alone_stays_valid():
    d = decide(speeds=["3000", "2200", "4100", "3000"])
    assert d["outcome"] == "clock-valid"
    assert d["reported"]["cpu_mhz_current_max"][1] == ["2200", "3000"]


def test_clock_decide_a_utc_step_alone_stays_valid():
    rows, ref, status = session()
    for r in rows:
        if r.get("kind") == "sample" and r["i"] > 300:
            r["host"]["utc_ticks"] += 20_000_000
    d = clock.decide(rows, ref, MANIFEST, "f" * 64, "session", status)
    assert d["outcome"] == "clock-valid"
    assert "other" in {w["band"] for w in d["reported"]["utc"]}


def test_clock_decide_a_40_s_reference_gap_is_d1():
    d = decide(ref_drop=[(700.0, 740.0)])
    assert d["outcome"] == "instrumentation failed" and "d1" in d["conditions"]
    assert "gap" in {e["band"] for e in d["external"]}


def test_clock_coverage_counts_short_gaps_on_the_qpc_axis():
    # 6% of the span uncovered in 6 s gaps: over the share, under the longest stretch.
    covered = [(t + 6.0, t + 100.0) for t in range(0, 1000, 100)]
    c = clock.coverage(0.0, 1000.0, covered)
    assert c["uncovered_share"] == pytest.approx(0.06) and c["longest_gap_s"] == pytest.approx(6.0)
    assert c["uncovered_share"] > clock.COVER_MAX_SHARE


def test_clock_decide_missing_sampler_coverage_is_d2():
    d = decide(drop=[(700.0, 740.0)])
    assert d["outcome"] == "instrumentation failed" and "d2" in d["conditions"]


@pytest.mark.parametrize("kw", [{"end": False}, {"host_end": False}, {"ref_end": False}, {"events": ["host_dead"]}],
                         ids=["sampler", "host-end", "reference", "host-dead"])  # fmt: skip
def test_clock_decide_a_dead_stream_is_d3(kw):
    d = decide(**kw)
    assert d["outcome"] == "instrumentation failed" and "d3" in d["conditions"]


def test_clock_decide_a_lost_reference_session_is_d3():
    rows, ref, status = session()
    ref.insert(-1, {"kind": "event", "event": "lost"})
    d = clock.decide(rows, ref, MANIFEST, "f" * 64, "session", status)
    assert "d3" in d["conditions"]


def test_clock_decide_a_short_pre_check_is_a():
    d = decide(pre=290.0)
    assert d["outcome"] == "instrumentation failed" and "a" in d["conditions"]


def test_clock_decide_a_fault_wins_over_missing_coverage():
    d = decide(perf=scaled(500.0, 13 / 12), ref_drop=[(700.0, 740.0)])
    assert "b" in d["conditions"] and "d1" in d["conditions"]
    assert d["outcome"] == "clock-invalid"


def test_clock_decide_a_self_hash_mismatch():
    other = {**MANIFEST, "clock_py_sha256": "0" * 64}
    d = decide(manifest=other)
    assert d["outcome"] == "instrumentation failed"
    assert d["conditions"]["selfhash"] == ["clock.py sha256 mismatch"]
    assert decide(scope="pre-check", manifest=other)["decision"] == "stop"


@pytest.mark.parametrize("rtt, precise, length", [(0.12, True, (20.0, 30.0)), (0.2, False, (30.0, 30.0))])
def test_clock_external_reads_are_merged_until_precise_up_to_30_s(rtt, precise, length):
    _, ref, _ = session(ref_rtt=rtt)
    ext = clock.external_intervals(clock.ref_points(ref))
    assert ext and all((e["band"] != "imprecise") == precise for e in ext)
    assert all(length[0] <= e["dt_s"] <= length[1] + 1e-6 for e in ext)


def test_clock_reads_with_a_long_round_trip_are_unusable():
    _, ref, _ = session(ref_rtt=0.3)
    assert clock.ref_points(ref) == []
    assert decide(ref_rtt=0.3)["outcome"] == "instrumentation failed"


def test_clock_the_first_sample_exclusion_applies_to_mono_raw_real_only():
    rows, _, _ = session(first_long=True)
    g = clock.guest_intervals([r for r in rows if r.get("kind") == "sample"])
    assert g[0]["band_mono"] == g[0]["band_raw"] == g[0]["band_real"] == "unclassified"
    assert g[0]["band_perf"] == "nominal"
    assert g[1]["band_mono"] == "nominal"


def test_clock_pre_check_gate():
    assert decide(scope="pre-check")["decision"] == "go"
    assert decide(scope="pre-check", ref_rtt=0.3)["decision"] == "stop"
    assert decide(scope="pre-check", perf=scaled(100.0, 13 / 12))["decision"] == "stop"


def test_clock_status_values_map_onto_the_qpc_axis():
    # guest monotonic = QPC + 1000 s: the timed span maps back to QPC [320.5, 919.5].
    d = decide(perf=lambda q: q + 1000.0)
    lo, hi = d["timed_qpc_s"]
    assert lo == pytest.approx(320.5) and hi == pytest.approx(919.5)


def test_clock_sampler_cpu_share_over_the_timed_span():
    d = decide(cpu_rate=0.02)
    assert d["cpu"]["sampler_share"] == pytest.approx(0.02, rel=0.01) and d["cpu"]["over_budget"] is True
    assert d["outcome"] == "clock-valid"
    assert d["cpu"]["host_reader_share"] == pytest.approx(0.0005)
    assert d["cpu"]["reference_loop_share"] == pytest.approx(0.0002)
    assert decide()["cpu"]["over_budget"] is False


# ----------------------------------------------------------- summarise


def write(path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def gate_rows(rc=0, drop=0):
    rows = []
    for r, i, c, m in driver.from_order(manifest.order(manifest.CASES, manifest.MODES, 1)):
        res = {"sps": 2.0 if c.startswith("tessera") else 1.0, "advantage_backend": "cuda", "scale": "full"}
        if m != "env":
            res["epochs_timed"] = driver.GATE_EPOCHS_TIMED
        if c.startswith("tessera"):
            res["tessera_rl"] = {"native_sha256": "e" * 64}
        rows.append({"round": r, "position": i, "case": c, "mode": m, "rc": 0, "group_left": [], "device": "cuda",
                     "result": res})  # fmt: skip
    rows[0]["rc"] = rc
    return rows[: len(rows) - drop]


def gate_manifest(**over):
    m = {"commit": "c" * 40, "wheel_sha256": "w" * 64, "cuda_toolkit": "13.2.86", "native_sha256": "e" * 64,
         "order": manifest.order(manifest.CASES, manifest.MODES, 1), **MANIFEST, "rerun_of": None}  # fmt: skip
    m.update(over)
    return m


def run_summarise(tmp_path, rows=None, m=None, outcome="clock-valid", decide_over=None, earlier=None, name="s"):
    d = tmp_path / name
    d.mkdir()
    rows = gate_rows() if rows is None else rows
    m = m or gate_manifest()
    rsha = write(d / "rows.jsonl", "".join(json.dumps(r) + "\n" for r in rows).encode())
    msha = write(d / "m.json", manifest.canonical(m))
    dec = {"scope": "session", "manifest_sha256": msha, "outcome": outcome, "conditions": {},
           "instruments": {"clock_py_sha256": m["clock_py_sha256"], "clock_ref_py_sha256": m["clock_ref_py_sha256"],
                           "clock_ref_host": "ref.example", "clock_ref_args": ["--out", "r.jsonl"], "clock_cpu": 3}}  # fmt: skip
    for k, v in (decide_over or {}).items():
        if k == "instruments":
            dec["instruments"].update(v)
        else:
            dec[k] = v
    dsha = write(d / "decide.json", json.dumps(dec).encode())
    argv = ["--rows", str(d / "rows.jsonl"), "--rows-sha256", rsha, "--manifest", str(d / "m.json"),
            "--manifest-sha256", msha, "--clock-decide", str(d / "decide.json"), "--clock-decide-sha256", dsha,
            "--sent-at", "2026-11-17T12:00:00Z", "--out", str(d / "summary.jsonl")]  # fmt: skip
    if earlier:
        argv += ["--rerun-of-manifest", earlier]
    rc = summarise.main(argv)
    out = d / "summary.jsonl"
    return rc, (json.loads(out.read_text()) if out.exists() else None), msha


def test_summarise_clock_valid_is_todays_summary(tmp_path):
    rc, s, msha = run_summarise(tmp_path)
    assert rc == 0 and s["clock"] == "clock-valid" and s["attempt"] == "complete"
    plain = driver.summary(gate_rows(), "cuda", True, gate_manifest(), msha)
    assert s["cells"] == plain["cells"] and s["pass"] is plain["pass"] is True


def test_summarise_clock_invalid_is_a_void_that_keeps_its_cells(tmp_path):
    rc, s, _ = run_summarise(tmp_path, outcome="clock-invalid")
    assert rc == 0 and s["attempt"] == "void: clock-invalid" and s["gate_eligible"] is False
    assert not set(summarise.VERDICT_KEYS) & set(s)
    assert s["cells"]["tessera-w1/e2e"]["median"] == 2.0


def test_summarise_instrumentation_failed_is_a_failed_attempt(tmp_path):
    rc, s, _ = run_summarise(tmp_path, outcome="instrumentation failed")
    assert rc == 0 and s["attempt"] == "failed, no result" and "pass" not in s


@pytest.mark.parametrize("rows", [gate_rows(drop=1), gate_rows(rc=1)], ids=["missing-run", "rc"])
@pytest.mark.parametrize("outcome", ["clock-valid", "clock-invalid"])
def test_summarise_applies_adr0061_a1_first(tmp_path, rows, outcome):
    rc, s, _ = run_summarise(tmp_path, rows=rows, outcome=outcome)
    assert rc == 0 and s["attempt"] == "failed, no result" and "pass" not in s
    assert s["clock"] == outcome


def test_summarise_a_stopped_group_is_incomplete(tmp_path):
    rows = gate_rows()
    rows[-1]["group_left"] = [123]
    rc, s, _ = run_summarise(tmp_path, rows=rows, outcome="clock-invalid")
    assert s["attempt"] == "failed, no result"


def test_summarise_writes_a_self_hash_mismatch_as_a_failed_row(tmp_path):
    over = {"outcome": "instrumentation failed", "conditions": {"selfhash": ["clock.py sha256 mismatch"]},
            "instruments": {"clock_py_sha256": "0" * 64}}  # fmt: skip
    rc, s, _ = run_summarise(tmp_path, outcome="instrumentation failed", decide_over=over)
    assert rc == 0 and s["attempt"] == "failed, no result"


@pytest.mark.parametrize(
    "over",
    [{"scope": "pre-check"}, {"manifest_sha256": "0" * 64}, {"instruments": {"clock_ref_host": "other"}},
     {"instruments": {"clock_ref_args": ["--out", "x"]}}, {"instruments": {"clock_cpu": None}},
     {"instruments": {"clock_py_sha256": "0" * 64}}],
    ids=["scope", "manifest", "host", "command", "cpu", "unreported-selfhash"],
)  # fmt: skip
def test_summarise_refuses_a_mismatch(tmp_path, capsys, over):
    rc, s, _ = run_summarise(tmp_path, decide_over=over)
    assert rc == summarise.EXIT_REFUSED and s is None
    assert "REFUSED" in capsys.readouterr().out


def test_summarise_refuses_a_wrong_rows_sha256(tmp_path, capsys):
    rc, _, _ = run_summarise(tmp_path)
    d = tmp_path / "s"
    argv = ["--rows", str(d / "rows.jsonl"), "--rows-sha256", "0" * 64, "--manifest", str(d / "m.json"),
            "--manifest-sha256", hashlib.sha256((d / "m.json").read_bytes()).hexdigest(),
            "--clock-decide", str(d / "decide.json"),
            "--clock-decide-sha256", hashlib.sha256((d / "decide.json").read_bytes()).hexdigest(),
            "--sent-at", "x", "--out", str(d / "again.jsonl")]  # fmt: skip
    assert summarise.main(argv) == summarise.EXIT_REFUSED and not (d / "again.jsonl").exists()


def earlier_manifest(tmp_path, name, m):
    data = manifest.canonical(m)
    (tmp_path / name).write_bytes(data)
    return str(tmp_path / name), hashlib.sha256(data).hexdigest()


def test_summarise_rerun_positions(tmp_path):
    first, fsha = earlier_manifest(tmp_path, "first.json", gate_manifest())
    second_m = gate_manifest(rerun_of={"manifest_sha256": fsha, "position": 2})
    rc, s, _ = run_summarise(tmp_path, m=second_m, outcome="clock-invalid", earlier=first, name="a")
    assert rc == 0 and s["attempt"] == "void: clock-invalid" and s["rerun_position"] == 2
    second, ssha = earlier_manifest(tmp_path, "second.json", second_m)
    third_m = gate_manifest(rerun_of={"manifest_sha256": ssha, "position": 3})
    rc, s, _ = run_summarise(tmp_path, m=third_m, outcome="clock-invalid", earlier=second, name="b")
    assert rc == 0 and s["attempt"] == "failed, no result"


@pytest.mark.parametrize("change", [{"commit": "d" * 40}, {"wheel_sha256": "x" * 64}, {"cuda_toolkit": "13.0.88"},
                                    {"order": []}, {"position": 3}], ids=["commit", "wheel", "toolkit", "order", "skip"])  # fmt: skip
def test_summarise_refuses_a_rerun_that_changes_the_attempt(tmp_path, change):
    pos = change.pop("position", 2)
    first, fsha = earlier_manifest(tmp_path, "first.json", gate_manifest(**change))
    m = gate_manifest(rerun_of={"manifest_sha256": fsha, "position": pos})
    rc, s, _ = run_summarise(tmp_path, m=m, outcome="clock-invalid", earlier=first)
    assert rc == summarise.EXIT_REFUSED and s is None


# -------------------------------------------------------- the driver


def sealed_attempt(tmp_path, capsys, monkeypatch, script="exit 1"):
    from test_gpu_split import make_manifest

    monkeypatch.setattr(driver, "HEARTBEAT_S", 0.05)
    wheel, argv = make_manifest(tmp_path, **{"--rounds": "1"})
    assert manifest.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    rows, status = tmp_path / "rows.jsonl", tmp_path / "status.json"
    rc = driver.main(["--python", f"sh -c '{script}' fake {wheel}", "--rows", str(rows), "--status", str(status),
                      "--device", "cpu", "--manifest", str(tmp_path / "manifest.json"), "--manifest-sha256", digest,
                      "--wheel", str(wheel)])  # fmt: skip
    return rc, capsys.readouterr().out, rows, status, digest


def test_driver_sealed_rows_leak_no_progress(tmp_path, capsys, monkeypatch):
    # One run fails (rc 1), the others pass slowly enough for heartbeats.
    script = 'case "$3" in ocean-serial) exit 1;; esac; sleep 0.06; exit 0'
    rc, out, rows, status, digest = sealed_attempt(tmp_path, capsys, monkeypatch, script)
    assert rc == 1
    lines = out.splitlines()
    assert lines[0] == f"started manifest={digest}"
    sha = hashlib.sha256(rows.read_bytes()).hexdigest()
    assert lines[-1] == f"rows sealed: {sha}"
    assert set(lines[1:-1]) <= {"alive"} and "alive" in lines
    for word in (*manifest.CASES, *manifest.MODES, "rc=", "sps", "status"):
        assert word not in "\n".join(lines[1:-1])
    assert oct(rows.stat().st_mode & 0o777) == "0o600"
    got = [json.loads(x) for x in rows.read_text().splitlines()]
    assert len(got) == len(manifest.CASES) * len(manifest.MODES) and all(r["log"] for r in got)
    st = json.loads(status.read_text())
    assert set(st) == {"rows_sha256", "manifest_sha256", "timed_start_mono", "timed_end_mono"}
    assert st["rows_sha256"] == sha and st["timed_start_mono"] < st["timed_end_mono"]


def test_driver_writes_timed_start_before_the_first_run_ends(tmp_path, capsys, monkeypatch):
    seen = tmp_path / "seen.json"
    script = f'test -e {seen} || cp {tmp_path / "status.json"} {seen}; exit 0'
    sealed_attempt(tmp_path, capsys, monkeypatch, script)
    early = json.loads(seen.read_text())
    assert early["timed_start_mono"] is not None and early["timed_end_mono"] is None


def test_driver_refuses_out_or_missing_rows_in_a_manifest_attempt(tmp_path, capsys):
    from test_gpu_split import make_manifest

    wheel, argv = make_manifest(tmp_path)
    manifest.main(argv)
    digest = capsys.readouterr().out.strip()
    common = ["--python", f"sh -c 'exit 0' fake {wheel}", "--device", "cpu", "--manifest",
              str(tmp_path / "manifest.json"), "--manifest-sha256", digest, "--wheel", str(wheel)]  # fmt: skip
    with pytest.raises(SystemExit):
        driver.main([*common, "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit):
        driver.main([*common, "--rows", str(tmp_path / "r.jsonl")])


# ---------------------------------------------------------- manifest


def test_manifest_records_the_clock_instruments_and_rerun_of(tmp_path, capsys):
    from test_gpu_split import make_manifest

    wheel, argv = make_manifest(tmp_path, **{"--clock-cpu": "3"})
    assert manifest.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    m = manifest.load(str(tmp_path / "manifest.json"), digest)
    assert m["clock_py_sha256"] == manifest.file_sha256(str(HARNESS / "clock.py"))
    assert m["clock_ref_py_sha256"] == manifest.file_sha256(str(HARNESS / "clock_ref.py"))
    assert (m["clock_ref_host"], m["clock_cpu"], m["case_cpus"], m["rerun_of"]) == ("ref.example", 3, None, None)
    _, argv2 = make_manifest(tmp_path, **{"--out": str(tmp_path / "m2.json"), "--clock-cpu": "3",
                                          "--rerun-of-manifest": str(tmp_path / "manifest.json"),
                                          "--rerun-of-sha256": digest})  # fmt: skip
    assert manifest.main(argv2) == 0
    d2 = capsys.readouterr().out.strip()
    assert manifest.load(str(tmp_path / "m2.json"), d2)["rerun_of"] == {"manifest_sha256": digest, "position": 2}
    _, argv3 = make_manifest(tmp_path, **{"--out": str(tmp_path / "m3.json"), "--rounds": "3",
                                          "--rerun-of-manifest": str(tmp_path / "manifest.json"),
                                          "--rerun-of-sha256": digest})  # fmt: skip
    assert manifest.main(argv3) == manifest.EXIT_MISMATCH  # the order changed
    assert "REFUSED" in capsys.readouterr().out


# ------------------------------------------- sampler, run_quiet, client

FAKE_LOOP = textwrap.dedent(
    """
    import sys, time
    n = 0
    for line in sys.stdin:
        line = line.strip()
        if line == "end":
            print("end", flush=True)
            break
        if line == "cpu":
            print(time.process_time(), flush=True)
            continue
        n += 1
        if MODE == "slow" and n == 3:
            time.sleep(0.4)
        if MODE == "die" and n == 3:
            sys.exit(0)
        print(time.monotonic_ns() // 100, 10000000, time.time_ns() // 100, flush=True)
    """
)


def fake_loop(tmp_path, mode="ok"):
    p = tmp_path / f"loop_{mode}.py"
    p.write_text(f"MODE = {mode!r}\n" + FAKE_LOOP)
    return f"{sys.executable} -I {p}"


def test_clock_and_clock_ref_share_one_loop():
    assert clock.LOOP == clock_ref.LOOP


def test_clock_sampler_with_a_fake_host(tmp_path):
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    (ctl / "tag").write_text("pre")
    out = tmp_path / "samples.jsonl"
    p = subprocess.Popen([sys.executable, "-I", str(HARNESS / "clock.py"), "sample", "--out", str(out), "--control",
                          str(ctl), "--interval", "0.05", "--host-cmd", fake_loop(tmp_path)],
                         stdin=subprocess.DEVNULL)  # fmt: skip
    for tag in ("session", "post"):
        time.sleep(0.4)
        (ctl / "tag").write_text(tag)
    time.sleep(0.4)
    (ctl / "stop").write_text("")
    assert p.wait(timeout=60) == 0
    rows = clock.read_jsonl(str(out))
    assert rows[0]["kind"] == "header" and rows[0]["clock_py_sha256"] == clock.self_sha256(str(HARNESS / "clock.py"))
    ctx = [r for r in rows if r["kind"] == "context"]
    assert [c["at"] for c in ctx] == ["start pre", "pre -> session", "session -> post", "stop post"]
    assert all(c["host_cpu_s"] is not None for c in ctx)
    samples = [r for r in rows if r["kind"] == "sample"]
    assert {s["tag"] for s in samples} == {"pre", "session", "post"} and all(s["host"] for s in samples)
    assert not any("cuda" in json.dumps(s) for s in samples)
    assert [r["event"] for r in rows if r["kind"] == "event"] == ["host_end", "end"]


def test_clock_sampler_records_a_dead_host_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(clock, "HOST_TIMEOUT_S", 0.2)
    s = clock.Sampler(str(tmp_path / "s.jsonl"), shlex.split(fake_loop(tmp_path, "slow")), None)
    for _ in range(4):
        s.sample("pre")
    rows = clock.read_jsonl(str(tmp_path / "s.jsonl"))
    assert {"kind": "event", "event": "host_dead", "why": "timeout", "i": 2} in rows
    assert rows[-1]["host"] is None


def test_clock_run_quiet_leaves_the_parents_stdin_alone(tmp_path):
    child = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(HARNESS)!r})
        import clock
        clock.run_quiet(["cat"])
        print(sys.stdin.read().strip())
        """
    )
    p = subprocess.run([sys.executable, "-I", "-c", child], input="sentinel\n", capture_output=True, text=True, timeout=60)
    assert p.stdout.strip() == "sentinel"


def run_ref(tmp_path, mode, seconds):
    out, stop = tmp_path / f"ref_{mode}.jsonl", tmp_path / f"stop_{mode}"
    p = subprocess.Popen([sys.executable, "-I", str(HARNESS / "clock_ref.py"), "--out", str(out), "--host",
                          "ref.example", "--stop-file", str(stop), "--every", "0.05", "--cpu-every", "0.2",
                          "--reply-timeout", "2", "--command", fake_loop(tmp_path, mode)])  # fmt: skip
    time.sleep(seconds)
    stop.write_text("")
    rc = p.wait(timeout=60)
    return rc, clock.read_jsonl(str(out))


def test_clock_ref_stamps_reads_with_monotonic_and_ends_cleanly(tmp_path):
    rc, rows = run_ref(tmp_path, "ok", 0.6)
    assert rc == 0 and rows[0]["clock"] == "time.monotonic"
    assert rows[0]["clock_ref_py_sha256"] == manifest.file_sha256(str(HARNESS / "clock_ref.py"))
    reads = [r for r in rows if r["kind"] == "read"]
    assert reads and all(r["t0"] < r["t1"] and r["usable"] for r in reads)
    assert any(r["kind"] == "cpu" for r in rows)
    assert rows[-1] == {"kind": "event", "event": "end", "loop_ended": True}


def test_clock_ref_marks_a_slow_reply_unusable(tmp_path):
    _, rows = run_ref(tmp_path, "slow", 0.8)
    reads = [r for r in rows if r["kind"] == "read"]
    assert reads[2]["usable"] is False and reads[2]["t1"] - reads[2]["t0"] > clock_ref.RTT_MAX_S


def test_clock_ref_records_a_lost_session_and_does_not_reconnect(tmp_path):
    rc, rows = run_ref(tmp_path, "die", 0.6)
    assert rc == 3
    assert rows[-1]["event"] == "lost" and sum(1 for r in rows if r["kind"] == "header") == 1
