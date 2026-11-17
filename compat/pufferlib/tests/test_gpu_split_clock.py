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
import socket as socket_mod
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


def session(pre=320.0, timed=600.0, post=320.0, step=2.0, qpc=ident, raw=None, slew=None, ref_rtt=0.02, ref_every=5.0,
            drop=(), ref_drop=(), sources=("tsc", "tsc", "tsc", "tsc"), speeds=None, end=True, host_end=True,
            ref_end=True, events=(), first_long=False, cpu_rate=0.001, scope="session"):  # fmt: skip
    """Synthetic sampler and reference streams on true time t (the
    reference's monotonic clock). ``qpc`` maps t to host QPC seconds; ``raw``
    maps QPC seconds to the guest's CLOCK_MONOTONIC_RAW (the binding clock
    under ADR-0064, default: equal); ``slew`` maps QPC seconds to the
    disciplined guest clocks perf, mono and real (default: equal to raw)."""
    raw = raw or ident
    disc = slew or raw
    total = pre + timed + post if scope == "session" else pre
    rows = [{"kind": "header", "clock_py_sha256": MANIFEST["clock_py_sha256"], "cpu": 3}]
    speeds = speeds or ["3000"] * 4
    ctx_at = [0.0, pre, pre + timed, total]

    def ctx(k):
        rows.append({"kind": "context", "at": str(k), "raw": raw(qpc(ctx_at[k])), "mono": disc(qpc(ctx_at[k])),
                     "current_clocksource": sources[k],
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
            g, d = raw(q), disc(q)
            w = 0.25 if (first_long and i == 0) else 0.001
            before = {"raw": g - w, **{k: d - w for k in ("perf", "mono", "real")}}
            after = {"raw": g + w, **{k: d + w for k in ("perf", "mono", "real")}}
            smp = {"kind": "sample", "i": i, "tag": tag, "guest_before": before, "guest_after": after,
                   "host": {"qpc": round(q * FREQ), "qpc_freq": FREQ, "utc_ticks": round(t * 1e7)},
                   "host_bracket_raw": [g - 0.0005, g + 0.0005], "rusage": [cpu_rate * t, 0.0]}  # fmt: skip
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
    status = {"timed_start_raw": raw(qpc(pre)) + 0.5, "timed_end_raw": raw(qpc(pre + timed)) - 0.5}
    return rows, ref, status


def decide(scope="session", manifest=MANIFEST, **kw):
    rows, ref, status = session(scope=scope, **kw)
    return clock.decide(rows, ref, manifest, "f" * 64, scope, status if scope == "session" else None)


# ------------------------------------------------------------- decide


def test_clock_decide_nominal_everywhere_is_clock_valid():
    d = decide()
    assert d["outcome"] == "clock-valid" and d["conditions"] == {}
    assert d["coverage"]["external"]["longest_gap_s"] <= clock.COVER_MAX_GAP_S


def test_clock_decide_raw_at_13_12_in_the_timed_span_is_b():
    d = decide(raw=scaled(500.0, 13 / 12))
    assert d["outcome"] == "clock-invalid" and "b" in d["conditions"] and "c" not in d["conditions"]
    assert d["binding_guest_clock"] == "CLOCK_MONOTONIC_RAW"


def test_clock_decide_a_perf_only_slew_is_reported_not_judged():
    # ADR-0064: the guest's time agents slew perf_counter (0.987842 in rehearsal 2),
    # not CLOCK_MONOTONIC_RAW; the session stays valid and the slew is reported.
    d = decide(slew=scaled(500.0, 0.987842))
    assert d["outcome"] == "clock-valid" and "b" not in d["conditions"]
    assert "other" in d["reported"]["other_guest_clocks"]["perf"]
    assert d["reported"]["perf_worst"] == pytest.approx(0.012158, abs=1e-4)


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
    assert any(e["gap"] for e in d["external"])


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


def test_clock_decide_a_fault_inside_a_sampler_gap_is_still_judged():
    # raw runs at 13/12 only while the sampler is silent (700 to 740 s).
    def fault(q):
        return q if q < 700 else 700 + min(q - 700, 40) * 13 / 12 + max(q - 740, 0)

    d = decide(raw=fault, drop=[(700.0, 740.0)])
    assert "b" in d["conditions"] and "d2" in d["conditions"]
    assert d["outcome"] == "clock-invalid"


def test_clock_decide_tags_out_of_order_fail_the_instrumentation():
    rows, ref, status = session()
    smp = [r for r in rows if r.get("kind") == "sample"]
    smp[400]["tag"] = "pre"
    d = clock.decide(rows, ref, MANIFEST, "f" * 64, "session", status)
    assert d["outcome"] == "instrumentation failed" and "tags" in d["conditions"]


def test_clock_decide_without_clocksource_readings():
    rows, ref, status = session()
    for r in rows:
        if r.get("kind") == "context":
            r["current_clocksource"] = None
    d = clock.decide(rows, ref, MANIFEST, "f" * 64, "session", status)
    assert d["outcome"] == "instrumentation failed" and "clocksource" in d["conditions"]
    assert decide(scope="pre-check", sources=(None,) * 4)["decision"] == "stop"


@pytest.mark.parametrize("over", [{"clock_ref_host": "other"}, {"clock_cpu": None},
                                  {"clock_ref_command": "python3 /r/clock_ref.py --out x.jsonl"}],
                         ids=["host", "cpu", "args"])  # fmt: skip
def test_clock_decide_checks_the_instruments_against_the_manifest(over):
    m = {**MANIFEST, **over}
    assert decide(manifest=m)["outcome"] == "instrumentation failed"
    pre = decide(scope="pre-check", manifest=m)
    assert pre["decision"] == "stop" and "instruments" in pre["conditions"]


def test_clock_decide_counts_corrupt_lines(tmp_path):
    rows, ref, status = session()
    lines = [json.dumps(r) for r in rows]
    (tmp_path / "trunc.jsonl").write_text("\n".join([*lines, '{"kind": "sam']) + "\n")
    (tmp_path / "mid.jsonl").write_text("\n".join([*lines[:50], "garbage", *lines[50:]]) + "\n")
    ok = clock.decide(clock.read_jsonl(str(tmp_path / "trunc.jsonl")), ref, MANIFEST, "f" * 64, "session", status)
    bad = clock.decide(clock.read_jsonl(str(tmp_path / "mid.jsonl")), ref, MANIFEST, "f" * 64, "session", status)
    assert "corrupt" not in ok["conditions"] and ok["outcome"] == "clock-valid"
    assert bad["outcome"] == "instrumentation failed" and "corrupt" in bad["conditions"]


def test_clock_decide_a_short_pre_check_is_a():
    d = decide(pre=290.0)
    assert d["outcome"] == "instrumentation failed" and "a" in d["conditions"]


def test_clock_decide_a_fault_wins_over_missing_coverage():
    d = decide(raw=scaled(500.0, 13 / 12), ref_drop=[(700.0, 740.0)])
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


def test_clock_the_first_sample_exclusion_applies_to_perf_mono_real_only():
    rows, _, _ = session(first_long=True)
    g = clock.guest_intervals([r for r in rows if r.get("kind") == "sample"])
    assert g[0]["band_perf"] == g[0]["band_mono"] == g[0]["band_real"] == "unclassified"
    assert g[0]["band_raw"] == "nominal"
    assert g[1]["band_perf"] == "nominal"


def test_clock_pre_check_gate():
    assert decide(scope="pre-check")["decision"] == "go"
    assert decide(scope="pre-check", ref_rtt=0.3)["decision"] == "stop"
    assert decide(scope="pre-check", raw=scaled(100.0, 13 / 12))["decision"] == "stop"
    assert decide(scope="pre-check", slew=scaled(100.0, 0.987842))["decision"] == "go"


def test_clock_status_values_map_onto_the_qpc_axis():
    # guest raw = QPC + 1000 s: the timed span (on raw) maps back to QPC [320.5, 919.5].
    d = decide(raw=lambda q: q + 1000.0)
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
        res = {"sps": 2.0 if c.startswith("tessera") else 1.0, "advantage_backend": "cuda", "scale": "full",
               "wall_clock": "CLOCK_MONOTONIC_RAW"}
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
         "order": manifest.order(manifest.CASES, manifest.MODES, 1), **MANIFEST, "rerun_of": None,
         "wall_clock": "CLOCK_MONOTONIC_RAW"}  # fmt: skip
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
           "binding_guest_clock": "CLOCK_MONOTONIC_RAW",
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


@pytest.mark.parametrize("ins", [{"clock_ref_host": "other"}, {"clock_ref_args": ["--out", "x"]}, {"clock_cpu": None}],
                         ids=["host", "command", "cpu"])  # fmt: skip
def test_summarise_writes_a_reported_instruments_mismatch_as_a_failed_row(tmp_path, ins):
    over = {"conditions": {"instruments": ["mismatch"]}, "instruments": ins}
    rc, s, _ = run_summarise(tmp_path, outcome="instrumentation failed", decide_over=over)
    assert rc == 0 and s["attempt"] == "failed, no result" and "pass" not in s


@pytest.mark.parametrize("over", [{"clock_py_sha256": "0" * 64}, {"clock_ref_host": "other"}], ids=["selfhash", "instruments"])
def test_clock_decide_a_wrong_instrument_beats_a_fault(over):
    d = decide(manifest={**MANIFEST, **over}, raw=scaled(500.0, 13 / 12))
    assert "b" in d["conditions"] and d["outcome"] == "instrumentation failed"


@pytest.mark.parametrize("cond", ["selfhash", "instruments"])
def test_summarise_a_wrong_instrument_with_a_fault_is_never_a_void(tmp_path, cond):
    ins = {"clock_py_sha256": "0" * 64} if cond == "selfhash" else {"clock_ref_host": "other"}
    why = "clock.py sha256 mismatch" if cond == "selfhash" else "mismatch"
    over = {"conditions": {cond: [why], "b": ["perf 1.08"]}, "instruments": ins}
    rc, s, _ = run_summarise(tmp_path, outcome="clock-invalid", decide_over=over)
    assert rc == 0 and s["attempt"] == "failed, no result" and s["clock"] == "instrumentation failed"


@pytest.mark.parametrize("cond", ["selfhash", "instruments"])
def test_summarise_refuses_a_wrong_instrument_reported_as_clock_valid(tmp_path, cond):
    over = {"conditions": {cond: ["clock.py sha256 mismatch"]}}
    rc, s, _ = run_summarise(tmp_path, outcome="clock-valid", decide_over=over)
    assert rc == summarise.EXIT_REFUSED and s is None


def test_summarise_refuses_an_unreported_self_hash_even_with_instruments_reported(tmp_path):
    over = {"conditions": {"instruments": ["mismatch"]},
            "instruments": {"clock_ref_host": "other", "clock_py_sha256": "0" * 64}}  # fmt: skip
    rc, s, _ = run_summarise(tmp_path, outcome="instrumentation failed", decide_over=over)
    assert rc == summarise.EXIT_REFUSED and s is None


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
    assert set(st) == {"rows_sha256", "manifest_sha256", "timed_start_raw", "timed_end_raw"}
    assert st["rows_sha256"] == sha and st["timed_start_raw"] < st["timed_end_raw"]


def test_driver_writes_timed_start_before_the_first_run_ends(tmp_path, capsys, monkeypatch):
    # Only the first run of the order copies the status file, through a temp
    # file and a rename, so the test reads a whole copy whatever the timing.
    seen = tmp_path / "seen.json"
    case, mode = manifest.order(manifest.CASES, manifest.MODES, 1)[0][0]
    script = (f'if [ "$3 $4" = "{case} {mode}" ]; then cp {tmp_path / "status.json"} {seen}.tmp && '
              f'mv {seen}.tmp {seen}; fi; exit 0')  # fmt: skip
    sealed_attempt(tmp_path, capsys, monkeypatch, script)
    early = json.loads(seen.read_text())
    final = json.loads((tmp_path / "status.json").read_text())
    assert early["timed_start_raw"] == final["timed_start_raw"] and early["timed_end_raw"] is None


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
    # Later calls build their throwaway wheels elsewhere, so the first wheel's
    # bytes (and sha256) stay as the first manifest recorded them.
    same = {"--wheel": str(wheel), "--wheel-sha256": manifest.file_sha256(str(wheel))}
    other = tmp_path / "other"
    other.mkdir()
    assert manifest.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    m = manifest.load(str(tmp_path / "manifest.json"), digest)
    assert m["clock_py_sha256"] == manifest.file_sha256(str(HARNESS / "clock.py"))
    assert m["clock_ref_py_sha256"] == manifest.file_sha256(str(HARNESS / "clock_ref.py"))
    assert (m["clock_ref_host"], m["clock_cpu"], m["case_cpus"], m["rerun_of"]) == ("ref.example", 3, None, None)
    _, argv2 = make_manifest(other, **same, **{"--out": str(tmp_path / "m2.json"), "--clock-cpu": "3",
                                          "--rerun-of-manifest": str(tmp_path / "manifest.json"),
                                          "--rerun-of-sha256": digest})  # fmt: skip
    assert manifest.main(argv2) == 0
    d2 = capsys.readouterr().out.strip()
    assert manifest.load(str(tmp_path / "m2.json"), d2)["rerun_of"] == {"manifest_sha256": digest, "position": 2}
    _, argv3 = make_manifest(other, **same, **{"--out": str(tmp_path / "m3.json"), "--rounds": "3",
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
            time.sleep(1.5)
        if MODE == "die" and n == 3:
            sys.exit(0)
        print(time.monotonic_ns() // 100, 10000000, time.time_ns() // 100, flush=True)
    """
)


def fake_loop(tmp_path, mode="ok"):
    p = tmp_path / f"loop_{mode}.py"
    p.write_text(f"MODE = {mode!r}\n" + FAKE_LOOP)
    return f"{sys.executable} -I {p}"


def set_tag(ctl, tag):
    """The documented way to change the tag: write a temp file, then rename."""
    (ctl / "tag.tmp").write_text(tag)
    (ctl / "tag.tmp").replace(ctl / "tag")


@pytest.mark.parametrize("content, current, want", [("", "session", "session"), ("pre", "session", "session"),
                                                    ("bogus", "pre", "pre"), ("post", "pre", "post"),
                                                    (None, "session", "session")])  # fmt: skip
def test_clock_read_tag_keeps_the_last_tag_and_moves_only_forwards(tmp_path, content, current, want):
    if content is not None:
        (tmp_path / "tag").write_text(content)
    assert clock.read_tag(str(tmp_path), current) == want


def test_clock_sampler_without_host_qpc_stops_non_zero(tmp_path):
    ctl = tmp_path / "ctl"
    ctl.mkdir()
    out = tmp_path / "s.jsonl"
    p = subprocess.run([sys.executable, "-I", str(HARNESS / "clock.py"), "sample", "--out", str(out), "--control",
                        str(ctl), "--interval", "0.05", "--host-cmd", "/nonexistent/loop"],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)  # fmt: skip
    assert p.returncode == 3 and "STOP" in p.stdout


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
        set_tag(ctl, tag)
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
    # The third read sleeps 1.5 s, beyond the 1 s timeout; the first read
    # first waits for the fake loop to start, so it gets a long deadline.
    s = clock.Sampler(str(tmp_path / "s.jsonl"), shlex.split(fake_loop(tmp_path, "slow")), None)
    assert s.host.ask("cpu", timeout=60) is not None
    monkeypatch.setattr(clock, "HOST_TIMEOUT_S", 1.0)
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


def wait_for(cond, deadline_s=60.0):
    """Polls ``cond`` until it is true, or fails at the deadline."""
    end = time.monotonic() + deadline_s
    while not cond():
        assert time.monotonic() < end, "condition not met before the deadline"
        time.sleep(0.02)


def rows_of(path):
    return clock.read_jsonl(str(path)) if path.exists() else []


# A local fake of the box side of clock_ref's transport (152-clockref-fix-dryrun.md):
# a fake ``ssh`` that checks its arguments (``-L 127.0.0.1:P:127.0.0.1:P``,
# remote command == clock_ref.tcp_loop(P, idle)), never reads its stdin (the
# box's DefaultShell takes it), and runs the same protocol as TCP_LOOP: a
# listener on 127.0.0.1:P that accepts one client, answers one line per
# request line, and exits on ``end``, on the client's disconnect, after the
# idle timeout, or if no client connects within the accept timeout. MODE
# varies the loop: ``slow`` (the third ``x`` is slow), ``die`` (closes at the
# third ``x``), ``noise`` (a line before the first reply).
FAKE_SSH_TCP = """
import socket, sys, time
sys.path.insert(0, HARNESS)
import clock_ref
args = sys.argv[1:]
fwd = args[args.index("-L") + 1]
lo, port, ro, rport = fwd.split(":")
assert lo == ro == "127.0.0.1" and port == rport, fwd
port = int(port)
assert args[-1] == clock_ref.tcp_loop(port, IDLE), "the remote command is not tcp_loop"
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(1)
srv.settimeout(ACCEPT_S)
try:
    c, _ = srv.accept()
except socket.timeout:
    open(EXITED, "w").write("accept timeout")
    sys.exit(3)
srv.close()
c.settimeout(IDLE)
f = c.makefile("rwb", buffering=0)
import os
f.write(f"listener {os.getpid()}\\n".encode())
if MODE == "noise":
    f.write(b"profile: Windows PowerShell, loaded in 120 ms\\n")
n = 0
why = "disconnect"
try:
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if line == b"end":
            f.write(b"end\\n")
            why = "end"
            break
        if line == b"cpu":
            f.write(f"{time.process_time()}\\n".encode())
            continue
        n += 1
        if MODE == "slow" and n == 3:
            time.sleep(0.6)
        if MODE == "die" and n == 3:
            why = "die"
            break
        f.write(f"{time.monotonic_ns() // 100} 10000000 {time.time_ns() // 100}\\n".encode())
except socket.timeout:
    why = "idle"
c.close()
open(EXITED, "w").write(why)
"""


def free_port() -> int:
    with socket_mod.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_ssh_tcp(tmp_path, mode="ok", idle=60.0, accept_s=60.0):
    exited = tmp_path / f"exited_{mode}"
    src = tmp_path / f"fake_ssh_{mode}.py"
    src.write_text(f"HARNESS = {str(HARNESS)!r}\nMODE = {mode!r}\nIDLE = {idle!r}\nACCEPT_S = {accept_s!r}\n"
                   f"EXITED = {str(exited)!r}\n" + FAKE_SSH_TCP)  # fmt: skip
    w = tmp_path / f"ssh_{mode}"
    w.write_text(f'#!/bin/sh\nexec {sys.executable} -I {src} "$@"\n')
    w.chmod(0o755)
    return str(w), exited


def run_ref(tmp_path, mode, reads, idle=60.0):
    """Runs the client through the fake tunnel until ``reads`` read rows and a
    cpu row exist (or it exits on its own), then stops it."""
    out, stop = tmp_path / f"ref_{mode}.jsonl", tmp_path / f"stop_{mode}"
    ssh, exited = fake_ssh_tcp(tmp_path, mode, idle)
    p = subprocess.Popen([sys.executable, "-I", str(HARNESS / "clock_ref.py"), "--out", str(out), "--host",
                          "ref.example", "--stop-file", str(stop), "--port", str(free_port()), "--every", "0.05",
                          "--cpu-every", "0.2", "--idle", str(idle), "--reply-timeout", "10", "--ssh", ssh],
                         stdin=subprocess.DEVNULL)  # fmt: skip

    def enough():
        rows = rows_of(out)
        return p.poll() is not None or (
            sum(r.get("kind") == "read" for r in rows) >= reads and any(r.get("kind") == "cpu" for r in rows))

    wait_for(enough)
    stop.write_text("")
    rc = p.wait(timeout=60)
    wait_for(exited.exists)
    return rc, clock.read_jsonl(str(out)), exited.read_text()


def test_clock_ref_stamps_reads_with_monotonic_and_ends_cleanly(tmp_path):
    rc, rows, why = run_ref(tmp_path, "ok", 3)
    assert rc == 0 and rows[0]["clock"] == "time.monotonic" and rows[0]["transport"] == "tcp over ssh -L"
    assert rows[0]["clock_ref_py_sha256"] == manifest.file_sha256(str(HARNESS / "clock_ref.py"))
    reads = [r for r in rows if r["kind"] == "read"]
    assert len(reads) >= 3 and all(r["t0"] < r["t1"] and r["qpc"] is not None for r in reads)
    assert all(r["usable"] == (r["t1"] - r["t0"] <= clock_ref.RTT_MAX_S) for r in reads)
    assert any(r["kind"] == "cpu" for r in rows)
    assert rows[-1] == {"kind": "event", "event": "end", "loop_ended": True}
    assert why == "end"
    listener = [r for r in rows if r["kind"] == "listener"]
    assert len(listener) == 1 and listener[0]["pid"] > 0
    # The first line from the box names the listener's PID, recorded before any read.
    kinds = [r["kind"] for r in rows]
    assert kinds.index("listener") < kinds.index("read")
    assert rows[0]["ssh_log"].endswith(".ssh.log") and pathlib.Path(rows[0]["ssh_log"]).exists()


def test_clock_ref_discards_lines_before_the_first_reply(tmp_path):
    rc, rows, _ = run_ref(tmp_path, "noise", 3)
    assert rc == 0 and {"kind": "discarded", "line": "profile: Windows PowerShell, loaded in 120 ms"} in rows
    assert all(r["qpc"] is not None for r in rows if r["kind"] == "read")


def test_clock_ref_marks_a_slow_reply_unusable(tmp_path):
    _, rows, _ = run_ref(tmp_path, "slow", 4)
    reads = [r for r in rows if r["kind"] == "read"]
    # The loop's third "x" is slow; the first one is the client's sync request.
    assert reads[1]["usable"] is False and reads[1]["t1"] - reads[1]["t0"] > clock_ref.RTT_MAX_S


def test_clock_ref_records_a_lost_session_and_does_not_reconnect(tmp_path):
    rc, rows, why = run_ref(tmp_path, "die", 1000)  # it exits on its own at the third read
    assert rc == 3 and why == "die"
    assert rows[-1]["event"] == "lost" and sum(1 for r in rows if r["kind"] == "header") == 1


def test_clock_ref_fake_loop_exits_on_disconnect_and_idle(tmp_path):
    # The protocol the client relies on: the box loop never outlives the client.
    port = free_port()
    for mode, idle, act in (("ok", 60.0, "close"), ("idle", 0.3, "wait")):
        d = tmp_path / mode
        d.mkdir()
        ssh, exited = fake_ssh_tcp(d, mode, idle)
        p = subprocess.Popen([ssh, "-T", "-L", f"127.0.0.1:{port}:127.0.0.1:{port}", "box",
                              clock_ref.tcp_loop(port, idle)], stdin=subprocess.DEVNULL)  # fmt: skip
        s = None
        def connected():
            nonlocal s
            try:
                s = socket_mod.create_connection(("127.0.0.1", port), timeout=1)
                return True
            except OSError:
                return False
        wait_for(connected)
        s.sendall(b"x\n")
        rf = s.makefile("rb")
        assert rf.readline().split()[0] == b"listener"
        assert len(rf.readline().split()) == 3
        if act == "close":
            rf.close()  # the makefile holds the socket open until closed too
            s.close()
        assert p.wait(timeout=30) == 0
        assert exited.read_text() == ("disconnect" if act == "close" else "idle")
        if s:
            s.close()


def test_clock_ref_tcp_loop_is_one_argument_on_localhost():
    loop = clock_ref.tcp_loop(47123, 60)
    assert '"' not in loop and "\n" not in loop and "\r" not in loop
    assert "[Net.IPAddress]::Loopback, 47123" in loop and "ReceiveTimeout = 60000" in loop
    assert "Wait(60000)" in loop and "'end'" in loop and "finally { $c.Close() }" in loop
    assert loop.count("AcceptTcpClientAsync") == 1 and "$l.Stop()" in loop and "'listener ' + $PID" in loop
    assert clock_ref.TCP_LOOP.count("PORT") == 1 and clock_ref.TCP_LOOP.count("IDLE_MS") == 1
    assert "catch { [Console]::Error.WriteLine('loop error: ' + $_) } finally" in loop
    for forbidden in ("New-NetFirewallRule", "netsh", "Out-File", "Set-Content", "Start-Process", "RunAs"):
        assert forbidden not in loop
    cmd = clock_ref.ssh_command("jb@box.example", 47123, 60)
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:47123:127.0.0.1:47123" and cmd[-1] == loop
    assert "ExitOnForwardFailure=yes" in cmd
    with pytest.raises(ValueError):
        clock_ref.tcp_loop(80, 60)


def test_clock_and_clock_ref_share_the_read_expressions():
    # One QPC/frequency/UTC read and one CPU read, verbatim in both loops.
    assert clock.READ_EXPR == clock_ref.READ_EXPR and clock.CPU_EXPR == clock_ref.CPU_EXPR
    for expr in (clock.READ_EXPR, clock.CPU_EXPR):
        assert expr in clock.LOOP and expr in clock_ref.TCP_LOOP


def test_clock_ref_header_passes_decide_and_summarise(tmp_path):
    import socket

    _, ref, _ = run_ref(tmp_path, "ok", 2)
    head = ref[0]
    assert head["host"] == socket.gethostname() and head["target"] == "ref.example"
    m = gate_manifest(clock_ref_host=socket.gethostname(), clock_ref_py_sha256=head["clock_ref_py_sha256"],
                      clock_ref_command="python3 -I " + shlex.join(head["command"]))  # fmt: skip
    msha = write(tmp_path / "m.json", manifest.canonical(m))
    rows, _, status = session()
    d = clock.decide(rows, ref, m, msha, "session", status)
    assert "instruments" not in d["conditions"] and "selfhash" not in d["conditions"]
    dsha = write(tmp_path / "d.json", json.dumps(d).encode())
    rsha = write(tmp_path / "rows.jsonl", "".join(json.dumps(r) + "\n" for r in gate_rows()).encode())
    rc = summarise.main(["--rows", str(tmp_path / "rows.jsonl"), "--rows-sha256", rsha, "--manifest",
                         str(tmp_path / "m.json"), "--manifest-sha256", msha, "--clock-decide", str(tmp_path / "d.json"),
                         "--clock-decide-sha256", dsha, "--sent-at", "x", "--out", str(tmp_path / "s.jsonl")])  # fmt: skip
    assert rc == 0


# ------------------------------------------------------------- ADR-0064


def test_split_wall_now_reads_clock_monotonic_raw(monkeypatch):
    import split

    seen = []
    monkeypatch.setattr(split.time, "clock_gettime", lambda c: seen.append(c) or 42.0)
    assert split.wall_now() == 42.0 and seen == [time.CLOCK_MONOTONIC_RAW]
    assert split.WALL_CLOCK == "CLOCK_MONOTONIC_RAW"


def test_split_refuses_to_import_without_clock_monotonic_raw():
    # No fallback: a platform without CLOCK_MONOTONIC_RAW cannot import split.py.
    code = ("import sys, time; delattr(time, 'CLOCK_MONOTONIC_RAW'); sys.path.insert(0, %r)\n"
            "try:\n    import split\nexcept ImportError as e:\n    print('refused:', e)\n") % str(HARNESS)
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    assert "refused: split.py needs time.CLOCK_MONOTONIC_RAW" in p.stdout


def test_split_has_no_perf_counter_wall():
    src = (HARNESS / "split.py").read_text()
    assert "perf_counter()" not in src and src.count("wall_now()") >= 5


def test_manifest_records_the_wall_clock(tmp_path, capsys):
    from test_gpu_split import make_manifest

    _, argv = make_manifest(tmp_path)
    assert manifest.main(argv) == 0
    digest = capsys.readouterr().out.strip()
    assert manifest.load(str(tmp_path / "manifest.json"), digest)["wall_clock"] == "CLOCK_MONOTONIC_RAW"


@pytest.mark.parametrize("where", ["row", "missing", "manifest", "decide"])
def test_summarise_refuses_anything_not_read_from_raw(tmp_path, capsys, where):
    rows, m, over = gate_rows(), gate_manifest(), None
    if where == "row":
        rows[3]["result"]["wall_clock"] = "perf_counter"
    elif where == "missing":
        del rows[5]["result"]["wall_clock"]
    elif where == "manifest":
        m["wall_clock"] = "perf_counter"
    else:
        over = {"binding_guest_clock": "perf_counter"}
    rc, s, _ = run_summarise(tmp_path, rows=rows, m=m, decide_over=over)
    assert rc == summarise.EXIT_REFUSED and s is None
    assert "REFUSED" in capsys.readouterr().out


def test_driver_refuses_a_manifest_without_the_raw_wall_clock(tmp_path, capsys):
    from test_gpu_split import make_manifest

    wheel, argv = make_manifest(tmp_path)
    manifest.main(argv)
    capsys.readouterr()
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["wall_clock"] = "perf_counter"
    blob = manifest.canonical(data)
    (tmp_path / "m2.json").write_bytes(blob)
    rc = driver.main(["--python", f"sh -c 'exit 0' fake {wheel}", "--rows", str(tmp_path / "r.jsonl"), "--status",
                      str(tmp_path / "s.json"), "--device", "cpu", "--manifest", str(tmp_path / "m2.json"),
                      "--manifest-sha256", hashlib.sha256(blob).hexdigest(), "--wheel", str(wheel)])  # fmt: skip
    assert rc == driver.EXIT_REFUSED and "CLOCK_MONOTONIC_RAW" in capsys.readouterr().out
    assert not (tmp_path / "r.jsonl").exists()


REHEARSAL2 = pathlib.Path(__file__).parent / "fixtures" / "rehearsal2"


def test_clock_decide_rehearsal2_pre_check_is_valid_with_perf_reported():
    # Rehearsal 2 (2026-11-17), the real samples its pre-check judged
    # clock-invalid on perf_counter (condition (b), perf at 0.987842). Under
    # ADR-0064 the binding clock is raw, which ran at 1.000001 against QPC.
    # That sampler predates host_bracket_raw, so the bracket is mapped here
    # from host_bracket_perf through each sample's own perf -> raw line.
    path = REHEARSAL2 / "samples-at-pre-check.jsonl"
    want = "0ad720082584f53a12bacbc7e9f4d7d5dd65e01f832ed94ff01fbd932b237400"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == want
    rows = clock.read_jsonl(str(path))
    for x in rows:
        if x.get("kind") == "sample" and x["host"] is not None:
            b, a = x["guest_before"], x["guest_after"]
            k = (a["raw"] - b["raw"]) / (a["perf"] - b["perf"])
            x["host_bracket_raw"] = [b["raw"] + (p - b["perf"]) * k for p in x["host_bracket_perf"]]
    ref = clock.read_jsonl(str(REHEARSAL2 / "ref.jsonl"))
    data = (REHEARSAL2 / "manifest.json").read_bytes()
    d = clock.decide(rows, ref, json.loads(data), hashlib.sha256(data).hexdigest(), "pre-check")
    assert d["decision"] == "go"
    assert d["outcome"] == "clock-valid"
    assert d["binding_guest_clock"] == "CLOCK_MONOTONIC_RAW"
    assert d["coverage"]["guest"]["uncovered_share"] == 0.0
    assert "other" in d["reported"]["other_guest_clocks"]["perf"]
    assert d["reported"]["perf_worst"] > 0.01
