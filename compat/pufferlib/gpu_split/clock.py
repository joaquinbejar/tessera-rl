# SPDX-License-Identifier: MIT OR Apache-2.0
"""Clock validation of a gate session (ADR-0063, docs/reviews/152-adr0063-harness-plan.md r3).

Two commands:

    python clock.py sample --out PATH --control DIR [--interval 2] [--cpu CORE]
    python clock.py decide --samples PATH --ref PATH --manifest PATH --scope pre-check|session \\
        [--status PATH] --out PATH

``sample`` runs inside the WSL2 guest from the start of the pre-check to the
end of the post-check. Every ``--interval`` seconds it reads the guest clocks
(``perf_counter``, ``monotonic``, ``CLOCK_MONOTONIC_RAW``, ``realtime``), then
the host's QPC, its frequency and UTC through one persistent PowerShell
process, then the guest clocks again. It makes no CUDA read and never imports
torch. The tag (``pre``, ``session``, ``post``) is the content of
``DIR/tag``; the sampler stops when ``DIR/stop`` exists. A context row
(clocksource, the processor's reported speed, nvidia-smi, the host reader's
CPU time) is written at the start, at every tag change and at the stop: that
is the start and end of ``pre`` and of ``post``. ``--cpu`` pins the sampler
to one guest core; the host reader is ``powershell.exe`` on the Windows side
and cannot be pinned from the guest.

``decide`` is a pure function of the recorded files (ADR-0063 item 4). It
writes one JSON object to ``--out`` and prints its sha256.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import select
import shlex
import subprocess
import sys
import time

PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
PS_CPU_SPEED = "Get-CimInstance Win32_Processor | ForEach-Object { $_.CurrentClockSpeed; $_.MaxClockSpeed }"
CLOCKSOURCE = "/sys/devices/system/clocksource/clocksource0/current_clocksource"
# One line in, one line out: "cpu" -> the loop's total processor seconds,
# "end" -> "end" and exit, anything else -> QPC, its frequency, UTC ticks.
LOOP = (
    "while (($l = [Console]::In.ReadLine()) -ne $null) { "
    "if ($l -eq 'end') { [Console]::Out.WriteLine('end'); [Console]::Out.Flush(); break } "
    "if ($l -eq 'cpu') { [Console]::Out.WriteLine([Diagnostics.Process]::GetCurrentProcess()"
    ".TotalProcessorTime.TotalSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)) } "
    "else { [Console]::Out.WriteLine(([Diagnostics.Stopwatch]::GetTimestamp()).ToString() + ' ' + "
    "([Diagnostics.Stopwatch]::Frequency).ToString() + ' ' + ([DateTime]::UtcNow.Ticks).ToString()) } "
    "[Console]::Out.Flush() }"
)

# Predeclared (ADR-0063 item 3 and item 4).
TOL = 0.005  # band half-width
BOUND_MAX = 0.005  # an interval is classified only if its bound is below this
MIN_INTERVAL_S = 10.0  # guest and external intervals are at least this long
MERGE_MAX_S = 30.0  # external reads are merged up to this length
WINDOW_MIN_S = 300.0  # pre-check, post-check and UTC windows
RTT_MAX_S = 0.25  # a reference read with a longer round trip is unusable
UTC_GRAIN_S = 0.0156
COVER_MAX_SHARE = 0.05  # (d1), (d2)
COVER_MAX_GAP_S = 30.0
CPU_BUDGET = 0.01  # the sampler, over the timed span (item 7)
HOST_TIMEOUT_S = 5.0
FACTORS = {"nominal": 1.0, "12/13": 12 / 13, "13/12": 13 / 12}


def self_sha256(path: str = __file__) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode()


def host_command() -> list[str]:
    return [PS, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded(LOOP)]


# ------------------------------------------------------------- sampling


def run_quiet(argv: list[str], timeout: float = 60.0) -> str:
    """Runs a short command with stdin from /dev/null, so it can never read
    the input of whoever started the sampler (152-clockcheck-e70e8289.md,
    Deviations), and returns its stdout, or "" on failure."""
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def run_ps(cmd: str) -> list[str]:
    return run_quiet([PS, "-NoProfile", "-NonInteractive", "-Command", cmd]).split()


def guest_clocks() -> dict:
    return {
        "perf": time.perf_counter(),
        "mono": time.monotonic(),
        "raw": time.clock_gettime(time.CLOCK_MONOTONIC_RAW),
        "real": time.clock_gettime(time.CLOCK_REALTIME),
    }


class HostReader:
    """One persistent process running LOOP; a host read is a pipe round trip.
    A read that times out marks the reader dead: it is killed and never used
    again."""

    def __init__(self, argv: list[str]):
        try:
            self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError:
            self.p = None
        self.why = None if self.p else "no host reader"

    def ask(self, line: str, timeout: float | None = None) -> str | None:
        timeout = HOST_TIMEOUT_S if timeout is None else timeout
        if self.p is None:
            return None
        try:
            self.p.stdin.write(line + "\n")
            self.p.stdin.flush()
            ready, _, _ = select.select([self.p.stdout], [], [], timeout)
            if not ready:
                self.dead("timeout")
                return None
            out = self.p.stdout.readline()
            if not out:
                self.dead("closed")
                return None
            return out.strip()
        except (OSError, ValueError):
            self.dead("error")
            return None

    def read(self):
        out = self.ask("x")
        try:
            q, f, t = out.split()
            return int(q), int(f), int(t)
        except (AttributeError, ValueError):
            if self.p is not None:
                self.dead("bad reply")
            return None

    def cpu(self):
        out = self.ask("cpu")
        try:
            return float(out)
        except (TypeError, ValueError):
            return None

    def dead(self, why):
        self.why = why
        if self.p is not None:
            try:
                self.p.kill()
                self.p.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.p = None

    def end(self) -> bool:
        """Asks the loop to stop; True if it answered ``end``."""
        ok = self.ask("end") == "end"
        if self.p is not None:
            try:
                self.p.stdin.close()
                self.p.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.dead("close timeout")
        self.p = None
        return ok


def rusage() -> list[float]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return [r.ru_utime, r.ru_stime]


class Sampler:
    def __init__(self, out: str, host_argv: list[str], cpu: int | None):
        self.out = out
        self.host = HostReader(host_argv)
        self.n = 0
        self.cpu = cpu

    def write(self, rec: dict):
        with open(self.out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def sample(self, tag: str) -> dict:
        g0 = guest_clocks()
        h0 = time.perf_counter()
        hv = self.host.read()
        h1 = time.perf_counter()
        g1 = guest_clocks()
        host = {"qpc": hv[0], "qpc_freq": hv[1], "utc_ticks": hv[2]} if hv else None
        rec = {"kind": "sample", "i": self.n, "tag": tag, "guest_before": g0, "guest_after": g1,
               "host": host, "host_bracket_perf": [h0, h1], "rusage": rusage()}  # fmt: skip
        self.n += 1
        self.write(rec)
        if hv is None and self.host.why:
            self.write({"kind": "event", "event": "host_dead", "why": self.host.why, "i": rec["i"]})
            self.host.why = None
        return rec

    def context(self, at: str) -> dict:
        try:
            with open(CLOCKSOURCE) as f:
                source = f.read().strip()
        except OSError:
            source = None
        speed = run_ps(PS_CPU_SPEED)
        smi = run_quiet(["nvidia-smi", "--query-gpu=pstate,clocks.gr,clocks.mem", "--format=csv,noheader"]).strip()
        rec = {"kind": "context", "at": at, "mono": time.monotonic(), "current_clocksource": source,
               "cpu_mhz_current_max": speed[:2] or None, "nvidia_smi": smi or None,
               "host_cpu_s": self.host.cpu(), "rusage": rusage()}  # fmt: skip
        self.write(rec)
        return rec


TAGS = ("pre", "session", "post")


def read_tag(control: str, current: str) -> str:
    """The tag in ``DIR/tag``, accepted only if it is one of TAGS and not
    before ``current``; an empty, unreadable or backward tag keeps
    ``current``. Writers replace the file atomically (temp file and rename)."""
    try:
        with open(os.path.join(control, "tag")) as f:
            new = f.read().strip()
    except OSError:
        return current
    if new in TAGS and TAGS.index(new) >= TAGS.index(current):
        return new
    return current


def sample_main(args, host_argv=None) -> int:
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    s = Sampler(args.out, host_argv or host_command(), args.cpu)
    s.write({"kind": "header", "clock_py_sha256": self_sha256(), "cpu": args.cpu, "pid": os.getpid(),
             "interval_s": args.interval})  # fmt: skip
    tag = read_tag(args.control, "pre")
    s.context(f"start {tag}")
    first = s.sample(tag)
    if first["host"] is None:
        print(f"STOP: first sample has no host QPC ({s.host.why})", flush=True)
        s.write({"kind": "event", "event": "end", "stopped": "no host QPC"})
        return 3
    stop = os.path.join(args.control, "stop")
    nxt = time.monotonic()
    while not os.path.exists(stop):
        nxt += args.interval
        time.sleep(max(0.0, nxt - time.monotonic()))
        new = read_tag(args.control, tag)
        if new != tag:
            s.context(f"{tag} -> {new}")
            tag = new
        s.sample(tag)
    s.context(f"stop {tag}")
    if s.host.end():
        s.write({"kind": "event", "event": "host_end"})
    s.write({"kind": "event", "event": "end"})
    return 0


# ------------------------------------------------------- pure analysis


def band(r, bound) -> str:
    if r is None:
        return "missing"
    if bound is None or bound >= BOUND_MAX:
        return "imprecise"
    for name, f in FACTORS.items():
        if abs(r - f) <= TOL:
            return name
    return "other"


def mid(smp, k):
    return (smp["guest_before"][k] + smp["guest_after"][k]) / 2


def qpc_s(smp) -> float:
    return smp["host"]["qpc"] / smp["host"]["qpc_freq"]


def host_perf(smp):
    h0, h1 = smp["host_bracket_perf"]
    return (h0 + h1) / 2, (h1 - h0) / 2


def guest_intervals(samples: list[dict]) -> list[dict]:
    """Consecutive host-read samples merged into intervals of at least
    MIN_INTERVAL_S QPC seconds. ``perf`` is placed at the host read's
    midpoint with a true bound; ``mono``, ``raw`` and ``real`` use the whole
    sample's midpoint, so they are not classified on an interval that starts
    at the first sample of a tag (ADR-0063 item 3)."""
    pts = [x for x in samples if x["host"] is not None]
    first = set()
    seen = set()
    for x in pts:
        if x["tag"] not in seen:
            seen.add(x["tag"])
            first.add(x["i"])
    out, a = [], 0
    for b in range(1, len(pts)):
        dq = qpc_s(pts[b]) - qpc_s(pts[a])
        if dq < MIN_INTERVAL_S:
            continue
        pa, ua = host_perf(pts[a])
        pb, ub = host_perf(pts[b])
        row = {"i": pts[a]["i"], "j": pts[b]["i"], "q0": qpc_s(pts[a]), "q1": qpc_s(pts[b]), "tag": pts[b]["tag"],
               "perf": (pb - pa) / dq, "bound_perf": (ua + ub) / dq}  # fmt: skip
        row["gap"] = dq > MERGE_MAX_S + 1e-6
        row["band_perf"] = band(row["perf"], row["bound_perf"])
        wa = (pts[a]["guest_after"]["perf"] - pts[a]["guest_before"]["perf"]) / 2
        wb = (pts[b]["guest_after"]["perf"] - pts[b]["guest_before"]["perf"]) / 2
        for k in ("mono", "raw", "real"):
            row[k] = (mid(pts[b], k) - mid(pts[a], k)) / dq
            row["band_" + k] = "unclassified" if pts[a]["i"] in first else band(row[k], (wa + wb) / dq)
        out.append(row)
        a = b
    return out


def ref_points(ref: list[dict]) -> list[tuple[float, float, float]]:
    """Usable reference reads as (t_ref, half round trip, qpc_s)."""
    pts = []
    for r in ref:
        if r.get("kind") != "read" or r.get("qpc") is None:
            continue
        rtt = r["t1"] - r["t0"]
        if rtt > RTT_MAX_S:
            continue
        pts.append(((r["t0"] + r["t1"]) / 2, rtt / 2, r["qpc"] / r["qpc_freq"]))
    return pts


def external_intervals(points) -> list[dict]:
    """QPC against the reference clock. From each start, reads are taken until
    the interval is at least MIN_INTERVAL_S long and its bound is below
    BOUND_MAX (classified), or it reaches MERGE_MAX_S (``imprecise``). An
    interval longer than MERGE_MAX_S spans missing reads: it is flagged
    ``gap`` and never counts as coverage, but its band is still judged under
    (b) and (c) when its bound is below BOUND_MAX (the same holds for guest
    intervals)."""
    out, a = [], 0
    for b in range(1, len(points)):
        dt = points[b][0] - points[a][0]
        if dt < MIN_INTERVAL_S:
            continue
        bound = (points[a][1] + points[b][1]) / dt
        gap = dt > MERGE_MAX_S + 1e-6
        if not gap and bound >= BOUND_MAX and dt < MERGE_MAX_S - 1e-6:
            continue
        r = (points[b][2] - points[a][2]) / dt
        out.append({"q0": points[a][2], "q1": points[b][2], "dt_s": dt, "r": r, "bound": bound,
                    "band": band(r, bound), "gap": gap})  # fmt: skip
        a = b
    return out


def utc_windows(samples) -> list[dict]:
    pts = [x for x in samples if x["host"] is not None]
    out, a = [], 0
    for b in range(1, len(pts)):
        dq = qpc_s(pts[b]) - qpc_s(pts[a])
        if dq < WINDOW_MIN_S:
            continue
        du = (pts[b]["host"]["utc_ticks"] - pts[a]["host"]["utc_ticks"]) / 1e7
        r = du / dq
        bound = 2 * UTC_GRAIN_S / dq
        out.append({"q0": qpc_s(pts[a]), "q1": qpc_s(pts[b]), "r": r, "bound": bound, "band": band(r, bound)})
        a = b
    return out


def coverage(lo: float, hi: float, covered: list[tuple[float, float]]) -> dict:
    """The share of [lo, hi] outside the covered segments, and the longest
    uncovered stretch, on the QPC axis."""
    span = hi - lo
    if span <= 0:
        return {"uncovered_share": 1.0, "longest_gap_s": 0.0, "span_s": span}
    gaps, at = [], lo
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in covered if b > lo and a < hi):
        if a > at:
            gaps.append(a - at)
        at = max(at, b)
    if hi > at:
        gaps.append(hi - at)
    return {"uncovered_share": sum(gaps) / span, "longest_gap_s": max(gaps, default=0.0), "span_s": span}


def mono_to_qpc(samples, mono: float) -> float | None:
    """A guest monotonic value placed on the QPC axis, by linear
    interpolation between the two host-read samples around it."""
    pts = [(mid(x, "mono"), qpc_s(x)) for x in samples if x["host"] is not None]
    for (m0, q0), (m1, q1) in zip(pts, pts[1:]):
        if m0 <= mono <= m1:
            return q0 if m1 == m0 else q0 + (mono - m0) * (q1 - q0) / (m1 - m0)
    return None


def nearest(samples, mono: float) -> dict:
    return min(samples, key=lambda x: abs(mid(x, "mono") - mono))


def decide(rows: list[dict], ref: list[dict], manifest: dict, manifest_sha256: str, scope: str,
           status: dict | None = None) -> dict:  # fmt: skip
    """ADR-0063 item 4. ``rows`` is the sampler's JSONL, ``ref`` the reference
    client's, ``status`` the driver's (session scope)."""
    if scope not in ("pre-check", "session"):
        raise ValueError("scope must be pre-check or session")
    head = next((r for r in rows if r.get("kind") == "header"), {})
    rhead = next((r for r in ref if r.get("kind") == "header"), {})
    samples = [r for r in rows if r.get("kind") == "sample"]
    contexts = [r for r in rows if r.get("kind") == "context"]
    events = [r.get("event") for r in rows if r.get("kind") == "event"]
    revents = [r.get("event") for r in ref if r.get("kind") == "event"]
    if scope == "pre-check":
        samples = [x for x in samples if x["tag"] == "pre"]
    host = [x for x in samples if x["host"] is not None]
    fired: dict[str, list[str]] = {}

    def fire(cond, why):
        fired.setdefault(cond, []).append(why)

    # Instruments (§2.1a): the self-hashes are judged here, the other values
    # are carried for summarise.py.
    instruments = {
        "clock_py_sha256": head.get("clock_py_sha256"),
        "clock_ref_py_sha256": rhead.get("clock_ref_py_sha256"),
        "clock_ref_host": rhead.get("host"),
        "clock_ref_target": rhead.get("target"),
        "clock_ref_args": (rhead.get("command") or [None])[1:],
        "clock_cpu": head.get("cpu"),
    }
    selfhash = []
    if instruments["clock_py_sha256"] != manifest.get("clock_py_sha256"):
        selfhash.append("clock.py sha256 mismatch")
    if instruments["clock_ref_py_sha256"] != manifest.get("clock_ref_py_sha256"):
        selfhash.append("clock_ref.py sha256 mismatch")
    for why in selfhash:
        fire("selfhash", why)
    for key, want in (("clock_ref_host", manifest.get("clock_ref_host")), ("clock_cpu", manifest.get("clock_cpu"))):
        if instruments[key] != want:
            fire("instruments", f"{key} {instruments[key]!r} is not the manifest's {want!r}")
    cmd = shlex.split(manifest.get("clock_ref_command") or "")
    args = instruments["clock_ref_args"] or []
    if not args or cmd[-len(args) :] != args or len(cmd) <= len(args) or not cmd[-len(args) - 1].endswith("clock_ref.py"):
        fire("instruments", "the reference client's arguments are not the manifest's clock_ref_command")
    order = [TAGS.index(x["tag"]) if x["tag"] in TAGS else -1 for x in samples]
    if any(o < 0 for o in order) or any(b < a for a, b in zip(order, order[1:])):
        fire("tags", "sample tags out of order")
    for name, stream in (("sampler", rows), ("reference", ref)):
        bad = [i for i, r in enumerate(stream) if r.get("kind") == "corrupt" and not r.get("last")]
        if bad:
            fire("corrupt", f"{name}: {len(bad)} corrupt line(s) before the last")

    pre = [qpc_s(x) for x in host if x["tag"] == "pre"]
    post = [qpc_s(x) for x in host if x["tag"] == "post"]
    if not pre or pre[-1] - pre[0] < WINDOW_MIN_S:
        fire("a", f"pre-check {pre[-1] - pre[0] if pre else 0:.1f} s")
    if scope == "session" and (not post or post[-1] - post[0] < WINDOW_MIN_S):
        fire("a", f"post-check {post[-1] - post[0] if post else 0:.1f} s")
    lo = qpc_s(host[0]) if host else 0.0
    hi = qpc_s(host[-1]) if host else 0.0

    guest = guest_intervals(samples)
    for g in guest:
        if g["band_perf"] not in ("nominal", "imprecise"):
            fire("b", f"perf {g['perf']:.6f} on QPC [{g['q0']:.1f}, {g['q1']:.1f}]")
    ext = [e for e in external_intervals(ref_points(ref)) if e["q1"] > lo and e["q0"] < hi]
    for e in ext:
        if e["band"] not in ("nominal", "imprecise"):
            fire("c", f"QPC/reference {e['r']:.6f} on QPC [{e['q0']:.1f}, {e['q1']:.1f}]")
    sources = sorted({c.get("current_clocksource") for c in contexts} - {None})
    if len(sources) > 1:
        fire("e", f"clocksource {sources}")
    if not sources:
        fire("clocksource", "no clocksource reading")

    cov_ext = coverage(lo, hi, [(e["q0"], e["q1"]) for e in ext if e["band"] != "imprecise" and not e["gap"]])
    cov_guest = coverage(lo, hi, [(g["q0"], g["q1"]) for g in guest if g["band_perf"] != "imprecise" and not g["gap"]])
    for cond, cov in (("d1", cov_ext), ("d2", cov_guest)):
        if cov["uncovered_share"] > COVER_MAX_SHARE or cov["longest_gap_s"] > COVER_MAX_GAP_S:
            fire(cond, f"uncovered {cov['uncovered_share']:.4f}, longest {cov['longest_gap_s']:.1f} s")
    if "host_dead" in events:
        fire("d3", "host reader died")
    if "lost" in revents:
        fire("d3", "reference session lost")
    if scope == "session":
        if "end" not in events:
            fire("d3", "sampler stream has no end marker")
        if "host_end" not in events:
            fire("d3", "host reader did not end")
        if "end" not in revents:
            fire("d3", "reference stream has no end marker")

    fault = sorted(set(fired) & {"b", "c", "e"})
    missing = sorted(set(fired) & {"a", "d1", "d2", "d3", "selfhash", "instruments", "tags", "corrupt", "clocksource"})
    outcome = "clock-invalid" if fault else "instrumentation failed" if missing else "clock-valid"
    out = {
        "scope": scope,
        "manifest_sha256": manifest_sha256,
        "outcome": outcome,
        "conditions": fired,
        "instruments": instruments,
        "span_qpc_s": [lo, hi],
        "guest_intervals": len(guest),
        "external_intervals": len(ext),
        "external_precise": sum(1 for e in ext if e["band"] != "imprecise" and not e["gap"]),
        "coverage": {"external": cov_ext, "guest": cov_guest},
        "worst_perf": max((abs(g["perf"] - 1) for g in guest if g["band_perf"] != "imprecise"), default=None),
        "external": ext,
        "reported": {
            "utc": utc_windows(samples),
            "clocksource": sources,
            "cpu_mhz_current_max": [c.get("cpu_mhz_current_max") for c in contexts],
            "other_guest_clocks": {k: sorted({g["band_" + k] for g in guest}) for k in ("mono", "raw", "real")},
        },
    }
    if scope == "pre-check":
        out["decision"] = "go" if outcome == "clock-valid" and out["external_precise"] >= 1 else "stop"
    if status and host:
        t0, t1 = status.get("timed_start_mono"), status.get("timed_end_mono")
        if t0 is not None and t1 is not None and t1 > t0:
            a, b = nearest(samples, t0), nearest(samples, t1)
            span = mid(b, "mono") - mid(a, "mono")
            cpu = sum(b["rusage"]) - sum(a["rusage"])
            share = cpu / span if span > 0 else None
            out["timed_qpc_s"] = [mono_to_qpc(samples, t0), mono_to_qpc(samples, t1)]
            out["cpu"] = {"sampler_share": share, "budget": CPU_BUDGET,
                          "over_budget": share is not None and share > CPU_BUDGET}  # fmt: skip
    hc = [(c["mono"], c["host_cpu_s"]) for c in contexts if c.get("host_cpu_s") is not None]
    rc = [(r["t"], r["cpu_s"]) for r in ref if r.get("kind") == "cpu" and r.get("cpu_s") is not None]
    out.setdefault("cpu", {})
    # The host reader's share spans its first to last context row (start of
    # pre to the stop); the reference loop's spans its first to last cpu row.
    out["cpu"]["host_reader_span"] = "context rows: start of pre to the stop"
    out["cpu"]["reference_loop_span"] = "cpu rows: first to last"
    for name, pts in (("host_reader_share", hc), ("reference_loop_share", rc)):
        out["cpu"][name] = (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]) if len(pts) > 1 and pts[-1][0] > pts[0][0] else None
    return out


def read_jsonl(path: str) -> list[dict]:
    """Rows of a JSONL file; a line that does not parse becomes
    ``{"kind": "corrupt", "last": <is it the last line>}``. Only a truncated
    last line is tolerated by ``decide``."""
    with open(path) as f:
        lines = [x.strip() for x in f if x.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"kind": "corrupt", "last": k == len(lines) - 1})
    return rows


def decide_main(args) -> int:
    with open(args.manifest, "rb") as f:
        data = f.read()
    status = None
    if args.status:
        with open(args.status) as f:
            status = json.load(f)
    out = decide(read_jsonl(args.samples), read_jsonl(args.ref), json.loads(data), hashlib.sha256(data).hexdigest(),
                 args.scope, status)  # fmt: skip
    blob = (json.dumps(out, sort_keys=True) + "\n").encode()
    with open(args.out, "wb") as f:
        f.write(blob)
    print(hashlib.sha256(blob).hexdigest(), flush=True)
    if args.scope == "pre-check":
        print(f"pre-check: {out['decision']}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--control", required=True)
    s.add_argument("--interval", type=float, default=2.0)
    s.add_argument("--cpu", type=int)
    s.add_argument("--host-cmd", help=argparse.SUPPRESS)  # tests: a fake host loop
    d = sub.add_parser("decide")
    d.add_argument("--samples", required=True)
    d.add_argument("--ref", required=True)
    d.add_argument("--manifest", required=True)
    d.add_argument("--scope", required=True, choices=("pre-check", "session"))
    d.add_argument("--status")
    d.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "sample":
        return sample_main(args, shlex.split(args.host_cmd) if args.host_cmd else None)
    return decide_main(args)


if __name__ == "__main__":
    sys.exit(main())
