#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Measures manifest scenarios under Callgrind, per object, by difference
(docs/reviews/pgo-wheels-design.md r4, section 5).

Refuses to run without `--approved-window`, the coordinator's approval of a host
window, and only on Linux. This has never run against Valgrind: command lines and
the handling of its output are unproven until the feasibility gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402

WORKLOAD = pathlib.Path(__file__).resolve().parent / "workload.py"
CONTROL = "control/interpreter-only"
AA_RUNS = 5
# The provenance every file must carry to be reused: the hashes of the wheel,
# manifest, kernels, interpreter and Valgrind it was measured with. main()
# sets it for part, probe, adopt and merge runs; None (the library default
# and the plain gate) reuses any finished file, as before.
STAMP: dict | None = None


def stamp_path(out: pathlib.Path) -> pathlib.Path:
    return out.with_name(out.name + ".stamp.json")


def stamped(out: pathlib.Path) -> bool:
    """True when no stamp is required, or `out`'s sidecar carries this one."""
    if STAMP is None:
        return True
    side = stamp_path(out)
    return side.exists() and json.loads(side.read_text(encoding="utf-8")) == STAMP


def stamp(out: pathlib.Path) -> None:
    """Records the provenance of a run that has just finished successfully."""
    if STAMP is not None:
        stamp_path(out).write_text(json.dumps(STAMP, sort_keys=True))


def command(python: str, manifest: str, scenario: str, steps: int, out: str, separate_threads: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    """The Callgrind command of one run. Threads are aggregated unless the
    feasibility control needs one file per thread."""
    return [
        "valgrind",
        "--tool=callgrind",
        "--cache-sim=yes",
        "--branch-sim=yes",
        f"--separate-threads={'yes' if separate_threads else 'no'}",
        f"--callgrind-out-file={out}",
        python,
        "-I",
        str(WORKLOAD),
        "--manifest",
        manifest,
        "--scenario",
        scenario,
        "--steps",
        str(steps),
        *extra,
    ]


def _reusable(out: pathlib.Path) -> evidence.Cost | None:
    """The cost in a finished file of an earlier run, or None. A file without
    Callgrind's closing `totals:` line is unfinished and is not reused."""
    if not out.exists():
        return None
    text = out.read_text(encoding="utf-8")
    if "\ntotals:" not in text:
        return None
    try:
        return evidence.native_cost(evidence.parse_callgrind(text))
    except ValueError:
        return None


def measure(python: str, manifest: str, scenario: str, steps: int, out_dir: pathlib.Path, runner=subprocess.run, tag: str = "", extra: tuple[str, ...] = (), reused: list | None = None) -> evidence.Cost:
    """The native cost of one run; a failing run raises. A finished file from an
    earlier run is reused, so a gate that does not fit one window can continue
    in the next with the same manifest and wheel; `reused` records which."""
    safe = scenario.replace("/", "_")
    out = out_dir / f"{safe}.{steps}{tag}.callgrind"
    found = _reusable(out)
    if found is not None and stamped(out):
        if reused is not None:
            reused.append(out.name)
        return found
    stamp_path(out).unlink(missing_ok=True)
    runner(command(python, manifest, scenario, steps, str(out), extra=extra), check=True)
    stamp(out)
    return evidence.native_cost(evidence.parse_callgrind(out.read_text(encoding="utf-8")))


def cost_of(python: str, manifest: str, s: evidence.Scenario, out_dir, runner=subprocess.run, tag: str = "", extra: tuple[str, ...] = (), reused: list | None = None) -> evidence.Cost:
    """The scenario's cost with setup removed: the run at 2N minus the run at N."""
    short = measure(python, manifest, s.name, s.steps, out_dir, runner, tag, extra, reused)
    return measure(python, manifest, s.name, 2 * s.steps, out_dir, runner, tag, extra, reused) - short


def thread_files(out: pathlib.Path) -> list[pathlib.Path]:
    """The `<out>-<tid>` files of a separate-threads run, by numeric thread id:
    tid 1 is the caller, and "-10" must not sort before "-2"."""
    found = []
    for path in out.parent.glob(out.name + "-*"):
        suffix = path.name[len(out.name) + 1 :]
        if suffix.isdigit():
            found.append((int(suffix), path))
    return [path for _, path in sorted(found)]


def worker_control(python: str, manifest: str, s: evidence.Scenario, out_dir, runner=subprocess.run, reused: list | None = None) -> dict:
    """Section 5 item 4. Sync threaded scenarios against the same steps at 0
    workers; modes that need workers from the per-thread files."""
    if s.mode == "sync":
        threaded = cost_of(python, manifest, s, out_dir, runner, ".workers", reused=reused).ir
        inline = cost_of(python, manifest, s, out_dir, runner, ".inline", ("--workers", "0"), reused).ir
        return {"kind": "against-inline", "passed": evidence.workers_collected(threaded, inline)}
    out = out_dir / f"{s.name.replace('/', '_')}.threads.callgrind"
    found = thread_files(out)
    if len(found) >= 2 and all("\ntotals:" in p.read_text(encoding="utf-8") for p in found) and stamped(out):
        # A finished per-thread run of an earlier window, reused like measure().
        if reused is not None:
            reused.extend(p.name for p in found)
    else:
        # Callgrind does not remove an earlier run's files for higher thread
        # numbers, so a rerun must not be read together with leftovers.
        for stale in found:
            stale.unlink()
        stamp_path(out).unlink(missing_ok=True)
        runner(command(python, manifest, s.name, s.steps, str(out), separate_threads=True), check=True)
        stamp(out)
        found = thread_files(out)
    texts = [p.read_text(encoding="utf-8") for p in found]
    others, share = evidence.non_caller_native(texts)
    return {"kind": "per-thread", "passed": others > 0, "non_caller_ir": others, "share": share}


def control_ir(python: str, manifest_path: str, out_dir: pathlib.Path, runner=subprocess.run, reused: list | None = None) -> int:
    """The interpreter-only control, by difference of its 2- and 1-step runs."""
    return (
        measure(python, manifest_path, CONTROL, 2, out_dir, runner, reused=reused)
        - measure(python, manifest_path, CONTROL, 1, out_dir, runner, reused=reused)
    ).ir


def scenario_runs(python: str, manifest_path: str, s: evidence.Scenario, out_dir: pathlib.Path, runner=subprocess.run, reused: list | None = None) -> tuple[list[evidence.Cost], dict]:
    """Every run of one scenario for the gate: the A/A pairs, then linearity
    (single-threaded) or the worker control. The entry lacks the control check."""
    entry: dict = {}
    aa = [cost_of(python, manifest_path, s, out_dir, runner, f".aa{i}", reused=reused) for i in range(AA_RUNS)]
    entry["aa_ir"] = [c.ir for c in aa]
    entry["aa_ir_spread"] = evidence.spread(entry["aa_ir"])
    if s.single_threaded:
        # N and 2N are the first A/A pair, reused; only 3N is a new run.
        irs = [measure(python, manifest_path, s.name, k * s.steps, out_dir, runner, ".aa0").ir for k in (1, 2, 3)]
        entry["linear"] = evidence.linear(*irs)
    else:
        entry["workers"] = worker_control(python, manifest_path, s, out_dir, runner, reused)
    return aa, entry


LINEAR = "lin"
WORKERS = "workers"


def unit_names(s: evidence.Scenario) -> list[str]:
    """The independent run units of one scenario: each A/A pair (N and 2N),
    then the 3N linearity run (single-threaded) or the worker control."""
    return [f"aa{i}" for i in range(AA_RUNS)] + [LINEAR if s.single_threaded else WORKERS]


def all_units(manifest: dict) -> list[str]:
    """Every unit of the gate, in run order: the control, then `name#unit`."""
    return [CONTROL, *(f"{s.name}#{u}" for s in evidence.scenarios(manifest) for u in unit_names(s))]


def select(manifest: dict, names: list[str]) -> list[str]:
    """The named units, in run order. A scenario name stands for all its units,
    `name#unit` for one (`aa0`..`aa4`, `lin` or `workers`); `CONTROL` may be
    named too. An unknown or repeated unit is refused, so two containers
    cannot both be given one by a typo."""
    by_name = {s.name: s for s in evidence.scenarios(manifest)}
    wanted: list[str] = []
    unknown = []
    for n in names:
        if n == CONTROL:
            wanted.append(n)
        elif n in by_name:
            wanted += [f"{n}#{u}" for u in unit_names(by_name[n])]
        elif "#" in n and n.split("#", 1)[0] in by_name and n.split("#", 1)[1] in unit_names(by_name[n.split("#", 1)[0]]):
            wanted.append(n)
        else:
            unknown.append(n)
    if unknown:
        raise ValueError(f"unknown scenarios or units: {unknown}")
    if len(set(wanted)) != len(wanted):
        raise ValueError("a unit is named twice")
    return [u for u in all_units(manifest) if u in wanted]


def run_unit(python: str, manifest_path: str, by_name: dict, unit: str, out_dir: pathlib.Path, runner=subprocess.run, reused: list | None = None) -> None:
    """The runs of one unit; the files are the ones the full gate reads."""
    if unit == CONTROL:
        control_ir(python, manifest_path, out_dir, runner, reused)
        return
    name, part = unit.split("#", 1)
    s = by_name[name]
    if part == LINEAR:
        measure(python, manifest_path, s.name, 3 * s.steps, out_dir, runner, ".aa0", reused=reused)
    elif part == WORKERS:
        worker_control(python, manifest_path, s, out_dir, runner, reused)
    else:
        cost_of(python, manifest_path, s, out_dir, runner, f".{part}", reused=reused)


def probe(python: str, manifest_path: str, manifest: dict, out_dir: pathlib.Path, names: list[str], runner=subprocess.run) -> dict:
    """One N-step run per named scenario, the first file of its first A/A pair,
    so a later part reuses it. Its time per scenario is the projection a
    parallel window plans from, instead of the first scenario's rate."""
    by_name = {s.name: s for s in evidence.scenarios(manifest)}
    reused: list[str] = []
    seconds = {}
    if any("#" in n for n in names):
        raise ValueError("--probe takes scenario names, not units")
    for name in dict.fromkeys(u.split("#", 1)[0] for u in select(manifest, names)):
        if name == CONTROL:
            continue
        s = by_name[name]
        start = time.monotonic()
        cost = measure(python, manifest_path, s.name, s.steps, out_dir, runner, ".aa0", reused=reused)
        seconds[name] = {"seconds": round(time.monotonic() - start, 1), "ir": cost.ir}
    return {"probe": seconds, "reused_files": reused}


def measure_only(python: str, manifest_path: str, manifest: dict, out_dir: pathlib.Path, names: list[str], runner=subprocess.run) -> dict:
    """The runs of the named scenarios or units only, for one of several
    containers that share a window on disjoint cpusets. Decides nothing:
    `--merge` evaluates the gate from the finished files of every part."""
    by_name = {s.name: s for s in evidence.scenarios(manifest)}
    reused: list[str] = []
    done = []
    seconds = {}
    for unit in select(manifest, names):
        start = time.monotonic()
        run_unit(python, manifest_path, by_name, unit, out_dir, runner, reused)
        seconds[unit] = round(time.monotonic() - start, 1)
        # One line per unit as it ends, so a run's log shows progress and
        # actual against modelled time unit by unit.
        print(f"unit {unit} {seconds[unit]} s", flush=True)
        done.append(unit)
    return {"units": done, "unit_seconds": seconds, "reused_files": reused}


def _run(argv, check):
    """`subprocess.run`, looked up at call time."""
    return subprocess.run(argv, check=check)


class MissingRun(Exception):
    """A merge found a run without a finished file."""


def refuse(argv, check):
    """The runner of `--merge`: it never starts Valgrind."""
    out = next(a.split("=", 1)[1] for a in argv if a.startswith("--callgrind-out-file="))
    raise MissingRun(pathlib.Path(out).name)


def feasibility(python: str, manifest_path: str, manifest: dict, out_dir: pathlib.Path, runner=subprocess.run) -> dict:
    """The gate of section 5, on W1. Every check is recorded; `passed` needs all."""
    all_scenarios = evidence.scenarios(manifest)
    reused: list[str] = []
    control = control_ir(python, manifest_path, out_dir, runner, reused)
    report: dict = {"control_native_ir": control, "scenarios": {}, "reused_files": reused}
    aa: dict[str, list[evidence.Cost]] = {}
    ok = True
    for s in all_scenarios:
        aa[s.name], entry = scenario_runs(python, manifest_path, s, out_dir, runner, reused)
        median_ir = int(statistics.median(entry["aa_ir"]))
        entry["interpreter_excluded"] = evidence.interpreter_excluded(control, median_ir)
        ok &= entry["interpreter_excluded"]
        ok &= entry["linear"] if s.single_threaded else entry["workers"]["passed"]
        report["scenarios"][s.name] = entry
    sets = evidence.gating_sets(all_scenarios, aa)
    report["gating"] = list(sets.gating)
    report["mispredict"] = list(sets.mispredict)
    report["mispredict_available"] = sets.mispredict_available
    report["usable"] = sets.usable
    report["passed"] = bool(ok and sets.usable)
    return report


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hashes(args) -> dict:
    """The measured inputs. The kernel list is not one: it enters no Ir, only
    the coverage acceptance of a PGO profile, so it is recorded beside the
    stamp (`kernels_sha256`) and not in it."""
    return {"wheel": sha256_of(args.wheel), "manifest": sha256_of(args.manifest)}


def kernels_sha256(args) -> str:
    return sha256_of(str(pathlib.Path(args.manifest).parent / "kernels.toml"))


def cpu(cpuinfo: str = "/proc/cpuinfo") -> dict:
    """The host CPU as the kernel reports it. Valgrind shows its guest a CPUID
    of its own, derived from the host's, so the host flags are the proxy: Ir
    can depend on the CPU features (ifunc-selected libc routines, runtime SIMD
    dispatch), and files measured on two hosts are not interchangeable unless
    shown equal. The model name and a hash of the first processor's feature
    flags; raises ValueError when either is missing, so two hosts can never
    stamp equal by both lacking them."""
    model, flags = None, None
    for line in pathlib.Path(cpuinfo).read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "flags" and flags is None:
            flags = hashlib.sha256(" ".join(sorted(value.split())).encode()).hexdigest()
    if not model or not flags:
        raise ValueError(f"{cpuinfo} has no 'model name' or 'flags' line; the host cannot be stamped")
    return {"cpu_model": model, "cpu_flags": flags}


def environment(args, cpuinfo: str = "/proc/cpuinfo") -> dict:
    """`hashes` plus the measuring toolchain and host: Ir depends on the
    interpreter binary, the Valgrind build and the CPU features, not only on
    the wheel."""
    version = subprocess.run(["valgrind", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    return {**hashes(args), "python": sha256_of(str(pathlib.Path(args.python).resolve())), "valgrind": version, **cpu(cpuinfo)}


def adopt(listing: pathlib.Path, out_dir: pathlib.Path) -> list[str]:
    """Stamps finished files of an earlier window with the current STAMP, after
    checking each against the window's record (`<sha256> <name> ...` lines,
    as KEPT.txt). Every file must carry a stamp of its own that equals the
    current one in every field but `kernels` (the field older stamps had);
    a file without a stamp, or with any other field different, is refused.
    All or nothing: a missing, unfinished, changed or differently stamped file
    refuses the whole adoption."""
    wanted = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            wanted.append((parts[0], parts[1]))
    if not wanted:
        raise ValueError(f"{listing} lists no files")
    for digest, name in wanted:
        path = out_dir / name
        if not path.exists() or _reusable(path) is None:
            raise ValueError(f"{name} is missing or unfinished")
        if sha256_of(str(path)) != digest:
            raise ValueError(f"{name} does not match its recorded sha256")
        side = stamp_path(path)
        if not side.exists():
            raise ValueError(f"{name} has no stamp; it cannot be adopted")
        old = {k: v for k, v in json.loads(side.read_text(encoding="utf-8")).items() if k != "kernels"}
        if old != STAMP:
            diff = sorted(k for k in set(old) | set(STAMP or {}) if old.get(k) != (STAMP or {}).get(k))
            raise ValueError(f"{name} was measured under another stamp (fields {diff})")
    for _, name in wanted:
        stamp(out_dir / name)
    return [name for _, name in wanted]


def merge(args) -> int:
    """The gate from the finished files of every part. Runs no Valgrind, so it
    needs no window and no Linux. Every part report in the out dir must carry
    the same wheel, manifest and kernels hashes as this call."""
    if not args.wheel:
        print("refused: a merge needs --wheel, the exact artifact measured", file=sys.stderr)
        return 2
    global STAMP
    want = hashes(args)
    # The gate reads only gate parts. Costs read both kinds: a wheel's costs
    # parts, or, for W1, the gate's own parts, whose A/A units are the same files.
    kinds = ("feasibility", "costs") if args.costs else ("feasibility",)
    parts = sorted(p for k in kinds for p in args.out_dir.glob(f"{k}.part.*.json"))
    if not parts:
        print("refused: no part reports to merge", file=sys.stderr)
        return 2
    windows, seen, env = [], set(), None
    for p in parts:
        part = json.loads(p.read_text(encoding="utf-8"))
        got = part.get("sha256") or {}
        if {k: got.get(k) for k in want} != want:
            print(f"refused: {p.name} was measured with other hashes: {got} != {want}", file=sys.stderr)
            return 2
        if env is not None and got != env:
            print(f"refused: {p.name} was measured with another toolchain: {got} != {env}", file=sys.stderr)
            return 2
        env = got
        if "units" not in part:
            print(f"refused: {p.name} is a part report of the older format (no units); delete it, it is not merged",
                  file=sys.stderr)
            return 2
        overlap = seen & set(part["units"])
        if overlap:
            print(f"refused: {p.name} repeats units of another part: {sorted(overlap)}", file=sys.stderr)
            return 2
        seen |= set(part["units"])
        windows.append(part.get("approved_window"))
    manifest = evidence.load_manifest(args.manifest)
    if args.costs:
        return merge_costs(args, manifest, parts, seen, env, windows)
    every = set(all_units(manifest))
    if seen != every:
        print(f"refused: the parts must cover every unit of the gate exactly once; missing "
              f"{sorted(every - seen)}, extra {sorted(seen - every)}", file=sys.stderr)
        return 2
    # Every file read must carry the parts' provenance: a finished file of
    # another window or toolchain counts as missing.
    STAMP = env
    try:
        report = feasibility(args.python, args.manifest, manifest, args.out_dir, runner=refuse)
    except MissingRun as e:
        print(f"refused: no finished file with this provenance for {e}; the gate is not evaluated", file=sys.stderr)
        return 3
    finally:
        STAMP = None
    report["merged_parts"] = [p.name for p in parts]
    report["approved_window"] = sorted({w for w in windows if w})
    report["sha256"] = env
    report["kernels_sha256"] = kernels_sha256(args)
    (args.out_dir / "feasibility.json").write_text(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


def aa_units(manifest: dict) -> list[str]:
    """The units a wheel's costs need: the five A/A pairs of every scenario."""
    return [u for u in all_units(manifest) if re.search(r"#aa[0-9]+$", u)]


def merge_costs(args, manifest: dict, parts: list, seen: set, env: dict, windows: list) -> int:
    """A wheel's costs.json from the finished files of every part: per scenario,
    the median of each event over its five A/A pairs (N and 2N, by difference).
    The parts must cover every A/A unit; a gate's parts (which also hold the
    control, linearity and worker units) serve as they are, so W1's costs come
    from the gate's own files. Runs no Valgrind."""
    global STAMP
    missing = sorted(set(aa_units(manifest)) - seen)
    if missing:
        print(f"refused: the parts do not cover every A/A unit; missing {missing}", file=sys.stderr)
        return 2
    STAMP = env
    costs, aa = {}, {}
    try:
        for s in evidence.scenarios(manifest):
            pairs = [cost_of(args.python, args.manifest, s, args.out_dir, refuse, f".aa{i}") for i in range(AA_RUNS)]
            costs[s.name] = {k: int(statistics.median(p.events[k] for p in pairs)) for k in pairs[0].events}
            aa[s.name] = [p.ir for p in pairs]
    except MissingRun as e:
        print(f"refused: no finished file with this provenance for {e}; no costs are written", file=sys.stderr)
        return 3
    finally:
        STAMP = None
    report = {"sha256": env, "kernels_sha256": kernels_sha256(args), "costs": costs, "aa_ir": aa,
              "merged_parts": [p.name for p in parts], "approved_window": sorted({w for w in windows if w})}
    (args.out_dir / "costs.json").write_text(json.dumps(report, indent=1))
    return 0


def _main(argv, platform) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approved-window", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--feasibility", action="store_true", help="the gate of section 5, on W1")
    parser.add_argument("--costs", action="store_true",
                        help="a wheel's costs for decide.py: with --pair, its A/A units only; with --merge, "
                        "costs.json as the per-event median of the five pairs")
    parser.add_argument("--wheel", default=None, help="the exact wheel measured")
    parser.add_argument("--scenario", "--pair", action="append", default=None, metavar="NAME[#UNIT]",
                        help="with --feasibility: run only these scenarios, or single units of one as NAME#aa0.."
                        f"NAME#aa{AA_RUNS - 1}, NAME#{LINEAR} or NAME#{WORKERS} (repeatable; the control is named "
                        f"{CONTROL!r}); writes a part report and decides nothing")
    parser.add_argument("--probe", action="store_true",
                        help="with --scenario: only the first N-step run of each, timed, for the window's projection")
    parser.add_argument("--adopt", type=pathlib.Path, default=None, metavar="LISTING",
                        help="with --feasibility: stamp an earlier window's finished files listed with their "
                        "sha256 (as KEPT.txt) with this toolchain; all or nothing. Per-thread sets are checked by "
                        "the stamp of their base name, so adopted per-thread files are measured again")
    parser.add_argument("--merge", action="store_true",
                        help="with --feasibility: evaluate the gate from finished files only, never running Valgrind")
    args = parser.parse_args(argv)
    if args.feasibility and args.costs:
        print("refused: --feasibility and --costs are two modes", file=sys.stderr)
        return 2
    if args.costs and args.adopt:
        print("refused: --adopt runs with --feasibility", file=sys.stderr)
        return 2
    if (args.scenario or args.merge or args.adopt) and not (args.feasibility or args.costs):
        print("refused: --scenario, --adopt and --merge apply to --feasibility or --costs", file=sys.stderr)
        return 2
    if args.adopt and (args.scenario or args.merge):
        print("refused: --adopt runs alone", file=sys.stderr)
        return 2
    if args.probe and not args.scenario:
        print("refused: --probe needs --scenario", file=sys.stderr)
        return 2
    if args.scenario and args.merge:
        print("refused: --merge reads every part; it takes no --scenario", file=sys.stderr)
        return 2
    if args.merge:
        return merge(args)
    if not args.approved_window:
        print("refused: Callgrind runs need --approved-window (design section 8)", file=sys.stderr)
        return 2
    if not args.wheel:
        print("refused: a measurement needs --wheel, the exact artifact measured", file=sys.stderr)
        return 2
    if platform != "linux":
        print("refused: Valgrind needs Linux", file=sys.stderr)
        return 2
    manifest = evidence.load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global STAMP
    if args.scenario:
        try:
            chosen = select(manifest, args.scenario)
        except ValueError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        other = [u for u in chosen if not re.search(r"#aa[0-9]+$", u)]
        if args.costs and other and not args.probe:
            print(f"refused: --costs runs A/A units only, not {other}", file=sys.stderr)
            return 2
    # Every run that measures stamps its files and its report with the full
    # stamp (inputs, interpreter, Valgrind, CPU), so a report compared across
    # wheels later can be checked field by field (decide.py).
    try:
        STAMP = environment(args)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if args.feasibility and args.adopt:
        try:
            names = adopt(args.adopt, args.out_dir)
        except ValueError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        (args.out_dir / "feasibility.adopted.json").write_text(
            json.dumps({"adopted": names, "listing": sha256_of(str(args.adopt)), "sha256": STAMP,
                        "kernels_sha256": kernels_sha256(args),
                        "approved_window": args.approved_window}, indent=1))
        return 0
    if (args.feasibility or args.costs) and args.scenario and args.probe:
        try:
            found = probe(args.python, args.manifest, manifest, args.out_dir, args.scenario, runner=_run)
        except ValueError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        found["approved_window"] = args.approved_window
        found["sha256"] = STAMP
        found["kernels_sha256"] = kernels_sha256(args)
        label = hashlib.sha256("\n".join(sorted(found["probe"])).encode()).hexdigest()[:12]
        (args.out_dir / f"feasibility.probe.{label}.json").write_text(json.dumps(found, indent=1))
        return 0
    if (args.feasibility or args.costs) and args.scenario:
        try:
            part = measure_only(args.python, args.manifest, manifest, args.out_dir, args.scenario, runner=_run)
        except ValueError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        part["approved_window"] = args.approved_window
        part["sha256"] = STAMP
        part["kernels_sha256"] = kernels_sha256(args)
        label = hashlib.sha256("\n".join(part["units"]).encode()).hexdigest()[:12]
        kind = "costs" if args.costs else "feasibility"
        (args.out_dir / f"{kind}.part.{label}.json").write_text(json.dumps(part, indent=1))
        return 0
    if args.feasibility:
        report = feasibility(args.python, args.manifest, manifest, args.out_dir)
        report["approved_window"] = args.approved_window
        report["sha256"] = STAMP
        report["kernels_sha256"] = kernels_sha256(args)
        (args.out_dir / "feasibility.json").write_text(json.dumps(report, indent=1))
        return 0 if report["passed"] else 1
    costs = {
        s.name: cost_of(args.python, args.manifest, s, args.out_dir, runner=_run).events
        for s in evidence.scenarios(manifest)
    }
    (args.out_dir / "costs.json").write_text(json.dumps(
        {"approved_window": args.approved_window, "sha256": STAMP, "kernels_sha256": kernels_sha256(args), "costs": costs},
        indent=1))
    return 0


def main(argv=None, platform=sys.platform) -> int:
    global STAMP
    try:
        return _main(argv, platform)
    finally:
        STAMP = None


if __name__ == "__main__":
    sys.exit(main())
