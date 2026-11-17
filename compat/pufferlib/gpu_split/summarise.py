# SPDX-License-Identifier: MIT OR Apache-2.0
"""The summary of a gate attempt, computed after the clock verdict was sent
(ADR-0063 item 6, docs/reviews/152-adr0063-harness-plan.md §2.1).

    python summarise.py --rows PATH --rows-sha256 HEX --manifest PATH --manifest-sha256 HEX \\
        --clock-decide PATH --clock-decide-sha256 HEX --sent-at ISO8601 --out PATH \\
        [--rerun-of-manifest PATH]

It refuses (exit 7, nothing written) unless the rows, the manifest and the
session-scope ``clock.py decide`` output hash to the declared values, the
decide output is for this manifest, its reference host, reference command and
sampler core equal the manifest's, and a rerun's ``rerun_of`` checks out
against the earlier manifest (ADR-0063 item 5(iii) and (iv)).

Then ADR-0061 A1 first: rows that miss a run, carry a non-zero rc or a
stopped group give ``failed, no result`` whatever the clock verdict. On
complete rows the clock outcome decides: ``clock-valid`` keeps the ADR-0061
summary; ``clock-invalid`` gives ``void: clock-invalid`` (``failed, no
result`` at rerun position 3) and ``instrumentation failed`` gives ``failed,
no result``, both with no ``pass`` and no scores, the cells kept. A clock.py
or clock_ref.py self-hash mismatch reported by decide is an instrumentation
failure, so it is written as a row, not refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manifest as manifestmod  # noqa: E402
import run  # noqa: E402

EXIT_REFUSED = 7
VERDICT_KEYS = ("pass", "ocean_score", "ocean_case", "tessera_score", "tessera_case")


def read_checked(path: str, sha256: str) -> bytes:
    with open(path, "rb") as f:
        data = f.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise ValueError(f"{path} hashes to {actual}, not the declared {sha256}")
    return data


def check_instruments(decide: dict, m: dict) -> None:
    ins = decide.get("instruments") or {}
    selfhash = set((decide.get("conditions") or {}).get("selfhash", []))
    if ins.get("clock_ref_host") != m.get("clock_ref_host"):
        raise ValueError("the reference host is not the manifest's")
    args = ins.get("clock_ref_args") or []
    cmd = shlex.split(m.get("clock_ref_command") or "")
    if not args or cmd[-len(args) :] != args or not cmd[-len(args) - 1].endswith("clock_ref.py"):
        raise ValueError("the reference command is not the manifest's")
    if ins.get("clock_cpu") != m.get("clock_cpu"):
        raise ValueError("the sampler core is not the manifest's")
    for key, why in (("clock_py_sha256", "clock.py sha256 mismatch"), ("clock_ref_py_sha256", "clock_ref.py sha256 mismatch")):
        if ins.get(key) != m.get(key) and why not in selfhash:
            raise ValueError(f"{key} differs from the manifest but decide did not report it")


def check_rerun(m: dict, earlier_path: str | None) -> int:
    """The rerun position (1 for a first session)."""
    ro = m.get("rerun_of")
    if ro is None:
        if earlier_path:
            raise ValueError("--rerun-of-manifest given for a first session")
        return 1
    if not earlier_path:
        raise ValueError("a rerun needs --rerun-of-manifest")
    earlier = json.loads(read_checked(earlier_path, ro["manifest_sha256"]))
    for k in manifestmod.SAME_ON_RERUN:
        if earlier.get(k) != m.get(k):
            raise ValueError(f"the rerun changed {k}")
    prev = (earlier.get("rerun_of") or {}).get("position", 1)
    if ro["position"] not in (2, 3) or prev != ro["position"] - 1:
        raise ValueError(f"rerun position {ro['position']} does not follow the earlier manifest's {prev}")
    return ro["position"]


def complete(rows: list[dict], m: dict) -> bool:
    """ADR-0061 A1: every run of the manifest's order, in order, rc 0, no
    process left."""
    want = run.from_order(m["order"])
    got = [(r.get("round"), r.get("position"), r.get("case"), r.get("mode")) for r in rows]
    return got == want and all(r.get("rc") == 0 and not r.get("group_left") for r in rows)


def summarise(rows, m, msha, decide, decide_sha, rows_sha, sent_at, position) -> dict:
    devices = sorted({r.get("device") for r in rows} - {None})
    device = devices[0] if len(devices) == 1 else "unknown"
    whole = complete(rows, m)
    s = run.summary(rows, device, whole, m, msha)
    clock = decide["outcome"]
    if whole and clock != "clock-valid":
        for k in VERDICT_KEYS:
            s.pop(k, None)
        s["gate_eligible"] = False
        s["attempt"] = "void: clock-invalid" if clock == "clock-invalid" and position < 3 else "failed, no result"
    s.update(clock=clock, clock_conditions=decide.get("conditions"), clock_decide_sha256=decide_sha,
             rows_sha256=rows_sha, clock_sent_at=sent_at, rerun_position=position)  # fmt: skip
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--rows", "--rows-sha256", "--manifest", "--manifest-sha256", "--clock-decide",
              "--clock-decide-sha256", "--sent-at", "--out"):  # fmt: skip
        p.add_argument(a, required=True)
    p.add_argument("--rerun-of-manifest")
    args = p.parse_args(argv)
    try:
        rows_data = read_checked(args.rows, args.rows_sha256)
        m = json.loads(read_checked(args.manifest, args.manifest_sha256))
        decide = json.loads(read_checked(args.clock_decide, args.clock_decide_sha256))
        if decide.get("scope") != "session":
            raise ValueError("the clock decide output is not of scope session")
        if decide.get("manifest_sha256") != args.manifest_sha256:
            raise ValueError("the clock decide output is for another manifest")
        check_instruments(decide, m)
        position = check_rerun(m, args.rerun_of_manifest)
        rows = [json.loads(x) for x in rows_data.decode().splitlines() if x.strip()]
    except (ValueError, OSError, KeyError, TypeError, IndexError) as e:
        print(f"REFUSED: {e}", flush=True)
        return EXIT_REFUSED
    s = summarise(rows, m, args.manifest_sha256, decide, args.clock_decide_sha256, args.rows_sha256,
                  args.sent_at, position)  # fmt: skip
    line = json.dumps(s, sort_keys=True)
    with open(args.out, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
