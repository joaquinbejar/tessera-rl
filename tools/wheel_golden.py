#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""An installed Tessera wheel against the committed golden fixtures
(docs/reviews/pgo-wheels-design.md, section 7, item 1).

A Rust golden binary has a different link graph from the wheel, so its pass
says nothing about the wheel. This tool replays each fixture's actions through
the installed wheel and compares its digests with the fixture.

Inputs: the fixtures, and the `.actions` files written by the Rust golden tests
under `TESSERA_GOLDEN_ACTIONS` (the actions depend on state through the scripted
baseline, which Python cannot reach).

What is compared, stated plainly:

- the fields the Python API exposes: obs, rewards, masks, the four flag arrays
  and alive, acting. Not records, terminal_obs, generations or the episode
  statistics;
- calls `t = 0` up to the one before the fixture's mid-run reseed. A partial
  reset with a new seed is not available in Sync mode from Python, so the
  fixture's later lines cannot be reproduced. The fixture's `final` line is
  not compared either.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIRS = (
    ROOT / "crates" / "tessera-rl-classic" / "tests" / "golden",
    ROOT / "crates" / "tessera-rl-games" / "tests" / "golden",
)
WORKERS = (0, 1, 2, 7, 64)
OUTPUTS = ("borrowed", "owned")
AUTO_RESET = {"NextStep": "next_step", "SameStep": "same_step"}
COMPARED = (
    "obs",
    "rewards",
    "masks",
    "agent_terminated",
    "agent_truncated",
    "episode_terminated",
    "episode_truncated",
    "alive",
    "acting",
)

_CROWDED = {"pursuers": 4, "evaders": 4, "slip": 300}
_GRID = {"size": 6, "max_steps": 40}

#: The config of each fixture as `make` options; the digests prove them.
CONFIGS = {
    "cartpole-v1-next-step": {},
    "cartpole-v1-same-step": {},
    "tag-v1-default-next-step": {},
    "tag-v1-default-same-step": {},
    "tag-v1-crowded-next-step": _CROWDED,
    "tag-v1-crowded-same-step": _CROWDED,
    "grid-v1-next-step": _GRID,
    "grid-v1-same-step": _GRID,
    "keydoor-v1-a1-next-step": {"size": 6, "agents": 1},
    "keydoor-v1-a3-same-step": {"size": 6, "agents": 3},
    "snake-v1-a1-pixels-next-step": {"size": 8, "agents": 1, "obs": "pixels"},
    "snake-v1-a2-state-same-step": {"size": 8, "agents": 2, "obs": "state"},
    "brickwall-v1-pixels-next-step": {"frameskip": 4, "lives": 2, "obs": "pixels"},
    "brickwall-v1-state-same-step": {"frameskip": 4, "lives": 2, "obs": "state"},
    "paddleduel-v1-a1-pixels-next-step": {"agents": 1, "frameskip": 4, "max_score": 2, "bounce": 1, "obs": "pixels"},
    "paddleduel-v1-a2-state-same-step": {"agents": 2, "frameskip": 4, "max_score": 2, "bounce": 1, "obs": "state"},
    "connect4-v1-next-step": {},
    "connect4-v1-same-step": {},
    "leduc-v1-next-step": {},
    "leduc-v1-same-step": {},
    "othello-v1-next-step": {},
    "othello-v1-same-step": {},
    "go9-v1-next-step": {},
    "go9-v1-same-step": {},
}


def parse_fixture(text: str) -> tuple[dict, dict[int, dict[str, str]]]:
    """(header, {t: {field: digest}}) of a format 3 fixture."""
    header: dict[str, str] = {}
    lines: dict[int, dict[str, str]] = {}
    fields: list[str] | None = None
    for raw in text.splitlines():
        if not raw or raw.startswith("#") or raw.startswith("final "):
            continue
        if "=" in raw:
            key, value = raw.split("=", 1)
            header[key] = value
            if key == "fields":
                fields = value.split(",")
            continue
        if fields is None:
            raise ValueError("a digest line before the fields header")
        parts = raw.split()
        if len(parts) != len(fields) + 1:
            raise ValueError(f"bad digest line: {raw[:40]!r}")
        lines[int(parts[0])] = dict(zip(fields, parts[1:], strict=True))
    if header.get("format") != "3":
        raise ValueError(f"unsupported fixture format {header.get('format')!r}")
    return header, lines


def reseed_step(header: dict) -> int:
    return int(header["reseed"].split(":", 1)[0])


def digests(step, digest64) -> dict[str, str]:
    """The fixture's digests of the Python-visible outputs of one call."""

    def hex_of(data: bytes) -> str:
        return f"{digest64(data):016x}"

    obs = b"".join(np.asarray(v).tobytes() for v in step.observations.values())
    out = {"obs": hex_of(obs), "masks": hex_of(b"" if step.masks is None else np.asarray(step.masks).tobytes())}
    for name in COMPARED:
        if name not in out:
            out[name] = hex_of(np.asarray(getattr(step, name)).tobytes())
    return out


def replay(name: str, header: dict, lines: dict, actions: bytes, workers: int, outputs: str, make, digest64) -> dict:
    """Replays one fixture; returns the mismatching calls."""
    env = make(
        header["env_id"],
        num_envs=int(header["num_envs"]),
        seed=int(header["seed"]),
        outputs=outputs,
        workers=workers,
        chunk_size=int(header["chunk"]),
        auto_reset=AUTO_RESET[header["auto_reset"]],
        max_episode_steps=int(header["max_episode_steps"]),
        **CONFIGS[name],
    )
    last = reseed_step(header) - 1
    rows = env.num_envs * env.agents
    expected = int(header["steps"]) * rows * 4
    if len(actions) != expected:
        env.close()
        raise ValueError(f"{name}: {len(actions)} action bytes, expected {expected}")
    every = np.frombuffer(actions, dtype="<i4").reshape(-1, env.num_envs, env.agents)
    mismatches = []
    try:
        step = env.reset()
        for t in range(last + 1):
            if t > 0:
                step = env.step(every[t - 1])
            got = digests(step, digest64)
            bad = [f for f in COMPARED if got[f] != lines[t][f]]
            if bad:
                mismatches.append({"t": t, "fields": bad})
    finally:
        env.close()
    return {
        "fixture": name,
        "workers": workers,
        "outputs": outputs,
        "calls": last + 1,
        "mismatches": mismatches,
    }


def fixtures() -> dict[str, pathlib.Path]:
    return {p.stem: p for d in FIXTURE_DIRS for p in sorted(d.glob("*.golden"))}


def run(actions_dir: pathlib.Path, make, digest64) -> dict:
    found = fixtures()
    if set(found) != set(CONFIGS):
        raise ValueError(f"fixtures and CONFIGS differ: {sorted(set(found) ^ set(CONFIGS))}")
    results = []
    for name, path in found.items():
        header, lines = parse_fixture(path.read_text(encoding="utf-8"))
        actions = (actions_dir / f"{name}.actions").read_bytes()
        for workers in WORKERS:
            for outputs in OUTPUTS:
                results.append(replay(name, header, lines, actions, workers, outputs, make, digest64))
    return {
        "compared_fields": list(COMPARED),
        "results": results,
        "passed": bool(results) and not any(r["mismatches"] for r in results),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actions", required=True, type=pathlib.Path, help="the TESSERA_GOLDEN_ACTIONS directory")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    import xxhash

    import tessera_rl as t

    report = run(args.actions, t.make, xxhash.xxh3_64_intdigest)
    report["build_profile"] = t._native.build_profile()
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    failed = [r for r in report["results"] if r["mismatches"]]
    print(json.dumps({"passed": report["passed"], "runs": len(report["results"]), "failed": len(failed)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
