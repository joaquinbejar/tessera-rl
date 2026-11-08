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

Every call of every fixture is compared, the mid-run reseed and the `final`
line included, through the native layer that the wheel ships. `records` is the
XXH3-64 of each instance's canonical `snapshot` bytes, so it covers all of the
persistent header and state.

Omitted, stated plainly: the output arrays `generations`, `episode_return`,
`episode_length`, `episode_illegal` and `episode_invalid_agent_events`. The wheel
does not expose them, so no user of it can read them. The report lists them.
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
    "records",
    "obs",
    "terminal_obs",
    "rewards",
    "masks",
    "agent_terminated",
    "agent_truncated",
    "episode_terminated",
    "episode_truncated",
    "alive",
    "acting",
)
OMITTED = (
    "generations",
    "episode_return",
    "episode_length",
    "episode_illegal",
    "episode_invalid_agent_events",
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


def parse_fixture(text: str) -> tuple[dict, dict[int, dict[str, str]], list[str]]:
    """(header, {t: {field: digest}}, final hashes) of a format 3 fixture."""
    header: dict[str, str] = {}
    lines: dict[int, dict[str, str]] = {}
    fields: list[str] | None = None
    final: list[str] | None = None
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("final "):
            final = raw.split()[1:]
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
    if final is None:
        raise ValueError("a fixture without a final line")
    return header, lines, final


def parse_reseed(header: dict) -> tuple[int, list[int], int]:
    """(step, ids, seed) of the fixture's mid-run reseed."""
    step, ids, seed = header["reseed"].split(":")
    return int(step), json.loads(ids), int(seed)


def digests(env, raw: dict, same_step: bool, digest64) -> dict[str, str]:
    """The fixture's digests of one call, from the native output `raw`."""

    def hex_of(data: bytes) -> str:
        return f"{digest64(data):016x}"

    def field(name):
        return bytes(raw[name]) if raw.get(name) is not None else b""

    terminal = bytearray()
    ended = np.frombuffer(field("episode_terminated"), np.uint8)
    truncated = np.frombuffer(field("episode_truncated"), np.uint8)
    if same_step:
        for leaf in raw["terminal_observations"]:
            data = bytes(leaf)
            per = len(data) // env.num_envs
            for i in np.flatnonzero(ended | truncated):
                terminal += data[int(i) * per : (int(i) + 1) * per]
    records = b"".join(digest64(env.snapshot(i)).to_bytes(8, "little") for i in range(env.num_envs))
    out = {
        "records": hex_of(records),
        "obs": hex_of(b"".join(bytes(v) for v in raw["observations"])),
        "terminal_obs": hex_of(bytes(terminal)),
    }
    for name in COMPARED:
        if name not in out:
            out[name] = hex_of(field(name))
    return out


def replay(name: str, header: dict, lines: dict, final: list[str], actions: bytes, workers: int, outputs: str, make, digest64) -> dict:
    """Replays one whole fixture; returns the mismatching calls."""
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
    steps = int(header["steps"])
    if len(actions) != steps * env.num_envs * env.agents * 4:
        env.close()
        raise ValueError(f"{name}: {len(actions)} action bytes, expected {steps * env.num_envs * env.agents * 4}")
    every = np.frombuffer(actions, dtype="<i4").reshape(-1, env.num_envs, env.agents)
    same_step = header["auto_reset"] == "SameStep"
    at, ids, seed = parse_reseed(header)
    mismatches = []
    final_ok = False
    try:
        raw = env._native.reset()
        for t in range(steps + 1):
            if t > 0:
                if t == at:
                    env._native.reset(seed, ids)
                raw = env._native.step(every[t - 1].tobytes())
            got = digests(env, raw, same_step, digest64)
            bad = [f for f in COMPARED if got[f] != lines[t][f]]
            if bad:
                mismatches.append({"t": t, "fields": bad})
        hashes = [f"{digest64(env.snapshot(i)):016x}" for i in range(env.num_envs)]
        final_ok = hashes == final
    finally:
        env.close()
    return {
        "fixture": name,
        "workers": workers,
        "outputs": outputs,
        "calls": steps + 1,
        "mismatches": mismatches,
        "final_ok": final_ok,
    }


def fixtures() -> dict[str, pathlib.Path]:
    return {p.stem: p for d in FIXTURE_DIRS for p in sorted(d.glob("*.golden"))}


def run(actions_dir: pathlib.Path, make, digest64) -> dict:
    found = fixtures()
    if set(found) != set(CONFIGS):
        raise ValueError(f"fixtures and CONFIGS differ: {sorted(set(found) ^ set(CONFIGS))}")
    results = []
    for name, path in found.items():
        header, lines, final = parse_fixture(path.read_text(encoding="utf-8"))
        actions = (actions_dir / f"{name}.actions").read_bytes()
        for workers in WORKERS:
            for outputs in OUTPUTS:
                results.append(replay(name, header, lines, final, actions, workers, outputs, make, digest64))
    return {
        "compared_fields": list(COMPARED),
        "omitted_fields": list(OMITTED),
        "results": results,
        "passed": bool(results) and all(r["final_ok"] and not r["mismatches"] for r in results),
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
    failed = [r for r in report["results"] if r["mismatches"] or not r["final_ok"]]
    print(json.dumps({"passed": report["passed"], "runs": len(report["results"]), "failed": len(failed)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
