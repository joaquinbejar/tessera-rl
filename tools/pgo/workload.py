#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""One manifest scenario through the installed wheel's Python API
(docs/reviews/pgo-wheels-design.md r4, section 4): what PGO trains on and what
Callgrind measures.

    workload.py --manifest tools/pgo/manifest.toml --scenario NAME [--steps N]

`--steps` overrides the manifest's count, for the N/2N/3N lengths. Every pool is
closed before returning, as profile counters are written at normal exit.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence  # noqa: E402


def choose(rng, masks, shape, choices: int) -> np.ndarray:
    """A seeded legal action per agent row; uniform without masks."""
    if masks is None:
        return rng.integers(0, choices, shape, dtype=np.int32)
    scores = rng.random(np.asarray(masks).shape)
    return np.argmax(scores * np.asarray(masks), axis=-1).astype(np.int32)


def _make(t, s: evidence.Scenario, manifest: dict):
    options = dict(s.config)
    if s.batch_size:
        options["batch_size"] = s.batch_size
    return t.make(
        s.env_id,
        num_envs=s.num_envs,
        seed=manifest["seed"],
        chunk_size=manifest["chunk_size"],
        workers=s.workers,
        mode=s.mode,
        outputs=s.outputs,
        **options,
    )


def _sync(env, rng, steps: int) -> None:
    step = env.reset()
    for _ in range(steps):
        step = env.step(choose(rng, step.masks, (env.num_envs, env.agents), env.action_choices))


def _pipelined(env, rng, steps: int) -> None:
    results = env.reset()
    for half in results:
        env.send(half.half, choose(rng, half.step.masks, (len(half.ids), env.agents), env.action_choices))
    for _ in range(steps):
        out = env.recv()
        env.send(out.half, choose(rng, out.step.masks, (len(out.ids), env.agents), env.action_choices))
    for _ in env.halves:
        env.recv()


def _async(env, rng, steps: int) -> None:
    batch = env.reset()
    env.send(batch.ids.tolist(), choose(rng, batch.step.masks, (len(batch.ids), env.agents), env.action_choices))
    sent = env.num_envs
    received = 0
    for _ in range(steps):
        batch = env.recv()
        received += len(batch.ids)
        env.send(batch.ids.tolist(), choose(rng, batch.step.masks, (len(batch.ids), env.agents), env.action_choices))
        sent += len(batch.ids)
    while received < sent:
        received += len(env.recv().ids)


def _state_operations(env, rng, steps: int) -> None:
    step = env.reset()
    for _ in range(steps):
        step = env.step(choose(rng, step.masks, (env.num_envs, env.agents), env.action_choices))
        env.import_record(1, env.snapshot(0))
        env.fork(0, [2, 3])
        env.load(env.encode())


def _uniform(obs, legal, hashes):
    priors = np.ones(legal.shape, np.float64) / legal.shape[1]
    return priors, np.zeros(legal.shape[0], np.float64)


def _search(s: evidence.Scenario, seed: int, steps: int) -> None:
    from tessera_rl.search import Search

    search = Search(env_id=s.env_id, simulations=s.simulations, batch=s.batch, eviction="oldest", seed=seed)
    for _ in range(steps):
        search.run(_uniform)


def run_scenario(s: evidence.Scenario, manifest: dict, steps: int | None = None) -> None:
    import tessera_rl as t

    count = s.steps if steps is None else steps
    if s.kind == "search":
        _search(s, manifest["seed"], count)
        return
    rng = np.random.default_rng(manifest["action_seed"])
    env = _make(t, s, manifest)
    try:
        if s.kind == "state-operations":
            _state_operations(env, rng, count)
        elif s.mode == "pipelined":
            _pipelined(env, rng, count)
        elif s.mode == "async":
            _async(env, rng, count)
        else:
            _sync(env, rng, count)
    finally:
        env.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args(argv)
    manifest = evidence.load_manifest(args.manifest)
    found = [s for s in evidence.scenarios(manifest) if s.name == args.scenario]
    if len(found) != 1:
        print(f"unknown scenario {args.scenario!r}", file=sys.stderr)
        return 2
    run_scenario(found[0], manifest, args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
