#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Behaviour of an installed candidate wheel against W0, and across targets
(docs/reviews/pgo-wheels-design.md r5, section 7, items 2 to 4).

    determinism.py record --wheel W.whl --out w1.json     # in the wheel's own environment
    determinism.py compare w0.json w1.json --out r.json   # exit 1 on any difference

Each instance's outputs are hashed row by row, so Sync, Pipelined and Async runs
are comparable although Async batches depend on timing. Actions depend only on
the instance, its slot, the step and its mask. Internal equalities (worker counts,
borrowed against owned, continuation) are recorded and must hold.

Several discrete heads (``MultiDiscrete``, the RTS of SPEC 11.3) take one
action per head and entity group (``action_nvec``, ``mask_heads``,
``entity_group_heads``): per unit, the type is drawn from the type mask, then
target and product from their masks conditioned on that type, each draw a
function of (instance, slot, t, unit, head) alone (``multi_action``). An
empty mask row plays choice 0, the no-op. Every mode runs for it; actions go
to the pool as ``(rows, agents, heads)``.

Items 5 (the Python suite) and 6 (the num crate's tests under the profile) are
commands of the wheel job, not of this tool. A difference blocks the wheel and is
a bug; the goldens are never re-blessed for it.

Discrete actions only: an environment with continuous actions (Pendulum,
``CONTINUOUS_GAP``) needs its own action path here, a stated gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

WORKERS = (0, 1, 2, 7, 64)
AUTO_RESET = ("next_step", "same_step")
OUTPUTS = ("borrowed", "owned")
NUM_ENVS = 16
CHUNK = 4
BATCH = 4
STEPS = 200
SEED = 20261108
SNAPSHOT_STEPS = (50, 200)
CONTINUE_AT = 100
CONTINUE_STEPS = 20

#: Registered environments this tool cannot drive yet (discrete actions only).
CONTINUOUS_GAP = frozenset({"tessera/Pendulum-v1"})
#: Every stated gap.
GAPS = CONTINUOUS_GAP

ENVS = {
    "cartpole": ("tessera/CartPole-v1", {}),
    "mountaincar": ("tessera/MountainCar-v1", {}),
    "acrobot": ("tessera/Acrobot-v1", {}),
    "tag": ("tessera/Tag-v1", {"pursuers": 2, "evaders": 2}),
    "grid": ("tessera/Grid-v1", {}),
    "keydoor": ("tessera/GridKeyDoor-v1", {}),
    "snake-pixels": ("tessera/Snake-v1", {"obs": "pixels"}),
    "snake-state": ("tessera/Snake-v1", {"obs": "state"}),
    "brickwall-pixels": ("tessera/BrickWall-v1", {"obs": "pixels"}),
    "brickwall-state": ("tessera/BrickWall-v1", {"obs": "state"}),
    "paddleduel-state": ("tessera/PaddleDuel-v1", {"obs": "state"}),
    "connect4": ("tessera/Connect4-v1", {}),
    "othello": ("tessera/Othello-v1", {}),
    "go9": ("tessera/Go9-v1", {}),
    "leduc": ("tessera/Leduc-v1", {}),
    # Few units, a small map: the per-unit Python draws stay cheap.
    "rts": ("tessera/RTS-v1", {"size": 12, "max_units": 8}),
}


def mix(x: int) -> int:
    """splitmix64."""
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def action(instance: int, slot: int, t: int, choices: int, legal) -> int:
    """A legal action from (instance, slot, t) alone; 0 if none is legal."""
    x = mix((instance << 40) ^ (slot << 32) ^ t ^ SEED)
    if legal is None:
        return x % choices
    allowed = [i for i, ok in enumerate(legal) if ok]
    return allowed[x % len(allowed)] if allowed else 0


def _layout(env):
    """The multi-head layout of ``env`` (``None`` for one discrete head):
    the heads of one entity group as ``(offset, parents, size)``, the period
    of an entity block in the mask row, and the entity count."""
    nvec = tuple(env.action_nvec)
    if len(nvec) <= 1:
        return None
    heads = list(env.mask_heads)
    per = env.entity_group_heads or len(heads)
    group = heads[:per]
    entities = len(nvec) // per
    period = heads[per][0] if entities > 1 else env._mask_len
    if group[0][1] != () or any(parents not in ((), (group[0][2],)) for _, parents, _ in group[1:]):
        raise ValueError("multi-head actions: heads other than the first may only depend on it")
    return group, period, entities


def multi_action(instance: int, slot: int, t: int, layout, row) -> list[int]:
    """Legal multi-head actions from (instance, slot, t) alone, one per head
    of each entity group: the first head under its mask, then each later
    head under its mask conditioned on the first head's choice. ``row`` is
    the agent's flat mask row, or ``None`` (every choice legal)."""
    group, period, entities = layout
    key = mix((instance << 40) ^ (slot << 32) ^ t ^ SEED)
    out = []
    for e in range(entities):
        first = 0
        for h, (offset, parents, size) in enumerate(group):
            x = mix(key ^ ((e << 8) | h))
            if row is None:
                choice = x % size
            else:
                start = e * period + offset + (first * size if parents else 0)
                allowed = [i for i, ok in enumerate(row[start : start + size]) if ok]
                choice = allowed[x % len(allowed)] if allowed else 0
            if h == 0:
                first = choice
            out.append(choice)
    return out


def actions_for(ids, step, agents: int, choices: int, t_of, layout=None):
    """One action per (instance, agent) row; with ``layout`` (multi-head),
    an array ``(rows, agents, heads)`` of ``multi_action`` rows."""
    if layout is not None:
        import numpy as np

        rows = [
            [
                multi_action(int(instance), slot, t_of(int(instance)), layout, None if step.masks is None else step.masks[row][slot].tolist())
                for slot in range(agents)
            ]
            for row, instance in enumerate(ids)
        ]
        return np.array(rows, np.int32).reshape(len(rows), agents, -1)
    out = []
    for row, instance in enumerate(ids):
        for slot in range(agents):
            legal = None if step.masks is None else step.masks[row][slot].tolist()
            out.append(action(int(instance), slot, t_of(int(instance)), choices, legal))
    return out


class Trace:
    """One sha256 per instance over its rows of every output."""

    def __init__(self, n: int):
        self.h = [hashlib.sha256() for _ in range(n)]

    def add(self, ids, step) -> None:
        arrays = [*step.observations.values(), *(a for f, a in zip(step._fields, step) if f != "observations" and a is not None)]
        for row, instance in enumerate(ids):
            digest = self.h[int(instance)]
            for array in arrays:
                digest.update(array[row].tobytes())

    def result(self) -> list[str]:
        return [d.hexdigest() for d in self.h]


def make_env(t, name: str, auto_reset: str, workers: int, outputs: str, mode: str = "sync", **more):
    env_id, config = ENVS[name]
    return t.make(env_id, num_envs=NUM_ENVS, seed=SEED, chunk_size=CHUNK, auto_reset=auto_reset, workers=workers, outputs=outputs, mode=mode, **config, **more)


def run_sync(env, steps: int = STEPS) -> dict:
    trace = Trace(env.num_envs)
    ids = range(env.num_envs)
    snapshots = {}
    step = env.reset()
    trace.add(ids, step)
    for t in range(1, steps + 1):
        acts = actions_for(ids, step, env.agents, env.action_choices, lambda _: t, _layout(env))
        step = env.step(_array(acts, env))
        trace.add(ids, step)
        if t in SNAPSHOT_STEPS:
            snapshots[str(t)] = [hashlib.sha256(env.snapshot(i)).hexdigest() for i in ids]
    return {"trace": trace.result(), "snapshots": snapshots}


def _array(acts, env):
    import numpy as np

    if _layout(env) is not None:
        return acts
    return np.array(acts, np.int32).reshape(-1, env.agents)


def run_pipelined(env, steps: int = STEPS) -> dict:
    trace = Trace(env.num_envs)
    counts = [0] * env.num_envs
    received = [0, 0]
    for half in env.reset():
        trace.add(half.ids, half.step)
        env.send(half.half, actions_for(half.ids, half.step, env.agents, env.action_choices, lambda i: 1, _layout(env)))
    while min(received) < steps:
        out = env.recv()
        trace.add(out.ids, out.step)
        received[out.half] += 1
        for i in out.ids:
            counts[i] += 1
        if received[out.half] < steps:
            next_t = received[out.half] + 1
            env.send(out.half, actions_for(out.ids, out.step, env.agents, env.action_choices, lambda i: next_t, _layout(env)))
    return {"trace": trace.result()}


def run_async(env, steps: int = STEPS) -> dict:
    trace = Trace(env.num_envs)
    counts = [0] * env.num_envs
    batch = env.reset()
    trace.add(batch.ids, batch.step)
    env.send(batch.ids.tolist(), actions_for(batch.ids, batch.step, env.agents, env.action_choices, lambda i: 1, _layout(env)))
    remaining = env.num_envs * steps
    while remaining:
        batch = env.recv()
        trace.add(batch.ids, batch.step)
        remaining -= len(batch.ids)
        for i in batch.ids.tolist():
            counts[i] += 1
        keep = [(row, int(i)) for row, i in enumerate(batch.ids) if counts[int(i)] < steps]
        if keep:
            rows = [r for r, _ in keep]
            ids = [i for _, i in keep]
            view = type(batch.step)(
                observations={k: v[rows] for k, v in batch.step.observations.items()},
                **{f: (None if a is None else a[rows]) for f, a in zip(batch.step._fields, batch.step) if f != "observations"},
            )
            env.send(ids, actions_for(ids, view, env.agents, env.action_choices, lambda i: counts[i] + 1, _layout(env)))
    return {"trace": trace.result()}


def continuation(t, name: str, auto_reset: str) -> dict:
    """The continuation of a pool, of its loaded file and of imported records."""
    ids = range(NUM_ENVS)

    def advance(env, step, first: int, count: int) -> list[str]:
        trace = Trace(env.num_envs)
        for s in range(first, first + count):
            acts = actions_for(ids, step, env.agents, env.action_choices, lambda _: s, _layout(env))
            step = env.step(_array(acts, env))
            trace.add(ids, step)
        return trace.result()

    env = make_env(t, name, auto_reset, 0, "owned")
    step = env.reset()
    for s in range(1, CONTINUE_AT + 1):
        step = env.step(_array(actions_for(ids, step, env.agents, env.action_choices, lambda _: s, _layout(env)), env))
    data = env.encode()
    records = [env.snapshot(i) for i in ids]
    held = step
    original = advance(env, step, CONTINUE_AT + 1, CONTINUE_STEPS)
    env.close()

    def resumed(install) -> list[str]:
        other = make_env(t, name, auto_reset, 0, "owned")
        other.reset()
        install(other)
        out = advance(other, held, CONTINUE_AT + 1, CONTINUE_STEPS)
        other.close()
        return out

    def imported(other) -> None:
        for i, record in enumerate(records):
            other.import_record(i, record)

    loaded = resumed(lambda other: other.load(data))
    via_records = resumed(imported)
    return {"trace": original, "load_equal": loaded == original, "import_equal": via_records == original}


def fork_equal(t, name: str, auto_reset: str) -> bool:
    """Forks of one instance, given the same actions, have the same outputs."""
    import numpy as np

    env = make_env(t, name, auto_reset, 0, "owned")
    step = env.reset()
    ids = range(NUM_ENVS)
    for s in range(1, 31):
        step = env.step(_array(actions_for(ids, step, env.agents, env.action_choices, lambda _: s, _layout(env)), env))
    env.fork(0, [1])
    same = True
    for s in range(31, 41):
        first = actions_for([0], _rows(step, [0]), env.agents, env.action_choices, lambda _: s, _layout(env))
        rest = actions_for(range(2, NUM_ENVS), _rows(step, range(2, NUM_ENVS)), env.agents, env.action_choices, lambda _: s, _layout(env))
        acts = first * 2 + rest if _layout(env) is None else np.concatenate([first, first, rest])
        step = env.step(_array(acts, env))
        same &= step.rewards[0].tobytes() == step.rewards[1].tobytes()
        same &= all(v[0].tobytes() == v[1].tobytes() for v in step.observations.values())
    env.close()
    return bool(same)


def _rows(step, rows):
    rows = list(rows)
    return type(step)(
        observations={k: v[rows] for k, v in step.observations.items()},
        **{f: (None if a is None else a[rows]) for f, a in zip(step._fields, step) if f != "observations"},
    )


def record(t, envs=tuple(ENVS), workers=WORKERS, steps: int = STEPS) -> dict:
    runs: dict = {}
    checks: dict = {}
    for name in envs:
        for auto in AUTO_RESET:
            base = None
            for w in workers:
                for outputs in OUTPUTS:
                    env = make_env(t, name, auto, w, outputs)
                    try:
                        result = run_sync(env, steps)
                    finally:
                        env.close()
                    runs[f"{name}/{auto}/sync/{w}/{outputs}"] = result
                    base = base or result
                    checks[f"{name}/{auto}/sync/{w}/{outputs}/same-as-first"] = result == base
            for mode, runner, more in (("pipelined", run_pipelined, {}), ("async", run_async, {"batch_size": BATCH})):
                for outputs in OUTPUTS:
                    env = make_env(t, name, auto, 2, outputs, mode, **more)
                    try:
                        runs[f"{name}/{auto}/{mode}/2/{outputs}"] = runner(env, steps)
                    finally:
                        env.close()
            c = continuation(t, name, auto)
            runs[f"{name}/{auto}/continuation"] = {"trace": c["trace"]}
            checks[f"{name}/{auto}/load"] = c["load_equal"]
            checks[f"{name}/{auto}/import"] = c["import_equal"]
            checks[f"{name}/{auto}/fork"] = fork_equal(t, name, auto)
    return {"runs": runs, "checks": checks}


def differences(a: dict, b: dict) -> list[str]:
    """Why two recorded reports do not describe the same behaviour."""
    out = []
    keys_a, keys_b = set(a["runs"]), set(b["runs"])
    out += [f"missing run: {k}" for k in sorted(keys_a ^ keys_b)]
    out += [f"differs: {k}" for k in sorted(keys_a & keys_b) if a["runs"][k] != b["runs"][k]]
    for name, report in (("first", a), ("second", b)):
        out += [f"{name}: check failed: {k}" for k, ok in sorted(report["checks"].items()) if ok is not True]
    if not keys_a:
        out.append("no runs")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--wheel", required=True, help="the exact wheel installed here")
    c = sub.add_parser("compare")
    c.add_argument("reports", nargs="+")
    c.add_argument("--out", default=None, help="the result, with the wheels compared, for decide.py")
    args = parser.parse_args(argv)
    if args.command == "record":
        import tessera_rl as t

        report = record(t)
        with open(args.wheel, "rb") as f:
            report["wheel_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f)
        return 0
    if len(args.reports) < 2:
        print("compare needs at least two reports", file=sys.stderr)
        return 2
    reports = []
    for path in args.reports:
        with open(path, encoding="utf-8") as f:
            reports.append(json.load(f))
    found = [d for other in reports[1:] for d in differences(reports[0], other)]
    result = {"passed": not found, "differences": found[:50], "count": len(found), "wheels": [r.get("wheel_sha256") for r in reports]}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    print(json.dumps(result))
    return 0 if not found else 1


if __name__ == "__main__":
    sys.exit(main())
