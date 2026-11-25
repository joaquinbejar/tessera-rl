# SPDX-License-Identifier: MIT OR Apache-2.0
"""``python -m tessera_rl.eval``: a policy against a set of league opponents.

Each opponent is played for exactly ``episodes`` episodes, from fixed seeds,
and the output is JSON: per opponent the win, draw and loss counts, the
Wilson 95% intervals of the win rate and of the score rate (draws count
half), the mean returns of the learner and opponent slots, then the run's
setup and provenance. The same arguments give byte-identical JSON.

How the episodes run:

- every opponent gets its own league of that one entry, so every episode
  plays it, and its ``pools`` single-instance pools (``auto_reset="disabled"``)
  are the league's members: episode ``k`` runs on pool ``k mod pools``, one
  episode at a time, so no episode beyond the last is ever started;
- every episode starts with a reset seeded ``seeds[k mod len(seeds)]``, so
  episode ``k`` starts the same against every opponent;
- the pools play in waves of at most 8 episodes each (a result ring's
  capacity), and every pool's ``epoch_barrier()`` runs once after each wave,
  so each wave's outcomes are committed by its league round;
- the counts are the engine's own outcomes, read from the league: the learner
  slots' mean episode return against the opponent slots' (equal means, an
  empty learner set or NaN are a draw). The evaluation league is not frozen,
  since a frozen league records nothing; its ratings are discarded and it is
  never saved or returned. Normalization is frozen.

The run stops with an error, and writes no JSON, unless the counted episodes
are exactly ``episodes`` and no result was dropped.

A policy is ``MODULE:NAME``, a callable ``policy(observations, info) ->
actions``: ``observations`` is the instance's observation, one array per leaf
with shape ``(agents, *leaf_shape)`` (a dict of leaves); ``info`` holds
``masks`` (``(agents, *mask_shape)`` ``uint8`` or ``None``) and
``learner_slots`` (the slots it plays). It returns integer actions of shape
``(agents,)``, or ``(agents, heads)`` for several heads; only its slots are
used, and the actions are checked before the step. An ``External`` opponent
is played by the callable given with ``--external TAG=MODULE:NAME``, with the
same interface over the opponent slots. An ONNX opponent is played by its
model, given with ``--onnx-model PATH`` (repeatable; ``--onnx-library`` names
the ONNX Runtime library, the ``[onnx]`` extra's by default), or as
``evaluate(..., onnx_opponents=tessera_rl.search.OnnxOpponents(...))``; its
report entry carries the model's ``onnx_provenance``, and its results are
repeatable only in the ONNX evaluator's class (same machine, library and
model bytes; never across platforms).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from typing import Any, Callable

import numpy as np

import tessera_rl
from tessera_rl import _native

#: The z of a two-sided 95% interval.
WILSON_Z = 1.959963984540054
#: Episodes per pool between two league rounds: a result ring's capacity.
WAVE = 8


def wilson(k: int, n: int, z: float = WILSON_Z) -> list[float]:
    """The Wilson score interval ``[low, high]`` of ``k`` successes in
    ``n`` (``k`` may be a half-integer for a score)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def _load(spec: str) -> Callable:
    module, sep, name = spec.partition(":")
    if not sep or not module or not name:
        raise _native.ArgumentError(f"a policy is MODULE:NAME, not {spec!r}")
    try:
        target = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        raise _native.ArgumentError(f"cannot load policy {spec!r}: {e}") from None
    if not callable(target):
        raise _native.ArgumentError(f"policy {spec!r} is not callable")
    return target


def parse_population(text: str) -> list[tuple[int, Any]]:
    """``"heuristic,external:7,onnx:<hex>"`` as ``(id, kind)`` pairs."""
    out: list[tuple[int, Any]] = []
    for i, item in enumerate(s.strip() for s in text.split(",") if s.strip()):
        if item == "heuristic":
            out.append((i, "heuristic"))
        elif item.startswith("external:"):
            out.append((i, ("external", int(item.split(":", 1)[1]))))
        elif item.startswith("onnx:"):
            out.append((i, ("onnx", item.split(":", 1)[1])))
        else:
            raise _native.ArgumentError(f"a population entry is heuristic, external:TAG or onnx:HEX, not {item!r}")
    return out


def from_export(text: str, ids: list[int] | None) -> list[tuple[int, Any]]:
    """The opponents of a ``League.export()``: the given population ids, or
    every non-retired entry."""
    population = json.loads(text)["population"]
    out: list[tuple[int, Any]] = []
    wanted = ids if ids is not None else [p["id"] for p in population if not p["retired"]]
    by_id = {p["id"]: p for p in population}
    for i in wanted:
        if i not in by_id:
            raise _native.ArgumentError(f"population id {i} is not in the export")
        kind = by_id[i]["kind"]
        if kind == "heuristic":
            out.append((i, "heuristic"))
        elif "external" in kind:
            out.append((i, ("external", int(kind["external"]))))
        else:
            out.append((i, ("onnx", kind["onnx"])))
    return out


def _check(actions: Any, agents: int, nvec: list[int], who: str) -> np.ndarray:
    a = np.asarray(actions)
    heads = len(nvec)
    shape = (agents,) if heads <= 1 else (agents, heads)
    if a.shape != shape or not np.issubdtype(a.dtype, np.integer):
        raise _native.ArgumentError(f"{who} returned {a.dtype} {a.shape}, expected integer {shape}")
    a = a.reshape(agents, max(heads, 1)).astype(np.int64)
    for h, n in enumerate(nvec or [0]):
        col = a[:, h]
        if n and ((col < 0) | (col >= n)).any():
            raise _native.ArgumentError(f"{who} returned an action outside 0..{n - 1} (head {h})")
    return a


def _git(*args: str) -> str:
    if args == ("rev-parse", "HEAD") and os.environ.get("TESSERA_GIT_SHA"):
        return os.environ["TESSERA_GIT_SHA"]
    if args[0] == "status" and os.environ.get("TESSERA_GIT_SHA"):
        return "dirty" if os.environ["TESSERA_GIT_SHA"].endswith("+worktree") else ""
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance() -> dict:
    """The environment block of ``tools/learning_test.py``, without the
    trainer's fields."""
    return {
        "git": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unknown"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "image": os.environ.get("TESSERA_IMAGE")
        or " ".join(filter(None, (os.environ.get("ImageOS"), os.environ.get("ImageVersion"))))
        or "unknown",
        "tessera_rl": tessera_rl.__version__,
        "numpy": np.__version__,
    }


#: The default step limit of one episode: a pool without auto-reset on an
#: env with no step cap would otherwise never end.
MAX_STEPS = 100_000


def _play(env, policy, external, seed: int, learners, opponents, nvec, max_steps: int = MAX_STEPS) -> np.ndarray:
    """One episode on a single-instance pool; returns each slot's return.
    An episode longer than ``max_steps`` steps is the run's error."""
    step = env.reset(seed=seed)
    agents = env.agents
    total = np.zeros(agents)
    for _ in range(max_steps):
        obs = {k: np.asarray(v)[0] for k, v in step.observations.items()}
        masks = None if step.masks is None else np.asarray(step.masks)[0]
        rows = np.zeros((agents, max(len(nvec), 1)), dtype=np.int64)
        mine = _check(policy(obs, {"masks": masks, "learner_slots": learners}), agents, nvec, "the policy")
        rows[list(learners)] = mine[list(learners)]
        if external is not None:
            theirs = _check(external(obs, {"masks": masks, "learner_slots": opponents}), agents, nvec, "the external opponent")
            rows[list(opponents)] = theirs[list(opponents)]
        actions = rows[:, 0] if len(nvec) <= 1 else rows
        step = env.step(actions.reshape((1, *actions.shape)).astype(np.int32))
        total += np.asarray(step.rewards, dtype=np.float64)[0]
        if bool(np.asarray(step.episode_terminated)[0]) or bool(np.asarray(step.episode_truncated)[0]):
            return total
    raise RuntimeError(
        f"an episode from seed {seed} ran past {max_steps} steps: set the env's max_episode_steps or --max-steps"
    )


def evaluate(
    env_id: str,
    policy: Callable,
    opponents: list[tuple[int, Any]],
    *,
    seeds: list[int],
    episodes: int,
    pools: int = 8,
    env_kwargs: dict | None = None,
    opponent_slots: list[int] | None = None,
    external: dict[int, Callable] | None = None,
    names: dict | None = None,
    max_steps: int = MAX_STEPS,
    onnx_opponents: Any = None,
) -> dict:
    """The evaluation report (the module doc's scheme)."""
    if episodes <= 0 or pools <= 0 or not seeds or max_steps <= 0:
        raise _native.ArgumentError("episodes, pools, seeds and max_steps must be positive")
    env_kwargs = dict(env_kwargs or {})
    external = dict(external or {})
    for _, kind in opponents:
        if isinstance(kind, tuple) and kind[0] == "onnx" and onnx_opponents is None:
            raise _native.ArgumentError(f"an ONNX opponent ({kind[1]}) needs onnx_opponents= (--onnx-model PATH)")
        if isinstance(kind, tuple) and kind[0] == "external" and kind[1] not in external:
            raise _native.ArgumentError(f"External opponent tag {kind[1]} needs --external {kind[1]}=MODULE:NAME")
    results = []
    for pid, kind in opponents:
        league = tessera_rl.League([kind], [{"matchmaking": "uniform"}])
        onnx = isinstance(kind, tuple) and kind[0] == "onnx"
        envs = []
        probe = tessera_rl.make(env_id, num_envs=1, seed=0, workers=0, auto_reset="disabled", outputs="owned", **env_kwargs)
        slots = list(range(1, probe.agents)) if opponent_slots is None else list(opponent_slots)
        probe.close()
        try:
            for _ in range(pools):
                env = tessera_rl.make(
                    env_id,
                    num_envs=1,
                    seed=0,
                    workers=0,
                    auto_reset="disabled",
                    outputs="owned",
                    policy_map={s: "opponent" for s in slots},
                    league=league,
                    role=0,
                    **({"onnx_opponents": onnx_opponents} if onnx else {}),
                    **env_kwargs,
                )
                env.freeze_normalization(True)
                envs.append(env)
            agents = envs[0].agents
            nvec = list(getattr(envs[0], "action_nvec", []) or [])
            learners = tuple(s for s in range(agents) if s not in slots)
            opp = tuple(slots)
            plays = isinstance(kind, tuple) and kind[0] == "external"
            ext = external[kind[1]] if plays else None
            queues = [[k for k in range(episodes) if k % pools == m] for m in range(pools)]
            learner_sum = opponent_sum = 0.0
            dropped = 0
            while any(queues):
                for m, env in enumerate(envs):
                    for k in queues[m][:WAVE]:
                        ret = _play(env, policy, ext, seeds[k % len(seeds)], learners, opp, nvec, max_steps)
                        learner_sum += float(np.mean(ret[list(learners)])) if learners else 0.0
                        opponent_sum += float(np.mean(ret[list(opp)]))
                    queues[m] = queues[m][WAVE:]
                # One league round: every member, every wave.
                for env in envs:
                    report = env.epoch_barrier()["league"]
                    if report is not None:
                        dropped += int(report["dropped"])
            counts = league.ratings(0)["counts"][0]
            w, d, l = int(counts["wins"]), int(counts["draws"]), int(counts["losses"])
            if w + d + l != episodes or dropped != 0:
                raise RuntimeError(
                    f"opponent {pid}: {w + d + l} episodes counted for {episodes} expected ({dropped} dropped)"
                )
        finally:
            for env in envs:
                env.close()
            del league
        results.append(
            {
                "id": pid,
                "kind": kind if isinstance(kind, str) else {kind[0]: kind[1]},
                "episodes": episodes,
                "wins": w,
                "draws": d,
                "losses": l,
                "win_rate": wilson(w, episodes),
                "score_rate": wilson(w + d / 2, episodes),
                "learner_return": learner_sum / episodes,
                "opponent_return": opponent_sum / episodes,
            }
        )
        if onnx:
            prov = onnx_opponents.provenance.get(str(kind[1]).lower())
            results[-1]["onnx_provenance"] = None if prov is None else dict(prov)
    setup = {
        "env": env_id,
        "env_kwargs": env_kwargs,
        "seeds": list(seeds),
        "episodes": episodes,
        "pools": pools,
        "opponent_slots": opponent_slots,
        "max_steps": max_steps,
        **(names or {}),
    }
    return {"opponents": results, "setup": setup, "provenance": provenance()}


def _seeds(text: str) -> list[int]:
    if ":" in text:
        a, b = (int(x) for x in text.split(":", 1))
        return list(range(a, b))
    return [int(x) for x in text.split(",") if x.strip()]


def _onnx(models: list[str], library: str | None) -> Any:
    """``--onnx-model`` and ``--onnx-library`` as an ``OnnxOpponents``, or
    ``None`` without models."""
    if not models:
        if library is not None:
            raise _native.ArgumentError("--onnx-library needs --onnx-model")
        return None
    from tessera_rl.search import OnnxOpponents

    return OnnxOpponents(models, library=library)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tessera_rl.eval", description=__doc__.splitlines()[0])
    p.add_argument("--env", required=True)
    p.add_argument("--env-kwargs", default="{}")
    p.add_argument("--policy", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--population")
    g.add_argument("--league-export")
    p.add_argument("--ids")
    p.add_argument("--external", action="append", default=[])
    p.add_argument("--seeds", required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--pools", type=int, default=8)
    p.add_argument("--opponent-slots")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--output")
    p.add_argument("--onnx-model", action="append", default=[])
    p.add_argument("--onnx-library")
    a = p.parse_args(argv)
    if a.population is not None:
        opponents = parse_population(a.population)
        if a.ids:
            raise _native.ArgumentError("--ids selects from --league-export")
    else:
        with open(a.league_export, encoding="utf-8") as f:
            opponents = from_export(f.read(), [int(x) for x in a.ids.split(",")] if a.ids else None)
    external = {}
    for item in a.external:
        tag, sep, spec = item.partition("=")
        if not sep:
            raise _native.ArgumentError(f"--external is TAG=MODULE:NAME, not {item!r}")
        external[int(tag)] = _load(spec)
    report = evaluate(
        a.env,
        _load(a.policy),
        opponents,
        seeds=_seeds(a.seeds),
        episodes=a.episodes,
        pools=a.pools,
        env_kwargs=json.loads(a.env_kwargs),
        opponent_slots=[int(x) for x in a.opponent_slots.split(",")] if a.opponent_slots else None,
        external=external,
        names={"policy": a.policy, "external": sorted(a.external)},
        max_steps=a.max_steps,
        onnx_opponents=_onnx(a.onnx_model, a.onnx_library),
    )
    text = json.dumps(report, indent=1) + "\n"
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
