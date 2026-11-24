# SPDX-License-Identifier: MIT OR Apache-2.0
"""Per-environment learning protocols (SPEC 16.1, 18; #370, design
docs/reviews/learning-370).

A protocol names the environment, its tier, the PPO hyperparameters, the
training budget and the adapter path a learning run goes through. CartPole's
protocol is the original one: ``learning_test.py`` keeps running it through
``reference_training.train`` unchanged, so its calibration fingerprint (which
hashes that file) does not move.

Tier T1 (discrete actions, one agent, no masks) runs the **same** PPO loop:
``train`` points ``reference_training``'s two environment factories at the
protocol's environment for the duration of one call, so no training code is
copied. Tiers T2 (continuous) and T3 (masked, two-player against the scripted
baseline and a uniform-legal player) come in later changes.

``random_scores`` gives the evaluation score of a uniform random policy on the
same evaluation seeds, the reference for the "learns at all" check
(``learning_gate.learns``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import tomllib
from dataclasses import asdict

#: The protocol table (compat/learning/protocols.toml). Each environment's
#: fingerprint covers only its own entry, so adding a protocol leaves every
#: other calibration valid. ``hyper`` overrides ``reference_training.Hyper``'s
#: CleanRL CartPole defaults; ``total_steps`` is the calibration budget.
PROTOCOLS_FILE = pathlib.Path(__file__).resolve().parents[1] / "compat" / "learning" / "protocols.toml"
PROTOCOLS: dict[str, dict] = tomllib.loads(PROTOCOLS_FILE.read_text())


def protocol(name: str) -> dict:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"no learning protocol for {name!r}; known: {sorted(PROTOCOLS)}") from None


def hyper(name: str, total_steps: int):
    import reference_training as rt  # noqa: PLC0415

    return rt.Hyper(**{**protocol(name)["hyper"], "total_steps": total_steps})


def fingerprint(name: str) -> str:
    """What determines a score of a non-CartPole protocol besides the seed:
    the protocol, both trainer sources and the numeric libraries."""
    import gymnasium  # noqa: PLC0415
    import numpy  # noqa: PLC0415
    import torch  # noqa: PLC0415

    import reference_training as rt  # noqa: PLC0415

    sources = {
        "reference_training": hashlib.sha256(pathlib.Path(rt.__file__).read_bytes()).hexdigest(),
        "learning_protocols": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }
    payload = json.dumps(
        {
            "protocol": protocol(name),
            "hyper": asdict(hyper(name, protocol(name)["total_steps"])),
            "sources": sources,
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@contextlib.contextmanager
def _factories(env_id: str):
    """``reference_training``'s vector and evaluation factories pointed at
    ``env_id`` (Tessera backend only), restored on exit, also when the run
    raises.

    Not concurrent-safe: the factories are module globals, so two protocols
    must never train at the same time in one process (threads). The
    calibration and the nightly run one protocol per process."""
    import gymnasium  # noqa: PLC0415

    import reference_training as rt  # noqa: PLC0415
    import tessera_rl.gym  # noqa: PLC0415, F401  (registers the ids)

    def make_vector(backend, num_envs, seed):
        if backend != "tessera":
            raise ValueError(f"protocols run the tessera backend, not {backend!r}")
        envs = tessera_rl.gym.make(env_id, num_envs=num_envs, seed=seed, autoreset_mode=rt.SAME_STEP)
        return gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)

    def make_single(backend):
        return gymnasium.make(env_id)

    saved = (rt.make_vector, rt.make_single)
    rt.make_vector, rt.make_single = make_vector, make_single
    try:
        yield rt
    finally:
        rt.make_vector, rt.make_single = saved


def train(name: str, seed: int, total_steps: int, eval_episodes: int) -> dict:
    """One training run of protocol ``name`` (tier T1)."""
    p = protocol(name)
    if p["tier"] != "T1":
        raise ValueError(f"tier {p['tier']} is not supported yet")
    with _factories(p["env"]) as rt:
        return rt.train("tessera", seed, hyper(name, total_steps), eval_episodes)


def random_scores(name: str, episodes: int, seed: int = 0) -> list[float]:
    """The evaluation score of a uniform random policy, per episode, on the
    evaluation seeds (``EVAL_SEED_BASE + i``) of ``reference_training``."""
    import gymnasium  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    import reference_training as rt  # noqa: PLC0415
    import tessera_rl.gym  # noqa: PLC0415, F401

    env = gymnasium.make(protocol(name)["env"])
    env.action_space.seed(seed)
    out = []
    for i in range(episodes):
        env.reset(seed=rt.EVAL_SEED_BASE + i)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, _ = env.step(env.action_space.sample())
            total += float(r)
            done = term or trunc
        out.append(total)
    env.close()
    return [float(x) for x in np.asarray(out)]
