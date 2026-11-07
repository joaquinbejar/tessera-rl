# SPDX-License-Identifier: MIT OR Apache-2.0
"""Gymnasium adapter (SPEC 14.2, ADR-0003): single-agent Tessera pools as a
``gymnasium.vector.VectorEnv``.

- Outputs are ``owned`` by default (fresh arrays), because code written for
  Gymnasium commonly keeps observations across steps; ``outputs="borrowed"``
  is available for callers that accept the borrow window.
- ``autoreset_mode`` is declared in ``metadata`` as Gymnasium 1.x requires.
  In ``SAME_STEP`` mode the final observation of an episode that ended is in
  ``info["final_obs"]`` (an object array, ``None`` elsewhere, with the mask
  ``info["_final_obs"]``), as ``SyncVectorEnv`` does; it is a step output
  (ADR-0039), valid only for that step. In ``DISABLED`` mode, reset ended
  instances with ``reset(options={"reset_mask": mask})``.
- Rewards are ``float32`` (the engine's precision); terminations and
  truncations are ``bool``.
- ``reset(seed=...)`` follows ``SyncVectorEnv``: ``s`` gives instance ``i``
  the seed ``s + i``; a list gives one seed (or ``None``, keep) per
  instance; ``reset_mask`` then selects which instances reset, validated
  with the same error types.
- Time limits: the vector env truncates at ``max_episode_steps`` (500 by
  default, the registered value under ``make_vec``; ``None`` or ``-1`` for
  none). Under ``gymnasium.make`` the single env has no native limit and
  Gymnasium's ``TimeLimit`` wrapper owns it.

Status: the adapter passes Gymnasium's env checker and the vector API
tests of this repository, and a nightly reference run (ADR-0018,
``tools/reference_training.py``, workflow ``reference-training.yml``)
trains PPO on ``tessera/CartPole-v1`` through it and gates it against the
same PPO on Gymnasium's ``CartPole-v1``. That validates this adapter; it
is not a drop-in compatibility claim for SB3, RLlib or Tianshou.

Snake-v1 and PaddleDuel-v1 are exposed with ``agents=1`` (PaddleDuel's
agent is the left paddle against the scripted right one); ``agents=2``
is refused, use the PettingZoo adapter. Snake's action space is plain
``Discrete(4)`` without its mask: the reverse of the current heading is
an illegal action the engine treats as a no-op (``IllegalActionPolicy::Noop``),
so the snake keeps its heading, with no error and no penalty.

Seeding: ``seed=None`` at construction draws one 64-bit seed from OS
entropy and exposes it as ``construction_seed``, so the run can be
reproduced by passing it back; a given seed is deterministic.

Importing this module registers the Tessera environment ids with Gymnasium
(``gymnasium.make_vec("tessera/CartPole-v1", num_envs=8)``); ``gymnasium.make``
returns :class:`TesseraEnv`, a one-instance ``gymnasium.Env`` with
``DISABLED`` auto-reset, which the Gymnasium env checker accepts.
"""

from __future__ import annotations

import secrets
from typing import Any

try:
    import gymnasium
    from gymnasium import spaces
    from gymnasium.vector import AutoresetMode, VectorEnv
    from gymnasium.vector.utils import batch_space
except ImportError as e:  # pragma: no cover - exercised only without gymnasium
    raise ImportError("tessera_rl.gym needs gymnasium>=1.4 (pip install 'tessera-rl[gymnasium]')") from e

import numpy as np

from tessera_rl import _native
from tessera_rl._env import _discrete_actions, native_kwargs

#: Single-agent environments the adapter exposes.
_MAKERS = {
    "tessera/CartPole-v1": _native.make_cartpole,
    "tessera/Grid-v1": _native.make_grid,
    "tessera/BrickWall-v1": _native.make_brickwall,
    "tessera/Snake-v1": _native.make_snake,
    "tessera/PaddleDuel-v1": _native.make_paddleduel,
}

#: Two-slot environments admitted with one agent (registered with
#: ``agents=1``). Admission rests on a lifetime invariant of each kernel:
#: with ``agents=1`` only slot 0 is ever alive or acting (their headers are
#: ``NoAdaptive`` and the kernels never bring slot 1 to life), so the
#: adapter reads and writes slot 0 of each row. ``initial_agents == 1``
#: alone is not admission: a dynamic population could start with one agent
#: and grow, which needs an explicit capability contract.
_SLOT0 = frozenset({"tessera/Snake-v1", "tessera/PaddleDuel-v1"})

#: The registered time limit, where it is not 500.
_LIMITS = {"tessera/BrickWall-v1": 4000, "tessera/Snake-v1": 1000, "tessera/PaddleDuel-v1": 5000}

_NATIVE_MODE = {
    AutoresetMode.NEXT_STEP: "next_step",
    AutoresetMode.SAME_STEP: "same_step",
    AutoresetMode.DISABLED: "disabled",
}


def _autoreset_mode(mode) -> AutoresetMode:
    if isinstance(mode, AutoresetMode):
        return mode
    for m in AutoresetMode:
        if mode in (m.value, m.name, m.name.lower()):
            return m
    raise _native.ArgumentError(f"unknown autoreset_mode {mode!r}")


def _native_env(env_id: str, num_envs: int, seed: int, mode: AutoresetMode, outputs: str, kwargs: dict):
    try:
        maker = _MAKERS[env_id]
    except KeyError:
        raise _native.ArgumentError(f"unknown single-agent environment {env_id!r}; known: {sorted(_MAKERS)}") from None
    if env_id in _SLOT0:
        kwargs = {"agents": 1, **kwargs}
    native = maker(**native_kwargs(num_envs=num_envs, seed=seed, auto_reset=_NATIVE_MODE[mode], outputs=outputs, **kwargs))
    layout = native.layout()
    if layout["agents"] != 1 and (env_id not in _SLOT0 or layout["initial_agents"] != 1):
        native.close()
        n = layout["initial_agents"]
        raise _native.ArgumentError(f"{env_id} with agents={n} is multi-agent; use the PettingZoo adapter")
    return native, layout


def _bounds(values, shape) -> np.ndarray:
    """A leaf's flat bounds in its shape: one per element, or one for all."""
    arr = np.asarray(values, dtype=np.float64)
    return np.full(shape, arr.item()) if arr.size == 1 else arr.reshape(shape)


def _single_space(layout) -> spaces.Space:
    boxes = {}
    for name, dtype, shape, low, high in layout["leaves"]:
        dt = np.dtype(dtype)
        lo = _bounds(low, shape)
        hi = _bounds(high, shape)
        boxes[name or "obs"] = spaces.Box(low=lo.astype(dt), high=hi.astype(dt), shape=tuple(shape), dtype=dt)
    if list(boxes) == ["obs"]:
        return boxes["obs"]
    return spaces.Dict(boxes)


class TesseraVectorEnv(VectorEnv):
    """``num_envs`` instances of a single-agent Tessera environment.

    Rewards are ``float32`` (the engine's precision), not the ``float64``
    of ``SyncVectorEnv``; terminations and truncations are ``bool``.
    ``construction_seed`` is the pool seed used (drawn from OS entropy when
    ``seed`` is ``None``).
    """

    def __init__(
        self,
        env_id: str = "tessera/CartPole-v1",
        num_envs: int = 1,
        seed: int | None = None,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
        outputs: str = "owned",
        max_episode_steps: int | None = 500,
        **kwargs: Any,
    ):
        mode = _autoreset_mode(autoreset_mode)
        if seed is None:
            seed = secrets.randbits(64)
        #: The pool seed: instance i starts from ``construction_seed + i``.
        self.construction_seed: int = int(seed)
        # The vector env owns its time limit (Gymnasium has no vector
        # TimeLimit): `make_vec` passes the registered 500 or an override;
        # None or -1 means no limit.
        limit = 0 if max_episode_steps is None or max_episode_steps < 0 else int(max_episode_steps)
        kwargs["max_episode_steps"] = limit
        self._native, layout = _native_env(env_id, num_envs, seed, mode, outputs, kwargs)
        self.num_envs = layout["num_envs"]
        self.outputs = outputs
        self.metadata = {"autoreset_mode": mode}
        self.render_mode = None
        self.single_observation_space = _single_space(layout)
        self.single_action_space = spaces.Discrete(layout["action_choices"])
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self._names = [name or "obs" for name, *_ in layout["leaves"]]
        #: Row width of the native outputs (capacity); the adapter uses slot 0.
        self._agents: int = layout["agents"]
        # Slot 0 carries the action; the other slots are never alive or
        # acting, and the engine zeroes them in validation anyway.
        self._actions = np.zeros((self.num_envs, self._agents), dtype="<i4")
        self._shapes = [(np.dtype(dtype), tuple(shape)) for _, dtype, shape, *_ in layout["leaves"]]

    # -- outputs -------------------------------------------------------------

    def _leaf(self, buf, k):
        if self.outputs == "borrowed":
            return np.asarray(buf)[:, 0]
        dtype, shape = self._shapes[k]
        rows = np.frombuffer(buf, dtype=dtype).reshape((self.num_envs, self._agents, *shape))
        return rows[:, 0] if self._agents == 1 else rows[:, 0].copy()

    def _obs(self, bufs):
        leaves = {name: self._leaf(buf, k) for k, (name, buf) in enumerate(zip(self._names, bufs, strict=True))}
        if isinstance(self.single_observation_space, spaces.Dict):
            return leaves
        return leaves["obs"]

    def _flags(self, out, key):
        if self.outputs == "borrowed":
            return np.asarray(out[key])
        return np.frombuffer(out[key], dtype=np.bool_)

    def _info(self, out, done) -> dict:
        info: dict = {}
        if self.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP and done.any():
            final = self._obs(out["terminal_observations"])
            final_obs = np.full(self.num_envs, None, dtype=object)
            for i in np.flatnonzero(done):
                if isinstance(final, dict):
                    final_obs[i] = {k: np.array(v[i]) for k, v in final.items()}
                else:
                    final_obs[i] = np.array(final[i])
            info["final_obs"] = final_obs
            info["_final_obs"] = done.copy()
            info["final_info"] = {}
            info["_final_info"] = done.copy()
        return info

    # -- Gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | list[int | None] | None = None, options: dict | None = None):
        """As ``SyncVectorEnv.reset``: the full seed vector is formed first
        (``None``, ``seed + i``, or one entry per instance, ``None`` keeping
        that instance's seed), then ``reset_mask`` selects the instances to
        reset, with the same validation and error types."""
        options = dict(options or {})
        seeds = _seed_vector(seed, self.num_envs)
        ids = None
        if "reset_mask" in options:
            mask = _check_reset_mask(options.pop("reset_mask"), self.num_envs)
            ids = [int(i) for i in np.flatnonzero(mask)]
            seeds = [seeds[i] for i in ids]
        if options:
            raise _native.ArgumentError(f"unsupported reset options {sorted(options)}")
        native_seed = None if all(s is None for s in seeds) else seeds
        out = self._native.reset(native_seed, ids)
        return self._obs(out["observations"]), {}

    def step(self, actions):
        acts = _discrete_actions(actions)
        if acts.size != self.num_envs:
            raise _native.ActionError(f"expected {self.num_envs} actions, got {acts.size}")
        if self._agents == 1:
            out = self._native.step(acts.tobytes())
        else:
            self._actions[:, 0] = acts.reshape(self.num_envs)
            out = self._native.step(self._actions.tobytes())
        terminated = self._flags(out, "episode_terminated")
        truncated = self._flags(out, "episode_truncated")
        if self.outputs == "borrowed":
            rewards = np.asarray(out["rewards"])[:, 0]
        else:
            rewards = np.frombuffer(out["rewards"], dtype="<f4").reshape(self.num_envs, self._agents)[:, 0]
            if self._agents != 1:
                rewards = rewards.copy()
        info = self._info(out, terminated | truncated)
        return self._obs(out["observations"]), rewards, terminated, truncated, info

    def close_extras(self, **kwargs):
        self._native.close()


class TesseraEnv(gymnasium.Env):
    """One instance as a ``gymnasium.Env`` (``DISABLED`` auto-reset, owned
    outputs): for the Gymnasium env checker and single-env tooling.

    It has no native time limit: ``gymnasium.make`` adds the ``TimeLimit``
    wrapper (registered ``max_episode_steps=500``), so overrides such as
    ``max_episode_steps=600`` or ``-1`` (no wrapper) behave as for
    Gymnasium's own environments."""

    metadata = {"render_modes": []}

    def __init__(self, env_id: str = "tessera/CartPole-v1", seed: int | None = None, **kwargs: Any):
        kwargs.setdefault("max_episode_steps", None)
        self._vec = TesseraVectorEnv(env_id, 1, seed, AutoresetMode.DISABLED, "owned", **kwargs)
        self.construction_seed: int = self._vec.construction_seed
        self.observation_space = self._vec.single_observation_space
        self.action_space = self._vec.single_action_space
        self.render_mode = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise _native.ArgumentError(f"unsupported reset options {sorted(options)}")
        obs, _ = self._vec.reset(seed=seed)
        return _first(obs), {}

    def step(self, action):
        obs, rewards, terminated, truncated, _ = self._vec.step(np.asarray([action]))
        return _first(obs), float(rewards[0]), bool(terminated[0]), bool(truncated[0]), {}

    def close(self):
        self._vec.close()


def _seed_vector(seed, n: int) -> list[int | None]:
    """The per-instance seeds of a reset, as ``SyncVectorEnv`` forms them;
    sums wrap modulo 2^64 (SPEC 9.2)."""
    if seed is None:
        return [None] * n
    if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
        if seed < 0:
            raise ValueError(f"seeds must be non-negative integers, got {seed}")
        return [(int(seed) + i) % 2**64 for i in range(n)]
    seeds = list(seed)
    if len(seeds) != n:
        raise ValueError(
            f"If seeds are passed as a list the length must match num_envs={n} but got length={len(seeds)}."
        )
    for s in seeds:
        if s is not None and (not isinstance(s, (int, np.integer)) or isinstance(s, bool) or s < 0):
            raise ValueError(f"seeds must be non-negative integers or None, got {s!r}")
    return [None if s is None else int(s) for s in seeds]


def _check_reset_mask(mask, n: int) -> np.ndarray:
    """``options["reset_mask"]``, validated exactly as ``SyncVectorEnv``."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"`options['reset_mask']` must be a numpy array, got {type(mask)}")
    if mask.shape != (n,):
        raise ValueError(f"`options['reset_mask']` must have shape `({n},)`, got {mask.shape}")
    if mask.dtype != np.bool_:
        raise TypeError(f"`options['reset_mask']` must have `dtype=np.bool_`, got {mask.dtype}")
    if not np.any(mask):
        raise ValueError(
            "`options['reset_mask']` must contain a boolean array with at least one True value, "
            f"got reset_mask={mask}"
        )
    return mask


def _first(obs):
    if isinstance(obs, dict):
        return {k: np.array(v[0]) for k, v in obs.items()}
    return np.array(obs[0])


def make(env_id: str, num_envs: int = 1, seed: int | None = None, **kwargs: Any) -> TesseraVectorEnv:
    """A :class:`TesseraVectorEnv` of ``num_envs`` instances of ``env_id``.

    ``seed=None`` draws the pool seed from OS entropy (see
    ``construction_seed``). Options: ``autoreset_mode`` (``AutoresetMode`` or its name, default
    ``NEXT_STEP``), ``outputs`` (``"owned"`` by default or ``"borrowed"``),
    and the native pool options (``max_episode_steps``, ``chunk_size``,
    ``workers``).
    """
    return TesseraVectorEnv(env_id, num_envs, seed, **kwargs)


def _register() -> None:
    for env_id in _MAKERS:
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=env_id,
            entry_point="tessera_rl.gym:TesseraEnv",
            vector_entry_point="tessera_rl.gym:TesseraVectorEnv",
            max_episode_steps=_LIMITS.get(env_id, 500),
            kwargs={"env_id": env_id, "agents": 1} if env_id in _SLOT0 else {"env_id": env_id},
        )


_register()
