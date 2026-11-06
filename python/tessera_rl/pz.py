# SPDX-License-Identifier: MIT OR Apache-2.0
"""PettingZoo adapters for the multi-agent environments (ADR-0013).

Each environment declares the PettingZoo interfaces it supports, and a
request for any other one is refused when it is made:

==========================  ===========  ============================
Environment                 Turns        Adapters
==========================  ===========  ============================
``tessera/Tag-v1``          simultaneous ``parallel_env``, ``aec_env``
``tessera/GridKeyDoor-v1``  simultaneous ``parallel_env``, ``aec_env``
``tessera/Snake-v1``        simultaneous ``parallel_env``, ``aec_env``
``tessera/Connect4-v1``     sequential   ``aec_env``
``tessera/Leduc-v1``        sequential   ``aec_env``
``tessera/Othello-v1``      sequential   ``aec_env``
``tessera/Go9-v1``          sequential   ``aec_env``
==========================  ===========  ============================

- A simultaneous environment is Parallel natively. Its AEC form buffers
  each agent's action and steps the environment once per cycle, when every
  live agent has acted; observations change only at cycle boundaries
  (PettingZoo's ``parallel_to_aec``).
- A sequential environment is AEC only, with the acting agent as
  ``agent_selection``. It is never presented as Parallel: that would
  change what each agent knows when it acts.

Agents are named ``agent_<slot>``. ``possible_agents`` are the slots
alive at reset (no shipped environment spawns agents later). An agent
leaves ``agents`` after the step in which it is terminated or truncated:
its final reward and flag are reported once, in that step, and at the end
of an episode every agent still alive gets its final reward and flag in
the same step (ADR-0013).

Observations are dicts of the environment's leaves plus ``action_mask``
(``int8``, one entry per action). Outputs are always owned copies, so
they stay valid across steps. One adapter runs one instance with
auto-reset disabled: call ``reset`` to start a new episode.

Actions are checked before anything changes: each must be an exact
integer scalar in ``[0, n)`` (integer types, or floats with no fractional
part); anything else, including ``None`` for a live agent in AEC, raises
``ActionError`` and leaves the environment, and in AEC the buffered
cycle, as it was. In Parallel, a name outside ``possible_agents`` raises
``ActionError``, and actions for agents already known to be dead are
ignored. A legal
index that the mask marks illegal, such as a move into a full Connect Four
column, is replaced by the environment's no-op (ADR-0028); it is not an
error.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from tessera_rl import _native
from tessera_rl._env import Env, _discrete_actions, native_kwargs

#: Native constructors and the PettingZoo interfaces each environment
#: declares (ADR-0013). Anything else is refused at construction.
_ENVS = {
    "tessera/Tag-v1": (_native.make_tag, frozenset({"parallel", "aec"})),
    "tessera/GridKeyDoor-v1": (_native.make_keydoor, frozenset({"parallel", "aec"})),
    "tessera/Snake-v1": (_native.make_snake, frozenset({"parallel", "aec"})),
    "tessera/Connect4-v1": (_native.make_connect4, frozenset({"aec"})),
    "tessera/Leduc-v1": (_native.make_leduc, frozenset({"aec"})),
    "tessera/Othello-v1": (_native.make_othello, frozenset({"aec"})),
    "tessera/Go9-v1": (_native.make_go9, frozenset({"aec"})),
}


def _native_pool(env_id: str, adapter: str, seed: int | None, kwargs: dict) -> tuple[Env, list[int]]:
    """One instance of ``env_id`` for ``adapter``, refused if undeclared,
    and the slots alive at reset. The slots are read from a probe pool
    that is then closed, so the returned pool has not started an episode
    and its first ``reset`` is the seed's first episode."""
    try:
        maker, declared = _ENVS[env_id]
    except KeyError:
        raise _native.ArgumentError(f"no PettingZoo adapter for {env_id!r}; known: {sorted(_ENVS)}") from None
    if adapter not in declared:
        raise _native.ArgumentError(
            f"{env_id} declares {sorted(declared)}, not {adapter!r}: undeclared conversions are refused (ADR-0013)"
        )
    for reserved in ("num_envs", "outputs", "auto_reset", "workers"):
        if reserved in kwargs:
            raise _native.ArgumentError(f"{reserved!r} is fixed by the PettingZoo adapter")
    options = native_kwargs(
        num_envs=1, seed=0 if seed is None else seed, outputs="owned", auto_reset="disabled", **kwargs
    )
    probe = Env(maker(**options))
    try:
        slots = [int(i) for i in np.flatnonzero(probe.reset().alive[0])]
    finally:
        probe.close()
    return Env(maker(**options)), slots


def _exact_action(value, n: int) -> int:
    """``value`` as an exact action index in ``[0, n)``; ``ActionError``
    otherwise (the native contract: no truncation, no rounding). An action
    is one scalar: lists, arrays and ``None`` are refused."""
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as e:
        raise _native.ActionError(f"an action is one integer: {e}") from None
    if arr.ndim != 0:
        raise _native.ActionError(f"an action is one integer, not shape {arr.shape}")
    action = int(_discrete_actions(arr.reshape(1))[0])
    if not 0 <= action < n:
        raise _native.ActionError(f"action {action} is outside [0, {n})")
    return action


class _Agents:
    """Spaces, names and per-agent views of a one-instance pool."""

    def __init__(self, pool: tuple[Env, list[int]], env_id: str):
        env, self.slots = pool
        self.env = env
        self.env_id = env_id
        self.names = [f"agent_{i}" for i in self.slots]
        self.slot_of = dict(zip(self.names, self.slots, strict=True))
        layout = env._native.layout()
        leaves = {}
        for name, dtype, shape, low, high in layout["leaves"]:
            dt = np.dtype(dtype)
            # Bounds come per element, or as one value for the whole leaf.
            lo = np.broadcast_to(np.asarray(low, dtype=np.float64).ravel(), (int(np.prod(shape)),))
            hi = np.broadcast_to(np.asarray(high, dtype=np.float64).ravel(), (int(np.prod(shape)),))
            lo, hi = lo.reshape(shape).astype(dt), hi.reshape(shape).astype(dt)
            leaves[name or "obs"] = spaces.Box(low=lo, high=hi, shape=tuple(shape), dtype=dt)
        n = env.action_choices
        leaves["action_mask"] = spaces.Box(low=0, high=1, shape=(n,), dtype=np.int8)
        self.observation_space = spaces.Dict(leaves)
        self.action_space = spaces.Discrete(n)
        self.last = None

    def observe(self, slot: int) -> dict:
        step = self.last
        if step is None:
            raise _native.ArgumentError("call reset() before observing")
        obs = {name: np.array(arr[0, slot]) for name, arr in step.observations.items()}
        if step.masks is not None:
            obs["action_mask"] = step.masks[0, slot].astype(np.int8)
        else:
            obs["action_mask"] = np.ones(self.env.action_choices, dtype=np.int8)
        return obs

    def flags(self, slot: int) -> tuple[float, bool, bool]:
        step = self.last
        terminated = bool(step.agent_terminated[0, slot] or step.episode_terminated[0])
        truncated = bool(step.agent_truncated[0, slot] or step.episode_truncated[0]) and not terminated
        return float(step.rewards[0, slot]), terminated, truncated


class TesseraParallelEnv(ParallelEnv):
    """A simultaneous Tessera environment as a PettingZoo ``ParallelEnv``."""

    metadata = {"render_modes": [], "name": "tessera", "is_parallelizable": True}

    def __init__(self, env_id: str, seed: int | None = None, **kwargs):
        self._a = _Agents(_native_pool(env_id, "parallel", seed, kwargs), env_id)
        self.metadata = {**self.metadata, "name": env_id}
        self.render_mode = None
        self.possible_agents = list(self._a.names)
        self.agents = list(self.possible_agents)

    def observation_space(self, agent):
        return self._a.observation_space

    def action_space(self, agent):
        return self._a.action_space

    def reset(self, seed=None, options=None):
        self._a.last = self._a.env.reset(seed)
        self.agents = [n for n, s in zip(self._a.names, self._a.slots, strict=True) if self._a.last.alive[0, s]]
        return {n: self._a.observe(self._a.slot_of[n]) for n in self.agents}, {n: {} for n in self.agents}

    def step(self, actions):
        # Everything is checked before the pool is touched. Actions for
        # agents that are known but no longer live are ignored, so the usual
        # `{a: policy(obs[a]) for a in obs}` loop works across a death.
        unknown = [n for n in actions if n not in self.possible_agents]
        if unknown:
            raise _native.ActionError(f"actions for unknown agents: {unknown}")
        missing = [n for n in self.agents if n not in actions]
        if missing:
            raise _native.ActionError(f"no action for live agents {missing}")
        n = self._a.action_space.n
        checked = {name: _exact_action(actions[name], n) for name in self.agents}
        acts = np.zeros((1, self._a.env.agents), dtype=np.int32)
        for name, action in checked.items():
            acts[0, self._a.slot_of[name]] = action
        self._a.last = self._a.env.step(acts)
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}
        for name in self.agents:
            slot = self._a.slot_of[name]
            obs[name] = self._a.observe(slot)
            rewards[name], terms[name], truncs[name] = self._a.flags(slot)
            infos[name] = {}
        self.agents = [n for n in self.agents if not (terms[n] or truncs[n])]
        return obs, rewards, terms, truncs, infos

    def close(self):
        self._a.env.close()


class TesseraAECEnv(AECEnv):
    """A sequential Tessera environment as a PettingZoo ``AECEnv``: the
    acting agent is ``agent_selection``, and each ``step`` is one engine
    step with that agent's action."""

    metadata = {"render_modes": [], "name": "tessera", "is_parallelizable": False}

    def __init__(self, env_id: str, seed: int | None = None, **kwargs):
        super().__init__()
        self._a = _Agents(_native_pool(env_id, "aec", seed, kwargs), env_id)
        if self._a.env.turns != "sequential":
            raise _native.ArgumentError(f"{env_id} is {self._a.env.turns}; TesseraAECEnv is for sequential turns")
        self.metadata = {**self.metadata, "name": env_id}
        self.render_mode = None
        self.possible_agents = list(self._a.names)
        # No reset here: the first reset() is the seed's first episode.
        self.agents = list(self.possible_agents)
        self.rewards = {n: 0.0 for n in self.agents}
        self._cumulative_rewards = {n: 0.0 for n in self.agents}
        self.terminations = {n: False for n in self.agents}
        self.truncations = {n: False for n in self.agents}
        self.infos = {n: {} for n in self.agents}
        self.agent_selection = self.agents[0]

    def observation_space(self, agent):
        return self._a.observation_space

    def action_space(self, agent):
        return self._a.action_space

    def observe(self, agent):
        return self._a.observe(self._a.slot_of[agent])

    def _acting(self) -> str | None:
        for name, slot in zip(self._a.names, self._a.slots, strict=True):
            if name in self.agents and self._a.last.acting[0, slot]:
                return name
        return None

    def _start(self):
        self.agents = [n for n, s in zip(self._a.names, self._a.slots, strict=True) if self._a.last.alive[0, s]]
        self.rewards = {n: 0.0 for n in self.agents}
        self._cumulative_rewards = {n: 0.0 for n in self.agents}
        self.terminations = {n: False for n in self.agents}
        self.truncations = {n: False for n in self.agents}
        self.infos = {n: {} for n in self.agents}
        self.agent_selection = self._acting() or self.agents[0]

    def reset(self, seed=None, options=None):
        self._a.last = self._a.env.reset(seed)
        self._start()

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        checked = _exact_action(action, self._a.action_space.n)
        self._cumulative_rewards[agent] = 0.0
        acts = np.zeros((1, self._a.env.agents), dtype=np.int32)
        acts[0, self._a.slot_of[agent]] = checked
        self._a.last = self._a.env.step(acts)
        for name in self.agents:
            reward, terminated, truncated = self._a.flags(self._a.slot_of[name])
            self.rewards[name] = reward
            self.terminations[name] = self.terminations[name] or terminated
            self.truncations[name] = self.truncations[name] or truncated
        nxt = self._acting()
        if nxt is not None:
            self.agent_selection = nxt
        self._accumulate_rewards()
        self._deads_step_first()

    def close(self):
        self._a.env.close()


def parallel_env(env_id: str, seed: int | None = None, **kwargs) -> TesseraParallelEnv:
    """``env_id`` as a PettingZoo ``ParallelEnv``; refused unless the
    environment declares it. ``kwargs`` are the environment's config and
    pool options (``max_episode_steps``, ``chunk_size``)."""
    return TesseraParallelEnv(env_id, seed=seed, **kwargs)


def aec_env(env_id: str, seed: int | None = None, **kwargs) -> AECEnv:
    """``env_id`` as a PettingZoo ``AECEnv``: native for a sequential
    environment, cycle buffering over the Parallel form for a simultaneous
    one; refused unless the environment declares it."""
    try:
        declared = _ENVS[env_id][1]
    except KeyError:
        raise _native.ArgumentError(f"no PettingZoo adapter for {env_id!r}; known: {sorted(_ENVS)}") from None
    if "aec" not in declared:
        raise _native.ArgumentError(f"{env_id} declares {sorted(declared)}, not 'aec' (ADR-0013)")
    if "parallel" in declared:
        return OrderEnforcingWrapper(_CheckedCycle(TesseraParallelEnv(env_id, seed=seed, **kwargs)))
    return TesseraAECEnv(env_id, seed=seed, **kwargs)


class _CheckedCycle(parallel_to_aec_wrapper):
    """Cycle buffering (PettingZoo's ``parallel_to_aec``) that checks each
    action when it is buffered, so an invalid one leaves the buffered
    cycle and the environment as they were."""

    def step(self, action):
        agent = self.agent_selection
        # A live agent's action is always checked, so None raises here.
        if not (self.terminations[agent] or self.truncations[agent]):
            action = _exact_action(action, self.action_space(agent).n)
        super().step(action)


def turn_masked_env(env_id: str, seed: int | None = None, **kwargs):
    """The optional turn-masked interface of ADR-0013: no shipped
    environment declares it, so every request is refused."""
    raise _native.ArgumentError(f"{env_id} does not declare 'turn_masked' (ADR-0013)")
