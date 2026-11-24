# SPDX-License-Identifier: MIT OR Apache-2.0
"""PettingZoo adapters for the multi-agent environments (ADR-0013).

Each environment declares the PettingZoo interfaces it supports, and a
request for any other one is refused when it is made:

=============================  ============  =============================
Environment                    Turns         Adapters
=============================  ============  =============================
``tessera/Tag-v1``             simultaneous  ``parallel_env``, ``aec_env``
``tessera/GridKeyDoor-v1``     simultaneous  ``parallel_env``, ``aec_env``
``tessera/Gather-v1``          simultaneous  ``parallel_env``, ``aec_env``
``tessera/Snake-v1``           simultaneous  ``parallel_env``, ``aec_env``
``tessera/PaddleDuel-v1``      simultaneous  ``parallel_env``, ``aec_env``
``tessera/RTS-v1``             simultaneous  ``parallel_env``
``tessera/LimitOrderBook-v1``  simultaneous  ``parallel_env``
``tessera/Connect4-v1``        sequential    ``aec_env``
``tessera/Leduc-v1``           sequential    ``aec_env``
``tessera/Lanterns-v1``        sequential    ``aec_env``
``tessera/Othello-v1``         sequential    ``aec_env``
``tessera/Go9-v1``             sequential    ``aec_env``
=============================  ============  =============================

- A simultaneous environment is Parallel natively. Its AEC form buffers
  each agent's action and steps the environment once per cycle, when every
  live agent has acted; observations change only at cycle boundaries
  (PettingZoo's ``parallel_to_aec``).
- A sequential environment is AEC only, with the acting agent as
  ``agent_selection``. It is never presented as Parallel: that would
  change what each agent knows when it acts.

An agent is one occupant of a slot: identity is (slot, generation)
(ADR-0013). Generation 0 is named ``agent_<slot>``, and a later
generation ``agent_<slot>_g<generation>``: PettingZoo forbids a name to
come back once its agent is done. ``possible_agents`` are the agents
alive at reset, and grow when an environment that spawns agents (a host
environment with ``spawns``, ADR-0065) fills a slot. An agent leaves
``agents`` after the step in which it is terminated or truncated: its
final reward and flag are reported once, in that step, and at the end of
an episode every agent still alive gets its final reward and flag in the
same step (ADR-0013). A freed slot is reused no earlier than the next
step; its new agent joins ``agents`` in the step it spawns, with its first
observation and info and the engine's reward and flags for that step (a
spawn in the episode's last step is flushed with the others and does not
stay in ``agents``). Native AEC refuses a spawning environment
(``CapabilityError``).

Observations are dicts of the environment's leaves plus ``action_mask``
(``int8``, one entry per action).

An environment with several discrete heads (``MultiDiscrete``, the RTS:
three heads per unit slot) has ``MultiDiscrete(nvec)`` as its action
space, and an action is one integer per head. Its heads are conditional,
which a Gymnasium ``MultiDiscrete`` mask (one independent mask per
component) cannot express, so its observations carry no ``action_mask``.
The masks are in ``infos[agent]``, and nothing else is (SPEC 11.3):

- ``"action_masks"`` holds the exact masks: every head's mask, one array per
  head of an entity group in the shapes of ``tessera_rl.masks``
  (``[units, types]``, ``[units, types, window]``, ``[units, types, products]``);
- ``"action_mask"`` is a tuple of ``int8`` masks, one per ``MultiDiscrete``
  component in ``nvec`` order (#311), which ``action_space.sample(mask)``
  accepts. A root head's entry is its exact mask, so a sampled root choice
  is legal. A conditional head's entry is the union over all its parent
  choices, legal or not: choice ``c`` is marked legal if the mask allows it
  under any parent choice. A sampled child is therefore often illegal for
  the parent actually sampled (for the RTS most sampled joint actions are);
  the tuple makes ``sample`` run and keeps root heads legal, and legal
  sampling of conditional heads needs ``"action_masks"``.

The Gymnasium adapter (``tessera_rl.gym``) keeps its own format:
its ``info["action_mask"]`` is the root head's array, not this tuple. An order other than the no-op on an empty or dead unit slot
counts as an illegal action: give empty slots zeros. Outputs are always owned copies, so
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

from tessera_rl import _envs, _host, _native, _render
from tessera_rl._env import Env, _discrete_actions, _multi_actions, native_kwargs
from tessera_rl.masks import head_views

#: Native constructors and the PettingZoo interfaces each environment
#: declares (ADR-0013). Anything else is refused at construction.
_MAKERS = {
    "tessera/Tag-v1": _native.make_tag,
    "tessera/GridKeyDoor-v1": _native.make_keydoor,
    "tessera/Gather-v1": _native.make_gather,
    "tessera/Snake-v1": _native.make_snake,
    "tessera/PaddleDuel-v1": _native.make_paddleduel,
    "tessera/Connect4-v1": _native.make_connect4,
    "tessera/Leduc-v1": _native.make_leduc,
    "tessera/Lanterns-v1": _native.make_lanterns,
    "tessera/Othello-v1": _native.make_othello,
    "tessera/Go9-v1": _native.make_go9,
    # Simultaneous; it declares Parallel only (cycle buffering of a
    # multi-head environment is not designed yet).
    "tessera/RTS-v1": _native.make_rts,
    # Simultaneous, multi-head like the RTS: Parallel only, for the same
    # reason.
    "tessera/LimitOrderBook-v1": _native.make_lob,
}

#: The maker of each first-party id and the PettingZoo interfaces it
#: declares, read from the registry (``Environment::ADAPTERS``, SPEC 12,
#: ADR-0080).
_ENVS = {env_id: (maker, _envs.adapters_of(env_id)) for env_id, maker in _MAKERS.items()}


#: The PettingZoo interfaces of a host environment, by its turns
#: (ADR-0013; ADR-0065 item 6: Mixed through ``parallel_env`` only).
_HOST_DECLARED = {
    "simultaneous": frozenset({"parallel", "aec"}),
    "sequential": frozenset({"aec"}),
    "mixed": frozenset({"parallel"}),
}


def _undeclared(env_id: str, adapter: str, declared, turns=None) -> Exception:
    """The error for ``adapter`` when ``env_id`` does not declare it
    (ADR-0013); a Mixed host environment as AEC is a ``CapabilityError``
    (ADR-0065 item 6)."""
    if turns == "mixed":
        return _native.CapabilityError(f"{env_id} is a Mixed host environment: use parallel_env (ADR-0065)")
    return _native.ArgumentError(
        f"{env_id} declares {sorted(declared)}, not {adapter!r}: undeclared conversions are refused (ADR-0013)"
    )


def _check_reserved(kwargs: dict) -> None:
    for reserved in ("num_envs", "outputs", "auto_reset", "workers"):
        if reserved in kwargs:
            raise _native.ArgumentError(f"{reserved!r} is fixed by the PettingZoo adapter")


def _host_pool(env_id: str, adapter: str, seed: int | None, kwargs: dict):
    """One instance of the host environment ``env_id`` for ``adapter``
    (SPEC 14.4), built once with the caller's config, the slots alive at
    reset and the interfaces it declares, read from its turns. No probe:
    the slots alive after a reset are ``0..initial_agents`` (ADR-0065
    item 4), so the returned pool has not started an episode."""
    _check_reserved(kwargs)
    native = _host.native_for(
        env_id, 1, 0 if seed is None else seed, "owned", {**kwargs, "auto_reset": "disabled"}
    )
    try:
        layout = native.layout()
        declared = _HOST_DECLARED.get(layout["turns"], frozenset())
        if adapter not in declared:
            raise _undeclared(env_id, adapter, declared, layout["turns"])
        if layout["action_bounds"] is not None:
            raise _native.ArgumentError(f"{env_id} has continuous actions; the PettingZoo adapter is discrete only")
        env = Env(native)
    except BaseException:
        native.close()
        raise
    # The agents alive at reset are slots 0..initial_agents; a spawning
    # environment adds agents as they spawn (_Agents.live, ADR-0065).
    return (env, list(range(layout["initial_agents"]))), declared


def _lookup(env_id: str) -> tuple:
    """The maker of the first-party ``env_id`` and the interfaces it
    declares (ADR-0013)."""
    if env_id in _ENVS:
        return _ENVS[env_id]
    known = sorted({*_ENVS, *_host.host_ids()})
    raise _native.ArgumentError(f"no PettingZoo adapter for {env_id!r}; known: {known}")


def _native_pool(env_id: str, adapter: str, seed: int | None, kwargs: dict) -> tuple[Env, list[int]]:
    """One instance of ``env_id`` for ``adapter``, refused if undeclared,
    and the slots alive at reset. For a first-party environment the slots
    are read from a probe pool that is then closed, so the returned pool
    has not started an episode and its first ``reset`` is the seed's
    first episode."""
    if env_id not in _ENVS and _host.is_host_id(env_id):
        return _host_pool(env_id, adapter, seed, kwargs)[0]
    maker, declared = _lookup(env_id)
    _envs.warn_if_deprecated(env_id)
    if adapter not in declared:
        raise _undeclared(env_id, adapter, declared)
    _check_reserved(kwargs)
    options = native_kwargs(
        num_envs=1, seed=0 if seed is None else seed, outputs="owned", auto_reset="disabled", **kwargs
    )
    probe = Env(maker(**options))
    try:
        if probe.action_bounds is not None:
            raise _native.ArgumentError(f"{env_id} has continuous actions; the PettingZoo adapter is discrete only")
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


def relaxed_masks(row, mask_heads) -> tuple[np.ndarray, ...]:
    """One ``int8`` mask per discrete head, in head order (``nvec`` order),
    for ``MultiDiscrete.sample`` (#311). ``row`` is one agent's mask row;
    ``mask_heads`` are ``(offset, parents, size)`` per head
    (``Env.mask_heads``). A head with parents is relaxed to the union over
    them all (legal or not): choice ``c`` is legal if any parent choice
    allows it, so a combination sampled from these masks can still be
    illegal."""
    row = np.asarray(row)
    out = []
    for offset, parents, size in mask_heads:
        width = int(np.prod(parents, dtype=np.int64)) * size
        block = row[offset : offset + width].reshape(*parents, size)
        legal = block.reshape(-1, size).any(axis=0)
        out.append(legal.astype(np.int8))
    return tuple(out)


def agent_name(slot: int, generation: int) -> str:
    """The PettingZoo name of the agent (slot, generation) (ADR-0013)."""
    return f"agent_{slot}" if generation == 0 else f"agent_{slot}_g{generation}"


class _Agents:
    """Spaces, names and per-agent views of a one-instance pool."""

    def __init__(self, pool: tuple[Env, list[int]], env_id: str):
        env, self.slots = pool
        self.env = env
        self.env_id = env_id
        self.names = [agent_name(i, 0) for i in self.slots]
        self.slot_of = dict(zip(self.names, self.slots, strict=True))
        layout = env._native.layout()
        #: Whether the environment fills empty slots (ADR-0065 ``spawns``).
        self.spawns = bool(layout.get("spawns", False))
        #: ``"host"`` for a third-party host environment, else ``"tessera"``.
        self.kind = "host" if layout.get("kind") == "host" else "tessera"
        #: Whether the host environment refuses agents (ADR-0070): every
        #: agent's infos then carry ``refused`` and ``refuse_code``.
        self.refuses = bool(layout.get("refuses_agents", False))
        leaves = {}
        # Set masks (ADR-0082 point 6): the last leaf of each set, exposed as
        # ``MultiBinary`` with ``int8`` 0/1 values, as ``action_mask``.
        mask_ids = {indices[-1] for _, _, indices in layout.get("sets", ())}
        #: Names of the set mask leaves.
        self.set_masks = frozenset(layout["leaves"][k][0] for k in mask_ids)
        for name, dtype, shape, low, high in layout["leaves"]:
            if name in self.set_masks:
                leaves[name] = spaces.MultiBinary(int(shape[0]))
                continue
            dt = np.dtype(dtype)
            # Bounds come per element, or as one value for the whole leaf.
            lo = np.broadcast_to(np.asarray(low, dtype=np.float64).ravel(), (int(np.prod(shape)),))
            hi = np.broadcast_to(np.asarray(high, dtype=np.float64).ravel(), (int(np.prod(shape)),))
            lo, hi = lo.reshape(shape).astype(dt), hi.reshape(shape).astype(dt)
            leaves[name or "obs"] = spaces.Box(low=lo, high=hi, shape=tuple(shape), dtype=dt)
        self.nvec = tuple(env.action_nvec)
        if len(self.nvec) > 1:
            # The conditional masks are what an agent of a multi-head
            # environment acts on: one without masks is not supported.
            if not env.mask_heads:
                raise _native.ArgumentError(f"{env_id} has several action heads and no masks; not supported")
            self.action_space = spaces.MultiDiscrete(np.asarray(self.nvec, dtype=np.int64))
        else:
            n = env.action_choices
            leaves["action_mask"] = spaces.Box(low=0, high=1, shape=(n,), dtype=np.int8)
            self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict(leaves)
        self.last = None

    @property
    def multi(self) -> bool:
        return len(self.nvec) > 1

    def head_masks(self, slot: int) -> tuple[np.ndarray, ...]:
        """Every head's mask of agent ``slot``, owned copies (multi-head)."""
        m = self.last.masks[:, slot : slot + 1]
        return tuple(np.array(v[0, 0], dtype=np.int8) for v in head_views(m, self.env.mask_heads, self.env.entity_group_heads))

    def info(self, slot: int) -> dict:
        """``infos[agent]``: that agent's masks for a multi-head environment
        (R2: nothing else), else empty. ``action_mask`` is the relaxed tuple
        (#311), ``action_masks`` the exact masks."""
        info = {}
        if self.multi:
            heads = self.head_masks(slot)
            relaxed = relaxed_masks(self.last.masks[0, slot], self.env.mask_heads)
            info = {"action_mask": relaxed, "action_masks": heads}
        if self.refuses:
            info.update(self.refusal(slot))
        return info

    def refusal(self, slot: int) -> dict:
        """``refused`` and ``refuse_code`` of agent ``slot`` in the last
        call (ADR-0070); all clear after a reset."""
        out = getattr(self.env, "_last_out", None)
        if out is None or "refuse_code" not in out:
            return {"refused": False, "refuse_code": 0}
        code = int(np.frombuffer(out["refuse_code"], dtype="<u4")[slot])
        return {"refused": code != 0, "refuse_code": code}

    def check(self, value):
        """One agent's action, checked: an index, or one per head."""
        if not self.multi:
            return _exact_action(value, self.action_space.n)
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError) as e:
            raise _native.ActionError(f"an action is one integer per head: {e}") from None
        if arr.shape != (len(self.nvec),):
            raise _native.ActionError(f"an action has shape ({len(self.nvec)},), not {arr.shape}")
        return _multi_actions(arr.reshape(1, 1, -1), 1, self.nvec)[0, 0]

    def observe(self, slot: int) -> dict:
        step = self.last
        if step is None:
            raise _native.ArgumentError("call reset() before observing")
        obs = {
            name: np.array(arr[0, slot], dtype=np.int8) if name in self.set_masks else np.array(arr[0, slot])
            for name, arr in step.observations.items()
        }
        # A multi-head environment's conditional masks are in infos (see
        # the module doc).
        if not self.multi:
            if step.masks is not None:
                obs["action_mask"] = step.masks[0, slot].astype(np.int8)
            else:
                obs["action_mask"] = np.ones(self.env.action_choices, dtype=np.int8)
        return obs

    def pending(self) -> bool:
        """Whether the instance waits for a reset: a host environment's
        ``pending_reset`` output (ADR-0065); never for a first-party one."""
        out = getattr(self.env, "_last_out", None)
        if out is None or "pending_reset" not in out:
            return False
        return bool(np.frombuffer(out["pending_reset"], dtype=np.bool_)[0])

    def live(self) -> list[str]:
        """The names of the agents alive in the last outputs, in slot
        order, registered in ``slot_of``."""
        step = self.last
        names = []
        for slot in np.flatnonzero(step.alive[0]):
            name = agent_name(int(slot), int(step.slot_generations[0, slot]))
            self.slot_of[name] = int(slot)
            names.append(name)
        return names

    def flags(self, slot: int) -> tuple[float, bool, bool]:
        step = self.last
        terminated = bool(step.agent_terminated[0, slot] or step.episode_terminated[0])
        truncated = bool(step.agent_truncated[0, slot] or step.episode_truncated[0]) and not terminated
        return float(step.rewards[0, slot]), terminated, truncated


class TesseraParallelEnv(ParallelEnv):
    """A simultaneous Tessera environment as a PettingZoo ``ParallelEnv``."""

    metadata = {"render_modes": [], "name": "tessera", "is_parallelizable": True}

    def __init__(self, env_id: str, seed: int | None = None, *, _pool=None, render_mode: str | None = None, **kwargs):
        pool = _pool if _pool is not None else _native_pool(env_id, "parallel", seed, kwargs)
        self._a = _Agents(pool, env_id)
        self.metadata = {
            **self.metadata,
            "name": env_id,
            "tessera_kind": self._a.kind,
            "refuses_agents": self._a.refuses,
        }
        renders = self._a.env.canvas is not None
        self.metadata["render_modes"] = ["rgb_array"] if renders else []
        self.metadata["render_fps"] = _render.RENDER_FPS
        try:
            self.render_mode = _render.check_mode(render_mode, renders, env_id)
        except Exception:
            self._a.env.close()
            raise
        self.possible_agents = list(self._a.names)
        self.agents = list(self.possible_agents)

    def observation_space(self, agent):
        return self._a.observation_space

    def action_space(self, agent):
        return self._a.action_space

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Archive resets (ADR-0075): forwarded to the wrapped
        ``tessera_rl`` environment's ``set_reset_source``."""
        self._a.env.set_reset_source(archive, weights)

    def reset_from(self, ids, entries):
        """Archive resets (ADR-0075): forwarded to the wrapped
        ``tessera_rl`` environment's ``reset_from``."""
        return self._a.env.reset_from(ids, entries)

    def reset(self, seed=None, options=None):
        self._a.last = self._a.env.reset(seed)
        self.agents = self._a.live()
        if self._a.spawns:
            # A slot's generation survives a reset, so its name does too.
            self.possible_agents = list(self.agents)
            self._a.slot_of = {name: self._a.slot_of[name] for name in self.agents}
        if self._a.pending():
            # An aborted reset leaves the instance pending (ADR-0065): no
            # agent can act, so `while env.agents:` ends at once.
            self.agents = []
        obs = {n: self._a.observe(self._a.slot_of[n]) for n in self.agents}
        return obs, {n: self._a.info(self._a.slot_of[n]) for n in self.agents}

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
        checked = {name: self._a.check(actions[name]) for name in self.agents}
        heads = (len(self._a.nvec),) if self._a.multi else ()
        acts = np.zeros((1, self._a.env.agents, *heads), dtype=np.int32)
        for name, action in checked.items():
            acts[0, self._a.slot_of[name]] = action
        self._a.last = self._a.env.step(acts)
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}
        for name in self.agents:
            slot = self._a.slot_of[name]
            obs[name] = self._a.observe(slot)
            rewards[name], terms[name], truncs[name] = self._a.flags(slot)
            infos[name] = self._a.info(slot)
        self.agents = [n for n in self.agents if not (terms[n] or truncs[n])]
        if self._a.spawns:
            # Agents that spawned this step: alive now, not alive before.
            for name in self._a.live():
                if name in self.agents or name in obs:
                    continue
                slot = self._a.slot_of[name]
                obs[name], infos[name] = self._a.observe(slot), self._a.info(slot)
                # The engine's reward and flags for the spawn step, as for
                # every agent: a spawn in the episode's last step is flushed
                # with the others (SPEC 12) and never joins `agents`.
                rewards[name], terms[name], truncs[name] = self._a.flags(slot)
                if not (terms[name] or truncs[name]):
                    self.agents.append(name)
                if name not in self.possible_agents:
                    self.possible_agents.append(name)
        return obs, rewards, terms, truncs, infos

    def render(self):
        """With ``render_mode="rgb_array"``, the frame, ``(height, width, 3)``
        ``uint8``, drawn from a snapshot (ADR-0083); ``None`` otherwise."""
        if self.render_mode is None:
            return None
        return self._a.env.render([0])[0]

    def close(self):
        self._a.env.close()


class TesseraAECEnv(AECEnv):
    """A sequential Tessera environment as a PettingZoo ``AECEnv``: the
    acting agent is ``agent_selection``, and each ``step`` is one engine
    step with that agent's action."""

    metadata = {"render_modes": [], "name": "tessera", "is_parallelizable": False}

    def __init__(self, env_id: str, seed: int | None = None, *, _pool=None, render_mode: str | None = None, **kwargs):
        super().__init__()
        pool = _pool if _pool is not None else _native_pool(env_id, "aec", seed, kwargs)
        self._a = _Agents(pool, env_id)
        if self._a.spawns:
            # Native AEC does not name or schedule spawned agents yet
            # (ADR-0067 follow-up); Simultaneous spawning environments use
            # the Parallel form and cycle buffering.
            self._a.env.close()
            raise _native.CapabilityError(f"{env_id} spawns agents; native AEC does not support spawning yet")
        if self._a.env.turns != "sequential":
            raise _native.ArgumentError(f"{env_id} is {self._a.env.turns}; TesseraAECEnv is for sequential turns")
        self.metadata = {
            **self.metadata,
            "name": env_id,
            "tessera_kind": self._a.kind,
            "refuses_agents": self._a.refuses,
        }
        renders = self._a.env.canvas is not None
        self.metadata["render_modes"] = ["rgb_array"] if renders else []
        self.metadata["render_fps"] = _render.RENDER_FPS
        try:
            self.render_mode = _render.check_mode(render_mode, renders, env_id)
        except Exception:
            self._a.env.close()
            raise
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

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Archive resets (ADR-0075): forwarded to the wrapped
        ``tessera_rl`` environment's ``set_reset_source``."""
        self._a.env.set_reset_source(archive, weights)

    def reset_from(self, ids, entries):
        """Archive resets (ADR-0075): forwarded to the wrapped
        ``tessera_rl`` environment's ``reset_from``."""
        return self._a.env.reset_from(ids, entries)

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
            if self._a.refuses:
                # The refused agent sees its refusal in its own infos; the
                # next agent_selection follows `acting` (ADR-0070 item 10).
                self.infos[name] = {**self.infos.get(name, {}), **self._a.refusal(self._a.slot_of[name])}
        nxt = self._acting()
        if nxt is not None:
            self.agent_selection = nxt
        self._accumulate_rewards()
        self._deads_step_first()

    def render(self):
        """With ``render_mode="rgb_array"``, the frame, ``(height, width, 3)``
        ``uint8``, drawn from a snapshot (ADR-0083); ``None`` otherwise."""
        if self.render_mode is None:
            return None
        return self._a.env.render([0])[0]

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
    if env_id not in _ENVS and _host.is_host_id(env_id):
        # One host object, built with the caller's config, serves both
        # the check and the environment.
        pool, declared = _host_pool(env_id, "aec", seed, kwargs)
        if "parallel" in declared:
            return OrderEnforcingWrapper(_CheckedCycle(TesseraParallelEnv(env_id, seed=seed, _pool=pool)))
        return TesseraAECEnv(env_id, seed=seed, _pool=pool)
    declared = _lookup(env_id)[1]
    if "aec" not in declared:
        raise _native.CapabilityError(f"{env_id} declares {sorted(declared)}, not 'aec' (ADR-0013)")
    if "parallel" in declared:
        return OrderEnforcingWrapper(_CheckedCycle(TesseraParallelEnv(env_id, seed=seed, **kwargs)))
    return TesseraAECEnv(env_id, seed=seed, **kwargs)


class _CheckedCycle(parallel_to_aec_wrapper):
    """Cycle buffering (PettingZoo's ``parallel_to_aec``) that checks each
    action when it is buffered, so an invalid one leaves the buffered
    cycle and the environment as they were. ``possible_agents`` is the
    Parallel environment's own list, which grows as agents spawn (the
    wrapper would otherwise keep the copy it made at construction)."""

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @possible_agents.setter
    def possible_agents(self, value):
        # parallel_to_aec_wrapper.__init__ assigns it; the property follows
        # the Parallel environment instead.
        pass

    def step(self, action):
        agent = self.agent_selection
        # A live agent's action is always checked, so None raises here.
        if not (self.terminations[agent] or self.truncations[agent]):
            # The Parallel environment's own check: one index, or one per
            # head for a MultiDiscrete space (#311).
            action = self.env._a.check(action)
        else:
            # An agent that spawned in the episode's last step is born
            # flushed: it has no buffered action, which the wrapper's dead
            # step deletes.
            self._actions.setdefault(agent, None)
        super().step(action)


def turn_masked_env(env_id: str, seed: int | None = None, **kwargs):
    """The optional turn-masked interface of ADR-0013: no shipped
    environment declares it, so every request is refused."""
    raise _native.ArgumentError(f"{env_id} does not declare 'turn_masked' (ADR-0013)")
