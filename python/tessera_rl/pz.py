# SPDX-License-Identifier: MIT OR Apache-2.0
"""PettingZoo adapters for the multi-agent environments.

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
``tessera/Survival-v1``        simultaneous  ``parallel_env``
``tessera/Connect4-v1``        sequential    ``aec_env``
``tessera/Leduc-v1``           sequential    ``aec_env``
``tessera/Lanterns-v1``        sequential    ``aec_env``
``tessera/Othello-v1``         sequential    ``aec_env``
``tessera/Go9-v1``             sequential    ``aec_env``
``tessera/Vrp-v1``             sequential    ``aec_env``
=============================  ============  =============================

- A simultaneous environment is Parallel natively. Its AEC form buffers
  each agent's action and steps the environment once per cycle, when every
  live agent has acted; observations change only at cycle boundaries
  (PettingZoo's ``parallel_to_aec``).
- A sequential environment is AEC only, with the acting agent as
  ``agent_selection``. It is never presented as Parallel: that would
  change what each agent knows when it acts.

An agent is one occupant of a slot: identity is (slot, generation).
Generation 0 is named ``agent_<slot>``, and a later
generation ``agent_<slot>_g<generation>``: PettingZoo forbids a name to
come back once its agent is done. ``possible_agents`` are the agents
alive at reset, and grow when an environment that spawns agents (a host
environment with ``spawns``) fills a slot. An agent leaves
``agents`` after the step in which it is terminated or truncated: its
final reward and flag are reported once, in that step, and at the end of
an episode every agent still alive gets its final reward and flag in the
same step. A freed slot is reused no earlier than the next
step; its new agent joins ``agents`` in the step it spawns, with its first
observation and info and the engine's reward and flags for that step (a
spawn in the episode's last step is flushed with the others and does not
stay in ``agents``). Native AEC refuses a spawning environment
(``CapabilityError``).

Observations are dicts of the environment's leaves plus ``action_mask``
(``int8``, one entry per action).

Policy maps: every first-party agent's ``infos`` carry its
``"policy_group"`` (``"learner"``, ``"external"``, ``"heuristic"`` or
``"privileged_heuristic"``). ``parallel_env(..., visible="learner")`` hides the
heuristic slots from ``agents``, ``possible_agents`` and the step dicts (the
engine plays them); their rewards stay in each visible agent's
``infos["heuristic_rewards"]``, by agent name. Native AEC refuses
``visible="learner"`` (``CapabilityError``). ``agents=`` is an environment's
own agent count (Snake, GridKeyDoor, PaddleDuel, LimitOrderBook), so both can
be set; ``agents="all"`` and ``agents="learner"`` are the filter's deprecated
spelling, which warns ``TesseraDeprecationWarning`` until 0.5.0 (#538).

Teams and roles: ``metadata["teams"]`` and ``metadata["roles"]`` map each
possible agent's name to its slot's team (an int) and role (a string,
``""`` for none), on the Parallel and AEC forms alike. They are
configuration facts, fixed for the environment's life, so they are not in
``infos``. Tag's pursuers are team 0 with role ``"pursuer"`` and its
evaders team 1 with role ``"evader"``; a host environment has team 0 and
role ``""`` for every agent.

An environment with several discrete heads (``MultiDiscrete``, the RTS:
three heads per unit slot) has ``MultiDiscrete(nvec)`` as its action
space, and an action is one integer per head. Its heads are conditional,
which a Gymnasium ``MultiDiscrete`` mask (one independent mask per
component) cannot express, so its observations carry no ``action_mask``.
The masks are in ``infos[agent]``, beside only the agent's policy group
(SPEC 11.3):

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
column, is replaced by the environment's no-op; it is not an
error.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec_wrapper
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from tessera_rl import _envs, _flat, _host, _native, _render
from tessera_rl._bounds import leaf_bound
from tessera_rl._env import Env, _discrete_actions, _multi_actions, native_kwargs
from tessera_rl.masks import head_views

#: Native constructors and the PettingZoo interfaces each environment
#: declares. Anything else is refused at construction.
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
    "tessera/Vrp-v1": _native.make_vrp,
    # Simultaneous; it declares Parallel only (cycle buffering of a
    # multi-head environment is not designed yet).
    "tessera/RTS-v1": _native.make_rts,
    # Simultaneous, multi-head like the RTS: Parallel only, for the same
    # reason.
    "tessera/LimitOrderBook-v1": _native.make_lob,
    # Simultaneous, two independent heads, 16 to 128 agents: Parallel only
    # (AEC cycle buffering is not designed for this many agents).
    "tessera/Survival-v1": _native.make_survival,
}

#: The maker of each first-party id and the PettingZoo interfaces it
#: declares, read from the registry (``Environment::ADAPTERS``, SPEC 12).
_ENVS = {env_id: (maker, _envs.adapters_of(env_id)) for env_id, maker in _MAKERS.items()}


#: The PettingZoo interfaces of a host environment, by its turns
#: (Mixed through ``parallel_env`` only).
_HOST_DECLARED = {
    "simultaneous": frozenset({"parallel", "aec"}),
    "sequential": frozenset({"aec"}),
    "mixed": frozenset({"parallel"}),
}


def _undeclared(env_id: str, adapter: str, declared, turns=None) -> Exception:
    """The error for ``adapter`` when ``env_id`` does not declare it; a
    Mixed host environment as AEC is a ``CapabilityError``."""
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
    the slots alive after a reset are ``0..initial_agents``, so the
    returned pool has not started an episode."""
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
    declares."""
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


#: The groups whose actions the engine computes.
#: The groups the engine plays, hidden under ``visible="learner"``: the
#: heuristic slots and the league's ``"opponent"`` slots (ADR-0088 section 10).
_HEURISTIC_GROUPS = frozenset({"heuristic", "privileged_heuristic", "opponent"})


#: `agents=` left unset (the filter's old spelling defaults to every agent).
_UNSET: Any = object()


def _warn_agents_filter(agents: str) -> None:
    """`TesseraDeprecationWarning` for `agents=` as the learner filter (#538),
    attributed to the first caller outside this package."""
    level = 2
    frame = sys._getframe(1)
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_envs._PACKAGE):
        frame = frame.f_back
        level += 1
    warnings.warn(
        f"agents={agents!r} as the learner filter is deprecated since 0.4.0: use visible={agents!r} "
        "(removed in 0.5.0, when agents= is only the environment's agent count)",
        _envs.TesseraDeprecationWarning,
        stacklevel=level,
    )


def _visible_option(visible: str | None, agents: Any, kwargs: dict) -> bool:
    """Whether to hide heuristic slots (ADR-0079 item 5.2): ``visible`` is
    ``"all"`` (the default) or ``"learner"``. ``agents`` is an environment's own
    agent-count configuration key (Snake, GridKeyDoor, PaddleDuel,
    LimitOrderBook), passed through to ``kwargs``; the strings ``"all"`` and
    ``"learner"`` there are the filter's deprecated spelling (#538)."""
    if isinstance(agents, str):
        if agents not in ("all", "learner"):
            raise _native.ArgumentError(f'agents must be an agent count, or the deprecated "all" or "learner", not {agents!r}')
        if visible is not None:
            raise _native.ArgumentError(
                f"visible={visible!r} and agents={agents!r} both set the learner filter; pass visible= only"
            )
        _warn_agents_filter(agents)
        visible = agents
    elif agents is not _UNSET:
        kwargs["agents"] = agents
    if visible is None:
        visible = "all"
    if visible not in ("all", "learner"):
        raise _native.ArgumentError(f'visible must be "all" or "learner", not {visible!r}')
    return visible == "learner"


def _state_space(state_leaves) -> spaces.Box | None:
    """The space of the privileged state, one ``Box`` as PettingZoo defines
    ``state()``: the leaf itself, or several leaves flattened in layout order
    to their lossless ``numpy.result_type`` (the PufferLib observation's
    rule); ``None`` without them."""
    leaves = list(state_leaves)
    if not leaves:
        return None
    if len(leaves) == 1:
        _, dtype, shape, low, high = leaves[0]
        dt = np.dtype(dtype)
        return spaces.Box(low=leaf_bound(low, shape, dt), high=leaf_bound(high, shape, dt), shape=tuple(shape), dtype=dt)
    dt = _flat.flat_dtype([leaf[1] for leaf in leaves], "state leaves", "the PettingZoo state()")
    low = np.concatenate([leaf_bound(lo, shape, dt).ravel() for _, _, shape, lo, _ in leaves])
    high = np.concatenate([leaf_bound(hi, shape, dt).ravel() for _, _, shape, _, hi in leaves])
    return spaces.Box(low=low, high=high, shape=(low.size,), dtype=dt)


def agent_name(slot: int, generation: int) -> str:
    """The PettingZoo name of the agent (slot, generation)."""
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
        #: Whether the environment fills empty slots (its ``spawns`` flag).
        self.spawns = bool(layout.get("spawns", False))
        #: ``"host"`` for a third-party host environment, else ``"tessera"``.
        self.kind = "host" if layout.get("kind") == "host" else "tessera"
        #: Whether the host environment refuses agents: every
        #: agent's infos then carry ``refused`` and ``refuse_code``.
        self.refuses = bool(layout.get("refuses_agents", False))
        #: The policy group of each slot; empty for a host
        #: environment, which has no policy map.
        self.groups: tuple[str, ...] = tuple(layout.get("policy_groups", ()))
        #: The team and the role of each slot, fixed for the pool's life;
        #: team 0 and role ``""`` when the layout has none.
        self.teams: tuple[int, ...] = tuple(layout.get("teams", ()))
        self.roles: tuple[str, ...] = tuple(layout.get("roles", ()))
        #: Whether heuristic slots are hidden from the agent lists
        #: (``visible="learner"``).
        self.learner_only = False
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
            lo, hi = leaf_bound(low, shape, dt), leaf_bound(high, shape, dt)
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
        #: The privileged state space (``privileged_state=True``): one Box,
        #: several leaves flattened in layout order; ``None`` without it.
        self.state_space = _state_space(layout.get("state_leaves", ()))

    @property
    def multi(self) -> bool:
        return len(self.nvec) > 1

    def state(self):
        """The privileged state of the one instance after the last call."""
        raw = getattr(self.env, "state", None)
        if self.state_space is None or raw is None:
            raise _native.CapabilityError(
                f"{self.env_id}: state() needs privileged_state=True and a reset first"
            )
        rows = [np.asarray(v[0]) for v in raw.values()]
        if len(rows) == 1:
            return np.array(rows[0], dtype=self.state_space.dtype)
        # Layout order (the dict keeps it), cast to the flat dtype.
        return np.concatenate([r.ravel() for r in rows]).astype(self.state_space.dtype)

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
        if self.groups:
            info["policy_group"] = self.groups[slot]
            if "opponent" in self.groups:
                # The instance's league opponent (ADR-0088 section 10).
                assignment = self.env.assignment
                if assignment is not None:
                    info["assignment"] = int(assignment[0])
        return info

    def team_metadata(self, names: list[str]) -> dict:
        """``metadata["teams"]`` and ``metadata["roles"]``: the team and the
        role of each of ``names``, by agent name."""
        teams, roles = {}, {}
        for name in names:
            self.add_team(name, teams, roles)
        return {"teams": teams, "roles": roles}

    def add_team(self, name: str, teams: dict, roles: dict) -> None:
        """Adds agent ``name``'s team and role to ``teams`` and ``roles``."""
        slot = self.slot_of[name]
        teams[name] = self.teams[slot] if slot < len(self.teams) else 0
        roles[name] = self.roles[slot] if slot < len(self.roles) else ""

    def hidden(self, slot: int) -> bool:
        """Whether agent ``slot`` is hidden: a heuristic or opponent slot under
        ``visible="learner"``."""
        return self.learner_only and slot < len(self.groups) and self.groups[slot] in _HEURISTIC_GROUPS

    def heuristic_rewards(self) -> dict[str, float]:
        """The last step's rewards of the hidden slots, by agent name, for
        the visible agents' infos."""
        step = self.last
        return {
            agent_name(slot, int(step.slot_generations[0, slot])): float(step.rewards[0, slot])
            for slot in range(len(self.groups))
            if self.hidden(slot)
        }

    def refusal(self, slot: int) -> dict:
        """``refused`` and ``refuse_code`` of agent ``slot`` in the last
        call; all clear after a reset."""
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
        ``pending_reset`` output; never for a first-party one."""
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

    def __init__(
        self,
        env_id: str,
        seed: int | None = None,
        *,
        _pool=None,
        render_mode: str | None = None,
        visible: str | None = None,
        agents: Any = _UNSET,
        **kwargs,
    ):
        learner_only = _visible_option(visible, agents, kwargs)
        pool = _pool if _pool is not None else _native_pool(env_id, "parallel", seed, kwargs)
        self._a = _Agents(pool, env_id)
        self._a.learner_only = learner_only
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
        self.possible_agents = [n for n in self._a.names if not self._a.hidden(self._a.slot_of[n])]
        self.agents = list(self.possible_agents)
        self.metadata.update(self._a.team_metadata(self.possible_agents))
        if self._a.state_space is not None:
            self.state_space = self._a.state_space

    def _visible(self, names: list[str]) -> list[str]:
        """``names`` without the hidden heuristic slots."""
        return [n for n in names if not self._a.hidden(self._a.slot_of[n])]

    def observation_space(self, agent):
        return self._a.observation_space

    def action_space(self, agent):
        return self._a.action_space

    def state(self):
        """The privileged state of the environment after the last ``reset``
        or ``step`` (``privileged_state=True``), as ``state_space`` describes:
        both hidden and public information, for a centralized critic. Never
        feed it to an acting policy. ``CapabilityError`` without the option."""
        return self._a.state()

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Archive resets: forwarded to the wrapped
        ``tessera_rl`` environment's ``set_reset_source``."""
        self._a.env.set_reset_source(archive, weights)

    def reset_from(self, ids, entries):
        """Archive resets: forwarded to the wrapped
        ``tessera_rl`` environment's ``reset_from``."""
        return self._a.env.reset_from(ids, entries)

    def reset(self, seed=None, options=None):
        self._a.last = self._a.env.reset(seed)
        self.agents = self._visible(self._a.live())
        if self._a.spawns:
            # A slot's generation survives a reset, so its name does too.
            self.possible_agents = list(self.agents)
            self._a.slot_of = {name: self._a.slot_of[name] for name in self.agents}
            # The names may carry a generation: keep the maps on them.
            self.metadata.update(self._a.team_metadata(self.possible_agents))
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
        if self._a.learner_only:
            # Hidden heuristic slots: their rewards stay in the infos, for
            # logging (ADR-0079 item 5.2).
            logged = self._a.heuristic_rewards()
            for name in infos:
                infos[name]["heuristic_rewards"] = logged
        self.agents = [n for n in self.agents if not (terms[n] or truncs[n])]
        if self._a.spawns:
            # Agents that spawned this step: alive now, not alive before.
            for name in self._visible(self._a.live()):
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
                    # A new occupant of the slot: the slot's team and role.
                    self._a.add_team(name, self.metadata["teams"], self.metadata["roles"])
        return obs, rewards, terms, truncs, infos

    def render(self):
        """With ``render_mode="rgb_array"``, the frame, ``(height, width, 3)``
        ``uint8``, drawn from a snapshot; ``None`` otherwise."""
        if self.render_mode is None:
            return None
        return self._a.env.render([0])[0]

    def close(self):
        self._a.env.close()


class TesseraAECEnv(AECEnv):
    """A sequential Tessera environment as a PettingZoo ``AECEnv``: the
    acting agent is ``agent_selection``, and each ``step`` is one engine
    step with that agent's action. It shows every agent: to hide heuristic
    seats, use the Parallel form with ``visible="learner"``."""

    metadata = {"render_modes": [], "name": "tessera", "is_parallelizable": False}

    def __init__(
        self,
        env_id: str,
        seed: int | None = None,
        *,
        _pool=None,
        render_mode: str | None = None,
        visible: str | None = None,
        agents: Any = _UNSET,
        **kwargs,
    ):
        super().__init__()
        if _visible_option(visible, agents, kwargs):
            # The turn order would have to skip hidden seats (ADR-0079 stage A
            # hides heuristic slots in the Parallel form only).
            raise _native.CapabilityError('visible="learner" is for the Parallel form; native AEC shows every agent')
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
        self.metadata.update(self._a.team_metadata(self.possible_agents))
        if self._a.state_space is not None:
            self.state_space = self._a.state_space
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

    def state(self):
        """The privileged state of the environment after the last ``reset``
        or ``step`` (``privileged_state=True``), as ``state_space`` describes:
        both hidden and public information, for a centralized critic. Never
        feed it to an acting policy. ``CapabilityError`` without the option."""
        return self._a.state()

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
        self.infos = {n: self._group_info(n) for n in self.agents}
        self.agent_selection = self._acting() or self.agents[0]

    def _group_info(self, name: str) -> dict:
        """An agent's infos at the start of an episode: its policy group,
        for a first-party environment."""
        slot = self._a.slot_of[name]
        return {"policy_group": self._a.groups[slot]} if self._a.groups else {}

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Archive resets: forwarded to the wrapped
        ``tessera_rl`` environment's ``set_reset_source``."""
        self._a.env.set_reset_source(archive, weights)

    def reset_from(self, ids, entries):
        """Archive resets: forwarded to the wrapped
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
        ``uint8``, drawn from a snapshot; ``None`` otherwise."""
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
    """The optional turn-masked PettingZoo interface: no shipped
    environment declares it, so every request is refused."""
    raise _native.ArgumentError(f"{env_id} does not declare 'turn_masked' (ADR-0013)")
