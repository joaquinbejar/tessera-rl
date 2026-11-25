# SPDX-License-Identifier: MIT OR Apache-2.0
"""Host environments from third-party wheels (host protocol 1, SPEC 14.4
draft).

A wheel declares its environments as entry points of the group
``tessera_rl.envs``: ``"<namespace>/<Name>-v<N>" = "<module>:<factory>"``.
``tessera_rl.make`` looks an id up in the first-party table first, then in
these entry points, and wraps the host object the factory returns in a
:class:`HostEnv`.

The entry points are read once per process and cached, without
invalidation: after installing or removing a host wheel, restart the
interpreter for ``tessera_rl.make`` to see it.

Only CPython objects, the buffer protocol and DLPack cross between the two
wheels; this module checks the host object against protocol 1 when it
wraps it, and refuses anything else with ``ConfigError``.
"""

from __future__ import annotations

import weakref
from importlib import metadata
from typing import NamedTuple

import numpy as np

from tessera_rl import _native
from tessera_rl._env import Env, Step, _index, _resolve_constructor_annotations

#: The newest host protocol this ``tessera_rl`` speaks (SPEC 14.4.7).
PROTOCOL = 2

#: Every host protocol this ``tessera_rl`` accepts: 1, and 2 for an
#: environment that refuses agents.
PROTOCOLS = (1, 2)

#: The outputs of a protocol 2 object, which a wrapper must read
#:.
_REFUSAL_KEYS = ("refused", "refuse_code")

#: The entry-point group of host environment factories (SPEC 14.4.1).
GROUP = "tessera_rl.envs"

#: The namespace reserved for first-party ids.
_RESERVED = "tessera/"

#: The members a host object must have (SPEC 14.4.2).
_MEMBERS = (
    "host_protocol",
    "tessera_rl_min",
    "tessera_rl_below",
    "layout",
    "reset",
    "step",
    "close",
    "is_poisoned",
    "_tessera_current_generation",
    "_tessera_set_fp_policy",
)

#: The layout keys a host object must return (SPEC 14.4.3).
_LAYOUT_KEYS = (
    "num_envs",
    "agents",
    "initial_agents",
    "spawns",
    "action_choices",
    "action_bounds",
    "action_nvec",
    "mask_len",
    "mask_heads",
    "entity_group_heads",
    "turns",
    "outputs",
    "leaves",
    "env_id",
    "env_version",
    "perfect_information",
    "kind",
    "auto_reset",
)

#: The output keys every reset and step must return (SPEC 14.4.4).
_OUTPUT_KEYS = (
    "observations",
    "terminal_observations",
    "rewards",
    "agent_terminated",
    "agent_truncated",
    "alive",
    "acting",
    "episode_terminated",
    "episode_truncated",
    "masks",
    "slot_generations",
    "episode_return",
    "episode_length",
    "aborted",
    "abort_code",
    "pending_reset",
)

#: First-party methods a host environment does not have (SPEC 14.4.2).
_ABSENT = (
    "snapshot_all",
    "send",
    "recv",
    # Host protocols carry no Render (ADR-0083).
    "render",
    "scene",
)

#: The pool options a factory takes by name (SPEC 14.4.1 item 4); every
#: other keyword is the environment's config.
_POOL_OPTIONS = ("num_envs", "seed", "auto_reset", "max_episode_steps", "workers", "cpu_set")

#: Every live wrapped host object (``HostEnv`` and the adapters' alike),
#: for ``set_fp_policy`` (SPEC 14.4.2).
_LIVE: weakref.WeakSet = weakref.WeakSet()


def _config(message: str) -> _native.ConfigError:
    return _native.ConfigError(message)


def _version(value, what: str) -> tuple[int, int, int]:
    """``MAJOR.MINOR.PATCH`` as an int tuple (no ``packaging``)."""
    if not isinstance(value, str):
        raise _config(f"host object {what} must be a str, got {type(value).__name__}")
    parts = value.split(".")
    if len(parts) != 3 or not all(p.isascii() and p.isdigit() for p in parts):
        raise _config(f"host object {what} {value!r} is not MAJOR.MINOR.PATCH")
    major, minor, patch = (int(p) for p in parts)
    return (major, minor, patch)


def _own_version() -> tuple[int, int, int]:
    core = _native.version().split("+")[0].split("-")[0]
    return _version(core, "tessera_rl version")


# -- discovery (SPEC 14.4.1) ---------------------------------------------------

#: id -> [(distribution name, entry point)], read once per process.
_CACHE: dict | None = None


def _entry_points() -> dict:
    global _CACHE
    if _CACHE is None:
        found: dict = {}
        for ep in metadata.entry_points(group=GROUP):
            dist = getattr(getattr(ep, "dist", None), "name", None) or "<unknown distribution>"
            found.setdefault(ep.name, []).append((dist, ep))
        _CACHE = found
    return _CACHE


def host_ids() -> list[str]:
    """The ids the installed host wheels declare (cached per process)."""
    return sorted(_entry_points())


def is_host_id(env_id: str) -> bool:
    """Whether an installed host wheel declares ``env_id``."""
    return env_id in _entry_points()


def _factory(env_id: str, known_first_party):
    entries = _entry_points().get(env_id)
    if not entries:
        known = sorted(set(known_first_party) | set(_entry_points()))
        raise _native.ArgumentError(f"unknown environment {env_id!r}; known: {known}")
    if env_id.startswith(_RESERVED):
        raise _config(f"{env_id!r} is in the reserved namespace {_RESERVED!r}; a host wheel cannot declare it")
    if len(entries) > 1:
        dists = sorted(d for d, _ in entries)
        raise _config(f"{env_id!r} is declared by more than one distribution: {dists}")
    _, ep = entries[0]
    return ep.load()


def _check_outputs(outputs: str) -> None:
    """Refuses a bad ``outputs`` before the factory builds anything."""
    if outputs not in ("borrowed", "owned"):
        raise _native.ArgumentError(f"outputs is 'borrowed' or 'owned', not {outputs!r}")


def _refuse_spin(kwargs: dict) -> None:
    """``spin`` is a first-party pool option; host protocol 1 has no spin
    budget, so it is refused here rather than passed on as config.
    ``spin=None`` (the default) is dropped."""
    if kwargs.pop("spin", None) is not None:
        raise _native.CapabilityError("spin is a first-party pool option; host environments do not take it")
    # Observation transforms run on first-party pools only (ADR-0078 item 1.4).
    if kwargs.pop("transforms", None):
        raise _native.CapabilityError("observation transforms are not supported for host environments (ADR-0078)")
    # The worker wait policy (ADR-0074 item 5) is not part of host protocol 1 either.
    for key in ("wait", "wait_cap_us"):
        if kwargs.pop(key, None) is not None:
            raise _native.CapabilityError(f"{key} is a first-party pool option; host environments do not take it")
    # ADR-0079 item 1.3: policy maps and the privileged state buffer are not
    # part of host protocol 1.
    for key in ("policy_map", "privileged_state"):
        if kwargs.pop(key, None) is not None:
            raise _native.CapabilityError(f"{key} is not available for host environments (ADR-0079)")


def make(env_id: str, known_first_party, num_envs: int, seed: int, outputs: str, kwargs: dict) -> HostEnv:
    """``tessera_rl.make`` for a host id: checks the mode, calls the factory
    with the pool options and the config, and wraps the object."""
    mode = kwargs.pop("mode", None)
    if mode not in (None, "sync"):
        raise _native.CapabilityError(f"host environments run in sync mode only, not {mode!r} (SPEC 14.4.5)")
    for split in ("batch_size", "gather"):
        if split in kwargs:
            raise _native.CapabilityError(f"{split} is for async mode, which host environments do not have")
    _refuse_spin(kwargs)
    _check_outputs(outputs)
    factory = _factory(env_id, known_first_party)
    return HostEnv(_call_factory(factory, num_envs, seed, kwargs), outputs)


def native_for(env_id: str, num_envs: int, seed: int, outputs: str, kwargs: dict) -> _HostNative:
    """A host object of ``env_id`` behind ``_native.Env``'s interface, for
    the Gymnasium and PettingZoo adapters. ``max_episode_steps=0`` (the
    adapters' "no limit") is ``None`` for the factory."""
    kwargs = dict(kwargs)
    for split in ("mode", "batch_size", "gather"):
        if kwargs.pop(split, None) not in (None, "sync"):
            raise _native.CapabilityError(f"host environments run in sync mode only ({split})")
    _refuse_spin(kwargs)
    if kwargs.get("max_episode_steps") == 0:
        kwargs["max_episode_steps"] = None
    _check_outputs(outputs)
    factory = _factory(env_id, ())
    return _HostNative(_call_factory(factory, num_envs, seed, kwargs), outputs)


def refuse(env_id: str, module: str) -> None:
    """The guard of the adapters that take no host environment (PufferLib,
    JAX): ``CapabilityError`` for a host id, before anything is built."""
    if is_host_id(env_id):
        raise _native.CapabilityError(f"{env_id} is a host environment; {module} does not support them")


def host_infos(out: dict, owned: bool, n: int, agents: int = 1, refuses: bool = False) -> dict:
    """The Gymnasium infos of a host call (SPEC 14.2): ``aborted``,
    ``abort_code`` and ``pending_reset``,
    and for an environment that refuses agents ``refused`` and
    ``refuse_code`` (shape ``(n, agents)``), each with its ``_``-prefixed
    mask, all true."""
    def field(key, dtype, shape):
        if owned:
            return np.frombuffer(out[key], dtype=dtype).reshape(shape).copy()
        return np.array(out[key])

    keys = [("aborted", np.bool_, (n,)), ("abort_code", "<u4", (n,)), ("pending_reset", np.bool_, (n,))]
    if refuses:
        keys += [("refused", np.bool_, (n, agents)), ("refuse_code", "<u4", (n, agents))]
    info = {}
    for key, dtype, shape in keys:
        info[key] = field(key, dtype, shape)
        info["_" + key] = np.ones(n, dtype=np.bool_)
    return info


def _call_factory(factory, num_envs, seed, kwargs):
    """The factory with exact built-in values: ints, strs, and the config
    as given (the host module refuses what it cannot take)."""
    options = {"num_envs": _index(num_envs, "num_envs"), "seed": _index(seed, "seed")}
    for key in _POOL_OPTIONS[2:]:
        if key in kwargs:
            value = kwargs.pop(key)
            if value is None:
                options[key] = None
            elif key == "auto_reset":
                options[key] = str(value)
            elif key == "cpu_set":
                options[key] = [_index(c, "cpu_set") for c in value]
            else:
                options[key] = _index(value, key)
    return factory(**options, **kwargs)


# -- the adapter-facing native object -----------------------------------------


def _dtype_bounds(dtype: str) -> tuple[list[float], list[float]]:
    """A leaf's bounds when the protocol gives none: its dtype's range, as
    exact ints for integer and bool dtypes, floats for float dtypes."""
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return [0], [1]
    if dt.kind in "iu":
        info = np.iinfo(dt)
        return [int(info.min)], [int(info.max)]
    return [float("-inf")], [float("inf")]


def _team_defaults(layout: dict) -> None:
    """Gives ``layout`` the ``teams`` and ``roles`` keys of a first-party
    layout: the host protocol has no team or role, so every slot has team 0
    and role ``""``."""
    agents = layout["agents"]
    layout["teams"] = [0] * agents
    layout["roles"] = [""] * agents


class _HostNative:
    """A host object behind the interface of ``_native.Env`` that
    ``tessera_rl.Env`` and the adapters read: ``layout()``, ``reset(seed,
    ids)``, ``step(bytes)``, ``is_poisoned()`` and ``close()``, with owned
    copies in ``outputs="owned"``."""

    def __init__(self, obj, outputs: str):
        _check_outputs(outputs)
        check(obj)
        protocol = obj.host_protocol
        self._obj = obj
        self._owned = outputs == "owned"
        layout = dict(obj.layout())
        missing = [k for k in _LAYOUT_KEYS if k not in layout]
        if missing:
            raise _config(f"host object layout() lacks {missing}")
        if layout["kind"] != "host" or layout["outputs"] != "borrowed":
            raise _config("host object layout() must have kind 'host' and outputs 'borrowed'")
        # Protocol 2 if and only if the environment refuses agents.
        refuses = layout.get("refuses_agents", False)
        if not isinstance(refuses, bool) or refuses != (protocol == 2):
            raise _config(
                f"host object speaks host protocol {protocol} with refuses_agents={refuses!r}; "
                "protocol 2 requires refuses_agents=True and protocol 1 requires it absent or False"
            )
        layout["refuses_agents"] = refuses
        # ADR-0073: optional; absent means no snapshots.
        snapshots = layout.get("snapshots", False)
        if not isinstance(snapshots, bool):
            raise _config(f"host object layout() has snapshots={snapshots!r}; it must be a bool")
        layout["snapshots"] = snapshots
        _team_defaults(layout)
        self._keys = _OUTPUT_KEYS + (_REFUSAL_KEYS if refuses else ())
        # The first-party leaf form, (name, dtype, shape, low, high): the
        # protocol has no bounds, so a leaf spans its dtype's range.
        layout["leaves"] = [(name, dtype, list(shape), *_dtype_bounds(dtype)) for name, dtype, shape in layout["leaves"]]
        layout["outputs"] = outputs
        self._layout = layout
        self._checked = False
        _LIVE.add(self)
        # The policy in force now; later changes reach it through
        # set_fp_policy (SPEC 14.4.2).
        obj._tessera_set_fp_policy(_native.get_fp_policy())

    def layout(self) -> dict:
        return self._layout

    def _out(self, out: dict) -> dict:
        if not self._checked:
            missing = [k for k in self._keys if k not in out]
            if missing:
                raise _config(f"host object outputs lack {missing}")
            self._checked = True
        if not self._owned:
            return out
        copy = {}
        for key, value in out.items():
            if isinstance(value, list):
                copy[key] = [bytearray(memoryview(v)) for v in value]
            elif key in self._keys:
                copy[key] = bytearray(memoryview(value))
        return copy

    def reset(self, seed=None, ids=None, *_):
        n = self._layout["num_envs"]
        if ids is not None and sorted(ids) != list(range(n)):
            raise _native.CapabilityError("a host environment resets every instance at once (SPEC 14.4.2)")
        if isinstance(seed, list):
            if any(s is None for s in seed):
                if all(s is None for s in seed):
                    seed = None
                else:
                    raise _native.CapabilityError("a host environment keeps all seeds or none in one reset")
            elif ids is not None:
                # Seeds given in `ids` order: put them in instance order.
                order = dict(zip(ids, seed, strict=True))
                seed = [order[i] for i in range(n)]
        return self._out(self._obj.reset(seed))

    def step(self, actions) -> dict:
        # Host protocol 1 takes exact bytes. Env.step hands the native step
        # an int32 array's memoryview (#152 E2); a host gets its bytes.
        if isinstance(actions, memoryview):
            actions = actions.tobytes()
        return self._out(self._obj.step(actions))

    def is_poisoned(self) -> bool:
        return self._obj.is_poisoned()

    def close(self) -> None:
        self._obj.close()


def check(obj) -> None:
    """Checks a host object against protocol 1 (SPEC 14.4.2): its members,
    its protocol version and the ``tessera_rl`` versions it accepts."""
    missing = [m for m in _MEMBERS if not hasattr(obj, m)]
    if missing:
        raise _config(f"not a host object (protocol {PROTOCOL}): it lacks {missing}")
    protocol = obj.host_protocol
    if isinstance(protocol, bool) or protocol not in PROTOCOLS:
        raise _config(f"host object speaks host protocol {protocol!r}; this tessera_rl speaks {PROTOCOLS}")
    low = _version(obj.tessera_rl_min, "tessera_rl_min")
    below = _version(obj.tessera_rl_below, "tessera_rl_below")
    own = _own_version()
    if not low <= own < below:
        raise _config(
            f"host object needs tessera_rl >= {obj.tessera_rl_min} and < {obj.tessera_rl_below}; "
            f"this is {'.'.join(map(str, own))}"
        )


# -- HostEnv ---------------------------------------------------------------------


class HostStep(NamedTuple):
    """The outputs of one host call: ``Step``'s fields, in the same order,
    then the host-only outputs.

    ``aborted`` and ``abort_code`` (``uint32``, 0 where not aborted) are
    per instance, of the last call; ``pending_reset`` is whether each
    instance waits for a reset (exact in every turn structure; ``aborted``
    alone never means pending). An aborted instance's observation rows
    are zeros and its masks false.
    """

    observations: dict[str, np.ndarray]
    rewards: np.ndarray
    agent_terminated: np.ndarray
    agent_truncated: np.ndarray
    alive: np.ndarray
    episode_terminated: np.ndarray
    episode_truncated: np.ndarray
    masks: np.ndarray | None
    acting: np.ndarray
    slot_generations: np.ndarray
    aborted: np.ndarray
    abort_code: np.ndarray
    pending_reset: np.ndarray
    #: ``(num_envs, agents)`` refused agents of the last call, or ``None``
    #: when the environment does not refuse agents.
    refused: np.ndarray | None = None
    #: ``(num_envs, agents)`` ``uint32`` refuse codes, 0 where not refused,
    #: or ``None``.
    refuse_code: np.ndarray | None = None


_resolve_constructor_annotations(HostStep)


def _absent(name: str):
    def method(self, *args, **kwargs):
        raise _native.CapabilityError(f"host environments have no {name} (SPEC 14.4.2)")

    method.__name__ = name
    method.__doc__ = f"Not available for host environments: raises ``CapabilityError``."
    return method


class HostEnv:
    """A pool of a third-party host environment (host protocol 1).

    It has ``Env``'s attributes and ``reset``, ``step``, ``is_poisoned``
    and ``close``, and returns :class:`HostStep`, ``Step`` plus
    ``aborted``, ``abort_code`` and ``pending_reset``. Host environments run in sync mode only and reset
    every instance at once; ``snapshot_all``, ``send`` and ``recv`` raise
    ``CapabilityError``.

    Snapshots exist only when the environment provides them
    (``snapshots`` is true): ``snapshot``, ``restore``, ``restore_all``,
    ``fork``, ``state_hash``, ``import_record``, ``encode`` and ``load``,
    over records and files in the host formats. Without them each raises
    ``CapabilityError``. There is no ``fork_branched``, and a poisoned host
    environment refuses all of them, ``restore_all`` and ``load``
    included: build a new one.

    ``outputs="borrowed"`` returns read-only views with ``Env``'s borrow
    window (``is_stale`` tracks them); ``outputs="owned"`` copies them.
    """

    #: Host environments do not render.
    canvas = None

    def __init__(self, obj, outputs: str = "borrowed"):
        self._native = _HostNative(obj, outputs)
        self._env = Env(self._native)
        layout = self._native.layout()
        self.mode = "sync"
        self.num_envs: int = layout["num_envs"]
        self.agents: int = layout["agents"]
        #: Slots alive after each reset (``agents`` is the capacity).
        self.initial_agents: int = layout["initial_agents"]
        #: Whether agents can spawn after the reset.
        self.spawns: bool = layout["spawns"]
        self.action_choices: int = layout["action_choices"]
        self.action_bounds = layout["action_bounds"]
        self.action_nvec: tuple[int, ...] = tuple(layout["action_nvec"])
        self.mask_heads = self._env.mask_heads
        self.entity_group_heads: int = layout["entity_group_heads"]
        self.turns: str = layout["turns"]
        self.outputs: str = outputs
        self.halves = None
        #: The registered id and outputs version of the environment.
        self.env_id: str = layout["env_id"]
        self.env_version: int = layout["env_version"]
        self.auto_reset: str = layout["auto_reset"]
        #: ``"host"``: a third-party host environment.
        self.kind: str = "host"
        #: Whether the environment refuses agents: ``HostStep.refused`` and
        #: ``refuse_code`` are set (host protocol 2).
        self.refuses_agents: bool = layout["refuses_agents"]
        #: Whether the environment provides snapshots; absent from
        #: the layout of older wheels, which have none.
        self.snapshots: bool = layout["snapshots"]

    def _snap(self, name: str):
        """The host object, for a snapshot method; ``CapabilityError``
        when the environment has no snapshots."""
        if not self.snapshots:
            raise _native.CapabilityError(f"this host environment has no snapshots, so no {name} (ADR-0073)")
        return self._native._obj

    def snapshot(self, index: int) -> bytes:
        """The record of instance ``index``: ``bytes`` holding an identity
        digest, the engine header and the environment's own state.
        Privileged: it holds full state."""
        return self._snap("snapshot").snapshot([_index(index, "index")])[0]

    def restore(self, ids, snapshots) -> None:
        """Restores the records ``snapshots`` into ``ids``, in order. Every
        record is checked first (``RecordError``); a refused call changes
        nothing."""
        obj = self._snap("restore")
        obj.restore([_index(i, "ids") for i in ids], [bytes(r) for r in snapshots])

    def restore_all(self, snapshots) -> None:
        """Restores one record per instance."""
        self._snap("restore_all").restore_all([bytes(r) for r in snapshots])

    def import_record(self, index: int, record: bytes) -> None:
        """Restores one (untrusted) record into instance ``index``."""
        self._snap("import_record").restore([_index(index, "index")], [bytes(record)])

    def fork(self, src: int, dst: list[int]) -> None:
        """Copies instance ``src`` into every instance of ``dst``."""
        self._snap("fork").fork(_index(src, "src"), [_index(d, "dst") for d in dst])

    def state_hash(self, index: int) -> int:
        """Instance ``index``'s host state hash. It carries no determinism
        tier and may differ across platforms."""
        raw = self._snap("state_hash").state_hash(_index(index, "index"))
        return int.from_bytes(raw, "little")

    def encode(self, ids=None) -> bytes:
        """A host snapshot file (``TESHOST``) of ``ids``, or of every
        instance. It holds full state and is privileged."""
        obj = self._snap("encode")
        if ids is None:
            ids = range(self.num_envs)
        return obj.encode([_index(i, "ids") for i in ids])

    def load(self, data: bytes, ids=None, max_instances=None, max_decoded_bytes=None) -> None:
        """Loads a file from ``encode`` into ``ids`` (default: every
        instance), with the default decode limits: ``max_instances`` and
        ``max_decoded_bytes`` must be ``None`` (``ArgumentError``).
        ``PersistError`` for a damaged, malformed or mismatched file. It is no recovery path: a
        poisoned host environment refuses it."""
        obj = self._snap("load")
        if max_instances is not None or max_decoded_bytes is not None:
            raise _native.ArgumentError("load: host files are read with the default decode limits; max_instances and max_decoded_bytes must be None")
        if ids is None:
            ids = range(self.num_envs)
        obj.import_records([_index(i, "ids") for i in ids], data)

    def _host_step(self, step: Step) -> HostStep:
        out = self._env._last_out
        n = self.num_envs

        def field(key, dtype, shape):
            if self.outputs == "borrowed":
                return np.asarray(out[key])
            return np.frombuffer(out[key], dtype=dtype).reshape(shape)

        a = self.agents
        refuses = self.refuses_agents
        return HostStep(
            *step,
            aborted=field("aborted", np.bool_, (n,)),
            abort_code=field("abort_code", "<u4", (n,)),
            pending_reset=field("pending_reset", np.bool_, (n,)),
            refused=field("refused", np.bool_, (n, a)) if refuses else None,
            refuse_code=field("refuse_code", "<u4", (n, a)) if refuses else None,
        )

    def reset(self, seed=None) -> HostStep:
        """Resets every instance: ``seed`` is ``None`` (keep the seeds), an
        int (instance ``i`` gets ``seed + i``) or a list of ``num_envs``
        ints. A full reset clears poison."""
        if isinstance(seed, (list, tuple, np.ndarray)):
            seeds = [_index(s, "seed") for s in seed]
            return self._host_step(self._env._wrap(self._native.reset(seeds)))
        return self._host_step(self._env.reset(seed))

    def step(self, actions) -> HostStep:
        """One step, with ``Env.step``'s action rules."""
        return self._host_step(self._env.step(actions))

    def is_poisoned(self) -> bool:
        return self._native.is_poisoned()

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Archive resets are for first-party environments."""
        raise _native.CapabilityError("archive resets need a first-party environment (ADR-0075 item 8)")

    def reset_from(self, ids, entries) -> None:
        """Archive resets are for first-party environments."""
        raise _native.CapabilityError("archive resets need a first-party environment (ADR-0075 item 8)")

    def restore_pool(self, snapshot, archive=None) -> None:
        """Whole-pool restore is for first-party environments."""
        raise _native.CapabilityError("restore_pool needs a first-party environment (ADR-0075 item 8)")

    def close(self) -> None:
        self._native.close()

    def __enter__(self) -> HostEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


for _name in _ABSENT:
    setattr(HostEnv, _name, _absent(_name))
del _name


def set_fp_policy(policy: str) -> None:
    """Sets the floating-point policy of ``tessera_rl`` and of
    every live host object, whose modules each keep their own."""
    _native.set_fp_policy(policy)
    for native in list(_LIVE):
        native._obj._tessera_set_fp_policy(policy)
