# SPDX-License-Identifier: MIT OR Apache-2.0
"""A native ``pufferlib.PufferEnv`` over Tessera, Sync mode, for the pinned
PufferLib 3.0.0 (``compat/pufferlib``). Importing ``tessera_rl`` never needs
PufferLib; only this module does.

    vecenv = pufferlib.vector.make(make_puffer("tessera/CartPole-v1", num_envs=64))
    obs, infos = vecenv.reset(seed=1)

**Importing PufferLib.** ``import pufferlib`` creates a ``resources`` symlink in
the current directory, and raises ``PermissionError`` in a read-only one. This
module does not change directory around it, since that is process-global: import
it from a directory you can write to and do not mind the link in. A failed
import raises ``ImportError`` naming the cause.

**Environments.** An allowlist: CartPole, Tag, Gather and GridKeyDoor. Sequential environments are
refused: PufferLib has no turns, and its trainer would train on idle seats.
Auto-reset is ``same_step``, fixed. ``terminal_observations`` holds final
observations, which PufferLib's tuple has no place for. PufferLib passes no action masks to a policy, so in Tag
an illegal action is a no-op (``env.action_masks`` has the masks); an out-of-range
one raises ``ActionError``. ``masks`` is PufferLib's "agent counted in this
batch", always true, so dead Tag agents count as steps.

**Buffers.** With ``buf=None`` (the native backend) the first ``reset`` replaces
the env's arrays with reshaped read-only views of the engine's buffers, with no
copy. They keep the overwrite semantics of borrowed outputs: a later step or
reset changes them in place. ``torch.as_tensor`` warns once per process that they
are not writable; nothing must write through it. ``vector.Serial`` and
``vector.Multiprocessing`` pass buffers they own, so those backends cost one copy
per step. ``tessera_rl.is_stale`` does not apply to these arrays: the views are
taken once, so it reports ``True`` even for arrays ``step`` has just returned.

**Observations with several leaves** (SPEC 11.1) are flattened: an agent's row
is every leaf raveled in C order, in layout order, so a ``Set`` gives its item
leaves then ``<leaf>_mask`` (ADR-0082). The dtype is ``numpy.result_type`` of the
leaf dtypes (Gather: ``int32``, 26 values), and a layout whose promotion is a
float that cannot hold every value of an integer leaf (a 64-bit integer with a
float, or ``uint64`` with a signed integer) is refused with ``ArgumentError``.
The engine keeps each leaf in its own buffer, so these rows are a copy into one
array the env owns, refilled in place by every ``reset`` and ``step`` (per half in
Pipelined). A single leaf stays a zero-copy view.

**Seeds.** ``vector.make`` drops its ``seed`` argument, so it has no effect. Use
``reset(seed=r)``. A reset derives the base of its instance seeds with
``mix_seed(constructor_seed, r)``, as PufferLib's reset seeds repeat across
Multiprocessing workers, and only its constructor seeds identify an env; instance
``k`` has seed ``base + k``. Ranges of two envs meeting is improbable, about
``E**2 * N / 2**64``, not impossible. Both seeds must be in ``0..2**32``, or
``ArgumentError``; keep the root reset seed below ``2**31`` on Multiprocessing,
which adds to it first: a child refuses a sum outside the range, and the parent
then hangs. The same seeds on the same backend and layout give the
same bytes; backends and layouts differ from each other. ``reset(None)`` first
uses reset seed 0, later keeps the seeds. ``Multiprocessing`` itself raises on
``reset(None)``: unsupported there.

**Processes.** The pool is built at the first reset, after the constructor reads
its layout from a temporary pool, so a driver env in a parent holds no pool at a
fork. Do not reset a driver env, or hold other Tessera pools in the parent,
before forking: an inherited pool refuses in the child. This module never sets a
start method. Spawn is tested at the pins on macOS. Fork is tested on Linux x86-64
at the pins. Spawn needs an
importable creator and a ``__main__`` guard. Forkserver is not tested.

**Pipelined.** ``make_puffer(env_id, num_envs=64, chunk_size=2, mode="pipelined")``
gives :class:`TesseraPufferPipelinedEnv`, on the native backend only: under
the pinned backends, ``vector.Serial`` and ``vector.Multiprocessing``, it refuses in
the parent, which it detects by looking for those backends' constructors in the
call stack; that ``vector.Ray`` is refused the same way is from reading only, as it
is not tested. A list of creators whose first is Sync passes the driver and raises
in the child, which is a parent hang under Multiprocessing. ``workers`` must be at
least 1. ``spin`` (the worker spin budget of ``make``) needs ``workers``, as do ``wait`` and ``wait_cap_us`` (the worker wait policy of ``make``), which pass through to ``make``. The constructor's probe of the halves starts real worker threads, closed
before it returns. ``log_interval`` counts half-steps. The two
halves must be equal, or ``ArgumentError`` naming the sizes; a chunk size that
cannot split the pool raises the engine's ``ConfigError`` (64 instances need a
``chunk_size`` of 32 or less). ``recv`` returns one half's rows and ``send`` pairs
with the half last received, and between ``send`` and the next ``recv`` the env
holds no output arrays, and ``action_masks`` is ``None``, so nothing handed to a
trainer spans a running half. Rows
returned by ``recv`` stay valid until that half's next ``send``. Interrupted while
waiting, a ``recv`` can be called again and returns the same half. There is no
``step`` and no ``terminal_observations``, as the wrapper's half does not carry
them. A policy map's heuristic slots are hidden as in Sync mode (#558): a half's
rows are its visible slots only, copied into arrays the env owns, and ``send``
takes one action per visible slot. With adaptive state a half reads a context two cycles old; CartPole
and Tag have none. No speed gain has been measured; the mode exists so that
workers step one half while the learner infers on the other.

**Multiprocessing hangs on a raise.** PufferLib 3.0.0's worker has no exception
reply: a raise prints in the child and the parent's ``recv`` blocks forever. What
this env can raise in a worker: ``ActionError`` (out-of-range action);
``Poisoned``, ``Panicked`` and ``WorkerDied`` after a panic in an environment;
``ArgumentError`` (seed out of range, a step before reset); ``ForkError`` (an
inherited pool). The constructor validates every option in the parent, but
the host can still refuse threads at the first reset, within ``MAX_WORKERS``.
Use a timeout around such runs. A process that raises after
``PuffeRL`` starts also never exits: its ``Utilization`` thread is not a daemon.

**Platform.** PufferLib 3.0.0 does not build on linux-arm64: its sdist links an
amd64 ``libraylib.a``.
"""

from __future__ import annotations

import sys
from functools import partial

import numpy as np

from tessera_rl import _flat, _host, _native
from tessera_rl._bounds import leaf_bound
from tessera_rl._env import _index, make

try:
    import gymnasium
    import pufferlib
except Exception as e:  # PermissionError in a read-only directory, among others.
    raise ImportError(
        f"tessera_rl.puffer needs PufferLib 3.0.0, and its import failed: {e!r}. "
        "It creates a 'resources' symlink in the current directory, which must be writable."
    ) from e

#: Environments tested through this adapter.
SUPPORTED = frozenset({"tessera/CartPole-v1", "tessera/Gather-v1", "tessera/GridKeyDoor-v1", "tessera/Tag-v1"})

#: Above this a worker count is refused here, rather than failing to start threads
#: at the first reset, which is a parent hang under Multiprocessing.
MAX_WORKERS = 1024

_U32 = 1 << 32
_U64_MASK = (1 << 64) - 1


def mix_seed(constructor_seed: int, reset_seed: int) -> int:
    """Mix version 1: the base of the instance seeds, a bijection on the pair.

    The SplitMix64 finalizer without its gamma addition, over
    ``(constructor_seed << 32) | reset_seed``. Part of the determinism contract.
    """
    c = _seed(constructor_seed, "constructor seed")
    r = _seed(reset_seed, "reset seed")
    z = (c << 32) | r
    z ^= z >> 30
    z = (z * 0xBF58476D1CE4E5B9) & _U64_MASK
    z ^= z >> 27
    z = (z * 0x94D049BB133111EB) & _U64_MASK
    z ^= z >> 31
    return z


def _seed(value, what: str) -> int:
    v = _index(value, what)
    if not 0 <= v < _U32:
        raise _native.ArgumentError(f"{what} must be in 0..2**32, got {v}; it is never truncated")
    return v


def _workers(value) -> int:
    """The worker count, refused in the constructor: the probe is threadless."""
    if type(value) is not int or not 0 <= value <= MAX_WORKERS:
        raise _native.ArgumentError(f"workers must be an int in 0..={MAX_WORKERS}, got {value!r}")
    return value


def flat_dtype(dtypes) -> np.dtype:
    """The dtype of a flattened observation: ``numpy.result_type`` of the leaf
    dtypes. ``ArgumentError`` when it is a float whose mantissa cannot hold every
    value of an integer leaf, the promotions that are not lossless."""
    return _flat.flat_dtype(dtypes, "observation leaves", "the PufferLib adapter")


def _observation_space(leaves: list) -> tuple:
    """``(dtype, shape, low, high)`` of the observation: the leaf itself, or the
    flattening of several leaves in layout order."""
    if len(leaves) == 1:
        _, dtype, shape, low, high = leaves[0]
        return np.dtype(dtype), tuple(shape), leaf_bound(low, shape, dtype), leaf_bound(high, shape, dtype)
    dtype = flat_dtype([leaf[1] for leaf in leaves])
    lows = [leaf_bound(low, shape, dtype).ravel() for _, _, shape, low, _ in leaves]
    highs = [leaf_bound(high, shape, dtype).ravel() for _, _, shape, _, high in leaves]
    low, high = np.concatenate(lows), np.concatenate(highs)
    return dtype, (low.size,), low, high


def _flatten_into(dst: np.ndarray, leaves: list) -> None:
    """Fills ``dst`` (``[rows, total]``) with the leaves, each ``[rows, ...]``."""
    col = 0
    for leaf in leaves:
        flat = leaf.reshape((dst.shape[0], -1))
        dst[:, col : col + flat.shape[1]] = flat
        col += flat.shape[1]


def _in_vector_backend() -> bool:
    """Whether a PufferLib vector backend is constructing this env, or running it
    in a worker. Its driver env gets no ``buf``, so nothing else tells it from the
    native backend, and a refusal in a worker would hang the parent."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get("__name__") == "pufferlib.vector" and frame.f_code.co_name != "make":
            return True
        frame = frame.f_back
    return False


def _check_env_id(env_id: str) -> None:
    if env_id not in SUPPORTED:
        raise _native.ArgumentError(
            f"{env_id!r} is not supported by the PufferLib adapter; supported: {sorted(SUPPORTED)}. "
            "Sequential environments are refused: PufferLib has no turns."
        )


def _spin(value) -> int:
    """The worker spin budget: an int in 0..2**32, refused like ``workers``."""
    if type(value) is not int or not 0 <= value < _U32:
        raise _native.ArgumentError(f"spin must be an int in 0..2**32, got {value!r}")
    return value


class TesseraPufferEnv(pufferlib.PufferEnv):
    """``num_envs`` instances as ``num_envs * agents`` PufferLib agents."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        buf=None,
        seed: int | None = 0,
        workers: int = 0,
        spin: int | None = None,
        log_interval: int | None = 128,
        auto_reset: str = "same_step",
        **env_kwargs,
    ):
        _check_env_id(env_id)
        if auto_reset != "same_step":
            raise _native.ArgumentError(f"auto_reset is same_step here, not {auto_reset!r}")
        if "mode" in env_kwargs or "outputs" in env_kwargs:
            raise _native.ArgumentError("the PufferLib adapter is Sync mode with borrowed outputs")
        if log_interval is not None and _index(log_interval, "log_interval") < 1:
            raise _native.ArgumentError("log_interval must be at least 1, or None")
        self._constructor_seed = _seed(0 if seed is None else seed, "constructor seed")
        self._make_args = {"num_envs": num_envs, "workers": _workers(workers), "auto_reset": auto_reset, **env_kwargs}
        if spin is not None:
            if _workers(workers) < 1:
                raise _native.ArgumentError("spin needs workers >= 1")
            self._make_args["spin"] = _spin(spin)
        self._env_id = env_id
        self._log_interval = log_interval
        self._env = None
        self.instance_seed_base: int | None = None
        self.infos: list = []

        # Validates here, not in a worker; holds no thread and is closed at once.
        # Worker-only options (spin, the ADR-0074 wait policy) are refused without workers.
        probe_args = {k: v for k, v in self._make_args.items() if k not in ("spin", "wait", "wait_cap_us")}
        probe = make(env_id, **{**probe_args, "workers": 0})
        try:
            layout = probe._native.layout()
            if layout["action_bounds"] is not None:
                raise _native.ArgumentError(f"{env_id} has continuous actions; the PufferLib adapter is discrete only")
            if len(layout["action_nvec"]) > 1:
                raise _native.ArgumentError(f"{env_id} has several action heads; the PufferLib adapter takes one")
            self._leaf_names = [leaf[0] for leaf in layout["leaves"]]
            dtype, shape, low, high = _observation_space(layout["leaves"])
            self._instances, self._agents = probe.num_envs, probe.agents
            #: Rows of the engine outputs, hidden slots included.
            self._engine_rows = self._instances * self._agents
            choices = probe.action_choices
            groups = tuple(layout.get("policy_groups", ()))
        finally:
            probe.close()
        #: The agent slots PufferLib sees: every slot, or with a policy map
        #: the learner and external ones; the engine plays the heuristic
        #: slots. `None` when every slot is seen.
        self._visible: list[int] | None = None
        if any(g in ("heuristic", "privileged_heuristic") for g in groups):
            self._visible = [s for s, g in enumerate(groups) if g in ("learner", "external")]
        #: Agents per instance as PufferLib counts them.
        self._row_agents = self._agents if self._visible is None else len(self._visible)
        self.num_agents = self._instances * self._row_agents
        self.single_observation_space = gymnasium.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        self.single_action_space = gymnasium.spaces.Discrete(choices)
        # Hidden slots need a gather each step, so the env keeps its own
        # arrays then, as with a vector backend's buffers.
        self._owned_buffers = buf is not None or self._visible is not None
        super().__init__(buf)
        self._episode_return = np.zeros(self.num_agents, dtype=np.float64)
        self._episode_length = np.zeros(self.num_agents, dtype=np.int64)
        self._tick = 0
        self._finished = [0.0, 0, 0]  # Sum of returns, sum of lengths, episodes.
        self._views: tuple | None = None
        self._masks: np.ndarray | None = None
        self._obs_leaves: list | None = None
        self._flat_obs: np.ndarray | None = None
        self._terminal: np.ndarray | None = None

    @property
    def _flattens(self) -> bool:
        return len(self._leaf_names) > 1

    def _obs_view(self, observations: dict, rows: int, flat: np.ndarray | None) -> np.ndarray:
        """One leaf as a reshaped view, or several flattened into ``flat``."""
        if flat is None:
            # A single leaf's layout name can differ from its output key (CartPole's is ""): take the only one.
            return next(iter(observations.values())).reshape((rows, *self.single_observation_space.shape))
        _flatten_into(flat, [observations[name] for name in self._leaf_names])
        return flat

    def _new_flat(self, rows: int) -> np.ndarray | None:
        if not self._flattens:
            return None
        return np.zeros((rows, *self.single_observation_space.shape), dtype=self.single_observation_space.dtype)

    @property
    def agents_per_batch(self) -> int:
        """``pufferl.py`` reads this name with an RNN; ``PufferEnv`` lacks it."""
        return self.num_agents

    @property
    def action_masks(self) -> np.ndarray | None:
        """Action masks, ``(num_agents, choices)``; ``None`` without masks."""
        if self._masks is not None and self._visible is not None:
            return self._gather(self._masks).reshape((self.num_agents, -1))
        return self._masks

    def _gather(self, rows: np.ndarray) -> np.ndarray:
        """The visible slots' rows of `[N, A, ...]` outputs, a copy."""
        return rows[:, self._visible]

    @property
    def terminal_observations(self) -> np.ndarray:
        """Final observations of the episodes that ended in the last call."""
        self._require_reset("terminal_observations")
        leaves = list(self._env.terminal_observations.values())
        shape = self.single_observation_space.shape
        if self._flattens:
            if self._terminal is None:
                self._terminal = self._new_flat(self._engine_rows)
            _flatten_into(self._terminal, [np.asarray(leaf) for leaf in leaves])
            full = self._terminal
        else:
            full = np.asarray(leaves[0]).reshape((self._engine_rows, *shape))
        if self._visible is not None:
            full = full.reshape((self._instances, self._agents, *shape))
            return self._gather(full).reshape((self.num_agents, *shape))
        return full

    def _require_reset(self, call: str) -> None:
        if self._env is None:
            raise _native.ArgumentError(f"{call} before the first reset")

    def _publish(self, out) -> None:
        """Takes the views of one call's outputs, once: the storage is stable."""
        n = self.num_agents
        self._flat_obs = self._new_flat(self._engine_rows)
        if self._flat_obs is not None:
            self._obs_leaves = [out.observations[name] for name in self._leaf_names]
        obs = self._obs_view(out.observations, self._engine_rows, self._flat_obs)
        if self._visible is not None:
            # `[N, A, ...]` views; `_copy_out` gathers the visible slots.
            obs = obs.reshape((self._instances, self._agents, *self.single_observation_space.shape))
            self._views = (obs, out.rewards, out.agent_terminated, out.agent_truncated)
            self._episode_ended = (out.episode_terminated, out.episode_truncated)
            self._masks = out.masks
            return
        self._views = (
            obs,  # `n` rows: every slot is seen here.
            out.rewards.reshape(n),
            out.agent_terminated.reshape(n),
            out.agent_truncated.reshape(n),
        )
        self._episode_ended = (out.episode_terminated, out.episode_truncated)
        if out.masks is not None:
            self._masks = out.masks.reshape((n, -1))
        if not self._owned_buffers:
            self.observations, self.rewards, self.terminals, self.truncations = self._views

    def _copy_out(self) -> None:
        if self._obs_leaves is not None:
            _flatten_into(self._flat_obs, self._obs_leaves)
        if self._visible is not None:
            n = self.num_agents
            obs, rewards, terminals, truncations = (self._gather(v) for v in self._views)
            self.observations[:] = obs.reshape((n, *self.single_observation_space.shape))
            self.rewards[:] = rewards.reshape(n)
            self.terminals[:] = terminals.reshape(n)
            self.truncations[:] = truncations.reshape(n)
            return
        if self._owned_buffers:
            self.observations[:] = self._views[0]
            self.rewards[:] = self._views[1]
            self.terminals[:] = self._views[2]
            self.truncations[:] = self._views[3]

    def set_reset_source(self, archive=None, weights=None) -> None:
        """Archive resets: forwarded to the wrapped
        ``tessera_rl`` environment's ``set_reset_source``."""
        self._env.set_reset_source(archive, weights)

    def reset_from(self, ids, entries):
        """Archive resets: forwarded to the wrapped
        ``tessera_rl`` environment's ``reset_from``."""
        return self._env.reset_from(ids, entries)

    def reset(self, seed: int | None = None):
        """Resets every instance; see the module notes on seeds."""
        first = self._env is None
        if seed is None and not first:
            base = None
        else:
            base = mix_seed(self._constructor_seed, 0 if seed is None else seed)
        if first:
            self._env = make(self._env_id, seed=base, **self._make_args)
        out = self._env.reset(base)
        if base is not None:
            self.instance_seed_base = base
        if first:
            self._publish(out)
        self._copy_out()
        self.masks[:] = True
        self._episode_return[:] = 0.0
        self._episode_length[:] = 0
        self._tick = 0
        self._finished = [0.0, 0, 0]
        self.infos = []
        return self.observations, self.infos

    def step(self, actions):
        self._require_reset("step")
        if self._visible is not None:
            # The hidden slots' rows are never read: the engine computes them.
            full = np.zeros((self._instances, self._agents), dtype=np.int32)
            full[:, self._visible] = np.asarray(actions).reshape((self._instances, self._row_agents))
            actions = full
        self._env.step(actions)
        self._copy_out()
        # This call's own whole-pool outputs, read now and dropped at once:
        # they are borrowed views, valid only until the next pool call.
        raw, self._env._last_out = self._env._last_out, None
        if raw is None:
            raise RuntimeError("the environment's last call was not this whole-pool step")
        self.infos = self._log(raw) if self._log_interval is not None else []
        return self.observations, self.rewards, self.terminals, self.truncations, self.infos

    def _log(self, raw: dict) -> list:
        # The engine's statistics of episodes that ended in this step
        # (`[N]`, zero elsewhere). Each ended episode counts once per agent
        # row, as PufferLib's per-agent logs do: its return is the sum over
        # its agents, and every agent row shares its length.
        self._tick += 1
        returns, lengths = np.asarray(raw["episode_return"]), np.asarray(raw["episode_length"])
        # Lengths are non-negative whole numbers in float32, so a row is
        # nonzero exactly when its bits are: counting on the uint32 view gives
        # the same count, several times faster than on float32 (#294).
        ended = int(np.count_nonzero(lengths.view(np.uint32)))
        if ended:
            self._finished[0] += float(returns.sum(dtype=np.float64))
            self._finished[1] += int(lengths.sum(dtype=np.float64)) * self._row_agents
            self._finished[2] += ended * self._row_agents
        if self._tick % self._log_interval != 0 or self._finished[2] == 0:
            return []
        total, length, n = self._finished
        self._finished = [0.0, 0, 0]
        return [{"episode_return": total / n, "episode_length": length / n, "n": n}]

    def recv(self):
        self._require_reset("recv")
        return super().recv()

    def close(self) -> None:
        """Frees the pool; idempotent. Arrays handed out stay readable."""
        if self._env is not None:
            self._env.close()


class TesseraPufferPipelinedEnv(TesseraPufferEnv):
    """Two equal halves on the native backend; see the module notes."""

    def __init__(self, env_id: str, num_envs: int = 2, buf=None, seed: int | None = 0, workers: int = 1, **kwargs):
        if buf is not None or _in_vector_backend():
            raise _native.ArgumentError(
                "Pipelined is for the native backend only: use pufferlib.vector.make without a backend"
            )
        if _workers(workers) < 1:
            raise _native.ArgumentError("workers must be at least 1 in Pipelined mode, got 0")
        super().__init__(env_id, num_envs=num_envs, seed=seed, workers=workers, **kwargs)
        self._make_args["mode"] = "pipelined"
        # Refuses here what only a Pipelined pool can tell, as ConfigError.
        probe = make(env_id, **self._make_args)
        try:
            first, second = probe.halves
        finally:
            probe.close()
        if len(first) != len(second):
            raise _native.ArgumentError(
                f"Pipelined needs equal halves, these have {len(first)} and {len(second)} instances"
            )
        #: Instances per half; rows PufferLib sees per half (visible slots), and
        #: the engine's rows per half (every slot).
        self._half_instances = len(first)
        self._half_agents = self._half_instances * self._row_agents
        self._half_engine_rows = self._half_instances * self._agents
        self._half_masks = np.ones(self._half_agents, dtype=bool)
        # One buffer per half, so a half's rows stay valid until its own next send.
        self._half_flat: dict = {}
        #: With hidden slots, each half's gathered rows, allocated once here:
        #: `(obs, rewards, terminals, truncations, masks)` per half index.
        self._half_rows: dict = {}
        if self._visible is not None:
            shape = self.single_observation_space.shape
            for index in (0, 1):
                self._half_rows[index] = (
                    np.zeros((self._half_agents, *shape), dtype=self.single_observation_space.dtype),
                    np.zeros(self._half_agents, dtype=np.float32),
                    np.zeros(self._half_agents, dtype=bool),
                    np.zeros(self._half_agents, dtype=bool),
                    np.zeros((self._half_agents, self.single_action_space.n), dtype=bool),
                )
        self._queue: list = []
        self._current: int | None = None
        self.observations = self.rewards = self.terminals = self.truncations = None
        self.masks = self.agent_ids = None

    @property
    def agents_per_batch(self) -> int:
        return self._half_agents

    @property
    def agent_per_batch(self) -> int:
        return self._half_agents

    @property
    def action_masks(self) -> np.ndarray | None:
        """The current half's masks, ``(half agents, choices)``; ``None`` between
        ``send`` and ``recv``, as a worker is writing those rows, and after ``reset``."""
        return self._masks

    @property
    def terminal_observations(self) -> np.ndarray:
        raise _native.ArgumentError("terminal_observations is not available in Pipelined mode")

    def _flat(self, half) -> tuple:
        out = half.step
        if self._visible is None:
            n = self._half_agents
            return (
                self._obs_view(out.observations, n, self._half_flat_for(half.half)),
                out.rewards.reshape(n),
                out.agent_terminated.reshape(n),
                out.agent_truncated.reshape(n),
            )
        # Hidden slots: every engine row, then the visible slots, copied into
        # this half's own arrays (#558), as the Sync env does.
        h, shape = self._half_instances, self.single_observation_space.shape
        obs = self._obs_view(out.observations, self._half_engine_rows, self._half_flat_for(half.half))
        rows = self._half_rows[half.half]
        rows[0][:] = self._gather(obs.reshape((h, self._agents, *shape))).reshape(rows[0].shape)
        rows[1][:] = self._gather(out.rewards.reshape((h, self._agents))).reshape(-1)
        rows[2][:] = self._gather(out.agent_terminated.reshape((h, self._agents))).reshape(-1)
        rows[3][:] = self._gather(out.agent_truncated.reshape((h, self._agents))).reshape(-1)
        return rows[:4]

    def _half_masks_of(self, half) -> np.ndarray | None:
        """The half's action masks, `(half agents, choices)`, visible slots only."""
        masks = half.step.masks
        if masks is None:
            return None
        if self._visible is None:
            return masks.reshape((self._half_agents, -1))
        dst = self._half_rows[half.half][4]
        dst[:] = self._gather(masks.reshape((self._half_instances, self._agents, -1))).reshape(dst.shape)
        return dst

    def _half_flat_for(self, index):
        if index not in self._half_flat:
            self._half_flat[index] = self._new_flat(self._half_engine_rows)
        return self._half_flat[index]

    def reset(self, seed: int | None = None):
        """Queues both halves' first outputs for ``recv``; returns half 0's rows."""
        first = self._env is None
        base = None if seed is None and not first else mix_seed(self._constructor_seed, 0 if seed is None else seed)
        if first:
            self._env = make(self._env_id, seed=base, **self._make_args)
        halves = self._env.reset(base)
        if base is not None:
            self.instance_seed_base = base
        self._queue = list(halves)
        self._current = None
        self._masks = None
        self.observations = self.rewards = self.terminals = self.truncations = None
        self._episode_return[:] = 0.0
        self._episode_length[:] = 0
        self._tick = 0
        self._finished = [0.0, 0, 0]
        self.infos = []
        return self._flat(halves[0])[0], self.infos

    def step(self, actions):
        raise _native.ArgumentError("Pipelined has no step: use send and recv")

    def send(self, actions) -> None:
        """Starts the next step of the half last received."""
        self._require_reset("send")
        if self._current is None:
            raise _native.ArgumentError("send without a half: call recv first")
        if self._visible is not None:
            # The hidden slots' rows are never read: the engine computes them.
            full = np.zeros((self._half_instances, self._agents), dtype=np.int32)
            full[:, self._visible] = np.asarray(actions).reshape((self._half_instances, self._row_agents))
            actions = full
        self._env.send(self._current, actions)
        self._current = None
        self._masks = None
        self.observations = self.rewards = self.terminals = self.truncations = None

    def recv(self):
        """One half: ``(obs, rewards, terminals, truncations, infos, ids, masks)``."""
        self._require_reset("recv")
        if self._current is not None:
            raise _native.ArgumentError(f"recv before half {self._current} was sent")
        sent = not self._queue
        if sent:
            half = self._env.recv()
        else:
            # A queued half needs no native call, but the fork guard must still
            # refuse it in a child: any native entry does, before any lock.
            _ = self._env.is_poisoned
            half = self._queue.pop(0)
        self.observations, self.rewards, self.terminals, self.truncations = self._flat(half)
        self._current = half.half
        ids = half.ids
        self.agent_ids = np.arange(ids.start * self._row_agents, ids.stop * self._row_agents)
        self.masks = self._half_masks
        out = half.step
        masks = self._half_masks_of(half)
        if masks is not None:
            self._masks = masks
        self.infos = self._log_half(ids, out) if sent and self._log_interval is not None else []
        return (self.observations, self.rewards, self.terminals, self.truncations,
                self.infos, self.agent_ids, self.masks)  # fmt: skip

    def _log_half(self, ids, out) -> list:
        rows = slice(ids.start * self._row_agents, ids.stop * self._row_agents)
        self._episode_return[rows] += self.rewards
        self._episode_length[rows] += 1
        self._tick += 1
        ended = out.episode_terminated | out.episode_truncated
        if ended.any():
            done = np.zeros(self.num_agents, dtype=bool)
            done[rows] = np.repeat(ended, self._row_agents)
            self._finished[0] += float(self._episode_return[done].sum())
            self._finished[1] += int(self._episode_length[done].sum())
            self._finished[2] += int(done.sum())
            self._episode_return[done] = 0.0
            self._episode_length[done] = 0
        if self._tick % self._log_interval != 0 or self._finished[2] == 0:
            return []
        total, length, n = self._finished
        self._finished = [0.0, 0, 0]
        return [{"episode_return": total / n, "episode_length": length / n, "n": n}]


def make_puffer(env_id: str, num_envs: int = 1, **kwargs):
    """A creator of :class:`TesseraPufferEnv` for ``pufferlib.vector.make``,
    usable with spawn. ``num_envs`` is instances per env object; the keyword
    arguments are the env's, including an environment's config fields; ``mode`` is
    ``"sync"`` (default) or ``"pipelined"``."""
    _host.refuse(env_id, "tessera_rl.puffer")
    _check_env_id(env_id)
    mode = kwargs.pop("mode", "sync")
    if mode not in ("sync", "pipelined"):
        raise _native.ArgumentError(f"mode must be sync or pipelined here, not {mode!r}")
    cls = TesseraPufferEnv if mode == "sync" else TesseraPufferPipelinedEnv
    return partial(cls, env_id, num_envs=num_envs, **kwargs)
