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

**Environments.** An allowlist: CartPole and Tag. Sequential environments are
refused: PufferLib has no turns, and its trainer would train on idle seats.
Observations with several leaves are refused too. Auto-reset is ``same_step``,
fixed. ``terminal_observations`` holds final observations, which PufferLib's
tuple has no place for. PufferLib passes no action masks to a policy, so in Tag
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
least 1. The constructor's probe of the halves starts real worker threads, closed
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
them. With adaptive state a half reads a context two cycles old (ADR-0047 section
6); CartPole and Tag have none. No speed gain has been measured; ADR-0006 is the
reason for this mode.

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

from tessera_rl import _native
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
SUPPORTED = frozenset({"tessera/CartPole-v1", "tessera/Tag-v1"})

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


def _bound(values: list, shape: list, dtype: str) -> np.ndarray:
    """A bound per element, from a list that may hold one value for all."""
    arr = np.asarray(values, dtype=np.float64)
    return np.broadcast_to(arr, shape).astype(dtype) if arr.size == 1 else arr.reshape(shape).astype(dtype)


def _single_leaf(leaves: list) -> tuple:
    """The only observation leaf; ``ArgumentError`` naming several."""
    if len(leaves) != 1:
        names = [leaf[0] for leaf in leaves]
        raise _native.ArgumentError(f"PufferLib needs one observation Box, this environment has leaves {names}")
    return leaves[0]


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


class TesseraPufferEnv(pufferlib.PufferEnv):
    """``num_envs`` instances as ``num_envs * agents`` PufferLib agents."""

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        buf=None,
        seed: int | None = 0,
        workers: int = 0,
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
        self._env_id = env_id
        self._log_interval = log_interval
        self._env = None
        self.instance_seed_base: int | None = None
        self.infos: list = []

        # Validates here, not in a worker; holds no thread and is closed at once.
        probe = make(env_id, **{**self._make_args, "workers": 0})
        try:
            layout = probe._native.layout()
            if layout["action_bounds"] is not None:
                raise _native.ArgumentError(f"{env_id} has continuous actions; the PufferLib adapter is discrete only")
            _, dtype, shape, low, high = _single_leaf(layout["leaves"])
            self._instances, self._agents = probe.num_envs, probe.agents
            choices = probe.action_choices
        finally:
            probe.close()
        self.num_agents = self._instances * self._agents
        self.single_observation_space = gymnasium.spaces.Box(
            low=_bound(low, shape, dtype),
            high=_bound(high, shape, dtype),
            dtype=np.dtype(dtype),
        )
        self.single_action_space = gymnasium.spaces.Discrete(choices)
        self._owned_buffers = buf is not None
        super().__init__(buf)
        self._episode_return = np.zeros(self.num_agents, dtype=np.float64)
        self._episode_length = np.zeros(self.num_agents, dtype=np.int64)
        self._tick = 0
        self._finished = [0.0, 0, 0]  # Sum of returns, sum of lengths, episodes.
        self._views: tuple | None = None
        self._masks: np.ndarray | None = None

    @property
    def agents_per_batch(self) -> int:
        """``pufferl.py`` reads this name with an RNN; ``PufferEnv`` lacks it."""
        return self.num_agents

    @property
    def action_masks(self) -> np.ndarray | None:
        """Action masks, ``(num_agents, choices)``; ``None`` without masks."""
        return self._masks

    @property
    def terminal_observations(self) -> np.ndarray:
        """Final observations of the episodes that ended in the last call."""
        self._require_reset("terminal_observations")
        leaf = self._env._native.outputs()["terminal_observations"][0]
        return np.asarray(leaf).reshape((self.num_agents, *self.single_observation_space.shape))

    def _require_reset(self, call: str) -> None:
        if self._env is None:
            raise _native.ArgumentError(f"{call} before the first reset")

    def _publish(self, out) -> None:
        """Takes the views of one call's outputs, once: the storage is stable."""
        n = self.num_agents
        obs = out.observations[next(iter(out.observations))]
        self._views = (
            obs.reshape((n, *self.single_observation_space.shape)),
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
        if self._owned_buffers:
            self.observations[:] = self._views[0]
            self.rewards[:] = self._views[1]
            self.terminals[:] = self._views[2]
            self.truncations[:] = self._views[3]

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
        self._env.step(actions)
        self._copy_out()
        self.infos = self._log() if self._log_interval is not None else []
        return self.observations, self.rewards, self.terminals, self.truncations, self.infos

    def _log(self) -> list:
        self._episode_return += self._views[1]
        self._episode_length += 1
        self._tick += 1
        ended = self._episode_ended[0] | self._episode_ended[1]
        if ended.any():
            done = np.repeat(ended, self._agents)
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
        self._half_agents = len(first) * self._agents
        self._half_masks = np.ones(self._half_agents, dtype=bool)
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
        n = self._half_agents
        out = half.step
        obs = out.observations[next(iter(out.observations))]
        return (
            obs.reshape((n, *self.single_observation_space.shape)),
            out.rewards.reshape(n),
            out.agent_terminated.reshape(n),
            out.agent_truncated.reshape(n),
        )

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
        self.agent_ids = np.arange(ids.start * self._agents, ids.stop * self._agents)
        self.masks = self._half_masks
        out = half.step
        if out.masks is not None:
            self._masks = out.masks.reshape((self._half_agents, -1))
        self.infos = self._log_half(ids, out) if sent and self._log_interval is not None else []
        return (self.observations, self.rewards, self.terminals, self.truncations,
                self.infos, self.agent_ids, self.masks)  # fmt: skip

    def _log_half(self, ids, out) -> list:
        rows = slice(ids.start * self._agents, ids.stop * self._agents)
        self._episode_return[rows] += self.rewards
        self._episode_length[rows] += 1
        self._tick += 1
        ended = out.episode_terminated | out.episode_truncated
        if ended.any():
            done = np.zeros(self.num_agents, dtype=bool)
            done[rows] = np.repeat(ended, self._agents)
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
    _check_env_id(env_id)
    mode = kwargs.pop("mode", "sync")
    if mode not in ("sync", "pipelined"):
        raise _native.ArgumentError(f"mode must be sync or pipelined here, not {mode!r}")
    cls = TesseraPufferEnv if mode == "sync" else TesseraPufferPipelinedEnv
    return partial(cls, env_id, num_envs=num_envs, **kwargs)
