# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native environment API: a pool of instances with owned NumPy outputs."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from tessera_rl import _native


class Step(NamedTuple):
    """The outputs of one call, as arrays the caller owns.

    Per-agent arrays have shape ``(num_envs, agents, ...)``; episode flags
    have shape ``(num_envs,)``. ``masks`` is ``None`` for environments
    without action masks.
    """

    observations: dict[str, np.ndarray]
    rewards: np.ndarray
    agent_terminated: np.ndarray
    agent_truncated: np.ndarray
    alive: np.ndarray
    episode_terminated: np.ndarray
    episode_truncated: np.ndarray
    masks: np.ndarray | None


class Env:
    """A pool of ``num_envs`` instances of one environment.

    Outputs are copies (``owned`` mode): arrays returned by one call stay
    valid and unchanged after later calls. Calls on one ``Env`` must not
    overlap; a concurrent call raises ``ConcurrencyError``.
    """

    def __init__(self, native: _native.Env):
        self._native = native
        layout = native.layout()
        self.num_envs: int = layout["num_envs"]
        self.agents: int = layout["agents"]
        self.action_choices: int = layout["action_choices"]
        self._mask_len: int = layout["mask_len"]
        self._leaves = [(name or "obs", np.dtype(dtype), tuple(shape)) for name, dtype, shape in layout["leaves"]]

    def reset(self, seed: int | None = None) -> Step:
        """Resets every instance; a new seed base if ``seed`` is given."""
        self._native.reset() if seed is None else self._native.reset(seed)
        return self._outputs()

    def step(self, actions) -> Step:
        """One step. ``actions`` has one int per agent row, ``(num_envs, agents)``."""
        acts = np.ascontiguousarray(actions, dtype="<i4")
        if acts.size != self.num_envs * self.agents:
            raise _native.ActionError(f"expected {self.num_envs * self.agents} actions, got {acts.size}")
        self._native.step(acts.tobytes())
        return self._outputs()

    def _outputs(self) -> Step:
        out = self._native.outputs()
        n, a = self.num_envs, self.agents
        obs = {
            name: np.frombuffer(buf, dtype=dtype).reshape((n, a, *shape))
            for (name, dtype, shape), buf in zip(self._leaves, out["observations"], strict=True)
        }

        def per_agent(key, dtype):
            return np.frombuffer(out[key], dtype=dtype).reshape((n, a))

        masks = None
        if self._mask_len:
            masks = np.frombuffer(out["masks"], dtype=np.bool_).reshape((n, a, self._mask_len))
        return Step(
            observations=obs,
            rewards=per_agent("rewards", "<f4"),
            agent_terminated=per_agent("agent_terminated", np.bool_),
            agent_truncated=per_agent("agent_truncated", np.bool_),
            alive=per_agent("alive", np.bool_),
            episode_terminated=np.frombuffer(out["episode_terminated"], dtype=np.bool_),
            episode_truncated=np.frombuffer(out["episode_truncated"], dtype=np.bool_),
            masks=masks,
        )

    def snapshot(self, index: int) -> bytes:
        """The canonical record of instance ``index``."""
        return self._native.snapshot(index)

    def import_record(self, index: int, record: bytes) -> None:
        """Imports an (untrusted) record into instance ``index``."""
        self._native.import_record(index, bytes(record))

    def fork(self, src: int, dst: list[int]) -> None:
        """Copies instance ``src`` into every instance of ``dst``."""
        self._native.fork(src, list(dst))

    @property
    def is_poisoned(self) -> bool:
        return self._native.is_poisoned()

    def close(self) -> None:
        """Frees the pool; idempotent. Arrays already returned stay valid."""
        self._native.close()

    def __enter__(self) -> Env:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make(env_id: str, num_envs: int = 1, seed: int = 0, **kwargs) -> Env:
    """Creates a pool of ``num_envs`` instances of ``env_id``.

    Pool options: ``max_episode_steps``, ``auto_reset`` (``"next_step"``,
    ``"same_step"`` or ``"disabled"``), ``chunk_size``, ``workers``. Tag also
    takes its config fields (``width``, ``height``, ``pursuers``,
    ``evaders``, ``slip``, ``radius``).
    """
    makers = {
        "tessera/CartPole-v1": _native.make_cartpole,
        "tessera/Tag-v1": _native.make_tag,
    }
    try:
        maker = makers[env_id]
    except KeyError:
        raise _native.ArgumentError(f"unknown environment {env_id!r}; known: {sorted(makers)}") from None
    return Env(maker(num_envs=num_envs, seed=seed, **kwargs))
