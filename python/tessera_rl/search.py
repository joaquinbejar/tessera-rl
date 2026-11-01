# SPDX-License-Identifier: MIT OR Apache-2.0
"""PUCT search over a Tessera environment (ADR-0043), with a Python
evaluator.

``Search.run(evaluator)`` performs one search call. The evaluator is called
once per batch of leaves as ``evaluator(obs, legal, hashes)``:

- ``obs``: the observation of the player to move at each leaf, an array of
  shape ``(n, *obs_shape)`` (a dict of arrays for environments with several
  observation leaves);
- ``legal``: ``(n, actions)`` bool, that player's legal actions;
- ``hashes``: ``(n,)`` uint64, each leaf's record hash (for canned or
  table evaluators).

It returns ``(priors, values)``: priors ``(n, actions)`` over **all** actions,
each row summing to 1, and values ``(n,)`` in ``[-1, 1]`` from the view of
the player to move at the leaf. The engine validates them, drops the
illegal priors and renormalizes over the legal ones.

A call is all or nothing: if the evaluator raises or returns an invalid
response, the call raises ``EvaluatorError`` (with the evaluator's own
exception as its ``__cause__``), and the tree and the call counter are as
before the call.

The evaluator runs under the caller's floating-point controls (ADR-0041:
it is called through a callout); the search arithmetic runs under the
canonical ones. The interpreter stays attached for the whole call, and the
zero-allocation guarantees of the engine do not extend across the Python
boundary.
"""

from __future__ import annotations

import numpy as np

from tessera_rl import _native
from tessera_rl._env import make


class Search:
    """A PUCT search. ``env_id`` is the environment (``tessera/Connect4-v1``);
    options: ``capacity`` (nodes, default 4096), ``batch`` (children stepped
    per batch, default 8), ``eviction`` (``"strict"`` or ``"oldest"``),
    ``c_puct``, ``fpu``, ``virtual_loss``, ``batch_size``, ``simulations``,
    ``noise_alpha`` with ``noise_epsilon`` (root Dirichlet noise), ``seed``.
    The search starts at a fresh game; ``set_root`` moves it."""

    def __init__(self, env_id: str = "tessera/Connect4-v1", **options):
        self._native = _native.make_search(env_id=env_id, **options)
        layout = self._native.layout()
        self.actions: int = layout["actions"]
        self._obs_bytes: int = layout["obs_bytes"]
        probe = make(env_id)
        self._leaves = list(probe._leaves)
        probe.close()

    def set_root(self, record: bytes) -> None:
        """Replaces the tree by the root ``record`` (``Env.snapshot`` bytes)."""
        self._native.set_root(bytes(record))

    def run(self, evaluator) -> None:
        """One search call with ``evaluator`` (see the module docs)."""
        raised = []
        actions = self.actions
        leaves = self._leaves
        obs_bytes = self._obs_bytes

        def call(obs: bytes, legal: bytes, hashes: bytes, n: int):
            try:
                # Each row is one leaf's observation leaves, concatenated.
                rows = np.frombuffer(obs, dtype=np.uint8).reshape((n, obs_bytes))
                arrays = {}
                at = 0
                for name, dtype, shape in leaves:
                    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    leaf = np.ascontiguousarray(rows[:, at : at + size])
                    arrays[name] = leaf.view(dtype).reshape((n, *shape))
                    at += size
                o = next(iter(arrays.values())) if len(arrays) == 1 else arrays
                m = np.frombuffer(legal, dtype=np.bool_).reshape((n, actions))
                h = np.frombuffer(hashes, dtype="<u8")
                priors, values = evaluator(o, m, h)
                return (
                    np.ascontiguousarray(priors, dtype="<f8").tobytes(),
                    np.ascontiguousarray(values, dtype="<f8").tobytes(),
                )
            except BaseException as exc:
                raised.append(exc)
                raise

        try:
            self._native.run(call)
        except BaseException as exc:
            if raised and exc is raised[0]:
                raise _native.EvaluatorError(f"the evaluator raised {type(exc).__name__}: {exc}") from exc
            raise

    def root_visits(self) -> np.ndarray:
        """The visit count of every root action."""
        return np.asarray(self._native.root_visits(), dtype=np.uint32)

    def select_action(self, temperature: float = 0.0) -> int:
        """The most visited action at 0, else a draw proportional to
        ``N^(1/T)`` from the last call's search stream."""
        return self._native.select_action(float(temperature))

    @property
    def calls(self) -> int:
        """Completed search calls."""
        return self._native.calls()

    def tree_hash(self) -> int:
        """A digest of the whole tree (for tests of determinism)."""
        return self._native.tree_hash()
