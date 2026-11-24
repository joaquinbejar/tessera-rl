# SPDX-License-Identifier: MIT OR Apache-2.0
"""A pure-Python host object implementing host protocol 1 (SPEC 14.4
draft, ADR-0065): a counter, so every path of tessera_rl's host wrapper and
adapters runs without a compiled host wheel. Its outputs are read-only
NumPy arrays (the buffer protocol); real host views add DLPack and the
generations, which the test host wheel covers.

Per instance i (seed s + i): the observation of every agent is
``[step + offset, seed % 1000]``, the reward ``0.5 + action``, and the
episode ends (terminated) at step ``episode``. Knobs: ``abort_step`` (the
step at which ``abort_seed``'s instance, or every one with -1, aborts with
``abort_code``), ``reset_aborts`` (the first resets of that instance abort
with ``abort_code + 100``), ``turns``, ``agents`` and ``initial_agents`` (the slots alive after a reset,
``0..initial_agents``; ``agents`` by default). With ``refuses`` (host
protocol 2, ADR-0070), an agent whose action is 2 is refused with code
``30 + slot``: it gets reward -1 and the others step as usual.
"""

from __future__ import annotations

import numpy as np

import tessera_rl

OUTPUT_KEYS = (
    "rewards", "agent_terminated", "agent_truncated", "alive", "acting", "episode_terminated",
    "episode_truncated", "masks", "slot_generations", "episode_return", "episode_length",
    "aborted", "abort_code", "pending_reset",
)


class FakeHost:
    def __init__(self, *, num_envs, seed=0, auto_reset=None, max_episode_steps=None, workers=None,
                 cpu_set=None, agents=2, initial_agents=None, turns="simultaneous", episode=4, offset=10,
                 abort_step=0, abort_code=7, abort_seed=-1, reset_aborts=0, protocol=1,
                 low=None, below=None, drop_layout=None, drop_output=None, layout_patch=None,
                 refuses=False):
        version = tessera_rl.__version__.split("+")[0].split("-")[0]
        major, minor, _ = (int(p) for p in version.split("."))
        self.host_protocol = protocol
        self.tessera_rl_min = low or version
        self.tessera_rl_below = below or f"{major}.{minor + 1}.0"
        self.n, self.a = num_envs, agents
        self.initial = agents if initial_agents is None else initial_agents
        self.auto_reset = auto_reset or "next_step"
        self.limit = max_episode_steps
        self.turns, self.episode, self.offset = turns, episode, offset
        self.abort_step, self.abort_code, self.abort_seed = abort_step, abort_code, abort_seed
        self.reset_aborts = reset_aborts
        self.drop_layout, self.drop_output = drop_layout, drop_output
        self.layout_patch = layout_patch or {}
        self.refuses = refuses
        self.seeds = [seed + i for i in range(num_envs)]
        self.steps = [0] * num_envs
        self.resets = [0] * num_envs
        self.pending = [True] * num_envs
        self.closed = False
        self.policies = []
        self.generation = 0

    # -- protocol members ----------------------------------------------------

    def layout(self):
        layout = {
            "num_envs": self.n, "agents": self.a, "initial_agents": self.initial, "spawns": False,
            "action_choices": 3, "action_bounds": None, "action_nvec": [3], "mask_len": 0,
            "mask_heads": [], "entity_group_heads": 0, "turns": self.turns, "outputs": "borrowed",
            "leaves": [("", "int32", [2])], "terminal": self.auto_reset == "same_step",
            "env_id": "fake/Counter-v1", "env_version": 1, "perfect_information": True,
            "kind": "host", "auto_reset": self.auto_reset,
        }
        if self.refuses:
            # The value as given, so a test can pass a non-bool.
            layout["refuses_agents"] = self.refuses
        layout.update(self.layout_patch)
        layout.pop(self.drop_layout, None)
        return layout

    def is_poisoned(self):
        self._open()
        return False

    def close(self):
        self.closed = True

    def _tessera_current_generation(self):
        return self.generation

    def _tessera_set_fp_policy(self, policy):
        self.policies.append(policy)

    def _open(self):
        if self.closed:
            raise tessera_rl.ArgumentError("the environment is closed")

    def _targeted(self, i):
        return self.abort_seed < 0 or self.seeds[i] == self.abort_seed

    def _rows(self):
        n, a = self.n, self.a
        alive = np.zeros((n, a), bool)
        alive[:, : self.initial] = True
        return {
            "obs": np.zeros((n, a, 2), np.int32), "term_obs": np.zeros((n, a, 2), np.int32),
            "rewards": np.zeros((n, a), np.float32), "agent_terminated": np.zeros((n, a), bool),
            "agent_truncated": np.zeros((n, a), bool), "alive": alive,
            "acting": alive.copy(), "episode_terminated": np.zeros(n, bool),
            "episode_truncated": np.zeros(n, bool), "masks": np.zeros((n, a, 0), bool),
            "slot_generations": np.zeros((n, a), np.uint32), "episode_return": np.zeros(n, np.float32),
            "episode_length": np.zeros(n, np.float32), "aborted": np.zeros(n, bool),
            "abort_code": np.zeros(n, np.uint32), "pending_reset": np.zeros(n, bool),
            "refused": np.zeros((n, a), bool), "refuse_code": np.zeros((n, a), np.uint32),
        }

    def _reset_one(self, r, i, seed):
        if seed is not None:
            self.seeds[i] = seed
        self.steps[i] = 0
        self.resets[i] += 1
        if self.resets[i] <= self.reset_aborts and self._targeted(i):
            r["aborted"][i], r["abort_code"][i] = True, self.abort_code + 100
            r["agent_truncated"][i] = True
            r["episode_truncated"][i] = True
            r["acting"][i] = False
            r["obs"][i] = 0
            self.pending[i] = True
            return
        self.pending[i] = False
        r["obs"][i] = [self.offset, self.seeds[i] % 1000]

    def _out(self, r):
        self.generation += 1
        r["pending_reset"][:] = self.pending
        keys = OUTPUT_KEYS + (("refused", "refuse_code") if self.refuses else ())
        out = {k: v for k, v in r.items() if k in keys}
        out["observations"] = [r["obs"]]
        out["terminal_observations"] = [r["term_obs"]] if self.auto_reset == "same_step" else []
        out.pop(self.drop_output, None)
        for v in [*out.values()]:
            for arr in v if isinstance(v, list) else [v]:
                arr.flags.writeable = False
        return out

    def reset(self, seed=None):
        self._open()
        if seed is None:
            seeds = [None] * self.n
        elif isinstance(seed, list):
            if len(seed) != self.n:
                raise tessera_rl.ArgumentError("a seed list needs num_envs seeds")
            seeds = seed
        else:
            seeds = [seed + i for i in range(self.n)]
        r = self._rows()
        for i in range(self.n):
            self._reset_one(r, i, seeds[i])
        return self._out(r)

    def step(self, actions):
        self._open()
        if type(actions) is not bytes:  # protocol 1: exact bytes, as a compiled host refuses anything else
            raise tessera_rl.ArgumentError("step: actions must be bytes")
        acts = np.frombuffer(actions, dtype="<i4") if len(actions) % 4 == 0 else None
        if acts is None or acts.size != self.n * self.a or (acts < 0).any() or (acts > 2).any():
            raise tessera_rl.ActionError("bad actions")
        if any(self.pending) and self.auto_reset == "disabled":
            raise tessera_rl.PendingResetError("pending")
        acts = acts.reshape(self.n, self.a)
        r = self._rows()
        for i in range(self.n):
            if self.pending[i]:
                self._reset_one(r, i, None)
                continue
            self.steps[i] += 1
            r["rewards"][i] = 0.5 + acts[i]
            if self.refuses:
                for s in range(self.a):
                    if acts[i, s] == 2 and r["acting"][i, s]:
                        r["refused"][i, s], r["refuse_code"][i, s] = True, 30 + s
                        r["rewards"][i, s] = -1.0
            if self.steps[i] == self.abort_step and self._targeted(i):
                r["aborted"][i], r["abort_code"][i] = True, self.abort_code
                r["agent_truncated"][i] = True
                r["episode_truncated"][i] = True
                r["acting"][i] = False
                self.pending[i] = True
            else:
                r["obs"][i] = [self.steps[i] + self.offset, self.seeds[i] % 1000]
                if self.steps[i] >= self.episode:
                    r["episode_terminated"][i] = True
                    r["agent_terminated"][i] = True
                    self.pending[i] = True
            if self.pending[i] and self.auto_reset == "same_step":
                r["term_obs"][i] = r["obs"][i]
                self._reset_one(r, i, None)
        return self._out(r)


class EntryPoint:
    """A stand-in for an ``importlib.metadata`` entry point."""

    def __init__(self, factory):
        self.factory = factory

    def load(self):
        return self.factory


def install(monkeypatch, entries):
    """Makes ``entries`` ({id: [(distribution, factory), ...]}) the host
    entry points tessera_rl sees, for one test."""
    from tessera_rl import _host

    monkeypatch.setattr(
        _host, "_CACHE", {k: [(d, EntryPoint(f)) for d, f in v] for k, v in entries.items()}
    )
