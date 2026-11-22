# SPDX-License-Identifier: MIT OR Apache-2.0
"""#322: the determinism recorder's Async run never asks for a batch that
cannot be filled. A stub Async pool refuses as the engine does
(``BatchUnreachable``, ADR-0047: ``recv`` never waits) and completes ids in
an order that mixes batches, the way load does."""

import importlib.util
import pathlib
import sys
from typing import NamedTuple

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("determinism_322", ROOT / "tools" / "pgo" / "determinism.py")
det = importlib.util.module_from_spec(_spec)
sys.modules["determinism_322"] = det
_spec.loader.exec_module(det)


class Step(NamedTuple):
    observations: dict
    rewards: np.ndarray
    masks: None


class Batch(NamedTuple):
    ids: np.ndarray
    step: Step


class Unreachable(Exception):
    pass


class StubAsync:
    """16 instances, batches of 4. An instance's output after its k-th step
    is (instance, k, its last action), so a trace depends only on each
    instance's own steps. ``order`` picks which in-flight ids complete next."""

    num_envs, agents, action_choices, batch = 16, 1, 2, 4
    action_bounds, action_nvec, mask_heads, entity_group_heads = None, (2,), (), 0

    def __init__(self, order):
        self.order = order
        self.steps = [0] * self.num_envs
        self.last = [0] * self.num_envs
        self.in_flight = []
        self.calls = 0

    def _step(self, ids):
        obs = np.array([[i, self.steps[i], self.last[i]] for i in ids], dtype=np.int64)
        return Batch(np.array(ids, dtype=np.uint32), Step({"obs": obs}, obs[:, 1].astype(np.float32), None))

    def reset(self):
        return self._step(list(range(self.num_envs)))

    def send(self, ids, actions):
        for i, a in zip(ids, actions):
            assert i not in self.in_flight, "an id sent twice"
            self.last[i] = int(a)
            self.in_flight.append(i)

    def recv(self):
        assert self.calls < 10_000, "the recorder did not finish"
        if len(self.in_flight) < self.batch:
            raise Unreachable(f"{len(self.in_flight)} in flight and 0 ready cannot fill a batch of {self.batch}")
        self.calls += 1
        done = self.order(list(self.in_flight), self.calls)[: self.batch]
        for i in done:
            self.in_flight.remove(i)
            self.steps[i] += 1
        return self._step(done)


def aligned(ids, call):
    return ids  # first sent, first done: batches stay the same groups of 4


def mixed(ids, call):
    # Oldest first, but instance 0 is held back two calls in three: it falls
    # behind its group, so groups break up (late, never starved).
    order = list(ids)
    if 0 in order and call % 3:
        order.remove(0)
        order.append(0)
    return order


@pytest.mark.parametrize("order", [aligned, mixed])
def test_async_recorder_records_every_instance_and_never_asks_an_unreachable_batch(order):
    env = StubAsync(order)
    got = det.run_async(env, steps=5)
    assert len(got["trace"]) == 16
    # Every instance reached at least its 5 recorded steps.
    assert min(env.steps) >= 5


def test_async_recorder_trace_does_not_depend_on_batch_order():
    a = det.run_async(StubAsync(aligned), steps=5)
    b = det.run_async(StubAsync(mixed), steps=5)
    assert a == b


def test_async_recorder_hashes_only_the_first_steps():
    # The trace equals hashing each instance's reset row and its first 5 steps.
    env = StubAsync(mixed)
    got = det.run_async(env, steps=5)
    ref = StubAsync(aligned)
    trace = det.Trace(16)
    trace.add(*ref.reset())
    for k in range(1, 6):
        for i in range(16):
            ref.steps[i] = k
            ref.last[i] = det.action(i, 0, k, 2, None)
        trace.add(*ref._step(list(range(16))))
    assert got["trace"] == trace.result()


def test_the_stub_refuses_like_the_engine():
    env = StubAsync(aligned)
    env.reset()
    env.send([0, 1, 2], [0, 0, 0])
    with pytest.raises(Unreachable, match="3 in flight and 0 ready cannot fill a batch of 4"):
        env.recv()
