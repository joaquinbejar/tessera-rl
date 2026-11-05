# SPDX-License-Identifier: MIT OR Apache-2.0
"""A view never reports fresh once a copy of it holds rewritten rows (TR's probe
on #89): the reader copies first and asks second, so the staleness signal must
be visible before a writer's first write."""

import threading

import pytest

import tessera_rl as t

# Async is smaller: gathering a token looks each id up linearly, so its
# whole-pool reset is quadratic in N (about 1 s per round at 2^16).
SIZES = {"sync": (1 << 16, 300), "pipelined": (1 << 16, 300), "async": (1 << 14, 100)}


def make(mode, n):
    kw = {"num_envs": n, "chunk_size": 64}
    if mode == "pipelined":
        kw["mode"] = "pipelined"
    elif mode == "async":
        kw.update(mode="async", batch_size=64, gather="indexed")
    return t.make("tessera/CartPole-v1", **kw)


def first_view(mode, out):
    if mode == "pipelined":
        return out[0].step.observations["obs"]
    if mode == "async":
        return out.step.observations["obs"]
    return out.observations["obs"]


def count_violations(mode):
    """Rounds with a copy that changed while the view read fresh, and rounds
    in which any change was seen at all."""
    n, rounds = SIZES[mode]
    env = make(mode, n)
    violations = changed = 0
    try:
        for round_ in range(rounds):
            view = first_view(mode, env.reset(seed=2 * round_ + 1))
            base = view.tobytes()
            writer = threading.Thread(target=env.reset, kwargs={"seed": 2 * round_ + 2})
            writer.start()
            violated = saw = False
            while writer.is_alive():
                copy = view.tobytes()
                if copy != base:
                    saw = True
                    violated |= not t.is_stale(view)
            writer.join()
            violations += violated
            changed += saw
    finally:
        env.close()
    return violations, changed


@pytest.mark.parametrize("mode", ["sync", "pipelined", "async"])
def test_view_copy_with_rewritten_rows_never_reads_fresh(mode):
    violations, changed = count_violations(mode)
    print(f"{mode}: {violations}/{SIZES[mode][1]} violations, {changed} rounds saw a change")
    assert changed, "the reader never observed a rewrite"
    assert violations == 0
