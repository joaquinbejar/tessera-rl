# SPDX-License-Identifier: MIT OR Apache-2.0
"""Search over accumulated returns from Python (ADR-0096, #524 head B):
``value="returns"`` with ``discount`` on ``Search`` and ``SearchGroup``,
on the single-agent ``tessera/Gather-v1``, and the refusals."""

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl.search import Search, SearchGroup

GATHER = "tessera/Gather-v1"


def gather_root():
    env = t.make(GATHER, num_envs=1, seed=4, outputs="owned", auto_reset="disabled")
    env.reset()
    record = env.snapshot(0)
    env.close()
    return record


def uniform(obs, legal, hashes):
    n = legal.shape[0]
    return np.full((n, 4), 0.25), np.zeros(n)


def run(**options):
    options.setdefault("simulations", 64)
    s = Search(env_id=GATHER, **options)
    s.set_root(gather_root())
    s.run(uniform)
    return s


@pytest.mark.parametrize("root", ["puct", "gumbel"])
def test_search_returns_gather_runs_and_is_reproducible(root):
    a = run(value="returns", discount=0.5, root=root)
    b = run(value="returns", discount=0.5, root=root)
    assert a.root_visits().sum() == 64
    assert a.root_visits().tolist() == b.root_visits().tolist()
    assert a.tree_hash() == b.tree_hash()
    assert 0 <= a.select_action(0.0) < 4


def test_search_returns_differs_from_terminal_on_dense_rewards():
    # A zero evaluator: Terminal values come from terminal rewards only,
    # Returns sees Gather's +1 per item on the way.
    terminal = run(root="gumbel", value="terminal")
    returns = run(root="gumbel", value="returns", discount=1.0)
    assert terminal.improved_policy().tobytes() != returns.improved_policy().tobytes()


def test_search_terminal_value_is_the_default():
    a = run(root="gumbel")
    b = run(root="gumbel", value="terminal")
    assert a.improved_policy().tobytes() == b.improved_policy().tobytes()
    assert a.tree_hash() == b.tree_hash()


def test_search_returns_refusals_are_typed():
    with pytest.raises(t.ArgumentError):
        Search(env_id=GATHER, discount=0.5)
    with pytest.raises(t.ArgumentError):
        Search(env_id=GATHER, value="terminal", discount=0.5)
    with pytest.raises(t.ArgumentError):
        Search(env_id=GATHER, value="bogus")
    for bad in [1.5, -0.25, float("nan")]:
        with pytest.raises(t.SearchError):
            Search(env_id=GATHER, value="returns", discount=bad)
    # Connect Four alternates movers.
    with pytest.raises(t.SearchError):
        Search(value="returns")


def test_search_group_checks_the_value_mode_per_slot():
    # Groups build Connect Four, whose movers alternate: Returns is refused.
    with pytest.raises(t.SearchError):
        SearchGroup(2, simulations=32, slot_options=[{"value": "returns", "discount": 0.5}, {}])
    with pytest.raises(t.ArgumentError):
        SearchGroup(2, slot_options=[{"value": "bogus"}, {}])
    with pytest.raises(t.ArgumentError):
        SearchGroup(2, slot_options=[{"discount": 0.5}, {}])
    SearchGroup(2, simulations=32, value="terminal")
