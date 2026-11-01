# SPDX-License-Identifier: MIT OR Apache-2.0
"""Connect Four from Python (P2-M1b): the turn structure, the acting mask,
the action mask and the observation planes, checked against a plain Python
reference game on random games, in borrowed and owned modes."""

import numpy as np
import pytest

import tessera_rl as t

ROWS, COLS = 6, 7


class Board:
    """A plain reference game: ``cells[row][col]`` is 0, 1 or 2, row 0 at the top."""

    def __init__(self):
        self.cells = np.zeros((ROWS, COLS), dtype=int)
        self.moves = 0

    def legal(self):
        return [c for c in range(COLS) if self.cells[0, c] == 0]

    def play(self, col):
        """Plays for the side to move; returns whether it completes four."""
        me = self.moves % 2 + 1
        row = max(r for r in range(ROWS) if self.cells[r, col] == 0)
        self.cells[row, col] = me
        self.moves += 1
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            run = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.cells[r, c] == me:
                    run += 1
                    r, c = r + sign * dr, c + sign * dc
            if run >= 4:
                return True
        return False

    def planes(self, agent):
        mine, theirs = agent + 1, 2 - agent
        return np.stack([self.cells == mine, self.cells == theirs]).astype(np.uint8)


def make(n=1, **kw):
    return t.make("tessera/Connect4-v1", num_envs=n, seed=5, **kw)


def actions_for(to_move, col):
    act = np.full((1, 2), 6)
    act[0, to_move] = col
    return act


def test_connect4_layout_is_sequential_with_seven_column_masks():
    env = make(3)
    assert env.turns == "sequential"
    assert env.agents == 2 and env.action_choices == 7
    s = env.reset()
    assert s.observations["board"].shape == (3, 2, 2, ROWS, COLS)
    assert s.observations["board"].dtype == np.uint8
    assert s.masks.shape == (3, 2, 7) and s.masks.all()
    assert s.acting.shape == (3, 2) and s.acting.dtype == np.bool_
    assert s.acting.tolist() == [[True, False]] * 3


def test_simultaneous_environments_report_every_alive_agent_acting():
    env = t.make("tessera/Tag-v1", num_envs=2, seed=3, pursuers=1, evaders=2)
    assert env.turns == "simultaneous"
    s = env.reset()
    assert np.array_equal(s.acting, s.alive)
    assert t.make("tessera/CartPole-v1").turns == "simultaneous"


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_connect4_random_games_match_the_python_reference(outputs):
    rng = np.random.default_rng(20261101)
    env = make(outputs=outputs, auto_reset="disabled")
    for _ in range(100):
        s = env.reset()
        board = Board()
        while True:
            me = board.moves % 2
            assert s.acting[0].tolist() == [me == 0, me == 1]
            legal = board.legal()
            assert np.flatnonzero(s.masks[0, me]).tolist() == legal
            for agent in (0, 1):
                assert np.array_equal(s.observations["board"][0, agent], board.planes(agent))
            col = int(rng.choice(legal))
            won = board.play(col)
            s = env.step(actions_for(me, col))
            over = won or board.moves == ROWS * COLS
            assert bool(s.episode_terminated[0]) == over
            if won:
                expected = [-1.0, -1.0]
                expected[me] = 1.0
                assert s.rewards[0].tolist() == expected
            else:
                assert s.rewards[0].tolist() == [0.0, 0.0]
            if over:
                assert not s.acting[0].any(), "nobody acts on a finished board"
                assert s.agent_terminated[0].all()
                break


def test_connect4_only_the_acting_agents_action_is_read():
    a, b = make(), make()
    a.reset()
    b.reset()
    sa = a.step(np.array([[3, 0]]))
    sb = b.step(np.array([[3, 6]]))
    assert np.array_equal(sa.observations["board"], sb.observations["board"])
    assert a.snapshot(0) == b.snapshot(0)


def test_connect4_full_column_is_masked_and_replaced_by_the_noop():
    env = make(auto_reset="disabled", outputs="owned")
    s = env.reset()
    for k in range(6):
        s = env.step(actions_for(k % 2, 3))
    assert s.masks[0, 0].tolist() == [True, True, True, False, True, True, True]
    # The illegal column becomes the centre-first no-op, column 2.
    s = env.step(actions_for(0, 3))
    assert s.observations["board"][0, 0, 0, ROWS - 1, 2] == 1


def test_connect4_snapshot_import_continues_the_turn_order():
    env = make(2, auto_reset="disabled", outputs="owned")
    env.reset()
    for col in (3, 3, 2):
        env.step(np.array([[col, col], [col, col]]))
    record = env.snapshot(0)
    other = make(2, auto_reset="disabled", outputs="owned")
    other.reset()
    other.import_record(1, record)
    s = other.step(np.array([[0, 0], [4, 4]]))
    assert s.acting[1].tolist() == [True, False]
    assert s.observations["board"][1, 1, 0, ROWS - 1, 4] == 1
