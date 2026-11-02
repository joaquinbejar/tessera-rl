# SPDX-License-Identifier: MIT OR Apache-2.0
"""Connect Four games for the example: self-play with the search, and the
candidate against the ADR-0042 bot.

A game runs on a one-instance Tessera environment without auto-reset. At
every move the search's root is the environment's record
(``Env.snapshot``), so the tree starts fresh each move.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

import tessera_rl as t
from tessera_rl.search import Search

ENV_ID = "tessera/Connect4-v1"
COLS = 7


class Board:
    """The environment of one game, with the move list."""

    def __init__(self, seed: int):
        self.env = t.make(ENV_ID, seed=seed, auto_reset="disabled", outputs="owned")
        self.step = self.env.reset()
        self.moves: list[int] = []
        self.mover = 0

    def play(self, col: int) -> None:
        self.step = self.env.step(np.array([[col, col]]))
        self.moves.append(col)
        self.mover = 1 - self.mover

    @property
    def over(self) -> bool:
        return bool(self.step.episode_terminated[0] or self.step.episode_truncated[0])

    def planes(self) -> np.ndarray:
        """The mover's planes, ``(2, 6, 7)``."""
        return np.array(self.step.observations["board"][0, self.mover])

    def reward(self, agent: int) -> float:
        """The last step's reward of ``agent`` (+1 win, -1 loss, 0 draw)."""
        return float(self.step.rewards[0, agent])

    def record(self) -> bytes:
        return self.env.snapshot(0)

    def moves_text(self) -> str:
        """The moves as 1-based column digits (the connect4-ref format)."""
        return "".join(str(c + 1) for c in self.moves) or "-"

    def close(self) -> None:
        self.env.close()


@dataclass
class Timing:
    """Wall time of the search calls and of the network section in them."""

    search_ns: int = 0
    net_ns: int = 0
    moves: int = 0
    batches: list[int] = field(default_factory=list)


def search_move(search: Search, board: Board, evaluator, timing: Timing, temperature: float) -> tuple[int, list[int]]:
    """One move of the search from the board's position."""
    search.set_root(board.record())
    net0 = evaluator.net_ns
    nb = len(evaluator.batches)
    start = time.perf_counter_ns()
    search.run(evaluator)
    timing.search_ns += time.perf_counter_ns() - start
    timing.net_ns += evaluator.net_ns - net0
    timing.batches.extend(evaluator.batches[nb:])
    timing.moves += 1
    visits = search.root_visits().tolist()
    return search.select_action(temperature), visits


@dataclass
class SelfPlayGame:
    planes: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    moves: list[int]
    timing: Timing
    wall_ns: int


def self_play(search: Search, evaluator, seed: int, temperature_moves: int) -> SelfPlayGame:
    """One self-play game. The first ``temperature_moves`` moves are drawn
    proportionally to the visits (temperature 1), then the most visited."""
    start = time.perf_counter_ns()
    board = Board(seed)
    timing = Timing()
    planes, policies, movers = [], [], []
    try:
        while not board.over:
            temperature = 1.0 if len(board.moves) < temperature_moves else 0.0
            planes.append(board.planes())
            movers.append(board.mover)
            col, visits = search_move(search, board, evaluator, timing, temperature)
            total = sum(visits)
            policies.append(np.asarray(visits, dtype=np.float64) / total)
            board.play(col)
        rewards = (board.reward(0), board.reward(1))
        moves = list(board.moves)
    finally:
        board.close()
    return SelfPlayGame(
        planes=np.stack(planes),
        policy=np.stack(policies),
        value=np.array([rewards[m] for m in movers]),
        moves=moves,
        timing=timing,
        wall_ns=time.perf_counter_ns() - start,
    )


class BotClient:
    """The ADR-0042 bot through ``tessera-rl-connect4-ref serve``: one
    persistent process, one line per move."""

    def __init__(self, binary: str):
        self.proc = subprocess.Popen(
            [binary, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

    def choose(self, tie_seed: int, moves_text: str) -> int:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(f"{tie_seed:#x} {moves_text}\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().strip()
        if not reply.isdigit():
            raise RuntimeError(f"connect4-ref serve replied {reply!r} to {moves_text!r}")
        return int(reply)

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


def read_openings(path: str) -> list[str]:
    """The opening move strings (1-based digits) of an ADR-0042 file."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            out.append(line.split()[2])
    return out


@dataclass
class BotGame:
    opening: int
    candidate_first: bool
    result: float
    log: list[tuple[str, int, list[int]]]
    timing: Timing
    bot_ns: int


def versus_bot(
    search: Search, evaluator, bot: BotClient, opening_index: int, opening: str, candidate_first: bool, tie_seed: int, seed: int
) -> BotGame:
    """The candidate (greedy search, no noise) against the bot from an
    opening. ``candidate_first`` means the candidate moves next from the
    opening position. The result is from the candidate's view."""
    board = Board(seed)
    timing = Timing()
    bot_ns = 0
    log: list[tuple[str, int, list[int]]] = []
    try:
        for ch in opening:
            board.play(int(ch) - 1)
        candidate = board.mover if candidate_first else 1 - board.mover
        while not board.over:
            if board.mover == candidate:
                col, visits = search_move(search, board, evaluator, timing, 0.0)
                log.append(("c", col, visits))
            else:
                start = time.perf_counter_ns()
                col = bot.choose(tie_seed, board.moves_text())
                bot_ns += time.perf_counter_ns() - start
                log.append(("b", col, []))
            board.play(col)
        result = board.reward(candidate)
    finally:
        board.close()
    return BotGame(opening_index, candidate_first, result, log, timing, bot_ns)
