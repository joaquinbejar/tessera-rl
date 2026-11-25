# SPDX-License-Identifier: MIT OR Apache-2.0
import os
from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from tessera_rl._env import Env

__all__ = ["FORMAT", "LogitsLog", "load", "logits_log", "payload", "side_path", "width_of"]

FORMAT: str

def side_path(replay_path: str | os.PathLike[str]) -> str: ...
def width_of(env: Env) -> int: ...
def heads_of(env: Env) -> tuple[int, ...]: ...

class LogitsLog:
    replay_path: str
    shape: tuple[int, int, int]
    def __init__(self, replay_path: str | os.PathLike[str], env: Env) -> None: ...
    def step(self, logits: Any) -> None: ...

def logits_log(replay_path: str | os.PathLike[str], env: Env) -> AbstractContextManager[LogitsLog]: ...

class Refused(Exception): ...

def load(
    side: str | os.PathLike[str], replay_path: str | os.PathLike[str], env: Env, step_count: int
) -> dict[int, np.ndarray[Any, Any]]: ...
def payload(row: Any, mask_row: Any, heads: tuple[int, ...]) -> list[dict[str, Any]]: ...
