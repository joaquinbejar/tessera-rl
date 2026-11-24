# SPDX-License-Identifier: MIT OR Apache-2.0
# pufferlib ships no stubs and is not in the lock: its classes are Any here.
from collections.abc import Sequence
from functools import partial
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt
import pufferlib

from tessera_rl._archive import Archive
from tessera_rl._env import make as make

SUPPORTED: frozenset[str]
MAX_WORKERS: int

def mix_seed(constructor_seed: int, reset_seed: int) -> int: ...
def flat_dtype(dtypes: Sequence[npt.DTypeLike]) -> np.dtype[Any]: ...


class TesseraPufferEnv(pufferlib.PufferEnv):
    instance_seed_base: int | None
    infos: list[dict[str, Any]]
    num_agents: int
    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        buf: Any = None,
        seed: int | None = 0,
        workers: int = 0,
        spin: int | None = None,
        log_interval: int | None = 128,
        auto_reset: str = "same_step",
        **env_kwargs: Any,
    ) -> None: ...
    @property
    def agents_per_batch(self) -> int: ...
    @property
    def action_masks(self) -> np.ndarray | None: ...
    @property
    def terminal_observations(self) -> np.ndarray: ...
    def set_reset_source(self, archive: Archive | None = None, weights: Sequence[int] | None = None) -> None: ...
    def reset_from(self, ids: Sequence[int], entries: Sequence[int]) -> Any: ...
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]: ...
    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]: ...
    def recv(self) -> Any: ...
    def close(self) -> None: ...

class TesseraPufferPipelinedEnv(TesseraPufferEnv):
    def __init__(
        self,
        env_id: str,
        num_envs: int = 2,
        buf: Any = None,
        seed: int | None = 0,
        workers: int = 1,
        **kwargs: Any,
    ) -> None: ...
    @property
    def agent_per_batch(self) -> int: ...
    def step(self, actions: npt.ArrayLike) -> NoReturn: ...
    def send(self, actions: npt.ArrayLike) -> None: ...

def make_puffer(env_id: str, num_envs: int = 1, **kwargs: Any) -> partial[TesseraPufferEnv]: ...
