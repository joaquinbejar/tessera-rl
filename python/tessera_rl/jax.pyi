# SPDX-License-Identifier: MIT OR Apache-2.0
from types import TracebackType
from typing import Any, NamedTuple

import jax

from tessera_rl import _env

__all__ = ["JAX_VERSION", "Env", "Step", "make"]

JAX_VERSION: str
UNPINNED_ENV: str

class Step(NamedTuple):
    observations: dict[str, jax.Array]
    rewards: jax.Array
    agent_terminated: jax.Array
    agent_truncated: jax.Array
    alive: jax.Array
    episode_terminated: jax.Array
    episode_truncated: jax.Array
    masks: jax.Array | None
    acting: jax.Array
    slot_generations: jax.Array

class Env:
    num_envs: int
    agents: int
    action_choices: int
    turns: str
    last_error: BaseException | None
    def __init__(self, pool: _env.Env) -> None: ...
    def reset(self, seed: int | jax.Array | None = None) -> Step: ...
    def step(self, actions: jax.Array | Any) -> Step: ...
    def clear_error(self) -> None: ...
    @property
    def is_poisoned(self) -> bool: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def __enter__(self) -> Env: ...
    def __exit__(
        self,
        *exc: type[BaseException] | BaseException | TracebackType | None,
    ) -> None: ...

def make(env_id: str, num_envs: int = 1, seed: int = 0, **kwargs: Any) -> Env: ...
