# SPDX-License-Identifier: MIT OR Apache-2.0
from typing import Any

import gymnasium
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

from tessera_rl._env import native_kwargs as native_kwargs
from tessera_rl.masks import head_views as head_views

class TesseraVectorEnv(VectorEnv[Any, np.ndarray, np.ndarray]):
    construction_seed: int
    def __init__(
        self,
        env_id: str = "tessera/CartPole-v1",
        num_envs: int = 1,
        seed: int | None = None,
        autoreset_mode: AutoresetMode | str = ...,
        outputs: str = "owned",
        max_episode_steps: int | None = 500,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]: ...
    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: ...
    def render(self) -> np.ndarray | None: ...  # type: ignore[override]
    def close_extras(self, **kwargs: Any) -> None: ...

class TesseraEnv(gymnasium.Env[Any, Any]):
    construction_seed: int
    observation_space: spaces.Space[Any]
    action_space: spaces.Space[Any]
    def __init__(
        self,
        env_id: str = "tessera/CartPole-v1",
        seed: int | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None: ...
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...
    def render(self) -> np.ndarray | None: ...
    def close(self) -> None: ...

def make(
    env_id: str, num_envs: int = 1, seed: int | None = None, **kwargs: Any
) -> TesseraVectorEnv: ...
