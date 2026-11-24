# SPDX-License-Identifier: MIT OR Apache-2.0
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["FrameStack", "Normalize", "OneHot"]

@dataclass(frozen=True)
class FrameStack:
    leaf: str
    k: int

@dataclass(frozen=True)
class OneHot:
    leaf: str
    n: int

@dataclass(frozen=True)
class Normalize:
    leaf: str
    clip: float = ...

def native_spec(transforms: Iterable[FrameStack | OneHot | Normalize]) -> str: ...
