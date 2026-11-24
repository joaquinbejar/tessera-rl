# SPDX-License-Identifier: MIT OR Apache-2.0
from collections.abc import Sequence

import numpy as np

def head_views(
    masks: np.ndarray,
    mask_heads: Sequence[tuple[int, tuple[int, ...], int]],
    entity_group_heads: int,
) -> tuple[np.ndarray, ...]: ...
