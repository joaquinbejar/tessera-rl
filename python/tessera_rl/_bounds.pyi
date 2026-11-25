# SPDX-License-Identifier: MIT OR Apache-2.0
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

def leaf_bound(values: Sequence[float], shape: Sequence[int], dtype: npt.DTypeLike) -> np.ndarray: ...
