# SPDX-License-Identifier: MIT OR Apache-2.0
"""Leaf bounds as NumPy arrays, exact for integer leaves.

``layout()`` gives each leaf's bounds as Python ints for integer leaves and
floats for float leaves. An integer bound such as ``2**63 - 1`` has no exact
``float64``, so integer arrays are built from the ints directly; float
bounds go through ``float64``, which keeps their bits.
"""

import numpy as np


def leaf_bound(values, shape, dtype) -> np.ndarray:
    """A leaf's bound per element in ``shape`` and ``dtype``: ``values``
    holds one value for every element, or one value per element.

    Integer and bool dtypes are built from Python ints, exact (bool as 0/1);
    float dtypes go through ``float64``."""
    dt = np.dtype(dtype)
    shape = tuple(int(s) for s in shape)
    if dt.kind in "iub":
        work = dt if dt.kind in "iu" else np.dtype(np.int64)
        arr = np.asarray([int(v) for v in values], dtype=work)
    else:
        arr = np.asarray(values, dtype=np.float64)
    arr = np.full(shape, arr.reshape(-1)[0], dtype=arr.dtype) if arr.size == 1 else arr.reshape(shape)
    return arr.astype(dt)
