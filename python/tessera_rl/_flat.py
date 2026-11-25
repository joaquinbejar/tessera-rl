"""The flattening dtype rule shared by the adapters that present several
leaves as one array (the PufferLib observation, the PettingZoo state)."""

import numpy as np

from tessera_rl import _native


def flat_dtype(dtypes, what: str, adapter: str) -> np.dtype:
    """``numpy.result_type`` of ``dtypes``. ``ArgumentError`` when it is a float
    whose mantissa cannot hold every value of an integer leaf, the promotions
    that are not lossless; ``what`` names the leaves and ``adapter`` the
    adapter in the message."""
    leaf_dtypes = [np.dtype(d) for d in dtypes]
    target = np.result_type(*leaf_dtypes)
    if target.kind == "f":
        digits = np.finfo(target).nmant + 1
        wide = [str(d) for d in leaf_dtypes if d.kind in "iu" and d.itemsize * 8 - (d.kind == "i") > digits]
        if wide:
            raise _native.ArgumentError(
                f"{what} {[str(d) for d in leaf_dtypes]} promote to {target}, "
                f"which cannot hold every value of {wide}; {adapter} never rounds"
            )
    return target
