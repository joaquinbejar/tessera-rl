from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

def flat_dtype(dtypes: Sequence[npt.DTypeLike], what: str, adapter: str) -> np.dtype[Any]: ...
