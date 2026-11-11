"""Per-head views of an action mask, in the shapes of SPEC 11.3.

An agent's mask row is one flat run of bools: each discrete head's block,
in head order. An environment with entity groups (ADR-0051) repeats the
same heads once per entity, so for the RTS the row is, per unit slot,
``[type (6) | target (6 x 81) | product (6 x 4)]``, 516 values. The views
here regroup it without a copy:

- a head with no parent: ``[N, A, entities, choices]``;
- a head with parents: ``[N, A, entities, *parent sizes, choices]``.

For the RTS: ``type [N, A, U, 6]``, ``target [N, A, U, 6, 81]``,
``product [N, A, U, 6, 4]``.

The three tensors are interleaved per entity, so the views are strided, not
contiguous. They are read-only and share memory with the mask array they
come from: a view of a ``borrowed`` mask follows that mask's borrow window
(``is_stale`` tracks it the same way), and a view of an ``owned`` mask is
valid as long as the array is.
"""

from __future__ import annotations

import numpy as np


def head_views(masks: np.ndarray, mask_heads, entity_group_heads: int) -> tuple[np.ndarray, ...]:
    """The views of ``masks`` (``[N, A, mask_len]``), one per head of an
    entity group (or per head, without groups), as described above.

    ``mask_heads`` and ``entity_group_heads`` are the ``Env`` attributes of
    the same names. Each view is a reshape and slice of ``masks``, so NumPy
    links it back to ``masks`` (``is_stale`` follows it).
    """
    if masks.ndim != 3:
        raise ValueError(f"masks have shape [N, A, mask_len], got {masks.shape}")
    heads = list(mask_heads)
    per = entity_group_heads or len(heads)
    if per == 0 or len(heads) % per:
        raise ValueError("the mask heads do not split into entity groups")
    entities = len(heads) // per
    n, a, row = masks.shape
    if row % entities:
        raise ValueError("the mask row does not split into entity blocks")
    period = row // entities
    for k, (offset, parents, size) in enumerate(heads):
        first = heads[k % per]
        if offset != first[0] + (k // per) * period or (parents, size) != first[1:]:
            raise ValueError("the entity groups are not laid out uniformly")
    blocks = masks.reshape(n, a, entities, period)
    out = []
    for offset, parents, size in heads[:per]:
        width = int(np.prod(parents, dtype=np.int64)) * size
        view = blocks[..., offset : offset + width].reshape(n, a, entities, *parents, size)
        view.flags.writeable = False
        out.append(view)
    return tuple(out)
