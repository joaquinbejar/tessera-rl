# SPDX-License-Identifier: MIT OR Apache-2.0
"""Policies for test_eval.py (``tessera_rl.eval`` loads them by
``MODULE:NAME``): PaddleDuel actions 0 up, 1 stay, 2 down."""

import numpy as np


def stay(observations, info):
    leaf = next(iter(observations.values()))
    return np.ones(leaf.shape[0], dtype=np.int64)


def down(observations, info):
    leaf = next(iter(observations.values()))
    return np.full(leaf.shape[0], 2, dtype=np.int64)


def chase(observations, info):
    """Up or down by the parity of the first observation byte: varied play."""
    leaf = next(iter(observations.values()))
    first = int(np.asarray(leaf).reshape(leaf.shape[0], -1)[0, 0])
    return np.full(leaf.shape[0], 2 * (first % 2), dtype=np.int64)


def wrong_shape(observations, info):
    return np.zeros(7, dtype=np.int64)


def float_actions(observations, info):
    leaf = next(iter(observations.values()))
    return np.zeros(leaf.shape[0], dtype=np.float32)


def out_of_range(observations, info):
    leaf = next(iter(observations.values()))
    return np.full(leaf.shape[0], 9, dtype=np.int64)
