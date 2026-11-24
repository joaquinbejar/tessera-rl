# SPDX-License-Identifier: MIT OR Apache-2.0
"""``tessera_rl.search.joint_log_prior`` (ADR-0086) against a brute-force
enumeration of the joint actions."""

import itertools
import math
import pathlib

import numpy as np
import pytest

from tessera_rl.search import joint_log_prior

# The search test environment's heads: kind (pass, add, bonus), amount
# under kind (add and bonus), bonus under kind (bonus only).
NVEC = (3, 4, 2)
PARENTS = [(), (0,), (0,)]


def picker_active(head, parents):
    kind = parents[0]
    return {1: kind != 0, 2: kind == 2}.get(head, True)


def logits(seed=3):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(3,)), rng.normal(size=(3, 4)), rng.normal(size=(3, 2))]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def brute(lg, legal=None):
    """Probability of each game action, put on its canonical joint."""
    p = np.zeros(math.prod(NVEC))
    for j, (k, a, b) in enumerate(itertools.product(*(range(n) for n in NVEC))):
        on_a, on_b = picker_active(1, [k, 0]), picker_active(2, [k, 0])
        if (not on_a and a) or (not on_b and b):
            continue
        q = softmax(lg[0])[k]
        q *= softmax(lg[1][k])[a] if on_a else 1.0
        q *= softmax(lg[2][k])[b] if on_b else 1.0
        p[j] = q
    if legal is not None:
        p = np.where(legal, p, 0.0)
    return p / p.sum()


def test_joint_log_prior_matches_brute_force_on_canonical_joints():
    lg = logits()
    got = np.exp(joint_log_prior(lg, NVEC, PARENTS, picker_active))
    np.testing.assert_allclose(got, brute(lg), rtol=1e-12, atol=0)
    assert got.sum() == pytest.approx(1.0, abs=1e-12)
    # 1 pass + 4 adds + 8 bonuses are canonical; the rest are -inf.
    assert np.count_nonzero(got) == 13


def test_joint_log_prior_without_masks_already_sums_to_one():
    # Marginalizing the inactive heads: the canonical joints hold all mass
    # before any renormalization.
    lg = logits(seed=9)
    assert brute(lg).sum() == pytest.approx(1.0)
    raw = np.zeros(24)
    for j in range(24):
        k, a, b = np.unravel_index(j, NVEC)
        if (k == 0 and a) or (k != 2 and b):
            continue
        raw[j] = softmax(lg[0])[k] * (softmax(lg[1][k])[a] if k else 1) * (softmax(lg[2][k])[b] if k == 2 else 1)
    assert raw.sum() == pytest.approx(1.0, abs=1e-12)


def test_joint_log_prior_renormalizes_over_the_legal_joints():
    lg = logits(seed=5)
    legal = np.ones(24, dtype=bool)
    legal[np.ravel_multi_index((1, 3, 0), NVEC)] = False  # add 3 masked
    legal[np.ravel_multi_index((0, 1, 0), NVEC)] = True  # non-canonical stays out
    got = np.exp(joint_log_prior(lg, NVEC, PARENTS, picker_active, legal))
    np.testing.assert_allclose(got, brute(lg, legal), rtol=1e-12, atol=0)
    assert got[np.ravel_multi_index((1, 3, 0), NVEC)] == 0.0
    assert got[np.ravel_multi_index((0, 1, 0), NVEC)] == 0.0


def test_joint_log_prior_with_flat_heads_is_the_product_of_softmaxes():
    rng = np.random.default_rng(1)
    lg = [rng.normal(size=(2,)), rng.normal(size=(3,))]
    got = np.exp(joint_log_prior(lg, (2, 3)))
    np.testing.assert_allclose(got, np.outer(softmax(lg[0]), softmax(lg[1])).ravel(), rtol=1e-12)


def test_joint_log_prior_refuses_bad_inputs():
    lg = logits()
    with pytest.raises(ValueError, match="one logits array"):
        joint_log_prior(lg[:2], NVEC, PARENTS)
    with pytest.raises(ValueError, match="expected"):
        joint_log_prior([lg[0], lg[1][0], lg[2]], NVEC, PARENTS)
    with pytest.raises(ValueError, match="earlier heads"):
        joint_log_prior(lg, NVEC, [(), (1,), (0,)])
    with pytest.raises(ValueError, match="no legal"):
        joint_log_prior(lg, NVEC, PARENTS, picker_active, np.zeros(24, dtype=bool))


# Cross-check with the native joints (review r1 of #454): the fixtures are
# written and checked by the Rust tests (crates/tessera-rl-search/tests/
# multi_discrete.rs and crates/tessera-rl-markets/tests/head_activity.rs),
# so the two sides cannot drift in radix order, the canonical rule or the
# reading of the activity tables.
FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "joint"


def _cases():
    for path in sorted(FIXTURES.glob("*.txt")):
        case = {}
        for line in path.read_text().splitlines():
            key, _, rest = line.partition(" ")
            if key == "case":
                case = {"name": rest}
            elif key == "nvec":
                case["nvec"] = [int(x) for x in rest.split()]
            elif key == "parents":
                case["parents"] = [() if p == "-" else tuple(int(x) for x in p.split(",")) for p in rest.split()]
            elif key == "active":
                case["active"] = [[b == "1" for b in t.split(",")] for t in rest.split()]
            elif key == "canonical":
                case["canonical"] = [int(x) for x in rest.split()]
            elif key == "legal":
                case["legal"] = [int(x) for x in rest.split()]
                yield case


CASES = list(_cases())


def test_joint_fixtures_exist():
    names = [c["name"] for c in CASES]
    assert any(n.startswith("picker_") for n in names), names
    assert any(n.startswith("lob_") for n in names), names


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_joint_log_prior_matches_the_native_joints(case):
    nvec, parents, tables = case["nvec"], case["parents"], case["active"]

    def active(head, values):
        ps = parents[head]
        second = nvec[ps[1]] if len(ps) > 1 else 1
        pair = (values[0] if ps else 0) * second + (values[1] if len(ps) > 1 else 0)
        return tables[head][pair]

    rng = np.random.default_rng(len(case["legal"]))
    logits = [rng.normal(size=tuple(nvec[p] for p in ps) + (n,)) for n, ps in zip(nvec, parents)]
    canonical = np.flatnonzero(np.isfinite(joint_log_prior(logits, nvec, parents, active)))
    assert canonical.tolist() == case["canonical"]
    legal = np.zeros(math.prod(nvec), dtype=bool)
    legal[case["legal"]] = True
    got = np.flatnonzero(np.isfinite(joint_log_prior(logits, nvec, parents, active, legal)))
    assert got.tolist() == case["legal"]
