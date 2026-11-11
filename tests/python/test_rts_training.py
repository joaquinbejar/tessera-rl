# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS reference PPO loop's parts (tools/rts_training.py), without any
training: no optimizer step is taken anywhere in this file."""

import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from rts_policy import Agent  # noqa: E402
from rts_training import Hyper, Seats, gae, ppo_loss, rollout, tensors, train  # noqa: E402

SMALL = Hyper(num_envs=4, num_steps=6, size=8, max_units=8, step_limit=40)


def agent(h=SMALL):
    torch.manual_seed(0)
    return Agent(11, h.size, 2, h.max_units, 6, 81, 4)


def test_rts_training_seats_put_half_the_envs_on_each_side():
    envs = Seats(SMALL, 3)
    try:
        assert [e.unwrapped.num_envs if hasattr(e, "unwrapped") else e.num_envs for e in envs.envs] == [2, 2]
        obs, masks = envs.reset(seed=3)
        assert obs["planes"].shape == (4, 11, 8, 8)
        assert [m.shape for m in masks] == [(4, 8, 6), (4, 8, 6, 81), (4, 8, 6, 4)]
    finally:
        envs.close()


def test_rts_training_rollout_shapes_and_legal_actions():
    h, a = SMALL, agent()
    envs = Seats(h, 5)
    try:
        obs, masks = envs.reset(seed=5)
        batch, state = rollout(a, envs, h, (obs, masks, np.zeros(h.num_envs, dtype=bool)))
        T, N, U = h.num_steps, h.num_envs, h.max_units
        assert batch["actions"].shape == (T, N, U, 3)
        assert batch["rewards"].shape == batch["values"].shape == batch["logprobs"].shape == (T, N)
        # Every sampled order is legal under the masks it was drawn from.
        tm, gm, pm, act = batch["type_mask"], batch["target_mask"], batch["product_mask"], batch["actions"]
        k = act[..., 0]
        ok_type = tm.gather(-1, k[..., None]).squeeze(-1) | ~tm.any(-1)
        assert ok_type.all()
        tg = gm.gather(-2, k[..., None, None].expand(*k.shape, 1, 81)).squeeze(-2)
        ok = tg.gather(-1, act[..., 1:2]).squeeze(-1) | ~tg.any(-1)
        assert ok.all()
    finally:
        envs.close()


def test_rts_training_gae_matches_a_hand_computation():
    r = torch.tensor([[1.0], [0.0], [2.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    d = torch.tensor([[0.0], [0.0], [1.0]])  # the obs at t=2 starts a new episode
    adv, ret = gae(r, v, d, torch.tensor([0.2]), torch.tensor([0.0]), 0.9, 0.8)
    # t=2: delta = 2 + 0.9*0.2 - 0.3 = 1.88
    # t=1: next is a new episode, so no bootstrap: delta = 0 - 0.4 = -0.4
    # t=0: delta = 1 + 0.9*0.4 - 0.5 = 0.86; adv = 0.86 + 0.72*(-0.4) = 0.572
    assert torch.allclose(adv.flatten(), torch.tensor([0.572, -0.4, 1.88]), atol=1e-6)
    assert torch.allclose(ret, adv + v)


def test_rts_training_loss_is_finite_with_finite_gradients_and_no_step():
    h, a = SMALL, agent()
    envs = Seats(h, 7)
    try:
        obs, masks = envs.reset(seed=7)
        batch, _ = rollout(a, envs, h, (obs, masks, np.zeros(h.num_envs, dtype=bool)))
    finally:
        envs.close()
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in batch.items()}
    flat["advantages"] = torch.randn(flat["rewards"].shape[0])
    flat["returns"] = flat["values"] + flat["advantages"]
    before = [p.detach().clone() for p in a.parameters()]
    loss, stats = ppo_loss(a, h, flat)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in a.parameters() if p.grad is not None)
    assert stats["entropy"] >= 0
    # No optimizer exists here: the weights are unchanged.
    assert all(torch.equal(b, p) for b, p in zip(before, a.parameters()))


def test_rts_training_train_refuses_without_the_frozen_protocol_hash():
    with pytest.raises(PermissionError):
        train(0, SMALL, authorization="", frozen_sha256="")
    with pytest.raises(PermissionError):
        train(0, SMALL, authorization="x", frozen_sha256="y")


def test_rts_training_tensors_keep_dtypes():
    obs = {"planes": np.zeros((2, 11, 8, 8), np.uint8), "globals": np.zeros((2, 2), np.uint16)}
    p, g = tensors(obs)
    assert p.dtype == torch.uint8 and g.dtype == torch.float32


def test_rts_training_rollout_bootstraps_truncated_episodes():
    # A step limit of 3 decisions truncates inside a rollout of 8, so the
    # final-observation bootstrap runs; rewards stay finite.
    h = Hyper(num_envs=2, num_steps=8, size=8, max_units=8, step_limit=3)
    a = agent(h)
    envs = Seats(h, 11)
    try:
        obs, masks = envs.reset(seed=11)
        batch, state = rollout(a, envs, h, (obs, masks, np.zeros(h.num_envs, dtype=bool)))
    finally:
        envs.close()
    assert batch["dones"][1:].any()
    assert torch.isfinite(batch["rewards"]).all()
