# SPDX-License-Identifier: MIT OR Apache-2.0
"""The RTS reference PPO loop's parts (tools/rts_training.py). No training
run: no test reaches train's loop. The update test applies optimizer steps
to a synthetic batch of random tensors (no environment, no rollout), which
is not a run on the environment."""

import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import rts_training  # noqa: E402
from rts_policy import Agent  # noqa: E402
from rts_training import Hyper, Seats, gae, lr_at, next_state, ppo_loss, rollout, tensors, train, update  # noqa: E402

import tessera_rl.gym  # noqa: E402

SMALL = Hyper(num_envs=4, num_steps=6, size=8, max_units=8, step_limit=40)
T, W, P = 6, 81, 4


def agent(h=SMALL):
    torch.manual_seed(0)
    return Agent(11, h.size, 2, h.max_units, T, W, P)


# -- the protocol's numbers ---------------------------------------------------


def test_rts_training_hyper_defaults_are_the_protocols():
    h = Hyper()
    assert h.total_steps == 1_720_320 and h.total_steps % (h.num_envs * h.num_steps) == 0
    assert h.total_steps // (h.num_envs * h.num_steps) == 840
    assert (h.num_envs, h.num_steps) == (16, 128)
    assert (h.learning_rate, h.anneal_lr, h.gamma, h.gae_lambda) == (2.5e-4, True, 0.99, 0.95)
    assert (h.num_minibatches, h.update_epochs, h.clip_coef) == (4, 4, 0.2)
    assert (h.ent_coef, h.vf_coef, h.max_grad_norm) == (0.01, 0.5, 0.5)
    assert (h.size, h.max_units, h.decision_ticks, h.step_limit) == (16, 16, 4, 600)


# -- the guard: a regression fails at once, never trains -------------------------


class Reached(Exception):
    pass


@pytest.mark.parametrize("authorization,frozen", [("", ""), ("x", "y"), ("x", ""), ("", "y")])
def test_rts_training_train_refuses_before_any_environment(monkeypatch, authorization, frozen):
    # Anything past the guard raises Reached in milliseconds: a weakened
    # guard fails this test instead of starting a run.
    def tripwire(*_a, **_k):
        raise Reached

    monkeypatch.setattr(rts_training, "Seats", tripwire)
    monkeypatch.setattr(rts_training, "Agent", tripwire)
    with pytest.raises(PermissionError):
        train(0, Hyper(total_steps=0), authorization, frozen)


def test_rts_training_train_with_the_hash_passes_the_guard(monkeypatch):
    # The positive control of the guard, stopped at the first environment.
    def tripwire(*_a, **_k):
        raise Reached

    monkeypatch.setattr(rts_training, "Seats", tripwire)
    with pytest.raises(Reached):
        train(0, Hyper(total_steps=0), "h", "h")


# -- seats: sides, seeds and row order ------------------------------------------


def test_rts_training_seats_match_two_independent_pools_row_for_row():
    """Half 0 plays side Zero (seat_swap False) from seed s, half 1 side One
    (seat_swap True) from seed s + half; every row of obs, rewards, flags and
    masks is its own instance's. Compared with two envs built here, and with
    their state hashes, which tell the sides apart where the own-frame
    observations of a mirrored map cannot."""
    h, seed = SMALL, 9
    half = h.num_envs // 2
    kw = dict(size=h.size, max_units=h.max_units, decision_ticks=h.decision_ticks, step_limit=h.step_limit)
    ref = [
        tessera_rl.gym.make("tessera/RTS-vsRush-v1", num_envs=half, seed=seed, autoreset_mode=rts_training.SAME_STEP,
                            seat_swap=swap, **kw)
        for swap in (False, True)
    ]  # fmt: skip
    seats = Seats(h, seed)
    try:
        obs, masks = seats.reset(seed=seed)
        ro = [e.reset(seed=seed + k * half) for k, e in enumerate(ref)]
        hashes = lambda envs: [[e._native.state_hash(i) for i in range(half)] for e in envs]  # noqa: E731
        assert hashes(seats.envs) == hashes(ref)
        assert (obs["planes"] == np.concatenate([o["planes"] for o, _ in ro])).all()
        gen = np.random.default_rng(1)
        for _ in range(8):
            act = gen.integers(0, 2, size=(h.num_envs, h.max_units * 3)).astype(np.int64)
            got = seats.step(act)
            want = [e.step(act[k * half : (k + 1) * half]) for k, e in enumerate(ref)]
            assert hashes(seats.envs) == hashes(ref)
            assert (got[0]["planes"] == np.concatenate([w[0]["planes"] for w in want])).all()
            for i, j in ((1, 1), (2, 2), (3, 3)):
                assert (got[i] == np.concatenate([w[j] for w in want])).all()
            for k in range(3):
                assert (got[4][k] == np.concatenate([w[4]["action_masks"][k] for w in want]).astype(bool)).all()
        # Side: the second half is not side Zero. Its games, from the same
        # seeds and the same orders, diverge from a seat_swap False pool's
        # (the opponent plays the other side).
        zero = tessera_rl.gym.make("tessera/RTS-vsRush-v1", num_envs=half, seed=seed, seat_swap=False, **kw)
        zero.reset(seed=seed + half)
        fresh = Seats(h, seed)
        fresh.reset(seed=seed)
        differ = False
        for _ in range(8):
            act = gen.integers(0, 2, size=(half, h.max_units * 3)).astype(np.int64)
            fresh.envs[1].step(act)
            zero.step(act)
            differ |= hashes([fresh.envs[1]]) != hashes([zero])
        assert differ
        zero.close()
        fresh.close()
    finally:
        seats.close()
        for e in ref:
            e.close()


# -- rollout on a fake environment: bootstrap and log-probabilities ------------


class FakeSeats:
    """Two instances (one per half), all orders legal. At step 2 instance 0
    is truncated and instance 1 terminated, each with its own final
    observation; rewards are 1 and 2."""

    half = 1

    def __init__(self, h):
        self.h, self.t = h, 0

    def obs(self, fill):
        return {"planes": np.full((2, 11, self.h.size, self.h.size), fill, np.uint8),
                "globals": np.zeros((2, 2), np.uint16)}  # fmt: skip

    def masks(self):
        u = self.h.max_units
        return (np.ones((2, u, T), bool), np.ones((2, u, T, W), bool), np.ones((2, u, T, P), bool))

    def step(self, actions):
        self.t += 1
        term = np.array([False, self.t == 2])
        trunc = np.array([self.t == 2, False])
        infos = [{}, {}]
        if self.t == 2:
            infos[0] = {"final_obs": np.array([{"planes": np.full((11, self.h.size, self.h.size), 7, np.uint8),
                                                "globals": np.zeros(2, np.uint16)}], dtype=object)}  # fmt: skip
            infos[1] = {"final_obs": np.array([{"planes": np.full((11, self.h.size, self.h.size), 9, np.uint8),
                                                "globals": np.zeros(2, np.uint16)}], dtype=object)}  # fmt: skip
        return self.obs(self.t), np.array([1.0, 2.0], np.float32), term, trunc, self.masks(), infos


def test_rts_training_rollout_bootstraps_only_truncated_rows_with_their_final_obs():
    h = Hyper(num_envs=2, num_steps=3, size=8, max_units=8)
    a = agent(h)
    # A known value function: V(planes) = the mean plane value.
    a.value = lambda planes, globals_: planes.float().mean(dim=(1, 2, 3))
    env = FakeSeats(h)
    batch, _ = rollout(a, env, h, (env.obs(0), env.masks(), np.zeros(2, bool)))
    r = batch["rewards"]
    # Step index 1 is the env's second step: row 0 truncated (final planes 7),
    # row 1 terminated (no bootstrap).
    assert r[1, 0].item() == pytest.approx(1.0 + h.gamma * 7.0)
    assert r[1, 1].item() == pytest.approx(2.0)
    assert r[0].tolist() == [1.0, 2.0] and r[2].tolist() == [1.0, 2.0]
    assert batch["dones"][2].tolist() == [1.0, 1.0]


def test_rts_training_rollout_logprobs_are_those_of_the_stored_actions():
    h = SMALL
    a = agent(h)
    envs = Seats(h, 5)
    try:
        obs, masks = envs.reset(seed=5)
        batch, _ = rollout(a, envs, h, (obs, masks, np.zeros(h.num_envs, dtype=bool)))
    finally:
        envs.close()
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in batch.items()}
    with torch.no_grad():
        dist, value = a.distribution(flat["planes"], flat["globals"], flat["type_mask"], flat["target_mask"],
                                     flat["product_mask"])  # fmt: skip
        lp, _ = dist.log_prob_entropy(flat["actions"])
    assert torch.allclose(lp, flat["logprobs"], atol=1e-5)
    assert torch.allclose(value, flat["values"], atol=1e-6)


# -- GAE -----------------------------------------------------------------------


def test_rts_training_gae_matches_a_hand_computation():
    r = torch.tensor([[1.0], [0.0], [2.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    d = torch.tensor([[0.0], [0.0], [1.0]])  # the obs at t=2 starts a new episode
    adv, ret = gae(r, v, d, torch.tensor([0.2]), torch.tensor([0.0]), 0.9, 0.8)
    assert torch.allclose(adv.flatten(), torch.tensor([0.572, -0.4, 1.88]), atol=1e-6)
    assert torch.allclose(ret, adv + v)


def test_rts_training_gae_stops_at_a_done_next_observation_per_column():
    # Two columns; column 1's next observation starts a new episode, so its
    # last step does not bootstrap from next_value; column 0's does.
    r = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    v = torch.tensor([[0.0, 0.0], [0.5, 0.5]])
    d = torch.zeros(2, 2)
    adv, _ = gae(r, v, d, torch.tensor([2.0, 2.0]), torch.tensor([0.0, 1.0]), 0.5, 1.0)
    # col 0: t=1 delta = 1 + 0.5*2 - 0.5 = 1.5; t=0 delta = 1 + 0.25 - 0 = 1.25, adv = 1.25 + 0.5*1.5 = 2.0
    # col 1: t=1 delta = 1 - 0.5 = 0.5;           t=0 adv = 1.25 + 0.5*0.5 = 1.5
    assert torch.allclose(adv, torch.tensor([[2.0, 1.5], [1.5, 0.5]]))


# -- the loss, against an independent recomputation ------------------------------


def _synthetic(h, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = h.max_units
    tm = torch.rand(n, u, T, generator=g) < 0.6
    gm = torch.rand(n, u, T, W, generator=g) < 0.6
    pm = torch.rand(n, u, T, P, generator=g) < 0.6
    a = agent(h)
    planes = torch.randint(0, 5, (n, 11, h.size, h.size), generator=g, dtype=torch.uint8)
    globals_ = torch.rand(n, 2, generator=g) * 100
    with torch.no_grad():
        dist, value = a.distribution(planes, globals_, tm, gm, pm)
        actions = dist.sample()
        lp, _ = dist.log_prob_entropy(actions)
    return a, {
        "planes": planes, "globals": globals_, "type_mask": tm, "target_mask": gm, "product_mask": pm,
        "actions": actions, "logprobs": lp + 0.3 * torch.randn(n, generator=g),
        "values": value + 0.5 * torch.randn(n, generator=g),
        "advantages": torch.randn(n, generator=g), "returns": value + torch.randn(n, generator=g),
    }  # fmt: skip


def test_rts_training_loss_equals_an_independent_cleanrl_recomputation():
    h = Hyper(size=8, max_units=8)
    a, mb = _synthetic(h, 12)
    loss, stats = ppo_loss(a, h, mb)
    dist, value = a.distribution(mb["planes"], mb["globals"], mb["type_mask"], mb["target_mask"], mb["product_mask"])
    newlp, ent = dist.log_prob_entropy(mb["actions"])
    ratio = torch.exp(newlp - mb["logprobs"])
    adv = (mb["advantages"] - mb["advantages"].mean()) / (mb["advantages"].std() + 1e-8)
    pg1, pg2 = -adv * ratio, -adv * torch.clamp(ratio, 1 - h.clip_coef, 1 + h.clip_coef)
    pg = torch.max(pg1, pg2).mean()
    vu = (value - mb["returns"]) ** 2
    vc = mb["values"] + torch.clamp(value - mb["values"], -h.clip_coef, h.clip_coef)
    vl = 0.5 * torch.max(vu, (vc - mb["returns"]) ** 2).mean()
    want = pg - h.ent_coef * ent.mean() + vl * h.vf_coef
    assert loss.item() == pytest.approx(want.item(), rel=1e-5, abs=1e-6)
    assert stats["pg"] == pytest.approx(pg.item(), rel=1e-5, abs=1e-6)
    assert stats["v"] == pytest.approx(vl.item(), rel=1e-5, abs=1e-6)
    assert stats["entropy"] == pytest.approx(ent.mean().item(), rel=1e-5, abs=1e-6)
    # The value clip and the advantage normalisation matter on this batch.
    assert not torch.allclose(vu.mean(), torch.max(vu, (vc - mb["returns"]) ** 2).mean())
    assert abs(mb["advantages"].mean().item()) > 1e-3 or abs(mb["advantages"].std().item() - 1) > 1e-3


# -- update on a synthetic batch (no environment) --------------------------------


def test_rts_training_update_steps_clips_and_uses_each_index_once_per_epoch(monkeypatch):
    """A synthetic batch of random tensors, no environment and no rollout:
    the update changes the parameters, clips the gradient norm at
    max_grad_norm each minibatch, and uses every index once per epoch."""
    h = Hyper(num_envs=2, num_steps=8, size=8, max_units=8, num_minibatches=4, update_epochs=2)
    n = h.num_envs * h.num_steps
    a, mb = _synthetic(h, n)
    batch = {k: v.reshape(h.num_steps, h.num_envs, *v.shape[1:]) for k, v in mb.items()
             if k not in ("advantages", "returns")}  # fmt: skip
    batch["rewards"] = torch.randn(h.num_steps, h.num_envs)
    batch["dones"] = torch.zeros(h.num_steps, h.num_envs)
    batch["ids"] = torch.arange(n).reshape(h.num_steps, h.num_envs)
    seen, clips = [], []
    real_loss, real_clip = rts_training.ppo_loss, torch.nn.utils.clip_grad_norm_

    def spy_loss(ag, hh, m):
        seen.append(m["ids"].tolist())
        return real_loss(ag, hh, m)

    def spy_clip(params, max_norm, *args, **kw):
        clips.append(max_norm)
        return real_clip(params, max_norm, *args, **kw)

    monkeypatch.setattr(rts_training, "ppo_loss", spy_loss)
    monkeypatch.setattr(rts_training.nn.utils, "clip_grad_norm_", spy_clip)
    before = [p.detach().clone() for p in a.parameters()]
    opt = torch.optim.Adam(a.parameters(), lr=1e-3, eps=1e-5)
    update(a, opt, h, batch, torch.zeros(h.num_envs), torch.zeros(h.num_envs), np.random.default_rng(0))
    assert clips == [h.max_grad_norm] * (h.update_epochs * h.num_minibatches)
    per_epoch = h.num_minibatches
    for e in range(h.update_epochs):
        ids = sum(seen[e * per_epoch : (e + 1) * per_epoch], [])
        assert sorted(ids) == list(range(n))
    assert any(not torch.equal(b, p) for b, p in zip(before, a.parameters()))


def test_rts_training_lr_anneals_from_the_rate_to_one_update_short_of_zero():
    h = Hyper()
    updates = h.total_steps // (h.num_envs * h.num_steps)
    assert lr_at(0, updates, h) == h.learning_rate
    assert lr_at(updates - 1, updates, h) == pytest.approx(h.learning_rate / updates)
    assert lr_at(5, updates, Hyper(anneal_lr=False)) == h.learning_rate


def test_rts_training_next_state_hands_off_value_and_done():
    h = Hyper(size=8, max_units=8)
    a = agent(h)
    a.value = lambda planes, globals_: planes.float().mean(dim=(1, 2, 3))
    obs = {"planes": np.stack([np.full((11, 8, 8), 3, np.uint8), np.full((11, 8, 8), 5, np.uint8)]),
           "globals": np.zeros((2, 2), np.uint16)}  # fmt: skip
    v, d = next_state(a, (obs, None, np.array([False, True])))
    assert v.tolist() == [3.0, 5.0] and d.tolist() == [0.0, 1.0]


def test_rts_training_tensors_keep_dtypes():
    obs = {"planes": np.zeros((2, 11, 8, 8), np.uint8), "globals": np.zeros((2, 2), np.uint16)}
    p, g = tensors(obs)
    assert p.dtype == torch.uint8 and g.dtype == torch.float32


class _HalfEnv:
    """One half's vector env, every output marked with its half ``k``."""

    def __init__(self, k, u=2):
        self.k, self.u = k, u

    def step(self, actions):
        k, u = self.k, self.u
        obs = {"planes": np.full((1, 11, 2, 2), k, np.uint8), "globals": np.full((1, 2), k, np.uint16)}
        masks = (np.full((1, u, T), k == 1), np.full((1, u, T, W), k == 0), np.full((1, u, T, P), k == 1))
        info = {"action_mask": masks[0], "action_masks": masks}
        return obs, np.array([float(k + 1)]), np.array([k == 0]), np.array([k == 1]), info


def test_rts_training_seats_step_keeps_each_half_in_its_rows():
    # Every output of half k lands in row k: obs, reward, flags and the
    # three masks (a reversed concatenation of any of them fails here).
    s = object.__new__(Seats)
    s.envs, s.half = [_HalfEnv(0), _HalfEnv(1)], 1
    obs, reward, term, trunc, masks, _ = s.step(np.zeros((2, 6), np.int64))
    assert obs["planes"][:, 0, 0, 0].tolist() == [0, 1] and obs["globals"][:, 0].tolist() == [0, 1]
    assert reward.tolist() == [1.0, 2.0] and term.tolist() == [True, False] and trunc.tolist() == [False, True]
    assert masks[0][:, 0, 0].tolist() == [False, True]
    assert masks[1][:, 0, 0, 0].tolist() == [True, False]
    assert masks[2][:, 0, 0, 0].tolist() == [False, True]


# -- TR on #214 round 2: zero_grad, final_obs rows, train's own body -----------


def test_rts_training_update_equals_an_independent_loop_with_zero_grad():
    """The update on a synthetic batch (no environment) equals a loop written
    here: same rng order, zero_grad, backward, clip, step per minibatch. A
    missing zero_grad accumulates gradients and diverges."""
    h = Hyper(num_envs=2, num_steps=8, size=8, max_units=8, num_minibatches=4, update_epochs=2)
    n = h.num_envs * h.num_steps
    a1, mb = _synthetic(h, n, seed=3)
    a2 = agent(h)
    a2.load_state_dict(a1.state_dict())
    batch = {k: v.reshape(h.num_steps, h.num_envs, *v.shape[1:]) for k, v in mb.items()
             if k not in ("advantages", "returns")}  # fmt: skip
    batch["rewards"] = torch.randn(h.num_steps, h.num_envs, generator=torch.Generator().manual_seed(4))
    batch["dones"] = torch.zeros(h.num_steps, h.num_envs)
    nv, nd = torch.zeros(h.num_envs), torch.zeros(h.num_envs)
    o1 = torch.optim.Adam(a1.parameters(), lr=1e-3, eps=1e-5)
    update(a1, o1, h, batch, nv, nd, np.random.default_rng(7))
    # The independent loop.
    o2 = torch.optim.Adam(a2.parameters(), lr=1e-3, eps=1e-5)
    adv, ret = gae(batch["rewards"], batch["values"], batch["dones"], nv, nd, h.gamma, h.gae_lambda)
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in batch.items()}
    flat["advantages"], flat["returns"] = adv.reshape(-1), ret.reshape(-1)
    rng, size = np.random.default_rng(7), n
    for _ in range(h.update_epochs):
        order = rng.permutation(size)
        for start in range(0, size, size // h.num_minibatches):
            idx = torch.as_tensor(order[start : start + size // h.num_minibatches])
            loss, _ = ppo_loss(a2, h, {k: v[idx] for k, v in flat.items()})
            o2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(a2.parameters(), h.max_grad_norm)
            o2.step()
    for p1, p2 in zip(a1.parameters(), a2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


class FakeSeatsLateRow(FakeSeats):
    """Row 0 runs on; row 1 is truncated at step 2 with its own final
    observation (planes 9). A final observation read from the wrong row gets
    row 0's zeros."""

    def step(self, actions):
        self.t += 1
        trunc = np.array([False, self.t == 2])
        infos = [{}, {}]
        if self.t == 2:
            infos[1] = {"final_obs": np.array([{"planes": np.full((11, self.h.size, self.h.size), 9, np.uint8),
                                                "globals": np.zeros(2, np.uint16)}], dtype=object)}  # fmt: skip
        return self.obs(self.t), np.array([1.0, 2.0], np.float32), np.array([False, False]), trunc, self.masks(), infos


def test_rts_training_rollout_bootstraps_a_later_row_from_its_own_final_obs():
    h = Hyper(num_envs=2, num_steps=3, size=8, max_units=8)
    a = agent(h)
    a.value = lambda planes, globals_: planes.float().mean(dim=(1, 2, 3))
    env = FakeSeatsLateRow(h)
    batch, _ = rollout(a, env, h, (env.obs(0), env.masks(), np.zeros(2, bool)))
    assert batch["rewards"][1, 1].item() == pytest.approx(2.0 + h.gamma * 9.0)
    assert batch["rewards"][1, 0].item() == pytest.approx(1.0)


def test_rts_training_train_body_wires_its_parts(monkeypatch):
    """train's own loop with recorders for Seats, rollout, next_state and
    update: nothing real runs (no environment, no rollout, no optimizer
    step). Pins the call order, the learning rate each update sees
    (lr_at(0..2)), the hand-off from next_state to update, one rng passed to
    every update, one CPU thread, envs closed and the agent returned."""
    h = Hyper(num_envs=2, num_steps=4, size=8, max_units=8, total_steps=3 * 2 * 4 + 5)  # not a whole number of updates: 3 run
    calls, seen, opts = [], [], []

    class RecSeats:
        def __init__(self, hh, seed):
            calls.append(("seats", seed))

        def reset(self, seed):
            calls.append(("reset", seed))
            return {"planes": None}, None

        def close(self):
            calls.append(("close",))

    def rec_rollout(ag, envs, hh, state):
        calls.append(("rollout",))
        return {"k": len(calls)}, ("obs", "masks", np.array([True, False]))

    def rec_next_state(ag, state):
        calls.append(("next_state",))
        return torch.tensor([float(len(calls)), 0.0]), torch.tensor([1.0, 0.0])

    def rec_update(ag, opt, hh, batch, nv, nd, rng):
        calls.append(("update",))
        seen.append((opt.param_groups[0]["lr"], nv.tolist(), nd.tolist(), id(rng), torch.get_num_threads()))
        opts.append((type(opt), opt.defaults["eps"], opt.defaults["lr"]))
        return {}

    monkeypatch.setattr(rts_training, "Seats", RecSeats)
    monkeypatch.setattr(rts_training, "rollout", rec_rollout)
    monkeypatch.setattr(rts_training, "next_state", rec_next_state)
    monkeypatch.setattr(rts_training, "update", rec_update)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        got = train(5, h, "h", "h")
    finally:
        torch.set_num_threads(threads)
    assert isinstance(got, Agent)
    assert got.decisions_trained == 3 * 2 * 4
    assert calls == [("seats", 5), ("reset", 5)] + [("rollout",), ("next_state",), ("update",)] * 3 + [("close",)]
    assert [s[0] for s in seen] == [lr_at(u, 3, h) for u in range(3)]
    assert [s[1] for s in seen] == [[4.0, 0.0], [7.0, 0.0], [10.0, 0.0]]
    assert all(s[2] == [1.0, 0.0] for s in seen)
    assert len({s[3] for s in seen}) == 1
    assert all(s[4] == 1 for s in seen)
    # TR on #214: the optimizer and its eps are CleanRL's, frozen by the protocol.
    assert opts and all(o == (torch.optim.Adam, 1e-5, h.learning_rate) for o in opts)
