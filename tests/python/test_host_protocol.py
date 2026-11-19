# SPDX-License-Identifier: MIT OR Apache-2.0
"""Host protocol 1, the host object (#272 step 4, SPEC 14.4.2 to 14.4.5
draft, ADR-0065 item 5): the test host wheel of tests/host-wheel
(`make py-test-host`), over the test counter of tests/host-env.

- every 14.4.2 member, and none of the first-party-only methods;
- every 14.4.3 layout key, with ADR-0065's `initial_agents` and `spawns`;
- every 14.4.4 output with its dtype and shape, including ADR-0065's
  `aborted`, `abort_code` and `pending_reset`, and their values after an
  abort in a step and in a reset;
- views: alive after `close`, overwritten by the next call, generations;
- calls: seeds and actions checked, reentrancy, poison, the FP policy.
"""

import json
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import tessera_rl

_package = pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")
host = getattr(_package, "_tessera_host", _package)

N = 4
A = 2
MODES = ["next_step", "same_step", "disabled"]

PER_AGENT = {
    "rewards": "float32",
    "agent_terminated": "bool",
    "agent_truncated": "bool",
    "alive": "bool",
    "acting": "bool",
    "slot_generations": "uint32",
}
PER_ENV = {
    "episode_terminated": "bool",
    "episode_truncated": "bool",
    "episode_return": "float32",
    "episode_length": "float32",
    "aborted": "bool",
    "abort_code": "uint32",
    "pending_reset": "bool",
}


def make(**kw):
    kw.setdefault("num_envs", N)
    kw.setdefault("seed", 0)
    return host.make(**kw)


def actions(value=0, n=N):
    return np.full(n * A, value, dtype="<i4").tobytes()


def arr(view):
    return np.asarray(view)


def version_tuple(s):
    return tuple(int(p) for p in s.split("."))


def test_host_object_has_every_protocol_member():
    env = make()
    assert env.host_protocol == 1
    low, below = version_tuple(env.tessera_rl_min), version_tuple(env.tessera_rl_below)
    assert len(low) == 3 and len(below) == 3 and low < below
    assert low <= version_tuple(tessera_rl.__version__.split("+")[0].split("-")[0]) < below
    for name in (
        "layout", "reset", "step", "close", "is_poisoned",
        "_tessera_current_generation", "_tessera_set_fp_policy",
    ):
        assert callable(getattr(env, name)), name
    # SPEC 14.4.2: none of the first-party-only methods.
    for name in (
        "snapshot", "restore", "restore_all", "snapshot_all", "fork", "state_hash",
        "encode", "load", "import_record", "send", "recv",
    ):
        assert not hasattr(env, name), name
    # Read-only attributes.
    with pytest.raises(AttributeError):
        env.host_protocol = 2
    env.close()


@pytest.mark.parametrize("mode", MODES)
def test_host_layout_has_every_key(mode):
    env = make(auto_reset=mode)
    l = env.layout()
    assert l["num_envs"] == N
    assert l["agents"] == A
    assert l["initial_agents"] == A
    assert l["spawns"] is False
    assert l["action_choices"] == 3
    assert l["action_bounds"] is None
    assert l["action_nvec"] == [3]
    assert l["mask_len"] == 0 and l["mask_heads"] == []
    assert l["entity_group_heads"] == 0
    assert l["turns"] == "mixed"
    assert l["outputs"] == "borrowed"
    assert l["leaves"] == [("", "int32", [2])]
    assert l["env_id"] == "test/FactoryCounter-v1"
    assert l["env_version"] == 1
    assert isinstance(l["perfect_information"], bool)
    assert l["kind"] == "host"
    assert l["auto_reset"] == mode
    env.close()


def check_outputs(out, mode):
    assert set(PER_AGENT) | set(PER_ENV) | {"observations", "terminal_observations", "masks"} <= set(out)
    (obs,) = out["observations"]
    assert (arr(obs).dtype, arr(obs).shape) == (np.dtype("int32"), (N, A, 2))
    if mode == "same_step":
        (term,) = out["terminal_observations"]
        assert (arr(term).dtype, arr(term).shape) == (np.dtype("int32"), (N, A, 2))
    else:
        assert out["terminal_observations"] == []
    for key, dtype in PER_AGENT.items():
        a = arr(out[key])
        assert (a.dtype, a.shape) == (np.dtype(dtype), (N, A)), key
    for key, dtype in PER_ENV.items():
        a = arr(out[key])
        assert (a.dtype, a.shape) == (np.dtype(dtype), (N,)), key
    masks = arr(out["masks"])
    assert (masks.dtype, masks.shape) == (np.dtype("bool"), (N, A, 0))
    for view in [*out["observations"], *(out[k] for k in (*PER_AGENT, *PER_ENV, "masks"))]:
        assert not arr(view).flags.writeable
        assert view.__dlpack_device__() == (1, 0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("workers", [0, 2])
def test_host_outputs_have_every_key_dtype_and_shape(mode, workers):
    env = make(auto_reset=mode, workers=workers, offset=10)
    out = env.reset()
    check_outputs(out, mode)
    assert not arr(out["pending_reset"]).any()
    assert not arr(out["aborted"]).any() and not arr(out["abort_code"]).any()
    # Seeds 0..N: the observation is [step + offset, seed % 1000].
    assert arr(out["observations"][0])[:, 0].tolist() == [[10, i] for i in range(N)]
    out = env.step(actions(2))
    check_outputs(out, mode)
    assert arr(out["rewards"]).tolist() == [[2.5, 2.5]] * N
    assert arr(out["observations"][0])[:, 0, 0].tolist() == [11] * N
    assert arr(out["slot_generations"]).dtype == np.uint32
    env.close()


def test_host_pending_reset_at_construction_and_after_reset():
    env = make()
    # Every instance is pending its first reset: a step is refused.
    with pytest.raises(tessera_rl.PendingResetError):
        env.step(actions())
    out = env.reset()
    assert arr(out["pending_reset"]).tolist() == [False] * N
    env.close()


@pytest.mark.parametrize("mode", MODES)
def test_host_abort_in_step_sets_the_abort_outputs(mode):
    # Instance 1 (seed 1) aborts at its second step with code 7.
    env = make(auto_reset=mode, abort_step=2, abort_seed=1, abort_code=7, offset=10)
    env.reset()
    env.step(actions(1))
    out = env.step(actions(1))
    aborted = arr(out["aborted"]).tolist()
    assert aborted == [False, True, False, False]
    assert arr(out["abort_code"]).tolist() == [0, 7, 0, 0]
    assert arr(out["episode_truncated"])[1] and not arr(out["episode_terminated"])[1]
    assert arr(out["agent_truncated"])[1].tolist() == [True, True]
    assert arr(out["rewards"])[1].tolist() == [1.5, 1.5], "the step's rewards are kept"
    # No observe on the aborted state: zero rows. Under SameStep the reset
    # succeeded in the same call: the row is the new episode's, and the
    # terminal row is the aborted state's zeros.
    if mode == "same_step":
        assert arr(out["terminal_observations"][0])[1].tolist() == [[0, 0], [0, 0]]
        assert arr(out["observations"][0])[1].tolist() == [[10, 1], [10, 1]]
    else:
        assert arr(out["observations"][0])[1].tolist() == [[0, 0], [0, 0]]
    pending = arr(out["pending_reset"]).tolist()
    # SameStep resets in the same call (the reset succeeds); the others
    # leave the instance pending.
    assert pending == [False, mode != "same_step", False, False]
    # The next call clears the abort rows.
    if mode == "disabled":
        with pytest.raises(tessera_rl.PendingResetError):
            env.step(actions())
        out = env.reset()
    else:
        out = env.step(actions())
    assert not arr(out["aborted"]).any() and not arr(out["abort_code"]).any()
    env.close()


def test_host_abort_in_reset_leaves_the_instance_pending():
    # Instance 2's first reset aborts with code 7 + 100.
    env = make(reset_aborts=1, abort_seed=2, abort_code=7)
    out = env.reset()
    assert arr(out["aborted"]).tolist() == [False, False, True, False]
    assert arr(out["abort_code"]).tolist() == [0, 0, 107, 0]
    assert arr(out["pending_reset"]).tolist() == [False, False, True, False]
    assert arr(out["observations"][0])[2].tolist() == [[0, 0], [0, 0]]
    # NextStep: the next call retries the reset.
    out = env.step(actions())
    assert arr(out["pending_reset"]).tolist() == [False] * N
    assert not arr(out["aborted"]).any()
    env.close()


def test_host_views_outlive_close_and_are_overwritten_by_the_next_call():
    env = make(offset=10)
    out = env.reset()
    rewards = out["rewards"]
    obs = out["observations"][0]
    first = arr(obs).copy()
    gen = rewards._tessera_view_generation()
    assert rewards._tessera_current_generation() == gen == env._tessera_current_generation()
    env.step(actions(1))
    # The same memory, now showing the step: the borrow window ended.
    assert arr(obs)[:, 0, 0].tolist() == [11] * N
    assert not np.array_equal(arr(obs), first)
    assert rewards._tessera_current_generation() > gen
    assert rewards._tessera_view_generation() == gen
    last = arr(rewards).copy()
    env.close()
    # Views keep the memory after close (ADR-0003 rule 1).
    assert np.array_equal(arr(rewards), last)
    assert np.from_dlpack(rewards).tolist() == last.tolist()
    with pytest.raises(tessera_rl.ArgumentError):
        env.layout()
    with pytest.raises(tessera_rl.ArgumentError):
        env.step(actions())
    env.close()


@pytest.mark.parametrize("seed", [-1, True, 2**64, 1.5, "1", [0, 1], [0, 1, 2, -1], [0, 1, 2, True]])
def test_host_reset_refuses_bad_seeds(seed):
    env = make()
    with pytest.raises(tessera_rl.ArgumentError):
        env.reset(seed)
    env.close()


def test_host_reset_accepts_none_an_int_and_a_list():
    env = make()
    for seed in (None, 2**64 - 1, [5, 6, 7, 8]):
        env.reset(seed)
    out = env.reset([5, 6, 7, 8])
    assert arr(out["observations"][0])[:, 0, 1].tolist() == [5, 6, 7, 8]
    env.close()


def test_host_step_refuses_bad_actions_and_mutates_nothing():
    env = make(offset=10)
    env.reset()
    gen = env._tessera_current_generation()
    for bad in (b"\x00" * 3, actions(n=N - 1), actions(5)):
        with pytest.raises(tessera_rl.ActionError):
            env.step(bad)
    with pytest.raises(tessera_rl.ArgumentError):
        env.step("not bytes")
    assert env._tessera_current_generation() == gen
    out = env.step(actions())
    assert arr(out["observations"][0])[:, 0, 0].tolist() == [11] * N
    env.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_host_concurrent_call_raises_concurrency_error(workers):
    env = make(sleep_ms=400, workers=workers)
    env.reset()
    errors = []

    def slow():
        try:
            env.step(actions())
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.1)
    with pytest.raises(tessera_rl.ConcurrencyError):
        env.step(actions())
    with pytest.raises(tessera_rl.ConcurrencyError):
        env.layout()
    t.join()
    assert errors == []
    env.close()


def test_host_panic_poisons_and_a_full_reset_clears_it():
    env = make(panic_step=1)
    env.reset()
    with pytest.raises(tessera_rl.TesseraError):
        env.step(actions())
    assert env.is_poisoned()
    with pytest.raises(tessera_rl.PoisonedError):
        env.step(actions())
    env.reset()
    assert not env.is_poisoned()
    env.close()


def test_host_fp_policy_names_are_checked():
    env = make()
    for bad in ("strict", 1, None):
        with pytest.raises(tessera_rl.ArgumentError):
            env._tessera_set_fp_policy(bad)
    env._tessera_set_fp_policy("reject")
    assert host._fp_policy() == "reject"
    env._tessera_set_fp_policy("canonicalize")
    assert host._fp_policy() == "canonicalize"
    env.close()


def test_host_fp_reject_policy_raises_under_non_default_controls():
    child = pathlib.Path(__file__).with_name("fp_child.py")
    proc = subprocess.run([sys.executable, str(child), "host_reject"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["errors"] == {
        "method": "FloatingPointEnvironmentError",
        "getbuffer": "FloatingPointEnvironmentError",
    }
    assert r["round_restored"]
    # The policy is the host module's: tessera_rl's own is unchanged.
    assert r["first_party_policy"] == "canonicalize"
    assert r["after_canonicalize"] == "ok"


def test_host_views_are_read_only_through_every_export():
    env = make()
    out = env.reset()
    for view in [*out["observations"], out["rewards"], out["alive"], out["abort_code"]]:
        assert not np.asarray(view).flags.writeable
        assert memoryview(view).readonly
        assert not np.from_dlpack(view).flags.writeable
        with pytest.raises(ValueError):
            np.asarray(view)[...] = 0
    env.close()


def test_host_reset_int_seed_is_the_list_of_consecutive_seeds():
    # ADR-0007: reset(s) gives instance i the seed s + i.
    env = make()
    by_int = arr(env.reset(500)["observations"][0]).copy()
    by_list = arr(env.reset([500, 501, 502, 503])["observations"][0]).copy()
    assert np.array_equal(by_int, by_list)
    assert by_int[:, 0, 1].tolist() == [500, 501, 502, 503]
    # None keeps the seeds of the last reset, here 900 + i.
    env.reset(900)
    kept = arr(env.reset(None)["observations"][0]).copy()
    assert kept[:, 0, 1].tolist() == [900, 901, 902, 903]
    assert not np.array_equal(kept, by_int)
    env.close()


def test_host_alive_and_acting_are_separate_outputs():
    # Slot 1 is alive but never acts (the counter's idle slot).
    env = make(idle_slot=1)
    out = env.reset()
    assert arr(out["alive"]).tolist() == [[True, True]] * N
    assert arr(out["acting"]).tolist() == [[True, False]] * N
    out = env.step(actions())
    assert arr(out["alive"]).tolist() == [[True, True]] * N
    assert arr(out["acting"]).tolist() == [[True, False]] * N
    env.close()


@pytest.mark.parametrize("extra", [1, 2, 3])
def test_host_step_refuses_trailing_bytes_after_a_full_action_set(extra):
    env = make()
    env.reset()
    with pytest.raises(tessera_rl.ActionError):
        env.step(actions() + b"\x00" * extra)
    env.close()


def test_host_view_dlpack_arguments():
    # The rules shared with the first-party OutputView (Export::dlpack_call).
    env = make()
    view = env.reset()["rewards"]
    assert "dltensor_versioned" in repr(view.__dlpack__(max_version=(1, 0)))
    assert '"dltensor"' in repr(view.__dlpack__(max_version=(0, 8)))
    assert '"dltensor"' in repr(view.__dlpack__())
    view.__dlpack__(stream=-1, copy=False, dl_device=(1, 0))
    for kwargs in ({"copy": True}, {"stream": 1}, {"dl_device": (2, 0)}):
        with pytest.raises(BufferError):
            view.__dlpack__(**kwargs)
    with pytest.raises(TypeError):
        view.__dlpack__(1)
    with pytest.raises(TypeError):
        view.__dlpack__(unknown=1)
    env.close()
