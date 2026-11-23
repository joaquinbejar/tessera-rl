# SPDX-License-Identifier: MIT OR Apache-2.0
"""tessera_rl's side of host protocol 1 (#272 step 5, SPEC 14.4.1 to 14.4.6
draft, ADR-0065): discovery, the HostEnv wrapper and its checks,
CapabilityError, set_fp_policy propagation and the Gymnasium and
PettingZoo mappings, over a pure-Python host object (fake_host.py). The
compiled test host wheel is covered by test_host_wheel_python.py
(``make py-test-host``)."""

import functools

import numpy as np
import pytest

import tessera_rl
from fake_host import FakeHost, install

FAKE = "fake/Counter-v1"


def factory(**overrides):
    return functools.partial(FakeHost, **overrides)


@pytest.fixture
def fake(monkeypatch):
    def use(**overrides):
        install(monkeypatch, {FAKE: [("fake-dist", factory(**overrides))]})

    use()
    return use


def _spy_hosts(monkeypatch, **overrides):
    """Installs FAKE with a factory that records every host object made."""
    made = []

    def spy(**kw):
        made.append(FakeHost(**{**kw, **overrides}))
        return made[-1]

    install(monkeypatch, {FAKE: [("fake-dist", spy)]})
    return made


# -- discovery (SPEC 14.4.1) ----------------------------------------------------


def test_host_make_returns_a_host_env(fake):
    env = tessera_rl.make(FAKE, num_envs=3, seed=5)
    assert isinstance(env, tessera_rl.HostEnv)
    assert (env.kind, env.mode, env.num_envs, env.agents) == ("host", "sync", 3, 2)
    assert (env.initial_agents, env.spawns, env.env_id, env.env_version) == (2, False, FAKE, 1)
    env.close()


def test_host_make_first_party_ids_come_first(fake, monkeypatch):
    install(monkeypatch, {"tessera/CartPole-v1": [("evil", factory())]})
    env = tessera_rl.make("tessera/CartPole-v1", num_envs=1)
    assert isinstance(env, tessera_rl.Env)
    env.close()


def test_host_make_refuses_the_reserved_namespace(monkeypatch):
    install(monkeypatch, {"tessera/Evil-v1": [("evil", factory())]})
    with pytest.raises(tessera_rl.ConfigError, match="reserved namespace"):
        tessera_rl.make("tessera/Evil-v1")


def test_host_make_refuses_an_id_two_distributions_declare(monkeypatch):
    install(monkeypatch, {FAKE: [("dist-a", factory()), ("dist-b", factory())]})
    with pytest.raises(tessera_rl.ConfigError, match=r"\['dist-a', 'dist-b'\]"):
        tessera_rl.make(FAKE)


def test_host_make_unknown_id_lists_every_known_id(fake):
    with pytest.raises(tessera_rl.ArgumentError) as e:
        tessera_rl.make("nobody/Nothing-v1")
    assert "tessera/CartPole-v1" in str(e.value) and FAKE in str(e.value)


@pytest.mark.parametrize("kwargs", [{"mode": "async", "batch_size": 1}, {"mode": "pipelined"}, {"gather": "indexed"}])
def test_host_make_split_phase_modes_are_capability_errors(fake, kwargs):
    with pytest.raises(tessera_rl.CapabilityError):
        tessera_rl.make(FAKE, num_envs=2, **kwargs)


@pytest.mark.parametrize("spin", [0, 2_000])
def test_host_make_spin_raises_capability_error(fake, spin):
    with pytest.raises(tessera_rl.CapabilityError, match="spin"):
        tessera_rl.make(FAKE, num_envs=2, workers=1, spin=spin)


def test_host_make_spin_none_is_dropped(fake):
    tessera_rl.make(FAKE, num_envs=2, spin=None).close()


def test_host_gym_vector_env_spin_raises_capability_error(fake):
    from tessera_rl import gym

    with pytest.raises(tessera_rl.CapabilityError, match="spin"):
        gym.TesseraVectorEnv(FAKE, num_envs=2, workers=1, spin=5)


def test_host_make_passes_pool_options_and_config(fake, monkeypatch):
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return FakeHost(**kw)

    install(monkeypatch, {FAKE: [("fake-dist", spy)]})
    tessera_rl.make(FAKE, num_envs=2, seed=9, auto_reset="same_step", max_episode_steps=5, workers=1, offset=3).close()
    assert seen == {
        "num_envs": 2, "seed": 9, "auto_reset": "same_step", "max_episode_steps": 5, "workers": 1, "offset": 3,
    }


def test_host_make_refuses_bad_outputs_before_the_factory(monkeypatch):
    made = _spy_hosts(monkeypatch)
    with pytest.raises(tessera_rl.ArgumentError, match="outputs"):
        tessera_rl.make(FAKE, outputs="bogus")
    assert made == []


def test_host_gym_refuses_bad_outputs_before_the_factory(monkeypatch):
    made = _spy_hosts(monkeypatch, agents=1)
    with pytest.raises(tessera_rl.ArgumentError, match="outputs"):
        gym.TesseraVectorEnv(FAKE, outputs="bogus")
    assert made == []


@pytest.mark.parametrize(
    "how, turns",
    [("make", "simultaneous"), ("gym", "simultaneous"), ("parallel_env", "simultaneous"),
     ("aec_env", "simultaneous"), ("aec_env", "sequential"), ("parallel_env", "mixed")],
)
def test_host_each_entry_point_builds_one_host_object(monkeypatch, how, turns):
    made = _spy_hosts(monkeypatch, turns=turns, **({"agents": 1} if how == "gym" else {}))
    if how == "make":
        env = tessera_rl.make(FAKE)
    elif how == "gym":
        env = pytest.importorskip("tessera_rl.gym", reason="needs gymnasium").TesseraVectorEnv(FAKE, num_envs=2)
    else:
        env = getattr(pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo"), how)(FAKE, seed=0)
    assert len(made) == 1
    env.close()


# -- the wrapper's checks (SPEC 14.4.2 to 14.4.4) -------------------------------


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"protocol": 2}, "host protocol 2"),
        ({"protocol": True}, "host protocol True"),
        ({"low": "999.0.0"}, "needs tessera_rl >= 999.0.0"),
        ({"below": "0.0.1"}, "< 0.0.1"),
        ({"low": "1.2"}, "not MAJOR.MINOR.PATCH"),
        ({"low": "1.2.3-rc1"}, "not MAJOR.MINOR.PATCH"),
        ({"drop_layout": "pending_reset"}, None),
        ({"drop_layout": "spawns"}, r"layout\(\) lacks \['spawns'\]"),
    ],
)
def test_host_wrap_checks_the_protocol(fake, overrides, match):
    fake(**overrides)
    if match is None:
        tessera_rl.make(FAKE).close()
        return
    with pytest.raises(tessera_rl.ConfigError, match=match):
        tessera_rl.make(FAKE)


def test_host_wrap_below_is_exclusive(fake):
    own = tessera_rl.__version__.split("+")[0].split("-")[0]
    fake(below=own)
    with pytest.raises(tessera_rl.ConfigError, match=f"< {own}"):
        tessera_rl.make(FAKE)


@pytest.mark.parametrize("patch", [{"kind": "tessera"}, {"outputs": "owned"}])
def test_host_wrap_refuses_another_kind_or_outputs(fake, patch):
    fake(layout_patch=patch)
    with pytest.raises(tessera_rl.ConfigError, match="kind 'host' and outputs 'borrowed'"):
        tessera_rl.make(FAKE)


def test_host_wrap_refuses_an_object_without_a_member():
    obj = FakeHost(num_envs=1)
    del obj.host_protocol
    with pytest.raises(tessera_rl.ConfigError, match="lacks \\['host_protocol'\\]"):
        tessera_rl.HostEnv(obj)


def test_host_outputs_missing_a_key_are_a_config_error(fake):
    fake(drop_output="pending_reset")
    env = tessera_rl.make(FAKE)
    with pytest.raises(tessera_rl.ConfigError, match=r"outputs lack \['pending_reset'\]"):
        env.reset()


@pytest.mark.parametrize("name", [
    "snapshot", "restore", "restore_all", "snapshot_all", "fork", "state_hash", "encode", "load",
    "import_record", "send", "recv",
])
def test_host_absent_methods_raise_capability_error(fake, name):
    env = tessera_rl.make(FAKE)
    with pytest.raises(tessera_rl.CapabilityError, match=name):
        getattr(env, name)(0)
    assert issubclass(tessera_rl.CapabilityError, tessera_rl.TesseraError)


# -- HostEnv ---------------------------------------------------------------------


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_host_env_reset_and_step_return_host_steps(fake, outputs):
    env = tessera_rl.make(FAKE, num_envs=3, seed=5, outputs=outputs)
    step = env.reset()
    assert isinstance(step, tessera_rl.HostStep)
    assert step._fields[:10] == tessera_rl.Step._fields
    assert step._fields[10:] == ("aborted", "abort_code", "pending_reset")
    assert step.observations["obs"][:, 0].tolist() == [[10, 5], [10, 6], [10, 7]]
    assert not step.pending_reset.any() and not step.aborted.any()
    assert step.abort_code.dtype == np.uint32 and step.slot_generations.shape == (3, 2)
    step = env.step(np.ones((3, 2), dtype=np.int64))
    assert step.rewards.tolist() == [[1.5, 1.5]] * 3
    assert step.observations["obs"].flags.writeable == (outputs == "owned")
    env.close()


@pytest.mark.parametrize("outputs", ["borrowed", "owned"])
def test_host_env_int32_ndarray_actions_reach_the_host_as_exact_bytes(fake, outputs):
    # Env.step passes an int32 C-contiguous array on as a memoryview (#152
    # E2); HostEnv turns it into the exact bytes protocol 1 takes, which
    # the fake host checks as strictly as a compiled one.
    a = tessera_rl.make(FAKE, num_envs=3, seed=5, outputs=outputs)
    b = tessera_rl.make(FAKE, num_envs=3, seed=5, outputs=outputs)
    a.reset()
    b.reset()
    for k in range(3):
        acts = np.full((3, 2), k % 3, dtype=np.int32)
        sa, sb = a.step(acts), b.step(acts.astype(np.int64))
        assert np.array_equal(sa.observations["obs"], sb.observations["obs"])
        assert np.array_equal(sa.rewards, sb.rewards)
    a.close()
    b.close()


def test_fake_host_step_takes_exact_bytes_only():
    host = FakeHost(num_envs=1, seed=1)
    host.reset(None)
    # A bytes subclass is the case that tells exact bytes from isinstance.
    subclass = type("B", (bytes,), {})(8)
    for other in (memoryview(bytes(8)), bytearray(8), np.zeros(2, dtype=np.int32), subclass):
        with pytest.raises(tessera_rl.ArgumentError, match="must be bytes"):
            host.step(other)
    host.step(bytes(8))


def test_host_env_reset_seed_list(fake):
    env = tessera_rl.make(FAKE, num_envs=2)
    step = env.reset([40, 41])
    assert step.observations["obs"][:, 0, 1].tolist() == [40, 41]
    with pytest.raises(TypeError):
        env.reset([1, True])
    env.close()


def test_host_env_abort_outputs(fake):
    fake(abort_step=2, abort_seed=1, abort_code=9)
    env = tessera_rl.make(FAKE, num_envs=3, seed=0)
    env.reset()
    env.step(np.zeros((3, 2)))
    step = env.step(np.zeros((3, 2)))
    assert step.aborted.tolist() == [False, True, False]
    assert step.abort_code.tolist() == [0, 9, 0]
    assert step.pending_reset.tolist() == [False, True, False]
    assert step.observations["obs"][1].tolist() == [[0, 0], [0, 0]]
    env.close()


def test_host_env_pending_reset_after_a_normal_end_is_not_aborted(fake):
    env = tessera_rl.make(FAKE, num_envs=2, auto_reset="disabled")
    env.reset()
    for _ in range(4):
        step = env.step(np.ones((2, 2), dtype=np.int64))
    assert step.episode_terminated.all()
    assert step.pending_reset.tolist() == [True, True]
    assert not step.aborted.any()
    env.close()


def test_host_native_reset_puts_seeds_given_in_ids_order_in_instance_order(fake):
    """``reset(seeds, ids)`` with ``ids`` shuffled. The adapters pass
    ``ids`` in ascending order, so this is reached at ``_HostNative``."""
    from tessera_rl import _host

    native = _host.native_for(FAKE, 3, 0, "borrowed", {})
    out = native.reset([12, 10, 11], [2, 0, 1])
    assert np.asarray(out["observations"][0])[:, 0, 1].tolist() == [10, 11, 12]
    native.close()


# -- set_fp_policy (SPEC 14.4.2) ------------------------------------------------


def test_host_set_fp_policy_reaches_every_live_host_object(fake):
    env = tessera_rl.make(FAKE)
    obj = env._native._obj
    assert obj.policies == ["canonicalize"], "set once when wrapped"
    try:
        tessera_rl.set_fp_policy("reject")
        assert obj.policies[-1] == "reject" and tessera_rl.get_fp_policy() == "reject"
    finally:
        tessera_rl.set_fp_policy("canonicalize")
    assert obj.policies[-1] == "canonicalize"
    env.close()


@pytest.mark.parametrize("adapter", ["gym", "pz"])
def test_host_set_fp_policy_reaches_the_adapters_host_objects(monkeypatch, adapter):
    if adapter == "gym":
        mod = pytest.importorskip("tessera_rl.gym", reason="needs gymnasium")
        made = _spy_hosts(monkeypatch, agents=1)
        env = mod.TesseraVectorEnv(FAKE, num_envs=2, seed=0)
    else:
        mod = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
        made = _spy_hosts(monkeypatch)
        env = mod.parallel_env(FAKE, seed=0)
    assert made and all(obj.policies[:1] == ["canonicalize"] for obj in made), "set once when wrapped"
    used = made[-1]
    try:
        tessera_rl.set_fp_policy("reject")
        assert used.policies[-1] == "reject"
    finally:
        tessera_rl.set_fp_policy("canonicalize")
    assert used.policies[-1] == "canonicalize"
    env.close()


# -- Gymnasium (SPEC 14.2 as ADR-0065 amends it) ----------------------------------

gym = pytest.importorskip("tessera_rl.gym", reason="needs gymnasium")


def test_host_gym_vector_env_carries_the_abort_infos(fake):
    fake(agents=1, abort_step=2, abort_seed=1)
    env = gym.TesseraVectorEnv(FAKE, num_envs=3, seed=0)
    assert env.metadata["tessera_kind"] == "host"
    obs, info = env.reset()
    for key in ("aborted", "abort_code", "pending_reset"):
        assert info["_" + key].all() and not info[key].any()
    env.step(np.zeros(3, dtype=np.int64))
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.int64))
    assert info["aborted"].tolist() == [False, True, False]
    assert info["abort_code"].tolist() == [0, 7, 0]
    assert info["pending_reset"].tolist() == [False, True, False]
    assert truncated.tolist() == [False, True, False] and not terminated.any()
    env.close()


def test_host_gym_same_step_abort_final_obs_is_the_zero_row(fake):
    fake(agents=1, abort_step=1, abort_seed=1)
    env = gym.TesseraVectorEnv(FAKE, num_envs=2, seed=0, autoreset_mode="same_step")
    env.reset()
    obs, _, _, truncated, info = env.step(np.zeros(2, dtype=np.int64))
    assert truncated.tolist() == [False, True]
    assert info["_final_obs"].tolist() == [False, True]
    assert info["final_obs"][1].tolist() == [0, 0]
    assert obs[1].tolist() == [10, 1], "the new episode's observation"
    env.close()


def test_host_gym_reset_reports_an_aborted_reset(fake):
    fake(agents=1, reset_aborts=1, abort_seed=0, abort_code=3)
    env = gym.TesseraVectorEnv(FAKE, num_envs=2, seed=0)
    _, info = env.reset()
    assert info["aborted"].tolist() == [True, False]
    assert info["abort_code"].tolist() == [103, 0]
    assert info["pending_reset"].tolist() == [True, False]
    env.close()


def test_host_gym_refuses_a_multi_agent_host_and_a_partial_reset(fake):
    with pytest.raises(tessera_rl.ArgumentError, match="PettingZoo"):
        gym.TesseraVectorEnv(FAKE, num_envs=1)
    fake(agents=1)
    env = gym.TesseraVectorEnv(FAKE, num_envs=2, autoreset_mode="disabled")
    env.reset()
    with pytest.raises(tessera_rl.CapabilityError):
        env.reset(options={"reset_mask": np.array([True, False])})
    env.close()


# -- PettingZoo, PufferLib, JAX ----------------------------------------------------


def test_host_pz_parallel_agents_empty_after_an_aborted_reset(fake):
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    fake(reset_aborts=1)
    env = pz.parallel_env(FAKE, seed=0)
    obs, infos = env.reset()
    assert env.agents == [] and obs == {}
    obs, infos = env.reset()
    assert env.agents == ["agent_0", "agent_1"]
    env.close()


@pytest.mark.parametrize("adapter", ["parallel_env", "aec_env"])
def test_host_pz_passes_the_callers_config(monkeypatch, adapter):
    """A host environment with a required config key works through
    PettingZoo, as through make() and Gymnasium."""
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    made = []

    def needs_size(*, size, **kw):
        made.append(FakeHost(**kw))
        made[-1].size = size
        return made[-1]

    install(monkeypatch, {FAKE: [("fake-dist", needs_size)]})
    env = getattr(pz, adapter)(FAKE, seed=0, size=5)
    assert [obj.size for obj in made] == [5]
    env.reset()
    env.close()


def test_host_pz_mixed_is_parallel_only(fake):
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    fake(turns="mixed")
    with pytest.raises(tessera_rl.CapabilityError, match="parallel_env"):
        pz.aec_env(FAKE)
    pz.parallel_env(FAKE).close()


def test_host_pz_sequential_is_aec_only(fake):
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    fake(turns="sequential")
    with pytest.raises(tessera_rl.ArgumentError, match="not 'parallel'"):
        pz.parallel_env(FAKE)


def test_host_adapters_report_tessera_kind_in_metadata(fake):
    """ADR-0062: both adapters say in ``metadata`` whether the environment
    is a host one."""
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    fake(agents=1)
    envs = (gym.TesseraVectorEnv(FAKE, num_envs=1), gym.TesseraEnv(FAKE), pz.parallel_env(FAKE), pz.aec_env(FAKE))
    for env in envs:
        assert env.metadata["tessera_kind"] == "host"
        env.close()
    envs = (
        gym.TesseraVectorEnv("tessera/CartPole-v1", num_envs=1), gym.TesseraEnv("tessera/CartPole-v1"),
        pz.parallel_env("tessera/Tag-v1"),
    )
    for env in envs:
        assert env.metadata["tessera_kind"] == "tessera"
        env.close()
    assert gym.TesseraEnv.metadata == {"render_modes": []}


@pytest.mark.parametrize("adapter", ["parallel_env", "aec_env"])
def test_host_pz_agents_are_the_initial_slots_not_the_capacity(fake, adapter):
    """ADR-0065 item 4: the slots alive after a reset are
    ``0..initial_agents``, below the ``agents`` capacity."""
    pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
    fake(agents=3, initial_agents=2)
    env = getattr(pz, adapter)(FAKE)
    assert env.possible_agents == ["agent_0", "agent_1"]
    env.reset(seed=1)
    assert env.agents == ["agent_0", "agent_1"]
    env.close()


def test_host_gym_observation_space_spans_an_integer_leafs_dtype(fake):
    """The protocol gives no bounds: a host leaf spans its dtype's range."""
    fake(agents=1)
    env = gym.TesseraVectorEnv(FAKE, num_envs=1)
    space = env.single_observation_space
    space = space["obs"] if hasattr(space, "spaces") else space
    info = np.iinfo(np.int32)
    assert (space.low == info.min).all() and (space.high == info.max).all()
    env.close()


@pytest.mark.parametrize("adapter", ["gym", "pz"])
def test_host_adapters_no_limit_is_none_for_the_factory(monkeypatch, adapter):
    """The adapters' "no limit" (``max_episode_steps`` 0) reaches the
    factory as ``None``."""
    if adapter == "gym":
        made = _spy_hosts(monkeypatch, agents=1)
        env = gym.TesseraVectorEnv(FAKE, num_envs=1, max_episode_steps=None)
    else:
        pz = pytest.importorskip("tessera_rl.pz", reason="needs pettingzoo")
        made = _spy_hosts(monkeypatch)
        env = pz.parallel_env(FAKE, max_episode_steps=0)
    assert [obj.limit for obj in made] == [None]
    env.close()


def test_host_puffer_and_jax_guard_refuses_host_ids(fake):
    """The guard tessera_rl.puffer and tessera_rl.jax call; it is reached
    here without PufferLib or JAX installed."""
    from tessera_rl import _host

    for module in ("tessera_rl.puffer", "tessera_rl.jax"):
        with pytest.raises(tessera_rl.CapabilityError, match=f"{module} does not support them"):
            _host.refuse(FAKE, module)
        _host.refuse("tessera/CartPole-v1", module)


@pytest.mark.parametrize("module, func", [("puffer", "make_puffer"), ("jax", "make")])
def test_host_puffer_and_jax_call_the_guard_first(module, func):
    """Each entry point calls the guard before anything else. Read from
    the source: importing the module needs PufferLib or JAX. The guard's
    behaviour through the import is tested where those are installed
    (test_jax.py, compat/pufferlib)."""
    import ast
    import pathlib

    source = pathlib.Path(tessera_rl.__file__).with_name(f"{module}.py").read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == func)
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert ast.unparse(body[0]) == f"_host.refuse(env_id, 'tessera_rl.{module}')"
