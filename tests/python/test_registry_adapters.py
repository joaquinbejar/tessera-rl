# SPDX-License-Identifier: MIT OR Apache-2.0
"""Declared adapter capabilities and deprecation warnings (ADR-0080)."""

import warnings

import pytest

import tessera_rl
from tessera_rl import _envs, _native, pz

#: The PettingZoo interfaces before ADR-0080 (tessera_rl.pz._ENVS, literal),
#: plus Gather-v1 (ADR-0082, #366), which declared the same pair there:
#: the registry's declarations must equal them, id by id.
_BEFORE = {
    "tessera/Tag-v1": {"parallel", "aec"},
    "tessera/GridKeyDoor-v1": {"parallel", "aec"},
    "tessera/Gather-v1": {"parallel", "aec"},
    "tessera/Snake-v1": {"parallel", "aec"},
    "tessera/PaddleDuel-v1": {"parallel", "aec"},
    "tessera/Connect4-v1": {"aec"},
    "tessera/Leduc-v1": {"aec"},
    "tessera/Othello-v1": {"aec"},
    "tessera/Go9-v1": {"aec"},
    "tessera/RTS-v1": {"parallel"},
}


def test_envs_adapters_equal_previous_table():
    declared = {r["id"]: set(r["adapters"]) for r in tessera_rl.envs() if r["adapters"]}
    assert declared == _BEFORE


def test_envs_single_agent_ids_declare_none():
    by_id = {r["id"]: r for r in tessera_rl.envs()}
    for env_id in ("tessera/CartPole-v1", "tessera/Pendulum-v1", "tessera/Grid-v1"):
        assert by_id[env_id]["adapters"] == frozenset()


def test_envs_facts_match_registered_envs():
    facts = [(r["id"], r["version"], r["max_agents"], r["tier"], r["perfect_information"]) for r in tessera_rl.envs()]
    assert facts == _native.registered_envs()


def test_envs_no_first_party_deprecated_none():
    assert all(r["deprecated"] is None for r in tessera_rl.envs())


def test_pz_declarations_read_from_registry_equal():
    assert {k: set(v[1]) for k, v in pz._ENVS.items()} == _BEFORE


def test_pz_host_table_equals_rust_default():
    # Adapters::for_turns (Rust) is tested against the same table.
    assert pz._HOST_DECLARED == {
        "simultaneous": frozenset({"parallel", "aec"}),
        "sequential": frozenset({"aec"}),
        "mixed": frozenset({"parallel"}),
    }


def test_deprecation_warning_category_future_warning_subclass():
    assert issubclass(tessera_rl.TesseraDeprecationWarning, FutureWarning)


@pytest.fixture
def deprecated_cartpole(monkeypatch):
    records = dict(_envs._records())
    records["tessera/CartPole-v1"] = {
        **records["tessera/CartPole-v1"],
        "deprecated": {"since": "0.4.0", "note": "use tessera/CartPole-v2"},
    }
    monkeypatch.setattr(_envs, "_records", lambda: records)


def test_make_deprecated_id_warns_at_caller(deprecated_cartpole):
    with pytest.warns(tessera_rl.TesseraDeprecationWarning, match="since 0.4.0: use tessera/CartPole-v2") as rec:
        tessera_rl.make("tessera/CartPole-v1", outputs="owned", workers=0).close()
    assert len(rec) == 1
    assert rec[0].filename == __file__


def test_gym_make_deprecated_id_warns_at_caller(deprecated_cartpole):
    from tessera_rl import gym

    with pytest.warns(tessera_rl.TesseraDeprecationWarning) as rec:
        gym.make("tessera/CartPole-v1").close()
    assert [w.filename for w in rec if w.category is tessera_rl.TesseraDeprecationWarning] == [__file__]


def test_make_not_deprecated_id_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", tessera_rl.TesseraDeprecationWarning)
        tessera_rl.make("tessera/CartPole-v1", outputs="owned", workers=0).close()


def test_pz_deprecated_id_warns(monkeypatch):
    records = dict(_envs._records())
    records["tessera/Connect4-v1"] = {**records["tessera/Connect4-v1"], "deprecated": {"since": "0.4.0", "note": "x"}}
    monkeypatch.setattr(_envs, "_records", lambda: records)
    with pytest.warns(tessera_rl.TesseraDeprecationWarning, match="Connect4-v1 is deprecated"):
        pz.aec_env("tessera/Connect4-v1").close()
