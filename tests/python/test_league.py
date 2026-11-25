# SPDX-License-Identifier: MIT OR Apache-2.0
"""Leagues from Python (ADR-0088 section 10, #384 head E3): ``tessera_rl.League``,
``make(..., policy_map={1: "opponent"}, league=, role=)``, ``Env.assignment``,
the shared barrier (order and thread independent, refusing an early barrier,
one round of lag with one pool as with many), save, load and export, the
adapters, and the refusals."""

import json
import threading

import numpy as np
import pytest

import tessera_rl
from tessera_rl import gym, pz

DUEL = "tessera/PaddleDuel-v1"
POP = ["heuristic", ("external", 7)]

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
)


def _league(**kw):
    return tessera_rl.League(POP, [{"matchmaking": "pfsp"}, {"matchmaking": "uniform"}], rate_population=True, **kw)


def _pool(league, role=0, seed=1):
    return tessera_rl.make(
        DUEL,
        num_envs=8,
        seed=seed,
        workers=0,
        agents=2,
        max_episode_steps=60,
        policy_map={1: "opponent"},
        league=league,
        role=role,
    )


def _run(env, t):
    acts = np.zeros((env.num_envs, env.agents), dtype=np.int32)
    for s in range(60):
        acts[:] = (t * 5 + s + np.arange(env.num_envs * env.agents).reshape(env.num_envs, env.agents)) % 3
        env.step(acts)


def _rounds(order_of, threads=False, rounds=6):
    league = _league()
    pools = [_pool(league, 0, 1), _pool(league, 1, 2), _pool(league, 0, 3)]
    for p in pools:
        p.reset()
    for r in range(rounds):
        for i, p in enumerate(pools):
            _run(p, r * 3 + i)
        if threads:
            ts = [threading.Thread(target=p.epoch_barrier) for p in pools]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        else:
            for i in order_of(r):
                pools[i].epoch_barrier()
    out = (league.export(), [p.assignment.tolist() for p in pools])
    for p in pools:
        p.close()
    return out


def test_league_pools_share_one_league_in_any_barrier_order():
    reference = _rounds(lambda r: [0, 1, 2])
    assert _rounds(lambda r: [2, 0, 1]) == reference
    assert json.loads(reference[0])["round"] == 6


def test_league_barriers_on_threads_equal_sequential_barriers():
    reference = _rounds(lambda r: [0, 1, 2])
    for _ in range(3):
        assert _rounds(None, threads=True) == reference


def test_league_early_barrier_raises_and_names_the_pending_pools():
    league = _league()
    a, b = _pool(league, 0, 1), _pool(league, 1, 2)
    a.reset()
    b.reset()
    _run(a, 0)
    a.epoch_barrier()
    with pytest.raises(tessera_rl.TesseraError, match=r"waiting for pools \[1\]"):
        a.epoch_barrier()
    assert not a.is_poisoned
    b.epoch_barrier()
    assert league.round == 1
    a.epoch_barrier()
    a.close()
    b.close()


def test_league_one_pool_lags_one_round():
    """A league from Python is always shared (rev, E3 ruling 1): with one
    pool, each barrier commits its round, and the table it installs is the
    previous round's, so the first barrier keeps the initial assignment
    table."""
    league = tessera_rl.League(POP, [{"matchmaking": "pfsp", "snapshot": ("every", 1)}])
    p = _pool(league)
    p.reset()
    for r in range(3):
        _run(p, r)
        before = len(league.population())
        p.epoch_barrier()
        assert league.round == r + 1
        assert len(league.population()) == before + 1
    # The pool's assignments only ever name entries of the table it had
    # installed, one round behind: never the snapshot just added.
    assert int(p.assignment.max()) < len(league.population()) - 1
    p.close()


def test_league_save_load_and_export_round_trip():
    league = tessera_rl.League(
        POP + [("onnx", "07" * 16)],
        [{"matchmaking": ("elo", 40.0), "self_play": (0.25, 99), "snapshot": ("win_rate", 0.6, 4), "snapshot_tags": 500}],
        elo_k=24.0,
    )
    data = league.save()
    loaded = tessera_rl.League.load(data)
    assert loaded.save() == data
    assert loaded.export() == league.export()
    pop = loaded.population()
    assert pop[2]["kind"] == {"onnx": "07" * 16}
    assert pop[3]["kind"] == {"external": 99}
    with pytest.raises(tessera_rl.PersistError):
        tessera_rl.League.load(data[:-1])


def test_league_onnx_entry_without_onnx_opponents_is_refused_naming_the_entry():
    league = tessera_rl.League(POP + [("onnx", "07" * 16)], [{"matchmaking": "uniform"}])
    with pytest.raises(tessera_rl.ArgumentError, match="league policy 2 is ONNX " + "07" * 16 + " and make has no onnx_opponents="):
        _pool(league)


def test_league_errors_remapped_for_619_subclass_tessera_error():
    # #619: league and opponent-policy errors now raise ConfigError and
    # OnnxError; both stay TesseraError for existing handlers.
    assert issubclass(tessera_rl.ConfigError, tessera_rl.TesseraError)
    assert issubclass(tessera_rl.OnnxError, tessera_rl.TesseraError)


@pytest.mark.parametrize(
    "population,roles",
    [
        ([], [{"matchmaking": "uniform"}]),
        (POP, []),
        (POP, [{"matchmaking": "nope"}]),
        (POP, [{"matchmaking": "uniform", "self_play": (1.5, 1)}]),
        (POP, [{"matchmaking": "uniform", "snapshot": ("every", 0)}]),
        (["bogus"], [{"matchmaking": "uniform"}]),
        (POP, [{"matchmaking": "uniform", "extra": 1}]),
    ],
)
def test_league_bad_configs_are_refused(population, roles):
    with pytest.raises((tessera_rl.ConfigError, tessera_rl.ArgumentError)):
        tessera_rl.League(population, roles)


def test_league_make_refusals():
    league = _league()
    common = dict(num_envs=2, workers=0, agents=2)
    with pytest.raises(tessera_rl.ArgumentError, match="needs league="):
        tessera_rl.make(DUEL, policy_map={1: "opponent"}, **common)
    with pytest.raises(tessera_rl.ArgumentError, match="needs an \"opponent\" slot"):
        tessera_rl.make(DUEL, league=league, **common)
    with pytest.raises(TypeError):
        tessera_rl.make(DUEL, policy_map={1: "opponent"}, league=object(), **common)
    with pytest.raises(tessera_rl.TesseraError):
        tessera_rl.make(DUEL, policy_map={1: "opponent"}, league=league, mode="async", batch_size=1, **common)


def test_league_pettingzoo_hides_opponents_and_reports_the_assignment():
    """A league pool is multi-agent (its opponent rows may be External), so
    it goes through PettingZoo or the raw Env, not the Gymnasium adapter."""
    league = _league()
    par = pz.parallel_env(DUEL, agents=2, visible="learner", policy_map={1: "opponent"}, league=league, role=1)
    _, infos = par.reset(seed=1)
    assert len(par.agents) == 1
    (agent,) = par.agents
    assert infos[agent]["policy_group"] == "learner"
    assert isinstance(infos[agent]["assignment"], int)
    par.close()


def test_league_replay_round_trips(tmp_path):
    """A league env records and replays through Python (head D's tag 12 and
    keyframe assignments, checked before any event)."""
    from tessera_rl import replay

    path = tmp_path / "league.trp"
    env = _pool(_league(), 0, 5)
    with replay.record(env, path):
        env.reset()
        _run(env, 0)
    env.close()
    fresh = _pool(_league(), 0, 5)
    assert replay.run(path, fresh).verified
    fresh.close()


def test_epoch_barrier_returns_the_fixed_report():
    """``epoch_barrier()`` returns a dict with fixed keys, ``None`` for a
    subsystem the pool does not have (rev, E3 ruling 3)."""
    plain = tessera_rl.make(DUEL, num_envs=2, workers=0, agents=2)
    plain.reset()
    report = plain.epoch_barrier()
    assert list(report) == ["episodes", "normalization", "curriculum", "league"]
    assert set(report["episodes"]) == {"count", "sum_return", "sum_length", "return_mean", "return_m2", "aborted"}
    assert report["normalization"] is None and report["curriculum"] is None and report["league"] is None
    plain.close()
    league = tessera_rl.League(POP, [{"matchmaking": "pfsp"}])
    p = _pool(league)
    p.reset()
    _run(p, 0)
    report = p.epoch_barrier()
    assert set(report["league"]) == {"submitted", "dropped", "uniform_fallback"}
    assert report["league"]["submitted"] > 0
    assert report["episodes"]["count"] > 0
    p.close()


def test_league_gym_refuses_a_league_pool_pointing_to_pettingzoo():
    """A league pool is multi-agent (its opponent rows may be External):
    the Gymnasium adapter refuses it with a typed error naming the way out."""
    with pytest.raises(tessera_rl.CapabilityError, match="PettingZoo adapter"):
        gym.make(DUEL, num_envs=2, workers=0, agents=2, policy_map={1: "opponent"}, league=_league(), role=0)
