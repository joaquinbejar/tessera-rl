# SPDX-License-Identifier: MIT OR Apache-2.0
"""ONNX league opponents from Python (#619; ADR-0088 item 2.3, erratum 2;
ADR-0057): ``make(..., league=, onnx_opponents=OnnxOpponents(...))``.

The gating follows ``test_onnx_evaluator.py``. The refusal tests need no
library. The real-library tests run only through ``make py-test-onnx``,
which sets ORT_DISABLE_TELEMETRY=1, a scratch HOME and TESSERA_ONNX_TEST=1;
under it every test must run (no skip), checked by
``test_onnx_league_real_run_has_the_feature``.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import search

sys.path.insert(0, os.path.dirname(__file__))

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "crates" / "tessera-rl-onnx" / "tests" / "fixtures"
MODEL = FIXTURES / "opponent_paddleduel.onnx"
MISSING = "/nonexistent/tessera-onnx/libonnxruntime.so.1.31.1"
REAL = os.environ.get("TESSERA_ONNX_TEST") == "1"
LIB = {"library": os.environ["TESSERA_ONNXRUNTIME_LIB"]} if os.environ.get("TESSERA_ONNXRUNTIME_LIB") else {}
DUEL = "tessera/PaddleDuel-v1"

pytestmark = pytest.mark.filterwarnings(
    "ignore:Observation is not a NumPy array:UserWarning",
    "ignore:Observation space for each agent probably should be:UserWarning",
)


def has_feature() -> bool:
    try:
        t._native.onnx_load(library=MISSING, accept_telemetry=True)
    except t.OnnxError as e:
        return "no ONNX evaluator" not in str(e)
    return True


real = pytest.mark.skipif(not REAL, reason="opens the real ONNX Runtime: run through `make py-test-onnx`")


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def pool(league, **kw):
    return t.make(
        DUEL,
        num_envs=8,
        seed=5,
        workers=0,
        agents=2,
        obs="state",
        max_episode_steps=40,
        outputs="owned",
        policy_map={1: "opponent"},
        league=league,
        role=0,
        **kw,
    )


def uniform(kind):
    return t.League([kind], [{"matchmaking": "uniform"}])


def test_onnx_league_without_onnx_opponents_is_refused_naming_the_hash():
    h = "07" * 16
    with pytest.raises(t.ArgumentError, match=f"league policy 0 is ONNX {h} and make has no onnx_opponents="):
        pool(uniform(("onnx", h)))


def test_onnx_league_onnx_opponents_must_be_the_type():
    with pytest.raises(TypeError, match="OnnxOpponents"):
        pool(uniform("heuristic"), onnx_opponents=[str(MODEL)])


def test_onnx_league_build_without_the_feature_refuses_onnx_opponents():
    # Never skipped: a build without the onnx feature refuses with
    # CapabilityError before any library is opened; a build with it reaches
    # the loader, which refuses the missing library with OnnxError.
    league = uniform("heuristic")  # kept alive: the native call takes its handle
    kwargs = {
        "num_envs": 1,
        "agents": 2,
        "policy_map": "1=opponent",
        "league": league._handle,
        "onnx_models": [os.fsencode(MODEL)],
        "onnx_library": MISSING,
        "onnx_accept_telemetry": True,
    }
    if has_feature():
        with pytest.raises(t.OnnxError):
            t._native.make_paddleduel(**kwargs)
    else:
        with pytest.raises(t.CapabilityError, match=r"no ONNX opponents \(the onnx feature\)"):
            t._native.onnx_model_provenance(model=os.fsencode(MODEL), library=MISSING, accept_telemetry=True)
        with pytest.raises(t.CapabilityError, match=r"no ONNX opponents \(the onnx feature\)"):
            t._native.make_paddleduel(**kwargs)


def test_onnx_league_onnx_opponents_without_a_league_is_refused():
    kwargs = {"num_envs": 1, "agents": 2, "onnx_models": [os.fsencode(MODEL)], "onnx_library": MISSING}
    want = (t.ArgumentError, "onnx_opponents= needs league=") if has_feature() else (t.CapabilityError, "the onnx feature")
    with pytest.raises(want[0], match=want[1]):
        t._native.make_paddleduel(**kwargs)


def test_onnx_league_errors_subclass_tessera_error():
    assert issubclass(t.OnnxError, t.TesseraError)
    assert issubclass(t.ConfigError, t.TesseraError)


@pytest.mark.skipif(not REAL, reason="checks the py-test-onnx run itself")
def test_onnx_league_real_run_has_the_feature():
    # Under `make py-test-onnx` nothing may skip for want of the feature.
    assert has_feature()
    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"


def opponents(*models, **kw):
    return search.OnnxOpponents([*models], **LIB, **kw)


@real
def test_onnx_league_opponent_plays_the_model_rule_as_an_external_caller_would():
    h = content_hash(MODEL)
    onnx = pool(uniform(("onnx", h)), onnx_opponents=opponents(MODEL))
    external = pool(uniform(("external", 1)))
    a, b = onnx.reset(), external.reset()
    seen = 0
    for s in range(60):
        oa = next(iter(a.observations.values()))
        ob = next(iter(b.observations.values()))
        np.testing.assert_array_equal(oa, ob, err_msg=f"step {s}")
        np.testing.assert_array_equal(a.rewards, b.rewards, err_msg=f"step {s}")
        x = np.zeros((8, 2), dtype=np.int32)
        x[:, 0] = (s + np.arange(8)) % 3
        y = x.copy()
        rows = ob.reshape(8, 2, -1)[:, 1]
        y[:, 1] = np.where(rows[:, 0] >= rows[:, 1], 2, 0)
        seen += int((y[:, 1] == 2).sum())
        a, b = onnx.step(x), external.step(y)
    assert seen > 0
    onnx.close()
    external.close()


@real
def test_onnx_league_same_seeds_repeat_bit_for_bit():
    h = content_hash(MODEL)
    runs = []
    for _ in range(2):
        env = pool(uniform(("onnx", h)), onnx_opponents=opponents(MODEL))
        step = env.reset()
        out = []
        for s in range(30):
            x = np.full((8, 2), s % 3, dtype=np.int32)
            step = env.step(x)
            out.append(next(iter(step.observations.values())).tobytes())
        env.close()
        runs.append(out)
    assert runs[0] == runs[1]


@real
def test_onnx_league_unserved_hash_is_refused_naming_it():
    h = "0f" * 16
    with pytest.raises(t.ArgumentError, match=f"league policy 0 is ONNX {h} and no onnx_opponents model has that content hash"):
        pool(uniform(("onnx", h)), onnx_opponents=opponents(MODEL))


@real
def test_onnx_league_unused_extra_model_is_allowed():
    env = pool(uniform("heuristic"), onnx_opponents=opponents(MODEL))
    env.reset()
    env.close()


@real
def test_onnx_league_duplicate_or_off_contract_model_is_refused():
    h = content_hash(MODEL)
    with pytest.raises(t.OnnxError, match="content hash of an earlier model"):
        pool(uniform(("onnx", h)), onnx_opponents=opponents(MODEL, MODEL))
    with pytest.raises(t.OnnxError):
        pool(uniform(("onnx", h)), onnx_opponents=opponents(FIXTURES / "constant.onnx"))


@real
def test_onnx_league_provenance_is_read_only_and_recorded():
    h = content_hash(MODEL)
    o = opponents(MODEL)
    env = pool(uniform(("onnx", h)), onnx_opponents=o)
    p = env.onnx_provenance
    assert list(p) == [h]
    assert p[h]["model_sha256"] == hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert p[h]["intra_threads"] == 1 and p[h]["inter_threads"] == 1
    assert p[h]["telemetry_enabled"] is False
    with pytest.raises(TypeError):
        p["x"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        p[h]["model_sha256"] = "x"  # type: ignore[index]
    assert dict(o.provenance) == dict(p)
    env.close()
    assert pool(uniform("heuristic")).onnx_provenance is None


@real
def test_onnx_league_barrier_keeps_the_onnx_entry():
    h = content_hash(MODEL)
    league = uniform(("onnx", h))
    env = pool(league, onnx_opponents=opponents(MODEL))
    env.reset()
    for s in range(20):
        env.step(np.full((8, 2), s % 3, dtype=np.int32))
    env.epoch_barrier()
    assert league.population()[0]["kind"] == {"onnx": h}
    env.close()


@real
@pytest.mark.skipif(sys.platform == "win32", reason="ADR-0057 platforms only")
def test_onnx_league_no_telemetry_directory_in_the_scratch_home():
    home = Path(os.environ["HOME"])
    assert not any(p.name.lower().startswith("onnxruntime") for p in home.rglob("*"))


def _rule(observations, info):
    """The fixture's rule on each slot's own state row: 2 if obs[0] >= obs[1]
    else 0 (an External opponent playing what the model plays)."""
    leaf = np.asarray(next(iter(observations.values())))
    rows = leaf.reshape(leaf.shape[0], -1)
    return np.where(rows[:, 0] >= rows[:, 1], 2, 0).astype(np.int64)


@real
def test_onnx_league_eval_counts_equal_an_external_opponent_playing_the_rule():
    import eval_policies as pol  # noqa: PLC0415, E402
    from tessera_rl import eval as tev  # noqa: PLC0415

    h = content_hash(MODEL)
    kw = {"agents": 2, "obs": "state", "max_episode_steps": 40}
    seeds = [11, 12, 13, 14, 15]
    o = opponents(MODEL)
    onnx = tev.evaluate(DUEL, pol.chase, [(0, ("onnx", h))], seeds=seeds, episodes=6, pools=3, env_kwargs=kw, onnx_opponents=o)
    ext = tev.evaluate(DUEL, pol.chase, [(0, ("external", 3))], seeds=seeds, episodes=6, pools=3, env_kwargs=kw, external={3: _rule})
    a, b = onnx["opponents"][0], ext["opponents"][0]
    for key in ("wins", "draws", "losses", "learner_return", "opponent_return"):
        assert a[key] == b[key], key
    assert a["onnx_provenance"] == dict(o.provenance[h])


@real
def test_onnx_league_model_path_with_a_newline_loads(tmp_path):
    # Paths cross as os.fsencode bytes, element by element: no separator.
    path = tmp_path / "opponent\nwith newline.onnx"
    path.write_bytes(MODEL.read_bytes())
    h = content_hash(MODEL)
    env = pool(uniform(("onnx", h)), onnx_opponents=opponents(path))
    env.reset()
    env.close()


@real
def test_onnx_league_non_utf8_model_path_loads_where_the_filesystem_allows(tmp_path):
    # Never skipped. Linux takes a non-UTF-8 name and the model loads from
    # it; macOS (APFS) refuses the name with OSError, and the bytes path is
    # still exercised with a valid name passed as os.fsencode bytes.
    h = content_hash(MODEL)
    path = os.fsencode(tmp_path) + b"/opponent-\xff.onnx"
    if sys.platform.startswith("linux"):
        with open(path, "wb") as f:
            f.write(MODEL.read_bytes())
    else:
        with pytest.raises(OSError):
            with open(path, "wb") as f:
                f.write(MODEL.read_bytes())
        path = os.fsencode(tmp_path / "opponent-bytes.onnx")
        with open(path, "wb") as f:
            f.write(MODEL.read_bytes())
    o = opponents(path)
    assert list(o.provenance) == [h]
    env = pool(uniform(("onnx", h)), onnx_opponents=o)
    env.reset()
    env.close()
