# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/pgo/evidence.py on hand-written samples. No Valgrind runs here: that
real Callgrind output meets the gate is unproven until a host window exists."""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("evidence", ROOT / "tools" / "pgo" / "evidence.py")
ev = importlib.util.module_from_spec(_spec)
sys.modules["evidence"] = ev
_spec.loader.exec_module(ev)

MANIFEST = ev.load_manifest(str(ROOT / "tools" / "pgo" / "manifest.toml"))
ALL = ev.scenarios(MANIFEST)
EVENTS = "Ir Dr Dw I1mr D1mr D1mw ILmr DLmr DLmw Bc Bcm Bi Bim"


def sample():
    return f"""# callgrind format
version: 1
positions: line
events: {EVENTS}
ob=(1) /venv/lib/tessera_rl/_native.abi3.so
fl=(1) a.rs
fn=(1) step
10 100 0 0 1 2 3 4 5 6 0 7 0 8
+2 50 0 0 0 0 0 0 0 0 0 1 0 1
cob=(2) /usr/lib/libc.so
cfn=(2) malloc
calls=1 0
12 999999 0 0 0 0 0 0 0 0 0 0 0 0
13 5 0 0 0 0 0 0 0 0 0 0 0 0
ob=(2)
fn=(2)
* 40 0 0 0 0 0 0 0 0 0 0 0 0
ob=(1)
fn=(1)
20 5 0 0 0 0 0 0 0 0 0 0 0 0
"""


def test_pgo_manifest_has_the_frozen_shape():
    assert MANIFEST["version"] == 1 and len(ALL) == 20
    options = (ROOT / "crates" / "tessera-rl-engine" / "src" / "options.rs").read_text()
    assert MANIFEST["chunk_size"] == 64 and "NonZeroU32::MIN.saturating_add(63)" in options
    assert {s.name for s in ALL if s.holdout} == {"holdout/PaddleDuel-v1-state", "holdout/Go9-v1", "holdout-split-phase/CartPole-v1"}
    assert sum(s.single_threaded and not s.holdout for s in ALL) == 14


def test_pgo_manifest_env_ids_exist():
    make = (ROOT / "python" / "tessera_rl" / "_env.py").read_text()
    assert all(f'"{s.env_id}"' in make for s in ALL)


def test_pgo_training_set_drops_the_holdout_for_w2h_only():
    assert len(ev.training_set(ALL, False)) == 20
    assert not any(s.holdout for s in ev.training_set(ALL, True)) and len(ev.training_set(ALL, True)) == 17


def test_pgo_manifest_refuses_duplicates_and_empty(tmp_path):
    g = {"name": "a", "envs": [{"id": "x/y"}, {"id": "x/y"}], "num_envs": 1, "steps": 1}
    with pytest.raises(ValueError, match="duplicate"):
        ev.scenarios({"group": [g]})
    with pytest.raises(ValueError, match="positive"):
        ev.scenarios({"group": [{**g, "envs": [{"id": "x/y"}], "steps": 0}]})
    p = tmp_path / "m.toml"
    p.write_text("seed = 1\n")
    with pytest.raises(ValueError, match="version"):
        ev.load_manifest(str(p))


def test_pgo_callgrind_is_exclusive_per_object_and_skips_inclusive_call_lines():
    objects = ev.parse_callgrind(sample())
    native = ev.native_cost(objects)
    assert native.ir == 100 + 50 + 5 + 5
    assert native.mispredicts == 7 + 8 + 1 + 1
    assert objects["/usr/lib/libc.so"].ir == 40


def test_pgo_callgrind_refuses_what_it_cannot_read():
    with pytest.raises(ValueError, match="no events"):
        ev.parse_callgrind("version: 1\n")
    with pytest.raises(ValueError, match="events missing"):
        ev.parse_callgrind("events: Ir\n")
    with pytest.raises(ValueError, match="unknown object"):
        ev.parse_callgrind(f"events: {EVENTS}\nob=(9)\n")


def test_pgo_native_cost_needs_exactly_one_object():
    c = ev.Cost({"Ir": 1})
    with pytest.raises(ValueError, match="found 0"):
        ev.native_cost({"/lib/libc.so": c})
    with pytest.raises(ValueError, match="found 2"):
        ev.native_cost({"/a/_native.so": c, "/b/_native.so": c})


def test_pgo_feasibility_checks():
    assert ev.interpreter_excluded(99, 100000) and not ev.interpreter_excluded(100, 100000)
    assert ev.linear(100, 200, 300) and ev.linear(100, 200, 301)
    assert not ev.linear(100, 200, 310) and not ev.linear(200, 200, 300)
    assert ev.workers_collected(90, 100) and not ev.workers_collected(89, 100)
    assert ev.spread([99, 100, 101]) == pytest.approx(0.02)


def cost(ir, mis=5000):
    return ev.Cost({"Ir": ir, "Bcm": mis, "Bim": 0})


def stable():
    return {s.name: [cost(1000000), cost(1000000), cost(1000000)] for s in ALL}


def test_pgo_gating_keeps_only_stable_single_threaded():
    sets = ev.gating_sets(ALL, stable())
    assert sets.usable and sets.mispredict_available and len(sets.gating) == 16
    assert not any("threaded" in n or "split" in n for n in sets.gating)


def test_pgo_an_unstable_scenario_leaves_the_gating_set():
    aa = stable()
    name = "state/CartPole-v1"
    aa[name] = [cost(1000000), cost(1002000), cost(1000000)]
    assert name not in ev.gating_sets(ALL, aa).gating


def test_pgo_a_missing_holdout_makes_the_evidence_unusable():
    aa = stable()
    aa["holdout/Go9-v1"] = [cost(1000000), cost(1100000), cost(1000000)]
    assert ev.gating_sets(ALL, aa).usable is False


def test_pgo_mispredict_set_has_its_own_rules():
    aa = {k: [cost(1000000, 500)] * 3 for k in stable()}
    sets = ev.gating_sets(ALL, aa)
    assert sets.mispredict == () and not sets.mispredict_available and sets.usable


def test_pgo_gating_needs_exact_coverage():
    aa = stable()
    aa.pop("state/CartPole-v1")
    with pytest.raises(ValueError, match="cover"):
        ev.gating_sets(ALL, aa)


def wheel(ir=1000000, mis=5000, **over):
    out = {s.name: cost(ir, mis) for s in ALL}
    out.update(over)
    return out


SETS = ev.gating_sets(ALL, stable())


@pytest.mark.parametrize(
    ("w1", "det", "want"),
    [(wheel(980000), True, True), (wheel(981000), True, False), (wheel(980000), False, False)],
)
def test_pgo_w1_acceptance(w1, det, want):
    assert ev.accept_w1(w1, wheel(), SETS.gating, det) is want


def test_pgo_w1_refuses_one_regression_over_two_percent():
    w1 = wheel(900000, **{"state/CartPole-v1": cost(1021000)})
    assert ev.accept_w1(w1, wheel(), SETS.gating, True) is False


def accept(w2f, w2h=None, sets=SETS, ok=True, det=True):
    return ev.accept_w2f(w2f, w2h or wheel(1000000), wheel(), ALL, sets, ok, det)


def test_pgo_w2f_passes_on_ir():
    assert accept(wheel(970000)) is True


def test_pgo_w2f_passes_on_mispredicts_only_if_available():
    w2f = wheel(1000000, 4500)
    assert accept(w2f) is True
    no = ev.Sets(SETS.gating, (), True, False)
    assert accept(w2f, sets=no) is False


def test_pgo_w2f_gates():
    assert accept(wheel(971000)) is False
    assert accept(wheel(970000), ok=False) is False
    assert accept(wheel(970000), det=False) is False
    assert accept(wheel(970000), wheel(1001000)) is False
    assert accept(wheel(970000, **{"holdout/Go9-v1": cost(1021000)})) is False
    assert accept(wheel(970000), sets=ev.Sets(SETS.gating, SETS.mispredict, False, True)) is False


@pytest.mark.parametrize(("w1", "w2", "want"), [(False, True, "W0"), (True, False, "W1"), (True, True, "W2F"), (False, False, "W0")])
def test_pgo_ship_ladder(w1, w2, want):
    assert ev.ship(w1, w2) == want
