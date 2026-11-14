# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ADR-0052 profile of the P2-M3 loop (examples/alphazero_connect4).

Covered:
- the schedule's horizon scaled to 10 generations, M3's unchanged;
- the second formal profile beside M3's, with M3's values and the gate's
  bars unchanged, and the budget capped at 250 core-hours;
- ``init`` requiring the owner's values and refusing overrides of fixed
  ones;
- looks only at the predeclared generations, a look elsewhere refused,
  and only the first look reaching the bar eligible for selection;
- tiny end-to-end runs: ``not eligible`` after the last generation and on
  a budget stop with no extra look, ``early`` on the first qualifying look,
  and a closed root refusing a further run.
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil
import sys

import pytest

torch = pytest.importorskip("torch")
fcntl = pytest.importorskip("fcntl", reason="POSIX only: the loop uses fcntl")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import gate  # noqa: E402
import h2h  # noqa: E402
import loop  # noqa: E402
import schedule  # noqa: E402
import train_step  # noqa: E402
from test_loop import bot_binary, make_root, run_loop  # noqa: E402


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


# ---------------------------------------------------------------- units


def test_schedule_horizon_ten_reaches_lr_min_at_generation_ten():
    assert math.isclose(schedule.lr(0, 10), 0.02, rel_tol=1e-15)
    assert math.isclose(schedule.lr(5000, 10), 0.011, rel_tol=1e-12)
    assert 0.002 < schedule.lr(9999, 10) < 0.002 + 1e-9
    with pytest.raises(ValueError):
        schedule.lr(10000, 10)
    with pytest.raises(ValueError, match="registered"):
        schedule.lr(0, 7)
    # M3's schedule is unchanged: the default horizon is 30 generations.
    assert schedule.lr(9999) == schedule.lr(9999, 30) > 0.015


def test_profile_m3_formal_values_unchanged():
    assert loop.FORMAL["k_max"] == 30 and loop.FORMAL["early_stop_wins"] == 380
    assert loop.FORMAL["budget_core_hours"] == 200.0 and loop.FORMAL["sims"] == 100
    assert loop.is_formal(dict(loop.FORMAL))
    assert loop.is_formal({**loop.FORMAL, "profile": "m3", "horizon_k": 30})
    assert not loop.is_formal({**loop.FORMAL, "horizon_k": 10})
    assert not loop.is_formal({**loop.FORMAL, "profile": "adr-0052"})
    # The gate's bars are unchanged.
    assert gate.ELIGIBLE_WINS == 336 and gate.LOOK_GAMES == 400
    assert gate.WINS_MIN == 815 and gate.GAMES == 1000


def adr0052_cfg(**over):
    return {**loop.FORMAL_ADR0052, "budget_core_hours": 250.0, **over}


def test_profile_adr0052_formal_needs_its_registered_values():
    assert loop.is_formal(adr0052_cfg())
    assert loop.is_formal(adr0052_cfg(budget_core_hours=210.0))
    for key, bad in (("budget_core_hours", 250.5), ("budget_core_hours", 0), ("budget_core_hours", None),
                     ("sims", 100), ("k_max", 30), ("horizon_k", 30), ("look_generations", [6, 8, 9]),
                     ("look_stop_wins", 335), ("c0_sha256", "x"), ("any_openings", True), ("games", 4)):  # fmt: skip
        assert not loop.is_formal(adr0052_cfg(**{key: bad})), (key, bad)
    assert loop.FORMAL_ADR0052["look_generations"] == [6, 8, 10] and loop.FORMAL_ADR0052["look_stop_wins"] == 336


def test_look_scheduled_m3_every_generation_adr0052_only_predeclared():
    assert all(loop.look_scheduled(dict(loop.FORMAL), g) for g in range(1, 31))
    cfg = adr0052_cfg()
    assert [g for g in range(1, 11) if loop.look_scheduled(cfg, g)] == [6, 8, 10]


def test_registered_runtime_hooks_pin_the_run_host_image_and_bot():
    import look

    assert h2h.REGISTERED_HOSTS == frozenset({h2h.REGISTERED_HOST, "swarmer03"})
    assert h2h.REGISTERED_IMAGES == frozenset({h2h.REGISTERED_IMAGE})
    # The run uses the registered bot, unchanged: one entry for the image.
    assert look.REGISTERED_BOTS == {
        h2h.REGISTERED_IMAGE: "ef274131e42f270222d907cd054b43aa19df1577f8a6bca2555b7f9bf311f857"
    }


def fake_loop(cfg):
    lp = loop.Loop.__new__(loop.Loop)
    lp.cfg = cfg
    lp.root = "/nonexistent"
    return lp


FROZEN_PROTOCOL = {"sims": 100, "batch_size": 8, "capacity": 8192, "openings": 200}


def fake_gens(wins_by_generation, protocols=None):
    """Ten generations; ``wins_by_generation`` maps a checkpoint generation
    to its look's wins (no entry: no look); ``protocols`` overrides a
    look's recorded protocol (default: the frozen one)."""
    gens = []
    for k in range(10):
        w = wins_by_generation.get(k + 1)
        lk = None
        if w is not None:
            protocol = (protocols or {}).get(k + 1, FROZEN_PROTOCOL)
            lk = {"attempt": f"g{k:03d}-look-0", "identity": {"checkpoint_sha256": f"sha{k + 1}", "wins": w,
                                                            "eligible": True, "log_sha256": f"log{k + 1}",
                                                            "protocol": protocol}}  # fmt: skip
        gens.append({"look": lk})
    return gens


def ok_entries(gens):
    return [{"run": f"{g['look']['attempt']} x", "status": "ok"} for g in gens if g["look"]]


def test_select_adr0052_only_the_first_look_at_the_bar_is_eligible(monkeypatch):
    lp = fake_loop(adr0052_cfg())
    gens = fake_gens({6: 330, 7: 400, 8: 340, 10: 350})
    eligible = [e for e in lp.looks(gens, ok_entries(gens)) if e["eligible"]]
    assert [e["generation"] for e in eligible] == [8]
    assert loop.choose(eligible) == {"generation": 8, "checkpoint": "sha8", "wins": 340}
    with pytest.raises(loop.Stop) as stop:
        lp.early_stop(gens, ok_entries(gens))
    assert stop.value.reason == "early" and stop.value.inputs["generation"] == 8


def test_select_adr0052_no_look_at_the_bar_selects_nothing():
    lp = fake_loop(adr0052_cfg())
    gens = fake_gens({6: 330, 8: 335, 10: 300})
    assert not any(e["eligible"] for e in lp.looks(gens, ok_entries(gens)))
    lp.early_stop(gens, ok_entries(gens))  # no stop


def test_select_adr0052_unscheduled_look_is_never_eligible():
    lp = fake_loop(adr0052_cfg())
    gens = fake_gens({5: 400, 7: 400, 9: 400})
    assert not any(e["eligible"] for e in lp.looks(gens, ok_entries(gens)))
    assert lp.stop_look(gens) is None


@pytest.mark.parametrize(
    "protocol",
    [{**FROZEN_PROTOCOL, "sims": 400}, {**FROZEN_PROTOCOL, "batch_size": 2}, {**FROZEN_PROTOCOL, "capacity": 64}, None],
)
def test_stop_look_adr0052_look_off_the_frozen_budget_never_stops(protocol):
    lp = fake_loop(adr0052_cfg())
    gens = fake_gens({6: 390}, protocols={6: protocol})
    assert lp.stop_look(gens) is None
    lp.early_stop(gens, ok_entries(gens))  # no "early"
    assert not any(e["eligible"] for e in lp.looks(gens, ok_entries(gens)))
    # A later frozen look at the bar still stops and is the eligible one.
    gens = fake_gens({6: 390, 8: 340}, protocols={6: protocol})
    assert lp.stop_look(gens) == 8


def test_look_sims_root_without_the_key_resolves_to_100_and_formal_needs_it():
    assert "look_sims" not in loop.FORMAL and loop.look_sims(dict(loop.FORMAL)) == 100
    assert loop.is_formal(dict(loop.FORMAL)) and loop.is_formal({**loop.FORMAL, "look_sims": 100})
    assert not loop.is_formal({**loop.FORMAL, "look_sims": 400})
    assert loop.FORMAL_ADR0052["look_sims"] == 100
    assert not loop.is_formal(adr0052_cfg(look_sims=400))
    cfg = adr0052_cfg()
    cfg.pop("look_sims")
    assert loop.look_sims(cfg) == 100 and loop.is_formal(cfg)


def test_select_adr0052_non_formal_root_has_no_eligible_look():
    lp = fake_loop(adr0052_cfg(budget_core_hours=251.0))
    gens = fake_gens({6: 390})
    assert not any(e["eligible"] for e in lp.looks(gens, ok_entries(gens)))


def test_run_phase_unscheduled_look_is_refused():
    lp = fake_loop(adr0052_cfg())
    with pytest.raises(loop.Stop) as stop:
        lp.run_phase([], 4, "look", 0)
    assert stop.value.reason == "unscheduled look" and stop.value.inputs["generation"] == 5


# ---------------------------------------------------------------- init


@pytest.fixture
def c0(tmp_path):
    path = tmp_path / "c0src" / "C0.pt"
    path.parent.mkdir(parents=True)
    assert train_step.main(["init", "--size", "tiny", "--seed", "7", "--out", str(path), "--source-sha", "t"]) == 0
    return path


def init_adr0052(tmp_path, c0, *extra):
    bot = tmp_path / "bot"
    bot.write_bytes(b"bot")
    argv = ["init", "--root", str(tmp_path / "root"), "--c0", str(c0), "--bot", str(bot), "--source-sha", "t",
            "--profile", "adr-0052", *extra]  # fmt: skip
    return loop.main(argv)


OWNER = ("--budget-core-hours", "240", "--selfplay-cores", "32", "--selfplay-wall-s", "3000")


@pytest.mark.parametrize(
    "missing", ["--budget-core-hours", "--selfplay-cores", "--selfplay-wall-s"],
)
def test_init_adr0052_without_an_owner_value_is_refused(tmp_path, c0, missing):
    i = OWNER.index(missing)
    with pytest.raises(SystemExit, match=missing):
        init_adr0052(tmp_path, c0, *OWNER[:i], *OWNER[i + 2 :])


@pytest.mark.parametrize(
    ("extra", "match"),
    [(("--budget-core-hours", "250.5"), "budget"), (("--budget-core-hours", "0"), "budget"),
     (("--k-max", "30"), "k_max"), (("--sims", "100"), "sims"), (("--horizon-k", "30"), "horizon_k"),
     (("--early-stop-wins", "380"), "early-stop")],
)  # fmt: skip
def test_init_adr0052_override_of_a_fixed_value_is_refused(tmp_path, c0, extra, match):
    owner = list(OWNER)
    if extra[0] == "--budget-core-hours":
        owner[1] = extra[1]
        extra = ()
    with pytest.raises(SystemExit, match=match):
        init_adr0052(tmp_path, c0, *owner, *extra)


def test_init_adr0052_writes_the_profile_and_trains_on_horizon_ten(tmp_path, c0):
    assert init_adr0052(tmp_path, c0, *OWNER) == 0
    cfg = json.loads((tmp_path / "root" / "config.json").read_text())
    assert cfg["profile"] == "adr-0052" and cfg["k_max"] == 10 and cfg["horizon_k"] == 10 and cfg["sims"] == 400
    assert cfg["look_generations"] == [6, 8, 10] and cfg["look_stop_wins"] == 336 and cfg["look_sims"] == 100
    assert cfg["budget_core_hours"] == 240.0 and "early_stop_wins" not in cfg
    assert cfg["selfplay_cores"] * cfg["selfplay_wall_s"] == 96000
    # Not the registered C_0, so not formal; every other value is the profile's.
    assert cfg["formal"] is False and loop.is_formal({**cfg, "c0_sha256": loop.REGISTERED_C0})
    lp = loop.Loop(str(tmp_path / "root"))
    gens = [{"selfplay": {"dir": "sp", "attempt": "a"}}]
    argv = lp.phase_argv(gens, 0, "train", 0)
    assert argv[argv.index("--horizon-k") + 1] == "10"


def argv_gens(k):
    """Generations 0..k with self-play and training dirs, enough to build
    argv for generation ``k``."""
    return [{"selfplay": {"dir": f"sp{j}", "attempt": f"g{j:03d}-selfplay-0"},
             "train": {"dir": f"tr{j}", "attempt": f"g{j:03d}-train-0"}} for j in range(k + 1)]  # fmt: skip


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_phase_argv_adr0052_real_profile_values_look_at_100_self_play_at_400(tmp_path, c0):
    """No test overrides: the profile's own values, as ``init`` writes them."""
    assert init_adr0052(tmp_path, c0, *OWNER) == 0
    lp = loop.Loop(str(tmp_path / "root"))
    sp = lp.phase_argv(argv_gens(5), 5, "selfplay", 0)
    assert (flag(sp, "--sims"), flag(sp, "--batch-size"), flag(sp, "--capacity")) == ("400", "8", "8192")
    assert (flag(sp, "--games"), flag(sp, "--workers"), flag(sp, "--wall-cap-s")) == ("10000", "32", "3000")
    lk = lp.phase_argv(argv_gens(5), 5, "look", 0)
    assert (flag(lk, "--sims"), flag(lk, "--batch-size"), flag(lk, "--capacity")) == ("100", "8", "8192")
    assert flag(lk, "--limit") == "200" and "--any-openings" not in lk
    tr = lp.phase_argv(argv_gens(5), 5, "train", 0)
    assert (flag(tr, "--horizon-k"), flag(tr, "--steps"), flag(tr, "--batch")) == ("10", "1000", "1024")


def test_phase_argv_m3_look_uses_the_run_sims_unchanged(tmp_path, bot):
    root = make_root(tmp_path, bot)  # M3, sims 2
    lp = loop.Loop(str(root))
    assert "look_sims" not in lp.cfg
    assert flag(lp.phase_argv(argv_gens(0), 0, "look", 0), "--sims") == "2"


def test_init_m3_defaults_unchanged(tmp_path, c0):
    bot = tmp_path / "bot"
    bot.write_bytes(b"bot")
    assert loop.main(["init", "--root", str(tmp_path / "root"), "--c0", str(c0), "--bot", str(bot), "--source-sha", "t"]) == 0
    cfg = json.loads((tmp_path / "root" / "config.json").read_text())
    assert "profile" not in cfg and cfg["horizon_k"] == 30 and cfg["k_max"] == 30 and cfg["sims"] == 100
    assert cfg["early_stop_wins"] == 380 and cfg["budget_core_hours"] == 200.0 and "look_sims" not in cfg
    assert (cfg["selfplay_cores"], cfg["selfplay_wall_s"]) == (32, 900)
    lp = loop.Loop(str(tmp_path / "root"))
    assert "--horizon-k" not in lp.phase_argv([{"selfplay": {"dir": "sp", "attempt": "a"}}], 0, "train", 0)


# ---------------------------------------------------------------- end to end


def tiny_adr0052_root(tmp_path, bot, stop_wins, frozen_looks=False, **over):
    """A tiny root under the ADR-0052 rules, scaled down: 3 generations,
    looks at C_2 and C_3 only. Never formal. Looks run at 2 sims, batch 2,
    capacity 64 unless ``frozen_looks`` (100, 8, 8192), the only budget
    that can stop the run."""
    if frozen_looks:
        over = {"--batch-size": 8, "--capacity": 8192, **over}
    root = make_root(tmp_path, bot, **{"--k-max": 3, **over})
    path = root / "config.json"
    cfg = json.loads(path.read_text())
    cfg.update(profile="adr-0052", look_generations=[2, 3], look_stop_wins=stop_wins,
               look_sims=100 if frozen_looks else 2)  # fmt: skip
    cfg.pop("early_stop_wins")
    path.write_text(json.dumps(cfg))
    return root


def runs(root):
    return [e["run"].split()[0] for e in loop.Loop(str(root)).ledger.entries()]


def stops(root):
    return [e for e in loop.Loop(str(root)).selection() if e["type"] == "stop"]


def test_run_adr0052_no_look_at_the_bar_ends_not_eligible_and_closes_the_root(tmp_path, bot):
    root = tiny_adr0052_root(tmp_path, bot, stop_wins=99)
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["stop"] == "not eligible"
    assert runs(root) == ["g000-selfplay-0", "g000-train-0",
                          "g001-selfplay-0", "g001-train-0", "g001-look-0",
                          "g002-selfplay-0", "g002-train-0", "g002-look-0"]  # fmt: skip
    assert [(e["reason"], e["inputs"]["cause"]) for e in stops(root)] == [("not eligible", "K_max")]
    lp = loop.Loop(str(root))
    assert [e["generation"] for e in lp.looks(lp.chain(lp.ledger.entries()), lp.ledger.entries())] == [2, 3]
    assert lp.select()["selected"] is None
    again = run_loop(root)
    assert again.returncode != 0 and "closed" in again.stderr
    assert len(runs(root)) == 8


def test_run_adr0052_first_look_at_the_bar_stops_early(tmp_path, bot):
    root = tiny_adr0052_root(tmp_path, bot, stop_wins=0, frozen_looks=True)
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result["stop"] == "early" and result["inputs"]["generation"] == 2
    assert runs(root)[-1] == "g001-look-0" and len(runs(root)) == 5
    again = run_loop(root)
    assert again.returncode != 0 and "closed" in again.stderr


def test_run_adr0052_look_off_the_frozen_budget_at_the_bar_does_not_stop(tmp_path, bot):
    # Looks at 2 sims reach the bar of 0 but are not on the frozen budget.
    root = tiny_adr0052_root(tmp_path, bot, stop_wins=0)
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result["stop"] == "not eligible" and result["inputs"]["cause"] == "K_max"
    assert len(runs(root)) == 8


def test_run_adr0052_budget_stop_between_looks_is_not_eligible_without_a_look(tmp_path, bot):
    # Generation 0 (no look) reserves 3000 + 60 core-seconds; generation 1
    # (with its look) needs 3120, more than what is left of 3100.
    root = tiny_adr0052_root(tmp_path, bot, stop_wins=0, **{"--selfplay-wall-s": 3000,
                                                             "--budget-core-hours": 3100 / 3600})  # fmt: skip
    r = run_loop(root)
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result["stop"] == "not eligible" and result["inputs"]["cause"] == "budget"
    assert runs(root) == ["g000-selfplay-0", "g000-train-0"]
    assert run_loop(root).returncode != 0


def test_chain_adr0052_look_at_an_unscheduled_generation_is_refused(tmp_path, bot):
    root = tiny_adr0052_root(tmp_path, bot, stop_wins=0, frozen_looks=True)
    assert run_loop(root).returncode == 0
    dst = tmp_path / "copy"
    shutil.copytree(root, dst)
    (dst / "gen-000" / "look-0").mkdir()
    lp = loop.Loop(str(dst))
    with pytest.raises(loop.Stop) as stop:
        lp.chain(lp.ledger.entries())
    assert stop.value.reason == "incomplete input" and "predeclared" in stop.value.inputs["reason"]
    with pytest.raises(SystemExit, match="closed"):
        lp.run()
