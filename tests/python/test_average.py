# SPDX-License-Identifier: MIT OR Apache-2.0
"""The predeclared checkpoint average (examples/alphazero_connect4/average.py,
docs/reviews/p2g1-checkpoint-averaging.md r3).

Covered, with averaging and calibration tested apart:
- averaging: identical inputs average to the input, the order and weights
  are recorded, integer counters are not averaged;
- calibration: deterministic for a fixed seed and data, mirrored exactly as
  ``train_window`` builds batches, counters set to the batch count;
- refusals: digest, schema, non-floating entry, too few samples,
  non-finite output, non-positive running variance, look runtime and
  look report;
- the pinned design constants.
"""

import os
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

EXAMPLE = pathlib.Path(__file__).parents[2] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))

import io  # noqa: E402

import artifacts  # noqa: E402
import augment  # noqa: E402
import average  # noqa: E402
import ckpt  # noqa: E402
import net as netmod  # noqa: E402


def save_net(tmp_path, name, seed, size="tiny"):
    path = str(tmp_path / f"{name}.pt")
    return path, ckpt.save(netmod.build(size, seed), path, size, {"test": name})


def write_window(directory, n, seed):
    """A self-play directory with one valid shard of ``n`` samples; returns
    its manifest digest."""
    rng = np.random.default_rng(seed)
    planes = np.zeros((n, 2, 6, 7), dtype=np.uint8)
    lower = rng.random((n, 6, 7)) < 0.4
    planes[:, 0, 1:, :] = lower[:, 1:, :]
    planes[:, 1, 1:, :] = (~lower[:, 1:, :]) & (rng.random((n, 5, 7)) < 0.4)
    arrays = {
        "planes": planes,
        "visits": rng.integers(1, 50, size=(n, 7)).astype(np.uint32),
        "z": rng.choice([-1, 0, 1], size=n).astype(np.int8),
        "g": np.zeros(n, dtype=np.uint32),
        "ply": np.zeros(n, dtype=np.uint16),
    }
    os.makedirs(directory)
    data = artifacts.encode_shard(arrays, {})
    with open(os.path.join(directory, "shard-0.bin"), "wb") as f:
        f.write(data)
    return artifacts.write_manifest(directory, {"files": [{"name": "shard-0.bin", "sha256": artifacts.sha256(data)}]})


def write_raw(path, state, size="tiny"):
    """A checkpoint written byte for byte from ``state`` with a consistent
    manifest, bypassing ``ckpt.save`` and any ``Net``; returns its digest."""
    buf = io.BytesIO()
    torch.save(state, buf)
    data = buf.getvalue()
    digest = artifacts.sha256(data)
    with open(path, "wb") as f:
        f.write(data)
    manifest = {"sha256": digest, "size": size, "complete": True}
    with open(os.path.splitext(path)[0] + ".json", "wb") as f:
        f.write(artifacts.canonical_json(manifest))
    return digest


# ----------------------------------------------------------------- averaging


def test_average_of_identical_checkpoints_equals_the_input(tmp_path):
    paths, digests = [], []
    for i in range(3):
        p = str(tmp_path / f"c{i}.pt")
        digests.append(ckpt.save(netmod.build("tiny", 5), p, "tiny", {"i": i}))
        paths.append(p)
    states, size = average.load_inputs(paths, digests)
    out = average.average(states)
    assert size == "tiny"
    for k, t in states[0].items():
        assert torch.equal(out[k], t), k


def test_average_is_the_uniform_float64_mean_and_counters_are_not_averaged(tmp_path):
    a, da = save_net(tmp_path, "a", 1)
    b, db = save_net(tmp_path, "b", 2)
    states, _ = average.load_inputs([a, b], [da, db])
    states[1] = {k: (v + 4 if not v.is_floating_point() else v) for k, v in states[1].items()}
    out = average.average(states)
    for k, t in out.items():
        if average.is_counter(k):
            assert t.dtype == torch.int64 and torch.equal(t, states[0][k])
        elif average.is_running_stat(k):
            assert torch.equal(t, states[0][k])
        else:
            want = ((states[0][k].double() + states[1][k].double()) * 0.5).float()
            assert t.dtype == torch.float32 and torch.equal(t, want), k


def test_average_refuses_a_digest_other_than_the_pinned_one(tmp_path):
    a, da = save_net(tmp_path, "a", 1)
    with pytest.raises(average.Refused, match="digest"):
        average.load_inputs([a], ["0" * 64])


def test_average_refuses_a_changed_file(tmp_path):
    a, da = save_net(tmp_path, "a", 1)
    with open(a, "ab") as f:
        f.write(b"x")
    with pytest.raises(artifacts.Incomplete):
        average.load_inputs([a], [da])


def test_average_refuses_a_schema_difference(tmp_path):
    a, da = save_net(tmp_path, "a", 1, "tiny")
    b, db = save_net(tmp_path, "b", 2, "small")
    with pytest.raises(average.Refused):
        average.load_inputs([a, b], [da, db])


def test_average_refuses_a_serialized_dtype_change(tmp_path):
    state = netmod.build("tiny", 1).state_dict()
    key = next(k for k, v in state.items() if v.is_floating_point())
    state[key] = state[key].double()
    path = str(tmp_path / "f64.pt")
    digest = write_raw(path, state)
    with pytest.raises(average.Refused, match="float64"):
        average.load_inputs([path], [digest])


def test_average_refuses_serialized_reordered_keys(tmp_path):
    state = netmod.build("tiny", 1).state_dict()
    keys = list(state)
    reordered = {k: state[k] for k in [keys[1], keys[0], *keys[2:]]}
    path = str(tmp_path / "reordered.pt")
    digest = write_raw(path, reordered)
    with pytest.raises(average.Refused, match="key order"):
        average.load_inputs([path], [digest])


def test_average_refuses_an_unexpected_non_floating_entry(tmp_path):
    state = netmod.build("tiny", 1).state_dict()
    state["extra_steps"] = torch.zeros((), dtype=torch.int64)
    path = str(tmp_path / "extra.pt")
    digest = write_raw(path, state)
    with pytest.raises(average.Refused, match="key order"):
        average.load_inputs([path], [digest])
    check = dict(netmod.build("tiny", 1).state_dict())
    wkey = next(k for k, v in check.items() if v.is_floating_point())
    check[wkey] = check[wkey].to(torch.int64)
    reference = netmod.build("tiny", 1).state_dict()
    reference[wkey] = reference[wkey].to(torch.int64)
    with pytest.raises(average.Refused, match="non-floating"):
        average.check_schema("x", check, reference)


def test_average_accepts_a_raw_state_that_matches_the_schema(tmp_path):
    state = netmod.build("tiny", 1).state_dict()
    path = str(tmp_path / "raw.pt")
    digest = write_raw(path, state)
    states, size = average.load_inputs([path], [digest])
    assert size == "tiny" and list(states[0]) == list(state)


# --------------------------------------------------------------- calibration


@pytest.fixture
def window(tmp_path):
    dirs = [str(tmp_path / f"w{i}") for i in range(2)]
    digests = [write_window(d, 40, seed=i) for i, d in enumerate(dirs)]
    return dirs, digests


def built(tmp_path, window, tag, seed=3, sample=64, batch=16):
    a, da = save_net(tmp_path, f"a-{tag}", 1)
    b, db = save_net(tmp_path, f"b-{tag}", 2)
    dirs, digests = window
    return average.build_model([a, b], [da, db], dirs, digests, seed=seed, sample=sample, batch=batch)


def test_calibration_is_deterministic_and_sets_the_counters(tmp_path, window):
    m1, _, meta1 = built(tmp_path, window, "first")
    m2, _, meta2 = built(tmp_path, window, "second")
    s1, s2 = m1.state_dict(), m2.state_dict()
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    for k, t in s1.items():
        if average.is_counter(k):
            assert int(t) == 64 // 16
    assert meta1 == meta2
    cal = meta1["calibration"]
    assert cal["n"] == 80 and cal["sample"] == 64 and cal["seed"] == 3 and cal["data_manifests"] == window[1]
    assert meta1["weights"] == [0.5, 0.5] and len(meta1["parents"]) == 2
    assert not m1.training


def test_calibration_changes_only_running_statistics(tmp_path, window):
    a, da = save_net(tmp_path, "a", 1)
    b, db = save_net(tmp_path, "b", 2)
    states, _ = average.load_inputs([a, b], [da, db])
    plain = average.average(states)
    model, _, _ = average.build_model([a, b], [da, db], *window, seed=3, sample=64, batch=16)
    for k, t in model.state_dict().items():
        if not (average.is_running_stat(k) or average.is_counter(k)):
            assert torch.equal(t, plain[k]), k


def test_calibration_batches_match_the_materialized_mirror_pool():
    rng = np.random.default_rng(1)
    planes = (rng.random((5, 2, 6, 7)) < 0.3).astype(np.uint8)
    policy = rng.dirichlet(np.ones(7), size=5)
    pool, _, _ = augment.augment(planes.astype(np.float32), policy, np.zeros(5))
    idx = average.sample_indices(5, seed=9, sample=8)
    x = average.batch_planes(torch.from_numpy(planes.astype(np.float32)), idx)
    assert torch.equal(x, torch.from_numpy(pool[idx]))


def test_calibration_sampling_is_without_replacement_unsorted_and_pinned():
    idx = average.sample_indices(100, seed=average.SEED, sample=150)
    assert len(set(idx.tolist())) == 150 and idx.min() >= 0 and idx.max() < 200
    assert not np.all(np.diff(idx) > 0)
    again = np.random.Generator(np.random.PCG64(average.SEED)).choice(200, size=150, replace=False)
    assert np.array_equal(idx, again)


def test_calibration_index_digest_is_little_endian_uint64():
    idx = np.array([1, 2, 258], dtype=np.int64)
    want = artifacts.sha256(b"".join(int(i).to_bytes(8, "little") for i in idx))
    assert average.indices_sha256(idx) == want
    assert average.indices_sha256(idx.astype(">i8")) == want


def test_calibration_refuses_too_few_samples():
    with pytest.raises(average.Refused, match="fewer than"):
        average.sample_indices(10, sample=21)


def test_calibration_refuses_other_data_manifests(tmp_path, window):
    a, da = save_net(tmp_path, "a", 1)
    with pytest.raises(average.Refused, match="data manifests"):
        average.build_model([a], [da], window[0], ["0" * 64, "1" * 64], seed=3, sample=64, batch=16)


def test_calibration_output_checks_refuse_non_finite_and_bad_variance():
    model = netmod.build("tiny", 1)
    state = model.state_dict()
    key = next(k for k in state if k.endswith("running_var"))
    bad = dict(state)
    bad[key] = torch.zeros_like(state[key])
    with pytest.raises(average.Refused, match="variance"):
        average.check_output(bad, 0)
    nan = dict(state)
    wkey = next(k for k, v in state.items() if v.is_floating_point() and not average.is_running_stat(k))
    nan[wkey] = torch.full_like(state[wkey], float("nan"))
    with pytest.raises(ckpt.NonFinite):
        average.check_output(nan, 0)


# --------------------------------------------------------------------- look


def test_look_preflight_accepts_only_the_registered_runtime():
    average.preflight(dict(average.LOOK_RUNTIME, boot_id="x", cpu_model="y"))
    for field, value in (("cpuset", "0-7"), ("host", "b5ac6e56fdc0"), ("affinity_cores", 8), ("registered", False)):
        with pytest.raises(average.Refused, match=field):
            average.preflight(dict(average.LOOK_RUNTIME, **{field: value}))


def look_report(wins=336, **over):
    sha = "a" * 64
    report = {
        "checkpoint": sha,
        "complete": True,
        "eligible": True,
        "games": 400,
        "wins": wins,
        "wdl": [wins, 4, 396 - wins],
        "by_colour": {"bot_first": [wins // 2, 2, 198 - wins // 2], "candidate_first": [wins - wins // 2, 2, 198 - (wins - wins // 2)]},
        "runtime": dict(average.LOOK_RUNTIME),
        "protocol": average.frozen_protocol(sha),
    }
    report.update(over)
    return report, sha


def test_look_verify_reports_the_candidate_rule():
    report, sha = look_report()
    assert average.verify_look(report, sha) == {"wins": 336, "eligible": True, "candidate": True}
    report, sha = look_report(wins=335)
    assert average.verify_look(report, sha)["candidate"] is False


def test_look_verify_accepts_a_real_m3_look_report_shape():
    report = {
        "checkpoint": "d9cc322d4175ae0dda4d6a3bd741a1ef6b67465e4b62198e7b64c54011f8901c",
        "complete": True,
        "eligible": True,
        "games": 400,
        "wins": 310,
        "wdl": [310, 12, 78],
        "by_colour": {"bot_first": [139, 9, 52], "candidate_first": [171, 3, 26]},
        "runtime": dict(average.LOOK_RUNTIME, boot_id="67bcf7e8", cpu_model="i7-12700H", mem_total="64302580 kB"),
        "protocol": {
            "batch_size": 8,
            "bot_sha256": "ef274131e42f270222d907cd054b43aa19df1577f8a6bca2555b7f9bf311f857",
            "capacity": 8192,
            "checkpoint": "d9cc322d4175ae0dda4d6a3bd741a1ef6b67465e4b62198e7b64c54011f8901c",
            "openings": 200,
            "openings_sha256": "3276d2d044c911ed29c36f2d42690697f19a4bac0183e3a77b1af3c5af72cc49",
            "search_seed": 20261102,
            "sims": 100,
            "size": "medium",
            "tie_base": 9103431398621970432,
        },
    }
    assert average.verify_look(report, report["checkpoint"]) == {"wins": 310, "eligible": True, "candidate": False}


def test_look_verify_refuses_other_runtime_protocol_or_checkpoint():
    report, sha = look_report()
    report["runtime"]["cpuset"] = "0-7"
    with pytest.raises(average.Refused):
        average.verify_look(report, sha)
    for field, value in (
        ("sims", 400),
        ("openings_sha256", "0" * 64),
        ("bot_sha256", "0" * 64),
        ("search_seed", 1),
        ("tie_base", 0),
        ("size", "small"),
        ("checkpoint", "b" * 64),
    ):
        report, sha = look_report()
        report["protocol"][field] = value
        with pytest.raises(average.Refused, match="frozen"):
            average.verify_look(report, sha)
    report, sha = look_report()
    report["protocol"]["extra"] = 1
    with pytest.raises(average.Refused, match="frozen"):
        average.verify_look(report, sha)
    for over in ({"wins": 300}, {"games": 398}, {"wdl": [336, 4, 50]}):
        report, sha = look_report()
        report.update(over)
        with pytest.raises(average.Refused, match="consistent"):
            average.verify_look(report, sha)
    report, sha = look_report()
    report["by_colour"]["bot_first"] = [100, 2, 98]
    with pytest.raises(average.Refused, match="consistent"):
        average.verify_look(report, sha)
    report, sha = look_report(eligible=False)
    with pytest.raises(average.Refused, match="eligible"):
        average.verify_look(report, sha)
    report, _ = look_report()
    with pytest.raises(average.Refused, match="report is for"):
        average.verify_look(report, "b" * 64)


# ------------------------------------------------------------------- pins


def test_average_pins_match_the_design_note():
    assert [d for d, _ in average.INPUTS] == [f"gen-{g:03d}" for g in range(20, 30)]
    assert [d for d, _ in average.DATA] == [f"gen-{g:03d}" for g in range(26, 30)]
    assert all(len(s) == 64 for _, s in average.INPUTS + average.DATA)
    assert (average.SEED, average.SAMPLE, average.BATCH) == (20261107, 65536, 1024)
    assert average.INPUTS[0][1].startswith("f9d5c128") and average.INPUTS[-1][1].startswith("30eaf149")
    assert average.LOOK_RUNTIME["cpuset"] == "2" and average.LOOK_RUNTIME["host"] == "xaskar"
