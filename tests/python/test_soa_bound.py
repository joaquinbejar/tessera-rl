# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/bench/soa_bound.py (#735): record size from Criterion's throughput,
per-instance cost against ADR-0010's bound, and its refusals of bad input."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "bench" / "soa_bound.py"
N = 8192


def load():
    spec = importlib.util.spec_from_file_location("soa_bound", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def written(fid):
    """Records the bench declares for an operation: N - 1 for the forks."""
    return N - 1 if fid.split("/")[-1] in ("fork", "fork_branched") else N


def case(root, group, fid, record_bytes, ns_per_instance, throughput=None):
    d = root / group / fid.replace("/", "_") / "new"
    d.mkdir(parents=True)
    n = written(fid)
    tp = {"Bytes": record_bytes * n} if throughput is None else throughput
    (d / "benchmark.json").write_text(json.dumps({"group_id": group, "function_id": fid, "throughput": tp}))
    total = ns_per_instance * n
    est = {"median": {"point_estimate": total, "confidence_interval": {"lower_bound": total * 0.9, "upper_bound": total * 1.1}}}
    (d / "estimates.json").write_text(json.dumps(est))


def test_soa_bound_bound_is_one_us_per_kib_per_instance():
    assert load().bound_ns(1024) == 1000.0
    assert load().bound_ns(96) == 93.75


def test_soa_bound_within_bound_passes_and_prints_record_size(tmp_path, capsys):
    case(tmp_path, "g", "soa_cartpole/fork", 96, 80.0)
    case(tmp_path, "g", "soa_cartpole/snapshot_all", 96, 38.8)
    assert load().main([str(tmp_path), "--group", "g", "--instances", str(N)]) == 0
    out = capsys.readouterr().out
    assert "soa_cartpole/fork: record 96 B, median 80.0 ns [72.0, 88.0], bound 93.8 ns, PASS" in out
    assert "soa_cartpole/snapshot_all: record 96 B, median 38.8 ns" in out


def test_soa_bound_over_bound_fails(tmp_path, capsys):
    case(tmp_path, "g", "soa_cartpole/fork", 96, 123.0)
    assert load().main([str(tmp_path), "--group", "g", "--instances", str(N)]) == 1
    assert "soa_cartpole/fork: record 96 B, median 123.0 ns" in capsys.readouterr().out


def test_soa_bound_report_only_ops_never_fail(tmp_path, capsys):
    case(tmp_path, "g", "soa_cartpole/state_hash", 96, 5000.0)
    case(tmp_path, "g", "soa_cartpole/snapshot_one", 96, 5000.0)
    assert load().main([str(tmp_path), "--group", "g", "--instances", str(N)]) == 0
    assert capsys.readouterr().out.count(", report") == 2


def test_soa_bound_real_criterion_fork_shape_divides_by_destinations(tmp_path, capsys):
    # The files Criterion 0.5 writes for benches/soa_snapshot.rs's
    # aos_grid/fork at N = 8192: 72-byte records, 8191 destinations
    # (589752 = 72 * 8191; gate-b735-r1 refused this before the fix).
    d = tmp_path / "soa_snapshot_grid_8192" / "aos_grid_fork" / "new"
    d.mkdir(parents=True)
    (d / "benchmark.json").write_text(json.dumps({
        "group_id": "soa_snapshot_grid_8192",
        "function_id": "aos_grid/fork",
        "value_str": None,
        "throughput": {"Bytes": 589752},
        "full_id": "soa_snapshot_grid_8192/aos_grid/fork",
        "directory_name": "soa_snapshot_grid_8192/aos_grid_fork",
        "title": "soa_snapshot_grid_8192/aos_grid/fork",
    }))
    def est(p):
        return {"confidence_interval": {"confidence_level": 0.95, "lower_bound": p * 0.95, "upper_bound": p * 1.05},
                "point_estimate": p, "standard_error": p * 0.01}
    (d / "estimates.json").write_text(json.dumps({
        "mean": est(163820.0), "median": est(163820.0), "median_abs_dev": est(800.0),
        "slope": est(163820.0), "std_dev": est(1200.0),
    }))
    assert load().main([str(tmp_path), "--group", "soa_snapshot_grid_8192", "--instances", str(N)]) == 0
    out = capsys.readouterr().out
    assert "aos_grid/fork: record 72 B, median 20.0 ns [19.0, 21.0], bound 70.3 ns, PASS" in out
    assert "per destination (8191 records)" in out


def test_soa_bound_fork_declared_over_all_instances_is_refused(tmp_path):
    case(tmp_path, "g", "soa_cartpole/fork", 96, 80.0, throughput={"Bytes": 96 * N})
    assert load().main([str(tmp_path), "--group", "g", "--instances", str(N)]) == 2


def test_soa_bound_fork_at_one_instance_is_refused(tmp_path):
    case(tmp_path, "g", "soa_cartpole/fork", 96, 80.0, throughput={"Bytes": 96})
    assert load().main([str(tmp_path), "--group", "g", "--instances", "1"]) == 2


def test_soa_bound_bad_input_is_refused(tmp_path, capsys):
    m = load()
    assert m.main([str(tmp_path), "--group", "missing", "--instances", str(N)]) == 2
    case(tmp_path, "odd", "soa_cartpole/fork", 96, 80.0, throughput={"Bytes": 96 * N + 1})
    assert m.main([str(tmp_path), "--group", "odd", "--instances", str(N)]) == 2
    case(tmp_path, "elems", "soa_cartpole/fork", 96, 80.0, throughput={"Elements": N})
    assert m.main([str(tmp_path), "--group", "elems", "--instances", str(N)]) == 2
    case(tmp_path, "noenv", "fork", 96, 80.0)
    assert m.main([str(tmp_path), "--group", "noenv", "--instances", str(N)]) == 2
    assert m.main([str(tmp_path), "--group", "g", "--instances", "0"]) == 2
    assert "soa_bound:" in capsys.readouterr().err
