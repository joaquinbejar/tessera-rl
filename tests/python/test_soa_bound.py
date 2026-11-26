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


def case(root, group, fid, record_bytes, ns_per_instance, throughput=None):
    d = root / group / fid.replace("/", "_") / "new"
    d.mkdir(parents=True)
    tp = {"Bytes": record_bytes * N} if throughput is None else throughput
    (d / "benchmark.json").write_text(json.dumps({"group_id": group, "function_id": fid, "throughput": tp}))
    total = ns_per_instance * N
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
