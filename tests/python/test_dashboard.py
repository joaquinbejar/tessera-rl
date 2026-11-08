"""The benchmark dashboard's admission, grouping and pages (synthetic data)."""

import copy
import importlib.util
import json
import pathlib
import shutil
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DIR = _ROOT / "tools" / "dashboard"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("dashboard_build", _DIR / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)
import grouping
import render
import schema
import status as st

SENTINEL = 271828.5
NAME = "rust/x/n1"


def report():
    stat = {"median": SENTINEL, "ci95": [1.0, 2.0], "n": 10, "samples": []}
    run = {
        "profile": "release",
        "cpu": "SyntheticCPU",
        "pinning": {"process_affinity": "1"},
        "method": {"warmup_s": 2.0, "reps": 10},
        "options": {"seed": 1},
        "plan_fingerprint": "ab",
        "samples": dict(stat, n=10),
    }
    return {
        "schema": "tessera-bench/3",
        "quick": False,
        "runs": 1,
        "method": {"warmup_s": 2.0, "reps": 10, "rep_s": 1.0, "batches": 2000, "runs_for_claims": 1},
        "pinning": {"applied": True, "cpu_set": [1]},
        "qualification": {"machine": "ref-syn", "exclusive_window": True},
        "environment": {
            "machine": "ref-syn",
            "machine_record": {"status": "reference", "cpu_id": "SyntheticCPU"},
            "hardware": {"cpu": "SyntheticCPU", "arch": "x", "logical_cpus": 2, "os": "SynOS 1", "frequency_policy": "fixed"},
            "software": {"git": "c" * 40, "git_dirty": False, "native_build_profile": "release", "rustc": "r1", "python": "3", "numpy": "2", "tessera_rl": "0"},
        },
        "load_average": {"before": [0.0], "after": [0.0]},
        "results": {NAME: {"config": {"name": NAME, "workers": 0, "metric": "throughput"}, "metric": "throughput", "samples": stat, "runs": [run]}},
        "derived": {},
        "valid_for_claims": True,
        "claim_check": {"valid": True, "reasons": []},
    }


def seal(rep, **extra):
    return {
        "schema": schema.ENVELOPE,
        "kind": "wall",
        "report_sha256": schema.report_hash(rep),
        "admitted_by": {"pr": 7, "approved_by": ["reviewer"]},
        "measured_at": "2026-01-01T00:00:00Z",
        "report": rep,
        **extra,
    }


@pytest.fixture
def root(tmp_path):
    for rel in ("tools/bench/configs.toml", "compat/machines.toml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_ROOT / rel, tmp_path / rel)
    (tmp_path / "benchmarks/results").mkdir(parents=True)
    return tmp_path


def put(root, name, doc):
    (root / "benchmarks/results" / name).write_text(json.dumps(doc), encoding="utf-8")


def verdict(doc):
    return st.judge("f.json", json.dumps(doc))


def test_status_complete_report_is_valid():
    e = verdict(seal(report()))
    assert (e.status, e.reasons) == (st.VALID, [])


def test_canonical_is_sorted_compact_ascii():
    assert schema.canonical({"b": "é", "a": [1, 2]}) == b'{"a":[1,2],"b":"\\u00e9"}'


def mutate(path, value):
    def apply(rep):
        at = rep
        for k in path[:-1]:
            at = at[k]
        at[path[-1]] = value
    return apply


SW = ("environment", "software")


@pytest.mark.parametrize(
    "change",
    [
        mutate((*SW, "git_dirty"), True),
        mutate((*SW, "git"), "unknown"),
        mutate(("valid_for_claims",), False),
        mutate(("claim_check", "valid"), False),
        mutate(("schema",), "tessera-bench/9"),
    ],
)
def test_status_report_defect_is_invalid(change):
    rep = report()
    change(rep)
    assert verdict(seal(rep)).status == st.INVALID


def test_status_nan_in_report_is_invalid():
    doc = seal(report())
    doc["report"]["quick"] = float("nan")
    e = verdict(doc)
    assert e.status == st.INVALID and "NaN" in e.reasons[0]


def test_status_changed_report_fails_hash():
    doc = seal(report())
    doc["report"]["runs"] = 2
    assert "report_sha256 does not match the report" in verdict(doc).reasons


def test_status_verdict_uses_embedded_record():
    rep = report()
    rep["environment"]["machine_record"]["status"] = "candidate"
    rep["valid_for_claims"] = False
    rep["claim_check"]["valid"] = False
    e = verdict(seal(rep))
    assert e.status == st.INDICATIVE and "not 'reference'" in e.reasons[0]


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda d: d.update(schema="other"), "unknown schema"),
        (lambda d: d["admitted_by"].update(pr=0), "pull request"),
        (lambda d: d["admitted_by"].update(pr=True), "pull request"),
        (lambda d: d["admitted_by"].update(approved_by=[]), "approved_by"),
        (lambda d: d.update(kind="instructions"), "unknown kind"),
        (lambda d: d.update(measured_at="yesterday"), "measured_at"),
        (lambda d: d.update(fixture=True), "fixture"),
    ],
)
def test_status_envelope_refused(edit, reason):
    doc = seal(report())
    edit(doc)
    e = verdict(doc)
    assert e.status == st.INVALID and reason in " ".join(e.reasons)


def test_status_not_json_is_invalid():
    assert st.judge("f.json", "{").status == st.INVALID


def statuses(root):
    return {e.name: e for e in st.load(root)}


def test_supersession_marks_replaced(root):
    put(root, "a.json", seal(report()))
    put(root, "b.json", seal(report(), supersedes="a.json"))
    got = statuses(root)
    assert (got["a.json"].status, got["b.json"].status) == (st.SUPERSEDED, st.VALID)


@pytest.mark.parametrize(
    ("links", "bad", "text"),
    [
        ({"a.json": "missing.json"}, ["a.json"], "does not exist"),
        ({"a.json": "a.json"}, ["a.json"], "itself"),
        ({"a.json": "b.json", "b.json": "a.json"}, ["a.json", "b.json"], "cycle"),
        ({"a.json": "c.json", "b.json": "c.json"}, ["a.json", "b.json"], "also supersedes"),
    ],
)
def test_supersession_refused(root, links, bad, text):
    for name in ("a.json", "b.json", "c.json"):
        extra = {"supersedes": links[name]} if name in links else {}
        put(root, name, seal(report(), **extra))
    got = statuses(root)
    for name in bad:
        assert got[name].status == st.INVALID and text in " ".join(got[name].reasons)


def test_supersession_by_invalid_file_changes_nothing(root):
    dirty = report()
    dirty["environment"]["software"]["git_dirty"] = True
    put(root, "a.json", seal(report()))
    put(root, "b.json", seal(dirty, supersedes="a.json"))
    got = statuses(root)
    assert (got["a.json"].status, got["b.json"].status) == (st.VALID, st.INVALID)


def test_admit_refuses_file_outside_results(root):
    outside = root / "elsewhere.json"
    outside.write_text(json.dumps(seal(report())))
    e = st.admit(outside, root)
    assert e.status == st.INVALID and "outside" in e.reasons[0]


def test_admit_refuses_symlink_out_of_results(root):
    outside = root / "elsewhere.json"
    outside.write_text(json.dumps(seal(report())))
    (root / "benchmarks/results/l.json").symlink_to(outside)
    assert st.load(root)[0].status == st.INVALID


def key(rep):
    return grouping.comparison_key(rep, NAME, "samples")


E, H = ("environment",), ("environment", "hardware")
KEY_CHANGES = [
    (("machine", "environment", "machine"), "other"),
    (("cpu", *H, "cpu"), "other"),
    (("cpu_id", *E, "machine_record", "cpu_id"), "other"),
    (("arch", *H, "arch"), "other"),
    (("os", *H, "os"), "other"),
    (("frequency_policy", *H, "frequency_policy"), "other"),
    (("software", *SW, "rustc"), "other"),
    (("software", *SW, "python"), "other"),
    (("software", *SW, "numpy"), "other"),
    (("software", *SW, "native_build_profile"), "debug"),
    (("cpu_set", "pinning", "cpu_set"), [2]),
    (("warmup_s", "method", "warmup_s"), 9.0),
    (("reps", "method", "reps"), 99),
    (("rep_s", "method", "rep_s"), 9.0),
    (("batches", "method", "batches"), 9),
    (("runs", "runs"), 99),
]


def test_key_change_list_covers_every_field():
    extra = {"kind", "configuration", "config", "work", "statistic"}
    assert {c[0][0] for c in KEY_CHANGES} | extra == {k for k, _ in key(report())}


@pytest.mark.parametrize(("spec", "value"), KEY_CHANGES, ids=lambda s: s[0] if isinstance(s, tuple) else None)
def test_key_field_splits_group(spec, value):
    a, b = report(), report()
    mutate(spec[1:], value)(b)
    assert spec[0] in grouping.differing(key(a), key(b))


def test_key_ignores_commit():
    a, b = report(), report()
    b["environment"]["software"]["git"] = "d" * 40
    assert key(a) == key(b)


@pytest.mark.parametrize("field", ["config", "work", "statistic", "configuration"])
def test_key_identity_fields_split_group(field):
    a, b = report(), report()
    name, stat = NAME, "samples"
    if field == "config":
        b["results"][NAME]["config"]["workers"] = 1
    elif field == "work":
        b["results"][NAME]["runs"][0]["plan_fingerprint"] = "cd"
    elif field == "statistic":
        stat = "other"
    else:
        name = "rust/y"
        b["results"][name] = copy.deepcopy(b["results"][NAME])
    assert field in grouping.differing(key(a), grouping.comparison_key(b, name, stat))


def index(root):
    return (build.build(root) / "index.html").read_text()


def test_page_empty_state(root):
    text = index(root)
    assert render.EMPTY in text and render.NO_REFERENCE in text
    assert "rust/cartpole/inline/n1024" in text


def test_page_is_byte_identical_twice(root):
    put(root, "a.json", seal(report()))
    assert index(root) == index(root)


def test_page_shows_valid_with_declared_window(root):
    put(root, "a.json", seal(report()))
    text = index(root)
    assert str(SENTINEL) in text and render.WINDOW in text and render.EMPTY not in text


def test_page_invalid_numbers_absent_and_no_copy(root):
    rep = report()
    rep["environment"]["software"]["git_dirty"] = True
    put(root, "a.json", seal(rep))
    text = index(root)
    assert str(SENTINEL) not in text and "dirty" in text
    assert not (root / "target/dashboard/results/a.json").exists()


def test_page_indicative_only_on_its_own_page(root):
    rep = report()
    rep["quick"] = True
    rep["valid_for_claims"] = False
    rep["claim_check"]["valid"] = False
    put(root, "a.json", seal(rep))
    text = index(root)
    assert str(SENTINEL) not in text
    assert str(SENTINEL) in (root / "target/dashboard/report-a.json.html").read_text()


def test_page_derived_invalid_has_no_number_and_others_stay(root):
    rep = report()
    rep["derived"] = {"bad": {"value": None, "invalid": "different plans"}, "good": {"value": 0.25, "ci95": [0.2, 0.3]}}
    put(root, "a.json", seal(rep))
    text = index(root)
    assert "invalid: different plans" in text and "0.25" in text and str(SENTINEL) in text


def test_page_unknown_metric_is_not_displayed(root):
    rep = report()
    rep["results"][NAME]["metric"] = "novel"
    put(root, "a.json", seal(rep))
    text = index(root)
    assert "not displayed" in text and str(SENTINEL) not in text


def test_page_two_hosts_never_share_a_table(root):
    other = report()
    other["environment"]["machine"] = "ref-two"
    other["qualification"]["machine"] = "ref-two"
    put(root, "a.json", seal(report()))
    put(root, "b.json", seal(other))
    tables = index(root).split("<table>")
    assert not any("a.json" in t.split("</table>")[0] and "b.json" in t.split("</table>")[0] for t in tables)


def test_page_newest_is_greatest_instant_then_name():
    def entry(name, at):
        return st.Entry(name, st.VALID, [], {"measured_at": at})
    a, b, c = entry("a", "2026-01-02T00:00:00Z"), entry("b", "2026-01-02T00:00:00Z"), entry("c", "2026-01-01T00:00:00Z")
    assert render.newest([c, a, b]) is b


def test_page_fixed_copy_has_no_claims(root):
    put(root, "a.json", seal(report()))
    text = index(root).lower()
    assert not any(word in text for word in ("faster", "fastest", " sps"))


def test_page_escapes_file_content(root):
    rep = report()
    rep["environment"]["machine"] = "<script>"
    rep["qualification"]["machine"] = "<script>"
    put(root, "a.json", seal(rep))
    assert "<script>" not in index(root)


def test_check_fails_on_invalid_file(root):
    put(root, "a.json", {"schema": "x"})
    assert build.check(root) == 1


def test_check_passes_on_committed_results():
    assert build.check(_ROOT) == 0
