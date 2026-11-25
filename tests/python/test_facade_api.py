# SPDX-License-Identifier: MIT OR Apache-2.0
"""The facade's public-API snapshot (ADR-0094 item 7, #642):
`scripts/check_facade_api.py` against the committed `api/tessera-rl.api`,
its exclusions, the experimental switch (ADR-0094 Q1), the JSON format
guard, and edits that must change the snapshot (an added public item, a lost
auto trait) or must not (hidden and `_` items). Source-level mutants are in
the #642 record."""

import copy
import importlib.util
import json
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_facade_api.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_facade_api", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def api():
    return load_module()


@pytest.fixture(scope="module")
def docs(api):
    """Every crate's rustdoc JSON, built once with the pinned toolchain."""
    if shutil.which("cargo") is None:
        pytest.skip("facade-api: no cargo on PATH, the JSON cannot be built")
    out = api.build(None)
    return {
        name: json.loads((out / f"{name}.json").read_text())
        for name in (k.replace("-", "_") for k in (api.FACADE, *api.MEMBERS))
    }


def lines(api, docs, include=None):
    w = api.Walker(api.Crates(docs))
    if include is not None:
        old = api.INCLUDE_EXPERIMENTAL
        api.INCLUDE_EXPERIMENTAL = include
    try:
        w.module("tessera_rl", str(docs["tessera_rl"]["root"]), "tessera_rl")
    finally:
        if include is not None:
            api.INCLUDE_EXPERIMENTAL = old
    return w


def test_facade_api_snapshot_is_current(api, docs, tmp_path):
    for name, doc in docs.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    assert api.snapshot(tmp_path) == api.SNAPSHOT.read_text()


def test_facade_api_header_records_toolchain_format_and_switch():
    head = [line for line in (ROOT / "api" / "tessera-rl.api").read_text().splitlines() if line.startswith("#")]
    text = "\n".join(head)
    assert "# toolchain: " in text
    assert "# rustdoc JSON format_version: " in text
    assert "# INCLUDE_EXPERIMENTAL: False" in text


def test_facade_api_experimental_prefixes_each_match_and_toggle(api, docs):
    off = lines(api, docs, include=False)
    assert off.matched == set(api.EXPERIMENTAL), set(api.EXPERIMENTAL) - off.matched
    on = lines(api, docs, include=True)
    added = on.lines - off.lines
    assert any("tessera_rl::core::HostEnvironment" in line for line in added)
    assert any(line.startswith("macro tessera_rl::register_host") for line in added)
    assert not any("HostEnvironment" in line.split(":", 1)[0] for line in off.lines)


def test_facade_api_refuses_another_json_format(api, docs):
    other = copy.deepcopy(docs)
    other["tessera_rl_num"]["format_version"] = api.FORMAT_VERSION + 1
    with pytest.raises(SystemExit, match="toolchain bump"):
        api.Crates(other)


def add_fn(doc, name, hidden=False):
    """A public `fn name()` at the crate root of `doc`."""
    root = doc["index"][str(doc["root"])]
    key = str(10**9 + len(doc["index"]))
    doc["index"][key] = {
        "id": int(key),
        "crate_id": 0,
        "name": name,
        "visibility": "public",
        "attrs": ["#[doc(hidden)]"] if hidden else [],
        "inner": {
            "function": {
                "sig": {"inputs": [], "output": None, "is_c_variadic": False},
                "generics": {"params": [], "where_predicates": []},
                "header": {"is_const": False, "is_unsafe": False, "is_async": False, "abi": "Rust"},
                "has_body": True,
            }
        },
    }
    if not hidden:
        root["inner"]["module"]["items"].append(int(key))


def test_facade_api_added_public_item_changes_the_snapshot(api, docs):
    base = lines(api, docs).lines
    mutated = copy.deepcopy(docs)
    add_fn(mutated["tessera_rl_num"], "api_probe")
    new = lines(api, mutated).lines
    assert new - base == {"fn tessera_rl::num::api_probe: fn api_probe()"}


def test_facade_api_hidden_and_underscore_items_do_not_change_it(api, docs):
    base = lines(api, docs).lines
    mutated = copy.deepcopy(docs)
    # rustdoc leaves a hidden item out of the module (no --document-hidden-items).
    add_fn(mutated["tessera_rl_num"], "hidden_probe", hidden=True)
    add_fn(mutated["tessera_rl_num"], "_private_probe")
    assert lines(api, mutated).lines == base


def test_facade_api_lost_auto_trait_changes_the_snapshot(api, docs):
    doc = docs["tessera_rl_num"]
    seed = next(
        k for k, e in doc["paths"].items() if e["crate_id"] == 0 and e["path"] == ["tessera_rl_num", "identity", "Seed"]
    )
    base = lines(api, docs).lines
    mutated = copy.deepcopy(docs)
    impls = mutated["tessera_rl_num"]["index"][seed]["inner"]["struct"]["impls"]
    send = [
        i
        for i in impls
        if (mutated["tessera_rl_num"]["index"][str(i)]["inner"]["impl"].get("trait") or {}).get("path", "").endswith("Send")
    ]
    assert send, "Seed has a Send impl"
    for i in send:
        impls.remove(i)
    lost = base - lines(api, mutated).lines
    assert any("core::marker::Send for tessera_rl_num::identity::Seed" in line for line in lost), lost
