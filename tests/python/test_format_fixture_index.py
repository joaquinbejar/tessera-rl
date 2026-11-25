# SPDX-License-Identifier: MIT OR Apache-2.0
"""The index of versioned formats, ``tests/format_fixtures.toml`` (#641):
every listed fixture exists and is not empty, every listed test is defined
in its file, and every version constant still reads as listed."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = tomllib.loads((ROOT / "tests" / "format_fixtures.toml").read_text())


def test_format_fixture_index_lists_every_format():
    names = [f["name"] for f in INDEX["format"]]
    assert len(names) == len(set(names))
    assert {"persisted record", "persisted record with assets", "replay", "canonical config",
            "league state", "frozen math", "host protocol"} <= set(names)


def test_format_fixture_index_fixtures_tests_and_constants_exist():
    for f in INDEX["format"]:
        for path in f["fixtures"]:
            p = ROOT / path
            assert p.is_file() and p.stat().st_size > 0, f"{f['name']}: {path}"
        source = (ROOT / f["tests"]["file"]).read_text()
        for name in f["tests"]["names"]:
            assert re.search(rf"\bdef {name}\b|\bfn {name}\b", source), f"{f['name']}: {name}"
        constant = (ROOT / f["constant"]["file"]).read_text()
        assert f["constant"]["text"] in constant, f"{f['name']}: {f['constant']['text']}"
