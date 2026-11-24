# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/pypi_attestations.py: PEP 740 attestations on PyPI (#350), offline."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pypi_attestations", ROOT / "scripts" / "pypi_attestations.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)


def _body(repository="joaquinbejar/tessera-rl", workflow="pypi.yml", kind="GitHub", attestations=({"x": 1},)):
    return json.dumps(
        {
            "version": 1,
            "attestation_bundles": [
                {
                    "publisher": {"kind": kind, "repository": repository, "workflow": workflow, "environment": "pypi"},
                    "attestations": list(attestations),
                }
            ],
        }
    ).encode()


def test_pypi_attestations_matching_bundle_passes():
    assert pa.judge(200, _body()) is None


@pytest.mark.parametrize(
    "status, body, reason",
    [
        (404, b"", "HTTP 404"),
        (200, b"<html>", "not JSON"),
        (200, json.dumps({"attestation_bundles": []}).encode(), "no attestation bundles"),
        (200, _body(repository="someone/fork"), "no bundle from GitHub"),
        (200, _body(workflow="other.yml"), "no bundle from GitHub"),
        (200, _body(kind="GitLab"), "no bundle from GitHub"),
        (200, _body(attestations=()), "no bundle from GitHub"),
    ],
)
def test_pypi_attestations_refuses_with_the_reason(status, body, reason):
    assert reason in pa.judge(status, body)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_pypi_attestations_polls_until_indexed():
    clock = _Clock()
    calls = []

    def fetch(url):
        calls.append(url)
        return (404, b"") if clock.t < 30 else (200, _body())

    failing = pa.check(["a.whl"], "0.4.0", fetch, timeout=300, now=clock.now, sleep=clock.sleep)
    assert failing == {}
    assert calls[0] == "https://pypi.org/integrity/tessera-rl/0.4.0/a.whl/provenance"
    assert len(calls) == 3


def test_pypi_attestations_stops_at_the_deadline_failing_closed():
    clock = _Clock()
    failing = pa.check(["a.whl", "b.whl"], "0.4.0", lambda url: (404, b""), timeout=60, now=clock.now, sleep=clock.sleep)
    assert failing == {"a.whl": "HTTP 404", "b.whl": "HTTP 404"}
    assert clock.t == 60
