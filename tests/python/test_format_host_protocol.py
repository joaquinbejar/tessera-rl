# SPDX-License-Identifier: MIT OR Apache-2.0
"""Host protocols 1 and 2 (SPEC 14.4, ADR-0065, ADR-0070) against their
committed surface fixture ``fixtures/formats/host_protocol.json``: for each
protocol, the members a host object must have, the layout keys it must
return and the outputs every reset and step must return. A protocol is a
runtime contract, not a file, so its fixture is this surface: the library
must still require exactly it, a host object that has it is accepted at
protocols 1 and 2, and an unknown protocol is refused with the error naming
it. A changed surface is resolved by a new protocol that keeps this one, or
by a fix; the fixture is regenerated only by
``test_format_host_protocol_write_fixture`` with ``TESSERA_BLESS_FORMATS=1``."""

import functools
import json
import os
from pathlib import Path

import pytest

import tessera_rl
from tessera_rl import _host
from fake_host import FakeHost, install

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "formats" / "host_protocol.json"
FAKE = "fake/Counter-v1"


def surface() -> dict:
    """The host protocol surface this library requires, as the fixture stores it."""
    base = {
        "members": list(_host._MEMBERS),
        "layout_keys": list(_host._LAYOUT_KEYS),
        "outputs": list(_host._OUTPUT_KEYS),
    }
    return {
        "protocols": {
            "1": base,
            "2": {
                **base,
                "layout_keys": [*base["layout_keys"], "refuses_agents"],
                "outputs": [*base["outputs"], *_host._REFUSAL_KEYS],
            },
        }
    }


@pytest.mark.skipif(os.environ.get("TESSERA_BLESS_FORMATS") != "1", reason="writes the fixture")
def test_format_host_protocol_write_fixture():
    FIXTURE.write_text(json.dumps(surface(), indent=2, sort_keys=True) + "\n")


def test_format_host_protocol_fixture_is_the_library_surface():
    assert _host.PROTOCOLS == (1, 2)
    assert json.loads(FIXTURE.read_text()) == surface()


@pytest.mark.parametrize("protocol", ["1", "2"])
def test_format_host_protocol_fixture_protocols_are_accepted(monkeypatch, protocol):
    spec = json.loads(FIXTURE.read_text())["protocols"][protocol]
    refuses = protocol == "2"
    obj = FakeHost(num_envs=2, protocol=int(protocol), refuses=refuses)
    # A host object with the fixture's surface...
    assert all(hasattr(obj, m) for m in spec["members"])
    assert set(spec["layout_keys"]) <= set(obj.layout())
    # ...passes the protocol check and wraps.
    _host.check(obj)
    install(monkeypatch, {FAKE: [("fake-dist", functools.partial(
        FakeHost, protocol=int(protocol), refuses=refuses))]})
    tessera_rl.make(FAKE).close()


@pytest.mark.parametrize("protocol", [0, 3, True])
def test_format_host_protocol_unknown_protocol_is_refused(monkeypatch, protocol):
    install(monkeypatch, {FAKE: [("fake-dist", functools.partial(FakeHost, protocol=protocol))]})
    with pytest.raises(tessera_rl.ConfigError, match=f"host protocol {protocol}"):
        tessera_rl.make(FAKE)
