# SPDX-License-Identifier: MIT OR Apache-2.0
"""tools/wheel_golden.py: fixture parsing and what the replay detects. The
real anchor runs through `make wheel-golden`."""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_TOOL = pathlib.Path(__file__).parents[2] / "tools" / "wheel_golden.py"
_spec = importlib.util.spec_from_file_location("wheel_golden", _TOOL)
tool = importlib.util.module_from_spec(_spec)
sys.modules["wheel_golden"] = tool
_spec.loader.exec_module(tool)

xxhash = pytest.importorskip("xxhash")
import tessera_rl as t  # noqa: E402

DIGEST = xxhash.xxh3_64_intdigest
FIELDS = ["records", *tool.COMPARED]


def fixture_text(env_id="tessera/CartPole-v1", steps=6, reseed=4):
    """A fixture whose digests come from a live pool, with its actions."""
    env = t.make(env_id, num_envs=4, seed=9, outputs="owned", chunk_size=2, auto_reset="next_step", max_episode_steps=500)
    actions = np.ones((steps, 4, 1), np.int32)
    rows = [env.reset()]
    rows += [env.step(a) for a in actions]
    env.close()
    head = [
        "# comment", "format=3", f"env_id={env_id}", "seed=9", "num_envs=4", "chunk=2", "auto_reset=NextStep",
        "max_episode_steps=500", f"steps={steps}", f"reseed={reseed}:[1]:7", "fields=" + ",".join(FIELDS),
    ]  # fmt: skip
    body = []
    for i, s in enumerate(rows):
        d = tool.digests(s, DIGEST)
        body.append(f"{i} {'0' * 16} " + " ".join(d[f] for f in tool.COMPARED))
    return "\n".join([*head, *body, "final 0"]), actions.astype("<i4").tobytes()


def play(text, actions, name="cartpole-v1-next-step", **kw):
    header, lines = tool.parse_fixture(text)
    args = {"workers": 0, "outputs": "owned", **kw}
    return tool.replay(name, header, lines, actions, make=t.make, digest64=DIGEST, **args)


def test_wheel_golden_parse_reads_the_header_and_lines():
    text, _ = fixture_text()
    header, lines = tool.parse_fixture(text)
    assert header["env_id"] == "tessera/CartPole-v1" and tool.reseed_step(header) == 4
    assert set(lines) == set(range(7)) and set(lines[0]) == set(FIELDS)


@pytest.mark.parametrize("bad", ["format=2\nfields=a\n", "format=3\n0 aa\n", "format=3\nfields=a,b\n0 aa\n"])
def test_wheel_golden_parse_refuses_a_bad_fixture(bad):
    with pytest.raises(ValueError):
        tool.parse_fixture(bad)


def test_wheel_golden_every_real_fixture_parses_and_has_a_config():
    found = tool.fixtures()
    assert set(found) == set(tool.CONFIGS) and len(found) == 24
    for path in found.values():
        header, lines = tool.parse_fixture(path.read_text(encoding="utf-8"))
        assert tool.reseed_step(header) - 1 in lines and header["auto_reset"] in tool.AUTO_RESET


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("outputs", tool.OUTPUTS)
def test_wheel_golden_replay_matches_and_stops_before_the_reseed(workers, outputs):
    text, actions = fixture_text()
    out = play(text, actions, workers=workers, outputs=outputs)
    assert out["mismatches"] == [] and out["calls"] == 4


@pytest.mark.parametrize("field", tool.COMPARED)
def test_wheel_golden_replay_detects_each_field(field):
    text, actions = fixture_text()
    index = FIELDS.index(field) + 1
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("2 "):
            parts = line.split()
            parts[index] = "f" * 16
            lines[i] = " ".join(parts)
    out = play("\n".join(lines), actions)
    assert out["mismatches"] == [{"t": 2, "fields": [field]}]


def test_wheel_golden_replay_detects_changed_actions():
    text, actions = fixture_text()
    changed = bytes(4 * 4) + actions[4 * 4 :]
    assert play(text, changed)["mismatches"]


def test_wheel_golden_replay_refuses_truncated_actions():
    text, actions = fixture_text()
    with pytest.raises(ValueError, match="action bytes"):
        play(text, actions[:-4])


def test_wheel_golden_a_digest_after_the_reseed_is_not_compared():
    text, actions = fixture_text()
    lines = ["5 " + " ".join(["f" * 16] * len(FIELDS)) if x.startswith("5 ") else x for x in text.splitlines()]
    assert play("\n".join(lines), actions)["mismatches"] == []


def test_wheel_golden_run_refuses_a_fixture_set_that_differs(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "CONFIGS", {"nope": {}})
    with pytest.raises(ValueError, match="differ"):
        tool.run(tmp_path, t.make, DIGEST)
