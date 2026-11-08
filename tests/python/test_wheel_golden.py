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
FIELDS = list(tool.COMPARED)


def fixture_text(env_id="tessera/CartPole-v1", steps=6, reseed=4, auto="SameStep"):
    """A fixture whose digests come from a live pool, with its actions."""
    env = t.make(env_id, num_envs=4, seed=9, outputs="owned", chunk_size=2, auto_reset=tool.AUTO_RESET[auto], max_episode_steps=500)
    actions = np.ones((steps, 4, 1), np.int32).astype("<i4")
    head = [
        "# comment", "format=3", f"env_id={env_id}", "seed=9", "num_envs=4", "chunk=2", f"auto_reset={auto}",
        "max_episode_steps=500", f"steps={steps}", f"reseed={reseed}:[1, 2]:7", "fields=" + ",".join(FIELDS),
    ]  # fmt: skip
    same = auto == "SameStep"
    body = []
    raw = env._native.reset()
    for i in range(steps + 1):
        if i > 0:
            if i == reseed:
                env._native.reset(7, [1, 2])
            raw = env._native.step(actions[i - 1].tobytes())
        d = tool.digests(env, raw, same, DIGEST)
        body.append(f"{i} " + " ".join(d[f] for f in FIELDS))
    final = " ".join(f"{DIGEST(env.snapshot(i)):016x}" for i in range(4))
    env.close()
    return "\n".join([*head, *body, f"final {final}"]), actions.tobytes()


def play(text, actions, name="cartpole-v1-next-step", **kw):
    header, lines, final = tool.parse_fixture(text)
    args = {"workers": 0, "outputs": "owned", **kw}
    return tool.replay(name, header, lines, final, actions, make=t.make, digest64=DIGEST, **args)


def edit(text, prefix, change):
    return "\n".join(change(x) if x.startswith(prefix) else x for x in text.splitlines())


def test_wheel_golden_parse_reads_the_header_lines_and_final():
    text, _ = fixture_text()
    header, lines, final = tool.parse_fixture(text)
    assert header["env_id"] == "tessera/CartPole-v1" and tool.parse_reseed(header) == (4, [1, 2], 7)
    assert set(lines) == set(range(7)) and set(lines[0]) == set(FIELDS) and len(final) == 4


@pytest.mark.parametrize(
    "bad", ["format=2\nfields=a\nfinal 0\n", "format=3\n0 aa\nfinal 0\n", "format=3\nfields=a,b\n0 aa\nfinal 0\n", "format=3\nfields=a\n0 aa\n"]
)
def test_wheel_golden_parse_refuses_a_bad_fixture(bad):
    with pytest.raises(ValueError):
        tool.parse_fixture(bad)


def test_wheel_golden_every_real_fixture_parses_and_has_a_config():
    found = tool.fixtures()
    assert set(found) == set(tool.CONFIGS) and len(found) == 24
    for path in found.values():
        header, lines, final = tool.parse_fixture(path.read_text(encoding="utf-8"))
        assert tool.parse_reseed(header)[0] in lines and header["auto_reset"] in tool.AUTO_RESET
        assert set(tool.COMPARED) | set(tool.OMITTED) == set(lines[0]) and len(final) == int(header["num_envs"])


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("outputs", tool.OUTPUTS)
def test_wheel_golden_replay_matches_through_the_reseed_and_final(workers, outputs):
    text, actions = fixture_text()
    out = play(text, actions, workers=workers, outputs=outputs)
    assert out["mismatches"] == [] and out["calls"] == 7 and out["final_ok"] is True


@pytest.mark.parametrize("field", tool.COMPARED)
@pytest.mark.parametrize("t_changed", [2, 5])  # before and after the reseed
def test_wheel_golden_replay_detects_each_field(field, t_changed):
    text, actions = fixture_text()
    index = FIELDS.index(field) + 1

    def change(line):
        parts = line.split()
        parts[index] = "f" * 16
        return " ".join(parts)

    out = play(edit(text, f"{t_changed} ", change), actions)
    assert out["mismatches"] == [{"t": t_changed, "fields": [field]}]


def test_wheel_golden_replay_detects_a_changed_final_hash():
    text, actions = fixture_text()
    out = play(edit(text, "final ", lambda x: x[:-1] + ("0" if x[-1] != "0" else "1")), actions)
    assert out["final_ok"] is False and out["mismatches"] == []


def test_wheel_golden_replay_detects_a_different_reseed():
    text, actions = fixture_text()
    out = play(edit(text, "reseed=", lambda x: "reseed=4:[1, 2]:8"), actions)
    assert [m["t"] for m in out["mismatches"]][0] == 4 and "records" in out["mismatches"][0]["fields"]


def test_wheel_golden_terminal_obs_is_compared_on_same_step_only():
    # Next step has no terminal rows: the digest is of empty bytes.
    text, actions = fixture_text(auto="NextStep")
    header, lines, _ = tool.parse_fixture(text)
    assert lines[1]["terminal_obs"] == f"{DIGEST(b''):016x}"
    assert play(text, actions)["mismatches"] == []


def test_wheel_golden_report_lists_what_is_omitted_and_fails_on_a_bad_final(monkeypatch, tmp_path):
    results = [{"fixture": "x", "mismatches": [], "final_ok": False}]
    fixture = tmp_path / "x.golden"
    fixture.write_text("format=3\nfields=a\nfinal 0\n")
    (tmp_path / "x.actions").write_bytes(b"")
    monkeypatch.setattr(tool, "fixtures", lambda: {"x": fixture})
    monkeypatch.setattr(tool, "CONFIGS", {"x": {}})
    monkeypatch.setattr(tool, "WORKERS", (0,))
    monkeypatch.setattr(tool, "OUTPUTS", ("owned",))
    monkeypatch.setattr(tool, "replay", lambda *a: results[0])
    report = tool.run(tmp_path, t.make, DIGEST)
    assert report["passed"] is False and report["omitted_fields"] == list(tool.OMITTED)


def test_wheel_golden_replay_detects_changed_actions():
    text, actions = fixture_text()
    changed = bytes(4 * 4) + actions[4 * 4 :]
    assert play(text, changed)["mismatches"]


def test_wheel_golden_replay_refuses_truncated_actions():
    text, actions = fixture_text()
    with pytest.raises(ValueError, match="action bytes"):
        play(text, actions[:-4])


def test_wheel_golden_run_refuses_a_fixture_set_that_differs(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "CONFIGS", {"nope": {}})
    with pytest.raises(ValueError, match="differ"):
        tool.run(tmp_path, t.make, DIGEST)
