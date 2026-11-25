# SPDX-License-Identifier: MIT OR Apache-2.0
"""Policy logits in the Viewer (#683, ADR-0083 item 6): the side file next
to a replay, its checks as untrusted input, and the live callable."""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import _viewer_logits as vl
from tessera_rl import replay, viewer

N = 2
STEPS = 6


def make():
    return t.make("tessera/CartPole-v1", num_envs=N, seed=5, outputs="owned", workers=0)


def logits_for(k):
    """Distinct logits per step ordinal ``k``: ``(N, agents, width)``."""
    env_width = 2
    return np.arange(N * env_width, dtype=np.float32).reshape(N, 1, env_width) + 10.0 * k


def record(path, *, log=True, fork_at=None, none_at=()):
    """A reset, STEPS steps, a second reset after step 2 and a fork after
    step ``fork_at``: calls are reset, s0, s1, s2, reset, s3, [fork], s4, s5."""
    env = make()
    ctx = viewer.logits_log(path, env) if log else None
    lg = ctx.__enter__() if ctx else None
    with replay.record(env, path, keyframe_every=3, checkpoint_every=2):
        env.reset(seed=3)
        for k in range(STEPS):
            env.step(((np.arange(N) + k) % 2).astype(np.int32))
            if lg is not None:
                lg.step(None if k in none_at else logits_for(k))
            if k == 2:
                env.reset(seed=4)
            if fork_at is not None and k == fork_at:
                env.fork(0, [1])
    if ctx:
        ctx.__exit__(None, None, None)
    env.close()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        with err:
            return err.code, err.read()


def api(v, route, **params):
    tok = v.url.split("?token=", 1)[1]
    query = "&".join(f"{k}={val}" for k, val in {**params, "token": tok}.items())
    status, body = get(f"http://127.0.0.1:{v.address[1]}{route}?{query}")
    assert status == 200, status
    return json.loads(body)


@pytest.fixture
def served_with(tmp_path):
    opened = []

    def serve(path, **kw):
        env = make()
        v = viewer.serve(path, env, **kw)
        opened.append((v, env))
        return v

    yield serve
    for v, env in opened:
        v.close()
        env.close()


def test_viewer_logits_side_file_leaves_replay_bytes_unchanged(tmp_path):
    a, b = tmp_path / "a.trpl", tmp_path / "b.trpl"
    record(a, log=True)
    record(b, log=False)
    assert a.read_bytes() == b.read_bytes()
    assert os.path.exists(vl.side_path(a)) and not os.path.exists(vl.side_path(b))
    assert not os.path.exists(vl.side_path(a) + ".tmp")


def test_viewer_logits_reset_and_fork_between_steps_map_calls_to_ordinals(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path, fork_at=3)
    v = served_with(path)
    meta = api(v, "/meta")
    assert meta["logits"] == {"source": "file", "note": None}
    # Calls: 0 reset, 1-3 steps 0-2, 4 reset, 5 step 3, 6 fork, 7-8 steps 4-5.
    assert meta["calls"] == 9
    ordinal_of_call = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4, 8: 5}
    for i in range(meta["calls"]):
        for inst in range(N):
            s = api(v, "/step", i=i, instance=inst)
            if i not in ordinal_of_call:
                assert "logits" not in s, (i, s)
                continue
            want = logits_for(ordinal_of_call[i])[inst, 0].tolist()
            assert s["logits"][0][0]["logits"] == pytest.approx(want), (i, inst)
            probs = s["logits"][0][0]["probs"]
            assert sum(probs) == pytest.approx(1.0)


def test_viewer_logits_none_step_has_no_field(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path, none_at=(1,))
    v = served_with(path)
    assert "logits" not in api(v, "/step", i=2, instance=0)
    assert "logits" in api(v, "/step", i=3, instance=0)


def test_viewer_logits_without_side_file_is_unchanged(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path, log=False)
    v = served_with(path)
    assert api(v, "/meta")["logits"] == {"source": None, "note": None}
    assert "logits" not in api(v, "/step", i=1, instance=0)


def test_viewer_logits_hash_mismatch_ignored_with_note(tmp_path, served_with):
    a, b = tmp_path / "a.trpl", tmp_path / "b.trpl"
    record(a)
    env = make()
    with replay.record(env, b):
        env.reset(seed=9)
        for k in range(STEPS):
            env.step(np.zeros(N, dtype=np.int32))
    env.close()
    os.replace(vl.side_path(a), vl.side_path(b))
    v = served_with(b)
    meta = api(v, "/meta")
    assert meta["logits"]["source"] is None
    assert "another recording" in meta["logits"]["note"]
    assert "logits" not in api(v, "/step", i=1, instance=0)


def rewrite(side, **changes):
    with np.load(side, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data.update(changes)
    for k in [k for k, v in changes.items() if v is None]:
        del data[k]
    with open(side, "wb") as f:
        np.savez(f, **data)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"format": np.array("tessera-logits/2")}, "format is not"),
        ({"shape": np.array([N, 1, 3])}, "shape is not"),
        ({"logits": np.zeros((STEPS, N, 1, 2), dtype=np.float64)}, "float32"),
        ({"logits": np.zeros((STEPS - 1, N, 1, 2), dtype=np.float32)}, "float32 of shape"),
        ({"ordinals": np.array([0, 1, 2, 3, 5, 4])}, "strictly increasing"),
        ({"ordinals": np.array([1, 2, 3, 4, 5, 6])}, "outside the replay"),
        ({"ordinals": np.array([0.0, 1, 2, 3, 4, 5])}, "integer"),
        ({"extra": np.array(1)}, "entries"),
        ({"replay_sha256": None}, "entries"),
    ],
)
def test_viewer_logits_malformed_side_file_refused_not_500(tmp_path, served_with, changes, reason):
    path = tmp_path / "run.trpl"
    record(path)
    rewrite(vl.side_path(path), **changes)
    v = served_with(path)
    meta = api(v, "/meta")
    assert meta["logits"]["source"] is None
    assert reason in meta["logits"]["note"], meta["logits"]["note"]
    assert "logits" not in api(v, "/step", i=1, instance=0)


def test_viewer_logits_pickled_object_array_refused(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path)
    rewrite(vl.side_path(path), logits=np.array([object()], dtype=object))
    v = served_with(path)
    assert "not a readable" in api(v, "/meta")["logits"]["note"]


def test_viewer_logits_garbage_file_refused(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path)
    with open(vl.side_path(path), "wb") as f:
        f.write(b"not a zip")
    v = served_with(path)
    assert "not a readable" in api(v, "/meta")["logits"]["note"]


def test_viewer_logits_explicit_missing_path_noted(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path, log=False)
    v = served_with(path, logits=tmp_path / "nope.npz")
    assert api(v, "/meta")["logits"]["note"] == "logits side file not found"


def test_viewer_logits_log_refuses_wrong_shape_and_writes_nothing_on_raise(tmp_path):
    path = tmp_path / "run.trpl"
    env = make()
    with pytest.raises(t.ArgumentError, match="shape"), viewer.logits_log(path, env) as lg:
        lg.step(np.zeros((N, 1, 3), dtype=np.float32))
    assert not os.path.exists(vl.side_path(path))
    env.close()


def test_viewer_logits_log_inside_record_refused(tmp_path):
    path = tmp_path / "run.trpl"
    env = make()
    with pytest.raises(t.ArgumentError, match="outside replay.record"):
        with replay.record(env, path), viewer.logits_log(path, env) as lg:
            env.reset(seed=1)
            env.step(np.zeros(N, dtype=np.int32))
            lg.step(logits_for(0))
    env.close()


def test_viewer_logits_callable_error_shown_in_panel(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path, log=False)

    def boom(i, instance):
        raise RuntimeError("policy gone")

    v = served_with(path, logits=boom)
    assert api(v, "/meta")["logits"]["source"] == "callable"
    s = api(v, "/step", i=1, instance=0)
    assert s["logits"] == {"error": "RuntimeError: policy gone"}
    # The handler still serves the next request.
    assert api(v, "/step", i=2, instance=1)["i"] == 2


def test_viewer_logits_callable_wrong_shape_is_an_error_entry(tmp_path, served_with):
    path = tmp_path / "run.trpl"
    record(path, log=False)
    v = served_with(path, logits=lambda i, inst: np.zeros(5))
    assert "expected" in api(v, "/step", i=1, instance=0)["logits"]["error"]


def test_viewer_logits_payload_masks_and_nonfinite():
    heads = (3, 2)
    row = np.array([1.0, np.nan, 2.0, np.inf, 0.0])
    mask = np.array([True, True, False, False, True])
    out = vl.payload(row, mask, heads)
    assert out[0]["logits"] == [1.0, None, 2.0]
    assert out[0]["legal"] == [True, True, False]
    assert out[0]["probs"] == pytest.approx([1.0, 0.0, 0.0])
    assert out[1]["logits"] == [None, 0.0]
    assert out[1]["probs"] == pytest.approx([0.0, 1.0])
    none_legal = vl.payload(np.array([1.0, 2.0]), np.array([False, False]), (2,))
    assert none_legal[0]["probs"] is None
    json.dumps(out)  # serves as JSON (no NaN)


def test_viewer_logits_live_callable_runs_on_request_thread_only():
    env = make()
    env.reset(seed=3)
    threads = []

    def fn(instance):
        threads.append(threading.current_thread().name)
        return np.full((1, 2), float(instance), dtype=np.float32)

    v = viewer.watch(env, logits=fn)
    try:
        out = {}
        client = threading.Thread(target=lambda: out.update(api(v, "/live", instance=1)))
        client.start()
        time.sleep(0.2)
        env.step(np.zeros(N, dtype=np.int32))
        client.join()
        assert not out.get("pending"), out
        assert out["logits"][0][0]["logits"] == [1.0, 1.0]
        main = threading.main_thread().name
        assert threads and all(name != main for name in threads)
        assert api(v, "/meta")["logits"]["source"] == "callable"
    finally:
        v.close()
        env.close()


def test_viewer_logits_watch_rejects_non_callable():
    env = make()
    try:
        with pytest.raises(ValueError, match="callable"):
            viewer.watch(env, logits="x.npz")
    finally:
        env.close()
