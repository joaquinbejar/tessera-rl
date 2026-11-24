# SPDX-License-Identifier: MIT OR Apache-2.0
"""Live watch (ADR-0083 item 7, issue #378 stage 4d): ``Env.watch()`` and
``tessera_rl.viewer.watch``. The pool serves a requested record at its next
batch boundary; a second thread takes it while the main thread steps."""

import json
import logging
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import viewer

N = 4
ENV = "tessera/CartPole-v1"


def make(**kwargs):
    return t.make(ENV, num_envs=N, seed=5, outputs="owned", workers=0, **kwargs)


def actions(k):
    return ((np.arange(N) + k) % 2).astype(np.int32)


def test_watch_second_thread_while_main_steps_records_match_snapshots():
    env = make()
    handle = env.watch()
    stop = threading.Event()
    got = []

    def view():
        while not stop.is_set():
            # Take first: a request from `READY` replaces the record.
            taken = handle.take()
            if taken is not None:
                got.append(taken)
            try:
                handle.request(1)
            except t.ConcurrencyError:
                pass
            time.sleep(0)

    snapshots = {}
    try:
        viewer_thread = threading.Thread(target=view)
        viewer_thread.start()
        env.reset(seed=3)
        snapshots[1] = env.snapshot(1)  # the first boundary since attach
        for k in range(300):
            env.step(actions(k))
            snapshots[k + 2] = env.snapshot(1)
        stop.set()
        viewer_thread.join()
        assert got, "no record was served"
        for generation, record in got:
            assert record == snapshots[generation], generation
    finally:
        stop.set()
        handle.close()
        env.close()


def test_watch_shadow_frames_equal_env_render_at_same_generation():
    env = make()
    handle = env.watch()
    shadow = t.make(ENV, num_envs=1, seed=0, outputs="owned", workers=0)
    try:
        env.reset(seed=3)
        for k in range(5):
            handle.request(2)
            env.step(actions(k))
            generation, record = handle.take()
            assert generation == k + 2
            shadow.import_record(0, record)
            assert np.array_equal(shadow.render([0])[0], env.render([2])[0])
            assert shadow.scene(0) == env.scene(2)
    finally:
        handle.close()
        shadow.close()
        env.close()


def test_watch_pipelined_serves_after_the_halfs_recv():
    env = t.make(ENV, num_envs=8, seed=5, outputs="owned", mode="pipelined", chunk_size=2, workers=1)
    handle = env.watch()
    try:
        env.reset(seed=1)
        a, b = env.halves
        handle.request(b.start)
        env.send(0, np.zeros(len(a), dtype=np.int32))
        env.send(1, np.zeros(len(b), dtype=np.int32))
        first = env.recv()
        assert first.half == 0
        assert handle.take() is None
        second = env.recv()
        assert second.half == 1
        assert handle.take() is not None
    finally:
        handle.close()
        env.close()


def test_watch_second_watch_raises_concurrency_until_closed():
    env = make()
    try:
        handle = env.watch()
        with pytest.raises(t.ConcurrencyError):
            env.watch()
        handle.close()
        env.watch().close()
    finally:
        env.close()


def test_watch_detached_after_env_close():
    env = make()
    handle = env.watch()
    env.reset(seed=1)
    handle.request(0)
    env.close()
    with pytest.raises(t.ArgumentError, match="detached"):
        handle.take()
    with pytest.raises(t.ArgumentError, match="detached"):
        handle.request(0)
    handle.close()


def test_watch_stopped_training_take_is_none_without_blocking():
    env = make()
    handle = env.watch()
    try:
        env.reset(seed=1)
        handle.request(0)
        start = time.monotonic()
        for _ in range(1000):
            assert handle.take() is None
        assert time.monotonic() - start < 5
        with pytest.raises(t.ConcurrencyError):
            handle.request(0)
    finally:
        handle.close()
        env.close()


# The live Viewer.


def base(v):
    return f"http://127.0.0.1:{v.address[1]}"


def token(v):
    return v.url.split("?token=", 1)[1]


def get(url, *, method="GET", headers=None):
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        with err:
            return err.code, err.headers, err.read()


def live(v, instance=0):
    status, _, body = get(f"{base(v)}/live?instance={instance}&token={token(v)}")
    assert status == 200, status
    return json.loads(body)


@pytest.fixture
def watched():
    env = make()
    env.reset(seed=3)
    v = viewer.watch(env)
    yield v, env
    v.close()
    env.close()


def test_watch_viewer_live_frame_equals_env_scene(watched):
    v, env = watched
    out = {}
    client = threading.Thread(target=lambda: out.update(live(v, 1)))
    client.start()
    time.sleep(0.2)
    env.step(actions(0))
    client.join()
    assert not out.get("pending"), out
    assert out["instance"] == 1 and out["generation"] >= 1
    want = json.loads(json.dumps([viewer._prim(p) for p in env.scene(1)]))
    assert out["prims"] == want


def test_watch_viewer_stopped_training_answers_pending(watched):
    v, _ = watched
    start = time.monotonic()
    assert live(v, 0) == {"pending": True}
    assert time.monotonic() - start >= 1.9


def test_watch_viewer_meta_is_live_and_replay_routes_404(watched):
    v, _ = watched
    status, _, body = get(f"{base(v)}/meta?token={token(v)}")
    assert status == 200 and json.loads(body)["live"] is True
    status, _, _ = get(f"{base(v)}/frame?i=0&token={token(v)}")
    assert status == 404


def test_watch_viewer_live_security_same_as_other_endpoints(watched):
    v, _ = watched
    status, _, body = get(f"{base(v)}/live?instance=0")
    assert (status, body) == (403, b"")
    status, _, body = get(f"{base(v)}/live?instance=0&token=x{token(v)}")
    assert (status, body) == (403, b"")
    url = f"{base(v)}/live?instance=0&token={token(v)}"
    status, _, body = get(url, headers={"Host": f"evil.example:{v.address[1]}"})
    assert (status, body) == (403, b"")
    status, _, _ = get(url, method="POST")
    assert status == 405
    status, headers, _ = get(f"{base(v)}/live?instance={N}&token={token(v)}")
    assert status == 400
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers.get("Access-Control-Allow-Origin") is None


def test_watch_viewer_500_logs_type_and_message_without_token(watched, caplog):
    v, _ = watched
    secret = token(v)

    def boom(instance):
        raise RuntimeError(f"injected failure near token={secret}")

    v._session.live = boom
    with caplog.at_level(logging.ERROR, logger="tessera_rl.viewer"):
        status, _, body = get(f"{base(v)}/live?instance=0&token={secret}")
    assert (status, body) == (500, b"")
    records = [r for r in caplog.records if r.name == "tessera_rl.viewer"]
    assert records, "the 500 was not logged"
    for record in records:
        assert record.levelno == logging.ERROR
        message = record.getMessage()
        assert "RuntimeError" in message
        assert secret not in message
        assert "/live" not in message and "instance=" not in message
