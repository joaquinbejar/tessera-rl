# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Tessera Viewer (ADR-0083 item 6, ADR-0085): a real server on
127.0.0.1 over a recorded CartPole replay."""

import http.client
import json
import subprocess
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest

import tessera_rl as t
from tessera_rl import replay, viewer

N = 2
STEPS = 12
CALLS = STEPS + 1  # the reset and the steps


def make():
    return t.make("tessera/CartPole-v1", num_envs=N, seed=5, outputs="owned", workers=0)


def record_file(path):
    env = make()
    with replay.record(env, path, keyframe_every=4, checkpoint_every=3):
        env.reset(seed=3)
        for k in range(STEPS):
            env.step(((np.arange(N) + k) % 2).astype(np.int32))
    env.close()


@pytest.fixture
def served(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    v = viewer.serve(path, env)
    yield v
    v.close()
    env.close()


def base(v):
    return f"http://127.0.0.1:{v.address[1]}"


def token(v):
    return v.url.split("?token=", 1)[1]


def get(url, *, method="GET", headers=None):
    """``(status, headers, body)``; HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        with err:
            return err.code, err.headers, err.read()


def api(v, route, **params):
    query = "&".join(f"{k}={val}" for k, val in {**params, "token": token(v)}.items())
    status, _, body = get(f"{base(v)}{route}?{query}")
    assert status == 200, status
    return json.loads(body)


def test_viewer_missing_token_403_empty(served):
    for route in ("/", "/viewer.js", "/meta", "/frame?i=0"):
        status, _, body = get(base(served) + route)
        assert (status, body) == (403, b"")


def test_viewer_wrong_token_403_empty(served):
    status, _, body = get(f"{base(served)}/meta?token=x{token(served)}")
    assert (status, body) == (403, b"")
    status, _, body = get(f"{base(served)}/index.html?token=")
    assert (status, body) == (403, b"")


def test_viewer_wrong_host_header_403(served):
    url = f"{base(served)}/meta?token={token(served)}"
    status, _, body = get(url, headers={"Host": f"evil.example:{served.address[1]}"})
    assert (status, body) == (403, b"")
    status, _, _ = get(url, headers={"Host": "127.0.0.1"})
    assert status == 403
    status, _, _ = get(url, headers={"Host": f"localhost:{served.address[1]}"})
    assert status == 200


def test_viewer_meta_fields_match_replay(served):
    meta = api(served, "/meta")
    assert meta["canvas"] == [600, 400]
    assert meta["num_envs"] == N and meta["agents"] == 1
    assert meta["keyframes"][0] == 0 and len(meta["keyframes"]) > 1
    assert meta["calls"] == CALLS and meta["calls_seen"] == CALLS


def test_viewer_frame_first_primitive_covers_canvas(served):
    frame = api(served, "/frame", i=3, instance=1)
    assert frame["i"] == 3 and frame["instance"] == 1
    first = frame["prims"][0]
    assert first[0] == "rect" and first[1:5] == [0, 0, 600, 400]
    assert len(first[5]) == 4


def test_viewer_step_fields_per_instance(served):
    step = api(served, "/step", i=2, instance=0)
    for key in ("rewards", "alive", "acting", "masks", "observations"):
        assert key in step
    assert len(step["rewards"]) == 1 and len(step["alive"]) == 1
    (leaf,) = step["observations"].values()
    assert leaf["shape"] == [1, 4] and not leaf["truncated"] and len(leaf["values"]) == 4


def test_viewer_backward_seek_matches_forward_seek(served, tmp_path):
    late = api(served, "/frame", i=CALLS - 1, instance=0)
    back = [api(served, "/frame", i=i, instance=0) for i in (9, 2, 0, 5)]
    back_steps = [api(served, "/step", i=i, instance=1) for i in (6, 1)]
    path = tmp_path / "run.trpl"
    for i, frame in zip((9, 2, 0, 5), back, strict=True):
        env = make()
        with viewer.serve(path, env) as fresh:
            assert api(fresh, "/frame", i=i, instance=0) == frame
        env.close()
    for i, step in zip((6, 1), back_steps, strict=True):
        env = make()
        with viewer.serve(path, env) as fresh:
            assert api(fresh, "/step", i=i, instance=1) == step
        env.close()
    assert api(served, "/frame", i=CALLS - 1, instance=0) == late


def test_viewer_bad_query_400(served):
    tok = token(served)
    for query in ("i=x", "i=-1", f"i={CALLS}", "", "i=1&i=2", "i=0&instance=9", "i=0&instance=a"):
        status, _, body = get(f"{base(served)}/frame?{query}&token={tok}")
        assert (status, body) == (400, b""), query


def test_viewer_post_405(served):
    status, _, body = get(f"{base(served)}/meta?token={token(served)}", method="POST")
    assert (status, body) == (405, b"")


def test_viewer_unknown_path_404_no_traversal(served):
    for route in ("/../viewer.py", "/viewer_static/index.html", "/viewer.py", "/%2e%2e/replay.py"):
        status, _, _ = get(f"{base(served)}{route}?token={token(served)}")
        assert status == 404, route


def test_viewer_static_files_served(served):
    for route, kind in (("/", "text/html"), ("/viewer.js", "text/javascript"), ("/viewer.css", "text/css")):
        status, headers, body = get(f"{base(served)}{route}?token={token(served)}")
        assert status == 200 and headers["Content-Type"].startswith(kind) and body
        assert b"http://" not in body and b"https://" not in body


def test_viewer_response_headers_safe(served):
    for url in (f"{base(served)}/meta?token={token(served)}", f"{base(served)}/meta"):
        _, headers, _ = get(url)
        assert headers["Referrer-Policy"] == "no-referrer"
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "Access-Control-Allow-Origin" not in headers


def test_viewer_bound_loopback_only(served):
    assert served.address[0] == "127.0.0.1"
    assert served.url.startswith(f"http://127.0.0.1:{served.address[1]}/?token=")
    assert token(served) not in repr(served)


def test_viewer_not_imported_by_package():
    code = "import sys, tessera_rl; print('tessera_rl.viewer' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_viewer_close_stops_server(tmp_path):
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    v = viewer.serve(path, env)
    port = v.address[1]
    v.close()
    v.close()
    with pytest.raises(OSError):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", "/meta")
            conn.getresponse()
        finally:
            conn.close()
    # The replay ended: the environment is free again.
    env.reset(seed=1)
    env.close()


def test_viewer_show_returns_iframe(tmp_path):
    pytest.importorskip("IPython")
    path = tmp_path / "run.trpl"
    record_file(path)
    env = make()
    frame = viewer.show(path, env, height=300)
    try:
        assert frame.src == frame.viewer.url and frame.height == 300
    finally:
        frame.viewer.close()
        env.close()
