# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Tessera Viewer (ADR-0083 item 6, ADR-0085): a local web page that
steps through a replay file, drawing each instance's scene and showing its
rewards, flags, masks and observations.

The backend is Python's standard library (``http.server``); the front end is
vendored HTML, CSS and JavaScript drawing on a Canvas 2D context, with no
build step and no external requests. This module is not imported by
``import tessera_rl`` and opens no socket until :func:`serve` or :func:`show`
is called.

``http.server`` is not a production server. The Viewer is a localhost,
single-user tool, guarded as follows (ADR-0085 item 2):

- bound to ``127.0.0.1`` only;
- a per-session token (``secrets.token_urlsafe(32)``) is required on every
  request, static files included, and compared with ``hmac.compare_digest``;
  the URL :attr:`Viewer.url` carries it;
- a request whose ``Host`` header is not ``127.0.0.1:<port>`` or
  ``localhost:<port>`` is refused (DNS rebinding);
- a refused request gets ``403`` with no body;
- responses carry ``Referrer-Policy: no-referrer``, ``Cache-Control:
  no-store`` and ``X-Content-Type-Options: nosniff``, and never
  ``Access-Control-Allow-Origin``;
- nothing is logged, so the token never reaches a log;
- the client never names a file: the server serves the one replay its caller
  chose and three fixed static files.

Requests are served on several threads, but every replay and pool operation
runs under one lock (ADR-0085 item 3), so they serialize on the pool.

The environment ``env`` is driven by a stepped replay (:func:`tessera_rl.
replay.steps`) while the Viewer is open: calls that change it raise until
:meth:`Viewer.close`. The caller still owns ``env`` and closes it.

Endpoints (``GET`` only, all token-checked): ``/meta``; ``/frame?i=&instance=``
(the scene after call ``i``, zero-based); ``/step?i=&instance=`` (rewards,
alive, acting, masks and observation leaves after call ``i``).
"""

from __future__ import annotations

import hmac
import importlib.resources
import json
import math
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from tessera_rl import replay as _replay

__all__ = ["Viewer", "serve", "show"]

_HOST = "127.0.0.1"
#: Observation leaves larger than this many numbers are reported by shape only.
_LEAF_LIMIT = 4096
_STATIC = {
    "index.html": "text/html; charset=utf-8",
    "viewer.js": "text/javascript; charset=utf-8",
    "viewer.css": "text/css; charset=utf-8",
}


class _BadRequest(Exception):
    pass


def _finite(values):
    # JSON has no NaN or infinity; report them as null.
    if isinstance(values, list):
        return [_finite(v) for v in values]
    if isinstance(values, float) and not math.isfinite(values):
        return None
    return values


def _prim(p) -> list:
    if p[0] == "polygon":
        return ["polygon", [[x, y] for x, y in p[1]], list(p[2])]
    return [p[0], *p[1:-1], list(p[-1])]


class _Session:
    """The replay cursor over ``env``; every method runs under ``lock``."""

    def __init__(self, path, env) -> None:
        self.path = os.fspath(path)
        self.env = env
        self.lock = threading.Lock()
        self.num_envs = int(env.num_envs)
        self.agents = int(env.agents)
        canvas = env.canvas
        self.canvas = None if canvas is None else [int(canvas[0]), int(canvas[1])]
        self.cursor: _replay.ReplaySteps | None = None
        #: Event index of every call seen, by call index.
        self.events: list[int] = []
        #: Calls applied by the current cursor.
        self.position = 0
        self.keyframes: tuple[int, ...] = ()
        # One pass over the whole file: the timeline knows every call.
        self._restart()
        while self._advance():
            pass
        self.calls = len(self.events)

    def _restart(self) -> None:
        if self.cursor is not None:
            self.cursor.close()
        self.cursor = _replay.steps(self.path, self.env)
        self.keyframes = self.cursor.keyframes()
        self.position = 0

    def _advance(self) -> bool:
        if self.cursor is None:
            return False
        try:
            event = next(self.cursor)
        except StopIteration:
            self.cursor = None
            return False
        if self.position == len(self.events):
            self.events.append(event)
        self.position += 1
        return True

    def seek(self, i: int) -> None:
        """Leaves ``env`` in the state after call ``i`` (zero-based)."""
        if not 0 <= i < self.calls:
            raise _BadRequest
        target = i + 1
        if self.cursor is None or target < self.position:
            self._seek_back(target)
        while self.position < target:
            if not self._advance():
                raise _BadRequest

    def _seek_back(self, target: int) -> None:
        # The nearest keyframe at or before the target call's event (a
        # keyframe is its own event, never a call's), then forward from it.
        bound = self.events[target - 1]
        known = set(self.events)
        usable = [k for k in self.keyframes if k < bound and k not in known]
        if self.cursor is not None and usable:
            k = max(usable)
            try:
                self.cursor.seek(k)
            except Exception:
                pass
            else:
                self.position = sum(1 for e in self.events if e < k)
                return
        self._restart()

    def meta(self) -> dict:
        layout = self.env._native.layout()
        env_id = getattr(self.env, "env_id", None) or layout.get("env_id") or layout.get("id")
        return {
            "env": env_id,
            "canvas": self.canvas,
            "num_envs": self.num_envs,
            "agents": self.agents,
            "keyframes": list(self.keyframes),
            "calls": self.calls,
            "calls_seen": len(self.events),
        }

    def frame(self, i: int, instance: int) -> dict:
        self.seek(i)
        prims = None if self.canvas is None else [_prim(p) for p in self.env.scene(instance)]
        return {"i": i, "instance": instance, "prims": prims}

    def step(self, i: int, instance: int) -> dict:
        self.seek(i)
        out = self.env._wrap(self.env._native.outputs())
        k = instance
        leaves = {}
        for name, arr in out.observations.items():
            leaf = arr[k]
            shape = list(leaf.shape)
            if leaf.size > _LEAF_LIMIT:
                leaves[name] = {"shape": shape, "dtype": str(leaf.dtype), "truncated": True}
            else:
                values = leaf.reshape(-1).tolist()
                leaves[name] = {"shape": shape, "dtype": str(leaf.dtype), "truncated": False, "values": _finite(values)}
        return {
            "i": i,
            "instance": instance,
            "rewards": _finite(out.rewards[k].tolist()),
            "alive": out.alive[k].tolist(),
            "acting": out.acting[k].tolist(),
            "agent_terminated": out.agent_terminated[k].tolist(),
            "agent_truncated": out.agent_truncated[k].tolist(),
            "episode_terminated": bool(out.episode_terminated[k]),
            "episode_truncated": bool(out.episode_truncated[k]),
            "masks": None if out.masks is None else out.masks[k].tolist(),
            "observations": leaves,
        }

    def close(self) -> None:
        if self.cursor is not None:
            cursor, self.cursor = self.cursor, None
            cursor.close()


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args) -> None:  # noqa: A002
        # Nothing is logged: request lines carry the token.
        pass

    def _headers(self) -> None:
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")

    def _empty(self, code: int) -> None:
        self.send_response(code)
        self._headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_error(self, code, message=None, explain=None) -> None:
        # The base class writes an HTML body naming the request; send none.
        self.close_connection = True
        self._empty(code)

    def _body(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self._headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self) -> tuple[bool, dict]:
        port = self.server.server_address[1]
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return False, {}
        query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        tokens = query.get("token", [])
        if len(tokens) != 1:
            return False, query
        given = tokens[0].encode("utf-8", "surrogateescape")
        return hmac.compare_digest(given, self.server.token.encode()), query

    def _refuse_method(self) -> None:
        ok, _ = self._allowed()
        self._empty(405 if ok else 403)

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse_method

    def do_GET(self) -> None:
        ok, query = self._allowed()
        if not ok:
            self._empty(403)
            return
        route = urlsplit(self.path).path
        try:
            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route.startswith("/") and route[1:] in _STATIC:
                self._static(route[1:])
            elif route == "/meta":
                with self.server.session.lock:
                    data = self.server.session.meta()
                self._json(data)
            elif route in ("/frame", "/step"):
                i = self._int(query, "i", None)
                instance = self._int(query, "instance", 0)
                session = self.server.session
                if not 0 <= instance < session.num_envs:
                    raise _BadRequest
                with session.lock:
                    data = session.frame(i, instance) if route == "/frame" else session.step(i, instance)
                self._json(data)
            else:
                self._empty(404)
        except _BadRequest:
            self._empty(400)
        except Exception:
            # Not logged: the request line carries the token.
            self._empty(500)

    @staticmethod
    def _int(query: dict, name: str, default: int | None) -> int:
        values = query.get(name)
        if not values:
            if default is None:
                raise _BadRequest
            return default
        if len(values) != 1:
            raise _BadRequest
        text = values[0]
        if not text.isascii() or not text.isdigit() or len(text) > 18:
            raise _BadRequest
        return int(text)

    def _json(self, data: dict) -> None:
        body = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
        self._body(body, "application/json")

    def _static(self, name: str) -> None:
        data = importlib.resources.files("tessera_rl").joinpath("viewer_static", name).read_bytes()
        self._body(data, _STATIC[name])


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, port: int, token: str, session: _Session) -> None:
        self.token = token
        self.session = session
        super().__init__((_HOST, port), _Handler)

    def handle_error(self, request, client_address) -> None:
        # The default prints a traceback that may hold the request line.
        pass


class Viewer:
    """A running Viewer server (see :func:`serve`)."""

    def __init__(self, path, env, *, port: int = 0) -> None:
        port = int(port)
        if not 0 <= port <= 65535:
            raise ValueError("port must be in 0..65535")
        self._session = _Session(path, env)
        try:
            self._server = _Server(port, secrets.token_urlsafe(32), self._session)
        except BaseException:
            self._session.close()
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, name="tessera-viewer", daemon=True)
        self._thread.start()
        self._closed = False

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)``; the host is always ``127.0.0.1``."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        """The page's URL, token included: treat it as a secret."""
        return f"http://{_HOST}:{self.address[1]}/?token={self._server.token}"

    def close(self) -> None:
        """Stops the server and ends the replay; ``env`` is free again."""
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        with self._session.lock:
            self._session.close()

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        # Never the URL: it carries the token.
        return f"<tessera_rl.viewer.Viewer port={self.address[1]}{' closed' if self._closed else ''}>"


def serve(path, env, *, port: int = 0, open_browser: bool = False) -> Viewer:
    """Serves the replay file at ``path``, replayed into ``env``, on
    ``127.0.0.1`` (``port`` 0: a free port). ``env`` must be built as for
    :func:`tessera_rl.replay.run`. Returns the running :class:`Viewer`; its
    ``url`` carries the session token. With ``open_browser``, opens the URL
    in the default browser."""
    viewer = Viewer(path, env, port=port)
    if open_browser:
        webbrowser.open(viewer.url)
    return viewer


def show(path, env, *, height: int = 600):
    """In Jupyter: starts a Viewer (see :func:`serve`) and returns an
    ``IPython.display.IFrame`` of its page. The frame's ``viewer`` attribute
    is the :class:`Viewer`; close it when done."""
    try:
        from IPython.display import IFrame
    except ImportError as exc:
        raise ImportError(
            "tessera_rl.viewer.show needs IPython (run it in Jupyter); use tessera_rl.viewer.serve outside it"
        ) from exc
    viewer = serve(path, env)
    frame = IFrame(viewer.url, width="100%", height=int(height))
    frame.viewer = viewer
    return frame
