# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Tessera Viewer: a local web page that
steps through a replay file, drawing each instance's scene and showing its
rewards, flags, masks and observations.

The backend is Python's standard library (``http.server``); the front end is
vendored HTML, CSS and JavaScript drawing on a Canvas 2D context, with no
build step and no external requests. This module is not imported by
``import tessera_rl`` and opens no socket until :func:`serve`, :func:`show` or
:func:`watch` is called.

``http.server`` is not a production server. The Viewer is a localhost,
single-user tool, guarded as follows:

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
- no request line, path or query is logged, so the token never reaches a
  log; an unexpected server error logs only its exception type and message
  (logger ``tessera_rl.viewer``, level ``ERROR``), with the token masked;
- the client never names a file: the server serves the one replay its caller
  chose and three fixed static files.

Requests are served on several threads, but every replay and pool operation
runs under one lock, so they serialize on the pool.

The environment ``env`` is driven by a stepped replay (:func:`tessera_rl.
replay.steps`) while the Viewer is open: calls that change it raise until
:meth:`Viewer.close`. The caller still owns ``env`` and closes it.

Endpoints (``GET`` only, all token-checked): ``/meta``; ``/frame?i=&instance=``
(the scene after call ``i``, zero-based); ``/step?i=&instance=`` (rewards,
alive, acting, masks and observation leaves after call ``i``).

Live mode (:func:`watch`): the same server and guards over a
training ``Env`` instead of a replay. ``/live?instance=`` asks the pool's watch
for that instance's record, waits at most 2 s for the pool to serve it at a
batch boundary, and answers ``{"generation", "instance", "prims"}``, drawn
from a private one-instance shadow pool of the same configuration
(``import_record`` then ``scene``), or ``{"pending": true}`` when no boundary
served it in time (training stopped, or the instance stayed in flight); the
request then stays pending. The training pool is never touched beyond the
watch slot. ``/frame`` and ``/step`` answer 404 in live mode.
"""

from __future__ import annotations

import bisect
import hmac
import importlib.resources
import json
import logging
import math
import os
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from tessera_rl import _viewer_logits as _logits
from tessera_rl import replay as _replay

__all__ = ["LogitsLog", "Viewer", "logits_log", "serve", "show", "watch"]

#: The side-file writer of policy logits (#683); see :func:`logits_log`.
LogitsLog = _logits.LogitsLog
logits_log = _logits.logits_log

_HOST = "127.0.0.1"
_LOG = logging.getLogger("tessera_rl.viewer")
#: How long ``/live`` waits for the pool to serve a request (seconds).
_LIVE_TIMEOUT = 2.0
#: How often ``/live`` polls the watch while it waits (seconds).
_LIVE_POLL = 0.01
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

    is_live = False

    def __init__(self, path, env, logits="auto") -> None:
        self.path = os.fspath(path)
        self.env = env
        self.heads = _logits.heads_of(env)
        self.width = _logits.width_of(env)
        #: The step ordinal of each ``step`` call's event (#683).
        self.ordinal_of: dict[int, int] = {}
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
        self._logits_init(logits)
        #: Per target position (calls applied, 1..calls): the keyframe a
        #: backward seek imports and the position it leaves the cursor at, or
        #: ``None`` to restart. Computed once, so a seek finds it in O(1).
        self._seek_plan = self._plan_seeks()

    def _plan_seeks(self) -> list[tuple[int, int] | None]:
        # A usable keyframe is its own event (never a call's) and lies before
        # the target call's event; the nearest one wins. Events and
        # keyframes are both increasing.
        known = set(self.events)
        usable = sorted(k for k in self.keyframes if k not in known)
        plan: list[tuple[int, int] | None] = [None]
        for target in range(1, self.calls + 1):
            bound = self.events[target - 1]
            j = bisect.bisect_left(usable, bound)
            if j == 0:
                plan.append(None)
            else:
                k = usable[j - 1]
                plan.append((k, bisect.bisect_left(self.events, k)))
        return plan

    def _restart(self) -> None:
        if self.cursor is not None:
            self.cursor.close()
        self.cursor = _replay.steps(self.path, self.env)
        self.keyframes = self.cursor.keyframes()
        if not self.ordinal_of:
            self.ordinal_of = {e: k for k, e in enumerate(self.cursor.step_events())}
        self.position = 0

    def _logits_init(self, logits) -> None:
        """The logits source: ``"auto"`` (the side file if present), a side
        file path, a callable ``fn(call_index, instance)``, or ``None``. A
        side file that cannot be served is ignored, and ``/meta`` says why."""
        self.logits_fn = None
        self.logits_by_ordinal: dict | None = None
        self.logits_note: str | None = None
        if logits is None:
            return
        if callable(logits):
            self.logits_fn = logits
            return
        side = _logits.side_path(self.path) if logits == "auto" else os.fspath(logits)
        if not os.path.exists(side):
            if logits != "auto":
                self.logits_note = "logits side file not found"
            return
        try:
            self.logits_by_ordinal = _logits.load(side, self.path, self.env, len(self.ordinal_of))
        except _logits.Refused as e:
            self.logits_note = f"logits side file ignored: {e}"

    def _logits_row(self, i: int, instance: int):
        """``(agents, width)`` logits for call ``i``, an ``{"error": ...}``
        dict, or ``None``."""
        if self.logits_fn is not None:
            return _call_logits(self.logits_fn, (i, instance))
        if self.logits_by_ordinal is None:
            return None
        ordinal = self.ordinal_of.get(self.events[i])
        if ordinal is None:
            return None
        arr = self.logits_by_ordinal.get(ordinal)
        return None if arr is None else arr[instance]

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
        step = self._seek_plan[target] if target < len(self._seek_plan) else None
        if self.cursor is not None and step is not None:
            k, position = step
            try:
                self.cursor.seek(k)
            except Exception:
                pass
            else:
                self.position = position
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
            "live": False,
            "logits": _logits_meta(self),
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
            **_logits_field(self, self._logits_row(i, instance), None if out.masks is None else out.masks[k]),
        }

    def close(self) -> None:
        if self.cursor is not None:
            cursor, self.cursor = self.cursor, None
            cursor.close()


class _LiveSession:
    """A live watch over a training ``env``; every method
    runs under ``lock``. Records are drawn by a private one-instance shadow
    pool, never by ``env``."""

    is_live = True

    def __init__(self, env, logits=None) -> None:
        if logits is not None and not callable(logits):
            raise ValueError("viewer.watch takes logits=fn(instance), a callable")
        self.logits_fn = logits
        self.logits_note: str | None = None
        self.heads = _logits.heads_of(env)
        self.width = _logits.width_of(env)
        env_id = getattr(env, "env_id", None)
        config = getattr(env, "_config", None)
        if env_id is None or config is None:
            raise ValueError("viewer.watch needs an Env made by tessera_rl.make")
        from tessera_rl import _env

        self.env = env
        self.lock = threading.Lock()
        self.num_envs = int(env.num_envs)
        self.agents = int(env.agents)
        canvas = env.canvas
        self.canvas = None if canvas is None else [int(canvas[0]), int(canvas[1])]
        self.env_id = env_id
        # The shadow first: a refused configuration attaches no watch.
        self.shadow = _env.make(env_id, num_envs=1, seed=0, outputs="owned", workers=0, **config)
        try:
            self.handle = env.watch()
        except BaseException:
            self.shadow.close()
            raise
        #: The instance of the request in the slot, if one is pending.
        self.pending: int | None = None

    def meta(self) -> dict:
        return {
            "env": self.env_id,
            "canvas": self.canvas,
            "num_envs": self.num_envs,
            "agents": self.agents,
            "live": True,
            "logits": _logits_meta(self),
        }

    def live(self, instance: int) -> dict:
        """The record of ``instance`` served within ``_LIVE_TIMEOUT``, drawn
        by the shadow pool, or ``{"pending": true}``."""
        from tessera_rl import ArgumentError, ConcurrencyError, PoisonedError

        deadline = time.monotonic() + _LIVE_TIMEOUT
        try:
            if self.pending is None:
                self.handle.request(instance)
                self.pending = instance
            while True:
                taken = self.handle.take()
                if taken is not None:
                    served, self.pending = self.pending, None
                    if served == instance:
                        return self._draw(instance, *taken)
                    # The answer to an older request for another instance.
                    self.handle.request(instance)
                    self.pending = instance
                    continue
                if time.monotonic() >= deadline:
                    return {"pending": True}
                time.sleep(_LIVE_POLL)
        except ConcurrencyError:
            return {"pending": True}
        except PoisonedError:
            return {"pending": True, "poisoned": True}
        except ArgumentError:
            # Detached: the training Env was closed.
            return {"pending": True, "detached": True}

    def _draw(self, instance: int, generation: int, record: bytes) -> dict:
        self.shadow.import_record(0, record)
        prims = None if self.canvas is None else [_prim(p) for p in self.shadow.scene(0)]
        row = None if self.logits_fn is None else _call_logits(self.logits_fn, (instance,))
        return {"generation": int(generation), "instance": instance, "prims": prims, **_logits_field(self, row, None)}

    def close(self) -> None:
        self.handle.close()
        self.shadow.close()


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
            elif route == "/live" and self.server.session.is_live:
                session = self.server.session
                instance = self._int(query, "instance", 0)
                if not 0 <= instance < session.num_envs:
                    raise _BadRequest
                with session.lock:
                    data = session.live(instance)
                self._json(data)
            elif route in ("/frame", "/step") and not self.server.session.is_live:
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
        except Exception as exc:
            # The exception's type and message only, never the path, query or
            # request line (they carry the token), and the token masked in
            # case a message quotes it.
            message = str(exc).replace(self.server.token, "<token>")
            _LOG.error("viewer request failed: %s: %s", type(exc).__name__, message)
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

    def __init__(self, port: int, token: str, session: _Session | _LiveSession) -> None:
        self.token = token
        self.session = session
        super().__init__((_HOST, port), _Handler)

    def handle_error(self, request, client_address) -> None:
        # The default prints a traceback that may hold the request line.
        pass


def _port(port) -> int:
    port = int(port)
    if not 0 <= port <= 65535:
        raise ValueError("port must be in 0..65535")
    return port


class Viewer:
    """A running Viewer server (see :func:`serve`)."""

    def __init__(self, path, env, *, port: int = 0, logits="auto") -> None:
        port = _port(port)
        self._start(_Session(path, env, logits), port)

    @classmethod
    def _live(cls, env, port: int, logits=None) -> Viewer:
        port = _port(port)
        viewer = cls.__new__(cls)
        viewer._start(_LiveSession(env, logits), port)
        return viewer

    def _start(self, session: _Session | _LiveSession, port: int) -> None:
        self._session = session
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
        """Stops the server and ends the replay; ``env`` is free again. In live
        mode, closes the watch handle and the shadow pool; the training
        ``env`` is not closed."""
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


def serve(path, env, *, port: int = 0, open_browser: bool = False, logits="auto") -> Viewer:
    """Serves the replay file at ``path``, replayed into ``env``, on
    ``127.0.0.1`` (``port`` 0: a free port). ``env`` must be built as for
    :func:`tessera_rl.replay.run`. Returns the running :class:`Viewer`; its
    ``url`` carries the session token. With ``open_browser``, opens the URL
    in the default browser.

    ``logits`` (#683): ``"auto"`` shows the replay's side file
    ``<path>.logits.npz`` (see :func:`logits_log`) if it exists and matches
    the replay; a path names another side file; a callable
    ``fn(call_index, instance)`` returns ``(agents, width)`` logits or
    ``None``; ``None`` shows none. A side file that does not match is
    ignored, and ``/meta`` says why."""
    viewer = Viewer(path, env, port=port, logits=logits)
    if open_browser:
        webbrowser.open(viewer.url)
    return viewer


def show(path, env, *, height: int = 600, logits="auto"):
    """In Jupyter: starts a Viewer (see :func:`serve`) and returns an
    ``IPython.display.IFrame`` of its page. The frame's ``viewer`` attribute
    is the :class:`Viewer`; close it when done."""
    try:
        from IPython.display import IFrame
    except ImportError as exc:
        raise ImportError(
            "tessera_rl.viewer.show needs IPython (run it in Jupyter); use tessera_rl.viewer.serve outside it"
        ) from exc
    viewer = serve(path, env, logits=logits)
    frame = IFrame(viewer.url, width="100%", height=int(height))
    frame.viewer = viewer
    return frame


def watch(env, *, port: int = 0, logits=None) -> Viewer:
    """Serves a live view of the training ``env`` on
    ``127.0.0.1`` (``port`` 0: a free port), with the guards of
    :func:`serve`. ``env`` must come from :func:`tessera_rl.make` (the
    Viewer builds a one-instance shadow pool of the same id and options to
    draw served records) and have no other watch open; keep stepping it as
    usual, from any thread. The page polls ``/live`` every 250 ms; each poll
    waits at most 2 s for the pool to serve the shown instance at a batch
    boundary, then answers ``{"pending": true}``. Returns the running
    :class:`Viewer`; :meth:`Viewer.close` closes the watch, not ``env``.

    ``logits`` (#683): a callable ``fn(instance)`` returning that
    instance's latest ``(agents, width)`` logits, or ``None``. It is called
    on the Viewer's request thread when the page polls, never from a
    worker; an exception shows in the page's logits panel."""
    return Viewer._live(env, port, logits)


def _call_logits(fn, args):
    """``fn(*args)`` as an ``(agents, width)`` array, ``None``, or an
    ``{"error": ...}`` dict if it raises or returns something else."""
    try:
        row = fn(*args)
    except Exception as e:  # shown in the panel; the handler keeps working
        _LOG.warning("viewer logits callable failed: %s", type(e).__name__)
        return {"error": f"{type(e).__name__}: {e}"}
    return row


def _logits_meta(session) -> dict:
    if getattr(session, "logits_fn", None) is not None:
        source = "callable"
    elif getattr(session, "logits_by_ordinal", None) is not None:
        source = "file"
    else:
        source = None
    return {"source": source, "note": getattr(session, "logits_note", None)}


def _logits_field(session, row, masks) -> dict:
    """The ``"logits"`` entry of a ``/step`` or ``/live`` answer: absent
    without a row; ``{"error": ...}`` for a failed or malformed one; else
    one list of heads per agent (``_viewer_logits.payload``)."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return {"logits": row}
    import numpy as np

    try:
        arr = np.asarray(row, dtype=np.float64)
    except (TypeError, ValueError):
        return {"logits": {"error": "logits are not numeric"}}
    agents = int(session.agents)
    if arr.shape != (agents, session.width):
        return {"logits": {"error": f"logits have shape {arr.shape}, expected {(agents, session.width)}"}}
    mask_rows = None
    if masks is not None:
        m = np.asarray(masks)
        if m.shape == (agents, session.width):
            mask_rows = m
    return {
        "logits": [
            _logits.payload(arr[a], None if mask_rows is None else mask_rows[a], session.heads) for a in range(agents)
        ]
    }
