# SPDX-License-Identifier: MIT OR Apache-2.0
"""A forked child and an inherited environment (issue #93). Every case runs in
a subprocess in its own process group, from a temporary directory, killed as a
group on timeout, because the failure this guards against is a hang."""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")

PRELUDE = textwrap.dedent(
    """
    import hashlib, os, sys, threading, time
    import numpy as np
    import tessera_rl as t

    N = 4096

    def build(mode, workers):
        kw = {"num_envs": N, "seed": 7, "chunk_size": 64, "workers": workers}
        if mode == "pipelined":
            kw["mode"] = "pipelined"
        elif mode == "async":
            kw.update(mode="async", batch_size=N)
        return t.make("tessera/CartPole-v1", **kw)

    def first(env, mode):
        out = env.reset()
        if mode == "pipelined":
            return out[0].step.observations["obs"]
        if mode == "async":
            return out.step.observations["obs"]
        return out.observations["obs"]

    def launch(env, mode):
        if mode == "pipelined":
            env.send(0, np.zeros(len(env.halves[0]), dtype=np.int32))
        elif mode == "async":
            env.send(list(range(N)), np.zeros(N, dtype=np.int32))
        else:
            env.step(np.zeros(N, dtype=np.int32))

    def finish(env, mode):
        if mode != "sync":
            env.recv()

    def play(mode, workers):
        env = build(mode, workers)
        digest = hashlib.sha256()
        first(env, mode)
        for _ in range(5):
            # Long enough for workers and the caller to reach their blocking
            # wait, which on macOS is what traps after a fork.
            time.sleep(0.02)
            if mode == "pipelined":
                for half, ids in enumerate(env.halves):
                    env.send(half, np.zeros(len(ids), dtype=np.int32))
                for _ in env.halves:
                    digest.update(env.recv().step.observations["obs"].tobytes())
            elif mode == "async":
                env.send(list(range(N)), np.zeros(N, dtype=np.int32))
                got = env.recv()
                order = np.argsort(got.ids)
                digest.update(got.step.observations["obs"][order].tobytes())
            else:
                digest.update(env.step(np.zeros(N, dtype=np.int32)).observations["obs"].tobytes())
        env.close()
        return digest.hexdigest()

    # The parent is multithreaded when it forks.
    stop = threading.Event()
    threading.Thread(target=lambda: [time.sleep(0.001) for _ in iter(lambda: stop.is_set(), True)], daemon=True).start()
    """
)


def run(body, tmp_path, timeout=int(os.environ.get("FORK_TEST_TIMEOUT", "60")), prelude=PRELUDE):
    proc = subprocess.Popen(
        [sys.executable, "-c", prelude + textwrap.dedent(body)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"timed out, killed the process group:\n{out}")
    return proc.returncode, out


CHILD = """
    mode, workers, action = {mode!r}, {workers}, {action!r}
    env = build(mode, workers)
    view = first(env, mode)
    before = view.tobytes()
    launch(env, mode)
    pid = os.fork()
    if pid == 0:
        if action == "use":
            calls = [lambda: env.reset(), lambda: env.is_poisoned]
            if mode == "sync":
                calls.append(lambda: env.step(np.zeros(N, dtype=np.int32)))
            else:
                calls.append(lambda: env.recv())
                calls.append(lambda: env.snapshot_all())
            for call in calls:
                try:
                    call()
                except t.ForkError as e:
                    assert "fork" in str(e), e
                else:
                    print("NOT REFUSED", flush=True)
                    os._exit(9)
        elif action == "close":
            env.close()
            env.close()
            try:
                env.reset()
            except t.ForkError:
                pass
            else:
                os._exit(9)
        elif action == "drop":
            del env
        elif action == "view":
            assert t.is_stale(view)
            assert view.shape[0] > 0 and len(view.tobytes()) == len(before)
            del view
        elif action == "new":
            print("DIGEST", play(mode, workers), flush=True)
        # Everything leaves through a real interpreter exit.
        sys.exit(0)
    _, status = os.waitpid(pid, 0)
    finish(env, mode)
    launch(env, mode)
    finish(env, mode)
    env.close()
    print("CHILD", status, flush=True)
    sys.exit(0)
"""

MODES = ["sync", "pipelined", "async"]
WORKERS = [0, 2]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("workers", WORKERS)
@pytest.mark.parametrize("action", ["use", "close", "drop", "exit", "view"])
def test_child_never_hangs_or_crashes_and_parent_is_unaffected(mode, workers, action, tmp_path):
    code, out = run(CHILD.format(mode=mode, workers=workers, action=action), tmp_path)
    assert code == 0, out
    assert "CHILD 0\n" in out, out
    assert "NOT REFUSED" not in out, out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("workers", WORKERS)
def test_new_pool_in_the_child_matches_a_fresh_process(mode, workers, tmp_path):
    code, out = run(CHILD.format(mode=mode, workers=workers, action="new"), tmp_path)
    assert code == 0, out
    assert "CHILD 0\n" in out, out
    digest = [line for line in out.splitlines() if line.startswith("DIGEST")]
    assert len(digest) == 1, out
    code, fresh = run(f"print('DIGEST', play({mode!r}, {workers}), flush=True)\nsys.exit(0)\n", tmp_path)
    assert code == 0, fresh
    assert digest[0] in fresh, (digest, fresh)


def test_guard_survives_a_second_fork_and_a_child_pool_in_a_grandchild(tmp_path):
    code, out = run(
        """
        env = build("sync", 2)
        first(env, "sync")
        pid = os.fork()
        if pid == 0:
            mine = build("sync", 2)
            first(mine, "sync")
            launch(mine, "sync")
            inner = os.fork()
            if inner == 0:
                try:
                    mine.reset()
                except t.ForkError:
                    sys.exit(0)
                sys.exit(5)
            _, status = os.waitpid(inner, 0)
            launch(mine, "sync")
            mine.close()
            sys.exit(0 if status == 0 else 6)
        _, status = os.waitpid(pid, 0)
        env.close()
        print("CHILD", status, flush=True)
        sys.exit(0)
        """,
        tmp_path,
    )
    assert code == 0, out
    assert "CHILD 0\n" in out, out


EARLY_HOOK = """
    import os, sys
    result = []
    env = None

    def early():
        try:
            {call}
            result.append("not refused")
        except BaseException as e:
            result.append(type(e).__name__)

    # Registered before tessera_rl is imported, so it runs first in the child.
    os.register_at_fork(after_in_child=early)
    import numpy as np
    import tessera_rl as t

    env = t.make("tessera/CartPole-v1", num_envs=4096, seed=7, chunk_size=64, workers=2, mode="pipelined")
    env.reset()
    env.send(0, np.zeros(len(env.halves[0]), dtype=np.int32))
    pid = os.fork()
    if pid == 0:
        print("EARLY", result, flush=True)
        sys.exit(0)
    print("CHILD", os.waitpid(pid, 0)[1], flush=True)
    env.recv()
    env.close()
    sys.exit(0)
"""


@pytest.mark.parametrize("call", ["env._native.layout()", "env.reset()", "env.recv()"])
def test_hook_registered_before_import_gets_fork_error(call, tmp_path):
    code, out = run(EARLY_HOOK.format(call=call), tmp_path, prelude="")
    assert code == 0, out
    assert "EARLY ['ForkError']" in out, out
    assert "CHILD 0\n" in out, out


def test_direct_libc_fork_is_covered(tmp_path):
    """A C extension calling fork directly runs no Python at-fork hook. PyDLL
    keeps the interpreter lock across the call, so the child can run Python."""
    code, out = run(
        """
        import ctypes
        env = build("pipelined", 2)
        first(env, "pipelined")
        launch(env, "pipelined")
        stop.set()
        time.sleep(0.05)
        pid = ctypes.PyDLL(None).fork()
        if pid == 0:
            try:
                env.reset()
                print("NOT REFUSED", flush=True)
            except t.ForkError:
                print("REFUSED", flush=True)
            del env
            os._exit(0)
        print("CHILD", os.waitpid(pid, 0)[1], flush=True)
        finish(env, "pipelined")
        env.close()
        sys.exit(0)
        """,
        tmp_path,
    )
    assert code == 0, out
    assert "REFUSED" in out and "NOT REFUSED" not in out, out
    assert "CHILD 0\n" in out, out
