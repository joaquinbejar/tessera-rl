# SPDX-License-Identifier: MIT OR Apache-2.0
"""A host object made before os.fork() refuses in the child (SPEC 14.4.2,
#329): the host module registers its own copy of the fork guard. Over the
compiled test host wheel (`make py-test-host`). Every case runs in a
subprocess in its own process group, killed as a group on timeout, and the
forked child has an alarm, because the failure this guards against is a
hang."""

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
pytest.importorskip("_tessera_host", reason="needs the test host wheel (make py-test-host)")

TIMEOUT = int(os.environ.get("FORK_TEST_TIMEOUT", "60"))
CHILD_ALARM = 20

PRELUDE = textwrap.dedent(
    f"""
    import os, signal, struct, threading, time
    import numpy as np
    import tessera_rl as t
    import _tessera_host as package
    host = getattr(package, "_tessera_host", package)

    ACTS = struct.pack("<4i", 0, 0, 0, 0)

    def in_child(call):
        # 0: refused with ForkError; 9: returned; 8: another error; the
        # alarm (SIGALRM) is a hang.
        pid = os.fork()
        if pid == 0:
            signal.alarm({CHILD_ALARM})
            try:
                call()
            except t.ForkError as e:
                os._exit(0 if "os.fork()" in str(e) else 7)
            except BaseException:
                os._exit(8)
            os._exit(9)
        _, status = os.waitpid(pid, 0)
        if os.WIFSIGNALED(status):
            return "signal %d" % os.WTERMSIG(status)
        return {{0: "refused", 7: "wrong message", 8: "other error", 9: "returned"}}[os.WEXITSTATUS(status)]

    # The parent is multithreaded when it forks.
    stop = threading.Event()
    threading.Thread(target=lambda: [time.sleep(0.001) for _ in iter(lambda: stop.is_set(), True)], daemon=True).start()
    """
)


def run(body, tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(body)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"timed out, killed the process group:\n{out}")
    assert proc.returncode == 0, out
    return out


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("call", ["reset", "step", "layout"])
def test_host_fork_object_refuses_in_the_child(tmp_path, workers, call):
    out = run(
        f"""
        obj = host.make(num_envs=2, workers={workers})
        obj.reset(None)
        calls = {{"reset": lambda: obj.reset(None), "step": lambda: obj.step(ACTS), "layout": lambda: obj.layout()}}
        print("CHILD", in_child(calls[{call!r}]))
        # The parent's object still works.
        obj.step(ACTS)
        obj.close()
        print("PARENT OK")
        """,
        tmp_path,
    )
    assert "CHILD refused" in out and "PARENT OK" in out, out


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("call", ["reset", "step"])
def test_host_fork_host_env_from_make_refuses_in_the_child(tmp_path, workers, call):
    out = run(
        f"""
        env = t.make("test/FactoryCounter-v1", num_envs=2, workers={workers})
        env.reset()
        calls = {{"reset": lambda: env.reset(), "step": lambda: env.step(np.zeros((2, 2), dtype=np.int32))}}
        print("CHILD", in_child(calls[{call!r}]))
        env.step(np.zeros((2, 2), dtype=np.int32))
        env.close()
        print("PARENT OK")
        """,
        tmp_path,
    )
    assert "CHILD refused" in out and "PARENT OK" in out, out


@pytest.mark.parametrize("workers", [0, 2])
def test_host_fork_a_new_object_in_the_child_works(tmp_path, workers):
    out = run(
        f"""
        old = host.make(num_envs=2, workers={workers})
        old.reset(None)

        def fresh():
            obj = host.make(num_envs=2, workers={workers})
            obj.reset(None)
            obj.step(ACTS)
            obj.close()

        print("CHILD", in_child(fresh))
        old.close()
        """,
        tmp_path,
    )
    assert "CHILD returned" in out, out


def test_host_fork_close_in_the_child_does_not_hang(tmp_path):
    out = run(
        """
        obj = host.make(num_envs=2, workers=2)
        obj.reset(None)

        def close_then_use():
            obj.close()
            obj.reset(None)

        print("CHILD", in_child(close_then_use))
        obj.close()
        """,
        tmp_path,
    )
    assert "CHILD refused" in out, out
