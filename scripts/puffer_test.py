# SPDX-License-Identifier: MIT OR Apache-2.0
"""Builds the wheel and runs compat/pufferlib/tests in the pinned environment,
from a temporary directory, under a hard timeout that kills the process group."""

import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(ROOT, "compat", "pufferlib")
TIMEOUT_S = 2400
BUILD_TIMEOUT_S = 1800
MATURIN = "1.16.0"


HANDLED_SIGNALS = tuple(getattr(signal, n) for n in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, n))


def run_group(cmd: list[str], cwd: str, timeout: float) -> int:
    """Runs ``cmd`` in its own process group and kills that group on every path,
    a timeout or a signal included, so nothing stays behind on a CI host."""
    posix = hasattr(os, "killpg")
    p = subprocess.Popen(cmd, cwd=cwd, start_new_session=posix)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        if posix:
            # The handled signals are blocked during cleanup, so a second one
            # cannot interrupt it; a pending one is delivered after.
            old = signal.pthread_sigmask(signal.SIG_BLOCK, HANDLED_SIGNALS)
            try:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                reap(p)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old)
        else:
            p.kill()
            p.wait()


def reap(p: subprocess.Popen) -> None:
    """Waits for ``p`` with ``os.waitpid``, not ``Popen.wait`` (#655).

    ``Popen.wait(timeout)`` takes its internal ``_waitpid_lock`` before the
    ``try`` that releases it. A signal handler raising ``SystemExit`` between
    the two leaves the lock held, and a later ``Popen.wait`` then blocks
    forever, leaving the killed child a zombie and this process alive.
    """
    if p.returncode is not None:
        return
    try:
        _, status = os.waitpid(p.pid, 0)
    except ChildProcessError:
        return  # already reaped
    p.returncode = os.waitstatus_to_exitcode(status)


def exit_on_signals() -> None:
    """A signal raises ``SystemExit``, so ``finally`` blocks run."""

    def handler(signum, _frame):
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def main() -> int:
    exit_on_signals()
    work = tempfile.mkdtemp(prefix="tessera-puffer-")
    try:
        build = ["uvx", "--from", f"maturin=={MATURIN}", "maturin", "build", "--release",
                 "-m", os.path.join(ROOT, "crates/tessera-rl-py/Cargo.toml"), "-o", work]  # fmt: skip
        status = run_group(build, work, BUILD_TIMEOUT_S)
        if status != 0:
            sys.stderr.write(f"the wheel build failed with status {status}\n")
            return status
        wheel = glob.glob(os.path.join(work, "*.whl"))[0]
        cmd = ["uv", "run", "--locked", "--python", "3.11", "--project", PROJECT, "--with", wheel,
               "python", "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               os.path.join(PROJECT, "tests"), *sys.argv[1:]]  # fmt: skip
        return run_group(cmd, work, TIMEOUT_S)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
