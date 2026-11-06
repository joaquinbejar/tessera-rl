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


def main() -> int:
    work = tempfile.mkdtemp(prefix="tessera-puffer-")
    try:
        subprocess.run(
            ["uvx", "--from", "maturin>=1.16,<2", "maturin", "build", "--release",
             "-m", os.path.join(ROOT, "crates/tessera-rl-py/Cargo.toml"), "-o", work],
            check=True,
        )  # fmt: skip
        wheel = glob.glob(os.path.join(work, "*.whl"))[0]
        cmd = ["uv", "run", "--locked", "--python", "3.11", "--project", PROJECT, "--with", wheel,
               "--with", "pytest", "python", "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               os.path.join(PROJECT, "tests"), *sys.argv[1:]]  # fmt: skip
        p = subprocess.Popen(cmd, cwd=work, start_new_session=True)
        try:
            return p.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return 124
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
