# SPDX-License-Identifier: MIT OR Apache-2.0
"""A local stand-in for a remote host (tests of remote.py only).

The host is a directory, ``$REMOTE_TEST_HOSTS/<host>``. A "container" is a
detached local process, in its own session, that runs the phase tool with
the container paths mapped back to the host directory and the repository.
Its state persists in files, so a fresh driver process sees it.

Like the real ``docker run --rm``, a container is gone once it exits: its
exit code and output are then only in ``root/_status/<name>.{json,log}``,
written by the container itself (or 137 by a kill).
"""

import json
import os
import shutil
import signal
import subprocess
import sys

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "alphazero_connect4")
sys.path.insert(0, EXAMPLE)

import remote  # noqa: E402


class FakeHost:
    def __init__(self, host: str):
        self.host = host
        self.kills: list[str] = []
        self.base = os.path.join(os.environ["REMOTE_TEST_HOSTS"], host)
        os.makedirs(os.path.join(self.base, "containers"), exist_ok=True)

    def path(self, rel: str) -> str:
        return os.path.join(self.base, rel)

    def push(self, local: str, rel: str) -> None:
        dst = self.path(rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(local):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(local, dst)
        else:
            shutil.copy2(local, dst)  # rsync -a keeps the mode

    def pull(self, rel: str, local: str) -> None:
        src = self.path(rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if os.path.isdir(src):
            shutil.rmtree(local, ignore_errors=True)
            shutil.copytree(src, local)
        else:
            shutil.copyfile(src, local)

    def mkdir(self, rel: str) -> None:
        os.makedirs(self.path(rel), exist_ok=True)

    def exists(self, rel: str) -> bool:
        return os.path.exists(self.path(rel))

    def isdir(self, rel: str) -> bool:
        return os.path.isdir(self.path(rel))

    def tree_sha256(self, rel: str) -> str:
        return remote.tree_sha256(self.path(rel))

    def remove(self, rel: str) -> None:
        p = self.path(rel)
        shutil.rmtree(p) if os.path.isdir(p) else (os.remove(p) if os.path.exists(p) else None)

    def clear_root(self) -> None:
        r = self.path("root")
        for n in os.listdir(r) if os.path.isdir(r) else []:
            if n not in ("ledger.json", "ledger.json.lock"):
                self.remove(os.path.join("root", n))

    def _rec(self, name: str) -> str:
        return os.path.join(self.base, "containers", name + ".json")

    def docker_run(self, name: str, cpuset: str, hostname: str, argv: list[str], deadline_s: float = 0) -> None:
        if os.path.exists(self._rec(name)):
            raise remote.TransportError(f"container {name} exists")
        host_argv = []
        for a in argv:
            if a.startswith(remote.CROOT + "/"):
                a = os.path.join(self.base, "root", a[len(remote.CROOT) + 1 :])
            elif a.startswith(remote.SRC + "/examples/alphazero_connect4/"):
                a = os.path.join(EXAMPLE, os.path.basename(a))
            host_argv.append(a)
        st = self._status(name)
        os.makedirs(os.path.dirname(st), exist_ok=True)
        # The container's own deadline, as `timeout` in the real one: it
        # holds without the driver. On exit it records its status and
        # removes itself (--rm).
        script = (
            "import json, os, signal, subprocess, sys\n"
            "st, rec = sys.argv[1], sys.argv[2]\n"
            "p = subprocess.Popen(sys.argv[4:])\n"  # in this group: a kill takes all of it
            "def done(rc):\n"
            "    open(st + '.json.tmp', 'w').write(json.dumps({'exit_code': rc}))\n"
            "    os.replace(st + '.json.tmp', st + '.json')\n"
            "    os.path.exists(rec) and os.remove(rec)\n"
            "try:\n    rc = p.wait(timeout=float(sys.argv[3]))\n"
            "except subprocess.TimeoutExpired:\n"
            "    done(124)\n"
            "    os.killpg(os.getpgrp(), signal.SIGKILL)\n"
            "done(rc)\n"
        )
        deadline = deadline_s or 3600.0
        env = {k: v for k, v in os.environ.items() if not k.startswith("REMOTE_TEST_CRASH")}
        proc = subprocess.Popen([sys.executable, "-c", script, st, self._rec(name), str(deadline), sys.executable,
                                 *host_argv], start_new_session=True, env=env, stdout=open(st + ".log", "w"),
                                stderr=subprocess.STDOUT)  # fmt: skip
        with open(self._rec(name), "w") as f:
            json.dump({"pid": proc.pid, "argv": host_argv}, f)

    def _status(self, name: str) -> str:
        return os.path.join(self.base, "root", remote.STATUS, name)

    def status(self, name: str):
        p = self._status(name) + ".json"
        return ("exited", int(json.load(open(p))["exit_code"])) if os.path.exists(p) else None

    def docker_state(self, name: str):
        if os.path.exists(self._rec(name)):
            pid = json.load(open(self._rec(name)))["pid"]
            try:
                os.kill(pid, 0)
                return ("running", None)
            except ProcessLookupError:
                pass
        # --rm: gone from docker; the exit is in the status record.
        return self.status(name)

    def docker_wait(self, name: str) -> None:
        import time

        while self.docker_state(name) and self.docker_state(name)[0] == "running":
            time.sleep(0.1)

    def docker_logs(self, name: str) -> str:
        p = self._status(name) + ".log"
        return open(p).read() if os.path.exists(p) else ""

    def docker_kill(self, name: str) -> None:
        """By name; records 137 unless the phase recorded its own exit."""
        rec = self._rec(name)
        self.kills.append(name)
        if os.path.exists(rec):
            pid = json.load(open(rec))["pid"]
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            import time

            for _ in range(100):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    break
            os.remove(rec)
        if not os.path.exists(self._status(name) + ".json"):
            with open(self._status(name) + ".json", "w") as f:
                f.write('{"exit_code": 137}')

    def docker_rm(self, name: str) -> None:
        rec = self._rec(name)
        if os.path.exists(rec):
            pid = json.load(open(rec))["pid"]
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if os.path.exists(rec):
            os.remove(rec)

    def kill(self, name: str) -> None:
        """Tests: the container vanishes (host reboot, OOM kill)."""
        self.docker_rm(name)

    def docker_ps_by_name(self, name: str) -> str:
        return name if os.path.exists(self._rec(name)) else ""

    def volumes(self) -> list[str]:
        return []


def factory(host: str) -> FakeHost:
    return FakeHost(host)
