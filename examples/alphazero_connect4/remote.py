# SPDX-License-Identifier: MIT OR Apache-2.0
"""The phase transport of the P2-M3 loop (step 2): one loop phase run in a
pinned container on a remote host, with a durable, verified round trip.

``remote.py run`` is the loop's per-phase ``runners`` command::

    {python} remote.py run --host swarmer02 --cpuset 0-31 --root ROOT -- {tool} {args}

``remote.py reconcile --root ROOT`` settles any open dispatch. The driver
runs it before every ``loop.py run``.

The round trip, for an attempt directory ``D``:

1. **Dispatch.** ``D.dispatch.json`` is written durably before any remote
   action. It records:
   - the attempt id, host, cpuset, container name and image;
   - the sha256 of the local ledger;
   - the remote paths, and the inputs with their hashes.

   An open (unacknowledged) dispatch anywhere in the root refuses a new
   one, so only one remote phase ever runs.
2. **Push.** The inputs and the ledger go to a mirror of the root on the
   host, then their remote hashes are checked against the dispatch.
3. **Run.** ``docker run -d --name NAME``, detached and named, with bind
   mounts by absolute path only. A driver or ssh death does not stop the
   phase. A resumed driver finds the container by name: if it is running,
   it waits; if it has exited, it collects.
4. **Import.** It pulls into ``D.staging`` and the staged ledger, and
   accepts the remote ledger only if it extends the one sent:
   - the entries sent are an exact prefix;
   - the tail is this attempt's entry, appended or settled.

   It then verifies the staged output's identity and publishes it: the
   output is renamed into place and the ledger replaced atomically. The
   dispatch is then acknowledged (``imported``) durably.
5. **Cleanup.** Only after the acknowledgement: the remote payload is
   removed, ``docker rm NAME`` runs, and a by-name ``docker ps -a`` and the
   volume set are checked.

**Recovery.**
- A successful remote attempt is imported once, never charged as
  interrupted and redone.
- If the container is gone but its entry is still ``running``, the entry
  is settled at its full reservation, but only after a by-name check on
  that host. The check's output goes into the dispatch.
- Before any reservation, a lost attempt charges nothing.
- A broken transfer is fetched again.
- A publication without its acknowledgement is acknowledged again.

The loop's owner lock: ``run`` is a child of ``loop.py run``, which holds
it, and ``reconcile`` takes it itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402

IMAGE = "rust:1.99.1@sha256:09027f873d45b8ecd1dc86f1fa13b7a62b875c961f017a0e331267b350430137"
BASE = "tessera-rl-m3loop"  # under the remote home: src/, uv, root/
CROOT = "/m3root"  # the root's mount point in the container
SRC = "/work/tessera"
PHASES = {"selfplay.py": "selfplay", "train_step.py": "train", "look.py": "look"}
# A container older than its wall cap plus these is killed by the driver:
# the supervisor's cleanup grace, uv sync and the build, and a margin.
SSH_TIMEOUT_S = 600
# Build products and caches never count in the source tree hash.
TREE_EXCLUDE = (".venv", ".uvcache", "target", "__pycache__", ".pytest_cache", "build")
# Built native extensions (maturin's editable install writes
# python/tessera_rl/_native.abi3.so): build products, not source.
FILE_EXCLUDE_SUFFIXES = (".so", ".pyd", ".dylib", ".pyc")
SETUP_S = float(os.environ.get("REMOTE_TEST_SETUP_S", "900"))
MARGIN_S = float(os.environ.get("REMOTE_TEST_MARGIN_S", "300"))


class TransportError(RuntimeError):
    """A remote step failed; the dispatch stays open for reconcile."""


def crash_point(name: str) -> None:
    """Tests: the whole driver (this process group) dies here."""
    if os.environ.get("REMOTE_TEST_CRASH") == name:
        import signal

        os.killpg(os.getpgrp(), signal.SIGKILL)


# ------------------------------------------------------------ transports


class SSH:
    """The real host: ssh, rsync and docker on it."""

    def __init__(self, host: str):
        self.host = host

    def _ssh(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        try:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host, cmd],
                               capture_output=True, text=True, timeout=SSH_TIMEOUT_S)  # fmt: skip
        except subprocess.TimeoutExpired:
            raise TransportError(f"{self.host}: {cmd[:80]!r} timed out after {SSH_TIMEOUT_S} s") from None
        # 255 is ssh's own failure: never an answer about the host.
        if r.returncode == 255:
            raise TransportError(f"{self.host}: ssh failed (255): {r.stderr.strip()[:300]}")
        if check and r.returncode != 0:
            raise TransportError(f"{self.host}: {cmd!r} exited {r.returncode}: {r.stderr.strip()[:300]}")
        return r

    def path(self, rel: str) -> str:
        return f"{BASE}/{rel}"

    def push(self, local: str, rel: str) -> None:
        self._ssh(f"mkdir -p {shlex.quote(os.path.dirname(self.path(rel)) or BASE)}")
        src = local + "/" if os.path.isdir(local) else local
        r = subprocess.run(["rsync", "-a", "--delete", src, f"{self.host}:{self.path(rel)}"], capture_output=True, text=True)
        if r.returncode != 0:
            raise TransportError(f"rsync to {self.host}:{rel} exited {r.returncode}")

    def pull(self, rel: str, local: str) -> None:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        r = subprocess.run(["rsync", "-a", "--delete", f"{self.host}:{self.path(rel)}/", local + "/"]
                           if self.isdir(rel) else ["rsync", "-a", f"{self.host}:{self.path(rel)}", local],
                           capture_output=True, text=True)  # fmt: skip
        if r.returncode != 0:
            raise TransportError(f"rsync from {self.host}:{rel} exited {r.returncode}")

    def mkdir(self, rel: str) -> None:
        self._ssh(f"mkdir -p {shlex.quote(self.path(rel))}")

    def _test(self, flag: str, rel: str) -> bool:
        r = self._ssh(f"test {flag} {shlex.quote(self.path(rel))}", check=False)
        if r.returncode not in (0, 1):
            raise TransportError(f"{self.host}: test {flag} {rel} exited {r.returncode}")
        return r.returncode == 0

    def exists(self, rel: str) -> bool:
        return self._test("-e", rel)

    def isdir(self, rel: str) -> bool:
        return self._test("-d", rel)

    def tree_sha256(self, rel: str) -> str:
        # The same digest as tree_sha256 below, computed on the host.
        script = TREE_SHA_PY.replace("'", "'\"'\"'")
        r = self._ssh(f"python3 -c '{script}' {shlex.quote(self.path(rel))}")
        return r.stdout.strip()

    def remove(self, rel: str) -> None:
        self._ssh(f"rm -rf {shlex.quote(self.path(rel))}")

    def clear_root(self) -> None:
        r = shlex.quote(self.path("root"))
        self._ssh(f"test ! -d {r} || find {r} -mindepth 1 -maxdepth 1 ! -name ledger.json ! -name ledger.json.lock "
                  "-exec rm -rf {} +")  # fmt: skip

    def docker_run(self, name: str, cpuset: str, hostname: str, argv: list[str], deadline_s: float) -> None:
        home = self._ssh("pwd").stdout.strip()
        base = f"{home}/{BASE}"
        # The deadline lives in the container, around uv setup and the
        # phase alike: it holds even if the driver or ssh is gone.
        phase = "uv run --locked --python 3.11 --group test --group torch python " + " ".join(map(shlex.quote, argv))
        inner = (
            f"timeout --kill-after 30 {int(deadline_s)} sh -c {shlex.quote(phase)}"
            + "; rc=$?; chown -R $HOST_UID:$HOST_GID " + CROOT + " " + SRC + "; exit $rc"
        )  # fmt: skip
        cmd = [
            "docker", "run", "-d", "--name", name, "--hostname", hostname, "--cpuset-cpus", cpuset,
            "-e", f"PILOT_IMAGE_DIGEST={IMAGE}", "-e", f"PILOT_CPUSET={cpuset}",
            "-e", "HOST_UID=$(id -u)", "-e", "HOST_GID=$(id -g)",
            "-v", f"{base}/src:{SRC}", "-v", f"{base}/root:{CROOT}", "-v", f"{base}/uv:/usr/local/bin/uv:ro",
            "-w", SRC, "-e", f"UV_CACHE_DIR={SRC}/.uvcache", "-e", "UV_LINK_MODE=copy", IMAGE, "sh", "-c", inner,
        ]  # fmt: skip
        self._ssh(" ".join(c if c.startswith("HOST_") else shlex.quote(c) for c in cmd))

    def docker_state(self, name: str) -> tuple[str, int | None] | None:
        r = self._ssh(f"docker inspect --format '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {shlex.quote(name)}",
                      check=False)  # fmt: skip
        if r.returncode != 0:
            # Absent only when docker says so; any other failure is not
            # evidence that the container is gone.
            if "No such object" in r.stderr or "No such container" in r.stderr:
                return None
            raise TransportError(f"{self.host}: docker inspect {name} exited {r.returncode}: {r.stderr.strip()[:300]}")
        status, code = r.stdout.split()
        return ("running", None) if status in ("created", "running", "restarting") else ("exited", int(code))

    def docker_wait(self, name: str) -> None:
        self._ssh(f"docker wait {shlex.quote(name)}", check=False)

    def docker_logs(self, name: str) -> str:
        return self._ssh(f"docker logs {shlex.quote(name)} 2>&1", check=False).stdout

    def docker_kill(self, name: str) -> None:
        self._ssh(f"docker kill {shlex.quote(name)}", check=False)

    def docker_rm(self, name: str) -> None:
        self._ssh(f"docker rm -f {shlex.quote(name)}", check=False)

    def docker_ps_by_name(self, name: str) -> str:
        return self._ssh(f"docker ps -a --filter name=^{name}$ --format '{{{{.Names}}}} {{{{.Status}}}}'").stdout.strip()

    def volumes(self) -> list[str]:
        return sorted(self._ssh("docker volume ls -q").stdout.split())


TREE_SHA_PY = """
import hashlib, os, sys
p = sys.argv[1]
h = hashlib.sha256()
if os.path.isdir(p):
    for base, dirs, files in os.walk(p):
        dirs[:] = sorted(x for x in dirs if x not in (".venv", ".uvcache", "target", "__pycache__", ".pytest_cache", "build"))
        for n in sorted(x for x in files if not x.endswith((".so", ".pyd", ".dylib", ".pyc"))):
            f = os.path.join(base, n)
            h.update(os.path.relpath(f, p).encode() + b"\\0")
            h.update(open(f, "rb").read())
else:
    h.update(open(p, "rb").read())
print(h.hexdigest())
"""


def tree_sha256(path: str, exclude_files: tuple[str, ...] = ()) -> str:
    h = hashlib.sha256()
    if os.path.isdir(path):
        for base, dirs, files in os.walk(path):
            dirs[:] = sorted(x for x in dirs if x not in TREE_EXCLUDE)
            for n in sorted(x for x in files if not (base == path and x in exclude_files)
                            and not x.endswith(FILE_EXCLUDE_SUFFIXES)):  # fmt: skip
                f = os.path.join(base, n)
                h.update(os.path.relpath(f, path).encode() + b"\0")
                with open(f, "rb") as fh:
                    h.update(fh.read())
    else:
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def transport(host: str):
    """The real transport, or a test double named by REMOTE_TEST_TRANSPORT
    (``module:factory``); a test double never reaches a real host."""
    spec = os.environ.get("REMOTE_TEST_TRANSPORT")
    if spec:
        mod, _, fn = spec.partition(":")
        sys.path.insert(0, os.environ.get("REMOTE_TEST_PATH", ""))
        return getattr(__import__(mod), fn)(host)
    return SSH(host)


# --------------------------------------------------------------- dispatch


def attempt_of(argv: list[str]) -> tuple[str, str, str]:
    """(phase, attempt id, attempt directory) from a phase command."""
    phase = PHASES[os.path.basename(argv[0])]
    attempt = argv[argv.index("--attempt") + 1]
    out = argv[argv.index("--out") + 1]
    return phase, attempt, os.path.dirname(out) if phase == "train" else out


def mapped(root: str, argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """The container argv and the inputs to push ``{local: remote rel}``.

    The tool maps to the source mount. Every existing absolute path under
    the root maps into the root mount. Any other existing absolute path,
    such as the bot binary, goes under ``root/_inputs`` by content.
    Outputs that do not exist yet map without being pushed."""
    root = os.path.abspath(root)
    out, inputs = [f"{SRC}/examples/alphazero_connect4/{os.path.basename(argv[0])}"], {}
    for a in argv[1:]:
        if os.path.isabs(a) and (a == root or a.startswith(root + os.sep)):
            rel = os.path.relpath(a, root)
            out.append(f"{CROOT}/{rel}")
            if os.path.exists(a):
                inputs[a] = f"root/{rel}"
                # A checkpoint travels with its manifest sidecar (ckpt.py).
                side = os.path.splitext(a)[0] + ".json"
                if os.path.isfile(a) and side != a and os.path.exists(side):
                    inputs[side] = f"root/{os.path.relpath(side, root)}"
        elif os.path.isabs(a) and os.path.exists(a):
            rel = f"_inputs/{tree_sha256(a)[:16]}/{os.path.basename(a)}"
            out.append(f"{CROOT}/{rel}")
            inputs[a] = f"root/{rel}"
        else:
            out.append(a)
    return out, inputs


def open_dispatches(root: str) -> list[str]:
    found = []
    for base, _, files in os.walk(root):
        for n in files:
            if n.endswith(".dispatch.json"):
                p = os.path.join(base, n)
                with open(p, encoding="utf-8") as f:
                    if json.load(f).get("closed") is None:
                        found.append(p)
    return sorted(found)


def write_dispatch(path: str, record: dict) -> None:
    artifacts.write_atomic(path, artifacts.canonical_json(record))


def read_ledger(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]


def entries_sha256(entries: list[dict]) -> str:
    return artifacts.sha256(artifacts.canonical_json(entries))


def dispatch(args, argv: list[str]) -> str:
    """Steps 1 to 3; returns the dispatch path. The caller holds the owner
    lock (loop.py run)."""
    root = os.path.abspath(args.root)
    opened = open_dispatches(root)
    if opened:
        raise TransportError(f"an open dispatch must be reconciled first: {opened[0]}")
    phase, attempt, d = attempt_of(argv)
    cargv, inputs = mapped(root, argv)
    config = os.path.join(root, "config.json")
    inputs[config] = "root/config.json"
    ledger = os.path.join(root, "ledger.json")
    sent = read_ledger(ledger)
    cores = int(argv[argv.index("--cores") + 1])
    if cores != cpuset_size(args.cpuset):
        raise TransportError(f"--cores {cores} does not match the cpuset {args.cpuset}")
    wall_cap = float(argv[argv.index("--wall-cap-s") + 1])
    record = {
        "kind": "p2-m3-dispatch",
        "attempt": attempt,
        "phase": phase,
        "host": args.host,
        "cpuset": args.cpuset,
        "cores": cores,
        # Unique per experiment (the root's config) and attempt.
        "container": f"tessera-m3-{artifacts.sha256_file(config)[:8]}-{attempt}",
        "image": IMAGE,
        "attempt_dir": os.path.relpath(d, root),
        "argv": argv,
        "container_argv": cargv,
        "config_sha256": artifacts.sha256_file(config),
        "source_tree_sha256": args.source_tree_sha256,
        "ledger_sha256": artifacts.sha256_file(ledger) if os.path.exists(ledger) else None,
        "ledger_entries_sent": len(sent),
        "sent_entries_sha256": entries_sha256(sent),
        "inputs": {inputs[p]: tree_sha256(p) for p in sorted(inputs)},
        "dispatched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dispatched_epoch": time.time(),
        "wall_cap_s": wall_cap,
        # The container's own deadline, around setup and the phase.
        "container_deadline_s": wall_cap + SETUP_S,
        "started": False,
        "closed": None,
    }
    path = d + ".dispatch.json"
    write_dispatch(path, record)
    crash_point("after-dispatch")
    t = transport(args.host)
    # The deployed source must be the reviewed tree (provision with
    # `remote.py provision`).
    got = t.tree_sha256("src")
    if got != args.source_tree_sha256:
        raise TransportError(f"the source on {args.host} is {got}, not the reviewed {args.source_tree_sha256}")
    for local, rel in inputs.items():
        t.push(local, rel)
    if phase == "train":
        t.mkdir(f"root/{record['attempt_dir']}")
    crash_point("mid-push")
    for rel, sha in record["inputs"].items():
        if t.tree_sha256(rel) != sha:
            raise TransportError(f"{rel} does not match its dispatch hash on {args.host}")
    t.docker_run(record["container"], args.cpuset, args.host, cargv, record["container_deadline_s"])
    crash_point("after-docker-run")  # before the start is recorded
    record["started"] = True
    write_dispatch(path, record)
    crash_point("after-start")
    return path


def cpuset_size(spec: str) -> int:
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += int(b or a) - int(a) + 1
    return n


# ----------------------------------------------------------------- import


def accept_ledger(sent: list[dict], remote: list[dict], attempt: str) -> list[dict]:
    """The remote ledger only if it extends what was sent with exactly this
    attempt's entry; raises ``TransportError`` otherwise."""
    if remote[: len(sent)] != sent:
        raise TransportError("the remote ledger does not keep the sent entries as an exact prefix")
    tail = remote[len(sent) :]
    if len(tail) > 1 or any(e["run"].split()[0] != attempt for e in tail):
        raise TransportError(f"the remote ledger appends {[e['run'] for e in tail]}, not this attempt only")
    return remote


def _finite(x) -> bool:
    import math

    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def validate_entry(e: dict, record: dict) -> None:
    """The attempt's settled entry against the frozen dispatch: the cores,
    the cap, the status and the charge arithmetic."""
    bad = lambda why: TransportError(f"ledger entry {e.get('run')!r}: {why}")  # noqa: E731
    if e.get("cores") != record["cores"]:
        raise bad(f"cores {e.get('cores')}, not the dispatched {record['cores']}")
    if not (_finite(e.get("deadline_s")) and 0 < e["deadline_s"] <= record["wall_cap_s"] + 1e-3):
        raise bad(f"deadline {e.get('deadline_s')} outside (0, {record['wall_cap_s']}]")
    if not _finite(e.get("core_hours")):
        raise bad(f"charge {e.get('core_hours')} is not finite and non-negative")
    status = e.get("status", "")
    # Settled at the full reservation by recovery or by hand (ledger.py);
    # every supervisor settlement charges the measured wall (supervisor.py).
    if status.startswith("interrupted (settled"):
        expected = round(e["deadline_s"] * e["cores"] / 3600, 6)
    elif status == "ok" or status.startswith(("ok (overran", "failed", "refused", "interrupted by signal",
                                              "killed at the deadline")):  # fmt: skip
        if not _finite(e.get("wall_s")):
            raise bad(f"wall {e.get('wall_s')} is not finite and non-negative")
        expected = round(e["wall_s"] * e["cores"] / 3600, 6)
    else:
        raise bad(f"status {status!r} is not a settled status")
    # The ledger rounds wall_s to 1 ms and charges from the unrounded wall.
    if abs(e["core_hours"] - expected) > 1e-6 + e["cores"] * 0.001 / 3600:
        raise bad(f"charge {e['core_hours']} is not {expected}")


def fsync_tree(path: str) -> None:
    for base, dirs, files in os.walk(path):
        for n in files:
            fd = os.open(os.path.join(base, n), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        fd = os.open(base, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def wait_or_kill(t, record: dict, path: str):
    """The container's state once it is not running. The container's own
    deadline stops it first; this is the driver's second safeguard: one
    older than its wall cap plus the setup allowance and a margin is
    killed (not removed), and the kill is recorded."""
    name = record["container"]
    deadline = record["dispatched_epoch"] + record["wall_cap_s"] + SETUP_S + MARGIN_S
    poll = float(os.environ.get("REMOTE_POLL_S", "15"))
    while True:
        state = t.docker_state(name)
        if state is None or state[0] != "running":
            return state
        if time.time() > deadline:
            t.docker_kill(name)
            record["killed_overdue"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "deadline_epoch": deadline}  # fmt: skip
            record["published_partial"] = True
            write_dispatch(path, record)
            for _ in range(100):
                state = t.docker_state(name)
                if state is None or state[0] != "running":
                    return state
                time.sleep(poll)
            raise TransportError(f"{name} did not stop after docker kill")
        time.sleep(poll)


def reconcile_one(root: str, path: str) -> dict:
    """Takes one open dispatch to a closed state; the caller holds the
    owner lock. Idempotent: every step replays to the same result, and any
    doubt leaves the dispatch open with the remote evidence in place."""
    import fcntl

    import loop as loopmod

    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    t = transport(record["host"])
    name, attempt = record["container"], record["attempt"]
    d = os.path.join(root, record["attempt_dir"])
    ledger = os.path.join(root, "ledger.json")
    if record.get("imported"):
        return cleanup(t, record, path)
    # The prefix comes from the immutable dispatch evidence, not from a
    # possibly altered local ledger.
    sent = read_ledger(ledger)[: record["ledger_entries_sent"]]
    if entries_sha256(sent) != record["sent_entries_sha256"]:
        raise TransportError("the local ledger no longer holds the entries sent with the dispatch")
    # Found by name, whether or not its start was recorded; never started
    # twice.
    state = wait_or_kill(t, record, path)
    record["exit_code"] = state[1] if state else None
    if state is not None:
        record["logs_tail"] = t.docker_logs(name)[-4000:]
    # Pull first: the remote ledger and the attempt output, staged.
    staging = d + ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    rledger = ledger + ".staging"
    rdoc = None
    if t.exists("root/ledger.json"):
        t.pull("root/ledger.json", rledger)
        with open(rledger, encoding="utf-8") as f:
            rdoc = json.load(f)
        remote = rdoc["entries"]
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            budget = json.load(f)["budget_core_hours"]
        if rdoc["budget_core_hours"] != budget:
            raise TransportError("the remote ledger's budget differs from the loop's")
    elif not record["started"] and state is None and not t.docker_ps_by_name(name):
        remote = sent  # never started: no container was ever created
    else:
        raise TransportError(f"no ledger on {record['host']}: no evidence of what the attempt did")
    rel = f"root/{record['attempt_dir']}"
    remote_out_sha = None
    if t.exists(rel):
        t.pull(rel, staging)
        remote_out_sha = t.tree_sha256(rel)
    crash_point("after-pull")
    remote = accept_ledger(sent, remote, attempt)
    mine = [i for i, e in enumerate(remote) if e["run"].split()[0] == attempt]
    if not mine:
        # Before any reservation: positive evidence that nothing ran. A
        # started dispatch needs an exited container and the remote ledger.
        if state is not None and state[0] == "running":
            raise TransportError(f"{name} is running with no ledger entry")
        if record["started"] and (state is None or rdoc is None or state[1] in (None, 0)):
            raise TransportError(f"{name} was started but is gone with no ledger entry: reconcile by hand")
        record["not_started_evidence"] = {
            "docker_ps_by_name": t.docker_ps_by_name(name) if state is None else f"exited {state[1]}",
            "remote_ledger_entries": len(remote),
            "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    elif remote[mine[0]]["status"] == "running":
        # The container is terminal (absent, or exited, killed included)
        # while its entry is open: settled conservatively at the full
        # reservation. A running container never is.
        if state is not None and state[0] == "running":
            raise TransportError(f"{name} is still running on {record['host']} with its entry open")
        listed = t.docker_ps_by_name(name)
        record["terminal_evidence"] = {
            "docker_state": list(state) if state else None,
            "docker_ps_by_name": listed,
            "checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if state is None and listed:
            raise TransportError(f"{name} is listed on {record['host']} but not inspectable: {listed}")
        from ledger import Ledger

        led = Ledger(rledger, rdoc["budget_core_hours"])
        # Ledger's own same-host recovery may already have settled it.
        if led.entries()[mine[0]]["status"] == "running":
            led.settle_interrupted(mine[0])
        remote = read_ledger(rledger)
        record["published_partial"] = True
    if mine:
        validate_entry(remote[mine[0]], record)
    spent = sum(e["core_hours"] for e in remote)
    if rdoc is not None and not _finite(spent):
        raise TransportError("the remote ledger's spend is not finite")
    # The output: an ok attempt must have a complete, valid output, equal
    # to the remote one; anything else keeps the remote evidence.
    out_sha = tree_sha256(staging) if os.path.isdir(staging) and os.listdir(staging) else None
    ok = bool(mine) and remote[mine[0]]["status"] == "ok"
    if ok:
        if out_sha is None:
            raise TransportError(f"{attempt} settled ok but its output is missing")
        if out_sha != remote_out_sha:
            raise TransportError(f"the staged output of {attempt} differs from the remote one")
        lp = loopmod.Loop(root)
        try:
            ident = lp.identity(record["phase"], staging)
        except loopmod.LOAD_ERRORS as e:
            raise TransportError(f"{attempt} settled ok but its output is invalid: {e}") from None
        record["output_identity"] = ident
    led_sha = artifacts.sha256_file(rledger) if os.path.exists(rledger) else None
    # Journal, then publish the output and the ledger, under the ledger's
    # own lock; replay checks each step against the journal.
    journal = record.get("publish")
    if journal is None:
        record["publish"] = journal = {"output_sha256": out_sha, "ledger_sha256": led_sha}
        write_dispatch(path, record)
    elif journal != {"output_sha256": out_sha, "ledger_sha256": led_sha}:
        raise TransportError("the remote evidence changed since publication began")
    with open(ledger + ".lock", "a+") as llock:
        fcntl.flock(llock, fcntl.LOCK_EX)
        if out_sha is not None:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
            if os.path.exists(d):
                # A replay: the loop may have written its descriptor since.
                if tree_sha256(d, exclude_files=("attempt.json",)) != out_sha:
                    raise TransportError(f"{d} already holds a different output: not overwritten")
                shutil.rmtree(staging)
            else:
                os.replace(staging, d)
            fsync_tree(d)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        if led_sha is not None:
            # Compare and replace: the local ledger is still the one sent,
            # or already this result (a replay).
            now = artifacts.sha256_file(ledger) if os.path.exists(ledger) else None
            if now not in (record["ledger_sha256"], led_sha):
                raise TransportError("the local ledger changed since the dispatch")
            with open(rledger, "rb") as f:
                artifacts.write_atomic(ledger, f.read())
    # Lineage before the acknowledgement: the loop's own chain check over
    # the published evidence (checkpoint, window, config). A broken link
    # keeps the remote evidence.
    try:
        lp = loopmod.Loop(root)
        lp.chain(lp.ledger.entries())
    except loopmod.Stop as e:
        raise TransportError(f"the imported {attempt} breaks the loop's lineage: {e.reason} {e.inputs}") from None
    crash_point("after-publish")
    record["imported"] = True
    record["imported_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["ledger_tail"] = remote[len(sent) :]
    write_dispatch(path, record)
    if os.path.exists(rledger):
        os.remove(rledger)
    return cleanup(t, record, path)


def cleanup(t, record: dict, path: str) -> dict:
    """After the acknowledgement only: the container, then everything in
    the host's root mirror but the ledger, then a by-name check."""
    name = record["container"]
    t.docker_rm(name)
    t.clear_root()
    left = t.docker_ps_by_name(name)
    if left:
        raise TransportError(f"{name} is still listed on {record['host']}: {left}")
    record["closed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_dispatch(path, record)
    return record


def provision(args) -> dict:
    """Deploys the reviewed source archive and the uv binary to a host,
    once, and records the evidence: the archive and uv digests and the
    deployed tree hash that every dispatch then checks."""
    import tarfile
    import tempfile

    t = transport(args.host)
    if t.exists("src"):
        raise TransportError(f"{args.host} already has a source tree: remove it first")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        with tarfile.open(args.archive) as tar:
            tar.extractall(src, filter="data")
        tree = tree_sha256(src)
        t.push(src, "src")
    t.push(args.uv, "uv")
    got = t.tree_sha256("src")
    if got != tree:
        raise TransportError(f"the deployed source on {args.host} is {got}, not {tree}")
    evidence = {
        "host": args.host,
        "archive_sha256": artifacts.sha256_file(args.archive),
        "uv_sha256": artifacts.sha256_file(args.uv),
        "source_tree_sha256": tree,
        "provisioned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(evidence, sort_keys=True))
    return evidence


# ------------------------------------------------------------------- main


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", required=True)
    r.add_argument("--cpuset", required=True)
    r.add_argument("--root", required=True)
    r.add_argument("--source-tree-sha256", required=True, help="the reviewed tree, from provision")
    c = sub.add_parser("reconcile")
    c.add_argument("--root", required=True)
    c.add_argument("--owner-held", action="store_true", help="called by loop.py, which holds the owner lock")
    v = sub.add_parser("provision")
    v.add_argument("--host", required=True)
    v.add_argument("--archive", required=True)
    v.add_argument("--uv", required=True)
    if "--" in argv:
        i = argv.index("--")
        args, phase_argv = p.parse_args(argv[:i]), argv[i + 1 :]
    else:
        args, phase_argv = p.parse_args(argv), []
    if args.command == "provision":
        provision(args)
        return 0
    root = os.path.abspath(args.root)
    if args.command == "run":
        if not phase_argv:
            p.error("run needs -- TOOL ARGS")
        path = dispatch(args, phase_argv)
        rec = reconcile_one(root, path)
        print(json.dumps({"attempt": rec["attempt"], "exit_code": rec.get("exit_code"), "closed": rec["closed"]}))
        return 0
    lock = None
    if not args.owner_held:
        # Standalone: the owner lock first, before any dispatch is read or
        # any host is contacted.
        import loop as loopmod

        lock = loopmod.Loop(root).owner()
    try:
        for path in open_dispatches(root):
            rec = reconcile_one(root, path)
            print(json.dumps({"attempt": rec["attempt"], "closed": rec["closed"]}))
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
