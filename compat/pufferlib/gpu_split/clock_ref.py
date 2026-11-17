# SPDX-License-Identifier: MIT OR Apache-2.0
"""The external clock reference of a gate session (ADR-0063 item 2).

Runs on the reference host named in the manifest, outside the GPU box:

    python clock_ref.py --out PATH --host USER@BOX --stop-file PATH [--every 5] [--cpu-every 60]

``--host`` is the ssh target (the box). The header records this host's own
name (``socket.gethostname()``) as ``host``, which must be the manifest's
``clock_ref_host``, and the target as ``target``.

It opens one ssh session to the box whose remote command is the PowerShell
loop itself (the same loop as ``clock.py``'s host reader), so the box's
DefaultShell runs it as the session's only stdin reader (``ssh_command``),
and keeps it for the whole session; it never opens a second one. A sync
request first discards anything the remote shell prints before the loop. Every read is one request and its reply: the request is
stamped just before it is sent and the reply just after it arrives, both with
this host's ``time.monotonic()``, so the box's QPC was read between the two
stamps. A read whose round trip exceeds 0.25 s is recorded as unusable. If
the session ends or a reply takes longer than ``--reply-timeout``, a ``lost``
row is written and the client stops. On the stop file it sends ``end`` and
writes an ``end`` row last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import select
import shlex
import socket
import subprocess
import sys
import time

# Kept identical to clock.LOOP (a test checks it); this file runs on its own.
LOOP = (
    "while (($l = [Console]::In.ReadLine()) -ne $null) { "
    "if ($l -eq 'end') { [Console]::Out.WriteLine('end'); [Console]::Out.Flush(); break } "
    "if ($l -eq 'cpu') { [Console]::Out.WriteLine([Diagnostics.Process]::GetCurrentProcess()"
    ".TotalProcessorTime.TotalSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)) } "
    "else { [Console]::Out.WriteLine(([Diagnostics.Stopwatch]::GetTimestamp()).ToString() + ' ' + "
    "([Diagnostics.Stopwatch]::Frequency).ToString() + ' ' + ([DateTime]::UtcNow.Ticks).ToString()) } "
    "[Console]::Out.Flush() }"
)
RTT_MAX_S = 0.25


def self_sha256() -> str:
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def ssh_command(host: str, ssh: str = "ssh") -> list[str]:
    """The ssh command whose remote command is LOOP itself.

    The box's OpenSSH DefaultShell is PowerShell, so sshd runs the remote
    command as ``powershell.exe -c "<command>"`` (152-gate2-p5-incident.md).
    A remote command that starts another ``powershell`` made two processes
    share the session's stdin: the DefaultShell wrapper and the loop as its
    native child. Only the first requests reached the loop, and the wrapper
    kept it alive after the session ended. Sending LOOP as the command makes
    the DefaultShell process run the loop in-process: it is the session's
    only stdin reader, and it exits when the session's stdin closes or on
    ``end``. LOOP holds no double quote, so sshd's ``-c "<command>"`` keeps
    it as one argument."""
    return [ssh, "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", host, LOOP]


SYNC_TIMEOUT_S = 30.0


class Session:
    def __init__(self, argv: list[str], timeout: float):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.timeout = timeout

    def ask(self, line: str) -> str | None:
        try:
            self.p.stdin.write(line + "\n")
            self.p.stdin.flush()
            ready, _, _ = select.select([self.p.stdout], [], [], self.timeout)
            if not ready:
                return None
            out = self.p.stdout.readline()
            return out.strip() if out else None
        except (OSError, ValueError):
            return None

    def sync(self, write) -> bool:
        """Aligns requests and replies once, at the start. The DefaultShell
        loads the user's profile (sshd gives it no -NoProfile), which may
        print lines before the loop reads its first request. One request is
        sent and lines are read until a reply that parses as QPC, frequency
        and UTC; anything before it is recorded as discarded."""
        try:
            self.p.stdin.write("x\n")
            self.p.stdin.flush()
        except (OSError, ValueError):
            return False
        end = time.monotonic() + SYNC_TIMEOUT_S
        while time.monotonic() < end:
            ready, _, _ = select.select([self.p.stdout], [], [], max(0.0, end - time.monotonic()))
            if not ready:
                return False
            line = self.p.stdout.readline()
            if not line:
                return False
            parts = line.split()
            if len(parts) == 3 and all(x.lstrip("-").isdigit() for x in parts):
                return True
            write({"kind": "discarded", "line": line.strip()[:200]})
        return False

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
            self.p.wait()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--stop-file", required=True)
    p.add_argument("--every", type=float, default=5.0)
    p.add_argument("--cpu-every", type=float, default=60.0)
    p.add_argument("--reply-timeout", type=float, default=10.0)
    p.add_argument("--command", help=argparse.SUPPRESS)  # tests: a local fake loop instead of ssh
    p.add_argument("--ssh", default="ssh", help=argparse.SUPPRESS)  # tests: a fake ssh with a DefaultShell
    args = p.parse_args(argv)
    cmd = shlex.split(args.command) if args.command else ssh_command(args.host, args.ssh)
    with open(args.out, "a") as f:

        def write(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()

        write({"kind": "header", "clock_ref_py_sha256": self_sha256(), "host": socket.gethostname(),
               "target": args.host,
               "command": sys.argv if argv is None else ["clock_ref.py", *argv],
               "platform": platform.platform(), "python": sys.version,
               "clock": "time.monotonic", "clock_info": str(time.get_clock_info("monotonic"))})  # fmt: skip
        s = Session(cmd, args.reply_timeout)
        if not s.sync(write):
            write({"kind": "event", "event": "lost", "at": time.monotonic(), "why": "no reply at sync"})
            s.close()
            return 3
        last_cpu = None
        nxt = time.monotonic()
        while not os.path.exists(args.stop_file):
            now = time.monotonic()
            if last_cpu is None or now - last_cpu >= args.cpu_every:
                t = time.monotonic()
                out = s.ask("cpu")
                if out is None:
                    write({"kind": "event", "event": "lost", "at": t})
                    s.close()
                    return 3
                try:
                    write({"kind": "cpu", "t": t, "cpu_s": float(out)})
                except ValueError:
                    write({"kind": "cpu", "t": t, "cpu_s": None})
                last_cpu = t
            t0 = time.monotonic()
            out = s.ask("x")
            t1 = time.monotonic()
            if out is None:
                write({"kind": "event", "event": "lost", "at": t1})
                s.close()
                return 3
            row = {"kind": "read", "t0": t0, "t1": t1, "usable": t1 - t0 <= RTT_MAX_S}
            try:
                q, fr, u = out.split()
                row.update(qpc=int(q), qpc_freq=int(fr), utc_ticks=int(u))
            except ValueError:
                row.update(qpc=None, usable=False, reply=out[:80])
            write(row)
            nxt += args.every
            time.sleep(max(0.0, nxt - time.monotonic()))
        ok = s.ask("end") == "end"
        s.close()
        write({"kind": "event", "event": "end", "loop_ended": ok})
    return 0


if __name__ == "__main__":
    sys.exit(main())
