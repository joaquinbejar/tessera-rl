# SPDX-License-Identifier: MIT OR Apache-2.0
"""The external clock reference of a gate session (ADR-0063 item 2).

Runs on the reference host named in the manifest, outside the GPU box:

    python clock_ref.py --out PATH --host USER@BOX --stop-file PATH --port N [--every 5] [--cpu-every 60] [--idle 60]

``--host`` is the ssh target (the box). The header records this host's own
name (``socket.gethostname()``) as ``host``, which must be the manifest's
``clock_ref_host``, and the target as ``target``.

**Transport: TCP over one ssh tunnel; the ssh session's stdin is unused.**
The box's OpenSSH DefaultShell is PowerShell, which runs a remote command as
``powershell.exe -c "<command>"``; that host takes the session's stdin, so a
loop that reads stdin gets nothing (``152-gate2-p5-incident.md``,
``152-clockref-fix-dryrun.md``). Here the remote command is ``TCP_LOOP``:
a PowerShell listener on ``127.0.0.1:<port>`` of the box that accepts one
client, answers one line per request line (the requests and replies of
``clock.py``'s ``LOOP``), and exits on ``end``, on the client's disconnect,
after ``--idle`` seconds without a request, or if no client connects within
60 s. The client runs ``ssh -T -L 127.0.0.1:<port>:127.0.0.1:<port>`` with
stdin from /dev/null and talks to its local end of the tunnel. Nothing on
the box listens beyond localhost, and nothing outlives the client.

Every read is one request and its reply: the request is stamped just before
it is sent and the reply just after it arrives, both with this host's
``time.monotonic()``, so the box's QPC was read between the two stamps. A
read whose round trip exceeds 0.25 s is recorded as unusable. If the session
ends or a reply takes longer than ``--reply-timeout``, a ``lost`` row is
written and the client stops; it never opens a second session. A sync
request first discards any line before the first QPC reply. On the stop file
it sends ``end`` and writes an ``end`` row last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

RTT_MAX_S = 0.25
SYNC_TIMEOUT_S = 30.0
CONNECT_RETRY_S = 0.5

# The box side: one PowerShell line, no double quote and no newline, so it
# stays one argument inside sshd's ``-c "<command>"``. PORT and IDLE_MS are
# filled in by tcp_loop(). The replies are those of clock.LOOP.
TCP_LOOP = (
    "$l = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, PORT); $l.Start(); "
    "$a = $l.AcceptTcpClientAsync(); if (-not $a.Wait(60000)) { $l.Stop(); exit 3 }; "
    "$c = $a.Result; $l.Stop(); $c.NoDelay = $true; $c.ReceiveTimeout = IDLE_MS; "
    "$s = $c.GetStream(); $r = [IO.StreamReader]::new($s); $w = [IO.StreamWriter]::new($s); $w.AutoFlush = $true; "
    "try { while (($x = $r.ReadLine()) -ne $null) { "
    "if ($x -eq 'end') { $w.WriteLine('end'); break } "
    "if ($x -eq 'cpu') { $w.WriteLine([Diagnostics.Process]::GetCurrentProcess()"
    ".TotalProcessorTime.TotalSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)) } "
    "else { $w.WriteLine(([Diagnostics.Stopwatch]::GetTimestamp()).ToString() + ' ' + "
    "([Diagnostics.Stopwatch]::Frequency).ToString() + ' ' + ([DateTime]::UtcNow.Ticks).ToString()) } } } "
    "catch { } finally { $c.Close() }"
)


def self_sha256() -> str:
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tcp_loop(port: int, idle_s: float) -> str:
    if not 1024 <= port <= 65535:
        raise ValueError("port must be in 1024..65535")
    return TCP_LOOP.replace("PORT", str(int(port))).replace("IDLE_MS", str(int(idle_s * 1000)))


def ssh_command(host: str, port: int, idle_s: float, ssh: str = "ssh") -> list[str]:
    """The tunnel: local 127.0.0.1:port to the box's 127.0.0.1:port, and the
    listener as the remote command. The session's stdin is not used."""
    return [ssh, "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{port}:127.0.0.1:{port}", host, tcp_loop(port, idle_s)]  # fmt: skip


class Session:
    """The ssh process (tunnel and remote listener) and one TCP connection
    through it."""

    def __init__(self, argv: list[str], port: int, timeout: float):
        self.p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=None)
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.buf = b""

    def readline(self, timeout: float) -> str | None:
        end = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = end - time.monotonic()
            if left <= 0:
                return None
            self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(4096)
            except (TimeoutError, OSError):
                return None
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def connect_and_sync(self, write) -> bool:
        """Connects through the tunnel and aligns once: one request, and every
        line before the first QPC reply is recorded as discarded. While the
        remote listener is not up yet, ssh accepts the local connection and
        then closes it; the client retries until SYNC_TIMEOUT_S."""
        end = time.monotonic() + SYNC_TIMEOUT_S
        while time.monotonic() < end:
            if self.p.poll() is not None:
                return False
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=2)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.sendall(b"x\n")
                while True:
                    line = self.readline(max(0.0, min(5.0, end - time.monotonic())))
                    if line is None:
                        break
                    parts = line.split()
                    if len(parts) == 3 and all(x.lstrip("-").isdigit() for x in parts):
                        return True
                    write({"kind": "discarded", "line": line[:200]})
            except OSError:
                pass
            self.close_socket()
            time.sleep(CONNECT_RETRY_S)
        return False

    def ask(self, line: str) -> str | None:
        try:
            self.sock.sendall((line + "\n").encode())
        except OSError:
            return None
        return self.readline(self.timeout)

    def close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.buf = b""

    def close(self):
        self.close_socket()
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.terminate()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--stop-file", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--every", type=float, default=5.0)
    p.add_argument("--cpu-every", type=float, default=60.0)
    p.add_argument("--idle", type=float, default=60.0, help="the box loop exits after this long without a request")
    p.add_argument("--reply-timeout", type=float, default=10.0)
    p.add_argument("--ssh", default="ssh", help=argparse.SUPPRESS)  # tests: a fake ssh
    args = p.parse_args(argv)
    cmd = ssh_command(args.host, args.port, args.idle, args.ssh)
    with open(args.out, "a") as f:

        def write(rec):
            f.write(json.dumps(rec) + "\n")
            f.flush()

        write({"kind": "header", "clock_ref_py_sha256": self_sha256(), "host": socket.gethostname(),
               "target": args.host, "transport": "tcp over ssh -L", "port": args.port, "idle_s": args.idle,
               "command": sys.argv if argv is None else ["clock_ref.py", *argv],
               "platform": platform.platform(), "python": sys.version,
               "clock": "time.monotonic", "clock_info": str(time.get_clock_info("monotonic"))})  # fmt: skip
        s = Session(cmd, args.port, args.reply_timeout)
        if not s.connect_and_sync(write):
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
