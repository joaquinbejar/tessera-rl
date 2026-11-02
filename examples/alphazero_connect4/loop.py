# SPDX-License-Identifier: MIT OR Apache-2.0
"""The resumable AlphaZero loop of P2-M3 (PR 3, design revision 2).

Generation ``k`` runs three phases:
1. self-play with ``C_k`` (run seed ``seeds.generation(base, k)``);
2. training ``C_{k+1}`` for ``schedule.STEPS`` steps over the window of
   generations ``max(0, k-3)..k``, warm-started from ``C_k`` with a fresh
   optimizer;
3. a held-out look at ``C_{k+1}`` (``look.py``).

``C_0`` is given at ``init`` and pinned by SHA-256.

**Exactly once.** Every phase attempt ``(k, phase, n)`` has a fresh
directory ``gen-<k>/<phase>-<n>/`` and its ledger run name. An attempt is
never reused or overwritten.

A phase is **done** if and only if an attempt has a complete,
digest-verified output **and** a ledger entry settled ``ok``; the lowest
such ``n`` counts. The ledger is authoritative for spend. ``loop.json`` is
only a cache, rebuilt from the attempts and the ledger on every start.

The selection log (``selection.jsonl``) is append-only and deduplicated by
checkpoint SHA-256, so a restart never appends a look twice; incomplete
looks never enter it. One owner at a time holds ``loop.lock``.

**Admission.** A phase starts only if the ledger's remaining budget (the
settled spend plus every open reservation) covers the frozen reservations
of the generation's remaining phases. Otherwise the loop stops with reason
``budget``. It stops with ``K_max`` after the last generation, and with
``early`` when two consecutive generations reach the early-stop wins.
Every stop decision is appended to the selection log with its inputs.

**Selection** (``select``) picks, among eligible complete looks with an ok
ledger entry, the most held-out wins, ties going to the later generation.
It writes ``selection-manifest.json``. With no eligible look there is no
selection, and therefore no gate. The selected checkpoint and the final
(last trained) checkpoint are named separately.

Repeated looks make the selected held-out score optimistically biased; it
is never an estimate of gate performance.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import ckpt  # noqa: E402
import look as lookmod  # noqa: E402
import schedule  # noqa: E402
import seeds  # noqa: E402
from ledger import Ledger  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PHASES = ("selfplay", "train", "look")
WINDOW = 4
MAX_ATTEMPTS = 3
EXIT_NO_SELECTION = 2


class Stop(Exception):
    """The loop stops: ``reason`` and its inputs."""

    def __init__(self, reason: str, inputs: dict):
        super().__init__(reason)
        self.reason = reason
        self.inputs = inputs


def crash_point(name: str) -> None:
    """Test hook: exits abruptly at the named boundary (``LOOP_TEST_CRASH``)."""
    if os.environ.get("LOOP_TEST_CRASH") == name:
        os._exit(137)


def reservations(cfg: dict) -> dict[str, int]:
    """Core-seconds reserved per phase (frozen in the config)."""
    return {
        "selfplay": cfg["selfplay_cores"] * cfg["selfplay_wall_s"],
        "train": cfg["train_wall_s"],
        "look": cfg["look_wall_s"],
    }


class Loop:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.ledger = Ledger(os.path.join(root, "ledger.json"), self.cfg["budget_core_hours"])

    # ------------------------------------------------------------ state

    def attempt_dir(self, k: int, phase: str, n: int) -> str:
        return os.path.join(self.root, f"gen-{k:03d}", f"{phase}-{n}")

    def attempt_id(self, k: int, phase: str, n: int) -> str:
        return f"g{k:03d}-{phase}-{n}"

    def ledger_ok(self, attempt: str, entries: list[dict]) -> bool:
        return any(e["run"].startswith(attempt + " ") and e["status"] == "ok" for e in entries)

    def output_complete(self, phase: str, d: str) -> bool:
        try:
            if phase == "selfplay":
                artifacts.load_manifest(d)
            elif phase == "train":
                ckpt.load(os.path.join(d, "C.pt"))
            else:
                lookmod.load(d)
            return True
        except (artifacts.Incomplete, OSError, KeyError, ValueError, ckpt.NonFinite):
            return False

    def phase_state(self, k: int, phase: str, entries: list[dict]) -> tuple[str | None, int]:
        """The done attempt's directory (or None) and the next attempt
        number."""
        n = 0
        done = None
        while os.path.exists(self.attempt_dir(k, phase, n)):
            d = self.attempt_dir(k, phase, n)
            if done is None and self.output_complete(phase, d) and self.ledger_ok(self.attempt_id(k, phase, n), entries):
                done = d
            n += 1
        return done, n

    def checkpoint(self, k: int, entries: list[dict]) -> str | None:
        """The path of C_k, or None if it does not exist yet."""
        if k == 0:
            return os.path.join(self.root, "c0", "C.pt")
        done, _ = self.phase_state(k - 1, "train", entries)
        return os.path.join(done, "C.pt") if done else None

    def selection(self) -> list[dict]:
        path = os.path.join(self.root, "selection.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append_selection(self, entry: dict) -> None:
        path = os.path.join(self.root, "selection.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def remaining_core_s(self, entries: list[dict]) -> float:
        spent = sum(e["core_hours"] for e in entries)
        return (self.cfg["budget_core_hours"] - spent) * 3600

    # ------------------------------------------------------------ phases

    def phase_argv(self, k: int, phase: str, n: int, entries: list[dict]) -> list[str]:
        cfg, d, attempt = self.cfg, self.attempt_dir(k, phase, n), self.attempt_id(k, phase, n)
        common = [
            "--source-sha", cfg["source_sha"], "--attempt", attempt,
            "--ledger", os.path.join(self.root, "ledger.json"),
            "--budget-core-hours", str(cfg["budget_core_hours"]),
        ]  # fmt: skip
        if phase == "selfplay":
            return [
                "--checkpoint", self.checkpoint(k, entries), "--games", str(cfg["games"]),
                "--workers", str(cfg["selfplay_cores"]), "--run-seed", str(seeds.generation(cfg["base"], k)),
                "--out", d, "--sims", str(cfg["sims"]), "--batch-size", str(cfg["batch_size"]),
                "--capacity", str(cfg["capacity"]), "--cores", str(cfg["selfplay_cores"]),
                "--wall-cap-s", str(cfg["selfplay_wall_s"]), *common,
            ]  # fmt: skip
        if phase == "train":
            window = []
            for j in range(max(0, k - WINDOW + 1), k + 1):
                done, _ = self.phase_state(j, "selfplay", entries)
                if done is None:
                    raise Stop("incomplete input", {"generation": k, "missing_selfplay": j})
                window.append(done)
            os.makedirs(d, exist_ok=True)
            return [
                "train-window", "--checkpoint", self.checkpoint(k, entries), "--data", *window,
                "--out", os.path.join(d, "C.pt"), "--generation", str(k), "--base", str(cfg["base"]),
                "--steps", str(cfg["steps"]), "--batch", str(cfg["train_batch"]), "--cores", "1",
                "--wall-cap-s", str(cfg["train_wall_s"]), *common,
            ]  # fmt: skip
        return [
            "--checkpoint", self.checkpoint(k + 1, entries), "--bot", cfg["bot"], "--out", d,
            "--limit", str(cfg["look_openings"]), "--sims", str(cfg["sims"]),
            "--batch-size", str(cfg["batch_size"]), "--capacity", str(cfg["capacity"]), "--cores", "1",
            "--wall-cap-s", str(cfg["look_wall_s"]), *common, *(["--any-openings"] if cfg.get("any_openings") else []),
        ]  # fmt: skip

    def run_phase(self, k: int, phase: str, n: int, entries: list[dict]) -> None:
        tool = {"selfplay": "selfplay.py", "train": "train_step.py", "look": "look.py"}[phase]
        argv = self.phase_argv(k, phase, n, entries)
        if os.environ.get("LOOP_TEST_INPROCESS") == "1":
            self.run_inprocess(phase, argv)
            return
        template = (self.cfg.get("runners") or {}).get(phase)
        if template:
            # An operator-supplied command (for example a remote container
            # for the registered look runtime); it must leave the attempt's
            # output in place. Placeholders: {python} {tool} {args}.
            cmd = template.format(python=sys.executable, tool=os.path.join(HERE, tool), args=" ".join(map(_q, argv)))
            subprocess.run(cmd, shell=True, check=False)
        else:
            subprocess.run([sys.executable, os.path.join(HERE, tool), *argv], check=False)

    def run_inprocess(self, phase: str, argv: list[str]) -> None:
        """Tests only (``LOOP_TEST_INPROCESS=1``): the phase's work in this
        process, with the same ledger reservation, settlement and run name
        as the supervisor, but no deadline process. Real runs always use
        the supervised tools."""
        import importlib

        module = importlib.import_module({"selfplay": "selfplay", "train": "train_step", "look": "look"}[phase])
        args = module.parse(argv)
        work = {"selfplay": "work", "train": "work_window", "look": "work"}[phase]
        out = os.path.dirname(os.path.abspath(args.out)) if phase == "train" else args.out
        if phase != "train":
            artifacts.fresh_dir(out)
        index, _ = self.ledger.reserve(f"{args.attempt} inprocess", args.cores, args.wall_cap_s)
        start = time.monotonic()
        crash_point(f"during-{phase}")

        class Clock:
            def check(self, projected_s, what):
                pass

        try:
            rc = getattr(module, work)(args, Clock())
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        self.ledger.settle(index, time.monotonic() - start, "ok" if rc == 0 else f"failed: exit {rc}")

    # ------------------------------------------------------------ driver

    def step(self) -> str | None:
        """Does the next unit of work; returns a stop reason or None."""
        entries = self.ledger.entries()
        res = reservations(self.cfg)
        for k in range(self.cfg["k_max"]):
            states = {p: self.phase_state(k, p, entries) for p in PHASES}
            if states["look"][0] is not None:
                self.record_look(k, states["look"][0], entries)
                reason = self.early_stop(k)
                if reason:
                    return reason
                continue
            pending = [p for p in PHASES if states[p][0] is None]
            need = sum(res[p] for p in pending)
            remaining = self.remaining_core_s(entries)
            if need > remaining:
                raise Stop("budget", {"generation": k, "pending": pending, "need_core_s": need, "remaining_core_s": remaining})
            used = disk_bytes(self.root)
            if used > self.cfg.get("max_disk_mb", 600) * 2**20:
                raise Stop("disk", {"generation": k, "used_bytes": used, "max_disk_mb": self.cfg.get("max_disk_mb", 600)})
            phase = pending[0]
            n = states[phase][1]
            if n >= MAX_ATTEMPTS:
                raise Stop("failed", {"generation": k, "phase": phase, "attempts": n})
            self.run_phase(k, phase, n, entries)
            crash_point(f"after-{phase}")
            return None
        raise Stop("K_max", {"k_max": self.cfg["k_max"]})

    def record_look(self, k: int, d: str, entries: list[dict]) -> None:
        """Appends generation k's look once (by checkpoint SHA)."""
        report = lookmod.load(d)
        known = {e["checkpoint"] for e in self.selection() if e["type"] == "look"}
        if report["checkpoint"] in known:
            return
        n = int(os.path.basename(d).rsplit("-", 1)[1])
        self.append_selection(
            {
                "type": "look",
                "generation": k + 1,
                "checkpoint": report["checkpoint"],
                "wins": report["wins"],
                "wdl": report["wdl"],
                "games": report["games"],
                "eligible": report["eligible"],
                "ledger_ok": self.ledger_ok(self.attempt_id(k, "look", n), entries),
                "log_sha256": report["log_sha256"],
                "attempt": self.attempt_id(k, "look", n),
            }
        )
        crash_point("after-selection-append")

    def early_stop(self, k: int) -> str | None:
        looks = {e["generation"]: e for e in self.selection() if e["type"] == "look" and e["eligible"] and e["ledger_ok"]}
        g = k + 1
        last = [looks.get(g - 1), looks.get(g)]
        if all(e is not None and e["wins"] >= self.cfg["early_stop_wins"] for e in last):
            inputs = {"generations": [g - 1, g], "wins": [e["wins"] for e in last], "threshold": self.cfg["early_stop_wins"]}
            raise Stop("early", inputs)
        return None

    def run(self) -> dict:
        lock = open(os.path.join(self.root, "loop.lock"), "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another loop owns this root") from None
        try:
            while True:
                try:
                    self.step()
                except Stop as s:
                    stops = [e for e in self.selection() if e["type"] == "stop"]
                    if not stops or stops[-1]["reason"] != s.reason or stops[-1]["inputs"] != s.inputs:
                        self.append_selection({"type": "stop", "reason": s.reason, "inputs": s.inputs})
                    self.write_cache(s.reason)
                    return {"stop": s.reason, "inputs": s.inputs}
                crash_point("before-state")
                self.write_cache(None)
        finally:
            lock.close()

    def write_cache(self, reason) -> None:
        entries = self.ledger.entries()
        gens = []
        for k in range(self.cfg["k_max"]):
            states = {p: self.phase_state(k, p, entries)[0] for p in PHASES}
            if not any(states.values()):
                break
            gens.append({"generation": k, **{p: (os.path.relpath(v, self.root) if v else None) for p, v in states.items()}})
        artifacts.write_atomic(
            os.path.join(self.root, "loop.json"),
            artifacts.canonical_json({"cache": True, "generations": gens, "stop": reason}),
        )

    def select(self) -> dict:
        looks = [e for e in self.selection() if e["type"] == "look"]
        eligible = [e for e in looks if e["eligible"] and e["ledger_ok"]]
        entries = self.ledger.entries()
        final = None
        for k in range(self.cfg["k_max"]):
            done, _ = self.phase_state(k, "train", entries)
            if done:
                final = {"generation": k + 1, "checkpoint": ckpt.manifest(os.path.join(done, "C.pt"))["sha256"]}
        manifest = {
            "kind": "p2-m3-selection",
            "rule": "most held-out wins among eligible complete looks with an ok ledger entry; ties to the later generation",
            "eligible_looks": eligible,
            "stops": [e for e in self.selection() if e["type"] == "stop"],
            "final_checkpoint": final,
            "note": "the selected held-out score is optimistically biased by repeated looks; never a gate estimate",
        }
        if eligible:
            best = max(eligible, key=lambda e: (e["wins"], e["generation"]))
            manifest["selected"] = {"generation": best["generation"], "checkpoint": best["checkpoint"], "wins": best["wins"]}
        else:
            manifest["selected"] = None
        artifacts.write_atomic(os.path.join(self.root, "selection-manifest.json"), artifacts.canonical_json(manifest))
        return manifest


def disk_bytes(root: str) -> int:
    """Bytes under the loop root (the retained local artifacts)."""
    total = 0
    for base, _, files in os.walk(root):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def init(args) -> int:
    if os.path.exists(args.root) and os.listdir(args.root):
        raise SystemExit(f"{args.root} is not empty")
    _, meta = ckpt.load(args.c0)
    if args.c0_sha256 and meta["sha256"] != args.c0_sha256:
        raise SystemExit(f"C_0 is {meta['sha256']}, not the registered {args.c0_sha256}")
    os.makedirs(os.path.join(args.root, "c0"))
    for ext in (".pt", ".json"):
        shutil.copyfile(os.path.splitext(args.c0)[0] + ext, os.path.join(args.root, "c0", "C" + ext))
    cfg = {
        "c0_sha256": meta["sha256"],
        "base": args.base,
        "k_max": args.k_max,
        "games": args.games,
        "sims": args.sims,
        "batch_size": args.batch_size,
        "capacity": args.capacity,
        "steps": args.steps,
        "train_batch": args.train_batch,
        "selfplay_cores": args.selfplay_cores,
        "selfplay_wall_s": args.selfplay_wall_s,
        "train_wall_s": args.train_wall_s,
        "look_wall_s": args.look_wall_s,
        "look_openings": args.look_openings,
        "early_stop_wins": args.early_stop_wins,
        "max_disk_mb": args.max_disk_mb,
        "budget_core_hours": args.budget_core_hours,
        "bot": os.path.abspath(args.bot),
        "source_sha": args.source_sha,
        "any_openings": args.any_openings,
        "runners": json.loads(args.runners) if args.runners else {},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    artifacts.write_atomic(os.path.join(args.root, "config.json"), artifacts.canonical_json(cfg))
    print(json.dumps({"root": args.root, "c0": meta["sha256"]}))
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--root", required=True)
    i.add_argument("--c0", required=True)
    i.add_argument("--c0-sha256", help="the registered C_0 digest, checked")
    i.add_argument("--base", type=int, default=20261103)
    i.add_argument("--k-max", type=int, default=schedule.K_MAX)
    i.add_argument("--games", type=int, default=10000)
    i.add_argument("--sims", type=int, default=100)
    i.add_argument("--batch-size", type=int, default=8)
    i.add_argument("--capacity", type=int, default=8192)
    i.add_argument("--steps", type=int, default=schedule.STEPS)
    i.add_argument("--train-batch", type=int, default=1024)
    i.add_argument("--selfplay-cores", type=int, default=32)
    i.add_argument("--selfplay-wall-s", type=int, default=900)
    i.add_argument("--train-wall-s", type=int, default=1200)
    i.add_argument("--look-wall-s", type=int, default=900)
    i.add_argument("--look-openings", type=int, default=200)
    i.add_argument("--early-stop-wins", type=int, default=380)
    i.add_argument("--max-disk-mb", type=int, default=600, help="stop before a phase when the root holds more")
    i.add_argument("--budget-core-hours", type=float, default=200.0)
    i.add_argument("--bot", required=True)
    i.add_argument("--source-sha", required=True)
    i.add_argument("--any-openings", action="store_true", help="tests only: looks are never eligible")
    i.add_argument("--runners", help="JSON {phase: command template}")
    for name in ("run", "select"):
        c = sub.add_parser(name)
        c.add_argument("--root", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(list(sys.argv[1:] if argv is None else argv))
    if args.command == "init":
        return init(args)
    loop = Loop(args.root)
    if args.command == "run":
        print(json.dumps(loop.run()))
        return 0
    manifest = loop.select()
    print(json.dumps({"selected": manifest["selected"], "final": manifest["final_checkpoint"]}))
    return 0 if manifest["selected"] else EXIT_NO_SELECTION


if __name__ == "__main__":
    sys.exit(main())
