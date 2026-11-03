# SPDX-License-Identifier: MIT OR Apache-2.0
"""The resumable AlphaZero loop of P2-M3 (PR 3, design revision 2).

**Generations.** Generation ``k`` runs three phases:
1. self-play with ``C_k``, run seed ``seeds.generation(base, k)``;
2. training ``C_{k+1}`` for ``schedule.STEPS`` steps over the window of
   generations ``max(0, k-3)..k``, warm-started from ``C_k`` with a fresh
   optimizer;
3. a held-out look at ``C_{k+1}`` (``look.py``). Looks start at ``C_1``,
   so ``C_0`` is never selectable.

``C_0`` is given at ``init`` and pinned by SHA-256.

**Evidence, not cache.** Every phase attempt ``(k, phase, n)`` has a
fresh directory ``gen-<k>/<phase>-<n>/`` and its ledger run name. An
attempt is never reused or overwritten.

An attempt is **done** when two things hold:
- its output is complete and digest-verified, or it was evicted after a
  verified upload, with its descriptor and its asset entry kept;
- its ledger entry is settled ``ok``.

At that point the loop writes ``attempt.json``, the attempt's immutable
identity: its output digests, its inputs and the command it ran.

Every decision is recomputed from these descriptors, the ledger and the
pinned lineage:
- ``C_0`` matches the configured digest;
- self-play names ``C_k``;
- training names ``C_k`` as its parent and the window's self-play
  manifests as its data;
- the look names ``C_{k+1}``.

A referenced attempt whose descriptor exists but whose output is gone or
changed stops the loop with ``incomplete input``, and so does any broken
link of the chain. The loop never regenerates a different history.

``selection.jsonl`` is an append-only audit log, one look entry per
generation plus every stop decision with its inputs. ``loop.json`` is a
cache. Selection and the early stop never trust either of them.

**Looks.** A look at a checkpoint that already has a verified look with
the same protocol is not run again: the generation references it.

**Admission.** A phase starts only if the ledger's remaining budget (the
settled spend plus every open reservation) covers the frozen reservations
of the generation's remaining phases. The stop reasons are ``K_max``,
``budget``, ``early``, ``failed``, ``disk`` and ``incomplete input``.

**Formal.** A run is formal only with the registered configuration:
- ``K_max`` 30, 1000 steps, early-stop at 380 wins, budget 200, window 4;
- 10,000 games, base 20261103, 100 sims, batch 8, capacity 8192;
- train batch 1024, 200 look openings;
- the registered ``C_0`` (medium), and test settings off.

Only a formal run has eligible looks. The look itself must also be
eligible: registered runtime and bot, frozen protocol, medium net.

**Selection** (``select``, under the owner lock) picks, among eligible
verified looks, the most held-out wins, ties going to the later
generation. It writes ``selection-manifest.json``, naming the selected
checkpoint and the final (last trained) checkpoint separately. With no
eligible look there is no selection, and therefore no gate.

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
REGISTERED_C0 = "89855dce4aa429d657af96c5effc42917cb18edbedb5f614e458832a6cb7ff7d"
FORMAL = {
    "k_max": schedule.K_MAX,
    "steps": schedule.STEPS,
    "early_stop_wins": 380,
    "budget_core_hours": 200.0,
    "window": WINDOW,
    "games": 10000,
    "base": 20261103,
    "sims": 100,
    "batch_size": 8,
    "capacity": 8192,
    "train_batch": 1024,
    "look_openings": 200,
    "c0_sha256": REGISTERED_C0,
    "any_openings": False,
}
LOAD_ERRORS = (artifacts.Incomplete, OSError, KeyError, ValueError, ckpt.NonFinite)


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


def is_formal(cfg: dict) -> bool:
    return all(cfg.get(k) == v for k, v in FORMAL.items())


class Loop:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.ledger = Ledger(os.path.join(root, "ledger.json"), self.cfg["budget_core_hours"])

    # ------------------------------------------------------------ attempts

    def attempt_dir(self, k: int, phase: str, n: int) -> str:
        return os.path.join(self.root, f"gen-{k:03d}", f"{phase}-{n}")

    def attempt_id(self, k: int, phase: str, n: int) -> str:
        return f"g{k:03d}-{phase}-{n}"

    def ledger_ok(self, attempt: str, entries: list[dict]) -> bool:
        return any(e["run"].startswith(attempt + " ") and e["status"] == "ok" for e in entries)

    def identity(self, phase: str, d: str) -> dict:
        """The verified identity of an attempt's output; raises one of
        ``LOAD_ERRORS`` if it is not complete."""
        if phase == "selfplay":
            manifest, digest = artifacts.load_manifest(d)
            return {"manifest_sha256": digest, "games_digest": manifest["games_digest"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"]}  # fmt: skip
        if phase == "train":
            _, meta = ckpt.load(os.path.join(d, "C.pt"))
            return {"checkpoint_sha256": meta["sha256"], "parent_sha256": meta["parent_sha256"],
                    "data_manifests": meta["data_manifests"], "size": meta["size"]}  # fmt: skip
        report = lookmod.load(d)
        return {"checkpoint_sha256": report["checkpoint"], "log_sha256": report["log_sha256"],
                "wins": report["wins"], "eligible": report["eligible"], "protocol": report["protocol"]}  # fmt: skip

    def descriptor(self, d: str) -> dict | None:
        path = os.path.join(d, "attempt.json")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def command(self, d: str) -> list[str] | None:
        path = d + ".command.json"
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def resolve(self, k: int, phase: str, entries: list[dict]) -> tuple[dict | None, int]:
        """The done attempt of ``(k, phase)`` as its descriptor (or None),
        and the next attempt number. Raises ``Stop("incomplete input")``
        when a recorded attempt lost or changed its output."""
        n, done = 0, None
        while os.path.exists(self.attempt_dir(k, phase, n)) or os.path.exists(self.attempt_dir(k, phase, n) + ".command.json"):
            d = self.attempt_dir(k, phase, n)
            attempt = self.attempt_id(k, phase, n)
            desc = self.descriptor(d) if os.path.isdir(d) else None
            if desc is not None:
                # The ledger is authoritative: a descriptor without its ok
                # settlement (a failed, missing or reset ledger) is broken
                # history, never a done attempt.
                if not self.ledger_ok(attempt, entries):
                    raise Stop("incomplete input", {"attempt": attempt, "reason": "no ok ledger settlement for a recorded attempt"})
                if not desc.get("evicted"):
                    try:
                        ok = self.identity(phase, d) == desc["identity"]
                    except LOAD_ERRORS:
                        ok = False
                    if not ok:
                        raise Stop("incomplete input", {"attempt": attempt, "reason": "recorded output missing or changed"})
                if done is None:
                    done = {**desc, "dir": d}
            elif done is None and self.ledger_ok(attempt, entries):
                try:
                    ident = self.identity(phase, d)
                except LOAD_ERRORS:
                    ident = None
                if ident is not None:
                    desc = {"attempt": attempt, "generation": k, "phase": phase, "identity": ident,
                            "command": self.command(d)}  # fmt: skip
                    artifacts.write_atomic(os.path.join(d, "attempt.json"), artifacts.canonical_json(desc))
                    done = {**desc, "dir": d}
            n += 1
        ref = os.path.join(self.root, f"gen-{k:03d}", f"{phase}-ref")
        if done is None and phase == "look" and os.path.exists(os.path.join(ref, "attempt.json")):
            desc = self.descriptor(ref)
            origin = self.origin(desc["ref"], entries)
            if origin["identity"] != desc["identity"]:
                raise Stop("incomplete input", {"attempt": desc["attempt"], "reason": "a look reference differs from its origin"})
            done = {**desc, "dir": ref}
        return done, n

    def origin(self, attempt: str, entries: list[dict]) -> dict:
        """The verified, settled look attempt a reference names."""
        gen, _, n = attempt.split("-")
        k = int(gen[1:])
        d = self.attempt_dir(k, "look", int(n))
        desc = self.descriptor(d) if os.path.isdir(d) else None
        if desc is None or desc["attempt"] != attempt or not self.ledger_ok(attempt, entries):
            raise Stop("incomplete input", {"attempt": attempt, "reason": "a referenced look is missing or unsettled"})
        if not desc.get("evicted"):
            try:
                ok = self.identity("look", d) == desc["identity"]
            except LOAD_ERRORS:
                ok = False
            if not ok:
                raise Stop("incomplete input", {"attempt": attempt, "reason": "recorded output missing or changed"})
        return desc

    # ------------------------------------------------------------ lineage

    def c0_sha(self) -> str:
        path = os.path.join(self.root, "c0", "C.pt")
        try:
            _, meta = ckpt.load(path)
        except LOAD_ERRORS as e:
            raise Stop("incomplete input", {"attempt": "c0", "reason": str(e)}) from None
        if meta["sha256"] != self.cfg["c0_sha256"]:
            raise Stop("incomplete input", {"attempt": "c0", "reason": f"C_0 is {meta['sha256']}, not the pinned one"})
        return meta["sha256"]

    def chain(self, entries: list[dict]) -> list[dict]:
        """Per generation, the verified done phases, with every lineage link
        checked, up to the first generation without a look. Stops on a
        broken link or a missing referenced input."""
        sha = self.c0_sha()
        gens = []
        for k in range(self.cfg["k_max"]):
            states = {p: self.resolve(k, p, entries) for p in PHASES}
            sp, tr, lk = (states[p][0] for p in PHASES)
            if tr is not None and sp is None:
                raise Stop("incomplete input", {"generation": k, "reason": "training references missing self-play"})
            if lk is not None and tr is None:
                raise Stop("incomplete input", {"generation": k, "reason": "a look references a missing checkpoint"})
            if sp is not None and sp["identity"]["checkpoint_sha256"] != sha:
                raise Stop("incomplete input", {"generation": k, "reason": "self-play does not name C_k"})
            c_next = None
            if tr is not None:
                window = [gens[j]["selfplay"]["identity"]["manifest_sha256"] for j in range(max(0, k - WINDOW + 1), k)]
                window.append(sp["identity"]["manifest_sha256"])
                if tr["identity"]["parent_sha256"] != sha or tr["identity"]["data_manifests"] != window:
                    raise Stop("incomplete input", {"generation": k, "reason": "training does not name C_k and its window"})
                c_next = tr["identity"]["checkpoint_sha256"]
            if lk is not None and lk["identity"]["checkpoint_sha256"] != c_next:
                raise Stop("incomplete input", {"generation": k, "reason": "the look does not name C_{k+1}"})
            gens.append({"selfplay": sp, "train": tr, "look": lk, "next": {p: states[p][1] for p in PHASES}, "c_k1": c_next})
            if lk is None:
                break
            sha = c_next
        return gens

    def checkpoint_path(self, gens: list[dict], k: int) -> str:
        if k == 0:
            return os.path.join(self.root, "c0", "C.pt")
        tr = gens[k - 1]["train"]
        if tr.get("evicted"):
            raise Stop("incomplete input", {"generation": k, "reason": f"restore {tr['attempt']} (the checkpoint) first"})
        return os.path.join(tr["dir"], "C.pt")

    # ------------------------------------------------------------ audit log

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

    def looks(self, gens: list[dict], entries: list[dict]) -> list[dict]:
        """Every generation's look, from verified evidence. Eligible only on
        a formal run, with an eligible report and an ok ledger entry for the
        look that ran."""
        out = []
        for k, g in enumerate(gens):
            lk = g["look"]
            if lk is None:
                continue
            source = lk.get("ref", lk["attempt"])
            eligible = is_formal(self.cfg) and lk["identity"]["eligible"] and self.ledger_ok(source, entries)
            out.append({"generation": k + 1, "checkpoint": lk["identity"]["checkpoint_sha256"],
                        "wins": lk["identity"]["wins"], "eligible": bool(eligible), "look": source,
                        "log_sha256": lk["identity"]["log_sha256"]})  # fmt: skip
        return out

    # ------------------------------------------------------------ phases

    def phase_argv(self, gens: list[dict], k: int, phase: str, n: int) -> list[str]:
        cfg, d, attempt = self.cfg, self.attempt_dir(k, phase, n), self.attempt_id(k, phase, n)
        common = [
            "--source-sha", cfg["source_sha"], "--attempt", attempt,
            "--ledger", os.path.join(self.root, "ledger.json"),
            "--budget-core-hours", str(cfg["budget_core_hours"]),
        ]  # fmt: skip
        if phase == "selfplay":
            return [
                "--checkpoint", self.checkpoint_path(gens, k), "--games", str(cfg["games"]),
                "--workers", str(cfg["selfplay_cores"]), "--run-seed", str(seeds.generation(cfg["base"], k)),
                "--out", d, "--sims", str(cfg["sims"]), "--batch-size", str(cfg["batch_size"]),
                "--capacity", str(cfg["capacity"]), "--cores", str(cfg["selfplay_cores"]),
                "--wall-cap-s", str(cfg["selfplay_wall_s"]), *common,
            ]  # fmt: skip
        if phase == "train":
            window = []
            for j in range(max(0, k - WINDOW + 1), k + 1):
                sp = gens[j]["selfplay"]
                if sp.get("evicted"):
                    raise Stop("incomplete input", {"generation": k, "reason": f"restore {sp['attempt']} (a window input)"})
                window.append(sp["dir"])
            return [
                "train-window", "--checkpoint", self.checkpoint_path(gens, k), "--data", *window,
                "--out", os.path.join(d, "C.pt"), "--generation", str(k), "--base", str(cfg["base"]),
                "--steps", str(cfg["steps"]), "--batch", str(cfg["train_batch"]), "--cores", "1",
                "--wall-cap-s", str(cfg["train_wall_s"]), *(["--formal"] if is_formal(cfg) else []), *common,
            ]  # fmt: skip
        return [
            "--checkpoint", self.checkpoint_path(gens, k + 1), "--bot", cfg["bot"], "--bot-sha256", cfg["bot_sha256"],
            "--out", d, "--limit", str(cfg["look_openings"]), "--sims", str(cfg["sims"]),
            "--batch-size", str(cfg["batch_size"]), "--capacity", str(cfg["capacity"]), "--cores", "1",
            "--wall-cap-s", str(cfg["look_wall_s"]), *common, *(["--any-openings"] if cfg.get("any_openings") else []),
        ]  # fmt: skip

    def run_phase(self, gens: list[dict], k: int, phase: str, n: int) -> None:
        tool = {"selfplay": "selfplay.py", "train": "train_step.py", "look": "look.py"}[phase]
        argv = self.phase_argv(gens, k, phase, n)
        d = self.attempt_dir(k, phase, n)
        # The command sits next to the attempt: self-play and looks need a
        # fresh output directory.
        os.makedirs(os.path.dirname(d), exist_ok=True)
        artifacts.write_atomic(d + ".command.json", artifacts.canonical_json([tool, *argv]))
        if phase == "train":
            os.makedirs(d)
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
        if phase != "train":
            artifacts.fresh_dir(args.out)
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
        crash_point(f"output-before-settle-{phase}")
        self.ledger.settle(index, time.monotonic() - start, "ok" if rc == 0 else f"failed: exit {rc}")

    # ------------------------------------------------------------ driver

    def reconcile_transport(self) -> None:
        """Closes any open remote dispatch before the next attempt is
        allocated (remote.py); a failure stops the loop with its evidence
        kept."""
        template = self.cfg.get("reconcile")
        if not template:
            return
        cmd = template.format(python=sys.executable, root=_q(self.root))
        r = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        if r.returncode != 0:
            raise Stop("transport", {"command": cmd, "exit": r.returncode, "stderr": r.stderr[-2000:]})

    def step(self) -> None:
        """Does the next unit of work, or raises ``Stop``."""
        self.reconcile_transport()
        entries = self.ledger.entries()
        gens = self.chain(entries)
        self.audit(gens, entries)
        self.early_stop(gens, entries)
        k = len(gens) - 1
        if gens[k]["look"] is not None:
            if k + 1 >= self.cfg["k_max"]:
                raise Stop("K_max", {"k_max": self.cfg["k_max"]})
            k += 1
            gens.append({"selfplay": None, "train": None, "look": None, "next": dict.fromkeys(PHASES, 0), "c_k1": None})
        g = gens[k]
        pending = [p for p in PHASES if g[p] is None]
        if pending == ["look"] and self.reuse_look(gens, k):
            return
        res = reservations(self.cfg)
        need = sum(res[p] for p in pending)
        remaining = self.remaining_core_s(entries)
        if need > remaining:
            raise Stop("budget", {"generation": k, "pending": pending, "need_core_s": need, "remaining_core_s": remaining})
        used = disk_bytes(self.root)
        if used > self.cfg.get("max_disk_mb", 600) * 2**20:
            raise Stop("disk", {"generation": k, "used_bytes": used, "max_disk_mb": self.cfg.get("max_disk_mb", 600)})
        phase = pending[0]
        n = g["next"][phase]
        if n >= MAX_ATTEMPTS:
            raise Stop("failed", {"generation": k, "phase": phase, "attempts": n})
        self.run_phase(gens, k, phase, n)
        crash_point(f"after-{phase}")

    def reuse_look(self, gens: list[dict], k: int) -> bool:
        """References an earlier verified look at the same checkpoint and
        protocol instead of running it again."""
        sha = gens[k]["c_k1"]
        for j in range(k):
            lk = gens[j]["look"]
            if lk is not None and lk["identity"]["checkpoint_sha256"] == sha:
                ref = os.path.join(self.root, f"gen-{k:03d}", "look-ref")
                os.makedirs(ref, exist_ok=True)
                desc = {"attempt": f"g{k:03d}-look-ref", "generation": k, "phase": "look",
                        "ref": lk.get("ref", lk["attempt"]), "identity": lk["identity"]}  # fmt: skip
                artifacts.write_atomic(os.path.join(ref, "attempt.json"), artifacts.canonical_json(desc))
                return True
        return False

    def audit(self, gens: list[dict], entries: list[dict]) -> None:
        """Appends each generation's look to the audit log, once."""
        logged = {e["generation"] for e in self.selection() if e["type"] == "look"}
        for e in self.looks(gens, entries):
            if e["generation"] not in logged:
                self.append_selection({"type": "look", **e})
                crash_point("after-selection-append")

    def early_stop(self, gens: list[dict], entries: list[dict]) -> None:
        looks = {e["generation"]: e for e in self.looks(gens, entries) if e["eligible"]}
        threshold = self.cfg["early_stop_wins"]
        for g in sorted(looks):
            last = [looks.get(g - 1), looks.get(g)]
            if all(e is not None and e["wins"] >= threshold for e in last):
                raise Stop("early", {"generations": [g - 1, g], "wins": [e["wins"] for e in last], "threshold": threshold})

    def owner(self):
        lock = open(os.path.join(self.root, "loop.lock"), "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise SystemExit("another loop owns this root") from None
        return lock

    def run(self) -> dict:
        lock = self.owner()
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
        try:
            gens = self.chain(self.ledger.entries())
        except Stop:
            gens = []
        summary = [{p: (g[p]["attempt"] if g[p] else None) for p in PHASES} for g in gens]
        artifacts.write_atomic(
            os.path.join(self.root, "loop.json"),
            artifacts.canonical_json({"cache": True, "formal": is_formal(self.cfg), "generations": summary, "stop": reason}),
        )

    def select(self) -> dict:
        lock = self.owner()
        try:
            entries = self.ledger.entries()
            gens = self.chain(entries)
            eligible = [e for e in self.looks(gens, entries) if e["eligible"]]
            trained = [(k + 1, g["c_k1"]) for k, g in enumerate(gens) if g.get("c_k1")]
            manifest = {
                "kind": "p2-m3-selection",
                "formal": is_formal(self.cfg),
                "rule": "most held-out wins among eligible verified looks; ties to the later generation",
                "eligible_looks": eligible,
                "stops": [e for e in self.selection() if e["type"] == "stop"],
                "final_checkpoint": {"generation": trained[-1][0], "checkpoint": trained[-1][1]} if trained else None,
                "note": "the selected held-out score is optimistically biased by repeated looks; never a gate estimate",
            }
            manifest["selected"] = choose(eligible)
            artifacts.write_atomic(os.path.join(self.root, "selection-manifest.json"), artifacts.canonical_json(manifest))
            return manifest
        finally:
            lock.close()


def choose(eligible: list[dict]) -> dict | None:
    """The selection rule: the most held-out wins, ties to the later
    generation; None without an eligible look."""
    if not eligible:
        return None
    best = max(eligible, key=lambda e: (e["wins"], e["generation"]))
    return {"generation": best["generation"], "checkpoint": best["checkpoint"], "wins": best["wins"]}


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
    bot_sha = artifacts.sha256_file(args.bot)
    if args.bot_sha256 and bot_sha != args.bot_sha256:
        raise SystemExit(f"the bot binary is {bot_sha}, not the expected {args.bot_sha256}")
    os.makedirs(os.path.join(args.root, "c0"))
    for ext in (".pt", ".json"):
        shutil.copyfile(os.path.splitext(args.c0)[0] + ext, os.path.join(args.root, "c0", "C" + ext))
    cfg = {
        "c0_sha256": meta["sha256"],
        "base": args.base,
        "k_max": args.k_max,
        "window": WINDOW,
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
        "bot_sha256": bot_sha,
        "source_sha": args.source_sha,
        "any_openings": args.any_openings,
        "runners": json.loads(args.runners) if args.runners else {},
        "reconcile": args.reconcile,
    }
    cfg["formal"] = is_formal(cfg)
    artifacts.write_atomic(os.path.join(args.root, "config.json"), artifacts.canonical_json(cfg))
    print(json.dumps({"root": args.root, "c0": meta["sha256"], "formal": cfg["formal"]}))
    return 0


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--root", required=True)
    i.add_argument("--c0", required=True)
    i.add_argument("--c0-sha256", help="the registered C_0 digest, checked")
    i.add_argument("--base", type=int, default=FORMAL["base"])
    i.add_argument("--k-max", type=int, default=schedule.K_MAX)
    i.add_argument("--games", type=int, default=FORMAL["games"])
    i.add_argument("--sims", type=int, default=FORMAL["sims"])
    i.add_argument("--batch-size", type=int, default=FORMAL["batch_size"])
    i.add_argument("--capacity", type=int, default=FORMAL["capacity"])
    i.add_argument("--steps", type=int, default=schedule.STEPS)
    i.add_argument("--train-batch", type=int, default=FORMAL["train_batch"])
    i.add_argument("--selfplay-cores", type=int, default=32)
    i.add_argument("--selfplay-wall-s", type=int, default=900)
    i.add_argument("--train-wall-s", type=int, default=1200)
    i.add_argument("--look-wall-s", type=int, default=900)
    i.add_argument("--look-openings", type=int, default=FORMAL["look_openings"])
    i.add_argument("--early-stop-wins", type=int, default=FORMAL["early_stop_wins"])
    i.add_argument("--max-disk-mb", type=int, default=600, help="stop before a phase when the root holds more")
    i.add_argument("--budget-core-hours", type=float, default=FORMAL["budget_core_hours"])
    i.add_argument("--bot", required=True)
    i.add_argument("--bot-sha256", help="the expected bot binary digest, checked")
    i.add_argument("--source-sha", required=True)
    i.add_argument("--any-openings", action="store_true", help="tests only: never formal, looks never eligible")
    i.add_argument("--runners", help="JSON {phase: command template}")
    i.add_argument("--reconcile", help="command run before every step, e.g. remote.py reconcile; placeholders {python} {root}")
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
    try:
        manifest = loop.select()
    except Stop as s:
        print(json.dumps({"stop": s.reason, "inputs": s.inputs}))
        return EXIT_NO_SELECTION
    print(json.dumps({"selected": manifest["selected"], "final": manifest["final_checkpoint"]}))
    return 0 if manifest["selected"] else EXIT_NO_SELECTION


if __name__ == "__main__":
    sys.exit(main())
