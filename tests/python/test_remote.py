# SPDX-License-Identifier: MIT OR Apache-2.0
"""The P2-M3 phase transport (examples/alphazero_connect4/remote.py), on a
local stand-in host (remote_fake.py): the round trip and every crash
boundary."""

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

HERE = pathlib.Path(__file__).parent
EXAMPLE = HERE.parents[1] / "examples" / "alphazero_connect4"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(HERE))

import loop  # noqa: E402
import remote  # noqa: E402
import remote_fake  # noqa: E402
from test_loop import bot_binary, identities, make_root  # noqa: E402

# The cpusets match the tiny loop's 1-core phases.
HOSTS = {"selfplay": ("swarmer02", "0"), "train": ("swarmer02", "0"), "look": ("xaskar", "2")}


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


def env_for(tmp_path, crash=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REMOTE_TEST", "LOOP_TEST"))}
    env.update(REMOTE_TEST_TRANSPORT="remote_fake:factory", REMOTE_TEST_PATH=str(HERE),
               REMOTE_TEST_HOSTS=str(tmp_path / "hosts"), REMOTE_POLL_S="0.2")  # fmt: skip
    if crash:
        env["REMOTE_TEST_CRASH"] = crash
    return env


def provision_hosts(tmp_path):
    """Deploys a stand-in source tree and uv to both stand-in hosts, as
    `remote.py provision` does on the real ones; returns the tree hash."""
    import tarfile

    src = tmp_path / "src-archive"
    src.mkdir(parents=True)
    (src / "README").write_text("the reviewed source tree (stand-in)\n")
    archive = tmp_path / "src.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(src, arcname=".")
    uv = tmp_path / "uv"
    uv.write_bytes(b"uv stand-in")
    tree = None
    for host in ("swarmer02", "xaskar"):
        p = subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "provision", "--host", host,
                            "--archive", str(archive), "--uv", str(uv)], env=env_for(tmp_path),
                           capture_output=True, text=True, check=True)  # fmt: skip
        tree = json.loads(p.stdout)["source_tree_sha256"]
    return tree


def remote_root(tmp_path, bot, **over):
    tree = provision_hosts(tmp_path)
    runners = {
        phase: f"{{python}} {EXAMPLE / 'remote.py'} run --host {h} --cpuset {c} --root {tmp_path / 'root'} "
        f"--source-tree-sha256 {tree} -- {{tool}} {{args}}"
        for phase, (h, c) in HOSTS.items()
    }
    reconcile = f"{{python}} {EXAMPLE / 'remote.py'} reconcile --root {{root}} --owner-held"
    return make_root(tmp_path, bot, **{"--k-max": 1, "--runners": json.dumps(runners), "--reconcile": reconcile, **over})


def drive(root, env):
    """The driver, loop.py run, in its own session: a crash point kills the
    whole driver group, never the stand-in containers. The loop reconciles
    any open dispatch before each step."""
    return subprocess.run([sys.executable, str(EXAMPLE / "loop.py"), "run", "--root", str(root)], env=env,
                          capture_output=True, text=True, start_new_session=True, timeout=300)  # fmt: skip


def cname(root, attempt):
    return f"tessera-m3-{remote.artifacts.sha256_file(str(pathlib.Path(root) / 'config.json'))[:8]}-{attempt}"


def last_stop(r):
    return json.loads(r.stdout.strip().splitlines()[-1])["stop"]


def runs(root):
    return [(e["run"].split()[0], e["status"]) for e in loop.Loop(str(root)).ledger.entries()]


def dispatches(root):
    return [json.loads(p.read_text()) for p in sorted(pathlib.Path(root).rglob("*.dispatch.json"))]


def host_clean(tmp_path):
    """No container record and no attempt payload left on any host."""
    left = []
    for h in (tmp_path / "hosts").glob("*"):
        left += [p.name for p in (h / "containers").glob("*")]
        left += [p.name for p in (h / "root").glob("gen-*/*")]
    return left


@pytest.fixture(scope="module")
def direct(tmp_path_factory, bot):
    """The same generation with the phases run locally, for identity."""
    tmp = tmp_path_factory.mktemp("direct")
    root = make_root(tmp, bot, **{"--k-max": 1})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REMOTE_TEST", "LOOP_TEST"))}
    r = subprocess.run([sys.executable, str(EXAMPLE / "loop.py"), "run", "--root", str(root)], env=env,
                       capture_output=True, text=True, timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stderr
    return identities(root)


def test_remote_one_generation_round_trip(tmp_path, bot, direct):
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path))
    assert r.returncode == 0, r.stderr
    assert last_stop(r) == "K_max"
    assert runs(root) == [("g000-selfplay-0", "ok"), ("g000-train-0", "ok"), ("g000-look-0", "ok")]
    assert identities(root) == direct, "the transport never changes a result"
    ds = dispatches(root)
    assert [d["phase"] for d in ds] == ["look", "selfplay", "train"] and all(d["imported"] and d["closed"] for d in ds)
    assert {d["host"] for d in ds} == {"swarmer02", "xaskar"} and all(d["image"] == remote.IMAGE for d in ds)
    assert host_clean(tmp_path) == []


def wait_for(cond, timeout=120):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.2)
    raise AssertionError("timed out")


@pytest.mark.parametrize("crash", ["after-docker-run", "after-start", "after-pull", "after-publish"])
def test_remote_driver_death_after_start_imports_the_work_once(tmp_path, bot, direct, crash):
    """The driver dies with the phase still running, after staging, or
    after publication before the acknowledgement: the next driver imports
    the successful remote attempt once, never charging it as interrupted."""
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, crash))
    assert r.returncode != 0  # the driver group was killed
    r = drive(root, env_for(tmp_path))
    assert r.returncode == 0, r.stderr
    assert last_stop(r) == "K_max"
    assert runs(root) == [("g000-selfplay-0", "ok"), ("g000-train-0", "ok"), ("g000-look-0", "ok")]
    assert identities(root) == direct
    assert host_clean(tmp_path) == []


def test_remote_death_before_the_container_charges_nothing(tmp_path, bot, direct):
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "mid-push"))
    assert r.returncode != 0
    r = drive(root, env_for(tmp_path))
    assert r.returncode == 0, r.stderr
    # The lost attempt 0 left no ledger entry; attempt 1 ran.
    assert runs(root)[0] == ("g000-selfplay-1", "ok")
    assert identities(root) == direct


def test_remote_vanished_container_is_settled_with_evidence(tmp_path, bot):
    """The container disappears after its reservation (host reboot): the
    entry is settled at the full reservation only after a by-name check,
    and the attempt is retried as n + 1."""
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    host = remote_fake.FakeHost.__new__(remote_fake.FakeHost)
    os.environ["REMOTE_TEST_HOSTS"] = str(tmp_path / "hosts")
    try:
        host = remote_fake.FakeHost("swarmer02")
        rl = tmp_path / "hosts" / "swarmer02" / "root" / "ledger.json"
        wait_for(lambda: rl.exists() and any(e["status"] == "running" for e in json.loads(rl.read_text())["entries"]))
        host.kill(cname(root, "g000-selfplay-0"))
    finally:
        del os.environ["REMOTE_TEST_HOSTS"]
    r = drive(root, env_for(tmp_path))
    assert r.returncode == 0, r.stderr
    (a0, s0), (a1, s1) = runs(root)[:2]
    assert (a0, a1, s1) == ("g000-selfplay-0", "g000-selfplay-1", "ok") and s0.startswith("interrupted")
    first = next(d for d in dispatches(root) if d["attempt"] == "g000-selfplay-0")
    assert first["terminal_evidence"]["docker_ps_by_name"] == "" and first["terminal_evidence"]["checked"]


def test_remote_open_dispatch_refuses_a_new_one(tmp_path, bot):
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "mid-push"))
    assert r.returncode != 0
    args = type("A", (), {"root": str(root), "host": "swarmer02", "cpuset": "0", "source_tree_sha256": "x"})()
    with pytest.raises(remote.TransportError, match="open dispatch"):
        remote.dispatch(args, ["selfplay.py", "--attempt", "x", "--out", str(root / "gen-000" / "selfplay-9")])


def test_remote_divergent_ledger_is_rejected_and_kept(tmp_path, bot):
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    rl = tmp_path / "hosts" / "swarmer02" / "root" / "ledger.json"
    wait_for(lambda: (tmp_path / "hosts" / "swarmer02" / "containers" / f"{cname(root, 'g000-selfplay-0')}.json.exit").exists())
    doc = json.loads(rl.read_text())
    doc["entries"].insert(0, {**doc["entries"][0], "run": "forged 0"})
    rl.write_text(json.dumps(doc))
    p = subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "reconcile", "--root", str(root)],
                       env=env_for(tmp_path), capture_output=True, text=True, timeout=120)  # fmt: skip
    assert p.returncode != 0 and "TransportError" in p.stderr
    d = dispatches(root)[0]
    assert d["closed"] is None and not d.get("imported")
    assert rl.exists(), "the remote evidence is kept"


def test_remote_accept_ledger_rules():
    sent = [{"run": "a 1", "status": "ok"}]
    assert remote.accept_ledger(sent, sent + [{"run": "g 2", "status": "ok"}], "g")[-1]["run"] == "g 2"
    assert remote.accept_ledger(sent, sent, "g") == sent
    for bad in ([], [{"run": "a 1", "status": "failed"}], sent + [{"run": "h 2", "status": "ok"}],
                sent + [{"run": "g 2", "status": "ok"}, {"run": "g 3", "status": "ok"}]):  # fmt: skip
        with pytest.raises(remote.TransportError):
            remote.accept_ledger(sent, bad, "g")


def test_remote_ssh_docker_command_uses_absolute_bind_mounts_only(monkeypatch):
    seen = []

    def fake(self, cmd, check=True):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "/home/u\n", "")

    monkeypatch.setattr(remote.SSH, "_ssh", fake)
    remote.SSH("swarmer02").docker_run("tessera-m3-x", "0-31", "swarmer02", ["/work/tessera/examples/a.py", "--x"], 1800)
    cmd = seen[-1]
    assert "timeout --kill-after 30 1800 sh -c" in cmd, "the deadline lives in the container"
    assert cmd.startswith("docker run -d --name tessera-m3-x ")
    mounts = [cmd.split()[i + 1] for i, t in enumerate(cmd.split()) if t == "-v"]
    assert mounts and all(m.startswith("/home/u/") for m in mounts), mounts
    assert remote.IMAGE in cmd and "--cpuset-cpus 0-31" in cmd and "--hostname swarmer02" in cmd


def test_remote_mapping_keeps_the_scientific_arguments(tmp_path):
    root = tmp_path / "root"
    (root / "c0").mkdir(parents=True)
    (root / "c0" / "C.pt").write_bytes(b"x")
    bin_ = tmp_path / "bot"
    bin_.write_bytes(b"b")
    argv = ["look.py", "--checkpoint", str(root / "c0" / "C.pt"), "--bot", str(bin_), "--sims", "100",
            "--out", str(root / "gen-000" / "look-0"), "--attempt", "g000-look-0"]  # fmt: skip
    cargv, inputs = remote.mapped(str(root), argv)
    assert cargv[0] == "/work/tessera/examples/alphazero_connect4/look.py"
    assert cargv[2] == "/m3root/c0/C.pt" and cargv[4].startswith("/m3root/_inputs/") and cargv[6] == "100"
    assert cargv[8] == "/m3root/gen-000/look-0"
    assert set(inputs) == {str(root / "c0" / "C.pt"), str(bin_)}


def test_remote_standalone_reconcile_takes_the_owner_lock(tmp_path, bot):
    import fcntl

    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    with open(root / "loop.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        p = subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "reconcile", "--root", str(root)],
                           env=env_for(tmp_path), capture_output=True, text=True, timeout=300)  # fmt: skip
        assert p.returncode != 0 and "another loop" in p.stderr
    p = subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "reconcile", "--root", str(root)],
                       env=env_for(tmp_path), capture_output=True, text=True, timeout=300)  # fmt: skip
    assert p.returncode == 0, p.stderr
    assert all(d["closed"] for d in dispatches(root))


def test_remote_overdue_container_is_killed_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("REMOTE_TEST_HOSTS", str(tmp_path / "hosts"))
    monkeypatch.setenv("REMOTE_POLL_S", "0.1")
    host = remote_fake.FakeHost("swarmer02")
    host.docker_run("tessera-m3-hang", "0", "swarmer02", ["-c", "import time; time.sleep(600)"], 3600)
    path = tmp_path / "x.dispatch.json"
    record = {"container": "tessera-m3-hang", "dispatched_epoch": time.time() - 10_000, "wall_cap_s": 60.0}
    state = remote.wait_or_kill(host, record, str(path))
    assert state == ("exited", 137) and record["killed_overdue"]["deadline_epoch"] < time.time()
    saved = json.loads(path.read_text())
    assert saved["killed_overdue"] == record["killed_overdue"] and saved["published_partial"] is True
    assert host.docker_ps_by_name("tessera-m3-hang"), "killed, not removed: cleanup removes it later"
    host.docker_rm("tessera-m3-hang")


def test_remote_container_names_are_unique_per_experiment(tmp_path, bot):
    a = remote_root(tmp_path / "a", bot)
    b = remote_root(tmp_path / "b", bot, **{"--base": 7})
    assert cname(a, "g000-selfplay-0") != cname(b, "g000-selfplay-0")


class Stub:
    """subprocess.run for the SSH class: canned results per command."""

    def __init__(self, table):
        self.table, self.seen = table, []

    def __call__(self, argv, **kw):
        cmd = argv[-1] if argv[0] == "ssh" else " ".join(argv)
        self.seen.append(cmd)
        for key, (rc, out, err) in self.table.items():
            if key in cmd:
                return subprocess.CompletedProcess(argv, rc, out, err)
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_remote_ssh_failure_is_never_read_as_gone(monkeypatch):
    stub = Stub({"": (255, "", "ssh: connect to host swarmer02: Connection refused")})
    monkeypatch.setattr(remote.subprocess, "run", stub)
    h = remote.SSH("swarmer02")
    for call in (lambda: h.docker_state("n"), lambda: h.docker_ps_by_name("n"), lambda: h.exists("root/x"),
                 lambda: h.isdir("root/x")):  # fmt: skip
        with pytest.raises(remote.TransportError, match="255"):
            call()


def test_remote_ssh_docker_state_absent_only_on_no_such_object(monkeypatch):
    h = remote.SSH("swarmer02")
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker inspect": (1, "", "Error: No such object: n")}))
    assert h.docker_state("n") is None
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker inspect": (1, "", "permission denied on the socket")}))
    with pytest.raises(remote.TransportError):
        h.docker_state("n")
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker inspect": (0, "running 0\n", "")}))
    assert h.docker_state("n") == ("running", None)
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker inspect": (0, "exited 3\n", "")}))
    assert h.docker_state("n") == ("exited", 3)
    monkeypatch.setattr(remote.subprocess, "run", Stub({"test -e": (2, "", "bad")}))
    with pytest.raises(remote.TransportError):
        h.exists("root/x")


def _docker_fixture(key):
    path = os.path.join(os.path.dirname(__file__), "fixtures", "docker_29_9_2.json")
    with open(path, encoding="utf-8") as f:
        c = json.load(f)["commands"][key]
    return c["returncode"], c["stdout"], c["stderr"]


def test_remote_ssh_docker_29_missing_name_reads_as_absent(monkeypatch):
    # Verbatim Docker 29.9.2 output captured from both hosts: lowercase
    # "error: no such object" on stderr and a lone newline on stdout.
    h = remote.SSH("xaskar")
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker inspect": _docker_fixture("inspect_absent")}))
    assert h.docker_state("tessera-m3-fixture-absent") is None
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker ps": _docker_fixture("ps_absent")}))
    assert h.docker_ps_by_name("tessera-m3-fixture-absent") == ""
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker logs": _docker_fixture("logs_absent")}))
    assert "No such container" in h.docker_logs("tessera-m3-fixture-absent")


def test_remote_ssh_docker_state_older_and_newer_wording_both_absent(monkeypatch):
    h = remote.SSH("xaskar")
    for err in ("Error: No such object: n", "error: no such object: n", "Error response from daemon: No such container: n"):
        monkeypatch.setattr(remote.subprocess, "run", Stub({"docker inspect": (1, "\n", err)}))
        assert h.docker_state("n") is None


def test_remote_ssh_commands_render_valid_templates(monkeypatch):
    stub = Stub({"docker ps": (0, "tessera-m3-n Up 1 minute\n", ""), "docker inspect": (0, "running 0\n", "")})
    monkeypatch.setattr(remote.subprocess, "run", stub)
    h = remote.SSH("swarmer02")
    assert h.docker_ps_by_name("tessera-m3-n") == "tessera-m3-n Up 1 minute"
    assert "--format '{{.Names}} {{.Status}}'" in stub.seen[-1]
    h.docker_state("tessera-m3-n")
    assert "--format '{{.State.Status}} {{.State.ExitCode}}'" in stub.seen[-1]
    monkeypatch.setattr(remote.subprocess, "run", Stub({"docker ps": (1, "", "boom")}))
    with pytest.raises(remote.TransportError):
        h.docker_ps_by_name("tessera-m3-n")


def test_remote_partial_attempts_are_never_packed(tmp_path):
    import assets

    d = tmp_path / "gen-000" / "selfplay-0"
    d.mkdir(parents=True)
    assert not assets.published_partial(str(d))
    (tmp_path / "gen-000" / "selfplay-0.dispatch.json").write_text(json.dumps({"published_partial": True}))
    assert assets.published_partial(str(d))


def crashed_after_start(tmp_path, bot):
    """A root whose first dispatch (self-play) finished remotely while the
    driver was dead: open, unimported, the remote result in place."""
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    exitf = tmp_path / "hosts" / "swarmer02" / "containers" / f"{cname(root, 'g000-selfplay-0')}.json.exit"
    wait_for(exitf.exists)
    return root, tmp_path / "hosts" / "swarmer02" / "root"


def reconcile(tmp_path, root):
    return subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "reconcile", "--root", str(root)],
                          env=env_for(tmp_path), capture_output=True, text=True, timeout=300)  # fmt: skip


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_remote_ok_attempt_with_bad_output_is_refused_and_kept(tmp_path, bot, damage):
    root, hroot = crashed_after_start(tmp_path, bot)
    out = hroot / "gen-000" / "selfplay-0"
    if damage == "corrupt":
        (out / "manifest.json").write_text("CORRUPT NOT JSON")
    else:
        import shutil

        shutil.rmtree(out)
    p = reconcile(tmp_path, root)
    assert p.returncode != 0 and ("invalid" in p.stderr or "missing" in p.stderr), p.stderr[-500:]
    d = dispatches(root)[0]
    assert not d.get("imported") and d["closed"] is None
    assert (hroot / "ledger.json").exists() and (damage == "missing" or out.exists()), "remote evidence kept"
    assert not (root / "gen-000" / "selfplay-0").exists()


def test_remote_local_ledger_changed_since_dispatch_is_refused(tmp_path, bot):
    """A concurrent writer of the local ledger after the dispatch."""
    from ledger import Ledger

    root, _ = crashed_after_start(tmp_path, bot)
    Ledger(str(root / "ledger.json"), 1.0).reserve("intruder 1", 1, 1)
    p = reconcile(tmp_path, root)
    assert p.returncode != 0 and "TransportError" in p.stderr
    assert not dispatches(root)[0].get("imported")


def test_remote_entry_validation_rules():
    record = {"cores": 32, "wall_cap_s": 900.0}
    good = {"run": "g 1", "cores": 32, "deadline_s": 900.0, "wall_s": 600.0, "core_hours": round(600 * 32 / 3600, 6),
            "status": "ok"}  # fmt: skip
    remote.validate_entry(good, record)
    remote.validate_entry({**good, "status": "interrupted (settled by hand at the full reservation)", "wall_s": None,
                           "core_hours": 8.0}, record)  # fmt: skip
    for st in ("interrupted by signal 15", "killed at the deadline (898.0 s of work)", "failed: child exit 1",
               "ok (overran the 900.0 s reservation)"):  # fmt: skip
        remote.validate_entry({**good, "status": st}, record)  # charged the measured wall
    bad = [
        {**good, "cores": 0}, {**good, "cores": 16}, {**good, "deadline_s": 0}, {**good, "deadline_s": 901},
        {**good, "core_hours": -1}, {**good, "core_hours": float("nan")}, {**good, "core_hours": 1.0},
        {**good, "status": "running"}, {**good, "status": "weird"}, {**good, "wall_s": None},
        {**good, "status": "interrupted (settled at the full reservation)", "core_hours": 1.0},
        {**good, "status": "interrupted by signal 15", "core_hours": 8.0},
    ]  # fmt: skip
    for e in bad:
        with pytest.raises(remote.TransportError):
            remote.validate_entry(e, record)


def test_remote_stale_source_is_refused_before_reserve(tmp_path, bot):
    root = remote_root(tmp_path, bot)
    (tmp_path / "hosts" / "swarmer02" / "src" / "README").write_text("a stale checkout\n")
    r = drive(root, env_for(tmp_path))
    assert last_stop(r) in ("failed", "transport")
    assert runs(root) == [], "nothing was reserved or run"
    assert not list((tmp_path / "hosts" / "swarmer02" / "containers").glob("*.json"))


def test_remote_container_deadline_holds_without_the_driver(tmp_path, monkeypatch):
    monkeypatch.setenv("REMOTE_TEST_HOSTS", str(tmp_path / "hosts"))
    host = remote_fake.FakeHost("swarmer02")
    host.docker_run("tessera-m3-setup-hang", "0", "swarmer02", ["-c", "import time; time.sleep(600)"], 1.0)
    wait_for(lambda: host.docker_state("tessera-m3-setup-hang") == ("exited", 124), timeout=20)
    host.docker_rm("tessera-m3-setup-hang")


def test_remote_owner_busy_reconcile_touches_nothing(tmp_path, bot, monkeypatch):
    import fcntl

    root, hroot = crashed_after_start(tmp_path, bot)
    calls = []
    monkeypatch.setattr(remote, "transport", lambda host: calls.append(host))
    before = {p: p.read_bytes() for p in pathlib.Path(root).rglob("*") if p.is_file() and p.name != "loop.lock"}
    with open(root / "loop.lock", "a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="another loop"):
            remote.main(["reconcile", "--root", str(root)])
    assert calls == []
    after = {p: p.read_bytes() for p in pathlib.Path(root).rglob("*") if p.is_file() and p.name != "loop.lock"}
    assert after == before


def test_remote_built_extension_does_not_change_the_source_hash(tmp_path):
    src = tmp_path / "src"
    (src / "python" / "tessera_rl").mkdir(parents=True)
    (src / "python" / "tessera_rl" / "__init__.py").write_text("x = 1\n")
    before = remote.tree_sha256(str(src))
    (src / "python" / "tessera_rl" / "_native.abi3.so").write_bytes(b"\x7fELF built")
    (src / "build").mkdir()
    (src / "build" / "lib.o").write_bytes(b"o")
    assert remote.tree_sha256(str(src)) == before
    (src / "python" / "tessera_rl" / "__init__.py").write_text("x = 2\n")
    assert remote.tree_sha256(str(src)) != before


def test_remote_exited_container_with_open_entry_is_settled_conservatively(tmp_path, bot):
    """A container killed (exit 137) while its entry is still open, e.g.
    by the driver's overdue safeguard: terminal, so it is settled at the
    full reservation with its state recorded, then retried as n + 1."""
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    rl = tmp_path / "hosts" / "swarmer02" / "root" / "ledger.json"
    wait_for(lambda: rl.exists() and any(e["status"] == "running" for e in json.loads(rl.read_text())["entries"]))
    os.environ["REMOTE_TEST_HOSTS"] = str(tmp_path / "hosts")
    try:
        host = remote_fake.FakeHost("swarmer02")
        host.docker_kill(cname(root, "g000-selfplay-0"))
        assert host.docker_state(cname(root, "g000-selfplay-0")) == ("exited", 137)
        # The killed supervisor never settles: put its entry back to open.
        doc = json.loads(rl.read_text())
        doc["entries"][-1].update(status="running", wall_s=None)
        doc["entries"][-1]["core_hours"] = round(doc["entries"][-1]["deadline_s"] * doc["entries"][-1]["cores"] / 3600, 6)
        rl.write_text(json.dumps(doc))
    finally:
        del os.environ["REMOTE_TEST_HOSTS"]
    r = drive(root, env_for(tmp_path))
    assert r.returncode == 0, r.stderr
    (a0, s0), (a1, s1) = runs(root)[:2]
    assert (a0, a1, s1) == ("g000-selfplay-0", "g000-selfplay-1", "ok") and s0.startswith("interrupted (settled")
    first = next(d for d in dispatches(root) if d["attempt"] == "g000-selfplay-0")
    assert first["terminal_evidence"]["docker_state"] == ["exited", 137] and first["published_partial"]
