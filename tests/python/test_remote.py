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

HOSTS = {"selfplay": ("swarmer02", "0-31"), "train": ("swarmer02", "0"), "look": ("xaskar", "2")}


@pytest.fixture(scope="module")
def bot():
    return bot_binary()


def env_for(tmp_path, crash=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REMOTE_TEST", "LOOP_TEST"))}
    env.update(REMOTE_TEST_TRANSPORT="remote_fake:factory", REMOTE_TEST_PATH=str(HERE),
               REMOTE_TEST_HOSTS=str(tmp_path / "hosts"))  # fmt: skip
    if crash:
        env["REMOTE_TEST_CRASH"] = crash
    return env


def remote_root(tmp_path, bot, **over):
    runners = {
        phase: f"{{python}} {EXAMPLE / 'remote.py'} run --host {h} --cpuset {c} --root {tmp_path / 'root'} -- {{tool}} {{args}}"
        for phase, (h, c) in HOSTS.items()
    }
    return make_root(tmp_path, bot, **{"--k-max": 1, "--runners": json.dumps(runners), **over})


def drive(root, env, *, reconcile=True):
    """The driver: reconcile, then loop.py run, in its own session (a crash
    point kills the whole driver group, never the stand-in containers)."""
    if reconcile:
        subprocess.run([sys.executable, str(EXAMPLE / "remote.py"), "reconcile", "--root", str(root)],
                       env=env, check=True, start_new_session=True, timeout=300)  # fmt: skip
    return subprocess.run([sys.executable, str(EXAMPLE / "loop.py"), "run", "--root", str(root)], env=env,
                          capture_output=True, text=True, start_new_session=True, timeout=300)  # fmt: skip


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


@pytest.mark.parametrize("crash", ["after-start", "after-pull", "after-publish"])
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
        host.kill("tessera-m3-g000-selfplay-0")
    finally:
        del os.environ["REMOTE_TEST_HOSTS"]
    r = drive(root, env_for(tmp_path))
    assert r.returncode == 0, r.stderr
    (a0, s0), (a1, s1) = runs(root)[:2]
    assert (a0, a1, s1) == ("g000-selfplay-0", "g000-selfplay-1", "ok") and s0.startswith("interrupted")
    first = next(d for d in dispatches(root) if d["attempt"] == "g000-selfplay-0")
    assert first["gone_evidence"]["docker_ps_by_name"] == "" and first["gone_evidence"]["checked"]


def test_remote_open_dispatch_refuses_a_new_one(tmp_path, bot):
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "mid-push"))
    assert r.returncode != 0
    args = type("A", (), {"root": str(root), "host": "swarmer02", "cpuset": "0"})()
    with pytest.raises(remote.TransportError, match="open dispatch"):
        remote.dispatch(args, ["selfplay.py", "--attempt", "x", "--out", str(root / "gen-000" / "selfplay-9")])


def test_remote_divergent_ledger_is_rejected_and_kept(tmp_path, bot):
    root = remote_root(tmp_path, bot)
    r = drive(root, env_for(tmp_path, "after-start"))
    assert r.returncode != 0
    rl = tmp_path / "hosts" / "swarmer02" / "root" / "ledger.json"
    wait_for(lambda: (tmp_path / "hosts" / "swarmer02" / "containers" / "tessera-m3-g000-selfplay-0.json.exit").exists())
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
    remote.SSH("swarmer02").docker_run("tessera-m3-x", "0-31", "swarmer02", ["/work/tessera/examples/a.py", "--x"])
    cmd = seen[-1]
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
