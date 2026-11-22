# SPDX-License-Identifier: MIT OR Apache-2.0
"""scripts/gungraun_apt.sh: the gate's Valgrind from a fixed Debian snapshot,
with the image's glibc checked after the install (the deb13u5 drift of
2026-11-22, docs/reviews/gungraun-glibc-drift)."""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gungraun_apt.sh"
TEXT = SCRIPT.read_text()
ENV = (ROOT / "benches" / "gungraun" / "environment.toml").read_text()
WORKFLOWS = ROOT / ".github" / "workflows"


def _const(name):
    return re.search(rf"^{name}=(\S+)$", TEXT, re.M).group(1)


def test_gungraun_apt_pins_the_baseline_glibc_and_valgrind():
    glibc, valgrind = _const("GLIBC"), _const("VALGRIND")
    assert f'glibc = "ldd (Debian GLIBC {glibc}) 2.41"' in ENV
    assert f'valgrind_package = "{valgrind}"' in ENV
    assert re.fullmatch(r"\d{8}T\d{6}Z", _const("SNAPSHOT"))


def test_gungraun_apt_uses_only_the_snapshot_source():
    assert "rm -f /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list" in TEXT
    assert re.search(r'^echo "deb \[check-valid-until=no\] https://snapshot\.debian\.org/archive/debian/\$SNAPSHOT trixie main" > /etc/apt/sources\.list$', TEXT, re.M)
    assert "deb.debian.org" not in TEXT.split("set -eu", 1)[1]


def test_gungraun_apt_is_the_one_install_in_every_place():
    for wf in ("ci.yml", "gungraun-bless.yml"):
        text = (WORKFLOWS / wf).read_text()
        assert "valgrind=" not in text, wf
        assert text.count("- run: scripts/gungraun_apt.sh") == 1, wf
    readme = (ROOT / "benches" / "gungraun" / "README.md").read_text()
    assert "valgrind=" not in readme and "scripts/gungraun_apt.sh &&" in readme


def _fakes(tmp_path, versions, ldd, update_rc=0, install_rc=0):
    fake = tmp_path / "fake"
    fake.mkdir()
    log = tmp_path / "log"

    def tool(name, body):
        p = fake / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    tool("apt-get", f'echo "apt-get $*" >> {log}\ncase " $* " in *" update "*) exit {update_rc};; esac\nexit {install_rc}\n')
    cases = "".join(f"  {pkg}) printf %s '{ver}';;\n" for pkg, ver in versions.items())
    tool("dpkg-query", f'eval last=\\${{$#}}\ncase "$last" in\n{cases}esac\n')
    tool("ldd", f"echo '{ldd}'\n")
    tool("rm", f'echo "rm $*" >> {log}\n')
    return fake, log


def _run(tmp_path, fake):
    dash = shutil.which("dash") or shutil.which("sh")
    # The script writes /etc/apt/sources.list: run a copy that writes into tmp_path.
    copy = tmp_path / "gungraun_apt.sh"
    copy.write_text(TEXT.replace("/etc/apt/sources.list", str(tmp_path / "sources.list")))
    env = {"PATH": f"{fake}:/usr/bin:/bin"}
    return subprocess.run([dash, str(copy)], env=env, capture_output=True, text=True, check=False)


GOOD = {"libc6": _const("GLIBC"), "libc6-dbg": _const("GLIBC"), "libc-bin": _const("GLIBC"), "valgrind": _const("VALGRIND")}
GOOD_LDD = f"ldd (Debian GLIBC {_const('GLIBC')}) 2.41"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX sh with fake tools")


def test_gungraun_apt_passes_on_the_pinned_versions(tmp_path):
    fake, log = _fakes(tmp_path, GOOD, GOOD_LDD)
    p = _run(tmp_path, fake)
    assert p.returncode == 0, p.stderr
    lines = log.read_text().splitlines()
    assert lines[1] == "apt-get -o Acquire::Retries=5 update"
    assert lines[2] == f"apt-get -o Acquire::Retries=5 install -y --no-upgrade valgrind={_const('VALGRIND')} libc6-dbg={_const('GLIBC')}"
    assert (tmp_path / "sources.list").read_text().startswith("deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/")
    assert p.stdout.startswith("gungraun_apt: " + GOOD_LDD)


@pytest.mark.parametrize("pkg", ["libc6", "libc6-dbg", "libc-bin", "valgrind"])
def test_gungraun_apt_stops_when_an_installed_version_is_not_the_pin(tmp_path, pkg):
    versions = dict(GOOD, **{pkg: "2.41-12+deb13u5"})
    fake, log = _fakes(tmp_path, versions, GOOD_LDD)
    p = _run(tmp_path, fake)
    assert p.returncode == 1 and f"{pkg} is 2.41-12+deb13u5 after the install" in p.stderr, p.stderr


def test_gungraun_apt_stops_when_ldd_is_not_the_pin(tmp_path):
    fake, log = _fakes(tmp_path, GOOD, "ldd (Debian GLIBC 2.41-12+deb13u5) 2.41")
    p = _run(tmp_path, fake)
    assert p.returncode == 1 and "ldd reports" in p.stderr


@pytest.mark.parametrize("stage, kw, msg", [
    ("update", {"update_rc": 100}, "snapshot service is unreachable"),
    ("install", {"install_rc": 100}, "apt-get install of valgrind="),
])
def test_gungraun_apt_fails_with_a_clear_message_when_apt_fails(tmp_path, stage, kw, msg):
    fake, log = _fakes(tmp_path, GOOD, GOOD_LDD, **kw)
    p = _run(tmp_path, fake)
    assert p.returncode == 1 and msg in p.stderr, (stage, p.stderr)
