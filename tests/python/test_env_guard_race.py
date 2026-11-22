# SPDX-License-Identifier: MIT OR Apache-2.0
"""The ambient CARGO_PROFILE_* / CARGO_TARGET_* guard of the three build scripts never fails
open. Its earlier form, `env | grep -Eq` under pipefail, missed a set variable when `env` died
of SIGPIPE (3.5 % to 25 % of runs on a Linux runner). The guard now uses no pipe, and each
script's guard is run here a hundred times, with a large environment, under the scripts' own shell
options."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "wheel_job": ROOT / "scripts" / "wheel_job.sh",
    "pgo_candidate_checks": ROOT / "scripts" / "pgo_candidate_checks.sh",
    "pgo_w2_build": ROOT / "scripts" / "pgo_w2_build.sh",
}
# A reintroduced pipe misses in nearly every run at this environment size (1985 of 2000 on a
# Linux runner), so a hundred runs detect it, and the no-pipe check pins the fix besides.
RUNS = 100

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX only: bash scripts")


def guard(script: Path) -> str:
    """The guard as the script has it: from `ambient=` to its `fi`."""
    m = re.search(r"^ambient=.*?^fi$", script.read_text(), re.M | re.S)
    assert m, f"no ambient guard in {script.name}"
    assert "|" not in m.group(0).replace("||", ""), "the guard uses no pipe"
    return m.group(0)


@pytest.mark.parametrize("name", sorted(SCRIPTS))
@pytest.mark.parametrize("value", ["x", ""], ids=["set", "empty"])
def test_env_guard_refuses_a_set_variable_in_every_run(name, value):
    # A large environment (about 128 KiB), the variable among many others, as the race needs.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_"))}
    env.update({f"TESSERA_PAD_{i:04d}": "p" * 120 for i in range(1000)})
    env["CARGO_PROFILE_RELEASE_OPT_LEVEL"] = value
    body = guard(SCRIPTS[name])
    loop = (
        "set -euo pipefail\n"
        f"refused=0\nfor _ in $(seq {RUNS}); do\n"
        f"    if ( {body}\n    ) 2>/dev/null; then :; else [ $? -eq 2 ] && refused=$((refused + 1)); fi\n"
        "done\necho \"$refused\"\n"
    )
    p = subprocess.run(["bash", "-c", loop], env=env, capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == str(RUNS), f"{name}: refused in {p.stdout.strip()} of {RUNS} runs"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_env_guard_passes_a_clean_environment_and_names_every_variable(name):
    body = guard(SCRIPTS[name])
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("CARGO_PROFILE_", "CARGO_TARGET_"))}
    p = subprocess.run(["bash", "-c", f"set -euo pipefail\n{body}\necho passed"], env=clean, capture_output=True, text=True, check=False)
    assert (p.returncode, p.stdout.strip()) == (0, "passed"), p.stderr
    both = {**clean, "CARGO_TARGET_DIR": "/t", "CARGO_PROFILE_RELEASE_LTO": ""}
    p = subprocess.run(["bash", "-c", f"set -euo pipefail\n{body}"], env=both, capture_output=True, text=True, check=False)
    assert p.returncode == 2
    assert f"{name}: CARGO_PROFILE_RELEASE_LTO CARGO_TARGET_DIR set in the environment; refused" in p.stderr
