# SPDX-License-Identifier: MIT OR Apache-2.0
"""Jobs that use uv install nothing: they put pipx's bin directory on the
job's PATH only if the uv there is exactly the pinned 0.13.6, and then check
that uv is found. The runner service's PATH may lack ~/.local/bin (it did
after a runner restart on 2026-11-13)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

INSTALL = """      - run: |
          d=$(pipx environment --value PIPX_BIN_DIR)
          v=$("$d/uv" --version) || { echo "no runnable uv in pipx's bin directory $d" >&2; exit 1; }
          case "$v" in "uv 0.13.6" | "uv 0.13.6 "*) ;; *) echo "uv in $d is '$v', not the pinned uv 0.13.6" >&2; exit 1 ;; esac
          echo "$d" >> "$GITHUB_PATH"
      - run: command -v uv >/dev/null || { echo "uv not on PATH" >&2; exit 1; }
"""


def _texts():
    return {p.name: p.read_text() for p in WORKFLOWS}


@pytest.mark.parametrize("name", [p.name for p in WORKFLOWS])
def test_uv_install_nothing_is_installed_and_the_pinned_uv_is_checked(name):
    text = _texts()[name]
    uses = len(re.findall(r"PIPX_BIN_DIR", text))
    assert text.count(INSTALL) == uses, name
    assert "pipx install" not in text, name


def test_uv_install_is_used_by_the_known_jobs():
    texts = _texts()
    assert texts["ci.yml"].count(INSTALL) == 2  # python, pufferlib
    assert texts["learning-test.yml"].count(INSTALL) == 1
    assert texts["reference-training.yml"].count(INSTALL) == 1
