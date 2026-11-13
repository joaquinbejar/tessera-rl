# SPDX-License-Identifier: MIT OR Apache-2.0
"""Every pipx install of uv puts pipx's bin directory on the job's PATH and
is followed by a check that uv is found: the runner service's PATH may lack
~/.local/bin (it did after a runner restart on 2026-11-13)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

INSTALL = """      - run: |
          pipx install uv==0.13.6
          echo "$(pipx environment --value PIPX_BIN_DIR)" >> "$GITHUB_PATH"
      - run: command -v uv >/dev/null || { echo "uv not on PATH after pipx install" >&2; exit 1; }
"""


def _texts():
    return {p.name: p.read_text() for p in WORKFLOWS}


@pytest.mark.parametrize("name", [p.name for p in WORKFLOWS])
def test_uv_install_every_pipx_install_puts_its_bin_dir_on_path(name):
    text = _texts()[name]
    installs = len(re.findall(r"pipx install uv", text))
    assert text.count(INSTALL) == installs, name
    assert "- run: pipx install uv" not in text, name


def test_uv_install_is_used_by_the_known_jobs():
    texts = _texts()
    assert texts["ci.yml"].count(INSTALL) == 2  # python, pufferlib
    assert texts["learning-test.yml"].count(INSTALL) == 1
    assert texts["reference-training.yml"].count(INSTALL) == 1
