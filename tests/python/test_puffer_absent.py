# SPDX-License-Identifier: MIT OR Apache-2.0
"""PufferLib cannot be installed next to the main environment's pins, so the
package must not need it. The adapter's tests live in compat/pufferlib."""

import subprocess
import sys

import pytest


def run(code, cwd):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd, timeout=120)


def test_import_tessera_rl_does_not_import_pufferlib(tmp_path):
    out = run("import sys, tessera_rl; print('pufferlib' in sys.modules)", tmp_path)
    assert out.stdout.strip() == "False", out.stderr


def test_import_of_the_adapter_without_pufferlib_names_the_cause(tmp_path):
    code = (
        "import sys\nsys.modules['pufferlib'] = None\n"
        "try:\n    import tessera_rl.puffer\nexcept ImportError as e:\n    print('needs PufferLib' in str(e))\n"
    )
    out = run(code, tmp_path)
    if out.returncode != 0:
        pytest.fail(out.stderr)
    assert out.stdout.strip() == "True"
