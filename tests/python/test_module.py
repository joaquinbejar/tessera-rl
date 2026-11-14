# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native module imports and reports its version."""

import tessera_rl


def test_module_reports_its_version():
    assert tessera_rl.version() == "0.2.0"
    assert tessera_rl.__version__ == "0.2.0"
