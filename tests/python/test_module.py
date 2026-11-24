# SPDX-License-Identifier: MIT OR Apache-2.0
"""The native module imports and reports its version."""

import tessera_rl


def test_module_reports_its_version():
    assert tessera_rl.version() == "0.3.0"
    assert tessera_rl.__version__ == "0.3.0"


def test_physical_cores_returns_positive_int_capped_by_cpu_count():
    import os

    cores = tessera_rl.physical_cores()
    assert isinstance(cores, int)
    assert cores >= 1
    assert cores <= (os.cpu_count() or cores)
    assert tessera_rl.physical_cores() == cores
