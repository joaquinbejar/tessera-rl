# SPDX-License-Identifier: MIT OR Apache-2.0
"""Test-only start gate for a supervised child (issue #174).

A test puts this directory on ``PYTHONPATH`` and sets
``TESSERA_TEST_START_GATE=PATH``. A process started with
``--child-deadline-epoch`` (a supervised child) then writes its pid to
``PATH.held`` and waits, before any line of its script runs, until ``PATH``
exists (at most 120 s). The test kills the supervisor inside that window
and then releases the child: a deterministic stand-in for a supervisor
that dies while its child is still importing. Every other process,
including the supervisor and the child's pool workers, passes through.
"""

import os
import sys
import time

_gate = os.environ.get("TESSERA_TEST_START_GATE")
if _gate and "--child-deadline-epoch" in sys.argv:
    del os.environ["TESSERA_TEST_START_GATE"]  # not for the child's own workers
    with open(_gate + ".held.tmp", "w", encoding="utf-8") as _f:
        _f.write(str(os.getpid()))
    os.replace(_gate + ".held.tmp", _gate + ".held")
    _end = time.monotonic() + 120
    while not os.path.exists(_gate) and time.monotonic() < _end:
        time.sleep(0.02)
