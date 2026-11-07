# SPDX-License-Identifier: MIT OR Apache-2.0
"""The synchronised copy of PufferLib's trainer for measurement mode (b).

``docs/reviews/gpu-split-plan.md`` r5: PuffeRL's ``Profile`` times phases
with ``time.time()`` and leaves ``torch.cuda.synchronize()`` commented out at
its two timing points, so on CUDA queued GPU work is charged to whichever
phase next blocks. Mode (b) runs a scratch copy of ``pufferl.py`` with exactly
those two blocks enabled, and nothing else changed. ``pufferl-sync.patch``
beside this file is that diff, published before any run; the tests check that
the copy produced here matches it.

The installed ``pufferl.py`` is never modified.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import os
import sys

# The two commented blocks, as they appear in PufferLib 3.0.0's pufferl.py
# (Profile.__call__ and Profile.end).
COMMENTED = "        #if torch.cuda.is_available():\n        #    torch.cuda.synchronize()\n"
ENABLED = "        if torch.cuda.is_available():\n            torch.cuda.synchronize()\n"
EXPECTED_BLOCKS = 2
MODULE = "pufferl_sync"
PATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pufferl-sync.patch")


class PatchError(RuntimeError):
    """The installed pufferl.py is not the one this patch was written for."""


def original_path() -> str:
    spec = importlib.util.find_spec("pufferlib.pufferl")
    if spec is None or spec.origin is None:
        raise PatchError("pufferlib.pufferl is not importable")
    return spec.origin


def patched_source(source: str) -> str:
    """``source`` with exactly the two synchronize blocks enabled."""
    found = source.count(COMMENTED)
    if found != EXPECTED_BLOCKS:
        raise PatchError(f"expected {EXPECTED_BLOCKS} commented synchronize blocks, found {found}")
    if ENABLED in source:
        raise PatchError("an enabled synchronize block is already present")
    return source.replace(COMMENTED, ENABLED)


def diff(source: str) -> str:
    """The unified diff from the installed file to the synchronised copy."""
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched_source(source).splitlines(keepends=True),
            fromfile="a/pufferlib/pufferl.py",
            tofile="b/pufferlib/pufferl.py",
        )
    )


def load(scratch_dir: str):
    """Writes the synchronised copy into ``scratch_dir`` and imports it as
    ``pufferl_sync``. Returns the module and the SHA-256 of the original and
    of the copy, for the run record."""
    path = original_path()
    with open(path, encoding="utf-8") as f:
        source = f.read()
    patched = patched_source(source)
    target = os.path.join(scratch_dir, f"{MODULE}.py")
    with open(target, "w", encoding="utf-8") as f:
        f.write(patched)
    spec = importlib.util.spec_from_file_location(MODULE, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    digests = {
        "pufferl_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "pufferl_sync_sha256": hashlib.sha256(patched.encode()).hexdigest(),
    }
    return module, digests


def main() -> int:
    """Prints the diff against the installed PufferLib, to regenerate the
    published patch."""
    with open(original_path(), encoding="utf-8") as f:
        sys.stdout.write(diff(f.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
