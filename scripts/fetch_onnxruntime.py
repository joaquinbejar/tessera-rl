# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fetch the pinned onnxruntime wheel for this platform and extract its library.

Usage: python3 scripts/fetch_onnxruntime.py <dest-dir>

Downloads the onnxruntime 1.31.1 wheel for this platform from PyPI's file URL,
checks it against the sha256 pinned below, and extracts the shared library
and its siblings (onnxruntime/capi/libonnxruntime*) into <dest-dir>. It prints
the library's path on stdout. Nothing is imported, loaded or installed, and
nothing is written outside <dest-dir>. Standard library only, so no tool has
to be installed on the runner.

The pins are the cp313 wheels; the library inside a wheel does not depend on
the CPython tag. Each sha256 is PyPI's digest for that file, and equals the
hash spike S1 recorded for the same file (ADR-0057). The version moves with
compat/targets.toml (`onnxruntime` in [env.main]).
"""

import hashlib
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

VERSION = "1.31.1"
PYPI = "https://files.pythonhosted.org/packages"
WHEELS = {
    ("Linux", "x86_64"): (
        f"{PYPI}/fc/0a/c887bcb84a9455b2723825381a77af38611a80f996f804e20fde5f671b63/"
        "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_x86_64.whl",
        "74a6a52994616c4d18b33db1e13c226333a70a6ba07480a5cab9c2a50dff1430",
        "libonnxruntime.so.1.31.1",
    ),
    ("Linux", "aarch64"): (
        f"{PYPI}/e1/e1/633d4994db0620d31f34d9962119bc4ebd0b2ef4c9a030f91c7843fb9672/"
        "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_aarch64.whl",
        "dcaedc50333876f3829ab966b242147f0e548229e29c36fe44622ecf748c25ee",
        "libonnxruntime.so.1.31.1",
    ),
    ("Darwin", "arm64"): (
        f"{PYPI}/b2/93/d2b855793302ef57d55cc0583b72a00a8dcb01d9bdb3adac380f09b050e8/"
        "onnxruntime-1.31.1-cp313-cp313-macosx_14_0_arm64.whl",
        "e4ad067d36e2220218e085d7223283b2ee1bc73d887e811fd14d21cf901a11ee",
        "libonnxruntime.1.31.1.dylib",
    ),
}
MEMBER_PREFIX = "onnxruntime/capi/libonnxruntime"


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: fetch_onnxruntime.py <dest-dir>")
    dest = Path(argv[1])
    key = (platform.system(), platform.machine())
    if key not in WHEELS:
        sys.exit(f"fetch_onnxruntime.py: no pinned onnxruntime {VERSION} wheel for {key}")
    url, sha256, library = WHEELS[key]
    dest.mkdir(parents=True, exist_ok=True)
    wheel = dest / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.exit(f"fetch_onnxruntime.py: {wheel.name} has sha256 {got}, pinned {sha256}")
    wheel.write_bytes(data)
    with zipfile.ZipFile(wheel) as archive:
        members = [m for m in archive.namelist() if m.startswith(MEMBER_PREFIX)]
        for member in members:
            target = dest / Path(member).name
            target.write_bytes(archive.read(member))
    path = dest / library
    if not path.is_file():
        sys.exit(f"fetch_onnxruntime.py: {library} not found in {wheel.name}")
    print(path)


if __name__ == "__main__":
    main(sys.argv)
