# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fetch the pinned onnxruntime wheel for this platform and extract its library.

Usage: python3 scripts/fetch_onnxruntime.py <dest-dir>

Downloads the onnxruntime 1.31.1 wheel for this platform from PyPI's file URL,
checks it against the sha256 pinned below, and extracts only the shared
library (onnxruntime/capi/libonnxruntime.*) into <dest-dir>, after checking
its own pinned sha256: no unchecked file is written next to it. It prints
the library's path on stdout. Nothing is imported, loaded or installed, and
nothing is written outside <dest-dir>. Standard library only, so no tool has
to be installed on the runner.

These pins are the only place the wheels and libraries are pinned in the
repository (tests/python/test_onnx_workflow.py checks that the workflow uses
this script and no other hash). Each wheel is the one whose library spike S1
examined for ADR-0057 (its platform rows: Linux x86-64 and Linux aarch64 from
the cp313 wheels, macOS arm64 from the cp312 wheel); the wheel's sha256 is
PyPI's digest for that file, and the library's sha256 is that of
that file inside it, as the spike extracted it. The
CPython tag of the wheel does not matter here (nothing Python is used), but
it matters for the bytes: on macOS arm64 the cp312 and cp313 wheels carry
different library files, so the tag is pinned too. The library's sha256 is
checked before the file is written, so nothing can load an unchecked file.
The version moves with compat/targets.toml (`onnxruntime` in [env.main]).
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
        "fc44112fbe06d7d689c57e11d3862031f64711007e4ee020f8eed5185fd2c1d3",
    ),
    ("Linux", "aarch64"): (
        f"{PYPI}/e1/e1/633d4994db0620d31f34d9962119bc4ebd0b2ef4c9a030f91c7843fb9672/"
        "onnxruntime-1.31.1-cp313-cp313-manylinux_2_28_aarch64.whl",
        "dcaedc50333876f3829ab966b242147f0e548229e29c36fe44622ecf748c25ee",
        "libonnxruntime.so.1.31.1",
        "8aa3b76b28c778a2b1dd660d34161adb1d508c9ccd04ff9dc80e26eca5df8467",
    ),
    ("Darwin", "arm64"): (
        f"{PYPI}/0e/f7/cecc0efeeea83bff2f7789ae1b80f27fa0f0210d6d9d66d1f0031a6b5e47/"
        "onnxruntime-1.31.1-cp312-cp312-macosx_14_0_arm64.whl",
        "96dd2564b10ea7153ad9c207e7cd8e5917cb3ba521e0e0a62ac03486fdc8d56c",
        "libonnxruntime.1.31.1.dylib",
        "d586cbe89effc085a09d3341b3cfbd5c322517f9d397aeb20814af21433f2b36",
    ),
}


def main(argv):
    if len(argv) != 2:
        sys.exit("usage: fetch_onnxruntime.py <dest-dir>")
    dest = Path(argv[1])
    key = (platform.system(), platform.machine())
    if key not in WHEELS:
        sys.exit(f"fetch_onnxruntime.py: no pinned onnxruntime {VERSION} wheel for {key}")
    url, sha256, library, library_sha256 = WHEELS[key]
    dest.mkdir(parents=True, exist_ok=True)
    wheel = dest / url.rsplit("/", 1)[1]
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.exit(f"fetch_onnxruntime.py: {wheel.name} has sha256 {got}, pinned {sha256}")
    wheel.write_bytes(data)
    with zipfile.ZipFile(wheel) as archive:
        member = f"onnxruntime/capi/{library}"
        payload = archive.read(member)
        got = hashlib.sha256(payload).hexdigest()
        if got != library_sha256:
            sys.exit(f"fetch_onnxruntime.py: {library} has sha256 {got}, pinned {library_sha256}")
    path = dest / library
    path.write_bytes(payload)
    print(path)


if __name__ == "__main__":
    main(sys.argv)
