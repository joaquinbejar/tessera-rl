# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fails unless a pytest junit report of tests/python/test_onnx_evaluator.py
has exactly the expected skips (scripts/py_onnx_test.sh, ADR-0057).

Usage: python check_onnx_skips.py <junit.xml>

In CI the library is pinned only by fetch_onnxruntime.py, so exactly the
two tests that need the onnxruntime package skip, with that reason. Any
other skip (a real-library test that silently stops running, for example
because TESSERA_ONNX_TEST did not reach the tests) or a missing one is an
error, as is a report with no test at all.
"""

import sys
import xml.etree.ElementTree as ET

REASON = (
    "needs the onnxruntime package ([onnx] extra); CI pins the library only, by sha256, "
    "through scripts/fetch_onnxruntime.py"
)
EXPECTED = {
    "test_onnx_default_library_lookup_never_imports_onnxruntime": REASON,
    "test_onnx_works_next_to_the_onnxruntime_package": REASON,
}


def main(path: str) -> int:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skipped = {}
    for case in cases:
        for skip in case.findall("skipped"):
            skipped[case.get("name")] = skip.get("message", "")
    print(f"check-onnx-skips: {len(cases)} tests, {len(skipped)} skipped")
    for name, message in sorted(skipped.items()):
        print(f"check-onnx-skips: skipped {name}: {message}")
    errors = []
    if not cases:
        errors.append("the report has no test")
    for name, reason in EXPECTED.items():
        if name not in skipped:
            errors.append(f"expected skip missing: {name}")
        elif reason not in skipped[name]:
            errors.append(f"{name} skipped for another reason: {skipped[name]}")
    for name in sorted(set(skipped) - set(EXPECTED)):
        errors.append(f"unexpected skip: {name}")
    for e in errors:
        print(f"check-onnx-skips: ERROR {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
