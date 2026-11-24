#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Checks that PyPI holds PEP 740 attestations for every uploaded file (#350).

    pypi_attestations.py --version 0.4.0 --dist dist/ [--timeout 300]

For each file in <dist> (the wheels the publish job uploaded), PyPI's
Integrity API (`/integrity/<project>/<version>/<file>/provenance`) must
return at least one attestation bundle from GitHub, repository
joaquinbejar/tessera-rl, workflow pypi.yml, with at least one attestation.
PyPI may index a fresh upload late, so missing answers are polled until
the timeout. Exit 0 if every file passes, 1 otherwise (it fails closed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

PROJECT = "tessera-rl"
REPOSITORY = "joaquinbejar/tessera-rl"
WORKFLOW = "pypi.yml"
URL = "https://pypi.org/integrity/{project}/{version}/{file}/provenance"

Fetch = Callable[[str], "tuple[int, bytes]"]


def judge(status: int, body: bytes) -> str | None:
    """None if the response proves an attestation from this workflow, else why not."""
    if status != 200:
        return f"HTTP {status}"
    try:
        data = json.loads(body)
    except ValueError:
        return "not JSON"
    bundles = data.get("attestation_bundles") if isinstance(data, dict) else None
    if not isinstance(bundles, list) or not bundles:
        return "no attestation bundles"
    for b in bundles:
        pub = b.get("publisher") or {}
        if (
            pub.get("kind") == "GitHub"
            and pub.get("repository") == REPOSITORY
            and pub.get("workflow") == WORKFLOW
            and b.get("attestations")
        ):
            return None
    return f"no bundle from GitHub {REPOSITORY} {WORKFLOW} with attestations"


def check(
    files: list[str],
    version: str,
    fetch: Fetch,
    timeout: float,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    interval: float = 15.0,
) -> dict[str, str]:
    """The files still failing at the deadline, with the last reason."""
    deadline = now() + timeout
    pending = {f: "not checked" for f in files}
    while True:
        for f in list(pending):
            status, body = fetch(URL.format(project=PROJECT, version=version, file=f))
            why = judge(status, body)
            if why is None:
                del pending[f]
            else:
                pending[f] = why
        if not pending or now() >= deadline:
            return pending
        sleep(interval)


def _fetch(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.pypi.integrity.v1+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (fixed https URL)
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except urllib.error.URLError:
        return 0, b""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    files = sorted(p.name for p in args.dist.glob("*.whl"))
    if not files:
        print("pypi_attestations: no wheels", file=sys.stderr)
        return 1
    failing = check(files, args.version, _fetch, args.timeout)
    for f, why in sorted(failing.items()):
        print(f"pypi_attestations: {f}: {why}", file=sys.stderr)
    if not failing:
        print(f"pypi_attestations: {len(files)} files attested by {REPOSITORY} {WORKFLOW}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
