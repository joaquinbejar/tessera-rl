# SPDX-License-Identifier: MIT OR Apache-2.0
"""Release-asset manifests of the P2-M3 loop (PR 3, design revision 2).

``pack`` turns each completed, ledger-settled phase attempt of a loop root
into one tarball and writes ``assets-manifest.json``. For each asset, the
manifest records the release tag, name, size, SHA-256, producing
generation, phase, attempt, source SHA and config hash. The coordinator
uploads the tarballs to the draft release named by the tag.

``verify`` checks a downloaded asset against the committed manifest and
refuses any substitution. A mismatching asset is a broken artifact, never
a silent re-upload.

The run host deletes a sole local copy only after ``verify`` passes on the
read-back of its upload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import loop as loopmod  # noqa: E402


def _tar(src: str, dst: str) -> None:
    """A tarball whose bytes depend on the content only (sorted names,
    zeroed times and owners)."""

    def clean(info):
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
        return info

    with tarfile.open(dst + ".tmp", "w", format=tarfile.PAX_FORMAT) as tar:
        for base, dirs, files in os.walk(src):
            dirs.sort()
            for name in sorted(files):
                if name.endswith(".tmp") or name.startswith("run-"):
                    continue
                full = os.path.join(base, name)
                tar.add(full, arcname=os.path.relpath(full, os.path.dirname(src)), filter=clean)
    os.replace(dst + ".tmp", dst)


def pack(root: str, out: str, tag: str) -> dict:
    lp = loopmod.Loop(root)
    entries = lp.ledger.entries()
    os.makedirs(out, exist_ok=True)
    assets = []
    for k in range(lp.cfg["k_max"]):
        for phase in loopmod.PHASES:
            done, _ = lp.phase_state(k, phase, entries)
            if done is None:
                continue
            name = f"g{k:03d}-{os.path.basename(done)}.tar"
            path = os.path.join(out, name)
            _tar(done, path)
            assets.append(
                {
                    "name": name,
                    "size": os.path.getsize(path),
                    "sha256": artifacts.sha256_file(path),
                    "generation": k,
                    "phase": phase,
                    "attempt": lp.attempt_id(k, phase, int(done.rsplit("-", 1)[1])),
                    "source_sha": lp.cfg["source_sha"],
                    "config_sha256": artifacts.sha256(artifacts.canonical_json(lp.cfg)),
                }
            )
    manifest = {"tag": tag, "assets": assets}
    artifacts.write_atomic(os.path.join(out, "assets-manifest.json"), artifacts.canonical_json(manifest))
    return manifest


def verify(manifest_path: str, asset: str) -> None:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    entry = next((a for a in manifest["assets"] if a["name"] == os.path.basename(asset)), None)
    if entry is None:
        raise artifacts.Incomplete(f"{asset} is not in the manifest")
    if os.path.getsize(asset) != entry["size"] or artifacts.sha256_file(asset) != entry["sha256"]:
        raise artifacts.Incomplete(f"{asset} does not match the manifest: a broken artifact, not a substitute")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("pack")
    a.add_argument("--root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tag", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("asset")
    args = p.parse_args(argv)
    if args.command == "pack":
        m = pack(args.root, args.out, args.tag)
        print(json.dumps({"assets": len(m["assets"])}))
        return 0
    verify(args.manifest, args.asset)
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
