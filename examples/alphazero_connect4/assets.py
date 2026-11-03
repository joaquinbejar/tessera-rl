# SPDX-License-Identifier: MIT OR Apache-2.0
"""Release assets of the P2-M3 loop (PR 3, design revision 2).

``pack`` turns each done phase attempt (one with an ``attempt.json``) into
a content-addressed tarball. It merges the entries into the root's
``assets-manifest.json``, keeping every prior entry: the manifest only
grows, and a changed entry for an existing name is refused.

An entry records:
- the release tag, name, size and SHA-256;
- the producing generation, phase and attempt;
- the command it ran;
- the source SHA and the config hash.

The coordinator uploads the tarballs to the draft release named by the
tag.

``evict`` removes an uploaded attempt's output, but only after the
read-back of its asset verifies against the manifest. It keeps the
attempt's descriptor and marks it ``evicted`` with its asset entry, so the
loop still resolves the attempt's identity. An evicted input that a later
phase needs stops the loop with ``incomplete input`` until ``restore``
brings it back. ``restore`` extracts a verified asset and clears the mark.

``pack``, ``evict`` and ``restore`` hold the loop's owner lock and refuse
a history whose recorded attempts lack their ok ledger settlement.

``verify`` refuses any substitute: a mismatching asset is a broken
artifact, never a silent re-upload.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import artifacts  # noqa: E402
import loop as loopmod  # noqa: E402

MANIFEST = "assets-manifest.json"


def _tar(src: str, dst: str) -> None:
    """A tarball whose bytes depend on the content only (sorted names,
    zeroed times and owners). The descriptor stays out: it is kept."""

    def clean(info):
        info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
        return info

    with tarfile.open(dst + ".tmp", "w", format=tarfile.PAX_FORMAT) as tar:
        for base, dirs, files in os.walk(src):
            dirs.sort()
            for name in sorted(files):
                if name.endswith(".tmp") or name.startswith("run-") or name == "attempt.json":
                    continue
                full = os.path.join(base, name)
                tar.add(full, arcname=os.path.relpath(full, src), filter=clean)
    os.replace(dst + ".tmp", dst)


def published_partial(d: str) -> bool:
    """A remote attempt that was killed or interrupted (remote.py) is never
    packed, whatever it left."""
    path = d + ".dispatch.json"
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return bool(json.load(f).get("published_partial"))


def load_manifest(root: str) -> dict:
    path = os.path.join(root, MANIFEST)
    if not os.path.exists(path):
        return {"tag": None, "assets": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pack(root: str, out: str, tag: str) -> dict:
    lp = loopmod.Loop(root)
    with _owned(lp):
        return _pack(lp, root, out, tag)


def _owned(lp):
    """Every operation that reads or changes the root holds the loop's
    owner lock, so it never races a running loop."""
    return contextlib.closing(lp.owner())


def _verified(lp) -> list[dict]:
    """The settled, verified history: every recorded attempt resolves
    against an ok ledger entry, or this refuses."""
    entries = lp.ledger.entries()
    try:
        lp.chain(entries)
        for gen in sorted(os.listdir(lp.root)):
            if gen.startswith("gen-"):
                for phase in loopmod.PHASES:
                    lp.resolve(int(gen[4:]), phase, entries)
    except loopmod.Stop as s:
        raise SystemExit(f"broken history: {s.reason} {json.dumps(s.inputs, sort_keys=True)}") from None
    return entries


def _pack(lp, root: str, out: str, tag: str) -> dict:
    _verified(lp)
    manifest = load_manifest(root)
    if manifest["tag"] not in (None, tag):
        raise SystemExit(f"the root's assets belong to release {manifest['tag']}, not {tag}")
    known = {a["name"]: a for a in manifest["assets"]}
    os.makedirs(out, exist_ok=True)
    for gen in sorted(os.listdir(root)):
        if not gen.startswith("gen-"):
            continue
        for attempt in sorted(os.listdir(os.path.join(root, gen))):
            d = os.path.join(root, gen, attempt)
            desc = lp.descriptor(d)
            if desc is None or desc.get("evicted") or "ref" in desc or published_partial(d):
                continue
            name = f"{desc['attempt']}.tar"
            path = os.path.join(out, name)
            _tar(d, path)
            entry = {
                "name": name,
                "size": os.path.getsize(path),
                "sha256": artifacts.sha256_file(path),
                "generation": desc["generation"],
                "phase": desc["phase"],
                "attempt": desc["attempt"],
                "command": desc.get("command"),
                "source_sha": lp.cfg["source_sha"],
                "config_sha256": artifacts.sha256(artifacts.canonical_json(lp.cfg)),
            }
            if name in known and known[name] != entry:
                raise SystemExit(f"{name} changed since it was packed: a broken artifact")
            known[name] = entry
    manifest = {"tag": tag, "assets": [known[n] for n in sorted(known)]}
    artifacts.write_atomic(os.path.join(root, MANIFEST), artifacts.canonical_json(manifest))
    shutil.copyfile(os.path.join(root, MANIFEST), os.path.join(out, MANIFEST))
    return manifest


def verify(manifest: dict, asset: str) -> dict:
    entry = next((a for a in manifest["assets"] if a["name"] == os.path.basename(asset)), None)
    if entry is None:
        raise artifacts.Incomplete(f"{asset} is not in the manifest")
    if os.path.getsize(asset) != entry["size"] or artifacts.sha256_file(asset) != entry["sha256"]:
        raise artifacts.Incomplete(f"{asset} does not match the manifest: a broken artifact, not a substitute")
    return entry


def attempt_dir(root: str, attempt: str) -> str:
    gen, phase, n = attempt.split("-")
    return os.path.join(root, f"gen-{gen[1:]}", f"{phase}-{n}")


def evict(root: str, readback: str) -> None:
    """Removes an attempt's output after its asset's read-back verifies.
    The mark is written first, so an interrupted eviction resumes by
    running it again, and restore can always bring the output back."""
    lp = loopmod.Loop(root)
    with _owned(lp):
        entry = verify(load_manifest(root), readback)
        _verified(lp)
        d = attempt_dir(root, entry["attempt"])
        desc = lp.descriptor(d)
        if desc is None:
            raise SystemExit(f"{entry['attempt']} has no descriptor")
        artifacts.write_atomic(os.path.join(d, "attempt.json"), artifacts.canonical_json({**desc, "evicted": entry}))
        for name in os.listdir(d):
            if name != "attempt.json":
                p = os.path.join(d, name)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


def restore(root: str, asset: str) -> None:
    """Extracts a verified asset into its attempt, then clears the mark;
    an interrupted restore resumes by running it again."""
    lp = loopmod.Loop(root)
    with _owned(lp):
        entry = verify(load_manifest(root), asset)
        d = attempt_dir(root, entry["attempt"])
        desc = lp.descriptor(d)
        if desc is None:
            raise SystemExit(f"{entry['attempt']} has no descriptor")
        with tarfile.open(asset) as tar:
            tar.extractall(d, filter="data")
        desc.pop("evicted", None)
        # The restored output must match the recorded identity before the
        # mark goes.
        if lp.identity(desc["phase"], d) != desc["identity"]:
            raise SystemExit(f"{entry['attempt']}: the restored output differs from its identity")
        artifacts.write_atomic(os.path.join(d, "attempt.json"), artifacts.canonical_json(desc))
        _verified(lp)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("pack")
    a.add_argument("--root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tag", required=True)
    for name in ("verify", "evict", "restore"):
        c = sub.add_parser(name)
        c.add_argument("--root", required=True)
        c.add_argument("asset")
    args = p.parse_args(argv)
    if args.command == "pack":
        m = pack(args.root, args.out, args.tag)
        print(json.dumps({"assets": len(m["assets"])}))
    elif args.command == "verify":
        verify(load_manifest(args.root), args.asset)
        print("ok")
    elif args.command == "evict":
        evict(args.root, args.asset)
        print("evicted")
    else:
        restore(args.root, args.asset)
        print("restored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
