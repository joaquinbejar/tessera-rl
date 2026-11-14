# SPDX-License-Identifier: MIT OR Apache-2.0
"""The PyPI workflow: trusted publishing of one wheels run's three wheels,
checked against that run and the release tag, on dispatch only, never from a
fork, with the OIDC token as the only credential of the upload job."""

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PYPI = (WORKFLOWS / "pypi.yml").read_text()
WHEELS = (WORKFLOWS / "wheels.yml").read_text()
VERIFY = ROOT / "scripts" / "pypi_verify.py"
GUARD = "if: github.event_name == 'workflow_dispatch' && github.repository == 'joaquinbejar/tessera-rl'"
COMMIT = "c" * 40
TAG = "v0.2.0"


def _job(text, name):
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))[name]


def _jobs(text):
    body = text.split("\njobs:\n", 1)[1]
    return re.findall(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.M)


def _step(job, name):
    """The run block of the step named `name`, dedented."""
    m = re.search(rf"^      - (?:id: \w+\n        )?name: {re.escape(name)}\n(.*?)(?=^      - |\Z)", job, re.M | re.S)
    assert m, name
    block = m.group(1)
    if "run: |\n" in block:
        lines = block.split("run: |\n", 1)[1].splitlines()
        return "\n".join(l[10:] for l in lines) + "\n"
    return _one(r"^        run: (.*)$", block) + "\n"


def _one(pattern, text):
    found = re.findall(pattern, text, re.M)
    assert len(found) == 1, (pattern, found)
    return found[0]


VERIFY_JOB = _job(PYPI, "verify")
PUBLISH_JOB = _job(PYPI, "publish")


# --- static: triggers, guard, permissions, pins ---------------------------------


def test_pypi_is_dispatch_only_with_the_run_and_the_tag():
    on = PYPI.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert re.findall(r"^  ([a-z_]+):", on, re.M) == ["workflow_dispatch"]
    assert re.findall(r"^      ([a-z_]+):$", on, re.M) == ["wheels_run_id", "tag"]
    assert on.count("required: true") == 2


def test_pypi_has_two_jobs_both_fork_guarded():
    assert _jobs(PYPI) == ["verify", "publish"]
    for job in (VERIFY_JOB, PUBLISH_JOB):
        assert _one(r"^    (if: .*)$", job) == GUARD


def test_pypi_jobs_run_on_hosted_runners_only():
    for job in (VERIFY_JOB, PUBLISH_JOB):
        assert _one(r"^    runs-on: (.*)$", job) == "ubuntu-24.04"
        assert "self-hosted" not in job
        assert re.search(r"^    timeout-minutes: \d+$", job, re.M)


def test_pypi_default_permissions_are_none():
    assert re.search(r"^permissions: \{\}$", PYPI, re.M)


def test_pypi_verify_reads_only():
    perms = VERIFY_JOB.split("    permissions:\n", 1)[1].split("\n    defaults:", 1)[0]
    assert sorted(re.findall(r"^      ([a-z-]+): (\w+)$", perms, re.M)) == [("actions", "read"), ("contents", "read")]
    assert "id-token" not in VERIFY_JOB
    assert "environment:" not in VERIFY_JOB


def test_pypi_publish_has_only_the_oidc_token_in_the_pypi_environment():
    perms = PUBLISH_JOB.split("    permissions:\n", 1)[1].split("\n    defaults:", 1)[0]
    assert re.findall(r"^      ([a-z-]+): (\w+)$", perms, re.M) == [("id-token", "write")]
    env = PUBLISH_JOB.split("    environment:\n", 1)[1]
    assert _one(r"^      name: (\S+)$", env.split("\n    permissions:", 1)[0]) == "pypi"
    assert _one(r"^    needs: (\S+)$", PUBLISH_JOB) == "verify"
    assert PYPI.count("id-token") == 1


def test_pypi_publish_uses_no_token_and_checks_out_nothing():
    assert "password" not in PYPI
    assert "PYPI_" not in PYPI and "secrets." not in PYPI
    assert "actions/checkout" not in PUBLISH_JOB
    assert "github.token" not in PUBLISH_JOB


def test_pypi_action_is_pinned_by_full_commit():
    uses = re.findall(r"uses: (\S+)", PYPI)
    assert uses, uses
    for u in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u), u
    assert _one(r"uses: (pypa/gh-action-pypi-publish@\S+) # v1\.14\.3$", PYPI) == (
        "pypa/gh-action-pypi-publish@7245906fb61fa33a13554f9d5fb856fe61758086"
    )


def test_pypi_actions_are_the_pins_the_other_workflows_use():
    others = "".join(p.read_text() for p in WORKFLOWS.glob("*.yml") if p.name != "pypi.yml")
    for u in re.findall(r"uses: (actions/\S+ # \S+)", PYPI):
        assert u in others, u


def test_pypi_publish_uploads_the_verify_jobs_artifact_of_this_run_only():
    up = _one(r"^          name: (pypi-dist-.+)$", VERIFY_JOB)
    down = _one(r"^          name: (pypi-dist-.+)$", PUBLISH_JOB)
    assert up == down == "pypi-dist-${{ github.run_id }}"
    assert "run-id" not in PUBLISH_JOB
    assert _one(r"^          packages-dir: (\S+)$", PUBLISH_JOB) == "dist/"


def test_pypi_no_run_block_interpolates_an_expression():
    for block in re.findall(r"run: \|\n((?:          .*\n|\n)+)", PYPI) + re.findall(r"run: (?!\|)(.*)", PYPI):
        assert "${{" not in block, block


def test_pypi_takes_the_wheels_from_a_green_dispatched_wheels_run_at_the_tag():
    block = _step(VERIFY_JOB, "The wheels run and the tag")
    assert "[ \"$path\" = .github/workflows/wheels.yml ]" in block
    assert "[ \"$event\" = workflow_dispatch ]" in block
    assert "[ \"$conclusion\" = success ]" in block
    assert "[ \"$head\" = \"$sha\" ]" in block
    assert "git/ref/tags/$TAG" in block


def test_pypi_artifact_names_are_those_wheels_yml_uploads():
    src = VERIFY.read_text()
    assert "name: wheel-aarch64-unknown-linux-gnu\n" in WHEELS
    assert "name: wheel-aarch64-apple-darwin\n" in WHEELS
    assert 'echo "name=wheel-$TARGET-$PROFILE-$COMMIT-$sha"' in WHEELS
    assert '"wheel-aarch64-unknown-linux-gnu"' in src and '"wheel-aarch64-apple-darwin"' in src
    assert "wheel-x86_64-unknown-linux-gnu-wheel-" in src
    # Each wheels artifact carries the checks.json the script reads.
    assert WHEELS.count("            checks.json\n") == 3


def test_pypi_reads_the_layout_wheels_yml_uploads():
    # upload-artifact keeps paths relative to the paths' common root: with
    # `dist/*.whl` and `checks.json` side by side, the artifact holds
    # dist/<wheel> next to checks.json (run 43594717096 failed on a flat layout).
    uploads = re.findall(r"actions/upload-artifact@\S+.*\n(?:        .*\n)+", WHEELS)
    assert len(uploads) == 3
    for u in uploads:
        paths = u.split("          path: |\n", 1)[1]
        assert "            dist/*.whl\n" in paths and "            checks.json\n" in paths, u
    src = VERIFY.read_text()
    assert 'd.glob("dist/*.whl")' in src
    assert '(d / "checks.json")' in src


def test_pypi_publish_downloads_the_flat_directory_verify_wrote():
    # verify writes the wheels and SHA256SUMS flat into $RUNNER_TEMP/dist and
    # uploads that one directory, so its files are at the artifact's root;
    # publish downloads them into dist/, where the sums step and the upload read.
    verify_run = _step(VERIFY_JOB, "Check the wheels against the run and the tag")
    assert verify_run.rstrip().endswith('"$RUNNER_TEMP/dist"')
    assert _one(r"^          path: (.*)$", VERIFY_JOB) == "${{ runner.temp }}/dist"
    assert _one(r"^          path: (.*)$", PUBLISH_JOB) == "dist"
    assert _step(PUBLISH_JOB, "The verified wheels").startswith("cd dist\n")
    assert _one(r"^          packages-dir: (\S+)$", PUBLISH_JOB) == "dist/"


# A copy of the artifact tree of wheels run 43592025059 (v0.2.0 at c156aa0), as
# `gh run download` laid it out: the real artifact names and paths, checks.json
# and compare.json verbatim, the wheels stubbed to their real METADATA and WHEEL
# files, and the other reports stubbed to `{}`. The stub wheels' sha256 is not
# the real one, so checks.json and the W1 name disagree with them on purpose.
RUN_TREE = Path(__file__).resolve().parent / "fixtures" / "pypi" / "run-43592025059"
RUN_COMMIT = "c156aa0b9e70b2b3f6232882dc7200169082c6ef"
RUN_W1 = f"wheel-x86_64-unknown-linux-gnu-wheel-{RUN_COMMIT}-1d614867bb51b2a9588811603f7647fca86351979a0452a9ceb6e8aa1a84f105"
RUN_WHEELS = {
    RUN_W1: "tessera_rl-0.2.0-cp311-abi3-manylinux_2_28_x86_64.whl",
    "wheel-aarch64-unknown-linux-gnu": "tessera_rl-0.2.0-cp311-abi3-manylinux_2_28_aarch64.whl",
    "wheel-aarch64-apple-darwin": "tessera_rl-0.2.0-cp311-abi3-macosx_13_0_arm64.whl",
}


def _run_tree(tmp_path) -> Path:
    arts = tmp_path / "artifacts"
    shutil.copytree(RUN_TREE, arts)
    return arts


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pypi_run_tree_fixture_is_the_real_layout():
    assert sorted(p.name for p in RUN_TREE.iterdir()) == sorted(RUN_WHEELS)
    for name, wheel in RUN_WHEELS.items():
        d = RUN_TREE / name
        assert [p.relative_to(d).as_posix() for p in d.rglob("*.whl")] == [f"dist/{wheel}"]
        assert (d / "checks.json").is_file() and (d / "golden-report.json").is_file()
    assert (RUN_TREE / RUN_W1 / "compare.json").is_file()


def test_pypi_verify_reads_the_real_run_tree_up_to_the_stub_bytes(tmp_path):
    # Every check but the byte identity passes on the real names and paths: the
    # only problems are the stubs' sha256 against checks.json and the W1 name.
    p, out = _verify(_run_tree(tmp_path), commit=RUN_COMMIT)
    assert p.returncode == 1
    problems = [l for l in p.stderr.splitlines() if l.startswith("pypi_verify: ")]
    assert len(problems) == 4, p.stderr
    assert sum("!= checks.json" in l for l in problems) == 3, p.stderr
    assert sum("!= the artifact name's 1d614867" in l for l in problems) == 1, p.stderr
    assert not out.exists()


def _reseal(arts: Path) -> Path:
    """Points checks.json and the W1 name at the stub wheels' own sha256."""
    for name, wheel in RUN_WHEELS.items():
        d = arts / name
        sha = _sha(d / "dist" / wheel)
        checks = d / "checks.json"
        data = json.loads(checks.read_text())
        data["wheel_sha256"] = sha
        checks.write_text(json.dumps(data, indent=1))
        if name == RUN_W1:
            d.rename(arts / (name.rsplit("-", 1)[0] + "-" + sha))
    return arts


def test_pypi_verify_accepts_the_real_run_tree_resealed(tmp_path):
    arts = _reseal(_run_tree(tmp_path))
    p, out = _verify(arts, commit=RUN_COMMIT)
    assert p.returncode == 0, p.stderr
    assert sorted(x.name for x in out.iterdir()) == sorted(["SHA256SUMS", *RUN_WHEELS.values()])
    sums = dict(reversed(l.split("  ")) for l in (out / "SHA256SUMS").read_text().splitlines())
    assert sums == {w: _sha(out / w) for w in RUN_WHEELS.values()}


@pytest.mark.parametrize("tag, commit, message", [("v0.2.1", RUN_COMMIT, "does not match"), (TAG, "d" * 40, "built at " + RUN_COMMIT)])
def test_pypi_verify_refuses_the_real_run_tree_for_another_release(tmp_path, tag, commit, message):
    p, out = _verify(_reseal(_run_tree(tmp_path)), tag=tag, commit=commit)
    assert p.returncode == 1 and message in p.stderr, p.stderr
    assert not out.exists()


def test_pypi_verify_refuses_the_real_run_tree_flattened(tmp_path):
    # The layout the first version assumed: each wheel at its artifact's root.
    arts = _reseal(_run_tree(tmp_path))
    for whl in list(arts.glob("*/dist/*.whl")):
        whl.rename(whl.parent.parent / whl.name)
    p, out = _verify(arts, commit=RUN_COMMIT)
    assert p.returncode == 1 and p.stderr.count("expected one wheel in dist/") == 3, p.stderr
    assert not out.exists()


REAL = os.environ.get("TESSERA_PYPI_ARTIFACTS")


@pytest.mark.skipif(not REAL, reason="set TESSERA_PYPI_ARTIFACTS to a downloaded wheels run (gh run download)")
def test_pypi_verify_accepts_a_real_wheels_run(tmp_path):
    arts = Path(REAL)
    tag = os.environ.get("TESSERA_PYPI_TAG", TAG)
    w1 = [d.name for d in arts.iterdir() if d.is_dir() and d.name.startswith("wheel-x86_64-")]
    assert len(w1) == 1, w1
    commit = w1[0].split("-wheel-", 1)[1].split("-")[0]
    p, out = _verify(arts, tag=tag, commit=commit, out=tmp_path / "dist")
    assert p.returncode == 0, p.stdout + p.stderr
    assert len((out / "SHA256SUMS").read_text().splitlines()) == 3


# --- the verify script ---------------------------------------------------------

PLAT = {
    "x86": "manylinux_2_28_x86_64",
    "aarch64-unknown-linux-gnu": "manylinux_2_28_aarch64",
    "aarch64-apple-darwin": "macosx_11_0_arm64",
}


def _wheel(path: Path, version="0.2.0", meta_version=None, name="tessera-rl"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tessera_rl/__init__.py", f"# {path.name}\n")
        z.writestr(
            f"tessera_rl-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {meta_version or version}\n\nbody\nVersion: 9.9.9\n",
        )
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _artifact(root: Path, key, version="0.2.0", commit=COMMIT, checks=None, sha_override=None, **kw):
    plat = PLAT[key]
    tmp = root / f".tmp-{key}"
    (tmp / "dist").mkdir(parents=True)
    # The layout wheels.yml uploads: the wheel under dist/, the reports at the root.
    whl = tmp / "dist" / f"tessera_rl-{version}-cp311-abi3-{plat}.whl"
    sha = _wheel(whl, version=version, **kw)
    name = f"wheel-x86_64-unknown-linux-gnu-wheel-{commit}-{sha_override or sha}" if key == "x86" else f"wheel-{key}"
    d = root / name
    tmp.rename(d)
    statuses = checks if checks is not None else {"golden_anchor": 0, "python_suite": 0, "num_tests": 0, "onnx_smoke": 0}
    (d / "checks.json").write_text(json.dumps({"wheel_sha256": sha, "checks": statuses}))
    return d


def _good(root: Path):
    root.mkdir()
    return [_artifact(root, k) for k in PLAT]


def _verify(artifacts, tag=TAG, commit=COMMIT, out=None):
    out = out or artifacts.parent / "dist"
    p = subprocess.run(
        [sys.executable, "-I", str(VERIFY), str(artifacts), tag, commit, str(out)],
        capture_output=True,
        text=True,
    )
    return p, out


def test_pypi_verify_accepts_the_three_wheels_and_writes_sums(tmp_path):
    arts = tmp_path / "a"
    _good(arts)
    p, out = _verify(arts)
    assert p.returncode == 0, p.stderr
    names = sorted(x.name for x in out.iterdir())
    assert names == sorted(["SHA256SUMS"] + [f"tessera_rl-0.2.0-cp311-abi3-{v}.whl" for v in PLAT.values()])
    check = subprocess.run(["shasum", "-a", "256", "-c", "SHA256SUMS"] if shutil.which("shasum") else ["sha256sum", "-c", "SHA256SUMS"], cwd=out, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    assert len((out / "SHA256SUMS").read_text().splitlines()) == 3


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: shutil.rmtree(a / "wheel-aarch64-apple-darwin"), "missing artifact wheel-aarch64-apple-darwin"),
        (lambda a: (a / "wheel-extra").mkdir(), "unexpected artifacts ['wheel-extra']"),
        (lambda a: _artifact(a, "x86", commit="d" * 40), "expected one W1 artifact"),
    ],
)
def test_pypi_verify_refuses_a_run_without_exactly_the_three_artifacts(tmp_path, mutate, message):
    arts = tmp_path / "a"
    _good(arts)
    mutate(arts)
    p, out = _verify(arts)
    assert p.returncode == 1 and message in p.stderr, p.stderr
    assert not out.exists()


def _replace(arts, key, **kw):
    for d in arts.iterdir():
        if (key == "x86" and d.name.startswith("wheel-x86_64")) or d.name == f"wheel-{key}":
            shutil.rmtree(d)
    return _artifact(arts, key, **kw)


@pytest.mark.parametrize(
    "key, kw, message",
    [
        ("aarch64-unknown-linux-gnu", {"version": "0.1.0"}, "does not match"),
        ("aarch64-apple-darwin", {"meta_version": "0.1.9"}, "METADATA says"),
        ("aarch64-apple-darwin", {"name": "tessera-other"}, "METADATA says"),
        ("aarch64-unknown-linux-gnu", {"checks": {"python_suite": 1, "golden_anchor": 0}}, "failed checks {'python_suite': 1}"),
        ("aarch64-unknown-linux-gnu", {"checks": {}}, "checks.json has no checks"),
        ("x86", {"commit": "d" * 40}, "not " + COMMIT),
        ("x86", {"sha_override": "e" * 64}, "the artifact name's"),
    ],
)
def test_pypi_verify_refuses_a_wheel_that_does_not_match(tmp_path, key, kw, message):
    arts = tmp_path / "a"
    _good(arts)
    _replace(arts, key, **kw)
    p, out = _verify(arts)
    assert p.returncode == 1 and message in p.stderr, p.stderr
    assert not out.exists()


def test_pypi_verify_refuses_a_wheel_changed_after_its_checks(tmp_path):
    arts = tmp_path / "a"
    _good(arts)
    whl = next((arts / "wheel-aarch64-apple-darwin").glob("dist/*.whl"))
    with zipfile.ZipFile(whl, "a") as z:
        z.writestr("tessera_rl/extra.py", "x = 1\n")
    p, out = _verify(arts)
    assert p.returncode == 1 and "!= checks.json" in p.stderr, p.stderr
    assert not out.exists()


def test_pypi_verify_refuses_two_wheels_in_one_artifact(tmp_path):
    arts = tmp_path / "a"
    _good(arts)
    d = arts / "wheel-aarch64-apple-darwin"
    _wheel(d / "dist" / "tessera_rl-0.2.0-cp311-abi3-macosx_12_0_arm64.whl")
    p, _ = _verify(arts)
    assert p.returncode == 1 and "expected one wheel in dist/" in p.stderr, p.stderr


@pytest.mark.parametrize("where", [".", "other"])
def test_pypi_verify_refuses_a_wheel_outside_dist(tmp_path, where):
    arts = tmp_path / "a"
    _good(arts)
    d = arts / "wheel-aarch64-apple-darwin"
    whl = next(d.glob("dist/*.whl"))
    (d / where).mkdir(exist_ok=True)
    if where == ".":
        # The flat layout the first version expected: only the wheel moved.
        whl.rename(d / whl.name)
    else:
        shutil.copyfile(whl, d / where / whl.name)
    p, out = _verify(arts)
    assert p.returncode == 1 and "expected one wheel in dist/" in p.stderr, p.stderr
    assert not out.exists()


def test_pypi_verify_refuses_a_wheel_of_the_wrong_platform(tmp_path):
    arts = tmp_path / "a"
    _good(arts)
    d = arts / "wheel-aarch64-unknown-linux-gnu"
    whl = next(d.glob("dist/*.whl"))
    wrong = whl.with_name(whl.name.replace("aarch64", "x86_64"))
    whl.rename(wrong)
    p, _ = _verify(arts)
    assert p.returncode == 1 and "does not match" in p.stderr, p.stderr


@pytest.mark.parametrize(
    "tag, commit",
    [("0.2.0", COMMIT), ("v0.2", COMMIT), ("v0.2.0-rc1", COMMIT), ("v01.2.0", COMMIT), (TAG, "c" * 39), (TAG, "C" * 40)],
)
def test_pypi_verify_refuses_bad_arguments(tmp_path, tag, commit):
    arts = tmp_path / "a"
    _good(arts)
    p, out = _verify(arts, tag=tag, commit=commit)
    assert p.returncode == 2, p.stderr
    assert not out.exists()


def test_pypi_verify_refuses_an_existing_output_directory(tmp_path):
    arts = tmp_path / "a"
    _good(arts)
    (tmp_path / "dist").mkdir()
    p, _ = _verify(arts)
    assert p.returncode == 2


def test_pypi_verify_refuses_a_version_other_than_the_tag(tmp_path):
    arts = tmp_path / "a"
    _good(arts)
    p, _ = _verify(arts, tag="v0.2.1")
    assert p.returncode == 1 and p.stderr.count("does not match") == 3, p.stderr


# --- the run blocks, executed under bash -eo pipefail with a fake gh -------------

FAKE_GH = r"""#!/bin/sh
case "$*" in
  "api repos/o/r/git/ref/tags/v0.2.0 --jq"*) echo "$FAKE_REF" ;;
  "api repos/o/r/git/tags/"*) echo "$FAKE_TAGOBJ" ;;
  "api repos/o/r/actions/runs/"*) echo "$FAKE_RUN" ;;
  *) echo "fake gh: unexpected $*" >&2; exit 1 ;;
esac
"""


def _run_head_step(tmp_path, ref="tag " + "1" * 40, tagobj="commit " + COMMIT, run=None, run_id="43481908321", tag=TAG):
    if run is None:
        run = f".github/workflows/wheels.yml workflow_dispatch completed success {COMMIT} o/r"
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    gh = bin_ / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(gh.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "out"
    out.write_text("")
    env = {
        "PATH": f"{bin_}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(out),
        "GH_TOKEN": "x",
        "REPO": "o/r",
        "RUN_ID": run_id,
        "TAG": tag,
        "FAKE_REF": ref,
        "FAKE_TAGOBJ": tagobj,
        "FAKE_RUN": run,
    }
    script = tmp_path / "step.sh"
    script.write_text(_step(VERIFY_JOB, "The wheels run and the tag"))
    p = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)], env=env, capture_output=True, text=True)
    return p, out.read_text()


bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@bash
def test_pypi_head_step_accepts_an_annotated_tag_and_its_green_run(tmp_path):
    p, out = _run_head_step(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert out == f"commit={COMMIT}\n"


@bash
def test_pypi_head_step_accepts_a_lightweight_tag(tmp_path):
    p, out = _run_head_step(tmp_path, ref="commit " + COMMIT, tagobj="unused")
    assert p.returncode == 0, p.stdout + p.stderr
    assert out == f"commit={COMMIT}\n"


@bash
@pytest.mark.parametrize(
    "kw, message",
    [
        ({"run_id": "0"}, "bad run id"),
        ({"run_id": "12; echo"}, "bad run id"),
        ({"tag": "v0.2.0 "}, "bad tag"),
        ({"tag": "main"}, "bad tag"),
        ({"ref": ""}, "not a commit"),
        ({"ref": "tree " + COMMIT}, "names a tree"),
        ({"run": f".github/workflows/ci.yml workflow_dispatch completed success {COMMIT} o/r"}, "not a wheels.yml run"),
        ({"run": f".github/workflows/wheels.yml push completed success {COMMIT} o/r"}, "was not dispatched"),
        ({"run": f".github/workflows/wheels.yml workflow_dispatch completed failure {COMMIT} o/r"}, "is not green"),
        ({"run": f".github/workflows/wheels.yml workflow_dispatch in_progress  {COMMIT} o/r"}, "is not green"),
        ({"run": f".github/workflows/wheels.yml workflow_dispatch completed success {COMMIT} fork/r"}, "built fork/r"),
        ({"run": f".github/workflows/wheels.yml workflow_dispatch completed success {'d' * 40} o/r"}, "tag v0.2.0 is"),
        ({"run": ""}, "not a wheels.yml run"),
    ],
)
def test_pypi_head_step_refuses(tmp_path, kw, message):
    p, out = _run_head_step(tmp_path, **kw)
    assert p.returncode != 0
    assert message in p.stdout + p.stderr, p.stdout + p.stderr
    assert out == ""


def test_pypi_publish_checks_the_sums_before_the_upload():
    steps = PUBLISH_JOB.split("    steps:\n", 1)[1]
    block = _step(PUBLISH_JOB, "The verified wheels")
    assert block.startswith("cd dist\nsha256sum -c SHA256SUMS\n"), block
    assert steps.index("sha256sum -c SHA256SUMS") < steps.index("pypa/gh-action-pypi-publish")


def _sha256sum_on_path(bin_: Path):
    """sha256sum where the host has one; on macOS a shim over `shasum -a 256`."""
    if shutil.which("sha256sum"):
        return
    if not shutil.which("shasum"):
        pytest.skip("needs sha256sum or shasum")
    shim = bin_ / "sha256sum"
    shim.write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)


@bash
@pytest.mark.parametrize("change", [None, "extra", "changed", "missing"])
def test_pypi_publish_sums_step_passes_only_the_three_checked_wheels(tmp_path, change):
    arts = tmp_path / "a"
    _good(arts)
    dist = tmp_path / "w" / "dist"
    p, _ = _verify(arts, out=dist)
    assert p.returncode == 0, p.stderr
    script = tmp_path / "sums.sh"
    script.write_text(_step(PUBLISH_JOB, "The verified wheels"))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _sha256sum_on_path(bin_)
    wheels = sorted(dist.glob("*.whl"))
    if change == "extra":
        (dist / "extra.whl").write_text("x")
    elif change == "changed":
        with wheels[0].open("ab") as f:
            f.write(b"y")
    elif change == "missing":
        wheels[1].unlink()
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path)}
    r = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)], cwd=tmp_path / "w", env=env, capture_output=True, text=True)
    if change is None:
        assert r.returncode == 0, r.stdout + r.stderr
        assert sorted(x.name for x in dist.iterdir()) == sorted(f"tessera_rl-0.2.0-cp311-abi3-{v}.whl" for v in PLAT.values())
        assert (tmp_path / "SHA256SUMS").is_file()
    else:
        assert r.returncode != 0, r.stdout + r.stderr
        assert (dist / "SHA256SUMS").is_file()


@bash
def test_pypi_publish_sums_step_on_the_real_run_tree(tmp_path):
    dist = tmp_path / "w" / "dist"
    p, _ = _verify(_reseal(_run_tree(tmp_path)), commit=RUN_COMMIT, out=dist)
    assert p.returncode == 0, p.stderr
    script = tmp_path / "sums.sh"
    script.write_text(_step(PUBLISH_JOB, "The verified wheels"))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _sha256sum_on_path(bin_)
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "RUNNER_TEMP": str(tmp_path)}
    r = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)], cwd=tmp_path / "w", env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(x.name for x in dist.iterdir()) == sorted(RUN_WHEELS.values())
