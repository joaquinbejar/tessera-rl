# SPDX-License-Identifier: MIT OR Apache-2.0
"""The Python package's public-API snapshot (#690): `scripts/check_py_api.py`
against the committed `api/tessera-rl-py.api`, mutants on copies of the
stubs that must change it (an added, a removed or a changed public item, an
adapter function) or must not (private names, a class moved between private
modules), the `__all__` and stub consistency check in both directions, and
the Python version guard."""

import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_py_api.py"


def run(root, mode="check"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), mode, "--root", str(root)], capture_output=True, text=True, cwd=ROOT
    )


def tree(tmp_path):
    """A copy of the package sources and the snapshot to mutate."""
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "python" / "tessera_rl", root / "python" / "tessera_rl", ignore=shutil.ignore_patterns("*.so", "*.pyd", "__pycache__"))
    (root / "api").mkdir()
    shutil.copy(ROOT / "api" / "tessera-rl-py.api", root / "api" / "tessera-rl-py.api")
    return root


def edit(path, old, new):
    text = path.read_text()
    assert text.count(old) >= 1, old
    path.write_text(text.replace(old, new, 1))


def diff_lines(out):
    return {line for line in out.splitlines() if re.match(r"^[+-][a-z]", line)}


def pkg(root):
    return root / "python" / "tessera_rl"


def test_py_api_snapshot_is_current():
    r = run(ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_py_api_header_records_python_modules_and_stubtest():
    head = [line for line in (ROOT / "api" / "tessera-rl-py.api").read_text().splitlines() if line.startswith("#")]
    text = "\n".join(head)
    assert "# Python: 3.11" in text
    assert "# modules: tessera_rl (__all__), gym, pz, puffer, replay, render" in text
    assert "EXPERIMENTAL_MODULES" in text
    assert "stubtest" in text


def test_py_api_added_public_name_is_caught(tmp_path):
    root = tree(tmp_path)
    edit(pkg(root) / "__init__.py", '__all__ = [\n', '__all__ = [\n    "Probe",\n')
    edit(pkg(root) / "__init__.pyi", "__all__ = [\n", 'class Probe: ...\n\n__all__ = [\n    "Probe",\n')
    r = run(root)
    assert r.returncode == 1, r.stdout
    assert "+class tessera_rl.Probe()" in diff_lines(r.stdout)


def test_py_api_removed_public_name_is_caught(tmp_path):
    root = tree(tmp_path)
    edit(pkg(root) / "__init__.py", '    "Archive",\n', "")
    edit(pkg(root) / "__init__.pyi", '    "Archive",\n', "")
    edit(pkg(root) / "__init__.pyi", "from tessera_rl._archive import Archive as Archive\n", "")
    r = run(root)
    assert r.returncode == 1, r.stdout
    removed = {line for line in diff_lines(r.stdout) if line.startswith("-")}
    assert any(line.startswith("-class tessera_rl.Archive") for line in removed), removed
    assert all("tessera_rl.Archive" in line for line in removed), removed


def test_py_api_changed_signature_is_caught(tmp_path):
    root = tree(tmp_path)
    stub = pkg(root) / "_env.pyi"
    edit(stub, "def freeze_normalization(self, frozen: bool = True) -> None: ...", "def freeze_normalization(self, frozen: bool = False) -> None: ...")
    r = run(root)
    assert r.returncode == 1
    lines = diff_lines(r.stdout)
    assert "-method tessera_rl.Env.freeze_normalization(self, frozen: bool=True) -> None" in lines, lines
    assert "+method tessera_rl.Env.freeze_normalization(self, frozen: bool=False) -> None" in lines, lines


def test_py_api_renamed_parameter_is_caught(tmp_path):
    root = tree(tmp_path)
    edit(pkg(root) / "_env.pyi", "def freeze_normalization(self, frozen: bool = True) -> None: ...", "def freeze_normalization(self, freeze: bool = True) -> None: ...")
    r = run(root)
    assert r.returncode == 1
    assert any("freeze: bool" in line for line in diff_lines(r.stdout))


def test_py_api_adapter_removal_is_caught(tmp_path):
    root = tree(tmp_path)
    stub = pkg(root) / "gym.pyi"
    text = stub.read_text()
    m = re.search(r"^def (\w+)\(", text, re.M)
    assert m, "gym.pyi has a public function"
    name = m.group(1)
    start = m.start()
    end = text.find("\ndef ", start + 1)
    end = len(text) if end == -1 else end + 1
    stub.write_text(text[:start] + text[end:])
    r = run(root)
    assert r.returncode == 1
    assert any(line.startswith(f"-function tessera_rl.gym.{name}") for line in diff_lines(r.stdout))


def test_py_api_private_names_do_not_change_it(tmp_path):
    root = tree(tmp_path)
    with (pkg(root) / "_env.pyi").open("a") as f:
        f.write("\ndef _helper(x: int) -> int: ...\n")
    (pkg(root) / "_private_probe.pyi").write_text("def anything() -> None: ...\n")
    edit(pkg(root) / "_env.pyi", "class Env:\n", "class Env:\n    def _hidden(self) -> None: ...\n")
    assert run(root).returncode == 0


def test_py_api_class_moved_between_private_modules_does_not_change_it(tmp_path):
    """Rev's mutant: Archive's stub moves from _archive.pyi to _league.pyi;
    its public path and signature are unchanged, so is the snapshot."""
    root = tree(tmp_path)
    src = pkg(root) / "_archive.pyi"
    body = src.read_text()
    m = re.search(r"^class Archive\b.*?(?=^\S|\Z)", body, re.M | re.S)
    assert m
    imports = "".join(line + "\n" for line in body.splitlines() if line.startswith(("import ", "from ")))
    src.write_text(body.replace(m.group(0), ""))
    dst = pkg(root) / "_league.pyi"
    dst.write_text(imports + dst.read_text() + "\n" + m.group(0))
    edit(pkg(root) / "__init__.pyi", "from tessera_rl._archive import Archive as Archive", "from tessera_rl._league import Archive as Archive")
    r = run(root)
    assert r.returncode == 0, r.stdout


def test_py_api_all_without_stub_is_refused(tmp_path):
    root = tree(tmp_path)
    edit(pkg(root) / "__init__.py", '__all__ = [\n', '__all__ = [\n    "Nowhere",\n')
    r = run(root)
    assert r.returncode == 1
    assert "__all__ differs" in r.stdout and "Nowhere" in r.stdout


def test_py_api_stub_reexport_missing_from_all_is_refused(tmp_path):
    root = tree(tmp_path)
    edit(pkg(root) / "__init__.py", '    "Archive",\n', "")
    edit(pkg(root) / "__init__.pyi", '    "Archive",\n', "")
    # The stub still re-exports Archive, which the runtime no longer lists.
    r = run(root)
    assert r.returncode == 1
    assert "re-exports Archive" in r.stdout


def test_py_api_name_in_all_without_definition_is_refused(tmp_path):
    root = tree(tmp_path)
    edit(pkg(root) / "__init__.py", '__all__ = [\n', '__all__ = [\n    "Ghost",\n')
    edit(pkg(root) / "__init__.pyi", "__all__ = [\n", '__all__ = [\n    "Ghost",\n')
    r = run(root)
    assert r.returncode == 1
    assert "Ghost is public but has no stub" in r.stdout


def test_py_api_refuses_another_python(tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import runpy, sys\n"
        "sys.version_info = (3, 12, 0, 'final', 0)\n"
        f"sys.argv = ['check_py_api.py', 'check']\n"
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
    )
    r = subprocess.run([sys.executable, str(shim)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "uv run --python 3.11" in r.stdout
