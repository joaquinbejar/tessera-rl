# SPDX-License-Identifier: MIT OR Apache-2.0
"""The reference-libraries nightly's per-entry gate (#501, #579): every
matrix entry has a boolean `gate`, the job is continue-on-error exactly when
the entry is not gated, and each entry runs its own make target. Parsed from
the workflow text, like test_pypi_workflow.py."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "reference-libraries.yml").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()

# The gate of each entry as recorded; a flip after 7 green nights updates it.
GATES = {"sb3": False, "tianshou": False, "rllib": False, "rllib-tag": False}

ENTRY = re.compile(r"^\s*- \{ job: ([\w-]+), project: ([\w-]+), gate: (\w+) \}\s*$", re.M)


def entries():
    return {job: (project, gate) for job, project, gate in ENTRY.findall(WORKFLOW)}


def test_reference_workflow_every_matrix_entry_has_a_boolean_gate():
    include = WORKFLOW.split("include:", 1)[1].split("steps:", 1)[0]
    lines = [line for line in include.splitlines() if line.strip().startswith("- ")]
    assert len(lines) == len(entries()), "every matrix entry is a one-line {job, project, gate} map"
    assert {gate for _, gate in entries().values()} <= {"true", "false"}


def test_reference_workflow_gates_are_as_recorded():
    assert {job: gate == "true" for job, (_, gate) in entries().items()} == GATES


def test_reference_workflow_continue_on_error_follows_the_gate():
    assert re.findall(r"^\s*continue-on-error:\s*(.+?)\s*$", WORKFLOW, re.M) == ["${{ !matrix.gate }}"]


def test_reference_workflow_each_entry_runs_its_make_target():
    assert "run: make reference-${{ matrix.job }}" in WORKFLOW
    for job, (project, _) in entries().items():
        assert re.search(rf"^reference-{re.escape(job)}:", MAKEFILE, re.M), job
        assert (ROOT / "compat" / "training" / project).is_dir(), project
