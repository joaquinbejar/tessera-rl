# SPDX-License-Identifier: MIT OR Apache-2.0
"""The flagship table (SPEC 22 phase 4, #640): every flagship number of the
valid and indicative result files, each row with its machine's current
``compat/machines.toml`` status and the report's ``valid_for_claims``, so an
indicative number cannot be read as a claim beside a reference one."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import status as st

PAGE = "flagships.html"
TITLE = "Tessera flagship benchmarks"
INTRO = (
    "Every flagship number of the admitted result files, valid or indicative. A row is a claim only when"
    " valid_for_claims is true, which needs a machine of reference status; every other row is indicative."
    f" Scaling rows are {SCALING}: k-worker Sync throughput over k times the Inline throughput, so they include"
    " the Sync scheduling overhead. Illegal-action no-ops says whether the throughput includes steps whose"
    " planned action was illegal and became a no-op (yes) or steps legal actions only (no)."
)
NONE = "No flagship result has been admitted yet."
UNLISTED = "not in compat/machines.toml"
EXACT = "exact (no CI)"
# Scaling is k-worker Sync throughput over k times Inline, not over one
# Sync worker (#640 review r1), so it carries the Sync overhead.
SCALING = "efficiency vs k×Inline"
# A configuration without ``illegal_action_noops`` (a file from before it).
UNRECORDED = "not recorded"
# Memory figures are exact counts, published without a CI.
MEMORY_FIELDS = ("record_bytes_per_instance", "mask_bytes_per_step")
COLUMNS = (
    "Flagship", "Configuration", "Metric", "Median", "95% CI", "Workers", "Schedule",
    "Illegal-action no-ops",
    "Machine", "Machine status", "Commit", "valid_for_claims", "File",
)  # fmt: skip


def _stats(result: dict) -> list[tuple[str, dict]]:
    return sorted((k, v) for k, v in result.items() if isinstance(v, dict) and "median" in v and "ci95" in v)


def _noops(config: dict) -> str:
    """Whether the throughput includes illegal-action no-ops."""
    value = config.get("illegal_action_noops")
    return UNRECORDED if value is None else ("yes" if value else "no")


def rows(entries: list[st.Entry], machines: dict) -> list[dict]:
    """One row per flagship statistic of every valid or indicative file."""
    out = []
    for entry in entries:
        if entry.status not in (st.VALID, st.INDICATIVE):
            continue
        report = entry.doc["report"]
        machine = report["environment"]["machine"]
        record = machines.get(machine) if machine else None
        common = {
            "machine": machine or "none named",
            "machine_status": record.get("status", UNLISTED) if record else UNLISTED,
            "commit": report["environment"]["software"]["git"],
            "valid_for_claims": report["valid_for_claims"] is True and entry.status == st.VALID,
            "file": entry.name,
        }
        flagship_of = {}
        for name, result in report["results"].items():
            flagship = result.get("config", {}).get("flagship")
            if not flagship:
                continue
            flagship_of[name] = flagship
            metric = result.get("metric", result["config"].get("metric"))
            run = {
                "workers": result.get("workers", "n/a"), "schedule": result.get("schedule") or "n/a",
                "noops": _noops(result["config"]),
            }  # fmt: skip
            if metric == "memory":
                for field in MEMORY_FIELDS:
                    if field in result:
                        out.append({**common, **run, "flagship": flagship, "configuration": name, "metric": f"memory: {field}", "median": result[field], "ci95": None})
                continue
            for stat, value in _stats(result):
                out.append({**common, **run, "flagship": flagship, "configuration": name, "metric": f"{metric}: {stat}", "median": value["median"], "ci95": value["ci95"]})
        for name, value in report.get("derived", {}).items():
            source = (value.get("from") or [None])[0]
            if source not in flagship_of or not name.startswith("scaling_efficiency/") or "invalid" in value:
                continue
            multi = report["results"][source]
            run = {
                "workers": multi.get("workers", "n/a"), "schedule": multi.get("schedule") or "n/a",
                "noops": _noops(multi.get("config", {})),
            }  # fmt: skip
            out.append({**common, **run, "flagship": flagship_of[source], "configuration": name, "metric": SCALING, "median": value["value"], "ci95": value.get("ci95")})
    return sorted(out, key=lambda r: (r["flagship"], r["configuration"], r["metric"], r["file"]))


def _cells(row: dict, esc) -> str:
    ci = EXACT if row["ci95"] is None else f"{esc(row['ci95'][0])} to {esc(row['ci95'][1])}"
    values = (
        esc(row["flagship"]), esc(row["configuration"]), esc(row["metric"]), esc(row["median"]), ci,
        esc(row["workers"]), esc(row["schedule"]), esc(row["noops"]), esc(row["machine"]), esc(row["machine_status"]),
        esc(row["commit"]), "true" if row["valid_for_claims"] else "false",
        f'<a href="results/{esc(row["file"])}">{esc(row["file"])}</a>',
    )  # fmt: skip
    return "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"


def body(entries: list[st.Entry], machines: dict, esc) -> list[str]:
    found = rows(entries, machines)
    if not found:
        return [f"<p>{INTRO}</p>", f"<p>{NONE}</p>"]
    header = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in COLUMNS) + "</tr>"
    return [f"<p>{INTRO}</p>", "<table>" + header + "".join(_cells(r, esc) for r in found) + "</table>"]
