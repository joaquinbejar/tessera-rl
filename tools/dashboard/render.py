# SPDX-License-Identifier: MIT OR Apache-2.0
"""Static pages from the judged result files. Fixed strings only: no
headline number, no superlative, no comparison across keys."""

from __future__ import annotations

import html
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grouping
import status as st

EMPTY = "No valid measurement exists yet. No host currently meets the wall-time method (ADR-0025)."
NO_REFERENCE = "No machine has reference status, so every configuration below is unmeasured."
WINDOW = "exclusive window declared by operator"
KNOWN_METRICS = frozenset({"throughput", "latency", "snapshot", "memory"})
STYLE = (
    "body{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:70rem;padding:0 1rem}"
    "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #888;padding:.25rem .5rem;text-align:left}"
    ".banner{border:2px solid #b60;padding:.5rem 1rem}"
)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def page(title: str, body: list[str]) -> str:
    return (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{esc(title)}</title>'
        f"<style>{STYLE}</style><h1>{esc(title)}</h1>" + "\n".join(body) + "</html>\n"
    )


def stats_of(result: dict) -> list[tuple[str, dict]]:
    return sorted((k, v) for k, v in result.items() if isinstance(v, dict) and "median" in v and "ci95" in v)


def number_cells(value: dict) -> str:
    low, high = value["ci95"]
    return f"<td>{esc(value['median'])}</td><td>{esc(low)} to {esc(high)}</td>"


def row(entry: st.Entry, value: dict) -> str:
    report = entry.doc["report"]
    return (
        f"<tr><td>{esc(entry.doc['measured_at'])}</td><td>{esc(report['environment']['software']['git'])}</td>"
        f"{number_cells(value)}<td>{esc(report['runs'])}</td><td>{WINDOW}</td>"
        f'<td><a href="results/{esc(entry.name)}">{esc(entry.name)}</a></td></tr>'
    )


HEADER = "<tr><th>Measured</th><th>Commit</th><th>Median</th><th>95% CI</th><th>Runs</th><th>Conditions</th><th>File</th></tr>"


def groups(valid: list[st.Entry]) -> dict:
    """Valid numbers by machine, then by comparison key."""
    out: dict = {}
    for entry in valid:
        report = entry.doc["report"]
        for name in sorted(report["results"]):
            result = report["results"][name]
            if result.get("metric") not in KNOWN_METRICS:
                continue
            for stat, value in stats_of(result):
                key = grouping.comparison_key(report, name, stat)
                out.setdefault(report["environment"]["machine"], {}).setdefault(key, []).append((entry, value))
    return out


def section(machine: str, by_key: dict) -> list[str]:
    body = [f"<h2>{esc(machine)}</h2>"]
    for key in sorted(by_key, key=repr):
        fields = dict(key)
        body.append(f"<h3>{esc(fields['configuration'])}: {esc(fields['statistic'])}</h3>")
        conditions = "; ".join(f"{k}={v}" for k, v in key)
        body.append(f"<p><small>{esc(conditions)}</small></p>")
        rows = sorted(by_key[key], key=lambda p: (p[0].doc["measured_at"], p[0].name))
        body.append("<table>" + HEADER + "".join(row(e, v) for e, v in rows) + "</table>")
        same = [k for k in by_key if k != key and dict(k)["configuration"] == fields["configuration"] and dict(k)["statistic"] == fields["statistic"]]
        for other in sorted(same, key=repr):
            body.append(f"<p>Separate series; differs in: {esc(', '.join(grouping.differing(key, other)))}.</p>")
    return body


def derived(entry: st.Entry) -> list[str]:
    """One report's own derived values; a value marked invalid has no number."""
    values = entry.doc["report"].get("derived", {})
    rows = []
    for name in sorted(values):
        v = values[name]
        if "invalid" in v:
            rows.append(f"<tr><td>{esc(name)}</td><td colspan=2>invalid: {esc(v['invalid'])}</td></tr>")
        else:
            low, high = v.get("ci95", ("not recorded", "not recorded"))
            rows.append(f"<tr><td>{esc(name)}</td><td>{esc(v['value'])}</td><td>{esc(low)} to {esc(high)}</td></tr>")
    if not rows:
        return []
    return [f"<h3>Derived in {esc(entry.name)}</h3>", "<table><tr><th>Name</th><th>Value</th><th>95% CI</th></tr>", *rows, "</table>"]


def undisplayed(valid: list[st.Entry]) -> list[str]:
    names = sorted(
        (e.name, n) for e in valid for n, r in e.doc["report"]["results"].items() if r.get("metric") not in KNOWN_METRICS
    )
    return [f"<p>{esc(n)} in {esc(f)}: not displayed.</p>" for f, n in names]


def missing(entries: list[st.Entry], configs: list[str], machines: dict) -> list[str]:
    reference = sorted(name for name, m in machines.items() if m.get("status") == "reference")
    body = ["<h2>Not measured</h2>"]
    if not reference:
        return [*body, f"<p>{NO_REFERENCE}</p>", "<ul>", *(f"<li>{esc(c)}</li>" for c in configs), "</ul>"]
    have = {
        (e.doc["report"]["environment"]["machine"], name)
        for e in entries
        if e.status == st.VALID
        for name in e.doc["report"]["results"]
    }
    rows = [f"<tr><td>{esc(c)}</td><td>{esc(m)}</td><td>not measured</td></tr>" for m in reference for c in configs if (m, c) not in have]
    return [*body, "<table><tr><th>Configuration</th><th>Machine</th><th>Status</th></tr>", *rows, "</table>"]


def listing(entries: list[st.Entry], wanted: str, title: str) -> list[str]:
    chosen = [e for e in entries if e.status == wanted]
    if not chosen:
        return []
    body = [f"<h2>{esc(title)}</h2>", "<ul>"]
    for e in chosen:
        target = f'<a href="report-{esc(e.name)}.html">{esc(e.name)}</a>' if wanted == st.INDICATIVE else esc(e.name)
        body.append(f"<li>{target}: {esc('; '.join(e.reasons))}</li>")
    return [*body, "</ul>"]


def newest(valid: list[st.Entry]) -> st.Entry | None:
    return max(valid, key=lambda e: (e.doc["measured_at"], e.name), default=None)


def index(entries: list[st.Entry], configs: list[str], machines: dict) -> str:
    valid = [e for e in entries if e.status == st.VALID]
    latest = newest(valid)
    body = [f"<p>{EMPTY}</p>"] if latest is None else [f"<p>Newest result: commit {esc(latest.doc['report']['environment']['software']['git'])}.</p>"]
    by_machine = groups(valid)
    for machine in sorted(by_machine):
        body += section(machine, by_machine[machine])
    for entry in valid:
        body += derived(entry)
    body += undisplayed(valid)
    body += missing(entries, configs, machines)
    body += listing(entries, st.INDICATIVE, "Indicative reports")
    body += listing(entries, st.SUPERSEDED, "Superseded reports")
    body += listing(entries, st.INVALID, "Invalid files")
    return page("Tessera benchmarks", body)


def report_page(entry: st.Entry) -> str:
    """One report on its own page. Indicative numbers appear only here."""
    report = entry.doc["report"]
    body = [f'<p class="banner">{esc(entry.status)}: {esc("; ".join(entry.reasons))}. Not comparable with any other report.</p>']
    body.append(f"<p>Commit {esc(report['environment']['software']['git'])}; machine {esc(report['environment']['machine'])}.</p>")
    body.append("<table><tr><th>Configuration</th><th>Statistic</th><th>Median</th><th>95% CI</th></tr>")
    for name in sorted(report["results"]):
        result = report["results"][name]
        if result.get("metric") not in KNOWN_METRICS:
            body.append(f"<tr><td>{esc(name)}</td><td colspan=3>not displayed</td></tr>")
            continue
        for stat, value in stats_of(result):
            body.append(f"<tr><td>{esc(name)}</td><td>{esc(stat)}</td>{number_cells(value)}</tr>")
    body.append("</table>")
    return page(entry.name, body)



def write(out: pathlib.Path, root: pathlib.Path, entries: list[st.Entry], configs: list[str], machines: dict) -> None:
    out.mkdir(parents=True)
    (out / "index.html").write_text(index(entries, configs, machines), encoding="utf-8")
    results = out / "results"
    for entry in entries:
        if entry.status not in (st.VALID, st.INDICATIVE):
            continue
        results.mkdir(exist_ok=True)
        source = root.joinpath(*st.RESULTS.parts) / entry.name
        (results / entry.name).write_bytes(source.read_bytes())
        if entry.status == st.INDICATIVE:
            (out / f"report-{entry.name}.html").write_text(report_page(entry), encoding="utf-8")
