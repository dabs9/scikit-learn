#!/usr/bin/env python3
"""Stage-7 fidelity + preservation auditors (deterministic, no agent).

(a) fidelity: every input finding id (D:F<n>/C:F<n>/I:F<n> from depth.md,
    closure.md, inverse.md) appears in synthesis_ledger.json with a valid
    disposition (kept-as | merged-into | duplicate-of) and a unit id that
    exists in findings.json.
(b) preservation: kept-ratio floor (delivered units / input findings >= 0.6),
    no merge group larger than --max-group, and every delivered unit carries
    at least one evidence anchor from each of its members.

Prints violations (one per line, prefixed FIDELITY/PRESERVATION) and a final
`missing_ids: ...` line for re-prompting. Exit 1 on any violation.
"""
import argparse
import json
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"^### F(\d+)\b.*$", re.M)
ANCHOR_RE = re.compile(r"[^\s\[\],`'\"]*[./][^\s\[\],`'\"]*:\d+(?:-\d+)?")
DISPOSITIONS = {"kept-as", "merged-into", "duplicate-of"}
INPUTS = [("depth.md", "D"), ("closure.md", "C"), ("inverse.md", "I")]


def blocks(text):
    matches = list(BLOCK_RE.finditer(text))
    out = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[m.group(1)] = text[m.start():end]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--min-kept-ratio", type=float, default=0.6)
    ap.add_argument("--max-group", type=int, default=6)
    a = ap.parse_args()
    run = Path(a.run_dir)

    inputs = {}  # "D:F3" -> block text
    for fname, prefix in INPUTS:
        p = run / fname
        if not p.exists():
            print(f"FIDELITY: input file missing: {fname}")
            return 1
        for fnum, block in blocks(p.read_text()).items():
            inputs[f"{prefix}:F{fnum}"] = block

    try:
        ledger = json.loads((run / "synthesis_ledger.json").read_text())
        findings = json.loads((run / "findings.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FIDELITY: cannot load synthesis outputs: {exc}")
        return 1

    if not isinstance(ledger, dict) or not isinstance(findings, list):
        print("FIDELITY: ledger must be an object and findings.json an array")
        return 1

    unit_ids = {f.get("id") for f in findings if isinstance(f, dict)}
    unit_bodies = {}
    for f in findings:
        if isinstance(f, dict):
            unit_bodies[f.get("id")] = " ".join(
                str(f.get(k, "")) for k in ("body", "evidence", "scenario", "instances", "title")
            )

    violations = []
    missing = []
    groups = {}  # unit -> [member ids]

    for iid in sorted(inputs):
        entry = ledger.get(iid)
        if entry is None:
            missing.append(iid)
            violations.append(f"FIDELITY: {iid} absent from ledger")
            continue
        disp = entry.get("disposition") if isinstance(entry, dict) else None
        unit = entry.get("unit") if isinstance(entry, dict) else None
        if disp not in DISPOSITIONS:
            violations.append(f"FIDELITY: {iid} invalid disposition {disp!r}")
            continue
        if unit not in unit_ids:
            violations.append(f"FIDELITY: {iid} maps to unknown unit {unit!r}")
            missing.append(iid)
            continue
        groups.setdefault(unit, []).append(iid)

    for lid in sorted(set(ledger) - set(inputs)):
        violations.append(f"FIDELITY: ledger id {lid} is not an input finding id")

    # preservation checks
    if inputs:
        ratio = len(findings) / len(inputs)
        if ratio < a.min_kept_ratio:
            violations.append(
                f"PRESERVATION: kept-ratio {ratio:.2f} below floor {a.min_kept_ratio}"
                f" ({len(findings)} delivered units / {len(inputs)} input findings)"
            )
    for unit, members in sorted(groups.items()):
        if len(members) > a.max_group:
            violations.append(
                f"PRESERVATION: unit {unit} merge group size {len(members)} > {a.max_group}"
            )
        body = unit_bodies.get(unit, "")
        for iid in members:
            anchors = set(ANCHOR_RE.findall(inputs[iid]))
            if anchors and not any(anc in body for anc in anchors):
                violations.append(
                    f"PRESERVATION: unit {unit} lost all evidence anchors of member {iid}"
                )
                missing.append(iid)

    delivered_without_members = sorted(unit_ids - set(groups))
    for unit in delivered_without_members:
        violations.append(f"FIDELITY: delivered unit {unit} has no ledger members")

    if violations:
        for v in violations:
            print(v)
        print("missing_ids: " + ", ".join(sorted(set(missing))))
        return 1
    print(
        f"AUDIT OK: {len(inputs)} input findings -> {len(findings)} delivered units, "
        f"largest merge group {max((len(m) for m in groups.values()), default=0)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
