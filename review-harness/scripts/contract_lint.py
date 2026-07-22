#!/usr/bin/env python3
"""Deterministic linter for harness finding files.

Rules (see build spec):
- every finding block has an `instances:` line ([file:line, ...] or single-instance)
- every finding has a `scenario: "..."` line with a trigger -> consequence arrow
- no hedged contracts (permissive-alternative wording)
- every finding cites at least one file:line evidence anchor
- severity present; F numbers unique

Exit 0 and print "LINT OK" when clean; otherwise print one violation per line
and exit 1. A file whose only content is `NO FINDINGS` passes with 0 findings.
"""
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"^### F(\d+)\b.*$", re.M)
# file-ish token (contains / or .) followed by :line or :line-range
ANCHOR_RE = re.compile(r"[^\s\[\],`'\"]*[./][^\s\[\],`'\"]*:\d+(?:-\d+)?")
SEVERITY_RE = re.compile(r"^severity:\s*(high|medium|low)\s*$", re.M)
SCENARIO_RE = re.compile(r"^scenario:\s*\"(.+)\"\s*$", re.M)
INSTANCES_RE = re.compile(r"^instances:\s*(single-instance\s*$|\[.*\]\s*$)", re.M)
CONTRACT_RE = re.compile(r"^contract:\s*(\S.*)$", re.M)
EVIDENCE_RE = re.compile(r"^evidence:\s*(\S.*)$", re.M)

HEDGES = [
    (re.compile(r"\bor alternatively\b", re.I), "'or alternatively'"),
    (re.compile(r"\bwhichever\b", re.I), "'whichever'"),
    (re.compile(r"\band/or\b", re.I), "'and/or'"),
    (re.compile(r"\beither\b.{0,120}?\bor\b", re.I | re.S), "'either ... or'"),
    (re.compile(r",\s*or\s+(?:you\s+could|consider|simply|just)\b", re.I), "', or you could/consider'"),
]


def blocks(text):
    """Yield (fnum, block_text) for each finding block."""
    matches = list(BLOCK_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield m.group(1), text[m.start():end]


def contract_text(block):
    """contract: value plus continuation lines up to the next key or blank."""
    m = CONTRACT_RE.search(block)
    if not m:
        return None
    lines = [m.group(1)]
    rest = block[m.end():].splitlines()
    for line in rest:
        if not line.strip() or re.match(
            r"^(severity|evidence|scenario|instances|contract):", line
        ) or line.startswith("### "):
            break
        lines.append(line)
    return "\n".join(lines)


def lint_text(text, label="findings"):
    violations = []
    found = list(blocks(text))

    if not found:
        if re.search(r"^\s*NO FINDINGS\s*$", text, re.M):
            return []
        return [f"{label}: no finding blocks and no 'NO FINDINGS' sentinel"]

    seen = {}
    for fnum, block in found:
        fid = f"F{fnum}"
        if fnum in seen:
            violations.append(f"{label} {fid}: duplicate finding number")
        seen[fnum] = True

        if not SEVERITY_RE.search(block):
            violations.append(f"{label} {fid}: missing/invalid 'severity:' (high|medium|low)")

        ev = EVIDENCE_RE.search(block)
        if not ev:
            violations.append(f"{label} {fid}: missing 'evidence:' line")
        elif not ANCHOR_RE.search(ev.group(1)):
            violations.append(f"{label} {fid}: evidence has no file:line anchor")

        sc = SCENARIO_RE.search(block)
        if not sc:
            violations.append(f"{label} {fid}: missing scenario: \"...\" line")
        elif ("→" not in sc.group(1)) and ("->" not in sc.group(1)):
            violations.append(f"{label} {fid}: scenario lacks trigger→consequence arrow")

        ct = contract_text(block)
        if ct is None:
            violations.append(f"{label} {fid}: missing 'contract:' line")
        else:
            for pat, name in HEDGES:
                if pat.search(ct):
                    violations.append(f"{label} {fid}: hedged contract ({name}); pin ONE recommendation")

        inst = INSTANCES_RE.search(block)
        if not inst:
            violations.append(f"{label} {fid}: missing 'instances:' ([file:line,...] or single-instance)")
        elif inst.group(1).strip() != "single-instance" and not ANCHOR_RE.search(inst.group(1)):
            violations.append(f"{label} {fid}: instances list has no file:line anchors")

    return violations


def main():
    if len(sys.argv) != 2:
        print("usage: contract_lint.py <findings.md>", file=sys.stderr)
        return 64
    path = Path(sys.argv[1])
    violations = lint_text(path.read_text(), label=path.name)
    if violations:
        for v in violations:
            print(v)
        return 1
    print("LINT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
