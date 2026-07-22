#!/usr/bin/env python3
"""Verify an ADD-ONLY stage: every finding block of OLD survives byte-identical
in NEW, and NEW has at least as many findings. Exit 1 listing violations."""
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"^### F(\d+)\b.*$", re.M)


def blocks(text):
    matches = list(BLOCK_RE.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), text[m.start():end].rstrip()))
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: snapshot_check.py <old.md> <new.md>", file=sys.stderr)
        return 64
    old = Path(sys.argv[1]).read_text()
    new = Path(sys.argv[2]).read_text()
    old_blocks = blocks(old)
    new_count = len(blocks(new))
    violations = []
    for fnum, block in old_blocks:
        if block not in new:
            violations.append(
                f"F{fnum}: original block missing or altered (must survive byte-identical)"
            )
    if new_count < len(old_blocks):
        violations.append(
            f"finding count shrank: old={len(old_blocks)} new={new_count} (add-only stage)"
        )
    if violations:
        for v in violations:
            print(v)
        return 1
    print(f"SNAPSHOT OK ({len(old_blocks)} preserved, {new_count - len(old_blocks)} added)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
