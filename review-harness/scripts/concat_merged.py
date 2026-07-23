#!/usr/bin/env python3
"""Deterministically concatenate per-theme merged findings into merged_all.md.

Blocks are renumbered globally (### F1..Fn) in fixed theme order (themes.tsv order);
only the heading line's F-number changes — every other byte of each block is preserved.
Also writes merged_map.json: global id -> {theme, original_id} for traceability.
merged_all.md is the byte-identical baseline the depth stage's snapshot check uses.
"""
import json
import re
import sys
from pathlib import Path

BLOCK_RE = re.compile(r"^### F(\d+)\b(.*)$", re.M)


def blocks(text):
    matches = list(BLOCK_RE.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), m.group(2), text[m.start():end].rstrip("\n")))
    return out


def main():
    if len(sys.argv) != 4:
        print("usage: concat_merged.py <themes.tsv> <merged_dir> <out_dir>", file=sys.stderr)
        return 64
    themes_tsv, merged_dir, out_dir = map(Path, sys.argv[1:])
    themes = [
        line.split("\t")[0].strip()
        for line in themes_tsv.read_text().splitlines()
        if line.strip()
    ]
    parts = []
    mapping = {}
    n = 0
    for theme in themes:
        p = merged_dir / f"{theme}.md"
        if not p.exists():
            print(f"error: missing merged file {p}", file=sys.stderr)
            return 1
        text = p.read_text()
        for orig_num, heading_rest, block in blocks(text):
            n += 1
            body = block.split("\n", 1)[1] if "\n" in block else ""
            parts.append(f"### F{n}{heading_rest}\n{body}".rstrip("\n"))
            mapping[f"F{n}"] = {"theme": theme, "original_id": f"F{orig_num}"}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "merged_all.md").write_text(
        ("\n\n".join(parts) + "\n") if parts else "NO FINDINGS\n"
    )
    (out_dir / "merged_map.json").write_text(json.dumps(mapping, indent=1))
    print(f"merged_all.md: {n} findings from {len(themes)} themes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
