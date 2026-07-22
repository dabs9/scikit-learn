#!/usr/bin/env python3
"""Build DIFF_MANIFEST.md plus per-file hunk diffs for a BASE..HEAD range.

Manifest: one line per changed file with A/M/D/R status and +/- counts,
grouped by top-level directory, with a total header.
Hunks: git diff BASE HEAD -- <file> written to <hunks-dir>/<path with / -> _>.diff
"""
import argparse
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


def git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def parse_entries(base, head, repo):
    """Return list of dicts: status, path (post-change), old_path (renames)."""
    out = git(["diff", "--name-status", "-M", "-z", base, head], repo)
    fields = out.split("\0")
    entries = []
    i = 0
    while i < len(fields):
        status = fields[i]
        if not status:
            i += 1
            continue
        if status[0] in ("R", "C"):
            entries.append(
                {"status": status, "old_path": fields[i + 1], "path": fields[i + 2]}
            )
            i += 3
        else:
            entries.append({"status": status, "old_path": None, "path": fields[i + 1]})
            i += 2
    return entries


def parse_numstat(base, head, repo):
    """Return {post-change path: (adds, dels)} (str counts; '-' for binary)."""
    out = git(["diff", "--numstat", "-M", "-z", base, head], repo)
    stats = {}
    fields = out.split("\0")
    i = 0
    while i < len(fields):
        if not fields[i]:
            i += 1
            continue
        parts = fields[i].split("\t")
        adds, dels = parts[0], parts[1]
        if len(parts) == 3 and parts[2]:
            # ordinary entry: adds\tdels\tpath in one field
            stats[parts[2]] = (adds, dels)
            i += 1
        else:
            # rename entry: adds\tdels\t<empty> NUL old NUL new
            stats[fields[i + 2]] = (adds, dels)
            i += 3
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hunks-dir", required=True)
    a = ap.parse_args()

    entries = parse_entries(a.base, a.head, a.repo)
    stats = parse_numstat(a.base, a.head, a.repo)

    hunks_dir = Path(a.hunks_dir)
    hunks_dir.mkdir(parents=True, exist_ok=True)

    total_add = total_del = 0
    groups = OrderedDict()
    for e in entries:
        adds, dels = stats.get(e["path"], ("0", "0"))
        if adds.isdigit():
            total_add += int(adds)
        if dels.isdigit():
            total_del += int(dels)
        top = e["path"].split("/")[0] if "/" in e["path"] else "(root)"
        groups.setdefault(top, []).append((e, adds, dels))

        paths = [p for p in (e["old_path"], e["path"]) if p]
        diff = git(["diff", a.base, a.head, "--", *paths], a.repo)
        hunk_name = e["path"].replace("/", "_") + ".diff"
        (hunks_dir / hunk_name).write_text(diff)

    lines = [
        f"# DIFF MANIFEST — {a.base[:12]}..{a.head[:12]}",
        f"Total: {len(entries)} files changed, +{total_add}/-{total_del}",
        "Hunk diffs: one file per changed path in the hunks directory "
        "(path with '/' replaced by '_', suffix .diff).",
        "",
    ]
    for top in sorted(groups):
        lines.append(f"## {top}")
        for e, adds, dels in sorted(groups[top], key=lambda t: t[0]["path"]):
            if e["old_path"]:
                name = f"{e['old_path']} -> {e['path']}"
            else:
                name = e["path"]
            lines.append(f"{e['status']:<5} {name}  +{adds}/-{dels}")
        lines.append("")

    Path(a.manifest).write_text("\n".join(lines))
    print(f"manifest: {a.manifest} ({len(entries)} files, +{total_add}/-{total_del})")
    print(f"hunks: {hunks_dir} ({len(entries)} diffs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
