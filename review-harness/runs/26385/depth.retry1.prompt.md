# Role

You are an ADD-ONLY strengthening pass over a merged findings document. You may append; you
may never rewrite, renumber, reorder, or remove. Your final response text IS the new
findings file — output only the findings document.

# Inputs (absolute paths; your cwd is the PR-head worktree)

- Merged findings (all themes, deterministically concatenated): /work/review-harness/runs/26385/merged_all.md
- Worktree at PR head: /tmp/rh-wt-26385
- Diff manifest: /work/review-harness/runs/26385/DIFF_MANIFEST.md
- Hunks: /tmp/rh-wt-26385/hunks/

# Task

1. Re-read EVERY medium and high finding in merged_all.md against the actual tree.
2. Strengthen ADD-ONLY by APPENDING new finding blocks (numbering continues after the last
   existing F number) for:
   - additional instances of an existing finding's defect that merged_all.md missed (title
     the block "additional instances of F<k>: ..." and cite the new anchors);
   - sibling defects you discovered while re-verifying (same area, distinct defect);
   - harder evidence that materially changes a finding (append as a new block referencing
     F<k>; do not touch the original).
3. Output = the ENTIRE merged_all.md content BYTE-IDENTICAL (every original block untouched,
   same order), followed by your appended blocks. A deterministic snapshot-diff will reject
   any run where an original block changed by even one byte.

# Output format

Original document verbatim, then appended blocks in the standard format (### F<N> — title /
severity / evidence / scenario / contract / instances). Appended blocks must satisfy the
same lint contract. If you have nothing to add, output the original document verbatim and
nothing else.


# RETRY NOTICE — your previous output failed deterministic checks
## Contract lint
depth.md F131: scenario lacks trigger→consequence arrow
depth.md F139: scenario lacks trigger→consequence arrow
depth.md F140: scenario lacks trigger→consequence arrow

## Snapshot (add-only) check
F1: original block missing or altered (must survive byte-identical)
F2: original block missing or altered (must survive byte-identical)
F3: original block missing or altered (must survive byte-identical)
F4: original block missing or altered (must survive byte-identical)
F5: original block missing or altered (must survive byte-identical)
F6: original block missing or altered (must survive byte-identical)
F7: original block missing or altered (must survive byte-identical)
F8: original block missing or altered (must survive byte-identical)
F9: original block missing or altered (must survive byte-identical)
F10: original block missing or altered (must survive byte-identical)
F11: original block missing or altered (must survive byte-identical)
F12: original block missing or altered (must survive byte-identical)
F13: original block missing or altered (must survive byte-identical)
F14: original block missing or altered (must survive byte-identical)
F15: original block missing or altered (must survive byte-identical)
F16: original block missing or altered (must survive byte-identical)
F17: original block missing or altered (must survive byte-identical)
F18: original block missing or altered (must survive byte-identical)
F19: original block missing or altered (must survive byte-identical)
F20: original block missing or altered (must survive byte-identical)
F21: original block missing or altered (must survive byte-identical)
F22: original block missing or altered (must survive byte-identical)
F23: original block missing or altered (must survive byte-identical)
F24: original block missing or altered (must survive byte-identical)
F25: original block missing or altered (must survive byte-identical)
F26: original block missing or altered (must survive byte-identical)
F27: original block missing or altered (must survive byte-identical)
F28: original block missing or altered (must survive byte-identical)
F29: original block missing or altered (must survive byte-identical)
F30: original block missing or altered (must survive byte-identical)
F31: original block missing or altered (must survive byte-identical)
F32: original block missing or altered (must survive byte-identical)
F33: original block missing or altered (must survive byte-identical)
F34: original block missing or altered (must survive byte-identical)
F35: original block missing or altered (must survive byte-identical)
F36: original block missing or altered (must survive byte-identical)
F37: original block missing or altered (must survive byte-identical)
F38: original block missing or altered (must survive byte-identical)
F39: original block missing or altered (must survive byte-identical)
F40: original block missing or altered (must survive byte-identical)
F41: original block missing or altered (must survive byte-identical)
F42: original block missing or altered (must survive byte-identical)
F43: original block missing or altered (must survive byte-identical)
F44: original block missing or altered (must survive byte-identical)
F45: original block missing or altered (must survive byte-identical)
F46: original block missing or altered (must survive byte-identical)
F47: original block missing or altered (must survive byte-identical)
F48: original block missing or altered (must survive byte-identical)
F49: original block missing or altered (must survive byte-identical)
F50: original block missing or altered (must survive byte-identical)
F51: original block missing or altered (must survive byte-identical)
F52: original block missing or altered (must survive byte-identical)
F53: original block missing or altered (must survive byte-identical)
F54: original block missing or altered (must survive byte-identical)
F55: original block missing or altered (must survive byte-identical)
F56: original block missing or altered (must survive byte-identical)
F57: original block missing or altered (must survive byte-identical)
F58: original block missing or altered (must survive byte-identical)
F59: original block missing or altered (must survive byte-identical)
F60: original block missing or altered (must survive byte-identical)
F61: original block missing or altered (must survive byte-identical)
F62: original block missing or altered (must survive byte-identical)
F63: original block missing or altered (must survive byte-identical)
F64: original block missing or altered (must survive byte-identical)
F65: original block missing or altered (must survive byte-identical)
F66: original block missing or altered (must survive byte-identical)
F67: original block missing or altered (must survive byte-identical)
F68: original block missing or altered (must survive byte-identical)
F69: original block missing or altered (must survive byte-identical)
F70: original block missing or altered (must survive byte-identical)
F71: original block missing or altered (must survive byte-identical)
F72: original block missing or altered (must survive byte-identical)
F73: original block missing or altered (must survive byte-identical)
F74: original block missing or altered (must survive byte-identical)
F75: original block missing or altered (must survive byte-identical)
F76: original block missing or altered (must survive byte-identical)
F77: original block missing or altered (must survive byte-identical)
F78: original block missing or altered (must survive byte-identical)
F79: original block missing or altered (must survive byte-identical)
F80: original block missing or altered (must survive byte-identical)
F81: original block missing or altered (must survive byte-identical)
F82: original block missing or altered (must survive byte-identical)
F83: original block missing or altered (must survive byte-identical)
F84: original block missing or altered (must survive byte-identical)
F85: original block missing or altered (must survive byte-identical)
F86: original block missing or altered (must survive byte-identical)
F87: original block missing or altered (must survive byte-identical)
F88: original block missing or altered (must survive byte-identical)
F89: original block missing or altered (must survive byte-identical)
F90: original block missing or altered (must survive byte-identical)
F91: original block missing or altered (must survive byte-identical)
F92: original block missing or altered (must survive byte-identical)
F93: original block missing or altered (must survive byte-identical)
F94: original block missing or altered (must survive byte-identical)
F95: original block missing or altered (must survive byte-identical)
F96: original block missing or altered (must survive byte-identical)
F97: original block missing or altered (must survive byte-identical)
F98: original block missing or altered (must survive byte-identical)
F99: original block missing or altered (must survive byte-identical)
F100: original block missing or altered (must survive byte-identical)
F101: original block missing or altered (must survive byte-identical)
F102: original block missing or altered (must survive byte-identical)
F103: original block missing or altered (must survive byte-identical)
F104: original block missing or altered (must survive byte-identical)
F105: original block missing or altered (must survive byte-identical)
F106: original block missing or altered (must survive byte-identical)
F107: original block missing or altered (must survive byte-identical)
F108: original block missing or altered (must survive byte-identical)
F109: original block missing or altered (must survive byte-identical)
F110: original block missing or altered (must survive byte-identical)
F111: original block missing or altered (must survive byte-identical)
F112: original block missing or altered (must survive byte-identical)
F113: original block missing or altered (must survive byte-identical)
F114: original block missing or altered (must survive byte-identical)
F115: original block missing or altered (must survive byte-identical)
F116: original block missing or altered (must survive byte-identical)
F117: original block missing or altered (must survive byte-identical)
F118: original block missing or altered (must survive byte-identical)
F119: original block missing or altered (must survive byte-identical)
F120: original block missing or altered (must survive byte-identical)
F121: original block missing or altered (must survive byte-identical)
F122: original block missing or altered (must survive byte-identical)
finding count shrank: old=122 new=15 (add-only stage)

Reproduce the ORIGINAL merged_all.md content byte-identical and only APPEND clean blocks.
