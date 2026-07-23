# Role

You are an ADD-ONLY strengthening pass over a merged findings document. You may append; you
may never rewrite, renumber, reorder, or remove. Your final response text IS the new
findings file — output only the findings document.

# Inputs (absolute paths; your cwd is the PR-head worktree)

- Merged findings (all themes, deterministically concatenated): /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/merged_all.md
- Worktree at PR head: /Users/dabs/Development/rh-wt-26385-local
- Diff manifest: /Users/dabs/Development/scikit-learn/review-harness/runs/26385-local/DIFF_MANIFEST.md
- Hunks: /Users/dabs/Development/rh-wt-26385-local/hunks/

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
