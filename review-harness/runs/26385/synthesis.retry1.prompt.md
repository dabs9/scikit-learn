# Role

You assemble the final deliverables by SELECTING findings verbatim. You are forbidden to
rewrite, soften, or drop finding content. Two deterministic auditors will diff your output
against every input finding and force restoration of anything missing.

# Inputs (absolute paths) and input-id namespaces

- /work/review-harness/runs/26385/depth.md    — ids D:F<n> (e.g. block "### F3" in depth.md is D:F3)
- /work/review-harness/runs/26385/closure.md  — ids C:F<n>
- /work/review-harness/runs/26385/inverse.md  — ids I:F<n>

A file containing only NO FINDINGS contributes zero ids.

# Task — write THREE files with the Write tool (absolute paths)

1. /work/review-harness/runs/26385/report.md — human-first report:
   - header: PR 26385, range 86541f2b3bc8a96264e265cb810cf79858544340..04c8b6954e3e4b8f8af086cea9d386f954b76bfe, counts by severity
   - findings grouped high → medium → low; each rendered as a severity-prefixed one-line
     title, then 2–4 plain sentences a maintainer can act on, then the verbatim block(s)
     inside a collapsed section:
     <details><summary>verbatim finding</summary>

     ```
     <original block(s), byte-identical>
     ```
     </details>
2. /work/review-harness/runs/26385/findings.json — JSON array; one object per delivered unit:
   {"id": "U<k>", "severity": "...", "title": "...", "file": "...", "line": <int>,
    "scenario": "...", "contract": "...", "instances": [...], "body": "<verbatim block(s)>"}
   (file/line = the primary evidence anchor; instances = the block's instances list, or
   [file:line] for single-instance.)
3. /work/review-harness/runs/26385/synthesis_ledger.json — JSON object mapping EVERY input finding id to
   {"disposition": "kept-as" | "merged-into" | "duplicate-of", "unit": "U<k>"}.
   kept-as: the unit is this finding verbatim. duplicate-of: byte-near-identical defect
   already delivered as unit U<k>. merged-into: grouped with siblings into unit U<k>.

# Rules

- Every input id appears in the ledger; every ledger unit exists in findings.json.
- Dedup only truly identical defects (same root cause, same site). When merging, keep the
  most specific member's wording verbatim and CARRY EVERY MEMBER'S evidence anchors into
  the unit body — the auditor rejects units that lost a member's anchors.
- No merge group larger than 6 members. At least 60% of input findings must survive as
  delivered units.
- Order findings.json by severity (high first), then by file path.

When the three files are written, reply with exactly: SYNTHESIS DONE


# RETRY NOTICE — the deterministic auditors rejected your output
PRESERVATION: unit U4 merge group size 7 > 6
missing_ids: 

Restore every missing/mishandled input finding and rewrite ALL THREE files.
