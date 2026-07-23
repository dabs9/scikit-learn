# review-harness

Self-contained fan-out code-review pipeline (built to the spec in
`REVIEW_HARNESS_MISSION.md`). Reviews a PR diff against its plan of record with k
mutually-blind reviewer agents, union-merges without summarizing, and mechanically audits
the one compressing step so nothing is silently dropped. Local output only.

## Run

```bash
bash review-harness/run.sh
```

Requirements: `git`, `python3`, and the `claude` CLI authenticated via `ANTHROPIC_API_KEY`.
Configuration in `config.env` (PR number, BASE/HEAD SHAs, k, model). The run is resumable —
completed seats leave markers in `runs/<label>/markers/` and are skipped on rerun.

For benchmark runs that must not collide with an existing run dir:

```bash
RUN_LABEL=26385-local bash review-harness/run.sh
```

## Pipeline

stage 0 inputs (manifest + hunks + per-theme prompt rendering) → stage 1 three blind waves
(a/b/c) of 12 theme seats each — 12 generic themes (prompts/themes.tsv) × 3 draws = 36
seats, 12 concurrent per wave; each seat: draft → contract lint → revise (≤3 rounds) →
stage 2 twelve parallel per-theme union merges (never drop; most specific wording verbatim;
per-theme merge-shrinkage invariant), then deterministic concatenation/renumbering into
merged_all.md (+merged_map.json) → stage 3 add-only depth pass (snapshot-diff enforced) →
stage 4 rule-closure sweep over the full manifest → stage 5 inverse plan↔change inventory
audit → stage 6 synthesis (verbatim selection into report.md + findings.json +
synthesis_ledger.json) → stage 7 deterministic fidelity + preservation auditors (loop back
into synthesis until clean, ≤3) → stage 8 local delivery + timings.

Artifacts land in `runs/<label>/`: PLAN.md, DIFF_MANIFEST.md, draws_a/ draws_b/ draws_c/,
merged/, merged_all.md, merged_map.json, depth.md, closure.md, inverse.md, report.md,
findings.json, synthesis_ledger.json, timings.tsv, harness_incidents.log, markers/.

## Selftest (no agents)

```bash
bash review-harness/scripts/selftest.sh
```

Exercises the deterministic machinery on fixtures: lint accept/reject, snapshot add-only
check, fidelity auditor (including the hand-deleted-finding case that must force
restoration), and the diff manifest against this repo.
