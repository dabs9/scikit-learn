# Run notes — review-harness PR 26385 (HDBSCAN)

## Environment fixes required before this run completed
- Default `REVIEW_WT` path (`dirname($ROOT)/rh-wt-<label>` = `/rh-wt-26385`) is not writable in
  this sandbox (`/` is root-owned). Fixed by passing `REVIEW_WT=/work/.rh-worktrees/rh-wt-26385`
  (an existing, documented override point in `run.sh`; no script changes made).
- `config.env`'s default `REVIEW_MODEL="claude-opus-4-8"` is rejected by the API in this
  environment (`400 The provided model identifier is invalid`). Fixed by overriding
  `REVIEW_MODEL=us.anthropic.claude-opus-4-7` via env var (again, an existing override point —
  no script/config edits made). The first attempt with the stale default failed all 6 stage-1
  draw seats after 3 retries each; that run's markers/logs were cleared and the harness was
  rerun clean with the corrected env vars.

## Findings counts by severity
- high: 5
- medium: 11
- low: 25
- **total: 41**

## Top 5 findings
1. **[high]** `_weighted_cluster_center` crashes when input has non-finite rows and `store_centers` is set — `sklearn/cluster/_hdbscan/hdbscan.py`
2. **[high]** `_weighted_cluster_center` miscounts clusters when missing-data label `-3` is present — `sklearn/cluster/_hdbscan/hdbscan.py`
3. **[high]** `_hdbscan_prims` never applies `alpha` — silent parameter drop for tree-based algorithms — `sklearn/cluster/_hdbscan/hdbscan.py`
4. **[high]** `_hdbscan_brute` unconditionally divides `distance_matrix /= alpha` even when `alpha is None` — `sklearn/cluster/_hdbscan/hdbscan.py`
5. **[high]** `HDBSCAN.dbscan_clustering` calls `labelling_at_cut` with the length-`n_samples` internal tree but returns labels sized to `finite_count`, breaking `dbscan_clustering` after non-finite handling — `sklearn/cluster/_hdbscan/hdbscan.py`

## Per-stage timings (from timings.tsv)
| stage | seconds | mm:ss |
|---|---|---|
| stage0_inputs | 0 | 0:00 |
| stage1_draws (k=6 parallel) | 1229 | 20:29 |
| stage2_merge | 382 | 6:22 |
| stage3_depth | 1479 | 24:39 |
| stage4_closure | 578 | 9:38 |
| stage5_inverse | 281 | 4:41 |
| stage67_synthesis_fidelity | 620 | 10:20 |
| stage8_deliver | 0 | 0:00 |

**Total wall clock (stage0 start → stage8 finish): 4569s ≈ 76 min 9 sec**
(harness log timestamps: 23:02:11 → 00:18:20 UTC)

## Where the harness misbehaved / self-corrected
- **No merge shrinkage**: stage 2 union merge produced 27 findings on the first try vs. a max
  single-draw count of 16 — invariant held, no retry needed.
- **No fidelity-audit failures**: stage 6+7 synthesis passed the deterministic fidelity +
  preservation auditors on the first attempt (`fidelity audit clean` at 00:18:20); no
  `harness_incidents.log` file was created (none of merge_shrinkage / depth_failed_after_3 /
  fidelity_violation triggered).
- **Lint revision loops (normal, self-healing)**: 7 total across the run — draw_2, draw_3,
  draw_4, draw_6 each needed one revision round in stage 1; the closure seat needed one revision;
  the inverse seat needed two revisions. All resolved within the allowed retry budget (≤3 rounds)
  and none escalated to a hard failure.
- **Stage 3 depth pass** needed one retry (`depth violations (try 1)` at 00:50:09) before
  producing a clean add-only doc (41 findings, up from merged.md's 27) — resolved on try 2,
  well inside the 3-try budget.

## Deliverables
- `report.md` (92,238 bytes)
- `findings.json` (108,772 bytes)
- `synthesis_ledger.json` (3,448 bytes)
- `timings.tsv` (285 bytes)
- `runner.log`, `harness_incidents.log` (absent — no incidents), draws/, markers/ (all staged)
