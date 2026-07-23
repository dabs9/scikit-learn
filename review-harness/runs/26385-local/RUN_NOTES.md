# Run notes — review-harness PR 26385 (HDBSCAN), LOCAL leg, themed architecture (v2)

Architecture: 12 generic themes × 3 blind waves = 36 draw seats (12 concurrent per wave)
→ 12 parallel per-theme union merges → deterministic concat (merged_all.md) → depth →
closure → inverse → synthesis + deterministic fidelity/preservation audit.
Model: `claude-opus-4-7` via the Anthropic API (user key), adaptive thinking
(MAX_THINKING_TOKENS empty — the API rejects forced `thinking.enabled` for this model).
Host: user's macOS laptop, caffeinated; seats run `claude -p` non-interactively.

## Findings counts by severity
- high: 19
- medium: 81
- low: 148
- **total delivered units: 248**

Calibration note: per the mission, 19 highs on merged, human-reviewed sklearn code likely
includes over-calls; the themed fan-out trades precision for recall. Top-of-report triage
recommended before treating counts as signal.

## Top 5 findings (one line each)
1. **[high]** `_weighted_cluster_center` shape mismatch when non-finite data + `store_centers` set — sklearn/cluster/_hdbscan/hdbscan.py:854
2. **[high]** `_weighted_cluster_center` treats `-3` (missing-data label) as a valid cluster — sklearn/cluster/_hdbscan/hdbscan.py:895
3. **[high]** `_hdbscan_prims` forwards `max_distance` to `DistanceMetric.get_metric` → unexpected-keyword failure — sklearn/cluster/_hdbscan/hdbscan.py:344
4. **[high]** `n_jobs` default is 4 while `_parameter_constraints` and docstring say None [out-of-theme] — sklearn/cluster/_hdbscan/hdbscan.py:486
5. **[high]** `plot_hdbscan.py` scale-invariance demo never scales the data — examples/cluster/plot_hdbscan.py:106

## Per-stage timings (timings.tsv; stage1b reconstructed from seat logs)
| stage | seconds | note |
|---|---|---|
| stage0 inputs | 2 | |
| stage1a draws (12 seats) | 326 | |
| stage1b draws (12 seats) | ~461 | timer split by crash; reconstructed from seat logs 02:46:05→02:53:46 |
| stage1c draws (12 seats) | 379 | |
| stage2 merges (12 parallel) + concat | 177 | no shrinkage incidents |
| stage3 depth | 810 | fell back after 3 tries (see misbehaviors) |
| stage4 closure | 233 | |
| stage5 inverse | 193 | |
| stage6+7 synthesis + fidelity | 748 | 1 fidelity bounce, forced restoration |
| stage8 deliver | 0 | |

**Wall clock as-lived: 56.9 min** (02:40:36 → 03:37:28 UTC), including ~1.4 min of
operator repair between wave-B failure and resume. Active pipeline ≈ 55.5 min.

## Where the harness misbehaved / self-corrected
- **Fidelity loop fired for real (the flagship mechanism):** synthesis attempt 1 dropped
  ~180 of 248 input findings; the deterministic auditors enumerated every missing id and
  the re-prompt restored them — audit clean on retry. Without stage 7 the run would have
  silently delivered ~25% of its own findings.
- **Depth stage contributed nothing:** 3 tries failed the byte-identical snapshot check on
  the (now much longer) merged_all.md; designed fallback used merged_all.md verbatim.
  Design fix for v3: depth should emit ONLY appended blocks; runner concatenates
  deterministically, removing the byte-identical reproduction requirement.
- **One draw seat lint-exhausted (wave B, readability):** root cause was a linter
  over-narrowing (required the file:line anchor on the `evidence:` line itself; the seat
  put anchors on continuation lines). Linter fixed mid-run (accept continuation-line
  anchors); the existing draft then passed unmodified and was marker-accepted. Also fixed
  earlier in this leg: misleading "missing instances" message for malformed
  `single-instance — <anchor>` lines (caused a 3-round revise loop death in the v1 run).
- **Environment deltas vs cloud leg:** macOS bash 3.2 empty-array expansion bug in seat.sh
  (fixed); Anthropic API rejects forced thinking.enabled for opus-4-7 (adaptive used);
  stale registered worktree after manual dir deletion (git worktree prune).
- Routine lint-revision rounds across draw seats: ~15 total, all self-healed within budget.

## Deliverables
- report.md (404,206 bytes), findings.json (469,985 bytes),
  synthesis_ledger.json (16,638 bytes; 248 units, full input-id coverage),
  merged_all.md + merged_map.json (traceability), timings.tsv,
  harness_incidents.log (2 entries), draws_a/b/c/, merged/, markers/.

## Benchmark context
- Cloud leg (same v2 architecture) NOT yet run: Functio execution backend down since
  02:39 UTC (every run fails in 0s, no image, null result; control plane up; reproduced
  with no-op prompts on two branches). Prior night's cloud run (v1 architecture,
  6 generalist seats): 76m09s pipeline / 87.6m end-to-end, 41 findings — not directly
  comparable to v2 numbers.
