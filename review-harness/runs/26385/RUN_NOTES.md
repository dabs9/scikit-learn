# Run notes — PR scikit-learn/scikit-learn#26385

## Configuration
- Model: `us.anthropic.claude-opus-4-7` (`REVIEW_MODEL=us.anthropic.claude-opus-4-7`)
- Thinking budget: adaptive default (`MAX_THINKING_TOKENS=` empty → unset, per run.sh:13, since forced Bedrock-style budgets aren't needed/supported for this endpoint)
- Worktree: `/tmp/rh-wt-26385` (`REVIEW_WT`)
- Pipeline: 12 themes × 3 blind draw waves (a/b/c) = 36 seats → per-theme merge → depth pass → rule-closure sweep → inverse audit → synthesis → fidelity/preservation audit loop → local delivery

## Findings counts by severity
Post-synthesis (`report.md` / `findings.json`): **high: 6, medium: 28, low: 43** (77 total)
Pre-synthesis merged pool (`merged_all.md`, 12 themes): 122 findings

## Top 5 findings
1. **high (U1)** — `HDBSCAN` docstring says `n_jobs` defaults to `None`; `__init__` actually defaults to `4` — silently pins 4 threads, contradicting documented/sklearn-standard behavior (`sklearn/cluster/_hdbscan/hdbscan.py:486,658`).
2. **high (U2)** — `_weighted_cluster_center` is called with the finite-row-reduced `X` but `self.labels_` has already been re-inflated to full raw length, so the boolean mask and data arrays disagree in size → `IndexError` (or silent wrong slicing) whenever `store_centers` is used on data with non-finite rows (`hdbscan.py:733,844,855,908-909`).
3. **high (U3)** — `_weighted_cluster_center`'s `n_clusters` computation hardcodes exclusion of `{-1, -2}` but omits the `-3` "missing" outlier label defined in `_OUTLIER_ENCODING`, inflating `n_clusters` and causing an extra empty-cluster loop iteration (`ZeroDivisionError`/`ValueError`) whenever NaN rows are present (`hdbscan.py:895`).
4. **high (U4)** — A test builds `clean_idx` via numpy broadcasting instead of list/array concatenation, so an outlier row is not actually removed from the "clean" model under test, weakening the test's intended assertion.
5. **high (U5)** — `test_hdbscan_precomputed_non_brute` references algorithm names that don't exist, so the assertion silently never exercises the code path it's meant to test.

(Full detail, evidence anchors, and all medium/low findings in `report.md` / `findings.json`.)

## Per-stage timings (from `timings.tsv`)
| Stage | Duration | Ended (UTC) |
|---|---|---|
| stage0_inputs | 3s | 07:21:39 |
| stage1a_draws (wave a, 12 seats) | 3301s (55.0m) | 08:16:40 |
| stage1b_draws (wave b, 12 seats) | 7622s (127.0m) | 10:23:42 |
| stage1c_draws (wave c, 12 seats) | 6451s (107.5m) | 12:11:13 |
| stage2_merge (12 per-theme unions) | 355s (5.9m) | 12:17:08 |
| stage3_depth | 2272s (37.9m) | 12:55:00 |
| stage4_closure | 914s (15.2m) | 13:10:14 |
| stage5_inverse | 219s (3.7m) | 13:13:53 |
| stage6+7_synthesis_fidelity | 2645s (44.1m) | 13:57:58 |
| stage8_deliver | 0s | 13:57:58 |

**Total wall-clock: 6h 36m 22s** (07:21:36 → 13:57:58 UTC), matching the sum of stage durations exactly (23782s).

## Harness misbehavior / incidents
- **Slow theme, timeout + retry (×2):** the `logic-and-correctness` seat exceeded the 4500s (75 min) per-attempt timeout on both wave b and wave c, each time succeeding on the automatic attempt-2 retry (alternate `--dangerously-skip-permissions` invocation style per `seat.sh`). Wave b attempt 2 succeeded in 7601s (~127m total for that seat including the failed attempt 1); wave c attempt 2 succeeded in 6437s. No data loss — retries are built into `seat.sh` (`SEAT_RETRIES=3`) and both eventually produced valid output.
- **Lint-loop revisions (self-healing, no data loss):** 15 total lint-violation → revise cycles across seats (see `harness_incidents.log` is silent on these — they're normal `contract_lint` revise-loop activity per stage 1's ≤3-round design): `draw_a_docs-and-comments`, `draw_a_readability-and-abstraction`, `draw_a_data-access-and-performance`, `draw_b_docs-and-comments`, `draw_b_dead-code-and-change-hygiene`, `draw_b_readability-and-abstraction`, `draw_b_build-config-and-operational-wiring`, `draw_b_logic-and-correctness`, `draw_c_docs-and-comments`, `draw_c_data-access-and-performance` (×2 revisions), `draw_c_readability-and-abstraction`, `draw_c_logic-and-correctness`, plus 2 revisions on the stage-5 `inverse` audit. All resolved within the allowed revise-round budget; no seat was abandoned.
- **Stage 3 depth-pass failure → verbatim fallback:** the add-only depth-strengthening pass failed its snapshot-diff (add-only) check on both try 1 and try 2 (`depth violations`), and after a 3rd unsuccessful try the harness fell back to using `merged_all.md` verbatim as the depth output (logged in `harness_incidents.log` as `depth_failed_after_3 falling back to merged_all.md`). This is a designed fallback — no findings were dropped (depth output findings count is 122, identical to the merged pool), but the intended "add-only strengthening" enrichment did not get applied to any finding this run.
- **Stage 6+7 fidelity/preservation audit — one violation, self-healed:** try 1 of the fidelity auditor failed with `PRESERVATION: unit U4 merge group size 7 > 6` (logged as `fidelity_violation try1` in `harness_incidents.log`). The loop-back-into-synthesis mechanism (≤3 tries) re-ran and try 2 came back clean (`fidelity audit clean` at 13:57:58), so the final `report.md`/`findings.json` are audit-clean with no missing findings (`missing_ids:` empty on the failing try, confirming the issue was oversized grouping, not dropped content).

All incidents were handled by the harness's own retry/self-heal logic exactly as designed; the run completed cleanly with exit code 0 and no manual intervention.
