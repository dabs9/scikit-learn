### F1 — Unsanctioned whitespace-only edits in Mean Shift documentation
severity: low
evidence:
  - Plan of record (`/work/review-harness/runs/26385/PLAN.md:4`) scopes this PR to "Add `HDBSCAN` as a new estimator in `sklearn.cluster`" plus three explicit novel Cython-side changes (`/work/review-harness/runs/26385/PLAN.md:15-17`): `cnp.*_t` → `*_t`, `*.shape[0]` → `len(*)`, and trimming unused Cython variables. Nothing in the plan sanctions edits to unrelated prose in the Mean Shift section of `doc/modules/clustering.rst`.
  - Diff `hunks/doc_modules_clustering.rst.diff:22-26` (removes trailing whitespace on "called hill " and "estimated probability density."), `hunks/doc_modules_clustering.rst.diff:30-31` ("depends on a kernel used for density estimation. "), and `hunks/doc_modules_clustering.rst.diff:34-35` ("is small enough and is ") — all inside the Mean Shift section of `doc/modules/clustering.rst:402-429`, unrelated to HDBSCAN.
scenario: "A reviewer scans the `-5` half of the `doc/modules/clustering.rst` manifest entry (`+118/-5`) expecting deletions tied to the HDBSCAN addition (e.g. the legitimate replacement of the external `HDBSCAN <https://hdbscan.readthedocs.io>` URL with `:class:`HDBSCAN`` at hunk line 165) → instead 4 of the 5 removed lines are trailing-whitespace-only cleanups in an unrelated section, complicating the diff review and mixing style-cleanup into a feature PR."
contract: Changes in inventory B must be accounted for by an item in inventory A (or the plan's own "novel changes" list). These edits touch a section of the file the plan does not enumerate.
instances: [doc/modules/clustering.rst:402, doc/modules/clustering.rst:415, doc/modules/clustering.rst:422, doc/modules/clustering.rst:429]

### F2 — Novel-change claim "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`" not applied to `_tree.pyx`
severity: low
evidence:
  - Plan of record `/work/review-harness/runs/26385/PLAN.md:15` explicitly lists this as one of the three novel changes included with the PR.
  - Verified in the worktree: `sklearn/cluster/_hdbscan/_linkage.pyx:42`, `_reachability.pyx:40`, and `_tree.pxd:30` all cimport typedefs (`from ...utils._typedefs cimport ...`). But `sklearn/cluster/_hdbscan/_tree.pyx` contains 90 occurrences of `cnp.*_t` (grep count) and has no `_typedefs` cimport — the file uses `cnp.float64_t`/`cnp.intp_t` throughout (e.g., diff `hunks/sklearn_cluster__hdbscan__tree.pyx.diff:45-46, 63-73, 90`).
scenario: "A reader audits the merged tree expecting the four HDBSCAN Cython files to share the uniform `_typedefs.pxd` typing convention the plan announces → because `_tree.pyx` retains the pre-modernization `cnp.*_t` typing, the 'novel change' description in the plan overstates what actually landed, leaving inconsistent typing conventions across sibling files in one module."
contract: Every requirement/promise in the plan (inventory A) must have a corresponding implementing change (inventory B). The novel-change item at PLAN.md:15 has no implementing change touching `_tree.pyx`, and no other file in the manifest converts it.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:45, sklearn/cluster/_hdbscan/_tree.pyx:46, sklearn/cluster/_hdbscan/_tree.pyx:63, sklearn/cluster/_hdbscan/_tree.pyx:90]
