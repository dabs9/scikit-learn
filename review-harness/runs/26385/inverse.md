### F1 — Plan says `cnp.*_t` typing was replaced with `*_t` from `_typedefs.pxd`, but many `cnp.*_t` usages remain in the changed files
severity: medium
evidence:
- Plan of record `/work/review-harness/runs/26385/PLAN.md:15` lists as a novel change #1: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`".
- Actual tree still uses `cnp.*_t` heavily: `sklearn/cluster/_hdbscan/_tree.pyx` has 90 `cnp.(intp_t|float64_t|uint8_t|int64_t|int32_t)` occurrences (e.g. `_tree.pyx:39` `cdef cnp.float64_t INFTY = np.inf`, `_tree.pyx:40` `cdef cnp.intp_t NOISE = -1`, `_tree.pyx:83`, `:90`, `:117`, `:119`, ... through `:700`).
- Beyond ndarray dtype slots, standalone scalar typings such as `cnp.intp_t bfs_root`, `cnp.intp_t node`, `cnp.float64_t lambda_value` remain in `_tree.pyx` even though `_tree.pxd` (added in this PR at `sklearn/cluster/_hdbscan/_tree.pxd:36`) already `cimport`s `intp_t, float64_t, uint8_t` from `..._typedefs`. The sibling files `_linkage.pyx` and `_reachability.pyx` do use the unprefixed `intp_t`, `float64_t`, etc. — so the substitution was applied to those files but not to `_tree.pyx`.
scenario: "Reviewer trusts the plan's novel-changes inventory that `cnp.*_t` → `*_t` is done → skips re-reviewing typings in `_tree.pyx` → the incomplete substitution ships unnoticed for the largest changed Cython module."
contract: Plan claims to describe the deltas this PR actually introduces on top of the already-reviewed sub-PRs; readers rely on that inventory being accurate for their diff-review budget.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:117, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:700]

### F2 — Plan says `*.shape[0]` was replaced with `len(*)` for ndarray objects, but many `.shape[0]` calls on ndarray objects remain
severity: low
evidence:
- Plan `/work/review-harness/runs/26385/PLAN.md:16` novel change #2: "Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects".
- Grep on the changed tree finds 20 residual `.shape[0]` occurrences across 4 changed files; several are clearly on ndarray-typed objects:
  - Python side: `sklearn/cluster/_hdbscan/hdbscan.py:223 if X.shape[0] != X.shape[1]:`; `:385 tree[tree.shape[0] - 1]…`; `:387 last_cluster_size = tree[tree.shape[0] - 1]["cluster_size"]`; `:752 if X.shape[0] == 1:`; `:758 if self._min_samples > X.shape[0]:`; `:840 np.empty(self._raw_data.shape[0], …)`; `:846 np.zeros(self._raw_data.shape[0], …)`; `:896 mask = np.empty((X.shape[0],), …)`.
  - Cython on `cnp.ndarray[…]`-declared objects: `_tree.pyx:90 hierarchy.shape[0] + 1`, `:145 2 * hierarchy.shape[0]`, `:146 hierarchy.shape[0] + 1`, `:396 2 * linkage.shape[0]`, `:531 labels.shape[0]`, `:563 children.shape[0] == 0`.
  - `_linkage.pyx:153 n_samples = raw_data.shape[0]` (raw_data is passed in as an ndarray).
scenario: "Reviewer trusts the plan's assertion that `.shape[0]` → `len(*)` is complete for ndarrays → does not re-check `.shape[0]` sites in the diff → the incomplete substitution ships with 15+ residual ndarray-operand call sites."
contract: See F1.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:223, sklearn/cluster/_hdbscan/hdbscan.py:385, sklearn/cluster/_hdbscan/hdbscan.py:387, sklearn/cluster/_hdbscan/hdbscan.py:752, sklearn/cluster/_hdbscan/hdbscan.py:758, sklearn/cluster/_hdbscan/hdbscan.py:840, sklearn/cluster/_hdbscan/hdbscan.py:846, sklearn/cluster/_hdbscan/hdbscan.py:896, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:396, sklearn/cluster/_hdbscan/_tree.pyx:531, sklearn/cluster/_hdbscan/_tree.pyx:563, sklearn/cluster/_hdbscan/_linkage.pyx:153]

### F3 — `_hierarchical_fast.pxd` (new) and edit to `_hierarchical_fast.pyx` are not sanctioned by any plan requirement
severity: low
evidence:
- Manifest `/work/review-harness/runs/26385/DIFF_MANIFEST.md:27-28`:
  - `A sklearn/cluster/_hierarchical_fast.pxd +9/-0`
  - `M sklearn/cluster/_hierarchical_fast.pyx +0/-5`
- The plan lists as novel changes only (1) `cnp.*_t` → `*_t`, (2) `.shape[0]` → `len`, (3) trim unused variables (`PLAN.md:14-17`). Promoting `UnionFind`'s cdef fields from the pyx to a new pxd is a public-API refactor of an unrelated module, not any of those three.
- Motivation is visible in the tree — `sklearn/cluster/_hdbscan/_linkage.pyx:39 from ...cluster._hierarchical_fast cimport UnionFind` requires `UnionFind` to have a pxd — but nothing in the plan of record sanctions this cross-module refactor.
scenario: "Reviewer uses the plan's novel-changes list as their diff-review inventory → never opens the `_hierarchical_fast.pxd`/`.pyx` hunks → an unlisted public-API refactor of an already-shipped module lands unreviewed."
contract: Plan is expected to enumerate every novel edit beyond the pre-reviewed sub-PRs; the `_hierarchical_fast.*` change is novel and unlisted.
instances: [sklearn/cluster/_hierarchical_fast.pxd:1, sklearn/cluster/_hierarchical_fast.pyx:5]

### F4 — Unrelated whitespace-only edits in `doc/modules/clustering.rst`'s Mean Shift section are not sanctioned by any plan requirement
severity: low
evidence:
- `hunks/doc_modules_clustering.rst.diff:19-45` shows trailing-whitespace strips on Mean Shift prose (lines 399, 419, 422, 428) inside the same PR that adds the HDBSCAN section (which is sanctioned by the overall "Add HDBSCAN" goal).
- Plan `PLAN.md:4` scope is "Add `HDBSCAN` as a new estimator" plus the three novel-change items at `PLAN.md:14-17`; drive-by whitespace fixes to Mean Shift documentation are outside both.
scenario: "Reviewer treats the plan as authoritative inventory of doc edits → skips the Mean Shift hunks assuming they are HDBSCAN-scope → unrelated whitespace churn on unrelated docs is bundled into the estimator PR without notice."
contract: See F3.
instances: [doc/modules/clustering.rst:399, doc/modules/clustering.rst:419, doc/modules/clustering.rst:422, doc/modules/clustering.rst:428]
