PR states novel change #2: "Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects" — but many `.shape[0]` remain in the new `hdbscan.py` (lines 223, 385, 387, 752, 758, 761, 840, 846, 896). This is inconsistent single-sourcing / patchy application of the stated rule. However, the plan says the change was applied — not a strict guarantee to eliminate all. This is a mild finding.

Now I have enough evidence to compose findings.

### F1 — `PyArray_SHAPE` extern re-declared in `_linkage.pyx` despite being available via `_tree.pxd` cimport
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:40-45 — file `cimport`s from `..._hdbscan._tree` (which via `_tree.pxd:48-49` already declares `intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`) and then re-declares the same `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` at lines 44-45.
scenario: "future signature/return-type change to `PyArray_SHAPE` in `_tree.pxd` → `_linkage.pyx` keeps stale local declaration and compiles against a mismatched signature"
contract: Remove the local `cdef extern from "numpy/arrayobject.h"` block in `_linkage.pyx`; rely on the single declaration in `_tree.pxd` (making it available by cimport) or a shared utility pxd.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44-45, sklearn/cluster/_hdbscan/_tree.pxd:48-49]

### F2 — `_weighted_cluster_center` hardcodes outlier set `{-1, -2}` instead of deriving from `_OUTLIER_ENCODING`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The single source of truth `_OUTLIER_ENCODING` (line 65) defines outlier labels including `-3` (missing). Elsewhere in the file the code correctly derives from `_OUTLIER_ENCODING` (lines 842-843, 964-969), and the test suite uses `OUTLIER_SET = {-1} | {out["label"] for _, out in _OUTLIER_ENCODING.items()}` (test_hdbscan.py:37).
scenario: "user fits data containing `np.nan` rows with `store_centers=\"centroid\"` → `-3` label is treated as a real cluster, `n_clusters` is over-counted by 1, and `self.centroids_[idx]` for `idx` in `range(n_clusters)` skips real clusters / includes a bogus empty slot → subsequent iteration masks yield empty `data`, and `np.average` raises ZeroDivisionError or writes garbage."
contract: Replace the literal `{-1, -2}` with a set derived from `_OUTLIER_ENCODING` (e.g. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}`) so the outlier-label set has a single source.
instances: single-instance

### F3 — `n_jobs` default drifts between `__init__` and its docstring
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring `n_jobs : int, default=None`; sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature uses `n_jobs=4`. The `_brute_mst`/`_hdbscan_brute`/`_hdbscan_prims` helper docstrings (lines 197, 303) also state `n_jobs : int, default=None`.
scenario: "user reads docstring, expects default single-thread joblib behavior, but constructor silently uses 4 workers → unexpected parallelism, memory usage, and non-reproducible resource consumption in constrained environments"
contract: Pin the constructor default to `n_jobs=None` to match the documented single-sourced default across the file and other sklearn estimators.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:486, sklearn/cluster/_hdbscan/hdbscan.py:658]

### F4 — Cython entry-point defaults for `min_cluster_size` (10) drift from the `HDBSCAN` estimator default (5)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:58 (`tree_to_labels(..., cnp.intp_t min_cluster_size=10, ...)`) and sklearn/cluster/_hdbscan/_tree.pyx:119 (`_condense_tree(..., cnp.intp_t min_cluster_size=10)`) both default to 10; sklearn/cluster/_hdbscan/hdbscan.py:649 has `min_cluster_size=5` for the `HDBSCAN.__init__` and sklearn/cluster/_hdbscan/hdbscan.py:923 has `dbscan_clustering(..., min_cluster_size=5)`.
scenario: "downstream user (or test) calls `_condense_tree(tree)` directly relying on the same 5 default → gets a silently different pruning threshold than the public estimator produces"
contract: Align the Cython defaults to 5 (matching the public estimator) or drop the defaults entirely so callers must pass an explicit value from the single source (`HDBSCAN.min_cluster_size`).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/hdbscan.py:649, sklearn/cluster/_hdbscan/hdbscan.py:923]

### F5 — Comment about non-finite label remap contradicts `_OUTLIER_ENCODING`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2", but the actual code (lines 842-843) writes `_OUTLIER_ENCODING["infinite"]["label"]` (= -2) and `_OUTLIER_ENCODING["missing"]["label"]` (= -3). The correct mapping (per `_OUTLIER_ENCODING` at lines 65-79 and per the class docstring at lines 534-537) is inf→-2, nan→-3.
scenario: "future maintainer trusts the inline comment while refactoring the outlier scheme → introduces off-by-one label collision with the -1 noise label"
contract: Update the comment to reference `_OUTLIER_ENCODING` as the single source (or state the correct mapping: inf→-2, nan→-3).
instances: single-instance

### F6 — `births = np.full(...)` allocated twice in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and sklearn/cluster/_hdbscan/_tree.pyx:254 — the exact same statement `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears on two consecutive lines with no intervening use. The first allocation is dead code.
scenario: "reader/refactorer edits only one of the two identical lines → creates subtle divergence in initialization (e.g., different fill value) that silently corrupts stability computation"
contract: Delete the redundant first `births = np.full(...)` at line 252 so only one initialization exists.
instances: single-instance

### F7 — `.shape[0]` pattern retained in new `hdbscan.py` despite plan's stated cleanup
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:385,387,752,758,761,840,846,896 — PR description explicitly lists "Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects" as a novel change; but `hdbscan.py` still uses `X.shape[0]` / `tree.shape[0]` / `self._raw_data.shape[0]` at these lines for `ndarray` objects where `len()` would apply.
scenario: "codebase drifts between two idioms for row-count access → readers and contributors are unsure which is canonical for `ndarray`, and the stated cleanup is only half-applied"
contract: Apply the same `.shape[0]` → `len(...)` transformation to `ndarray` accesses in `hdbscan.py` (lines 385, 387, 752, 758, 761, 840, 846, 896) to complete the single-sourced convention declared by the plan.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:385, sklearn/cluster/_hdbscan/hdbscan.py:387, sklearn/cluster/_hdbscan/hdbscan.py:752, sklearn/cluster/_hdbscan/hdbscan.py:758, sklearn/cluster/_hdbscan/hdbscan.py:761, sklearn/cluster/_hdbscan/hdbscan.py:840, sklearn/cluster/_hdbscan/hdbscan.py:846, sklearn/cluster/_hdbscan/hdbscan.py:896]

### F8 — Test file uses invalid `algorithm` strings (`prims_kdtree`/`prims_balltree`) not in `_parameter_constraints` [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`; the estimator's `_parameter_constraints` (sklearn/cluster/_hdbscan/hdbscan.py:629-638) only accepts `{"auto", "brute", "kdtree", "balltree"}`. The test asserts a `ValueError` is raised, but the ValueError is raised by parameter validation ("The 'algorithm' parameter … must be a str among …"), not by the intended check "algorithm cannot be tree-based with precomputed metric".
scenario: "test purports to validate that precomputed data + tree algorithm raises → actually it validates that an invalid algorithm string raises; the real precomputed-vs-tree combination is never exercised"
contract: Replace the invalid algorithm strings with the valid ones (`kdtree`/`balltree`) so the test exercises the intended precomputed-vs-tree guard.
instances: single-instance
