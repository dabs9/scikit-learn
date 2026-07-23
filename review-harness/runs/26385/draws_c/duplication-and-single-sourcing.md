### F1 — Hard-coded outlier label set `{-1, -2}` drifts from `_OUTLIER_ENCODING` (drops `-3`)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` hard-codes the noise/infinite labels while the module-level `_OUTLIER_ENCODING` (lines 65-79) defines the outlier labels (infinite=-2, missing=-3). The docstring at hdbscan.py:561-562 and 573 explicitly promises: "the `-1, -2, -3` labels for the outlier clusters are excluded." Consumers such as `test_hdbscan.py:37` correctly derive their set as `{-1} | {out["label"] for _, out in _OUTLIER_ENCODING.items()}`.
scenario: "Fit HDBSCAN with `store_centers='centroid'` (or `'medoid'` / `'both'`) on data containing `np.nan` rows (label -3) → `n_clusters` at line 895 counts -3 as a real cluster; the `for idx in range(n_clusters)` loop at line 907 iterates one past the true number of clusters; the extra iteration masks zero rows and `np.average(empty, weights=empty, axis=0)` raises `ZeroDivisionError: Weights sum to zero`, breaking `fit`."
contract: Compute the excluded label set from `_OUTLIER_ENCODING` in a single place (e.g. `_OUTLIER_LABELS = {-1} | {enc["label"] for enc in _OUTLIER_ENCODING.values()}` at module scope) and reuse it here so `-3` is never dropped from the exclusion set.
instances: single-instance

### F2 — Duplicate assignment of `births` in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and :254 — identical statements `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` are executed back-to-back with no intervening use of the first result. This is a copy-paste artifact from the tree refactor; the first assignment is dead code.
scenario: "Every call into `_compute_stability` (every `fit`) allocates the `births` array twice and discards the first allocation → wasted allocation on the hot path and misleading source that suggests two distinct initializations."
contract: Delete the first `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252 so only the assignment at line 254 remains.
instances: single-instance

### F3 — `cdef extern PyArray_SHAPE` declared in two places [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`, and sklearn/cluster/_hdbscan/_linkage.pyx:44-45 re-declares the same extern block. Because the declaration is not `cimport`ed from a single header, two independent copies exist and can drift (e.g. if one is later updated to return a different pointer/type).
scenario: "A future maintainer edits the extern signature in one file (say switching return type to `Py_ssize_t*`) but forgets the other → silent type-cast bug in one translation unit that only surfaces at runtime on architectures where `intp_t` and `Py_ssize_t` differ, or subtle miscompilation."
contract: Declare `PyArray_SHAPE` in exactly one `.pxd` (e.g. keep it only in `_tree.pxd`) and `cimport` it from `_linkage.pyx` rather than re-declaring the extern locally.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:48-49, sklearn/cluster/_hdbscan/_linkage.pyx:44-45]
