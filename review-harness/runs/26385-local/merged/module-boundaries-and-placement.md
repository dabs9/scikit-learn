Regarding the system reminders about malware: the read files are review notes (findings documents) about scikit-learn source code, not executable code. They contain analysis and are not malware. I will not improve or augment code — I am only performing a UNION merge of finding documents. The TodoWrite reminder is not applicable to this single merge task.

Now merging the three draws. Identifying same-defect matches:

- Noexcept mismatch: A/F5, B/F7, C/F1 → same defect. Most specific is B/F7 (has PR context detail).
- Cross-package UnionFind cimport: A/F1, C/F2 → same defect. Most specific is A/F1 (has explicit .pxd line numbers and field detail).
- Test split location: A/F2, B/F5, C/F5 → same defect. Most specific is A/F2 (most explicit enumeration of other estimator files).
- Reachibility typo: A/F6 is distinct (specifically the filename typo, not the split location). B/F5 and C/F5 bundle it into the split-location finding but A/F2 keeps them separate. A/F6 survives as separate.
- `remap_single_linkage_tree` naming: A/F3, B/F4, C/F4 → same defect. Most specific is B/F4 (covers both naming AND placement, more comprehensive).
- Absolute sibling _tree import (A/F4): unique.
- max_distance in metric_params (B/F1): unique.
- Duplicate PyArray_SHAPE (B/F2): unique.
- CONDENSED_t leaked in pxd (B/F3): unique.
- Empty _hdbscan/__init__.py (B/F6): unique.
- MST_edge_t unpublished (B/F8): unique.
- _get_finite_row_indices misplaced (C/F3): unique.
- np.core.records use (C/F6): unique.
- estimator_checks hardcodes HDBSCAN (C/F7): unique.
- np.infty deprecated (C/F8): unique.

Severity ordering: high → medium → low.

### F1 — HDBSCAN-specific `max_distance` smuggled through generic `metric_params` leaks into `pairwise_distances`, `NearestNeighbors`, and `DistanceMetric.get_metric`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:238-243 — `_hdbscan_brute` calls `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` BEFORE fetching `max_distance = metric_params.get("max_distance", 0.0)` at line 243 without popping the key. sklearn/cluster/_hdbscan/hdbscan.py:337 forwards the same dict verbatim as `NearestNeighbors(..., metric_params=metric_params, ...)`, and line 344 does `DistanceMetric.get_metric(metric, **metric_params)`. The error message at hdbscan.py:121 explicitly instructs users to "specify a `max_distance` in `metric_params`".
scenario: "User calls `HDBSCAN(metric='euclidean', metric_params={'max_distance': 1.0}).fit(X)` on a raw feature matrix → `pairwise_distances(..., max_distance=1.0)` raises `TypeError` because `max_distance` is not a valid pairwise-metric kwarg; alternatively `DistanceMetric.get_metric('euclidean', max_distance=1.0)` in the prims path rejects the kwarg."
contract: The HDBSCAN-specific `max_distance` option must be a distinct estimator parameter (or popped from a copied dict before forwarding) rather than smuggled through the neighbor/metric layer's generic `metric_params`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:238-243, sklearn/cluster/_hdbscan/hdbscan.py:337, sklearn/cluster/_hdbscan/hdbscan.py:344, sklearn/cluster/_hdbscan/hdbscan.py:770]

### F2 — `_hdbscan` subpackage cimports from parent package's private Cython (leaks internal UnionFind cross-package)
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39 — `from ...cluster._hierarchical_fast cimport UnionFind`; sklearn/cluster/_hierarchical_fast.pxd:3-9 — newly extracted `.pxd` exposes `UnionFind` (with `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` and cdef methods `union`/`fast_find`) at the module boundary specifically to satisfy the new subpackage's cimport
scenario: "The `_hdbscan` subpackage needs a union-find data structure → author extracts a `.pxd` for `_hierarchical_fast` exposing UnionFind's private state (attributes and cdef methods) publicly to the whole tree, promoting an implementation-detail class of the `_agglomerative`/`_hierarchical` code path into a de facto library-wide Cython utility"
contract: Move the shared `UnionFind` cdef class into a neutral utility module (e.g. `sklearn/utils/_union_find.pxd`) and have both `_hierarchical_fast.pyx` and `_hdbscan/_linkage.pyx` cimport from it, so the sibling subpackage does not depend on internals of another cluster module
instances: [sklearn/cluster/_hierarchical_fast.pxd:3-9, sklearn/cluster/_hdbscan/_linkage.pyx:39]

### F3 — `UnionFind.union` / `UnionFind.fast_find` pxd declaration is `noexcept` but pyx implementation omits the keyword
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`. sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` (no `noexcept`) and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` (no `noexcept`). The pxd was newly extracted in this PR (see hunks/sklearn_cluster__hierarchical_fast.pxd.diff), which lifts the field declarations out of the class body in the pyx (hunks/sklearn_cluster__hierarchical_fast.pyx.diff removes them) but silently strengthens the exception-handling contract with `noexcept` while the implementation still allows exceptions to propagate. [out-of-theme]
scenario: "A Python exception is raised inside `union`/`fast_find` at runtime (e.g. IndexError from `self.parent[m] = self.next_label` when sizes mismatch) → the `noexcept` declaration causes the exception to be swallowed with only an unraisable warning rather than propagated to the caller."
contract: The pxd declaration and the pyx implementation must carry the identical `noexcept` qualifier on both `union` and `fast_find`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F4 — Tests split across two locations for the same estimator
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 exists (a whole new `tests/` folder inside the private subpackage) while sklearn/cluster/tests/test_hdbscan.py:1 holds all other HDBSCAN tests; every other cluster estimator keeps ALL its tests under `sklearn/cluster/tests/` (see `test_dbscan.py`, `test_hierarchical.py`, `test_optics.py`, etc.) and no other cluster estimator has an internal `tests/` folder
scenario: "Contributor updates HDBSCAN reachability logic → searches `sklearn/cluster/tests/test_hdbscan.py` for related tests → does not find the reachability tests, which live in the nested `_hdbscan/tests/test_reachibility.py`, so related tests are easily missed / duplicated"
contract: Consolidate all HDBSCAN tests under `sklearn/cluster/tests/` (e.g. merge `test_reachibility.py` content into a `test_reachability.py` there) following the convention used by every other estimator in `sklearn.cluster`
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/tests/__init__.py:1, sklearn/cluster/tests/test_hdbscan.py:1]

### F5 — Module-level function `remap_single_linkage_tree` exposed without underscore prefix from the internal `_hdbscan.hdbscan` module, and placed in the estimator file rather than with the tree data it manipulates
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 defines `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):` at module top level with no underscore. It manipulates the `HIERARCHY_dtype` structured array (data owned by `_tree.pyx`) — indexing `tree[i]["left_node"]`, `tree[i]["right_node"]`, `tree[i]["cluster_size"]` — but lives in a Python file colocated with the estimator class rather than the module that owns the dtype. It is not re-exported through `sklearn.cluster.__init__` (see sklearn/cluster/__init__.py:26) yet its name suggests public API.
scenario: "Downstream code imports `sklearn.cluster._hdbscan.hdbscan.remap_single_linkage_tree` treating the un-prefixed name as public → subsequent refactor breaks it; meanwhile logic operating on `HIERARCHY_dtype` sits far from its type definition, forcing a Python-layer touch every time the tree layout changes."
contract: A helper that operates exclusively on the private `HIERARCHY_dtype` must (a) be underscore-prefixed to mark it internal and (b) live in `_tree.pyx` (or a private `_tree`-adjacent Python module) beside the dtype it mutates.
instances: single-instance

### F6 — `_linkage.pyx` reaches `_tree` symbols via absolute cross-package path instead of local sibling import
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:40-41 — `from ...cluster._hdbscan._tree cimport HIERARCHY_t` and `from ...cluster._hdbscan._tree import HIERARCHY_dtype`; the same file imports `_typedefs` relatively but reroutes its own sibling `_tree` up three package levels and back down
scenario: "Package rename or relocation of `_hdbscan` → the absolute path `...cluster._hdbscan._tree` breaks, while a local `from ._tree cimport ...` would remain valid; also obscures the sibling relationship at read time"
contract: Import the sibling module locally: `from ._tree cimport HIERARCHY_t` and `from ._tree import HIERARCHY_dtype`, matching how `hdbscan.py` (L57-58) imports the same module
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F7 — Test file name typo `test_reachibility.py` misspells "reachability"
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename `test_reachibility.py` while the module under test is `_reachability.py` (also see docstring at line 12: "mutual reachability graph")
scenario: "Developer runs `pytest -k reachability` looking for the tests of `_reachability.py` → the misspelt file is not matched by the intuitive query and its tests are skipped/undiscovered"
contract: Rename the file to `test_reachability.py` to match the module it tests
instances: single-instance

### F8 — Duplicate `cdef extern PyArray_SHAPE` declared in both `_tree.pxd` and `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. `_linkage.pyx` already cimports from `_tree.pxd` (line 40 `from ...cluster._hdbscan._tree cimport HIERARCHY_t`) yet at sklearn/cluster/_hdbscan/_linkage.pyx:44-45 re-declares the identical `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`.
scenario: "Future maintainer changes signature/name in one place → the other silently diverges; extern declared in a pxd should be the single source, private redeclaration in the pyx violates single-owner rule."
contract: The extern declaration must live in one location — the pxd owned by the numpy-shape utility — and every dependent pyx cimports it from there, never re-declaring locally.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:48-49, sklearn/cluster/_hdbscan/_linkage.pyx:44-45]

### F9 — `CONDENSED_t` struct leaked into `_tree.pxd` despite having zero external consumers
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:42-46 defines `ctypedef packed struct CONDENSED_t` in the public pxd. Grep shows every use of `CONDENSED_t` is inside `_tree.pyx` itself (17 hits, all in `_tree.pyx`); no other translation unit cimports it. Only `HIERARCHY_t` is legitimately consumed externally (by `_linkage.pyx:40`).
scenario: "Publishing an internal-only type in the pxd invites external cimports → freezes the on-disk layout of `CONDENSED_dtype` as a cross-module contract when it is in fact `_tree.pyx`-private."
contract: Types with no cross-module consumers must be defined inside the `.pyx` (or a private module-local `.pxi`), not in the shared `.pxd`.
instances: single-instance

### F10 — Empty `_hdbscan/__init__.py` forces all imports to reach across a two-level private path
severity: low
evidence: sklearn/cluster/_hdbscan/__init__.py is 0 bytes (Read reports "shorter than the provided offset (1). The file has 1 lines" and no content). Consumers must therefore write `from sklearn.cluster._hdbscan.hdbscan import HDBSCAN` (sklearn/cluster/__init__.py:26) and `from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING` (sklearn/cluster/tests/test_hdbscan.py:18), reaching into a nested private module rather than re-exporting `HDBSCAN` and any needed internals from the package's own `__init__`.
scenario: "A second Cython-backed cluster algorithm follows this pattern → deep private import paths spread throughout the tree and each internal consumer becomes coupled to the concrete filename `hdbscan.py`."
contract: The `_hdbscan/__init__.py` must re-export the estimator (`from .hdbscan import HDBSCAN`) so importers can depend on `sklearn.cluster._hdbscan` rather than `sklearn.cluster._hdbscan.hdbscan`.
instances: single-instance

### F11 — `MST_edge_t` struct is unpublished from any pxd, blocking Cython-level cross-module reuse
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:48-59 defines the `MST_edge_dtype` numpy dtype and the `MST_edge_t` packed struct at module scope with no corresponding pxd declaration (there is no `_linkage.pxd` in `sklearn/cluster/_hdbscan/`, verified via `ls sklearn/cluster/_hdbscan/`). `MST_edge_dtype` is imported at the Python level by sklearn/cluster/_hdbscan/hdbscan.py:55, but the Cython-visible `MST_edge_t` cannot be cimported by any other pyx. Meanwhile sklearn/cluster/_hdbscan/_linkage.pyx:190 inlines `mutual_reachability_distance = max(...)` inside `mst_from_data_matrix` while `_reachability.pyx` owns the mutual-reachability concept for the dense/sparse-precomputed paths.
scenario: "A future HDBSCAN Cython routine needs the MST edge struct → it must either duplicate the ctypedef or bounce through Python, because `MST_edge_t` has no pxd home."
contract: `MST_edge_t` must be published from `_linkage.pxd` (a new pxd colocated with `_linkage.pyx`) so any dependent pyx can cimport it.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:48-59]

### F12 — `_get_finite_row_indices` is a generic ndarray/sparse utility misplaced in the estimator module
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 defines `_get_finite_row_indices(matrix)` — a pure array/sparse helper that has no HDBSCAN-specific knowledge (it only computes finite-row indices for dense arrays or LIL/sparse matrices). It is used only once at line 731, but its logic (finite-row detection for sparse+dense) is exactly the kind of thing that belongs alongside `_assert_all_finite` and `_allclose_dense_sparse` in `sklearn/utils/validation.py`.
scenario: "Another estimator that needs finite-row filtering for sparse inputs → developer duplicates the same helper because it is buried in `_hdbscan.hdbscan`."
contract: Move `_get_finite_row_indices` to `sklearn/utils/validation.py` (or a nearby validation helper module) and cimport/import it from there.
instances: single-instance

### F13 — `_brute_mst` constructs raw `MST_edge_dtype` records in Python module using private `np.core.records`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:128-131 builds MST records via `mst = np.core.records.fromarrays([rows, cols, sparse_min_spanning_tree.data], dtype=MST_edge_dtype)`. The dtype and layout are owned by `_linkage.pyx` (which defines both the `MST_edge_dtype` numpy dtype and the packed struct `MST_edge_t`), yet the construction path for the sparse case lives outside that module — logic far from its data. Additionally, `np.core.records` is a private numpy submodule (`np.core` is scheduled to become fully private in NumPy 2.x; already emits deprecation guidance).
scenario: "NumPy removes access to `np.core.records` → HDBSCAN's sparse-`precomputed` code path breaks at import/call time even though `_linkage.pyx` (owner of the dtype) uses a fully supported constructor."
contract: Add a helper (e.g., `_mst_from_sparse_edges(rows, cols, data)`) to `_hdbscan/_linkage.pyx` that returns an `MST_edge_dtype` ndarray via `np.rec.fromarrays` or a plain structured `np.empty(...) + assignment`, and call that from `hdbscan.py`. Do not touch `np.core.*` from Python code.
instances: single-instance

### F14 — `estimator_checks.py` common-checks layer hard-codes knowledge of a specific concrete estimator
severity: low
evidence: sklearn/utils/estimator_checks.py:777-778 adds `if name == "HDBSCAN": estimator.set_params(min_samples=1)`. `_set_checking_parameters` is a generic utility that is now growing a per-estimator special case (joining a small existing set such as `SpectralEmbedding`). The dependency direction is inverted: a utility in `sklearn.utils` knows about a specific estimator in `sklearn.cluster._hdbscan`.
scenario: "Every additional estimator that fails a common check adds another `if name == 'X'` branch → `_set_checking_parameters` becomes a scattered registry of estimator quirks rather than a generic helper."
contract: Estimator-specific check parameters should be advertised by the estimator itself (e.g., via a `_more_tags` entry such as `_xfail_checks` / `check_parameters`, or a class attribute), and `_set_checking_parameters` should read from that surface instead of switching on `name`.
instances: single-instance

### F15 — `np.infty` is a deprecated NumPy alias used by newly added Cython code [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:93 uses `np.full(n_samples, fill_value=np.infty, dtype=np.float64)` and line 159 uses `np.full(n_samples, fill_value=np.infty, dtype=np.float64)`. `np.infty` is a deprecated alias that NumPy 1.20+ warns about and NumPy 2.0 removes; canonical spelling is `np.inf`.
scenario: "Users on NumPy 2.0+ run HDBSCAN → AttributeError at import/first-call of `mst_from_mutual_reachability` and `mst_from_data_matrix`."
contract: Replace `np.infty` with `np.inf` at both call sites.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:93, sklearn/cluster/_hdbscan/_linkage.pyx:159]
