### F1 — `UnionFind.union`/`fast_find` declared `noexcept` in .pxd but not in .pyx
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, but sklearn/cluster/_hierarchical_fast.pyx:331 and :339 define these methods without the `noexcept` specifier. The `.pxd` (a header/boundary declaration) contradicts the corresponding implementation in `.pyx`.
scenario: "Cython compilation of `_hierarchical_fast.pyx` sees the pxd declaration and the implementation with mismatched exception specifications → build warnings/errors and, at minimum, contract drift where downstream consumers cimporting via the pxd expect noexcept but the implementation allows exception propagation."
contract: Add `noexcept` to both method definitions in `sklearn/cluster/_hierarchical_fast.pyx` so the pyx signatures match the pxd declarations exactly.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F2 — Cross-package Cython cimport creates coupling from `_hdbscan` back to sibling `_hierarchical_fast`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39 does `from ...cluster._hierarchical_fast cimport UnionFind`. `_hdbscan` is a nominally private subpackage of `cluster`, but its Cython build now depends on the compiled state of a sibling module (`_hierarchical_fast`) that lives one directory up. `_hierarchical_fast.pxd` was created specifically to expose `UnionFind`'s previously private fields (see hunks/sklearn_cluster__hierarchical_fast.pyx.diff removing three `cdef` field declarations from the .pyx).
scenario: "Any refactor of `_hierarchical_fast.UnionFind` (field layout, method sigs) silently breaks `_hdbscan/_linkage.pyx` → build/runtime failures across packages, and the two subpackages can no longer be reasoned about independently."
contract: `UnionFind` used across subpackages must be relocated to a shared location (e.g., `sklearn/utils/_union_find.pxd` or under `sklearn/cluster/_shared/`) so the dependency direction is subpackage → shared utility, not sibling → sibling.
instances: single-instance

### F3 — `_get_finite_row_indices` is a generic ndarray/sparse utility misplaced in the estimator module
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 defines `_get_finite_row_indices(matrix)` — a pure array/sparse helper that has no HDBSCAN-specific knowledge (it only computes finite-row indices for dense arrays or LIL/sparse matrices). It is used only once at line 731, but its logic (finite-row detection for sparse+dense) is exactly the kind of thing that belongs alongside `_assert_all_finite` and `_allclose_dense_sparse` in `sklearn/utils/validation.py`.
scenario: "Another estimator that needs finite-row filtering for sparse inputs → developer duplicates the same helper because it is buried in `_hdbscan.hdbscan`."
contract: Move `_get_finite_row_indices` to `sklearn/utils/validation.py` (or a nearby validation helper module) and cimport/import it from there.
instances: single-instance

### F4 — Public-named helper `remap_single_linkage_tree` defined in private `_hdbscan.hdbscan` module without underscore prefix
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 defines `def remap_single_linkage_tree(tree, internal_to_raw, non_finite)`. All sibling helpers in the same module use a leading underscore (`_brute_mst`, `_hdbscan_brute`, `_hdbscan_prims`, `_process_mst`, `_get_finite_row_indices`, `_weighted_cluster_center`). This helper is only used internally by `HDBSCAN.fit` at line 834; there is no reason for it to appear public.
scenario: "External code that scrapes `sklearn.cluster._hdbscan.hdbscan` sees `remap_single_linkage_tree` as an intended public helper and imports it → any refactor of its signature is a silent breaking change even though the module was meant to be private."
contract: Rename to `_remap_single_linkage_tree` to match the module's private-helper convention.
instances: single-instance

### F5 — Test files for `_hdbscan` are split across two directories in an inconsistent way
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 exists as a 64-line unit test file for the `_reachability` module, while sklearn/cluster/tests/test_hdbscan.py:1 (533 lines) holds the main HDBSCAN estimator tests. Every other clustering subpackage places all its tests in `sklearn/cluster/tests/` (see sklearn/cluster/tests/test_dbscan.py:1, sklearn/cluster/tests/test_optics.py:1, sklearn/cluster/tests/test_hierarchical.py:1, sklearn/cluster/tests/test_birch.py:1). Additionally, the file name is misspelled ("reachibility" instead of "reachability" — the module itself is named `_reachability.pyx`).
scenario: "Contributors looking for HDBSCAN tests only find the main file and miss the reachability suite; test discovery configuration relying on `sklearn/cluster/tests/` glob will not pick up the second suite → coverage gaps silently develop."
contract: Consolidate `test_reachibility.py` into `sklearn/cluster/tests/test_hdbscan_reachability.py` (correctly spelled), delete `sklearn/cluster/_hdbscan/tests/` including its `__init__.py`.
instances: [sklearn/cluster/_hdbscan/tests/__init__.py:1, sklearn/cluster/_hdbscan/tests/test_reachibility.py:1]

### F6 — `_brute_mst` constructs raw `MST_edge_dtype` records in Python module using private `np.core.records`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:128-131 builds MST records via `mst = np.core.records.fromarrays([rows, cols, sparse_min_spanning_tree.data], dtype=MST_edge_dtype)`. The dtype and layout are owned by `_linkage.pyx` (which defines both the `MST_edge_dtype` numpy dtype and the packed struct `MST_edge_t`), yet the construction path for the sparse case lives outside that module — logic far from its data. Additionally, `np.core.records` is a private numpy submodule (`np.core` is scheduled to become fully private in NumPy 2.x; already emits deprecation guidance).
scenario: "NumPy removes access to `np.core.records` → HDBSCAN's sparse-`precomputed` code path breaks at import/call time even though `_linkage.pyx` (owner of the dtype) uses a fully supported constructor."
contract: Add a helper (e.g., `_mst_from_sparse_edges(rows, cols, data)`) to `_hdbscan/_linkage.pyx` that returns an `MST_edge_dtype` ndarray via `np.rec.fromarrays` or a plain structured `np.empty(...) + assignment`, and call that from `hdbscan.py`. Do not touch `np.core.*` from Python code.
instances: single-instance

### F7 — `estimator_checks.py` common-checks layer hard-codes knowledge of a specific concrete estimator
severity: low
evidence: sklearn/utils/estimator_checks.py:777-778 adds `if name == "HDBSCAN": estimator.set_params(min_samples=1)`. `_set_checking_parameters` is a generic utility that is now growing a per-estimator special case (joining a small existing set such as `SpectralEmbedding`). The dependency direction is inverted: a utility in `sklearn.utils` knows about a specific estimator in `sklearn.cluster._hdbscan`.
scenario: "Every additional estimator that fails a common check adds another `if name == 'X'` branch → `_set_checking_parameters` becomes a scattered registry of estimator quirks rather than a generic helper."
contract: Estimator-specific check parameters should be advertised by the estimator itself (e.g., via a `_more_tags` entry such as `_xfail_checks` / `check_parameters`, or a class attribute), and `_set_checking_parameters` should read from that surface instead of switching on `name`.
instances: single-instance

### F8 — `np.infty` is a deprecated NumPy alias used by newly added Cython code [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:93 uses `np.full(n_samples, fill_value=np.infty, dtype=np.float64)` and line 159 uses `np.full(n_samples, fill_value=np.infty, dtype=np.float64)`. `np.infty` is a deprecated alias that NumPy 1.20+ warns about and NumPy 2.0 removes; canonical spelling is `np.inf`.
scenario: "Users on NumPy 2.0+ run HDBSCAN → AttributeError at import/first-call of `mst_from_mutual_reachability` and `mst_from_data_matrix`."
contract: Replace `np.infty` with `np.inf` at both call sites.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:93, sklearn/cluster/_hdbscan/_linkage.pyx:159]
