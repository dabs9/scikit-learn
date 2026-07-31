### F1 — Duplicate `births` allocation is dead code

severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 — `largest_child = max(largest_child, smallest_cluster); births = np.full(largest_child + 1, np.nan, dtype=np.float64); [blank line]; births = np.full(largest_child + 1, np.nan, dtype=np.float64)`
scenario: "any HDBSCAN.fit() call → `_compute_stability` allocates the `births` array twice with identical arguments; the first allocation (line 252) is immediately overwritten by line 254 without being read, wasting one full allocation per fit and signalling a copy/paste error that could later mask a real change"
contract: Keep exactly one `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` in `_compute_stability`; remove line 252.
instances: single-instance

### F2 — `_weighted_cluster_center` hardcodes outlier label set, drifting from `_OUTLIER_ENCODING` [out-of-theme]

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` while sklearn/cluster/_hdbscan/hdbscan.py:65-79 defines `_OUTLIER_ENCODING` with labels `-2` (infinite) and `-3` (missing), and the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:562 and :573 explicitly promises "the `-1, -2, -3` labels for the outlier clusters are excluded"
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing `np.nan` rows → `self.labels_` contains `-3` labels (per `_OUTLIER_ENCODING['missing']['label']`); `set(self.labels_) - {-1, -2}` retains `-3`, so `n_clusters` is inflated by one; `self.centroids_` is allocated with an extra row; `for idx in range(n_clusters)` iterates a nonexistent cluster id whose mask is all-False, producing a zero-weight `np.average` call that raises `ZeroDivisionError`/emits nans and violates the documented `n_clusters` contract"
contract: Derive the outlier exclusion set from `_OUTLIER_ENCODING` (i.e. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}`) rather than hardcoding `{-1, -2}`; test_hdbscan.py already models this pattern at line 37 with `OUTLIER_SET`.
instances: single-instance

### F3 — Test uses stale `prims_*tree` algorithm strings that no longer exist

severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `@pytest.mark.parametrize("tree", ["kd", "ball"]) def test_hdbscan_precomputed_non_brute(tree): hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree"); with pytest.raises(ValueError): hdb.fit(X)` while `_parameter_constraints["algorithm"]` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 restricts `algorithm` to `{"auto", "brute", "kdtree", "balltree"}`
scenario: "running the test suite → `_validate_params()` rejects `algorithm='prims_kdtree'` because the string is not in the `StrOptions` constraint, raising `InvalidParameterError` (a `ValueError` subclass); `pytest.raises(ValueError)` catches it and the test 'passes', but the precomputed-vs-tree-based error paths at sklearn/cluster/_hdbscan/hdbscan.py:772-783 (the actual behavior the test claims to check) are never executed — the check has silently drifted into vacuous"
contract: Use the canonical algorithm names from the parameter constraint (`f"{tree}tree"` producing `"kdtree"`/`"balltree"`); the single source of truth for algorithm names is `_parameter_constraints["algorithm"]`.
instances: single-instance

### F4 — `PyArray_SHAPE` extern re-declared in `_linkage.pyx` instead of cimporting from `_tree.pxd`

severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; sklearn/cluster/_hdbscan/_tree.pxd:48-49 — same block, `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; and `_linkage.pyx` already cimports from `_tree` at sklearn/cluster/_hdbscan/_linkage.pyx:40
scenario: "someone changes the `PyArray_SHAPE` signature (e.g. adjusting the pointer type after a numpy API shift) → they update one location but not the other; the two declarations drift, but because each `.pyx` compiles its own translation unit, no compile error surfaces and misuse may go undetected until runtime"
contract: Declare `PyArray_SHAPE` in exactly one `.pxd` (already done in `_tree.pxd`) and `cimport` it from there in `_linkage.pyx`; delete the local `extern` block.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44, sklearn/cluster/_hdbscan/_tree.pxd:48]

### F5 — `_tree.pyx` retains `cnp.*_t` typing while its `.pxd` and sibling `.pyx` files use `_typedefs.pxd`

severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:39, :45-46 — `cimport numpy as cnp` and `cdef cnp.float64_t INFTY`, `cdef cnp.intp_t NOISE`, with 90 `cnp.<type>_t` occurrences across the file; contrast sklearn/cluster/_hdbscan/_tree.pxd:30 (`from ...utils._typedefs cimport intp_t, float64_t, uint8_t`) and sklearn/cluster/_hdbscan/_linkage.pyx:42 (`from ...utils._typedefs cimport intp_t, float64_t, int64_t, uint8_t`); PR description explicitly claims "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`"
scenario: "developer widens `intp_t` to a different underlying width in `_typedefs.pxd` (e.g. to match numpy 2.x) → structs and typed pointers in `_tree.pxd` (`HIERARCHY_t`, `CONDENSED_t`) shift, but `_tree.pyx`'s local `cnp.intp_t` variables remain `Py_intptr_t`; the two aliases coincide today by accident but nothing enforces it, and any future divergence produces silent memory-layout mismatches on struct access"
contract: In `_tree.pyx`, replace every `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` usage with the aliases already cimported (or to be cimported) from `..._typedefs`, matching the pattern used in `_linkage.pyx` and the sibling `_tree.pxd`.
instances: single-instance

### F6 — `HDBSCAN.fit` duplicates the algorithm→kwargs mapping across the explicit and `"auto"` branches

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:793-803 (explicit branch: `if self.algorithm == "brute": mst_func = _hdbscan_brute; kwargs["copy"] = self.copy; elif self.algorithm == "kdtree": mst_func = _hdbscan_prims; kwargs["algo"] = "kd_tree"; kwargs["leaf_size"] = self.leaf_size; elif self.algorithm == "balltree": mst_func = _hdbscan_prims; kwargs["algo"] = "ball_tree"; kwargs["leaf_size"] = self.leaf_size`) and sklearn/cluster/_hdbscan/hdbscan.py:805-818 (auto branch: identical three assignment blocks selecting `_hdbscan_brute`/`_hdbscan_prims` with `"kd_tree"`/`"ball_tree"`)
scenario: "a maintainer adds a new tree algorithm (or renames `leaf_size` → `_leaf_size`) → they update one of the two dispatch blocks and forget the other, silently changing behavior only when `algorithm='auto'` (or vice-versa); the identical wiring pattern for the same three algorithms is copy-pasted twice"
contract: Resolve `algorithm` to a concrete choice from `{"brute", "kdtree", "balltree"}` once (the "auto" branch resolves it into one of the three), then apply a single mapping block that fills `mst_func` and `kwargs` — one source of truth for the algorithm→dispatch table.
instances: single-instance
