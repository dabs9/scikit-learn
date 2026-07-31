### F1 — `_weighted_cluster_center` hardcodes outlier label set, drifting from `_OUTLIER_ENCODING` [out-of-theme]

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` while sklearn/cluster/_hdbscan/hdbscan.py:65-79 defines `_OUTLIER_ENCODING` with labels `-2` (infinite) and `-3` (missing), and the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:562 and :573 explicitly promises "the `-1, -2, -3` labels for the outlier clusters are excluded"
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing `np.nan` rows → `self.labels_` contains `-3` labels (per `_OUTLIER_ENCODING['missing']['label']`); `set(self.labels_) - {-1, -2}` retains `-3`, so `n_clusters` is inflated by one; `self.centroids_` is allocated with an extra row; `for idx in range(n_clusters)` iterates a nonexistent cluster id whose mask is all-False, producing a zero-weight `np.average` call that raises `ZeroDivisionError`/emits nans and violates the documented `n_clusters` contract"
contract: Derive the outlier exclusion set from `_OUTLIER_ENCODING` (i.e. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}`) rather than hardcoding `{-1, -2}`; test_hdbscan.py already models this pattern at line 37 with `OUTLIER_SET`.
instances: single-instance

### F2 — `store_centers` + non-finite input crashes because `X` is finite-filtered but `self.labels_` is raw-sized [out-of-theme]

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 reduces `X = X[finite_index]`; hdbscan.py:840-844 rebuilds `self.labels_` at `self._raw_data.shape[0]` (full raw length); hdbscan.py:854-855 then calls `self._weighted_cluster_center(X)` with the still-reduced `X`; inside `_weighted_cluster_center` (hdbscan.py:908-909) `mask = self.labels_ == idx` produces a boolean array whose length is the raw sample count, and `data = X[mask]` fails when that length differs from `X.shape[0]`.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_row)` → `X` shape mismatch with `mask` → `IndexError: boolean index did not match indexed array along dimension 0; dimension is <finite_n> but corresponding boolean dimension is <raw_n>`."
contract: Inside `_weighted_cluster_center`, index `self.labels_` by `self._finite_index` before comparing (`labels = self.labels_[self._finite_index]; mask = labels == idx`) so the mask and the finite-filtered `X` share a length.
instances: single-instance

### F3 — Duplicate `births` allocation is dead code

severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 — `largest_child = max(largest_child, smallest_cluster); births = np.full(largest_child + 1, np.nan, dtype=np.float64); [blank line]; births = np.full(largest_child + 1, np.nan, dtype=np.float64)`
scenario: "any HDBSCAN.fit() call → `_compute_stability` allocates the `births` array twice with identical arguments; the first allocation (line 252) is immediately overwritten by line 254 without being read, wasting one full allocation per fit and signalling a copy/paste error that could later mask a real change"
contract: Keep exactly one `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` in `_compute_stability`; remove line 252.
instances: single-instance

### F4 — Test uses stale `prims_*tree` algorithm strings that no longer exist

severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `@pytest.mark.parametrize("tree", ["kd", "ball"]) def test_hdbscan_precomputed_non_brute(tree): hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree"); with pytest.raises(ValueError): hdb.fit(X)` while `_parameter_constraints["algorithm"]` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 restricts `algorithm` to `{"auto", "brute", "kdtree", "balltree"}`
scenario: "running the test suite → `_validate_params()` rejects `algorithm='prims_kdtree'` because the string is not in the `StrOptions` constraint, raising `InvalidParameterError` (a `ValueError` subclass); `pytest.raises(ValueError)` catches it and the test 'passes', but the precomputed-vs-tree-based error paths at sklearn/cluster/_hdbscan/hdbscan.py:772-783 (the actual behavior the test claims to check) are never executed — the check has silently drifted into vacuous"
contract: Use the canonical algorithm names from the parameter constraint (`f"{tree}tree"` producing `"kdtree"`/`"balltree"`); the single source of truth for algorithm names is `_parameter_constraints["algorithm"]`.
instances: single-instance

### F5 — Documented `n_jobs` default (`None`) contradicts the constructor default (`4`)

severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring: `n_jobs : int, default=None` with `` `None` means 1 unless in a :obj:`joblib.parallel_backend` context. `` sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`.
scenario: "A user reads the documented default and expects single-threaded behavior in a serial environment (or under `parallel_backend`) → HDBSCAN silently spawns 4 workers, breaking pinned CPU budgets and the `joblib.parallel_backend` contract that other sklearn estimators honor."
contract: Change `__init__` to `n_jobs=None,` so the value matches the docstring and the sklearn-wide `n_jobs` convention.
instances: single-instance

### F6 — `PyArray_SHAPE` extern re-declared in `_linkage.pyx` instead of cimporting from `_tree.pxd`

severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; sklearn/cluster/_hdbscan/_tree.pxd:48-49 — same block, `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; and `_linkage.pyx` already cimports from `_tree` at sklearn/cluster/_hdbscan/_linkage.pyx:40
scenario: "someone changes the `PyArray_SHAPE` signature (e.g. adjusting the pointer type after a numpy API shift) → they update one location but not the other; the two declarations drift, but because each `.pyx` compiles its own translation unit, no compile error surfaces and misuse may go undetected until runtime"
contract: Declare `PyArray_SHAPE` in exactly one `.pxd` (already done in `_tree.pxd`) and `cimport` it from there in `_linkage.pyx`; delete the local `extern` block.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44, sklearn/cluster/_hdbscan/_tree.pxd:48]

### F7 — Cython typedefs inconsistently sourced across the new `_hdbscan` module

severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx uses `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` 90 times (e.g. lines 39, 40, 58, 61, 66, 67, 90, 145-155), whereas sklearn/cluster/_hdbscan/_linkage.pyx and _reachability.pyx cimport `intp_t`, `float64_t`, `int64_t`, `uint8_t` directly from `...utils._typedefs` (_linkage.pyx:42, _reachability.pyx:40) and use them exclusively. The plan of record explicitly claims "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`", but `_tree.pyx` was not migrated.
scenario: "Future refactor of `_typedefs.pxd` (e.g. shifting `intp_t` from `np.intp` to another width) → the two files diverge in their storage/API and the mixed usage inside the same module makes the drift harder to spot."
contract: Replace every `cnp.<type>_t` in `_tree.pyx` with the corresponding typedef cimported from `...utils._typedefs`, matching the style of `_linkage.pyx`/`_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:61, sklearn/cluster/_hdbscan/_tree.pyx:66, sklearn/cluster/_hdbscan/_tree.pyx:67, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:91, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:147, sklearn/cluster/_hdbscan/_tree.pyx:150, sklearn/cluster/_hdbscan/_tree.pyx:151, sklearn/cluster/_hdbscan/_tree.pyx:153, sklearn/cluster/_hdbscan/_tree.pyx:154, sklearn/cluster/_hdbscan/_tree.pyx:155, sklearn/cluster/_hdbscan/_tree.pyx:182, sklearn/cluster/_hdbscan/_tree.pyx:240, sklearn/cluster/_hdbscan/_tree.pyx:241, sklearn/cluster/_hdbscan/_tree.pyx:243, sklearn/cluster/_hdbscan/_tree.pyx:244, sklearn/cluster/_hdbscan/_tree.pyx:246, sklearn/cluster/_hdbscan/_tree.pyx:247, sklearn/cluster/_hdbscan/_tree.pyx:248, sklearn/cluster/_hdbscan/_tree.pyx:249, sklearn/cluster/_hdbscan/_tree.pyx:281, sklearn/cluster/_hdbscan/_tree.pyx:286, sklearn/cluster/_hdbscan/_tree.pyx:289]

### F8 — `HDBSCAN.fit` duplicates the algorithm→kwargs mapping across the explicit and `"auto"` branches

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:793-803 (explicit branch: `if self.algorithm == "brute": mst_func = _hdbscan_brute; kwargs["copy"] = self.copy; elif self.algorithm == "kdtree": mst_func = _hdbscan_prims; kwargs["algo"] = "kd_tree"; kwargs["leaf_size"] = self.leaf_size; elif self.algorithm == "balltree": mst_func = _hdbscan_prims; kwargs["algo"] = "ball_tree"; kwargs["leaf_size"] = self.leaf_size`) and sklearn/cluster/_hdbscan/hdbscan.py:805-818 (auto branch: identical three assignment blocks selecting `_hdbscan_brute`/`_hdbscan_prims` with `"kd_tree"`/`"ball_tree"`)
scenario: "a maintainer adds a new tree algorithm (or renames `leaf_size` → `_leaf_size`) → they update one of the two dispatch blocks and forget the other, silently changing behavior only when `algorithm='auto'` (or vice-versa); the identical wiring pattern for the same three algorithms is copy-pasted twice"
contract: Resolve `algorithm` to a concrete choice from `{"brute", "kdtree", "balltree"}` once (the "auto" branch resolves it into one of the three), then apply a single mapping block that fills `mst_func` and `kwargs` — one source of truth for the algorithm→dispatch table.
instances: single-instance

### F9 — Comment claims label mapping (`inf → -1`, `nan → -2`) inconsistent with `_OUTLIER_ENCODING`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2"; the code immediately below (lines 842-843) actually maps `inf → -2` and `nan → -3` via `_OUTLIER_ENCODING`.
scenario: "future maintainer trusts the comment and adjusts downstream label handling to match `-1/-2` instead of `-2/-3` → introduces a semantic regression"
contract: rewrite the comment to state the actual mapping (`inf → -2`, `nan → -3`) and reference `_OUTLIER_ENCODING` as the source of truth.
instances: single-instance

### F10 — `_hdbscan_prims` docstring documents a nonexistent `copy` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature has no `copy` parameter; sklearn/cluster/_hdbscan/hdbscan.py:313-318 — docstring still contains the `copy : bool, default=False ...` block copy-pasted from `_hdbscan_brute` (lines 207-212).
scenario: "user reads the `_hdbscan_prims` docstring and calls it with `copy=True` → `TypeError: _hdbscan_prims() got an unexpected keyword argument 'copy'`, or (more commonly) reader is misled about the function's contract"
contract: delete the `copy` parameter block from `_hdbscan_prims`'s docstring; the single source of truth for `copy` is `_hdbscan_brute` (and the public `HDBSCAN.copy`).
instances: single-instance

### F11 — `_hdbscan_brute` / `_hdbscan_prims` docstring defaults drift from actual signatures
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160-161 — actual defaults `min_samples=5, alpha=None`; docstring 179 and 183 say `min_samples : int, default=None` and `alpha : float, default=1.0`. Same drift in `_hdbscan_prims`: sklearn/cluster/_hdbscan/hdbscan.py:272 has `min_samples=5` while sklearn/cluster/_hdbscan/hdbscan.py:290 says `default=None`.
scenario: "reader/maintainer takes the doc-stated defaults at face value and calls `_hdbscan_brute(X)` expecting `alpha=1.0` → gets `alpha=None`, which then produces a `TypeError` on the `distance_matrix /= alpha` line (241)"
contract: reconcile each docstring `default=...` clause with the actual signature default, using the signature as the source of truth.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:161, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:272, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F12 — `plot_hdbscan.py` scale-invariance loop does not scale `X`, contradicting the paired DBSCAN loop
severity: low
evidence: examples/cluster/plot_hdbscan.py:87-89 — DBSCAN demo scales input as `dbs.fit(X * scale)`; examples/cluster/plot_hdbscan.py:106-110 — mirroring HDBSCAN demo calls `hdb.fit(X)` instead of `hdb.fit(X * scale)`, then labels the plot with the (unused) scale value.
scenario: "reader executes the notebook expecting three visibly different clusterings at scales `(1, 0.5, 3)` demonstrating HDBSCAN's scale-invariance → sees three identical panels (all fit on unscaled X), and the pedagogical claim in surrounding prose is not actually demonstrated"
contract: replace `hdb.fit(X)` with `hdb.fit(X * scale)` (and pass `X * scale` into `plot(...)`) to mirror the DBSCAN loop and single-source the scaling behavior across the two paired demos.
instances: single-instance

### F13 — Dead `mask = np.empty(...)` allocation immediately overwritten [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)` is assigned before any use, and the very first statement in the loop at line 908 reassigns it as `mask = self.labels_ == idx`; nothing between lines 896 and 908 reads it.
scenario: "Every `store_centers` fit reaches line 896 → an unused bool array of length `n_samples` is allocated and immediately overwritten, wasting `n_samples` bytes and one heap allocation per fit."
contract: Delete the dead allocation on line 896.
instances: single-instance
