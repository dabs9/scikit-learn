### F1 — `n_jobs` signature default (`4`) contradicts documented default (`None`)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the standard "`None` means 1 unless in a joblib.parallel_backend context" prose; sklearn/cluster/_hdbscan/hdbscan.py:658 declares `n_jobs=4` in `HDBSCAN.__init__`, and sklearn/cluster/_hdbscan/hdbscan.py:640 allows `[Integral, None]`.
scenario: "User instantiates `HDBSCAN()` expecting the documented sequential/`None` default → the estimator silently spawns 4 workers, giving different perf/reproducibility characteristics than the public contract advertises"
contract: Set the `HDBSCAN.__init__` default to `n_jobs=None` so the constructor default matches the docstring and the standard scikit-learn convention.
instances: single-instance

### F2 — Private `_hdbscan_brute` signature defaults contradict its documented defaults (`alpha=None`, `min_samples=5` vs docs)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-165 declares `def _hdbscan_brute(X, min_samples=5, alpha=None, ...)`; the docstring at sklearn/cluster/_hdbscan/hdbscan.py:179-184 states `min_samples : int, default=None` and `alpha : float, default=1.0`. Line sklearn/cluster/_hdbscan/hdbscan.py:241 then unconditionally runs `distance_matrix /= alpha`, which raises `TypeError: unsupported operand type(s) for /=: '...' and 'NoneType'` if the signature default is used.
scenario: "Any direct caller (test, downstream code) relying on the documented `alpha=1.0` default → `/= alpha` blows up with a `TypeError` because the true default is `None`"
contract: Change `_hdbscan_brute`'s signature to `min_samples=None, alpha=1.0` (matching its docstring), or update the docstring and signature to a single, consistent set of defaults; the function body must never divide by `None`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:160-161, sklearn/cluster/_hdbscan/hdbscan.py:179-184]

### F3 — `_hdbscan_prims` docstring documents `copy` parameter that its signature does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 defines `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric=..., leaf_size=40, n_jobs=None, **metric_params)` with no `copy` parameter, while sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False`. Conversely, the mandatory `algo` positional parameter is undocumented, and `min_samples : int, default=None` in the docstring (sklearn/cluster/_hdbscan/hdbscan.py:290-292) contradicts the signature default of `5`.
scenario: "Caller passes `copy=True` following the documented interface → `TypeError: _hdbscan_prims() got an unexpected keyword argument 'copy'`"
contract: Sync the `_hdbscan_prims` docstring with the actual signature: remove the `copy` entry, add an `algo` entry describing the `{"kd_tree","ball_tree"}` contract, and correct the `min_samples` default to `5`.
instances: single-instance

### F4 — `HDBSCAN.labels_` dtype flips between `np.intp` and `np.int32` depending on input
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:822-829 assigns `self.labels_` from `tree_to_labels`, which returns an `intp` array (`_do_labelling` declared `cnp.ndarray[cnp.intp_t, ndim=1, mode='c']` at sklearn/cluster/_hdbscan/_tree.pyx:431); sklearn/cluster/_hdbscan/hdbscan.py:840-844 then replaces `self.labels_` with a fresh `np.empty(..., dtype=np.int32)` array when the input contained non-finite rows.
scenario: "User writes code that keys on `hdb.labels_.dtype` (e.g. joining to another `np.intp`-indexed array on a 64-bit system) → the dtype silently changes to `int32` when the same estimator is run on data containing NaN/Inf rows"
contract: Allocate the remapped label buffer as `np.empty(self._raw_data.shape[0], dtype=np.intp)` so `HDBSCAN.labels_` has a single, path-independent dtype.
instances: single-instance

### F5 — `_compute_stability` returns a dict keyed by `numpy.float64` scalars instead of cluster ids
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:269-276 builds `result_pre_dict = np.vstack((np.arange(smallest_cluster, np.max(parents) + 1), result)).T` and then `return dict(result_pre_dict)`. Because `result` is `float64` (`cnp.float64_t[::1]`), `vstack` promotes the parent-id column to `float64`, so the dict's keys are `numpy.float64` scalars, not `intp` cluster ids.
scenario: "Downstream code in `_get_clusters` (sklearn/cluster/_hdbscan/_tree.pyx:707-759) sorts these float keys and mixes them with `intp` values from `cluster_tree['child']`/`cluster_tree['parent']` (see the `set(eom_clusters)` / `is_cluster` dictionaries) → membership tests rely on `float(3.0) == int(3)` Python equality; the contract 'cluster id maps to stability' becomes 'float → float', silently broken and fragile to any consumer that does `isinstance(k, int)` or uses the id as an array index"
contract: Populate the stability mapping explicitly as `dict(zip(range(smallest_cluster, max_parent + 1), np.asarray(result)))` (or equivalent int-keyed construction) so keys are `intp`/`int` cluster ids matching the rest of the tree contract.
instances: single-instance

### F6 — `_get_finite_row_indices` returns a `float64` array for sparse input, `int64` for dense
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 — the sparse branch does `np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])`, which yields `dtype=float64` when the list is empty (numpy's default) and `int` otherwise; the dense branch (`(row_indices,) = np.isfinite(matrix.sum(axis=1)).nonzero()`) always yields `int64`.
scenario: "Sparse input in which every row contains a non-finite entry → `finite_index` is a `float64` array; the subsequent `X = X[finite_index]` uses floating indices (deprecated / errors in modern numpy), and `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` stores float values that are later written into `HIERARCHY_dtype`'s intp `left_node`/`right_node` fields via `remap_single_linkage_tree`"
contract: Construct the sparse index array with an explicit int dtype (`np.fromiter((i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))), dtype=np.intp)`) so `_get_finite_row_indices` returns the same integer dtype in both branches.
instances: single-instance

### F7 — `test_hdbscan_precomputed_non_brute` uses `algorithm` values that violate the parameter constraint
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 fits with `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"` / `"prims_balltree"`); the `HDBSCAN` constraint set at sklearn/cluster/_hdbscan/hdbscan.py:629-638 accepts only `{"auto","brute","kdtree","balltree"}`, so `_validate_params()` raises `InvalidParameterError` before the code under test (the precomputed-vs-tree guard on sklearn/cluster/_hdbscan/hdbscan.py:772-783) is exercised.
scenario: "A regression removes the precomputed+tree ValueError in `fit` → this test still passes because its ValueError is emitted by parameter validation, so the check the docstring claims to enforce silently rots"
contract: Rewrite the test to use the real valid `algorithm` names (`"kdtree"`, `"balltree"`) so the raised ValueError actually comes from the precomputed guard.
instances: single-instance

### F8 — `test_dbscan_clustering_outlier_data` does elementwise ndarray addition where list concatenation is intended [out-of-theme]
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:206-212 — `missing_labels_idx` and `infinite_labels_idx` are `np.flatnonzero` results (ndarrays), then `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))` uses `+` on those ndarrays, which broadcasts elementwise (`np.array([2,5]) + np.array([0]) → np.array([2,5])`) rather than concatenating the index lists. The infinite index `0` is therefore *not* excluded from `clean_idx`, so `X_outlier[clean_idx]` still contains the `np.inf` row.
scenario: "Reader trusts the test's 'clean_idx' as verifying `dbscan_clustering` on cleaned data → the test is actually re-running HDBSCAN on data that still contains an infinite row, and only passes because the outlier-encoding path assigns the same label to the same row in both runs"
contract: Combine the two index arrays with concatenation (`np.concatenate([missing_labels_idx, infinite_labels_idx])` or explicit `list(...) + list(...)`) so `clean_idx` truly excludes every non-finite row.
instances: single-instance
