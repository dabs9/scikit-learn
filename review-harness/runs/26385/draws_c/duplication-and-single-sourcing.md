### F1 — Duplicate `births` allocation, first result immediately discarded
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears back-to-back on lines 252 and 254 with an identical signature; the loop between them (line 253 is blank) never uses the first array.
scenario: "`_compute_stability` runs on every fit → an extra `largest_child+1`-long float array is allocated and immediately garbage-collected on every call."
contract: Delete the redundant line 252 allocation; keep only the assignment on line 254 (the one whose result is actually used).
instances: single-instance

### F2 — `_weighted_cluster_center` hard-codes outlier labels instead of deriving them from `_OUTLIER_ENCODING`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` literally spells out the sentinel labels; `_OUTLIER_ENCODING` (lines 65-79) is the module's declared single source of truth and holds `-2` (`infinite`) and `-3` (`missing`), and the sibling `dbscan_clustering` method on lines 964-969 reads those labels via `_OUTLIER_ENCODING["infinite"]["label"]`/`["missing"]["label"]`.
scenario: "`HDBSCAN(store_centers='both')` fit over data whose only outliers are `np.nan`-rows → those rows carry label `-3`, `{-1, -2}` fails to filter them, `n_clusters` is inflated by one, and the extra iteration produces a spurious zero/NaN centroid row (or a `ZeroDivisionError` in `np.average` when the mask is empty)."
contract: Derive the excluded set from the single source: `n_clusters = len(set(self.labels_) - {NOISE_LABEL} - {v["label"] for v in _OUTLIER_ENCODING.values()})` (with `NOISE_LABEL = -1` sourced from `_tree.pyx`'s `NOISE`, or promoted into `_OUTLIER_ENCODING`).
instances: single-instance

### F3 — Documented `n_jobs` default (`None`) contradicts the constructor default (`4`)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring: `n_jobs : int, default=None` with `` `None` means 1 unless in a :obj:`joblib.parallel_backend` context. `` sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`.
scenario: "A user reads the documented default and expects single-threaded behavior in a serial environment (or under `parallel_backend`) → HDBSCAN silently spawns 4 workers, breaking pinned CPU budgets and the `joblib.parallel_backend` contract that other sklearn estimators honor."
contract: Change `__init__` to `n_jobs=None,` so the value matches the docstring and the sklearn-wide `n_jobs` convention.
instances: single-instance

### F4 — Cython typedefs inconsistently sourced across the new `_hdbscan` module
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx uses `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` 90 times (e.g. lines 39, 40, 58, 61, 66, 67, 90, 145-155), whereas sklearn/cluster/_hdbscan/_linkage.pyx and _reachability.pyx cimport `intp_t`, `float64_t`, `int64_t`, `uint8_t` directly from `...utils._typedefs` (_linkage.pyx:42, _reachability.pyx:40) and use them exclusively. The plan of record explicitly claims "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`", but `_tree.pyx` was not migrated.
scenario: "Future refactor of `_typedefs.pxd` (e.g. shifting `intp_t` from `np.intp` to another width) → the two files diverge in their storage/API and the mixed usage inside the same module makes the drift harder to spot."
contract: Replace every `cnp.<type>_t` in `_tree.pyx` with the corresponding typedef cimported from `...utils._typedefs`, matching the style of `_linkage.pyx`/`_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:61, sklearn/cluster/_hdbscan/_tree.pyx:66, sklearn/cluster/_hdbscan/_tree.pyx:67, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:91, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:147, sklearn/cluster/_hdbscan/_tree.pyx:150, sklearn/cluster/_hdbscan/_tree.pyx:151, sklearn/cluster/_hdbscan/_tree.pyx:153, sklearn/cluster/_hdbscan/_tree.pyx:154, sklearn/cluster/_hdbscan/_tree.pyx:155, sklearn/cluster/_hdbscan/_tree.pyx:182, sklearn/cluster/_hdbscan/_tree.pyx:240, sklearn/cluster/_hdbscan/_tree.pyx:241, sklearn/cluster/_hdbscan/_tree.pyx:243, sklearn/cluster/_hdbscan/_tree.pyx:244, sklearn/cluster/_hdbscan/_tree.pyx:246, sklearn/cluster/_hdbscan/_tree.pyx:247, sklearn/cluster/_hdbscan/_tree.pyx:248, sklearn/cluster/_hdbscan/_tree.pyx:249, sklearn/cluster/_hdbscan/_tree.pyx:281, sklearn/cluster/_hdbscan/_tree.pyx:286, sklearn/cluster/_hdbscan/_tree.pyx:289]

### F5 — `PyArray_SHAPE` cdef-extern declaration duplicated between `_tree.pxd` and `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. sklearn/cluster/_hdbscan/_linkage.pyx:44-45 — identical block re-declared locally, even though the file already `cimport`s from `_tree` on line 40.
scenario: "Someone changes the signature (or type-source) in `_tree.pxd` (e.g. under the F4 migration) → `_linkage.pyx`'s local copy silently drifts, producing conflicting extern declarations at compile time or a stale return type."
contract: Delete the local `cdef extern` block in `_linkage.pyx:44-45` and cimport `PyArray_SHAPE` from `..._hdbscan._tree` alongside `HIERARCHY_t` — a single declaration site.
instances: single-instance

### F6 — Dead `mask = np.empty(...)` allocation immediately overwritten [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)` is assigned before any use, and the very first statement in the loop at line 908 reassigns it as `mask = self.labels_ == idx`; nothing between lines 896 and 908 reads it.
scenario: "Every `store_centers` fit reaches line 896 → an unused bool array of length `n_samples` is allocated and immediately overwritten, wasting `n_samples` bytes and one heap allocation per fit."
contract: Delete the dead allocation on line 896.
instances: single-instance

### F7 — `store_centers` + non-finite input crashes because `X` is finite-filtered but `self.labels_` is raw-sized [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 reduces `X = X[finite_index]`; hdbscan.py:840-844 rebuilds `self.labels_` at `self._raw_data.shape[0]` (full raw length); hdbscan.py:854-855 then calls `self._weighted_cluster_center(X)` with the still-reduced `X`; inside `_weighted_cluster_center` (hdbscan.py:908-909) `mask = self.labels_ == idx` produces a boolean array whose length is the raw sample count, and `data = X[mask]` fails when that length differs from `X.shape[0]`.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_row)` → `X` shape mismatch with `mask` → `IndexError: boolean index did not match indexed array along dimension 0; dimension is <finite_n> but corresponding boolean dimension is <raw_n>`."
contract: Inside `_weighted_cluster_center`, index `self.labels_` by `self._finite_index` before comparing (`labels = self.labels_[self._finite_index]; mask = labels == idx`) so the mask and the finite-filtered `X` share a length.
instances: single-instance
