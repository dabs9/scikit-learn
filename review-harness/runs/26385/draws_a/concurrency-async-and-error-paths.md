I've verified the key concerns. Numpy is not available to test locally, but the boolean-index-mismatch behavior in numpy has been an IndexError for years. Now emitting findings.

### F1 — `_weighted_cluster_center` crashes with `IndexError` when `store_centers` is set and X has non-finite rows
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855, sklearn/cluster/_hdbscan/hdbscan.py:895-909 — after line 733 `X = X[finite_index]` reduces X to `n_finite` rows, and after line 844 `self.labels_ = new_labels` is of length `n_original = self._raw_data.shape[0]`. Line 855 then passes the reduced `X` to `_weighted_cluster_center`, which computes `mask = self.labels_ == idx` (length `n_original`) and does `data = X[mask]` (X has `n_finite < n_original` rows).
scenario: "user fits `HDBSCAN(store_centers='centroid')` on data containing any `np.nan` or `np.inf` row → NumPy raises `IndexError: boolean index did not match indexed array along dimension 0` inside `fit`, so the estimator never returns a fitted model even though the label/probability computation succeeded"
contract: Pass `self._raw_data` (or restrict computation to the finite subset via `finite_index`) to `_weighted_cluster_center` when non-finite rows were removed, and cover the combination `(non-finite input) x (store_centers set)` in the test suite.
instances: single-instance

### F2 — Default `n_jobs=4` silently overrides `joblib.parallel_backend` context and contradicts the documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` has `n_jobs=4,`; the class docstring at lines 486-490 states `n_jobs : int, default=None` with "`None` means 1 unless in a :obj:`joblib.parallel_backend` context." `n_jobs` is threaded verbatim into `pairwise_distances(..., n_jobs=n_jobs)` (line 239) and `NearestNeighbors(..., n_jobs=n_jobs)` (line 338).
scenario: "user wraps `HDBSCAN().fit(X)` in `with joblib.parallel_backend('threading', n_jobs=1): ...` expecting single-threaded execution → sklearn still spawns 4 workers because the explicit `n_jobs=4` bypasses backend deference; conversely users get unexpected multi-process parallelism even when they didn't opt in"
contract: Change the `__init__` default to `n_jobs=None` so the documented "defer to `joblib.parallel_backend`" contract holds.
instances: single-instance

### F3 — `_weighted_cluster_center` silently produces meaningless centers when `metric="precomputed"` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 — `fit` unconditionally calls `self._weighted_cluster_center(X)` when `self.store_centers`, regardless of `self.metric`. The docstring at lines 882-883 explicitly warns "This requires `X` to be a raw feature array, not precomputed distances", but no guard exists. For `metric="precomputed"`, `X` at that point is the distance matrix passed by the user, so `np.average(data, weights=strength, axis=0)` at line 912 averages distance-vector rows and `pairwise_distances(data, metric=self.metric, ...)` at 915-917 requests pairwise-distance-of-distance-vectors.
scenario: "user fits `HDBSCAN(metric='precomputed', store_centers='centroid').fit(D)` → `self.centroids_` is populated with averages of rows of the distance matrix (garbage) with no warning or error; the estimator reports success but the stored attribute is nonsensical"
contract: In `fit`, raise `ValueError` if `store_centers is not None and self.metric == "precomputed"` before dispatching to `_weighted_cluster_center`.
instances: single-instance

### F4 — `n_clusters` computation in `_weighted_cluster_center` fails to exclude the `-3` (missing) outlier label
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. `_OUTLIER_ENCODING` at lines 65-79 defines labels `-1` (noise, implicit), `-2` (infinite), and `-3` (missing); only `-1` and `-2` are subtracted from the set.
scenario: "input contains rows with `np.nan` (label `-3`) and `store_centers` is set (assume F1 is fixed so the length mismatch is resolved) → `n_clusters` includes `-3`, the loop `for idx in range(n_clusters)` iterates one past the highest real cluster id, and `data = X[self.labels_ == n_clusters-1]` is empty, so `np.average(empty, weights=empty, axis=0)` raises `ZeroDivisionError` / warns and writes `nan` to `centroids_`"
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` (or equivalently subtract the full set of outlier labels defined in `_OUTLIER_ENCODING`).
instances: single-instance
