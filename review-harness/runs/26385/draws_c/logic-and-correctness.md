### F1 — `_weighted_cluster_center` boolean-index mismatch when non-finite data is present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 calls `self._weighted_cluster_center(X)` where `X` was reduced to finite rows at line 733 (`X = X[finite_index]`), while `self.labels_` was expanded to the full raw length at line 844 (`new_labels = np.empty(self._raw_data.shape[0], ...)` then `self.labels_ = new_labels`). Inside `_weighted_cluster_center` (line 908 `mask = self.labels_ == idx`, line 909 `data = X[mask]`) the mask has shape `(n_raw,)` but `X` has shape `(n_finite, n_features)`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing `np.inf` or `np.nan` rows → boolean index dimension mismatch raises `IndexError` before any centroid/medoid is stored"
contract: pass `self._raw_data` (the full validated data) to `_weighted_cluster_center` when non-finite rows were filtered, so the mask indexes over the same rows as `self.labels_`.
instances: single-instance

### F2 — `n_clusters` in `_weighted_cluster_center` fails to exclude the `-3` (missing) outlier label
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. Per `_OUTLIER_ENCODING` (lines 65-79) and the docstring at lines 561-563/572-573, `-3` is the "missing data" outlier label and should be excluded from cluster counts alongside `-1` and `-2`.
scenario: "fit with samples containing `np.nan` and `store_centers='centroid'` → `n_clusters` is off by one; the `for idx in range(n_clusters)` loop then processes a non-existent `idx == n_actual_clusters` cluster whose `mask` is all `False`, triggering `ZeroDivisionError` from `np.average(empty, weights=empty, axis=0)`"
contract: compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` (or equivalently `len(set(self.labels_[self.labels_ >= 0]))`) so all three outlier encodings are excluded.
instances: single-instance

### F3 — `_weighted_cluster_center` `mask` allocation shape wrong when non-finite rows exist
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)`. This pre-allocation is immediately overwritten on line 908 (`mask = self.labels_ == idx`), so the allocation is dead. However, the shape `(X.shape[0],)` documents an incorrect assumption that `X` and `self.labels_` share length — the same assumption that leads to F1.
scenario: "maintainer reads the dead `mask = np.empty((X.shape[0],), ...)` allocation and infers `X.shape[0] == self.labels_.shape[0]` → propagates the wrong invariant into a future edit (the same assumption that produces F1), while the allocation itself provides no runtime safety since line 908 rebinds `mask` to size `self.labels_.shape[0]`"
contract: remove the dead pre-allocation on line 896; the loop assigns `mask` directly on line 908.
instances: single-instance

### F4 — `n_jobs` default of `4` contradicts the documented default of `None` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the standard sklearn semantics ("`None` means 1 unless in a `joblib.parallel_backend` context"), but the constructor signature at line 658 sets `n_jobs=4`. A user reading only the docstring will silently get 4-way parallelism instead of serial execution.
scenario: "user constructs `HDBSCAN()` expecting serial execution per the docstring → gets 4-worker parallelism, oversubscribing threads inside a `joblib.parallel_backend` or on a shared host"
contract: change the constructor default to `n_jobs=None` to match the documented sklearn convention.
instances: single-instance
