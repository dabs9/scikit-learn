### F1 — `_weighted_cluster_center` crashes on non-finite input when `store_centers` is set
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855, 907-909 — `fit` passes the finite-subset `X` (shape `(n_finite, n_features)` after `X = X[finite_index]` at line 733) to `_weighted_cluster_center`, but by this point `self.labels_` has been reassigned at line 844 to shape `(n_total,)`. Inside the helper, `mask = self.labels_ == idx` yields a boolean array of size `n_total`, then `data = X[mask]` boolean-indexes X (size `n_finite`) with a mismatched mask.
scenario: "`HDBSCAN(store_centers='centroid').fit(X)` where `X` contains any `np.nan` or `np.inf` row → `IndexError: boolean index did not match indexed array along dimension 0; dimension is n_finite but corresponding boolean dimension is n_total`"
contract: Pass `self._raw_data` (full n_total rows) rather than the finite-subset `X` when invoking `self._weighted_cluster_center`, and mask both `self._raw_data` and `self.probabilities_` with the label mask.
instances: single-instance

### F2 — `_weighted_cluster_center` over-counts clusters by treating missing-data label `-3` as a real cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. `_OUTLIER_ENCODING` defines three outlier labels: -1 (noise), -2 (infinite), -3 (missing). Only -1 and -2 are subtracted, so when `self.labels_` contains any -3 the count is inflated by 1.
scenario: "Input `X` contains any `np.nan` row, `store_centers='centroid'`; the extra iteration of `for idx in range(n_clusters)` reaches an `idx` no sample carries → `mask` is all-False → `data`/`strength` empty → `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError: Weights sum to zero, can't be normalized` and `centroids_[idx]` is never assigned"
contract: Use `set(self.labels_) - {-1, -2, -3}` (or `- set(v['label'] for v in _OUTLIER_ENCODING.values()) - {-1}`) so every outlier label defined in `_OUTLIER_ENCODING` is excluded from the cluster count.
instances: single-instance

### F3 — `n_jobs` default disagrees with documented default [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` has `n_jobs=4`, while the class-docstring parameter description at line 486 says "`n_jobs : int, default=None`" and further states "`None` means 1 unless in a `joblib.parallel_backend` context." The signature and the documented API disagree.
scenario: "User reads the class docstring, expects the default to inherit from a `joblib.parallel_backend` context (or fall back to 1), but `HDBSCAN()` is instantiated with no `n_jobs` argument → constructor uses `n_jobs=4`, spawning parallelism the user did not opt into and clashing with the surrounding backend"
contract: Change the default in `__init__` to `n_jobs=None` to match the documented behavior.
instances: single-instance
