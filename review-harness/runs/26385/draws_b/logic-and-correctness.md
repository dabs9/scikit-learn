Live repro confirms F1: `IndexError: boolean index did not match indexed array along axis 0; size of axis is 198 but size of corresponding boolean axis is 200` when `store_centers` is set on data with non-finite rows.

### F1 — `_weighted_cluster_center` shape mismatch when non-finite samples are present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-909 — inside `fit`, when `metric != "precomputed"` and the input contains non-finite entries, `X = X[finite_index]` (line 733) reduces `X` to just the finite rows and `self.labels_` is later reassigned to a raw-sized array via `new_labels = np.empty(self._raw_data.shape[0], ...)` (line 844); then `if self.store_centers: self._weighted_cluster_center(X)` (line 855) passes the finite-only `X`. Inside `_weighted_cluster_center`, `mask = self.labels_ == idx` has raw length and `data = X[mask]` mixes raw-length mask with finite-length `X`, producing `IndexError: boolean index did not match indexed array along axis 0`.
scenario: "`HDBSCAN(store_centers='both').fit(X_with_nan_or_inf)` where any row of `X` contains `np.nan`/`np.inf` → `fit` raises `IndexError` instead of returning a fitted estimator (confirmed by live reproduction)."
contract: In `_weighted_cluster_center`, build the mask on `self.labels_[finite_index]` and index the finite `X` so mask and data always share axis-0 length.
instances: single-instance

### F2 — `_weighted_cluster_center` cluster count ignores the `-3` (missing) outlier label
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` subtracts only `-1` and `-2`, but `_OUTLIER_ENCODING["missing"]["label"] == -3` (line 80) and the class docstring at lines 542-543 asserts `-3` is a valid outlier label; when `self.labels_` contains any `-3`, `n_clusters` is inflated by one and the subsequent `for idx in range(n_clusters)` iterates past the real clusters (the extra `idx` will match no samples, so `np.average(data, weights=strength, axis=0)` on an empty slice raises).
scenario: "Once F1 is fixed and `_weighted_cluster_center` is invoked on data containing samples labeled `-3` (missing) → the extra loop iteration hits an empty cluster mask and `np.average` raises `ZeroDivisionError`/warns and produces `nan` centroids, or `pairwise_distances` fails on empty input."
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` so every non-cluster label — including missing — is excluded from the iteration.
instances: single-instance

### F3 — Inline comment in `fit` states the wrong outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2", but `_OUTLIER_ENCODING` (lines 71-85) and the public class docstring (lines 540-543) fix the mapping as `np.inf → -2`, `np.nan → -3` (and `-1` is reserved for noise). Immediately below, the code assigns `new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` (i.e. `-2`) and `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` (i.e. `-3`), contradicting the comment.
scenario: "Reader/maintainer trusts the inline comment while reasoning about `_weighted_cluster_center`/`dbscan_clustering` outlier handling → introduces a follow-on bug that treats `-1`/`-2` as the outlier encodings when the true encodings are `-2`/`-3`."
contract: Update the comment to "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3" so it matches `_OUTLIER_ENCODING` and the public docstring.
instances: single-instance
