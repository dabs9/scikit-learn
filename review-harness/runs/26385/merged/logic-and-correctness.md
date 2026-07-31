### F1 — `_weighted_cluster_center` shape mismatch when non-finite samples are present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-909 — inside `fit`, when `metric != "precomputed"` and the input contains non-finite entries, `X = X[finite_index]` (line 733) reduces `X` to just the finite rows and `self.labels_` is later reassigned to a raw-sized array via `new_labels = np.empty(self._raw_data.shape[0], ...)` (line 844); then `if self.store_centers: self._weighted_cluster_center(X)` (line 855) passes the finite-only `X`. Inside `_weighted_cluster_center`, `mask = self.labels_ == idx` has raw length and `data = X[mask]` mixes raw-length mask with finite-length `X`, producing `IndexError: boolean index did not match indexed array along axis 0`.
scenario: "`HDBSCAN(store_centers='both').fit(X_with_nan_or_inf)` where any row of `X` contains `np.nan`/`np.inf` → `fit` raises `IndexError` instead of returning a fitted estimator (confirmed by live reproduction)."
contract: In `_weighted_cluster_center`, build the mask on `self.labels_[finite_index]` and index the finite `X` so mask and data always share axis-0 length.
instances: single-instance

### F2 — `n_clusters` count omits the `-3` (missing) outlier label
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The class docstring at lines 561-563 and 571-573 states that "the `-1, -2, -3` labels for the outlier clusters are excluded", and `_OUTLIER_ENCODING["missing"]["label"] == -3`.
scenario: "fit on data with np.nan samples and `store_centers='both'` → labels contain -3; n_clusters is one too large; loop iterates `idx = n_clusters - 1` for which `mask = labels_ == idx` is all False → `np.average(empty, ...)` raises ZeroDivisionError / medoid path calls `pairwise_distances` on an empty array; even if fixed, the extra empty row would corrupt `centroids_`/`medoids_`"
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` (or subtract the full `_OUTLIER_ENCODING` label set) so counts match the documented semantics.
instances: single-instance

### F3 — Sparse underpopulated rows silently keep original edge weights when `max_distance <= 0`
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:193-212 — rows with `row_data.size <= further_neighbor_idx` get `core_distances[i] = INFINITY`; then in the update loop `if isfinite(mutual_reachibility_distance): data[i] = ...; elif max_distance > 0: data[i] = max_distance` — otherwise `data[i]` is left untouched. `_brute_mst` (sklearn/cluster/_hdbscan/hdbscan.py:111-123) only errors when `connected_components > 1`, which is unchanged because the sparse pattern was not modified.
scenario: "user passes a sparse precomputed matrix where some row has 1..(min_samples-1) explicit distances and does not set `metric_params={'max_distance': ...}` → core distance is infinite but data values are kept as the raw small distances → MST includes those small edges → clustering silently uses wrong reachability for the underpopulated points without ever raising the documented `There exists points with fewer than …` error"
contract: When `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, raise the same "fewer than {min_samples} neighbors" ValueError instead of leaving `data[i]` unchanged (or overwrite `data[i]` with `INFINITY` so the downstream MST/warning path catches it).
instances: single-instance

### F4 — Row containing both `+inf` and `-inf` (no NaN) is mis-labeled as "missing" instead of "infinite"
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:721-728 — `reduced_X = X.sum(axis=1)`; `missing_index = np.isnan(reduced_X).nonzero()[0]`; `infinite_index = np.isinf(reduced_X).nonzero()[0]`. For a row like `[np.inf, -np.inf]`, the sum is `nan`, so the sample is added to `missing_index` and never to `infinite_index`. Combined with the write order at lines 842-843 (`missing` overwrites), the sample ends up with label `-3`, contradicting the class docstring (lines 534-537) which reserves `-3` for "samples with missing data".
scenario: "fit on a feature array with a row containing both +inf and -inf and no NaN → sample gets label -3 (missing) and `probabilities_ = np.nan`, though the point has no missing values; users filtering `labels_ == -2` to find infinite-outlier rows will miss it"
contract: Detect infinite/missing rows per-element (e.g., via `np.isnan(X).any(axis=1)` and `np.isinf(X).any(axis=1)`), not via the row sum, so `+inf`/`-inf` presence is not aliased into NaN.
instances: single-instance

### F5 — `_hdbscan_brute`'s `alpha=None` default fails the `distance_matrix /= alpha` divide
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-241 — signature `def _hdbscan_brute(..., alpha=None, ...)` while line 241 does `distance_matrix /= alpha`. The docstring at line 183 states `alpha : float, default=1.0`. HDBSCAN.fit always supplies `alpha=self.alpha`, so the default is dead through the class path, but the function is `cpdef`-adjacent and importable.
scenario: "any direct caller of `_hdbscan_brute` relying on the documented `alpha=1.0` default → `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'` on line 241"
contract: Change the default to `alpha=1.0` to match the docstring and the semantics used elsewhere (e.g., `_hdbscan_prims` line 273).
instances: single-instance

### F6 — Inline comment in `fit` states the wrong outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2", but `_OUTLIER_ENCODING` (lines 71-85) and the public class docstring (lines 540-543) fix the mapping as `np.inf → -2`, `np.nan → -3` (and `-1` is reserved for noise). Immediately below, the code assigns `new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` (i.e. `-2`) and `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` (i.e. `-3`), contradicting the comment.
scenario: "Reader/maintainer trusts the inline comment while reasoning about `_weighted_cluster_center`/`dbscan_clustering` outlier handling → introduces a follow-on bug that treats `-1`/`-2` as the outlier encodings when the true encodings are `-2`/`-3`."
contract: Update the comment to "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3" so it matches `_OUTLIER_ENCODING` and the public docstring.
instances: single-instance

### F7 — `n_jobs` default disagrees with documented default [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` has `n_jobs=4`, while the class-docstring parameter description at line 486 says "`n_jobs : int, default=None`" and further states "`None` means 1 unless in a `joblib.parallel_backend` context." The signature and the documented API disagree.
scenario: "User reads the class docstring, expects the default to inherit from a `joblib.parallel_backend` context (or fall back to 1), but `HDBSCAN()` is instantiated with no `n_jobs` argument → constructor uses `n_jobs=4`, spawning parallelism the user did not opt into and clashing with the surrounding backend"
contract: Change the default in `__init__` to `n_jobs=None` to match the documented behavior.
instances: single-instance
