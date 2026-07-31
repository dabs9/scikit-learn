### F1 — `_weighted_cluster_center` uses filtered X but full-size labels after non-finite remap
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:833-855 — after `not all_finite` branch, `self.labels_` is expanded to `self._raw_data.shape[0]`, then `self._weighted_cluster_center(X)` is called with the local `X` that was replaced by `X = X[finite_index]` on line 733. Inside, line 908 computes `mask = self.labels_ == idx` (length `n_raw`) and line 909 does `data = X[mask]` (X has only `n_finite` rows).
scenario: "`HDBSCAN(store_centers='both').fit(X)` on X containing any np.nan/np.inf sample → boolean mask length (n_raw) does not match X length (n_finite) → IndexError from `X[mask]` inside `_weighted_cluster_center`, blocking every centroid/medoid computation on non-finite input"
contract: Pass the full-length feature array (e.g., `self._raw_data`) to `_weighted_cluster_center` so `X.shape[0] == self.labels_.shape[0]`, or restrict the mask to finite rows before indexing X.
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
