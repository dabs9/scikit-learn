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

### F8 — `HDBSCAN.dbscan_clustering` accepts unvalidated caller-supplied parameters
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-969 — the public method `dbscan_clustering(self, cut_distance, min_cluster_size=5)` never calls `_validate_params()` and its two parameters are not listed in `_parameter_constraints` (lines 616-645). Values flow straight into the Cython `labelling_at_cut(self._single_linkage_tree_, cut_distance, min_cluster_size)` at line 960-962 where `cut_distance` is coerced to `cnp.float64_t` and `min_cluster_size` to `cnp.intp_t` with no bounds/type gate at the Python boundary.
scenario: "caller invokes `clusterer.dbscan_clustering(cut_distance=-1.0, min_cluster_size=-5)` after fitting → in `_tree.pyx:419` the guard `cluster_size[cluster] < min_cluster_size` is vacuously false for every cluster, so every merged group is emitted as a real cluster label instead of noise — silently wrong output rather than a `ValueError`."
contract: Both `cut_distance` and `min_cluster_size` must be validated (e.g., via an `Interval` constraint in `_parameter_constraints` and a call to `_validate_params()` at the start of `dbscan_clustering`) so out-of-range or wrong-type inputs raise `InvalidParameterError` before reaching Cython.
instances: single-instance

### F9 — Dense precomputed distance matrix only rejects NaN, silently accepts negative values and -inf
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:741-751 — for `metric="precomputed"` on a dense array the code calls `_validate_data(X, force_all_finite=False, dtype=np.float64)` then checks only `if np.isnan(X).any(): raise ValueError("np.nan values found in precomputed-dense")`. The comment at line 743-744 documents intent as "allowed to contain numpy.inf for missing distances", but there is no check for negative values or `-np.inf`. In `_hdbscan_brute` at line 241 the matrix is passed to `mutual_reachability_graph` unchanged, and in `_reachability.pyx:144-149` `max(core_i, core_j, distance_matrix[i,j])` propagates any `-inf`/negatives through the MST without complaint.
scenario: "user passes a precomputed dense matrix containing `-np.inf` or negative pseudo-distances (e.g., from a similarity matrix mistakenly reused as distances) → passes validation, `mutual_reachability_graph` folds them in, and clustering silently returns nonsensical labels/probabilities instead of raising the expected input-validation error."
contract: Dense precomputed matrices must be rejected when they contain negative values or `-np.inf`; only NaN and (per docs) `+np.inf` should be tolerated.
instances: single-instance

### F10 — Length mismatch between `X` and `self.labels_` crashes `_weighted_cluster_center` for non-finite inputs
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 (`X = X[finite_index]` shrinks `X` to only finite rows), sklearn/cluster/_hdbscan/hdbscan.py:840-844 (`self.labels_` is rebuilt at `self._raw_data.shape[0]`, i.e. FULL length), sklearn/cluster/_hdbscan/hdbscan.py:854-855 (`self._weighted_cluster_center(X)` is invoked with the SHRUNKEN `X`), sklearn/cluster/_hdbscan/hdbscan.py:908-910 (`mask = self.labels_ == idx` has full length, `X[mask]` then indexes the shrunken `X`).
scenario: "user calls `HDBSCAN(store_centers=\"centroid\").fit(X)` where `X` contains any `np.nan`/`np.inf` row → `_validate_data(force_all_finite=False)` accepts it, `all_finite=False`, `X` is reduced to finite rows only while `self.labels_` is stretched back to `self._raw_data.shape[0]`; `X[mask]` in `_weighted_cluster_center` raises `IndexError: boolean index did not match indexed array along dimension 0` and `fit` aborts with `centroids_`/`medoids_` never set"
contract: pass `self._raw_data` (or another full-length reference to the original data) into `_weighted_cluster_center` in the non-finite branch, so that `mask` and the data it indexes have identical lengths.
instances: single-instance

### F11 — HDBSCAN default `n_jobs=4` diverges from documented default and defeats joblib.parallel_backend
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature has `n_jobs=4,`, while the same file's docstring at lines 486–490 states `n_jobs : int, default=None. Number of jobs to run in parallel to calculate distances. None means 1 unless in a joblib.parallel_backend context.` The value propagates verbatim through `kwargs["n_jobs"] = self.n_jobs` (line 769–775) to `pairwise_distances(..., n_jobs=n_jobs, ...)` (line 245) and `NearestNeighbors(..., n_jobs=n_jobs, ...)` (line 341).
scenario: "user does `with joblib.parallel_backend('loky', n_jobs=1): HDBSCAN().fit(X)` → 4 worker processes still spawn because the concrete `n_jobs=4` is passed through rather than sentinel `None` → resource-oversubscription and behavior contrary to what the docstring promises"
contract: Change the `__init__` default to `n_jobs=None` so both `pairwise_distances` and `NearestNeighbors` see `None` and respect the joblib parallel backend context, matching the documented contract.
instances: single-instance

### F12 — `_sparse_mutual_reachability_graph` silently retains original edge weights when core distance is infinite and `max_distance <= 0`
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:199-200 (`core_distances[i] = INFINITY` when the row has fewer than `further_neighbor_idx+1` explicit non-zeros), sklearn/cluster/_hdbscan/_reachability.pyx:206-212 (in the `nogil` loop, if `mutual_reachibility_distance` is not finite AND `max_distance <= 0`, `data[i]` is left at its ORIGINAL value — no branch updates or errors). The downstream guard in `_brute_mst` at sklearn/cluster/_hdbscan/hdbscan.py:111-123 only detects disconnected components, not per-row "fewer than min_samples neighbors" for still-connected rows.
scenario: "sparse CSR distance matrix where some row has ≥1 but <`min_samples` non-zeros and the whole graph stays connected, `metric_params` omits `max_distance` (default 0.0) → `core_distances[row]=INFINITY`, every edge from that row satisfies `!isfinite(max(...))` and skips both branches, so the returned mutual-reachability graph has the caller's original edge weights for those rows; `csgraph.connected_components` still returns 1 (no error), the MST/clustering are built from stale values, and the user gets a silently wrong result"
contract: when `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, explicitly set `data[i] = INFINITY` so `_brute_mst`'s connected-components check will fire instead of leaving stale finite values in place.
instances: single-instance

### F13 — `_weighted_cluster_center` silently produces meaningless centers when `metric="precomputed"` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 — `fit` unconditionally calls `self._weighted_cluster_center(X)` when `self.store_centers`, regardless of `self.metric`. The docstring at lines 882-883 explicitly warns "This requires `X` to be a raw feature array, not precomputed distances", but no guard exists. For `metric="precomputed"`, `X` at that point is the distance matrix passed by the user, so `np.average(data, weights=strength, axis=0)` at line 912 averages distance-vector rows and `pairwise_distances(data, metric=self.metric, ...)` at 915-917 requests pairwise-distance-of-distance-vectors.
scenario: "user fits `HDBSCAN(metric='precomputed', store_centers='centroid').fit(D)` → `self.centroids_` is populated with averages of rows of the distance matrix (garbage) with no warning or error; the estimator reports success but the stored attribute is nonsensical"
contract: In `fit`, raise `ValueError` if `store_centers is not None and self.metric == "precomputed"` before dispatching to `_weighted_cluster_center`.
instances: single-instance

### F14 — `_weighted_cluster_center` counts label `-3` (missing) as a cluster, producing empty-mask `np.average` crash [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 (`n_clusters = len(set(self.labels_) - {-1, -2})` excludes noise (-1) and infinite-outlier (-2) but NOT missing (-3, `_OUTLIER_ENCODING["missing"]["label"]`) as defined at hdbscan.py:74). The subsequent loop at hdbscan.py:907-920 iterates `range(n_clusters)`; when a `-3` label is present the count is inflated and one `idx` value never matches any real cluster, giving an empty `data`/`strength` slice passed to `np.average(data, weights=strength, axis=0)` (line 912) which raises `ZeroDivisionError: Weights sum to zero`.
scenario: "hypothetical fit path with a `-3` label surviving into `self.labels_` combined with `store_centers != None` → `n_clusters` overcounts by 1, the extra iteration hits `data=X[mask]` with an all-False mask, `np.average` raises ZeroDivisionError. Currently unreachable because F1 raises first, but the bug is intrinsic to this function and will surface as soon as F1 is fixed."
contract: subtract every outlier label defined in `_OUTLIER_ENCODING` (i.e. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}` → `{-1, -2, -3}`) from `set(self.labels_)` when computing `n_clusters`.
instances: single-instance

### F15 — `test_dbscan_clustering_outlier_data` broadcasts arrays instead of concatenating, so "clean" data still contains the infinite outlier
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` (`np.array([2, 5])`) and `infinite_labels_idx` (`np.array([0])`) are NumPy arrays; `+` performs element-wise broadcast → `[2, 5]`, not `[2, 5, 0]`. Index 0 (the np.inf point) is NOT removed from `clean_idx`.
scenario: "future refactor changes how infinite outliers are labeled in `dbscan_clustering` on non-outlier inputs → test still passes because the 'clean' subset still contains the outlier and gets the same infinite label from the outlier path in both models, masking the regression"
contract: replace with an explicit concatenation such as `np.concatenate([missing_labels_idx, infinite_labels_idx]).tolist()` (or `list(...) + list(...)`) so all three outlier indices (0, 2, 5) are truly excluded from `clean_idx`.
instances: single-instance

### F16 — `test_labelling_thresholding` asserts only the noise-count, not which samples are noise
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:519-520,532-533 — `num_noise = condensed_tree["value"] < 1; assert sum(num_noise) == sum(labels == -1)` (and again with `< MAX_LAMBDA`). The assertion compares two totals; it never checks *which* sample indices `labels` marks as -1.
scenario: "a regression that mislabels a valid-cluster point as noise while promoting a noise point to the cluster (identity-swap) → totals still match, assertion passes"
contract: assert full-array equality against the expected label vector, e.g. `assert_array_equal(labels, np.where(condensed_tree['value'] < 1, -1, 0))` (adjusted for indexing), so identity — not just count — is verified.
instances: [sklearn/cluster/tests/test_hdbscan.py:519, sklearn/cluster/tests/test_hdbscan.py:520, sklearn/cluster/tests/test_hdbscan.py:532, sklearn/cluster/tests/test_hdbscan.py:533]

### F17 — `test_hdbscan_algorithms` never asserts clustering quality for the parametrized (algo, metric) combinations
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:143-175 — the only labels assertion (`n_clusters == n_clusters_true`) is made on a model built as `HDBSCAN(algorithm=algo)` (no metric, no metric_params). Under the valid-metric branch at 174-175 `hdb.fit(X)` is called with the actual (algo, metric) combination, but no assertion follows. Additionally, when `algo in ("brute", "auto")` the function returns at 149, so the `metric` parametrization is a no-op — the same identical assertion is repeated across every metric.
scenario: "a regression breaks kdtree+manhattan (returns garbage labels but doesn't raise) → parametrized run is green because `hdb.fit(X)` output is never checked"
contract: for every non-erroring (algo, metric) combination, compute `labels` and assert `len(set(labels) - OUTLIER_SET) == n_clusters_true` (and `fowlkes_mallows_score >= 0.98` for combos where the clustering is expected to be good) so the parametrization actually validates behavior.
instances: single-instance

### F18 — `cluster_selection_method="leaf"` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — grep for `cluster_selection_method` returns only `"eom"` (lines 340, 354); the `leaf` branch in `sklearn/cluster/_hdbscan/_tree.pyx:761-782` (including the empty-`leaves` fallback at :763-766) has no test coverage.
scenario: "Any regression in the `leaf` branch of `_get_clusters` (e.g. `is_cluster[condensed_tree['parent'].min()] = True` fallback, or `epsilon_search` interaction) ships silently → users passing `cluster_selection_method='leaf'` get wrong labels with no test failure."
contract: Add at least one parametrized test that runs `HDBSCAN(cluster_selection_method="leaf")` on `X` and verifies both cluster count and label sanity (including with `cluster_selection_epsilon > 0`).
instances: single-instance

### F19 — `max_cluster_size` parameter is completely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py grep for `max_cluster_size` — zero occurrences. The parameter is declared in `_parameter_constraints` (`sklearn/cluster/_hdbscan/hdbscan.py:622-625`) and drives an override branch inside `_get_clusters` (`sklearn/cluster/_hdbscan/_tree.pyx:716-717,733`) where a cluster is split when `cluster_sizes[node] > max_cluster_size`.
scenario: "a regression in the `max_cluster_size` split path silently returns clusters larger than the requested limit → no test detects it"
contract: add a test that fits HDBSCAN on data producing one large natural cluster, then re-fits with `max_cluster_size=<smaller>` and asserts that no returned non-noise cluster exceeds `max_cluster_size` in bincount size.
instances: single-instance

### F20 — test_hdbscan_precomputed_non_brute exercises non-existent algorithm names
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` uses `"prims_kdtree"` / `"prims_balltree"`, which are NOT in `_parameter_constraints["algorithm"] = StrOptions({"auto", "brute", "kdtree", "balltree"})` in `sklearn/cluster/_hdbscan/hdbscan.py:629-638`.
scenario: "user calls `HDBSCAN(metric='precomputed', algorithm='kdtree').fit(X)` → the test asserts nothing about that combination; the actual test passes only because `_validate_params()` rejects the invalid algorithm string, so the precomputed-vs-tree logic (`hdbscan.py:772-783`) is never exercised."
contract: Rename to the valid values `"kdtree"` / `"balltree"` so the test actually verifies the precomputed-with-tree rejection its docstring claims.
instances: single-instance

### F21 — test_hdbscan_centers uses rtol=1, making the centroid/medoid check meaningless for the (3,3) center
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:318-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)`; for `center=(3,3)` the tolerance evaluates to `atol + rtol*|3| = 0.05 + 3 = 3.05` per component, i.e. the centroid could be anywhere in `[-0.05, 6.05]` and still pass.
scenario: "A regression that shifts the (3,3) centroid to (0.1, 0.1) or (5.9, 5.9) → the assertion still passes; the centroid computation could be arbitrarily broken for non-origin clusters and this test would not detect it."
contract: Use `rtol=0.05` (or `atol` only), matching the intent of "centroid accurate to ~5%".
instances: [sklearn/cluster/tests/test_hdbscan.py:319, sklearn/cluster/tests/test_hdbscan.py:320]

### F22 — `_weighted_cluster_center` excludes only labels `{-1, -2}`, not the `-3` missing label [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The `-3` "missing" outlier label defined at hdbscan.py:74 (`_OUTLIER_ENCODING["missing"]["label"]`) is *not* excluded from the count; and no test combines `store_centers` with data that contains `np.nan` rows.
scenario: "User fits `HDBSCAN(store_centers='centroid')` on data containing `np.nan` rows → `-3` is counted as an additional 'cluster', so `range(n_clusters)` iterates one index past the largest real cluster label; the resulting `mask = self.labels_ == idx` is all-False, `X[mask]` is empty, and `np.average` raises `ZeroDivisionError`/`Weights sum to zero`."
contract: Use the `_OUTLIER_ENCODING` labels (i.e. `set(self.labels_) - {-1} - {v['label'] for v in _OUTLIER_ENCODING.values()}`) so every outlier encoding is excluded when counting clusters, and add a test that combines `store_centers` with NaN-containing input.
instances: single-instance

### F23 — `test_hdbscan_min_cluster_size` silently no-ops when every point is noise
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:259-263 — `for min_cluster_size in range(2, len(X), 1):` iterates up to 199. For large `min_cluster_size` values, all labels equal `-1`, so `true_labels` is empty and the `if len(true_labels) != 0:` guard skips the assertion; those iterations verify nothing.
scenario: "a regression that returns noise for ALL points at moderately-large min_cluster_size → test still passes because every non-trivial iteration hits the skipped branch"
contract: replace the silent skip with an explicit assertion for the noise-only case, e.g. `else: assert set(labels) == {-1}` (or bound the loop range so every iteration produces at least one cluster), so no iteration passes without an assertion.
instances: single-instance

### F24 — test_hdbscan_sparse does not verify np.nan rows receive the missing-outlier label
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:297-301 — after `sparse_X_nan[0, 0] = np.nan` and `labels = HDBSCAN().fit(sparse_X_nan).labels_`, the test only checks `n_clusters = len(set(labels) - OUTLIER_SET) == 3`; it never asserts `labels[0] == _OUTLIER_ENCODING["missing"]["label"]`.
scenario: "A regression in the `_get_finite_row_indices` / `_OUTLIER_ENCODING['missing']` remapping for sparse inputs (`hdbscan.py:714-744`, `830-852`) that fails to label the nan row as `-3` → test still passes because `-3` ∈ OUTLIER_SET is subtracted before counting."
contract: Add `assert labels[0] == _OUTLIER_ENCODING["missing"]["label"]` (and that the remaining rows are not `-3`).
instances: single-instance

### F25 — `store_centers` "centroid"-only and "medoid"-only variants are untested
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:316,324 — the only `store_centers` invocations pass `"both"`. The `_weighted_cluster_center` helper (`sklearn/cluster/_hdbscan/hdbscan.py:894-921`) has separate `make_centroids`/`make_medoids` toggles and does *not* create the unused attribute; if only `"centroid"` is requested, `self.medoids_` must NOT exist (and vice versa). No test guards this.
scenario: "a regression sets both `self.centroids_` and `self.medoids_` regardless of `store_centers` value → tests pass because only `'both'` is exercised, but user code that inspects `hasattr(hdb, 'medoids_')` silently breaks"
contract: parametrize `test_hdbscan_centers` (or add a companion test) over `["centroid", "medoid", "both"]`, asserting both that the requested attribute is present and correct, and that the un-requested attribute is absent.
instances: single-instance

### F26 — Sparse mutual-reachability `max_distance > 0` fallback and `INFINITY` core-distance branches are untested
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:199-200 (`else: core_distances[i] = INFINITY` when `further_neighbor_idx >= row_data.size`) and sklearn/cluster/_hdbscan/_reachability.pyx:211-212 (`elif max_distance > 0: data[i] = max_distance`). `sklearn/cluster/_hdbscan/tests/test_reachibility.py` never constructs a sparse matrix with an under-populated row and never passes a non-zero `max_distance`.
scenario: "a regression breaks the max_distance truncation of non-finite mutual reachabilities (e.g., writes `max_distance` even for finite entries) → both dedicated test files stay green because no case triggers the branch"
contract: add tests in `test_reachibility.py` that (a) call `mutual_reachability_graph` with a sparse row shorter than `min_samples` and assert that the resulting core-distance is `inf`, and (b) call it with a sparse matrix containing enough missing distances to yield an infinite mutual reachability plus `max_distance > 0`, and assert that infinities are replaced by `max_distance`.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:199, sklearn/cluster/_hdbscan/_reachability.pyx:199-200, sklearn/cluster/_hdbscan/_reachability.pyx:211, sklearn/cluster/_hdbscan/_reachability.pyx:211-212]

### F27 — test_hdbscan_centers parametrizes `algorithm` but the first assertion ignores it
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:308-320 — `@pytest.mark.parametrize("algorithm", ALGORITHMS)` runs the test 4×, but the first fit at :316 is `HDBSCAN(store_centers="both").fit(H)` with no `algorithm=` — only the second fit at :322-325 uses the parametrized `algorithm`. The centroid/medoid accuracy check therefore runs 4× against the same "auto" algorithm.
scenario: "The centroid/medoid computation path for `algorithm='brute'` / `'kdtree'` / `'balltree'` is never covered by the accuracy assertion → a regression that affects only non-auto algorithms passes the tolerance check because it's always executed under `algorithm='auto'`."
contract: Pass `algorithm=algorithm` to the first HDBSCAN construction at :316 so the centroid/medoid accuracy is verified once per algorithm.
instances: single-instance

### F28 — `n_jobs` documented default is `None` but constructor default is `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the text "`None` means 1 unless in a :obj:`joblib.parallel_backend` context", while sklearn/cluster/_hdbscan/hdbscan.py:658 declares the parameter as `n_jobs=4` in `HDBSCAN.__init__`.
scenario: "User reads docs → constructs `HDBSCAN()` expecting single-threaded / joblib-parallel-backend-controlled execution → estimator spawns 4 worker jobs unconditionally"
contract: Change the constructor default to `n_jobs=None` to match the documented, parameter-constraint-permitted (`[Integral, None]`) contract and the sibling `_hdbscan_brute` / `_hdbscan_prims` helpers (which both default `n_jobs=None`).
instances: single-instance

### F29 — `_weighted_cluster_center` receives reduced X but full-length labels when non-finite samples exist
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-910 — `X = X[finite_index]` reduces X to (n_finite, n_features); later `self.labels_ = new_labels` is expanded back to (n_raw,); then `self._weighted_cluster_center(X)` is called with the reduced X while `mask = self.labels_ == idx` produces a mask of length n_raw and `data = X[mask]` boolean-indexes X (length n_finite) with that longer mask.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_or_inf)` → `IndexError: boolean index did not match indexed array along dimension 0` from `X[mask]` inside `_weighted_cluster_center`"
contract: `_weighted_cluster_center` must be given data whose leading axis matches `self.labels_`; pass `self._raw_data` (and skip the outlier rows via the mask) rather than the finite-only `X`.
instances: single-instance

### F30 — `remap_single_linkage_tree(non_finite=...)` docstring says `ndarray` (boolean array) but caller passes a `set`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-365 documents `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`; the sole call site at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` (a `set` of raw integer indices). The body at hdbscan.py:369, 383 relies on `len(non_finite)` (count of unique indices) and hdbscan.py:388 iterates `enumerate(non_finite)` treating each element as a raw sample index — behavior that would silently give the wrong `outlier_tree` if a caller followed the docstring and passed a boolean mask.
scenario: "External caller (or future maintainer) reads the docstring, passes a boolean ndarray of length n_samples → `len(non_finite)` returns n_samples instead of the outlier count, `outlier` in the loop iterates the booleans (0/1) instead of raw indices, producing a corrupt hierarchy"
contract: Update the docstring to state `non_finite : set of int  A set of raw-data row indices that are non-finite (union of infinite and missing indices)`, and rename the parameter or type-annotate accordingly so the declared type matches what the function actually consumes.
instances: single-instance

### F31 — Length-1 ndarray coerced to scalar / used in `if` context in `_do_labelling` and `traverse_upwards`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 assigns `parent_lambda = lambda_array[child_array == n]` (an ndarray, not cdef-typed) and then `_tree.pyx:505` writes `if parent_lambda >= threshold:` — a length-1 boolean-array truth test. sklearn/cluster/_hdbscan/_tree.pyx:582 declares `cdef cnp.intp_t root, parent` and `_tree.pyx:586` assigns `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` (an ndarray) into the scalar; `_tree.pyx:593` likewise assigns `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']` (an ndarray) into `cdef cnp.float64_t parent_eps`. The comment at `_tree.pyx:496-497` acknowledges the pattern relies on the boolean mask matching exactly one row.
scenario: "NumPy ≥ 1.25 deprecates length-1-ndarray-to-scalar coercion → these coercions raise `DeprecationWarning` today and will raise `TypeError` in a future NumPy, breaking `_do_labelling`/`traverse_upwards` at runtime; if the invariant ever fails (e.g. duplicate child edge) the length-N array coercion raises `ValueError` with no diagnostic hinting at the real cause"
contract: Extract the scalar explicitly at each site — replace `lambda_array[child_array == n]` / `cluster_tree[...][field]` with `...[0]` (or `.item()`) so the RHS is a genuine scalar, mirroring the existing correct pattern at `_tree.pyx:621` (`eps = 1 / distances[leaf_nodes][0]`).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:498, sklearn/cluster/_hdbscan/_tree.pyx:505, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593]

### F32 — `UnionFind` cdef signatures disagree between `.pxd` and `.pyx` (`noexcept` present in one, absent in the other)
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; the corresponding definitions at sklearn/cluster/_hierarchical_fast.pyx:331 (`cdef void union(self, intp_t m, intp_t n):`) and sklearn/cluster/_hierarchical_fast.pyx:339 (`cdef intp_t fast_find(self, intp_t n):`) omit the `noexcept` specifier.
scenario: "Under Cython 3, a `cdef` function returning a value with no exception specifier defaults to propagating Python exceptions (equivalent to `except *`); the `.pxd` promises callers no exception propagation, but the definition does not opt into that guarantee → either a Cython signature-mismatch/deprecation warning at build time, or a silent semantic divergence between what the header advertises and what the body implements"
contract: Add `noexcept` to both definitions in `_hierarchical_fast.pyx` so the definition matches the header exactly (`cdef void union(self, intp_t m, intp_t n) noexcept:` and `cdef intp_t fast_find(self, intp_t n) noexcept:`).
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F33 — `labels_` docstring promises `-2` for infinite samples but precomputed path never assigns it
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:534-537 states "Samples with infinite elements (+/- np.inf) are given the label -2"; but the `-2` remapping at sklearn/cluster/_hdbscan/hdbscan.py:830-844 is gated on `self.metric != "precomputed"`. For `metric="precomputed"` (dense) the code at sklearn/cluster/_hdbscan/hdbscan.py:741-751 sets `force_all_finite=False`, keeps `np.inf` in the matrix, and never runs the outlier remap — so a row full of `np.inf` distances flows through `mst_from_mutual_reachability` untouched (warning only, line 256-265) and receives ordinary tree-derived labels (typically -1 or a cluster id).
scenario: "User passes a precomputed distance matrix containing `np.inf` entries → `labels_` for those samples is -1 (or a cluster label), never -2, silently violating the documented Attributes contract."
contract: Apply the `_OUTLIER_ENCODING` remap for infinite-distance samples on the precomputed path so `labels_` uniformly assigns `-2` to infinite-distance rows regardless of `metric`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:534, sklearn/cluster/_hdbscan/hdbscan.py:830, sklearn/cluster/_hdbscan/hdbscan.py:955]

### F34 — `test_hdbscan_precomputed_non_brute` uses non-existent algorithm names and passes for the wrong reason
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 sets `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"` / `"prims_balltree"`), but the `_parameter_constraints` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only accept `{"auto", "brute", "kdtree", "balltree"}`. `_validate_params()` therefore raises `InvalidParameterError` (a `ValueError` subclass) *before* the precomputed-vs-tree check at sklearn/cluster/_hdbscan/hdbscan.py:772-783 ever runs. The test only asserts `pytest.raises(ValueError)` with no `match=`, so it succeeds without ever exercising the code path its docstring claims to cover.
scenario: "Someone regresses the `metric=='precomputed'` + `algorithm=='kdtree'` guard → this test still passes because it fails at parameter validation instead, and the missing coverage lets the real regression ship."
contract: Change the test to `algorithm="kdtree"` / `algorithm="balltree"` and match the actual error message so it verifies the documented behaviour.
instances: single-instance

### F35 — `n_clusters` excludes only `{-1, -2}`, so label `-3` (missing) is counted as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`, but `_OUTLIER_ENCODING["missing"]["label"] == -3` (lines 74) and hdbscan.py:843 writes `-3` into `self.labels_` for missing samples.
scenario: "Fit with `store_centers` set and data containing `np.nan` → the -3 outlier label is treated as a real cluster id; `centroids_`/`medoids_` are allocated an extra row, and `range(n_clusters)` iterates one iteration too far, producing an empty-slice `np.average` warning/NaN row"
contract: enumerate the outlier labels from `_OUTLIER_ENCODING` (i.e., subtract `{-1} | {out["label"] for out in _OUTLIER_ENCODING.values()}`) so all outlier encodings are excluded.
instances: single-instance

### F36 — `_hdbscan_brute` signature `alpha=None` contradicts its `default=1.0` docstring and would crash when invoked with the default
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`; the docstring at sklearn/cluster/_hdbscan/hdbscan.py:183-184 says `alpha : float, default=1.0`; sklearn/cluster/_hdbscan/hdbscan.py:241 does `distance_matrix /= alpha`, which raises `TypeError: unsupported operand type(s) for /=: ... 'NoneType'` if `alpha` really is `None`.
scenario: "Any refactor that calls `_hdbscan_brute(X, ..., **kwargs)` without an explicit `alpha` (relying on the signature-advertised default) → immediate `TypeError` at `/= alpha`."
contract: Change the signature default to `alpha=1.0` to match the docstring and match the sibling `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.
instances: single-instance

### F37 — `_hdbscan_prims` docstring documents a `copy` parameter absent from the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature is `def _hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)` — no `copy`. Yet the docstring at sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False` with a full description. Also `leaf_size` (in the signature) is not documented in the docstring.
scenario: "Caller reading the docstring passes `copy=True` → it silently lands in `**metric_params` and is forwarded to the distance metric, producing a `TypeError` from the distance backend at fit time rather than being ignored as the docstring implies."
contract: Remove the `copy` paragraph from the `_hdbscan_prims` docstring and add a `leaf_size` entry so signature and docstring match exactly.
instances: single-instance

### F38 — `_condense_tree` uses an inconsistent, redundant `<cnp.intp_t>` cast on `right_count` only
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:177 assigns `left_count = hierarchy[left - n_samples].cluster_size` with no cast; sklearn/cluster/_hdbscan/_tree.pyx:182 assigns `right_count = <cnp.intp_t> hierarchy[right - n_samples].cluster_size`. The `HIERARCHY_t.cluster_size` field is already declared `intp_t` in sklearn/cluster/_hdbscan/_tree.pxd:38, and `right_count` is declared `cnp.intp_t` at sklearn/cluster/_hdbscan/_tree.pyx:155 — the cast is a no-op and diverges from the parallel left-side assignment for no reason.
scenario: "A future reviewer sees the asymmetric cast and copies the pattern elsewhere thinking it's meaningful → the misleading precedent propagates and later masks a real widening/truncation when the struct layout changes."
contract: Drop the `<cnp.intp_t>` cast on line 182 so both branches read the field with identical typing.
instances: single-instance

### F39 — `_tree.pyx` retains `cnp.*_t` typing throughout, contradicting the PR's stated typedef migration
severity: low
evidence: The PR description lists as change #1: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`". `_linkage.pyx` (line 42) and `_reachability.pyx` (line 40) both `cimport` `intp_t, float64_t, int64_t, uint8_t` from `...utils._typedefs`, and use the unqualified names. But `_tree.pyx` still imports numpy as `cnp` at sklearn/cluster/_hdbscan/_tree.pyx:33 and uses `cnp.intp_t / cnp.float64_t / cnp.uint8_t` in 90 spots (grep count) — including cdef declarations at sklearn/cluster/_hdbscan/_tree.pyx:39-40, 66-67, 82-91, 144-156, 240-249, 302-305, 329-337, 388-394, 467-474, 519-525, 692-700. The paired `_tree.pxd` already imports `intp_t, float64_t, uint8_t` from `_typedefs`, so `_tree.pyx` is the only member of `_hdbscan/` still on the `cnp.*_t` typing regime.
scenario: "A later reader checks whether `_typedefs` migration is complete for `_hdbscan` → sees the plan says yes, sees `_tree.pyx` still uses `cnp.*_t`, has to spend time reconciling; future numpy header decoupling work has to re-open a file the plan claimed was already done."
contract: Complete the migration in `_tree.pyx` by replacing all `cnp.intp_t / cnp.float64_t / cnp.uint8_t` occurrences with the unqualified `intp_t / float64_t / uint8_t` imported from `...utils._typedefs`, matching `_linkage.pyx` and `_reachability.pyx`.
instances: single-instance

### F40 — `_weighted_cluster_center` hardcodes outlier label set, drifting from `_OUTLIER_ENCODING` [out-of-theme]

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` while sklearn/cluster/_hdbscan/hdbscan.py:65-79 defines `_OUTLIER_ENCODING` with labels `-2` (infinite) and `-3` (missing), and the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:562 and :573 explicitly promises "the `-1, -2, -3` labels for the outlier clusters are excluded"
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing `np.nan` rows → `self.labels_` contains `-3` labels (per `_OUTLIER_ENCODING['missing']['label']`); `set(self.labels_) - {-1, -2}` retains `-3`, so `n_clusters` is inflated by one; `self.centroids_` is allocated with an extra row; `for idx in range(n_clusters)` iterates a nonexistent cluster id whose mask is all-False, producing a zero-weight `np.average` call that raises `ZeroDivisionError`/emits nans and violates the documented `n_clusters` contract"
contract: Derive the outlier exclusion set from `_OUTLIER_ENCODING` (i.e. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}`) rather than hardcoding `{-1, -2}`; test_hdbscan.py already models this pattern at line 37 with `OUTLIER_SET`.
instances: single-instance

### F41 — `store_centers` + non-finite input crashes because `X` is finite-filtered but `self.labels_` is raw-sized [out-of-theme]

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 reduces `X = X[finite_index]`; hdbscan.py:840-844 rebuilds `self.labels_` at `self._raw_data.shape[0]` (full raw length); hdbscan.py:854-855 then calls `self._weighted_cluster_center(X)` with the still-reduced `X`; inside `_weighted_cluster_center` (hdbscan.py:908-909) `mask = self.labels_ == idx` produces a boolean array whose length is the raw sample count, and `data = X[mask]` fails when that length differs from `X.shape[0]`.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_row)` → `X` shape mismatch with `mask` → `IndexError: boolean index did not match indexed array along dimension 0; dimension is <finite_n> but corresponding boolean dimension is <raw_n>`."
contract: Inside `_weighted_cluster_center`, index `self.labels_` by `self._finite_index` before comparing (`labels = self.labels_[self._finite_index]; mask = labels == idx`) so the mask and the finite-filtered `X` share a length.
instances: single-instance

### F42 — Duplicate `births` allocation is dead code

severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 — `largest_child = max(largest_child, smallest_cluster); births = np.full(largest_child + 1, np.nan, dtype=np.float64); [blank line]; births = np.full(largest_child + 1, np.nan, dtype=np.float64)`
scenario: "any HDBSCAN.fit() call → `_compute_stability` allocates the `births` array twice with identical arguments; the first allocation (line 252) is immediately overwritten by line 254 without being read, wasting one full allocation per fit and signalling a copy/paste error that could later mask a real change"
contract: Keep exactly one `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` in `_compute_stability`; remove line 252.
instances: single-instance

### F43 — Test uses stale `prims_*tree` algorithm strings that no longer exist

severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `@pytest.mark.parametrize("tree", ["kd", "ball"]) def test_hdbscan_precomputed_non_brute(tree): hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree"); with pytest.raises(ValueError): hdb.fit(X)` while `_parameter_constraints["algorithm"]` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 restricts `algorithm` to `{"auto", "brute", "kdtree", "balltree"}`
scenario: "running the test suite → `_validate_params()` rejects `algorithm='prims_kdtree'` because the string is not in the `StrOptions` constraint, raising `InvalidParameterError` (a `ValueError` subclass); `pytest.raises(ValueError)` catches it and the test 'passes', but the precomputed-vs-tree-based error paths at sklearn/cluster/_hdbscan/hdbscan.py:772-783 (the actual behavior the test claims to check) are never executed — the check has silently drifted into vacuous"
contract: Use the canonical algorithm names from the parameter constraint (`f"{tree}tree"` producing `"kdtree"`/`"balltree"`); the single source of truth for algorithm names is `_parameter_constraints["algorithm"]`.
instances: single-instance

### F44 — Documented `n_jobs` default (`None`) contradicts the constructor default (`4`)

severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring: `n_jobs : int, default=None` with `` `None` means 1 unless in a :obj:`joblib.parallel_backend` context. `` sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`.
scenario: "A user reads the documented default and expects single-threaded behavior in a serial environment (or under `parallel_backend`) → HDBSCAN silently spawns 4 workers, breaking pinned CPU budgets and the `joblib.parallel_backend` contract that other sklearn estimators honor."
contract: Change `__init__` to `n_jobs=None,` so the value matches the docstring and the sklearn-wide `n_jobs` convention.
instances: single-instance

### F45 — `PyArray_SHAPE` extern re-declared in `_linkage.pyx` instead of cimporting from `_tree.pxd`

severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; sklearn/cluster/_hdbscan/_tree.pxd:48-49 — same block, `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; and `_linkage.pyx` already cimports from `_tree` at sklearn/cluster/_hdbscan/_linkage.pyx:40
scenario: "someone changes the `PyArray_SHAPE` signature (e.g. adjusting the pointer type after a numpy API shift) → they update one location but not the other; the two declarations drift, but because each `.pyx` compiles its own translation unit, no compile error surfaces and misuse may go undetected until runtime"
contract: Declare `PyArray_SHAPE` in exactly one `.pxd` (already done in `_tree.pxd`) and `cimport` it from there in `_linkage.pyx`; delete the local `extern` block.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44, sklearn/cluster/_hdbscan/_tree.pxd:48]

### F46 — Cython typedefs inconsistently sourced across the new `_hdbscan` module

severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx uses `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` 90 times (e.g. lines 39, 40, 58, 61, 66, 67, 90, 145-155), whereas sklearn/cluster/_hdbscan/_linkage.pyx and _reachability.pyx cimport `intp_t`, `float64_t`, `int64_t`, `uint8_t` directly from `...utils._typedefs` (_linkage.pyx:42, _reachability.pyx:40) and use them exclusively. The plan of record explicitly claims "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`", but `_tree.pyx` was not migrated.
scenario: "Future refactor of `_typedefs.pxd` (e.g. shifting `intp_t` from `np.intp` to another width) → the two files diverge in their storage/API and the mixed usage inside the same module makes the drift harder to spot."
contract: Replace every `cnp.<type>_t` in `_tree.pyx` with the corresponding typedef cimported from `...utils._typedefs`, matching the style of `_linkage.pyx`/`_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:61, sklearn/cluster/_hdbscan/_tree.pyx:66, sklearn/cluster/_hdbscan/_tree.pyx:67, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:91, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:147, sklearn/cluster/_hdbscan/_tree.pyx:150, sklearn/cluster/_hdbscan/_tree.pyx:151, sklearn/cluster/_hdbscan/_tree.pyx:153, sklearn/cluster/_hdbscan/_tree.pyx:154, sklearn/cluster/_hdbscan/_tree.pyx:155, sklearn/cluster/_hdbscan/_tree.pyx:182, sklearn/cluster/_hdbscan/_tree.pyx:240, sklearn/cluster/_hdbscan/_tree.pyx:241, sklearn/cluster/_hdbscan/_tree.pyx:243, sklearn/cluster/_hdbscan/_tree.pyx:244, sklearn/cluster/_hdbscan/_tree.pyx:246, sklearn/cluster/_hdbscan/_tree.pyx:247, sklearn/cluster/_hdbscan/_tree.pyx:248, sklearn/cluster/_hdbscan/_tree.pyx:249, sklearn/cluster/_hdbscan/_tree.pyx:281, sklearn/cluster/_hdbscan/_tree.pyx:286, sklearn/cluster/_hdbscan/_tree.pyx:289]

### F47 — `HDBSCAN.fit` duplicates the algorithm→kwargs mapping across the explicit and `"auto"` branches

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:793-803 (explicit branch: `if self.algorithm == "brute": mst_func = _hdbscan_brute; kwargs["copy"] = self.copy; elif self.algorithm == "kdtree": mst_func = _hdbscan_prims; kwargs["algo"] = "kd_tree"; kwargs["leaf_size"] = self.leaf_size; elif self.algorithm == "balltree": mst_func = _hdbscan_prims; kwargs["algo"] = "ball_tree"; kwargs["leaf_size"] = self.leaf_size`) and sklearn/cluster/_hdbscan/hdbscan.py:805-818 (auto branch: identical three assignment blocks selecting `_hdbscan_brute`/`_hdbscan_prims` with `"kd_tree"`/`"ball_tree"`)
scenario: "a maintainer adds a new tree algorithm (or renames `leaf_size` → `_leaf_size`) → they update one of the two dispatch blocks and forget the other, silently changing behavior only when `algorithm='auto'` (or vice-versa); the identical wiring pattern for the same three algorithms is copy-pasted twice"
contract: Resolve `algorithm` to a concrete choice from `{"brute", "kdtree", "balltree"}` once (the "auto" branch resolves it into one of the three), then apply a single mapping block that fills `mst_func` and `kwargs` — one source of truth for the algorithm→dispatch table.
instances: single-instance

### F48 — Comment claims label mapping (`inf → -1`, `nan → -2`) inconsistent with `_OUTLIER_ENCODING`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2"; the code immediately below (lines 842-843) actually maps `inf → -2` and `nan → -3` via `_OUTLIER_ENCODING`.
scenario: "future maintainer trusts the comment and adjusts downstream label handling to match `-1/-2` instead of `-2/-3` → introduces a semantic regression"
contract: rewrite the comment to state the actual mapping (`inf → -2`, `nan → -3`) and reference `_OUTLIER_ENCODING` as the source of truth.
instances: single-instance

### F49 — `_hdbscan_prims` docstring documents a nonexistent `copy` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature has no `copy` parameter; sklearn/cluster/_hdbscan/hdbscan.py:313-318 — docstring still contains the `copy : bool, default=False ...` block copy-pasted from `_hdbscan_brute` (lines 207-212).
scenario: "user reads the `_hdbscan_prims` docstring and calls it with `copy=True` → `TypeError: _hdbscan_prims() got an unexpected keyword argument 'copy'`, or (more commonly) reader is misled about the function's contract"
contract: delete the `copy` parameter block from `_hdbscan_prims`'s docstring; the single source of truth for `copy` is `_hdbscan_brute` (and the public `HDBSCAN.copy`).
instances: single-instance

### F50 — `_hdbscan_brute` / `_hdbscan_prims` docstring defaults drift from actual signatures
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160-161 — actual defaults `min_samples=5, alpha=None`; docstring 179 and 183 say `min_samples : int, default=None` and `alpha : float, default=1.0`. Same drift in `_hdbscan_prims`: sklearn/cluster/_hdbscan/hdbscan.py:272 has `min_samples=5` while sklearn/cluster/_hdbscan/hdbscan.py:290 says `default=None`.
scenario: "reader/maintainer takes the doc-stated defaults at face value and calls `_hdbscan_brute(X)` expecting `alpha=1.0` → gets `alpha=None`, which then produces a `TypeError` on the `distance_matrix /= alpha` line (241)"
contract: reconcile each docstring `default=...` clause with the actual signature default, using the signature as the source of truth.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:161, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:272, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F51 — `plot_hdbscan.py` scale-invariance loop does not scale `X`, contradicting the paired DBSCAN loop
severity: low
evidence: examples/cluster/plot_hdbscan.py:87-89 — DBSCAN demo scales input as `dbs.fit(X * scale)`; examples/cluster/plot_hdbscan.py:106-110 — mirroring HDBSCAN demo calls `hdb.fit(X)` instead of `hdb.fit(X * scale)`, then labels the plot with the (unused) scale value.
scenario: "reader executes the notebook expecting three visibly different clusterings at scales `(1, 0.5, 3)` demonstrating HDBSCAN's scale-invariance → sees three identical panels (all fit on unscaled X), and the pedagogical claim in surrounding prose is not actually demonstrated"
contract: replace `hdb.fit(X)` with `hdb.fit(X * scale)` (and pass `X * scale` into `plot(...)`) to mirror the DBSCAN loop and single-source the scaling behavior across the two paired demos.
instances: single-instance

### F52 — Dead `mask = np.empty(...)` allocation immediately overwritten [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)` is assigned before any use, and the very first statement in the loop at line 908 reassigns it as `mask = self.labels_ == idx`; nothing between lines 896 and 908 reads it.
scenario: "Every `store_centers` fit reaches line 896 → an unused bool array of length `n_samples` is allocated and immediately overwritten, wasting `n_samples` bytes and one heap allocation per fit."
contract: Delete the dead allocation on line 896.
instances: single-instance

### F53 — Stale algorithm strings in `test_hdbscan_precomputed_non_brute`
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282-290 — the test constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` with `tree ∈ {"kd","ball"}`, producing `algorithm="prims_kdtree"` or `"prims_balltree"`. Neither value is in the `StrOptions({"auto","brute","kdtree","balltree"})` constraint declared on hdbscan.py:635-644, so `_validate_params` rejects the parameter before `fit` ever reaches the `if self.algorithm != "auto": ... "Sparse data matrices only support algorithm 'brute'."` or precomputed-plus-non-brute branches the test claims to exercise.
scenario: "Test parametrized as `test_hdbscan_precomputed_non_brute[kd|ball]` runs → `_validate_params` raises `InvalidParameterError` (a `ValueError` subclass), `pytest.raises(ValueError)` passes → the intended behavior ('precomputed + tree algorithm should error') is silently uncovered; a future regression that permitted `algorithm='kdtree'` with `metric='precomputed'` would go undetected by this test."
contract: Change the algorithm literal to the current valid tree name — `HDBSCAN(metric="precomputed", algorithm=f"{tree}tree")` — so the test actually reaches the precomputed-vs-tree-algorithm path it documents.
instances: single-instance

### F54 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented default `None` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4,` in the constructor signature, while the class docstring (lines 492-496) states "n_jobs : int, default=None ... `None` means 1 unless in a `joblib.parallel_backend` context." No parameter constraint pins this; the `_parameter_constraints` entry for `n_jobs` (line 646) accepts `[Integral, None]`.
scenario: "A user constructs `HDBSCAN()` expecting the documented single-threaded behavior → instead 4 worker threads are silently spawned inside `pairwise_distances` (through `kwargs['n_jobs']=self.n_jobs`) → surprising CPU usage and non-reproducible behavior against the sklearn convention that every estimator's default is `n_jobs=None`."
contract: Change the constructor signature to `n_jobs=None,` so the default matches both the docstring and sklearn's global convention.
instances: single-instance

### F55 — `plot_hdbscan.py` "Scale Invariance" loop never rescales `X` [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:113-116 — `hdb = HDBSCAN(); for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`. The scale variable is threaded only into the plot label — `fit` receives the unscaled `X` and `plot` receives the unscaled `X` in every iteration, so all three subplots show identical clusterings against identical coordinates. Contrast with the DBSCAN loop above (lines 92-95) which correctly calls `dbs.fit(X * scale)` / `plot(X * scale, ...)`.
scenario: "Reader opens the rendered gallery example titled 'Scale Invariance' → three panels look identical with different `scale=` annotations → the demonstration of HDBSCAN's scale invariance is fake because no scaling ever occurs."
contract: Replace `hdb.fit(X)` and `plot(X, ...)` inside the loop with `hdb.fit(X * scale)` and `plot(X * scale, ...)`, matching the DBSCAN comparison block immediately above.
instances: single-instance

### F56 — Duplicate `births = np.full(...)` assignment (first is dead, immediately overwritten)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two identical statements back-to-back:
```
    births = np.full(largest_child + 1, np.nan, dtype=np.float64)

    births = np.full(largest_child + 1, np.nan, dtype=np.float64)
```
No read of `births` occurs between the two writes, so the first allocation is unreachable/wasted.
scenario: "Every call to `_compute_stability` (one per `HDBSCAN.fit`) allocates and discards a `largest_child+1`-length float64 array before immediately overwriting it → dead code that directly contradicts the PR-description item 'Trimmed unused variables (thanks to Cython linting pre-commit)'"
contract: Delete the redundant first `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252.
instances: single-instance

### F57 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist in the function signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — the Parameters section of `_hdbscan_prims` describes `copy : bool, default=False ...`, but the signature at lines 269-278 has no `copy` argument (only X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params)
scenario: "reader / documentation build inspects `_hdbscan_prims` → sees a documented `copy` parameter that cannot be passed and has no effect; likely copy-paste residue from `_hdbscan_brute` which does accept `copy`"
contract: Remove the `copy` parameter block from the `_hdbscan_prims` docstring.
instances: single-instance

### F58 — `_hdbscan_brute` default `alpha=None` is dead-and-broken code (docstring says `default=1.0`, and the only path that would ever hit `None` would crash on `distance_matrix /= alpha`)
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — signature `alpha=None,`; docstring at lines 183-184 declares `alpha : float, default=1.0`; the body unconditionally executes `distance_matrix /= alpha` on line 241
scenario: "any direct caller of `_hdbscan_brute` relying on the documented default → `distance_matrix /= None` raises TypeError; sole in-tree caller (`HDBSCAN.fit`) always passes `self.alpha` (default 1.0) so the None default is never actually exercised — it is dead code that also contradicts documented behavior"
contract: Change the signature default to `alpha=1.0` to match the docstring and to prevent latent breakage of the internal API.
instances: single-instance

### F59 — Dead `mst_func = None` initialization in `HDBSCAN.fit`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:763 — `mst_func = None` is placed immediately before the `if self.algorithm != "auto": ... else: ...` block. `_parameter_constraints` (lines 636-644) restricts `algorithm` to `{"auto","brute","kdtree","balltree"}`, and every branch inside both arms assigns `mst_func` to `_hdbscan_brute` or `_hdbscan_prims`; the only exits without assignment are `raise ValueError` paths. The `None` sentinel can never reach `mst_func(**kwargs)` on line 820.
scenario: "Static readers see a `None` initialization suggesting `mst_func(**kwargs)` might be invoked on `None` → misleading future maintenance; the sentinel is otherwise unreachable so cannot be exercised."
contract: Remove the `mst_func = None` line; rely on the exhaustive assignment guaranteed by `_validate_params`.
instances: single-instance

### F60 — `_weighted_cluster_center` breaks on non-finite input with `store_centers` [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 reduces `X = X[finite_index]` to only finite rows; sklearn/cluster/_hdbscan/hdbscan.py:840-844 rebuilds `self.labels_` to `self._raw_data.shape[0]`; sklearn/cluster/_hdbscan/hdbscan.py:855 then calls `self._weighted_cluster_center(X)` with the *reduced* X but sklearn/cluster/_hdbscan/hdbscan.py:908-909 does `mask = self.labels_ == idx` (raw length) then `data = X[mask]` (reduced length) → boolean-index length mismatch → IndexError.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_rows)` → IndexError: boolean index did not match indexed array along dimension 0."
contract: index `self._raw_data` (or the equivalent finite-only labels) inside `_weighted_cluster_center` so mask and data share a shape.
instances: single-instance

### F61 — `remap_single_linkage_tree` docstring lies about `non_finite`: says "Boolean array" but caller passes a `set`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-366 — `non_finite : ndarray\n    Boolean array of which entries in the raw data are non-finite`; sklearn/cluster/_hdbscan/hdbscan.py:838 caller passes `non_finite=set(infinite_index + missing_index)` (a set of integer indices). Inside the function (line 388-389), `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)` — `outlier` is written into the tree's `left_node` field, which only makes sense if it is an integer index, not a boolean.
scenario: "A reader takes the docstring at its word and calls `remap_single_linkage_tree(tree, mapping, non_finite=np.isnan(X.sum(1)) | np.isinf(X.sum(1)))` → outlier_tree rows get `left_node ∈ {0,1}` (True/False cast), silently corrupting the extended tree."
contract: rewrite the parameter as `non_finite : iterable of int — raw-data indices of non-finite rows`.
instances: single-instance

### F62 — HDBSCAN `n_jobs` default: signature is `4`, docstring says `None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — the class docstring reads "`n_jobs : int, default=None ... ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.``-1`` means using all processors.`" ; line 658 sets `n_jobs=4` in the signature, and line 673 stores that value on `self.n_jobs`, which is later passed straight into `NearestNeighbors`/`pairwise_distances`.
scenario: "user reads the docstring and expects the estimator to behave like every other sklearn estimator (respect `joblib.parallel_backend`); instead the constructor spawns 4 processes unconditionally → surprise CPU/memory usage and non-composition with parallel_backend contexts"
contract: Set the default to `n_jobs=None` to match the docstring and sklearn-wide convention.
instances: single-instance

### F63 — Duplicated `births = np.full(...)` allocation is dead code
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — line 252 executes `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`, then line 254 (after a blank line) immediately overwrites it with the identical assignment. The first assignment is unconditionally discarded.
scenario: "Reader tries to understand `_compute_stability` → sees two identical allocations, hunts for a subtle difference, wastes review time; every fit pays for one throwaway numpy array allocation proportional to `largest_child + 1`"
contract: Delete the redundant `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252.
instances: single-instance

### F64 — `n_clusters` in `_weighted_cluster_center` omits the `-3` (missing) label [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; sklearn/cluster/_hdbscan/hdbscan.py:843 shows -3 is written into `labels_` for missing samples; the outlier encoding at sklearn/cluster/_hdbscan/hdbscan.py:65-79 also names -3 as an outlier.
scenario: "Data with missing rows → labels_ contains `{-1, -3, 0, ..., k-1}` → `n_clusters = k+1` → `range(n_clusters)` iterates to `idx == k` for which no row matches → `np.average` over an empty slice raises."
contract: `n_clusters = len(set(self.labels_) - {-1, -2, -3})` to exclude every outlier label defined in `_OUTLIER_ENCODING`.
instances: single-instance

### F65 — `TreeUnionFind.is_component` is written but never read
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:330 declares `cdef cnp.uint8_t[::1] is_component`; line 337 initializes it; line 355 writes `self.is_component[x] = False` inside `find`; no reader anywhere in the codebase (`grep is_component sklearn/cluster` returns only these three write/init sites).
scenario: "Reader of `TreeUnionFind.find` sees a boolean-side-effect write and assumes the flag matters for correctness, spends time reasoning about it → the flag has zero observable effect; every path-compressing call also pays for a memoryview store that only exists as noise"
contract: Remove the `is_component` attribute, its allocation in `__init__`, and its write in `find`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:330, sklearn/cluster/_hdbscan/_tree.pyx:337, sklearn/cluster/_hdbscan/_tree.pyx:355]

### F66 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist and omits `algo` / `leaf_size`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature has params `(X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params)`; sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents a full `copy` block; there is no `copy=` in the signature, and `algo` / `leaf_size` (which are required/consumed) are undocumented.
scenario: "Caller reads the docstring and passes `_hdbscan_prims(X, copy=True, ...)` → TypeError: unexpected keyword; conversely a caller looking for `algo` finds no documentation and guesses."
contract: delete the fabricated `copy` block; document `algo` and `leaf_size`.
instances: single-instance

### F67 — `_hdbscan_prims` docstring claims `metric='precomputed'` support that the code does not provide
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-283 — `Builds a single-linkage tree (SLT) from the input data \`X\`. If \`metric="precomputed"\` then \`X\` must be a symmetric array of distances.`; sklearn/cluster/_hdbscan/hdbscan.py:328-347 uses `NearestNeighbors`/`DistanceMetric.get_metric`, neither of which accepts `"precomputed"`. The estimator itself routes precomputed only through `_hdbscan_brute` (hdbscan.py:793-803).
scenario: "Reader sees `_hdbscan_prims` docstring, believes prims supports precomputed, invokes it with `metric='precomputed'` → NearestNeighbors immediately rejects the metric."
contract: strip the precomputed sentence from `_hdbscan_prims`; describe it only as the tree-based path for feature arrays.
instances: single-instance

### F68 — `_hdbscan_brute` docstring `alpha : float, default=1.0` contradicts signature `alpha=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None,`; sklearn/cluster/_hdbscan/hdbscan.py:183-184 — `alpha : float, default=1.0`; hdbscan.py:241 does `distance_matrix /= alpha` which errors on the documented "default".
scenario: "Reader calls `_hdbscan_brute(X, metric='euclidean')` per the doc's default → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`."
contract: set the signature default to `alpha=1.0` (matching the docstring and the class default).
instances: single-instance

### F69 — `plot_hdbscan.py` "scale invariance" demonstration does not scale the data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ..., parameters={"scale": scale})`. Every iteration fits and plots the *same* `X` (compare with the DBSCAN loop at examples/cluster/plot_hdbscan.py:86-89 which correctly uses `X * scale`).
scenario: "Reader runs the example to see HDBSCAN's scale-invariance advantage → three identical subplots labelled `scale=1, 0.5, 3` → the ‘proof' is inert and misleads about what invariance means."
contract: fit and plot on `X * scale`, mirroring the DBSCAN loop just above.
instances: single-instance

### F70 — Inline comment lies about outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 says "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The code immediately below (lines 842-843) actually assigns `_OUTLIER_ENCODING["infinite"]["label"]` (= -2) for inf and `_OUTLIER_ENCODING["missing"]["label"]` (= -3) for nan. The comment values (-1, -2) do not match the code (-2, -3).
scenario: "Reader trusts the comment → uses `-1` to filter np.inf samples in downstream code and gets nothing, or `-2` to filter np.nan and misses them entirely"
contract: Update the comment to "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3."
instances: single-instance

### F71 — Misspelled `mutual_reachibility_distance` local diverges from `reachability` naming
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:127, :144, :149, :182, :206, :209, :210 all use the identifier `mutual_reachibility_distance` while the file/function/module all consistently use "reachability" (see line 44 `mutual_reachability_graph`, file name `_reachability.pyx`).
scenario: "Someone greps for `reachability` in the reachability code → misses the loop bodies that actually compute the reachability distance; a copy-paste of the variable name into new call-sites propagates the misspelling"
contract: Rename the local to `mutual_reachability_distance` at every occurrence (7 lines above), matching the module and function names.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F72 — Public docstrings render "single-linkage tree tree" and misspell "reahability"/"collecteion"
severity: low
evidence: The strings appear verbatim in user-visible NumPy-style docstrings that Sphinx will render: `single-linkage tree tree` at sklearn/cluster/_hdbscan/hdbscan.py:149, :220, :326, :361 and sklearn/cluster/_hdbscan/_linkage.pyx:234; `mutual-reahability graph` at sklearn/cluster/_hdbscan/hdbscan.py:102, :143 and sklearn/cluster/_hdbscan/_linkage.pyx:75, :137, :228; `collecteion of edges` at sklearn/cluster/_hdbscan/hdbscan.py:103, :144 and sklearn/cluster/_hdbscan/_linkage.pyx:76, :138, :229; `simbling` at sklearn/cluster/_hdbscan/_tree.pyx:503 and sklearn/cluster/tests/test_hdbscan.py:530; `smaler` at sklearn/cluster/_hdbscan/_tree.pyx:133.
scenario: "Docs build publishes broken prose in official scikit-learn documentation → user searches Sphinx for `reachability` and misses these entries; documentation credibility hit"
contract: Fix each misspelling to "reachability", "collection", "sibling", "smaller", and drop the doubled "tree" in "single-linkage tree tree" — a single sweep across the enumerated line-set is sufficient.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/tests/test_hdbscan.py:530]

### F73 — `_weighted_cluster_center` pre-allocates a `mask` buffer it never uses
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 allocates `mask = np.empty((X.shape[0],), dtype=np.bool_)`. Inside the loop at line 908 the very first statement rebinds `mask = self.labels_ == idx`, discarding the pre-allocated buffer. `mask` is never read between allocation and rebinding.
scenario: "Reader sees a pre-allocated typed buffer and assumes it's part of an in-place reuse optimization → wastes time proving no reuse exists; every fit with `store_centers` also allocates and immediately abandons the buffer"
contract: Delete line 896 — the loop's `mask = self.labels_ == idx` already produces the correct array.
instances: single-instance

### F74 — `enumerate(tree)` with `_` while indexing `tree[i]` misleads the reader
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370 uses `for i, _ in enumerate(tree):` and the loop body at lines 371-381 accesses `tree[i]["left_node"]`, `tree[i]["right_node"]`, and writes `tree[i]["left_node"] = ...`. The `_` implies "value unused", but each iteration re-fetches the same row that `enumerate` already produced.
scenario: "Reader parses `for i, _ in enumerate(tree)` as an index-only walk and misses that the loop mutates `tree[i]` in place → refactoring to `for entry in tree` is tempted and silently breaks because structured-array element access is view-vs-copy dependent"
contract: Replace with `for i in range(len(tree)):` — it accurately signals index-only iteration and matches the mutation intent.
instances: single-instance

### F75 — `_tree.pyx` docstrings pin single-linkage/condensed-tree arrays to shape `(n_samples,)` when they are `n_samples-1` long
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 (`hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`), :138 (condensed_tree output), :371 (`linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`), :447, :656. The construction at sklearn/cluster/_hdbscan/_linkage.pyx:252 (`single_linkage = np.zeros(n_samples - 1, dtype=HIERARCHY_dtype)`) is the authoritative shape.
scenario: "Downstream user sizes their preallocated linkage buffer using the documented `(n_samples,)` → off-by-one when they later pass it here."
contract: use `(n_samples - 1,)` (or an explicit `(n_edges,)`) in the docstrings for the SLT and condensed-tree parameters.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F76 — `_weighted_cluster_center` ends with a stray bare `return` after a series of side-effect assignments
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:921 — the helper is documented (lines 882-887) to "instead stores them in the `self.{centroids, medoids}_` attributes" rather than returning anything, yet it ends with `return` on line 921 which returns `None`. Reads as if the helper were previously value-returning.
scenario: "reader reaches the trailing `return` on a documented mutator → wonders what value they missed and pays a small but real cognitive tax that is out of line with the surrounding style (`fit` uses `return self`; other helpers omit it)"
contract: Remove the bare `return` line.
instances: single-instance

### F77 — `_hdbscan_brute` / `_hdbscan_prims` docstring `min_samples default=None` contradicts signature `min_samples=5`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160,272 signatures both use `min_samples=5`; sklearn/cluster/_hdbscan/hdbscan.py:179-181,290-292 docstrings both claim `default=None`.
scenario: "Reader believes omitting `min_samples` uses `None` → configures unrelated logic downstream (e.g. sets min_samples from external code only if not-None) → default of 5 silently applied."
contract: change both docstrings to `default=5`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F78 — `_do_labelling` names the sample count `root_cluster`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-490 — `root_cluster = np.min(parent_array)` then used as an array size (`result = np.empty(root_cluster, ...)`) and loop bound (`for n in range(root_cluster)`); the identifier reads as "id of the root cluster", but the semantics used here are "number of samples".
scenario: "A future editor changes the labelling convention so the smallest parent is no longer equal to `n_samples` → every use of `root_cluster` as a size silently breaks because the name concealed the invariant."
contract: rename to `n_samples` (or introduce `n_samples = root_cluster` and use it in the sizing/loop) so the size and the id are not overloaded on one identifier.
instances: single-instance

### F79 — `_brute_mst`: `mutual_reachability` parameter docstring header uses a misspelled name
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82 signature has `mutual_reachability`; sklearn/cluster/_hdbscan/hdbscan.py:91-93 docstring documents it as `mututal_reachability_graph:` (typo *mututal*, and different name). Also documents `min_samples : int, default=None` for a parameter that has no signature default (sklearn/cluster/_hdbscan/hdbscan.py:82).
scenario: "Reader searches the source for `mutual_reachability` and matches only the code / not the docstring → trusts the misspelled `mututal` when following a doc-generation warning and propagates the typo into downstream references."
contract: rename the docstring parameter to `mutual_reachability : {ndarray, sparse matrix}` and remove the phantom `default=None` for `min_samples`.
instances: single-instance

### F80 — `enumerate(non_finite)` over a `set` produces non-deterministic outlier ordering
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:388-391 iterates a Python set (constructed at sklearn/cluster/_hdbscan/hdbscan.py:838 via `set(infinite_index + missing_index)`); the loop then assigns `last_cluster_id`/`last_cluster_size` in that iteration order.
scenario: "Two processes with different `PYTHONHASHSEED` fit HDBSCAN on identical data with non-finite rows → resulting `_single_linkage_tree_` outlier-appended rows differ in id assignment → downstream reproducibility checks fail even though algorithmic output is equivalent."
contract: iterate `sorted(non_finite)` (or accept a sequence in the API) so the appended outlier tree is deterministic.
instances: single-instance

### F81 — Comment misdocuments outlier label mapping, contradicting `_OUTLIER_ENCODING`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — the comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." Immediately below, `new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` (which is -2 at line 73) and `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` (which is -3 at line 80). The class docstring at line 535-537 correctly documents -2 for infinite and -3 for missing.
scenario: "Future maintainer reads block comment while investigating non-finite handling → uses -1/-2 assumption to write dependent logic (e.g. new outlier subclass) and ships a mismatch against the actual encoding."
contract: Rewrite the comment to state that samples with `np.inf` are relabelled to `_OUTLIER_ENCODING["infinite"]["label"]` (-2) and samples with `np.nan` to `_OUTLIER_ENCODING["missing"]["label"]` (-3), matching the class docstring.
instances: single-instance

### F82 — User Guide names parameter `minimum_cluster_size` that does not exist
severity: high
evidence: doc/modules/clustering.rst:1057-1060 — "HDBSCAN can be smoothed with an additional hyperparameter `min_cluster_size` … components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples`". The public constructor at sklearn/cluster/_hdbscan/hdbscan.py:649 exposes `min_cluster_size`, not `minimum_cluster_size`; the same paragraph even switches between the two names.
scenario: "Reader copies the recommended snippet `minimum_cluster_size = min_samples` into `HDBSCAN(minimum_cluster_size=…)` → `TypeError: __init__() got an unexpected keyword argument 'minimum_cluster_size'` (or the arg is silently absorbed and ignored)."
contract: Replace every occurrence of `minimum_cluster_size` in the paragraph with `min_cluster_size`.
instances: [doc/modules/clustering.rst:1059, doc/modules/clustering.rst:1060]

### F83 — Class docstring lists `n_jobs` default as `None` but constructor defaults to `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 — "n_jobs : int, default=None" followed by "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context." But `__init__` at line 658 defines `n_jobs=4`, so calling `HDBSCAN()` runs with 4 workers, not 1.
scenario: "User reads docstring expecting single-threaded default behaviour and deploys `HDBSCAN()` on a resource-constrained machine → the estimator runs with 4 workers, causing unexpected multi-core usage / oversubscription."
contract: Change the docstring to `n_jobs : int, default=4` and rewrite the description to reflect the actual default (4 parallel jobs), or change the constructor default to `None` to match the documented contract.
instances: single-instance

### F84 — `_hdbscan_prims` docstring documents non-existent `copy` param and omits `algo`, `leaf_size`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature is `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric='euclidean', leaf_size=40, n_jobs=None, **metric_params)`, but the docstring (lines 285-322) documents `X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy`, `metric_params` — `copy` does not exist on this function, and required `algo` / `leaf_size` are undocumented.
scenario: "developer maintaining this file reads the docstring and assumes `_hdbscan_prims` accepts `copy=` (mirroring `_hdbscan_brute`), calls `_hdbscan_prims(..., copy=True)` from a new code path → TypeError; conversely, no docstring guidance exists for the required `algo` positional argument"
contract: Docstring parameter list must be the exact set of the function signature: drop the `copy` block, add `algo` and `leaf_size` sections mirroring the constructor's descriptions.
instances: single-instance

### F85 — `_sparse_mutual_reachability_graph` docstring documents a `distance_matrix` param that does not exist
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:152-179 — function signature is `_sparse_mutual_reachability_graph(data, indices, indptr, n_samples, further_neighbor_idx, max_distance)`, but the Parameters block documents only `distance_matrix`, `further_neighbor_idx`, `max_distance`. Neither the actual positional CSR components (`data`, `indices`, `indptr`, `n_samples`) nor their contracts are described.
scenario: "Contributor extending the sparse path consults the docstring for how to call this helper → writes a call using the phantom `distance_matrix` argument that fails at runtime, and remains misled about how CSR components are passed."
contract: Rewrite the Parameters section to describe `data`, `indices`, `indptr`, and `n_samples` as the CSR components with their shapes/dtypes, and delete the phantom `distance_matrix` entry.
instances: single-instance

### F86 — `_get_clusters` docstring advertises a `stabilities` return that is not returned
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:686-690 — Returns section lists three arrays including "stabilities : ndarray (n_clusters,) / The cluster coherence strengths of each cluster." The function actually ends at line 796 with `return (labels, probs)` — a two-tuple; no stability array is ever produced.
scenario: "Caller does `labels, probs, stabilities = _get_clusters(...)` based on the docstring → `ValueError: not enough values to unpack (expected 3, got 2)`."
contract: Delete the `stabilities` entry from the Returns block so it describes only `(labels, probabilities)`.
instances: single-instance

### F87 — `_do_labelling` docstring omits two of its five parameters
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:437-465 — signature declares `condensed_tree, clusters, cluster_label_map, allow_single_cluster, cluster_selection_epsilon`, but the Parameters block only documents the first three. `allow_single_cluster` and `cluster_selection_epsilon` are undocumented, even though the docstring's own prose ("The determination of some points as noise … is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters.") admits they are load-bearing.
scenario: "User of the public-facing tests (which import `_do_labelling`) reads the incomplete Parameters block → passes wrong types or values for the two undocumented parameters and produces silently wrong labels."
contract: Add Parameters entries for `allow_single_cluster : int` and `cluster_selection_epsilon : float` describing their roles in the epsilon-based noise assignment.
instances: single-instance

### F88 — `_hdbscan_brute`, `_hdbscan_prims`, and `_brute_mst` misstate `min_samples` default (and `_brute_mst` misnames its first parameter)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:179 "min_samples : int, default=None" for `_hdbscan_brute` whose signature at line 160 has `min_samples=5`; line 290 same doc for `_hdbscan_prims` whose signature at line 272 has `min_samples=5`; line 95 same doc for `_brute_mst` whose signature at line 82 makes `min_samples` a required positional (no default). Additionally, `_brute_mst`'s Parameters block at line 91 documents the first argument as `mututal_reachability_graph` (also a typo) while the actual parameter is named `mutual_reachability` (line 82).
scenario: "Contributor relies on the stated `default=None` and calls the helper without `min_samples` → gets the wrong number of neighbours (`_hdbscan_brute`/`_hdbscan_prims` silently use 5) or a `TypeError: missing 1 required positional argument` (`_brute_mst`)."
contract: Fix every entry to state the real default: `min_samples : int, default=5` for `_hdbscan_brute` and `_hdbscan_prims`, `min_samples : int` (required) for `_brute_mst`, and rename `_brute_mst`'s documented parameter from `mututal_reachability_graph` to `mutual_reachability` to match the signature.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:95, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F89 — `_condense_tree` docstring reports wrong shape for `hierarchy`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 "hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype". The function derives `n_samples = hierarchy.shape[0] + 1` at line 146, so the input's shape is actually `(n_samples - 1,)`. Every peer function that produces or consumes the hierarchy (`make_single_linkage` in _linkage.pyx:239, `labelling_at_cut` at _tree.pyx:377) documents the correct `(n_samples - 1,)` shape.
scenario: "Contributor writing new tests sizes a synthetic hierarchy array to `n_samples` rows per the docstring → `_condense_tree` computes `n_samples + 1` internally and downstream indexing goes off by one."
contract: State the correct shape: `hierarchy : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype`. Similarly correct the `condensed_tree` return-shape from `(n_samples,)` to describe that its size varies with the number of surviving edges.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F90 — RST math syntax broken by stray trailing colons and stray space in the HDBSCAN user guide
severity: medium
evidence: doc/modules/clustering.rst:1009-1010 — "removing any edges with value greater than :math:`\varepsilon`: from the original graph. Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." The trailing `:` after the closing backtick of each `:math:` role renders as a literal ":" following the math span. Line 1011 also has "at this staged marked" (should be "at this stage"). Line 1025-1026 renders "the fully -connected mutual reachability graph" with an orphan space before the hyphen.
scenario: "Sphinx builds the User Guide with the current markup → renders awkward '`ε`:' punctuation and the sentence 'are at this staged marked as noise', which reads nonsensically to end users."
contract: Delete the stray colons after both `` :math:`\varepsilon` `` tokens, change "at this staged" to "at this stage", and join "fully-connected" without the intervening whitespace.
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011, doc/modules/clustering.rst:1025, doc/modules/clustering.rst:1026]

### F91 — Docstrings list wrong parameter defaults for helper functions
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:101 (`min_samples : int, default=None` for `_brute_mst` whose signature `def _brute_mst(mutual_reachability, min_samples)` at line 82 has no default), :179 (`min_samples : int, default=None` — but signature line 160 has `min_samples=5`), :183 (`alpha : float, default=1.0` — but signature line 161 has `alpha=None`), :290 (`min_samples : int, default=None` — but `_hdbscan_prims` signature line 272 has `min_samples=5`).
scenario: "A developer reads a docstring, believes the documented default matches the signature and omits the argument expecting `alpha=1.0` → at runtime line 241 executes `distance_matrix /= alpha` with `alpha=None`, raising `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`."
contract: Correct each parameter's default in the docstring to match the actual signature (`_brute_mst.min_samples`: no default, `_hdbscan_brute.min_samples`: 5, `_hdbscan_brute.alpha`: `None`, `_hdbscan_prims.min_samples`: 5).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:101, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F92 — Example narrator claims scale-invariance demo but code never scales the fitted data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104-110 (narrative: "HDBSCAN is scale-invariant. ... One immediate advantage is that HDBSCAN is scale-invariant.") followed by lines 108-110 (`for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ..., parameters={"scale": scale})`). The `scale` loop variable is used only in the plot title — `hdb.fit(X)` and `plot(X, ...)` both use the unscaled `X`, so the three sub-plots are identical and do not demonstrate scale invariance.
scenario: "A user reads the example gallery to convince themselves HDBSCAN is scale-invariant → sees three identical panels labelled with different `scale` values but no evidence of actual rescaling; the narrative and code disagree and the intended pedagogical point is not conveyed."
contract: Fit and plot the rescaled data — `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` — so the code matches the "HDBSCAN is scale-invariant" narrative directly above it.
instances: single-instance

### F93 — Class docstring narration lists non-existent parameter name in error text
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:118-122 error message tells users to `Ensure your distance matrix has non-zero values for at least `min_sample`={min_samples} neighbors for each points` — the parameter is `min_samples` (with the trailing `s`), and grammar reads "for each points" instead of "for each point". [out-of-theme]
scenario: "user hits the ValueError, searches the estimator API for `min_sample` → nothing found, wastes time; the message also has a grammar error visible to end users"
contract: Rename to `min_samples` (the actual parameter) and fix "for each points" → "for each point".
instances: single-instance

### F94 — Example uses a `:ref:` target that does not exist (case mismatch)
severity: low
evidence: examples/cluster/plot_hdbscan.py:104 has `see :ref:`User Guide <HDBSCAN>`` — the only defined label is `.. _hdbscan:` at doc/modules/clustering.rst:971 (lowercase). ReST/Sphinx reference targets are case-sensitive; no `HDBSCAN` label exists in the docs tree.
scenario: "sphinx builds this example → the ref becomes an unresolved reference warning; the rendered example page has broken link text"
contract: Use `:ref:`User Guide <hdbscan>`` matching the declared label exactly.
instances: single-instance

### F95 — `dbscan_clustering` docstring has typo and dropped period
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:927 "particular cut_distance (or epsilon) DBSCAN* can be thought of as" — missing period between `(or epsilon)` and `DBSCAN*`, producing a run-on sentence; sklearn/cluster/_hdbscan/hdbscan.py:946 "Clusters smaller than this value with be called 'noise'" — `with be called` should be `will be called`; sklearn/cluster/_hdbscan/hdbscan.py:937 "and cluster smaller than `min_cluster_size`" — `cluster smaller` should be `clusters smaller` (subject/verb agreement).
scenario: "user reads the public method's docstring and encounters a run-on sentence and a "with be" grammar error → the method is a user-facing API entry point, so this is high-visibility"
contract: Add the missing period after `(or epsilon)`; `with be called` → `will be called`; `and cluster smaller than` → `and clusters smaller than`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:927, sklearn/cluster/_hdbscan/hdbscan.py:937, sklearn/cluster/_hdbscan/hdbscan.py:946]

### F96 — Broken narration in `_weighted_cluster_center` comment obscures the invariant
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 "Need to handle iteratively seen each cluster may have a different number of samples, hence we can't create a homogenous 3D array." The word `seen` should be `since` (or the sentence is otherwise malformed); "homogenous" is also nonstandard for "homogeneous".
scenario: "reader trying to understand why the loop can't be vectorized reads a nonsensical sentence and cannot recover the invariant (each cluster's mask has variable size) → the whole point of the comment is lost"
contract: `iteratively seen each cluster` → `iteratively since each cluster`; `homogenous` → `homogeneous`.
instances: single-instance

### F97 — `_do_labelling` docstring says "The determination of some points as noise is in large, single-cluster datasets is controlled by" — duplicated "is"
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:441-443 "The determination of some points as noise is in large, single-cluster datasets is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters." — grammatically broken (two auxiliary `is` verbs).
scenario: "reader trying to understand which parameters govern noise labeling parses the sentence with two `is` verbs → sentence is unparseable, so the intended mapping from `allow_single_cluster` / `cluster_selection_epsilon` to noise-labelling behavior is lost on the reader"
contract: Drop the first `is`: "The determination of some points as noise in large, single-cluster datasets is controlled by …".
instances: single-instance

### F98 — `whats_new/v1.3.rst` uses bare `:class:`DBSCAN`` where surrounding text uses fully-qualified names
severity: low
evidence: doc/whats_new/v1.3.rst:194 "Similarly to :class:`cluster.OPTICS`, it can be seen as a generalization of :class:`DBSCAN` by allowing…" — `DBSCAN` is the only unqualified `:class:` reference in the entry; both `cluster.OPTICS` (line 194) and `cluster.HDBSCAN` (line 192) use the `cluster.` prefix.
scenario: "sphinx may fail to resolve `:class:`DBSCAN`` unambiguously depending on intersphinx config → nitpicky warning at build time; the anchor text in the rendered changelog is inconsistent with the neighbors"
contract: Change `:class:`DBSCAN`` to `:class:`cluster.DBSCAN`` for consistency and to guarantee resolution.
instances: single-instance

### F99 — `_reachability.pyx` `mutual_reachability_graph` Returns section names the wrong (and misspelled) key
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:76-78 Returns block uses `mututal_reachability_graph:` (typo `mututal_`, and a bare colon rather than the numpydoc `name : type` format used elsewhere in the file).
scenario: "sphinx renders the Returns section without a properly formatted type/description separator → visually inconsistent with siblings, plus a visible misspelling"
contract: Fix to `mutual_reachability : {ndarray, sparse matrix} of shape (n_samples, n_samples)` — the actual function returns the modified input, whose canonical name is `distance_matrix` or `mutual_reachability`.
instances: single-instance

### F100 — Test file misnamed `test_reachibility.py` (misspelling)
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py — the module and file name are misspelled (`reachibility` → should be `reachability`). Same misspelling of the local variable `mutual_reachibility_distance` in sklearn/cluster/_hdbscan/_reachability.pyx:127, 144, 149, 182, 206, 209, 210.
scenario: "developer searches for `test_reachability` to locate the reachability tests → the ripgrep hit is empty because the file is misspelled `test_reachibility`, and the misspelled variable then surfaces in every git log and stack trace involving the Cython inner loop"
contract: Rename file to `test_reachability.py` and rename `mutual_reachibility_distance` → `mutual_reachability_distance` throughout `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:150, sklearn/cluster/_hdbscan/_reachability.pyx:155, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:188, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/_reachability.pyx:212]

### F101 — Example prose in `plot_hdbscan.py` has awkward `min_cluster_size` / `min_samples` grammar
severity: low
evidence: examples/cluster/plot_hdbscan.py:178 "Clusters smaller than the ones of this size will be left as noise." (redundant "the ones of"); examples/cluster/plot_hdbscan.py:181 "However values which too small will lead to false sub-clusters" (missing `are` — should be "values which are too small"); examples/cluster/plot_hdbscan.py:203 "`min_samples` better be tuned after finding a good value for `min_cluster_size`." (colloquial and grammatically broken).
scenario: "rendered example page uses ungrammatical English in explanatory paragraphs → poor first impression on new users being introduced to the estimator via `sphinx-gallery`"
contract: Rewrite each fragment: `smaller than the ones of this size` → `smaller than this size`; `values which too small` → `values which are too small`; `better be tuned after` → `should be tuned after`.
instances: [examples/cluster/plot_hdbscan.py:178, examples/cluster/plot_hdbscan.py:181, examples/cluster/plot_hdbscan.py:203]

### F102 — `remap_single_linkage_tree` docstring type of `non_finite` disagrees with the only caller
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370-371 — "non_finite : ndarray / Boolean array of which entries in the raw data are non-finite". The sole caller at line 838 passes `non_finite=set(infinite_index + missing_index)` — a Python `set` of integer indices, and the function body uses `len(non_finite)` and `enumerate(non_finite)` for both `outlier_count` and per-outlier `outlier_tree` row values (line 389-395). A Boolean mask would produce meaningless `outlier` values (0/1) and wrong `outlier_count` semantics.
scenario: "Contributor writing a second call site follows the docstring and passes a Boolean mask → `outlier_count` becomes the mask length (n) rather than the number of non-finite entries, and every `outlier_tree` row gets a `left_node` of 0 or 1 instead of the raw sample index."
contract: Correct the docstring to describe `non_finite` as a set (or iterable) of integer indices into the raw data marking the non-finite samples.
instances: single-instance

### F103 — Systemic doc typos: "reahability", "collecteion", duplicated "tree tree", "simbling", "smaler", "with be called", "homogenous"
severity: low
evidence: Recurring typos in newly-added docstrings and comments:
- "reahability" for "reachability" — sklearn/cluster/_hdbscan/_linkage.pyx:81, 143, 234; sklearn/cluster/_hdbscan/hdbscan.py:108, 149
- "collecteion" for "collection" — sklearn/cluster/_hdbscan/_linkage.pyx:82, 144, 235; sklearn/cluster/_hdbscan/hdbscan.py:109, 150
- "single-linkage tree tree" (duplicated word) — sklearn/cluster/_hdbscan/_linkage.pyx:240; sklearn/cluster/_hdbscan/hdbscan.py:155, 226, 332, 361
- "simbling" for "sibling" — sklearn/cluster/_hdbscan/_tree.pyx:509; sklearn/cluster/tests/test_hdbscan.py:536
- "smaler" for "smaller" — sklearn/cluster/_hdbscan/_tree.pyx:139
- "with be called" for "will be called" — sklearn/cluster/_hdbscan/hdbscan.py:952
- "homogenous" (informal) — sklearn/cluster/_hdbscan/hdbscan.py:912
scenario: "Reviewer runs grep-based doc searches and API surface skimming (`git grep reachability`, `git grep sibling`) → misses these locations, and long-term the misspellings ossify into related identifiers and get copy-pasted."
contract: Correct each typo to its intended spelling ("reachability", "collection", "single-linkage tree (dendrogram)" with the duplication removed, "sibling", "smaller", "will be called", "homogeneous").
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:81, sklearn/cluster/_hdbscan/_linkage.pyx:82, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:143, sklearn/cluster/_hdbscan/_linkage.pyx:144, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/_linkage.pyx:235, sklearn/cluster/_hdbscan/_linkage.pyx:240, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:108, sklearn/cluster/_hdbscan/hdbscan.py:109, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:150, sklearn/cluster/_hdbscan/hdbscan.py:155, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:226, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:332, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/hdbscan.py:912, sklearn/cluster/_hdbscan/hdbscan.py:952, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:139, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/_tree.pyx:509, sklearn/cluster/tests/test_hdbscan.py:530, sklearn/cluster/tests/test_hdbscan.py:536]

### F104 — Dense NaN/Inf detection recomputes `X.sum(axis=1)` twice per fit
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:721 computes `reduced_X = X.sum(axis=1)` for identifying NaN/Inf rows, and immediately after at hdbscan.py:731 calls `_get_finite_row_indices(X)` which (for the dense branch, hdbscan.py:406) runs `np.isfinite(matrix.sum(axis=1)).nonzero()` — a second full O(n·d) sum over the same array.
scenario: "fit(X) with any non-finite entry in a large dense X → two full-array reductions where one would suffice, doubling the memory-bandwidth cost of finite-row bookkeeping"
contract: Compute `reduced_X = X.sum(axis=1)` once, then derive `finite_index` from that same array (e.g. `np.isfinite(reduced_X).nonzero()[0]`) instead of re-summing inside `_get_finite_row_indices`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:721, sklearn/cluster/_hdbscan/hdbscan.py:406]

### F105 — Repeated `cluster_tree['parent' | 'child'] == v` full scans build an O(N²) cluster-selection loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:728-739 iterates `for node in node_list` and at line 729 evaluates `child_selection = (cluster_tree['parent'] == node)` — a full O(|cluster_tree|) linear scan per node; and at 737 calls `bfs_from_cluster_tree` whose step (line 294) is `np.isin(parents, process_queue)`, another O(|parents|·|queue|) scan.  `recurse_leaf_dfs` (line 562) and `traverse_upwards` (lines 586, 593) exhibit the same "filter by field equality" pattern once per recursion.
scenario: "large dataset producing an O(n) condensed cluster tree → EOM cluster selection performs Θ(n²) structured-array scans in `_get_clusters`, and epsilon-search on a deep tree multiplies that by additional per-leaf recursive scans"
contract: Group `cluster_tree` once by parent (e.g. build a `defaultdict(list)` of `parent → (child, value, size)` or a sorted-by-parent index with searchsorted boundaries) at entry to `_get_clusters` / `epsilon_search`, and consult that index inside the hot loops instead of re-scanning the structured array on every iteration.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:729, sklearn/cluster/_hdbscan/_tree.pyx:737, sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:562, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593, sklearn/cluster/_hdbscan/_tree.pyx:620, sklearn/cluster/_hdbscan/_tree.pyx:725]

### F106 — Sparse `_get_finite_row_indices` densifies via LIL and iterates in Python
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 uses `matrix.tolil().data`, which materializes a Python object array of per-row Python lists, then walks it with a Python list-comp calling `np.all(np.isfinite(row))` per row.
scenario: "large sparse precomputed distance matrix containing any non-finite entry → conversion to LIL and per-row Python iteration allocates one Python list per sample and forces per-row NumPy dispatch, giving O(nnz) work with a per-row Python-level constant instead of the vectorised O(nnz) `np.isfinite` over `matrix.data`"
contract: Do a single vectorised pass on the CSR representation: compute `bad = ~np.isfinite(matrix.data)` and use `np.add.reduceat(bad, matrix.indptr[:-1])` (or `np.diff(np.searchsorted(...))` over bad indices) to derive the fully-finite row mask, without going through LIL.
instances: single-instance

### F107 — `_dense_mutual_reachability_graph` allocates a full n×n temporary just to extract n core distances
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133-137 — `core_distances = np.ascontiguousarray(np.partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx])`.  `np.partition(..., axis=1)` allocates a full n×n copy of the distance matrix before the `[:, k]` column slice throws all of it away.
scenario: "brute-force HDBSCAN on a dense n=20 000 pairwise-distance matrix → an additional ~3 GB float64 temporary is allocated for a computation whose output is n floats, doubling peak memory of the brute path"
contract: Partition row-by-row into an O(n) scratch buffer (e.g. `for i in range(n_samples): row = distance_matrix[i].copy(); row.partition(k); core_distances[i] = row[k]`) so the temporary is O(n), not O(n²).
instances: single-instance

### F108 — `_dense_mutual_reachability_graph` recomputes each cell twice
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:142-155 — nested `for i in range(n_samples): for j in range(n_samples):` computes `max(core_distances[i], core_distances[j], distance_matrix[i, j])` and writes `distance_matrix[i, j]`. The expression is symmetric in `(i, j)` and the input matrix is required to be symmetric (`pairwise_distances` output or the `_allclose_dense_sparse(X, X.T)` check in `_hdbscan_brute`), so cell `(i, j)` and `(j, i)` are recomputed and rewritten with the same value.
scenario: "brute algorithm on any dense input of size n → the mutual-reachability rewrite performs 2× the necessary comparisons and writes on the hot path"
contract: Iterate only the upper (or lower) triangle (`for j in range(i, n_samples)`) and mirror the assignment to `distance_matrix[j, i]`, halving loop work.
instances: single-instance

### F109 — `_do_labelling` recomputes constant `parent_array == cluster` scan inside the per-sample loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-506 — inside `for n in range(root_cluster):`, the `allow_single_cluster` + single-cluster branch executes `threshold = lambda_array[parent_array == cluster].max()` at line 504, but that branch is only entered when `cluster == root_cluster` (guaranteed by the enclosing `elif` after `if cluster != root_cluster:`), so the max is invariant across iterations. Similarly `1 / cluster_selection_epsilon` at line 500 is constant.
scenario: "`allow_single_cluster=True`, `cluster_selection_epsilon=0.0`, dataset resolves to one root cluster with N samples → the O(len(condensed_tree)) boolean scan+max runs up to N times, yielding O(N·|tree|) work where O(|tree|) suffices"
contract: Hoist both `threshold` computations out of the `for n in range(root_cluster):` loop; compute once before the loop starts.
instances: single-instance

### F110 — `_do_labelling` per-sample `child_array == n` scan is O(N) per iteration
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 — `parent_lambda = lambda_array[child_array == n]` scans the entire condensed tree once for every `n` in `range(root_cluster)`. The comment above the line acknowledges the child value is unique, so a precomputed mapping would give O(1) lookup.
scenario: "`allow_single_cluster=True` single-cluster path with N samples → O(N²) child-array scans instead of the O(N + |tree|) achievable with a `dict` built once from `zip(child_array, lambda_array)`"
contract: Build a `child → lambda` dict once before the loop (e.g., `child_to_lambda = dict(zip(child_array, lambda_array))`) and replace `lambda_array[child_array == n]` with a direct lookup.
instances: single-instance

### F111 — `_get_clusters` EOM loop: per-node full mask scan of `cluster_tree`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — `for node in node_list: child_selection = (cluster_tree['parent'] == node); subtree_stability = np.sum([stability[child] for child in cluster_tree['child'][child_selection]])` and, in the else-branch, `for sub_node in bfs_from_cluster_tree(cluster_tree, node): ...`
scenario: "large tree passed to `HDBSCAN.fit` with `cluster_selection_method='eom'` (default) → per-node boolean mask over the whole `cluster_tree` for every internal cluster gives O(|node_list| × |cluster_tree|) work in the mainline hot path; a single precomputed `parent → children` adjacency dict makes the same sweep O(|cluster_tree|)"
contract: Build a `parent → list-of-children` (or `parent → list-of-(child, stability)`) dict once up-front from `cluster_tree` and index it in the EOM loop (and in `bfs_from_cluster_tree`) rather than re-masking the array per node.
instances: single-instance

### F112 — `traverse_upwards` scans full `cluster_tree` twice per recursion level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586-602 — `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` then `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']`, followed by tail-recursive call
scenario: "`cluster_selection_epsilon > 0` with many leaves near the root → for each leaf, every level of ancestor traversal performs two full O(|cluster_tree|) mask scans of `cluster_tree['child']`, giving O(leaves × depth × |cluster_tree|) worst-case work in `epsilon_search`"
contract: Precompute a `child → (parent, value)` dict once and follow parent pointers in O(1) per step; the recursive scans must not remain in the per-leaf loop.
instances: single-instance

### F113 — `recurse_leaf_dfs` combines per-recursion full scan with O(n²) `sum([...], [])`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:562-566 — `children = cluster_tree[cluster_tree['parent'] == current_node]['child']` then `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`
scenario: "`cluster_selection_method='leaf'` on a tree with many leaves → each recursion masks all of `cluster_tree` again, and `sum([...], [])` reallocates the growing leaf list on every element, so leaf-finding is O(nodes × |cluster_tree|) with an extra O(leaves²) from the list concatenation"
contract: Build a `parent → children` dict once, and replace `sum([...], [])` with `itertools.chain.from_iterable(...)` or an in-place-appending accumulator so the concat is O(leaves).
instances: single-instance

### F114 — `_do_labelling` single-cluster branch: per-iteration full mask scans and non-hoisted constant threshold
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-508 — inside `for n in range(root_cluster):`, the `elif len(clusters) == 1 and allow_single_cluster:` branch runs `parent_lambda = lambda_array[child_array == n]` and `threshold = lambda_array[parent_array == cluster].max()` (with `cluster == root_cluster` fixed) per iteration
scenario: "`allow_single_cluster=True` on data producing one cluster with n samples → each iteration scans `child_array` and `parent_array` (each size = |condensed_tree|) again; `threshold` is invariant across the loop (`cluster` is constant here and `cluster_selection_epsilon` is a scalar) but is still recomputed n times, giving O(n × |condensed_tree|) work in what could be O(n) after hoisting"
contract: Precompute `threshold` and a `child → lambda` dict once before the `for n` loop; use dict lookups inside.
instances: single-instance

### F115 — `epsilon_search` uses a `list` for the "already processed" set
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:612 declares `list processed = list()`, line 623 tests `if leaf not in processed` and line 634 does `processed.append(sub_node)`.  Membership on a list is O(len(processed)).
scenario: "cluster tree with many leaves under a common epsilon-ancestor → `leaf not in processed` becomes an O(k²) accumulator over the leaves already merged into ancestor subtrees"
contract: Declare `processed` as a `set` and use `add` / `in` for O(1) membership.
instances: single-instance

### F116 — `bfs_from_cluster_tree` rescans the full `parents` array on every BFS layer
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-296 — the loop body is `process_queue = children[np.isin(parents, process_queue)]`, which walks the entire `parents` array once per BFS layer. Called from `_get_clusters` at line 737 (`for sub_node in bfs_from_cluster_tree(cluster_tree, node)`) inside `for node in node_list`, so overlapping subtrees are re-walked from each retained cluster.
scenario: "EOM selection on a hierarchy where many candidate clusters are retained → each retained cluster triggers a fresh `np.isin(parents, queue)` scan per BFS layer, giving O(retained · depth · |tree|) work on the cluster-selection hot path"
contract: Build a `parent → list-of-children` adjacency (e.g. via `np.argsort(parents)` + `np.searchsorted`) once at the top of `_get_clusters` and reuse it inside `bfs_from_cluster_tree`, dropping the per-layer full-array scan.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]

### F117 — `_hdbscan_brute` unconditionally executes `distance_matrix /= alpha` even when `alpha == 1.0`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:241 — `distance_matrix /= alpha` runs regardless of `alpha`; `HDBSCAN.__init__` sets `alpha=1.0` by default (hdbscan.py:655) and passes it through in `kwargs` (hdbscan.py:764-770)
scenario: "default `HDBSCAN(...)` with `algorithm='brute'` (or brute chosen by dispatch) on n=20 000 → n² = 4×10⁸ float divisions-by-1.0 are executed and n² cells are re-written, dirtying the whole distance matrix for a numerical no-op"
contract: Guard the scaling with `if alpha != 1.0: distance_matrix /= alpha`.
instances: single-instance

### F118 — `_compute_stability` allocates `births` twice
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252 is immediately overwritten by the same expression on line 254 with no interposed use
scenario: "any `HDBSCAN.fit` → the first `np.full` allocation is unused (an obvious merge artifact) and its buffer is garbage-collected without ever being read; wastes an O(largest_child) allocation per call"
contract: Delete the first `births = np.full(...)` on line 252.
instances: single-instance

### F119 — HDBSCAN default `n_jobs=4` contradicts its own docstring and every other sklearn estimator [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature has `n_jobs=4`, while sklearn/cluster/_hdbscan/hdbscan.py:486-491 documents `n_jobs : int, default=None` with the standard "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context" wording. Every other clusterer's default is `n_jobs=None` (sklearn/cluster/_dbscan.py:36,330; sklearn/cluster/_optics.py:274; sklearn/cluster/_spectral.py:633; sklearn/cluster/_mean_shift.py:41,132,427), and the private helpers in this same file (sklearn/cluster/_hdbscan/hdbscan.py:163, 276) also default to `None`.
scenario: "user runs `HDBSCAN().fit(X)` inside a joblib parallel context or a CPU-capped environment → the estimator silently pins 4 workers regardless of the surrounding backend / `os.sched_getaffinity`, violating the documented 'one job unless in `parallel_backend`' contract and breaking joblib nesting."
contract: change the constructor default to `n_jobs=None` to match both the docstring and the sklearn-wide convention.
instances: single-instance

### F120 — UnionFind's private cdef state is leaked through .pxd to enable direct cross-module field access
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:1-9 — a new `.pxd` was created solely to publish `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` (previously declared inline in `_hierarchical_fast.pyx` and therefore invisible to other Cython modules), and sklearn/cluster/_hdbscan/_linkage.pyx:265 reaches directly into that state with `U.size[current_node_cluster] + U.size[next_node_cluster]` after `cimport UnionFind` at sklearn/cluster/_hdbscan/_linkage.pyx:39.
scenario: "any future refactor of `UnionFind`'s storage (e.g. packing `parent`/`size` into a single struct, moving off memoryviews, or making `size` lazy) → silently breaks `_hdbscan/_linkage.pyx::make_single_linkage`, because the sibling submodule now depends on the concrete memoryview layout rather than a documented interface."
contract: keep the `.pxd` limited to the two methods (`union`, `fast_find`) and have `union` return the newly-merged cluster's size (or add a `cdef intp_t get_size(intp_t label) noexcept`); remove the `next_label`/`parent`/`size` declarations from the `.pxd` so they stay private to `_hierarchical_fast.pyx`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:4-6, sklearn/cluster/_hdbscan/_linkage.pyx:265, sklearn/cluster/_hierarchical_fast.pxd:3-8, sklearn/cluster/_hdbscan/_linkage.pyx:266]

### F121 — `_linkage.pyx` imports its own sibling via a top-level round-trip path
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39-41 — `from ...cluster._hierarchical_fast cimport UnionFind`, `from ...cluster._hdbscan._tree cimport HIERARCHY_t`, `from ...cluster._hdbscan._tree import HIERARCHY_dtype`. The last two go up three levels to `sklearn` and back down two into the file's own package. The peer file `sklearn/cluster/_hdbscan/hdbscan.py:58` uses the correct relative form `from ._tree import HIERARCHY_dtype`, and elsewhere in `sklearn.cluster` sibling Cython files use `from ._sibling cimport ...` (e.g. `sklearn/cluster/_k_means_lloyd.pyx:22-24`).
scenario: "rename or move of the `_hdbscan` subpackage → self-referencing `...cluster._hdbscan.*` imports break even though the sibling relationships are unchanged; also masks that `_tree` is a sibling of `_linkage`, not an external dependency"
contract: Rewrite the three sibling-facing lines in `_linkage.pyx` as `from ._tree cimport HIERARCHY_t`, `from ._tree import HIERARCHY_dtype`, and `from .._hierarchical_fast cimport UnionFind`, matching the sibling-import convention used elsewhere in `sklearn.cluster`.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:39, sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F122 — HIERARCHY-format tree remapping lives in the estimator file, not with the dtype it manipulates
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-393 — `remap_single_linkage_tree` reads/writes `tree[i]["left_node"]`, `tree[i]["right_node"]`, `tree[i]["cluster_size"]` and allocates `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)`, i.e. pure structured-array manipulation of the tree format whose dtype and Cython struct are owned by sklearn/cluster/_hdbscan/_tree.pyx:42-46 (`HIERARCHY_dtype`) and sklearn/cluster/_hdbscan/_tree.pxd:34-38 (`HIERARCHY_t`). The dtype is imported back into the estimator at sklearn/cluster/_hdbscan/hdbscan.py:58.
scenario: "HIERARCHY layout gains or renames a field in `_tree.pyx`/`_tree.pxd` → maintainer updates the tree module and its callers (`_linkage.pyx`, `_condense_tree`, `tree_to_labels`) but overlooks `remap_single_linkage_tree` because it is out-of-module; result: silent producer/consumer divergence on the tree format."
contract: move `remap_single_linkage_tree` into `sklearn/cluster/_hdbscan/_tree.pyx` (next to `HIERARCHY_dtype` and the other tree-format functions) and call it from `hdbscan.py`; keep `hdbscan.py` free of raw HIERARCHY structured-array indexing.
instances: single-instance

### F123 — Split test placement: one estimator's tests live in two directories
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1-64 — reachability tests live inside a nested `_hdbscan/tests/` package; sklearn/cluster/tests/test_hdbscan.py:1-533 — the estimator's main tests live in the sibling `cluster/tests/` directory. Every other clusterer (DBSCAN, OPTICS, Birch, KMeans, Bicluster, MeanShift, hierarchical, spectral, AffinityPropagation) has all of its tests under `sklearn/cluster/tests/` — this PR is the only clusterer that splits its tests across two locations.
scenario: "contributor adding a test for another `_hdbscan/*.pyx` file (e.g. `_linkage`, `_tree`) → has no canonical location to write it; over time the split leads to duplicate coverage or forgotten tests in the less-visible nested directory."
contract: relocate `sklearn/cluster/_hdbscan/tests/test_reachibility.py` to `sklearn/cluster/tests/test_reachability.py` and delete the `sklearn/cluster/_hdbscan/tests/` package so that all HDBSCAN tests live next to every other clusterer's tests.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/tests/__init__.py:1]

### F124 — HDBSCAN reaches into private `_dist_metrics` module for `DistanceMetric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:46 — `from ...metrics._dist_metrics import DistanceMetric`, while `sklearn/metrics/__init__.py:40` publicly re-exports `DistanceMetric` and every other `sklearn.cluster` Python file uses the public path (e.g. `sklearn/cluster/_agglomerative.py:21` — `from ..metrics import DistanceMetric`; the class docstrings themselves advertise `from sklearn.metrics import DistanceMetric`).
scenario: "downstream refactor renames/relocates `_dist_metrics` → HDBSCAN's import breaks even though the public `sklearn.metrics.DistanceMetric` surface is unchanged"
contract: Import `DistanceMetric` in `hdbscan.py` via the public entry point (`from ...metrics import DistanceMetric`), matching the convention used by every other Python file in `sklearn.cluster`.
instances: single-instance

### F125 — `remap_single_linkage_tree` lacks underscore prefix despite being an internal helper
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 — `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):`. It is called only from `hdbscan.py:834` inside `HDBSCAN.fit`, and is not re-exported anywhere (`sklearn/cluster/__init__.py:26` re-exports only `HDBSCAN` from this file). Every other private helper in the same file uses the underscore convention: `_brute_mst`, `_process_mst`, `_hdbscan_brute`, `_hdbscan_prims`, `_get_finite_row_indices`.
scenario: "future contributor treats `remap_single_linkage_tree` as public because of its unadorned name → downstream code depends on it → the module's private-vs-public boundary is quietly widened without review"
contract: Rename to `_remap_single_linkage_tree` and update the call site at `hdbscan.py:834` accordingly.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:351, sklearn/cluster/_hdbscan/hdbscan.py:834]

### F126 — Test module for `_reachability.pyx` is misnamed `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename contains "reachibility"; the target module `sklearn/cluster/_hdbscan/_reachability.pyx:1` and its function `mutual_reachability_graph` (imported at `sklearn/cluster/_hdbscan/tests/test_reachibility.py:15`) spell it "reachability".
scenario: "developer searches for `test_reachability` (correct spelling) via file finder / grep-by-basename → finds no matches and assumes the module is untested; discoverability of the test file relies on knowing about the typo"
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py`.
instances: single-instance

### F127 — `PyArray_SHAPE` numpy internal declared in `_tree.pxd` public surface instead of `_tree.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. No file outside `_tree.pyx` cimports this symbol from `_tree.pxd` (verified by grep — only uses are inside `_tree.pyx`), and the peer file that needs the same extern re-declares its own copy locally (`sklearn/cluster/_hdbscan/_linkage.pyx:44-45`), confirming the .pxd declaration is not being reused externally.
scenario: "numpy header layout changes or the sklearn build stops assuming numpy internals → the `.pxd` public surface breaks external cimports that never needed the symbol; also invites future cimports of a numpy-C-API primitive as if it were part of the module's contract"
contract: Move the `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` block from `_tree.pxd` into `_tree.pyx` so it remains a private compilation-unit helper; the `_linkage.pyx` local copy is fine to keep for its own use.
instances: single-instance

### F128 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented default `None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — the constructor signature line reads `n_jobs=4,` while the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None ... None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.` Peer estimators use `n_jobs=None` (sklearn/cluster/_dbscan.py:330, sklearn/cluster/_optics.py:274).
scenario: "A user reads the docstring and instantiates `HDBSCAN()` under `with joblib.parallel_backend('threading', n_jobs=1):` → the estimator ignores the context and silently spawns 4 workers via `pairwise_distances`/`NearestNeighbors`, hardcoding parallelism that contradicts the API contract and the pattern of every other clustering estimator."
contract: Set `n_jobs=None` in `HDBSCAN.__init__` so the wired default matches the docstring and the sklearn-wide `joblib.parallel_backend` contract.
instances: single-instance

### F129 — Test file name misspelled ("reachibility" → "reachability") diverges from module under test
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — the file tests the `mutual_reachability_graph` API exposed by `sklearn/cluster/_hdbscan/_reachability.pyx` (correctly spelled), but the test file itself is registered on disk as `test_reachibility.py` (missing the second `a`). The DIFF_MANIFEST also confirms `A sklearn/cluster/_hdbscan/tests/test_reachibility.py`.
scenario: "Maintainer runs `git grep test_reachability` or `pytest sklearn/**/test_reachability.py` to locate tests for the `_reachability` module → nothing is returned because the test file is misspelled `test_reachibility.py`; the tests are not discoverable through the canonical module name and the wiring between production module and its test file is broken by spelling."
contract: Rename `sklearn/cluster/_hdbscan/tests/test_reachibility.py` to `sklearn/cluster/_hdbscan/tests/test_reachability.py` so the test file name matches the module it tests (`_reachability.pyx`).
instances: single-instance

### F130 — `test_hdbscan_precomputed_non_brute` passes invalid algorithm names `prims_kdtree`/`prims_balltree`, so the ValueError comes from parameter validation instead of the guard the test claims to cover [out-of-theme]
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:282-290 — `@pytest.mark.parametrize("tree", ["kd", "ball"])` builds `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`, but the registered `StrOptions` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 accepts only `{"auto","brute","kdtree","balltree"}`. `_validate_params()` raises `InvalidParameterError` (a `ValueError`) before reaching the tree/metric compatibility guard at sklearn/cluster/_hdbscan/hdbscan.py:772-783 that the test's docstring says it is exercising.
scenario: "A future contributor changes the tree/metric compatibility check (or removes it) → this test still passes because it now trips on parameter validation, so the intended regression protection for the tree+precomputed guard is silently gone."
contract: Rename the parametrization values to the currently-registered `"kdtree"`/`"balltree"` algorithm names so the test actually exercises the metric-vs-tree guard it claims to cover.
instances: single-instance
