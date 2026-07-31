### F1 — `test_dbscan_clustering_outlier_data` broadcasts arrays instead of concatenating, so "clean" data still contains the infinite outlier
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` (`np.array([2, 5])`) and `infinite_labels_idx` (`np.array([0])`) are NumPy arrays; `+` performs element-wise broadcast → `[2, 5]`, not `[2, 5, 0]`. Index 0 (the np.inf point) is NOT removed from `clean_idx`.
scenario: "future refactor changes how infinite outliers are labeled in `dbscan_clustering` on non-outlier inputs → test still passes because the 'clean' subset still contains the outlier and gets the same infinite label from the outlier path in both models, masking the regression"
contract: replace with an explicit concatenation such as `np.concatenate([missing_labels_idx, infinite_labels_idx]).tolist()` (or `list(...) + list(...)`) so all three outlier indices (0, 2, 5) are truly excluded from `clean_idx`.
instances: single-instance

### F2 — `test_labelling_thresholding` asserts only the noise-count, not which samples are noise
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:519-520,532-533 — `num_noise = condensed_tree["value"] < 1; assert sum(num_noise) == sum(labels == -1)` (and again with `< MAX_LAMBDA`). The assertion compares two totals; it never checks *which* sample indices `labels` marks as -1.
scenario: "a regression that mislabels a valid-cluster point as noise while promoting a noise point to the cluster (identity-swap) → totals still match, assertion passes"
contract: assert full-array equality against the expected label vector, e.g. `assert_array_equal(labels, np.where(condensed_tree['value'] < 1, -1, 0))` (adjusted for indexing), so identity — not just count — is verified.
instances: [sklearn/cluster/tests/test_hdbscan.py:519, sklearn/cluster/tests/test_hdbscan.py:520, sklearn/cluster/tests/test_hdbscan.py:532, sklearn/cluster/tests/test_hdbscan.py:533]

### F3 — `test_hdbscan_algorithms` never asserts clustering quality for the parametrized (algo, metric) combinations
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:143-175 — the only labels assertion (`n_clusters == n_clusters_true`) is made on a model built as `HDBSCAN(algorithm=algo)` (no metric, no metric_params). Under the valid-metric branch at 174-175 `hdb.fit(X)` is called with the actual (algo, metric) combination, but no assertion follows. Additionally, when `algo in ("brute", "auto")` the function returns at 149, so the `metric` parametrization is a no-op — the same identical assertion is repeated across every metric.
scenario: "a regression breaks kdtree+manhattan (returns garbage labels but doesn't raise) → parametrized run is green because `hdb.fit(X)` output is never checked"
contract: for every non-erroring (algo, metric) combination, compute `labels` and assert `len(set(labels) - OUTLIER_SET) == n_clusters_true` (and `fowlkes_mallows_score >= 0.98` for combos where the clustering is expected to be good) so the parametrization actually validates behavior.
instances: single-instance

### F4 — `cluster_selection_method="leaf"` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — grep for `cluster_selection_method` returns only `"eom"` (lines 340, 354); the `leaf` branch in `sklearn/cluster/_hdbscan/_tree.pyx:761-782` (including the empty-`leaves` fallback at :763-766) has no test coverage.
scenario: "Any regression in the `leaf` branch of `_get_clusters` (e.g. `is_cluster[condensed_tree['parent'].min()] = True` fallback, or `epsilon_search` interaction) ships silently → users passing `cluster_selection_method='leaf'` get wrong labels with no test failure."
contract: Add at least one parametrized test that runs `HDBSCAN(cluster_selection_method="leaf")` on `X` and verifies both cluster count and label sanity (including with `cluster_selection_epsilon > 0`).
instances: single-instance

### F5 — `max_cluster_size` parameter is completely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py grep for `max_cluster_size` — zero occurrences. The parameter is declared in `_parameter_constraints` (`sklearn/cluster/_hdbscan/hdbscan.py:622-625`) and drives an override branch inside `_get_clusters` (`sklearn/cluster/_hdbscan/_tree.pyx:716-717,733`) where a cluster is split when `cluster_sizes[node] > max_cluster_size`.
scenario: "a regression in the `max_cluster_size` split path silently returns clusters larger than the requested limit → no test detects it"
contract: add a test that fits HDBSCAN on data producing one large natural cluster, then re-fits with `max_cluster_size=<smaller>` and asserts that no returned non-noise cluster exceeds `max_cluster_size` in bincount size.
instances: single-instance

### F6 — test_hdbscan_precomputed_non_brute exercises non-existent algorithm names
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` uses `"prims_kdtree"` / `"prims_balltree"`, which are NOT in `_parameter_constraints["algorithm"] = StrOptions({"auto", "brute", "kdtree", "balltree"})` in `sklearn/cluster/_hdbscan/hdbscan.py:629-638`.
scenario: "user calls `HDBSCAN(metric='precomputed', algorithm='kdtree').fit(X)` → the test asserts nothing about that combination; the actual test passes only because `_validate_params()` rejects the invalid algorithm string, so the precomputed-vs-tree logic (`hdbscan.py:772-783`) is never exercised."
contract: Rename to the valid values `"kdtree"` / `"balltree"` so the test actually verifies the precomputed-with-tree rejection its docstring claims.
instances: single-instance

### F7 — test_hdbscan_centers uses rtol=1, making the centroid/medoid check meaningless for the (3,3) center
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:318-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)`; for `center=(3,3)` the tolerance evaluates to `atol + rtol*|3| = 0.05 + 3 = 3.05` per component, i.e. the centroid could be anywhere in `[-0.05, 6.05]` and still pass.
scenario: "A regression that shifts the (3,3) centroid to (0.1, 0.1) or (5.9, 5.9) → the assertion still passes; the centroid computation could be arbitrarily broken for non-origin clusters and this test would not detect it."
contract: Use `rtol=0.05` (or `atol` only), matching the intent of "centroid accurate to ~5%".
instances: [sklearn/cluster/tests/test_hdbscan.py:319, sklearn/cluster/tests/test_hdbscan.py:320]

### F8 — `_weighted_cluster_center` excludes only labels `{-1, -2}`, not the `-3` missing label [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The `-3` "missing" outlier label defined at hdbscan.py:74 (`_OUTLIER_ENCODING["missing"]["label"]`) is *not* excluded from the count; and no test combines `store_centers` with data that contains `np.nan` rows.
scenario: "User fits `HDBSCAN(store_centers='centroid')` on data containing `np.nan` rows → `-3` is counted as an additional 'cluster', so `range(n_clusters)` iterates one index past the largest real cluster label; the resulting `mask = self.labels_ == idx` is all-False, `X[mask]` is empty, and `np.average` raises `ZeroDivisionError`/`Weights sum to zero`."
contract: Use the `_OUTLIER_ENCODING` labels (i.e. `set(self.labels_) - {-1} - {v['label'] for v in _OUTLIER_ENCODING.values()}`) so every outlier encoding is excluded when counting clusters, and add a test that combines `store_centers` with NaN-containing input.
instances: single-instance

### F9 — `test_hdbscan_min_cluster_size` silently no-ops when every point is noise
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:259-263 — `for min_cluster_size in range(2, len(X), 1):` iterates up to 199. For large `min_cluster_size` values, all labels equal `-1`, so `true_labels` is empty and the `if len(true_labels) != 0:` guard skips the assertion; those iterations verify nothing.
scenario: "a regression that returns noise for ALL points at moderately-large min_cluster_size → test still passes because every non-trivial iteration hits the skipped branch"
contract: replace the silent skip with an explicit assertion for the noise-only case, e.g. `else: assert set(labels) == {-1}` (or bound the loop range so every iteration produces at least one cluster), so no iteration passes without an assertion.
instances: single-instance

### F10 — test_hdbscan_sparse does not verify np.nan rows receive the missing-outlier label
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:297-301 — after `sparse_X_nan[0, 0] = np.nan` and `labels = HDBSCAN().fit(sparse_X_nan).labels_`, the test only checks `n_clusters = len(set(labels) - OUTLIER_SET) == 3`; it never asserts `labels[0] == _OUTLIER_ENCODING["missing"]["label"]`.
scenario: "A regression in the `_get_finite_row_indices` / `_OUTLIER_ENCODING['missing']` remapping for sparse inputs (`hdbscan.py:714-744`, `830-852`) that fails to label the nan row as `-3` → test still passes because `-3` ∈ OUTLIER_SET is subtracted before counting."
contract: Add `assert labels[0] == _OUTLIER_ENCODING["missing"]["label"]` (and that the remaining rows are not `-3`).
instances: single-instance

### F11 — `store_centers` "centroid"-only and "medoid"-only variants are untested
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:316,324 — the only `store_centers` invocations pass `"both"`. The `_weighted_cluster_center` helper (`sklearn/cluster/_hdbscan/hdbscan.py:894-921`) has separate `make_centroids`/`make_medoids` toggles and does *not* create the unused attribute; if only `"centroid"` is requested, `self.medoids_` must NOT exist (and vice versa). No test guards this.
scenario: "a regression sets both `self.centroids_` and `self.medoids_` regardless of `store_centers` value → tests pass because only `'both'` is exercised, but user code that inspects `hasattr(hdb, 'medoids_')` silently breaks"
contract: parametrize `test_hdbscan_centers` (or add a companion test) over `["centroid", "medoid", "both"]`, asserting both that the requested attribute is present and correct, and that the un-requested attribute is absent.
instances: single-instance

### F12 — Sparse mutual-reachability `max_distance > 0` fallback and `INFINITY` core-distance branches are untested
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:199-200 (`else: core_distances[i] = INFINITY` when `further_neighbor_idx >= row_data.size`) and sklearn/cluster/_hdbscan/_reachability.pyx:211-212 (`elif max_distance > 0: data[i] = max_distance`). `sklearn/cluster/_hdbscan/tests/test_reachibility.py` never constructs a sparse matrix with an under-populated row and never passes a non-zero `max_distance`.
scenario: "a regression breaks the max_distance truncation of non-finite mutual reachabilities (e.g., writes `max_distance` even for finite entries) → both dedicated test files stay green because no case triggers the branch"
contract: add tests in `test_reachibility.py` that (a) call `mutual_reachability_graph` with a sparse row shorter than `min_samples` and assert that the resulting core-distance is `inf`, and (b) call it with a sparse matrix containing enough missing distances to yield an infinite mutual reachability plus `max_distance > 0`, and assert that infinities are replaced by `max_distance`.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:199, sklearn/cluster/_hdbscan/_reachability.pyx:199-200, sklearn/cluster/_hdbscan/_reachability.pyx:211, sklearn/cluster/_hdbscan/_reachability.pyx:211-212]

### F13 — test_hdbscan_centers parametrizes `algorithm` but the first assertion ignores it
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:308-320 — `@pytest.mark.parametrize("algorithm", ALGORITHMS)` runs the test 4×, but the first fit at :316 is `HDBSCAN(store_centers="both").fit(H)` with no `algorithm=` — only the second fit at :322-325 uses the parametrized `algorithm`. The centroid/medoid accuracy check therefore runs 4× against the same "auto" algorithm.
scenario: "The centroid/medoid computation path for `algorithm='brute'` / `'kdtree'` / `'balltree'` is never covered by the accuracy assertion → a regression that affects only non-auto algorithms passes the tolerance check because it's always executed under `algorithm='auto'`."
contract: Pass `algorithm=algorithm` to the first HDBSCAN construction at :316 so the centroid/medoid accuracy is verified once per algorithm.
instances: single-instance
