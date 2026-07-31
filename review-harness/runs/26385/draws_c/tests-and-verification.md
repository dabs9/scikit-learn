### F1 — Broadcast-add instead of set-union leaves an outlier in the "clean" test data
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))` — `missing_labels_idx` is `np.array([2, 5])` and `infinite_labels_idx` is `np.array([0])`, so `missing_labels_idx + infinite_labels_idx` is element-wise NumPy addition with broadcasting, yielding `array([2, 5])`, not the union `{0, 2, 5}`.
scenario: "Contributor introduces a regression that causes np.inf points to be labeled as clustered instead of -2 → test still passes because index 0 (an infinite outlier) is retained in `clean_idx`, so both models see the same outlier and produce the same -2 labels for it. The advertised contract (clustering equivalence with outliers *excluded*) is never actually exercised."
contract: Replace `set(missing_labels_idx + infinite_labels_idx)` with `set(missing_labels_idx).union(infinite_labels_idx)` (or concatenate then set) so `clean_idx` truly excludes every outlier index.
instances: single-instance

### F2 — `max_cluster_size` parameter has zero test coverage
severity: medium
evidence: `max_cluster_size` is a documented `HDBSCAN` parameter (sklearn/cluster/_hdbscan/hdbscan.py:443) with real behavior at sklearn/cluster/_hdbscan/_tree.pyx:733 (`cluster_sizes[node] > max_cluster_size`), but grep across `sklearn/cluster/tests/` and `sklearn/cluster/_hdbscan/tests/` shows no test constructs `HDBSCAN(max_cluster_size=...)` nor calls `_get_clusters` with a non-default `max_cluster_size`.
scenario: "Refactor of the EOM selection breaks the `max_cluster_size` filter (e.g., inverts the comparison, or accidentally keeps parents whose size exceeds the cap) → no test fails, silent regression."
contract: Add at least one test that fits HDBSCAN on a dataset containing an obviously oversized cluster and asserts that setting `max_cluster_size` splits/rejects that cluster while leaving smaller clusters intact.
instances: single-instance

### F3 — `cluster_selection_method="leaf"` branch is entirely untested
severity: medium
evidence: Every `cluster_selection_method=` in the test suite is `"eom"` (sklearn/cluster/tests/test_hdbscan.py:340, 354). The `"leaf"` branch at sklearn/cluster/_hdbscan/_tree.pyx:761 (`elif cluster_selection_method == 'leaf':`) is documented in `HDBSCAN` (sklearn/cluster/_hdbscan/hdbscan.py:492) but never exercised.
scenario: "Code path taken when a user chooses leaf-based selection could raise, return wrong dtypes, or silently return degenerate labels → CI stays green because no test ever hits that branch."
contract: Add a test that fits `HDBSCAN(cluster_selection_method="leaf")` on a small blob dataset and asserts a plausible label vector (n_clusters within a known range, correct dtype, no exceptions).
instances: single-instance

### F4 — `test_labelling_thresholding` only verifies noise *count*, not which points are noise
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:519-520 and 532-533 — `num_noise = condensed_tree["value"] < 1; assert sum(num_noise) == sum(labels == -1)` (and the analogous `MAX_LAMBDA` block). Both assertions compare only the *sum* of two boolean vectors — they pass as long as the total count matches, even if `_do_labelling` labels the wrong specific samples as noise.
scenario: "A regression in `_do_labelling` that swaps two sample assignments (e.g., marks sample 0 as noise and sample 1 as cluster instead of vice-versa) keeps the noise count identical → the test still passes."
contract: Assert the boolean identity `assert_array_equal(labels == -1, condensed_tree['value'] < threshold_per_sample)` (or an equivalent per-index equality) instead of comparing sums.
instances: [sklearn/cluster/tests/test_hdbscan.py:520, sklearn/cluster/tests/test_hdbscan.py:533]

### F5 — `test_hdbscan_algorithms` silently drops the `metric` parametrization for `brute`/`auto`
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:143 constructs `HDBSCAN(algorithm=algo).fit_predict(X)` with **no** `metric=metric` — always the default `"euclidean"`. Lines 148-149 then `return` early for `algo in ("brute", "auto")`. Thus the outer `@pytest.mark.parametrize("metric", _VALID_METRICS)` runs `len(_VALID_METRICS)` copies of the exact same euclidean test for those two algorithms.
scenario: "A metric-specific bug in the brute or auto path (e.g., a broken haversine or wminkowski flow) is invisible → the parametrize matrix appears to cover it, but the fitted estimator never actually uses that metric."
contract: For `brute` and `auto`, pass `metric=metric` (together with the existing `metric_params` block) to `HDBSCAN` before the fit, and assert the fit succeeds for supported metrics.
instances: single-instance

### F6 — `store_centers="centroid"` and `store_centers="medoid"` individual modes are untested
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:316 and 324 are the only `store_centers` uses, both `"both"`. The single-mode branches at sklearn/cluster/_hdbscan/hdbscan.py:897-903 (`make_centroids = self.store_centers in ("centroid", "both")` and the analogous medoid guard) are exercised only for the `"both"` value.
scenario: "A future refactor of `_weighted_cluster_center` conditionally sets `self.centroids_` / `self.medoids_` incorrectly for the single-mode values (e.g., always sets both, or leaves the wrong one unset) → no test catches it."
contract: Parametrize the existing centers test over `store_centers in ("centroid", "medoid", "both")` and assert the presence/absence of `centroids_` / `medoids_` matches the requested mode.
instances: single-instance

### F7 — `_weighted_cluster_center` excludes only labels `{-1, -2}`, not the `-3` missing label [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The `-3` "missing" outlier label defined at hdbscan.py:74 (`_OUTLIER_ENCODING["missing"]["label"]`) is *not* excluded from the count; and no test combines `store_centers` with data that contains `np.nan` rows.
scenario: "User fits `HDBSCAN(store_centers='centroid')` on data containing `np.nan` rows → `-3` is counted as an additional 'cluster', so `range(n_clusters)` iterates one index past the largest real cluster label; the resulting `mask = self.labels_ == idx` is all-False, `X[mask]` is empty, and `np.average` raises `ZeroDivisionError`/`Weights sum to zero`."
contract: Use the `_OUTLIER_ENCODING` labels (i.e. `set(self.labels_) - {-1} - {v['label'] for v in _OUTLIER_ENCODING.values()}`) so every outlier encoding is excluded when counting clusters, and add a test that combines `store_centers` with NaN-containing input.
instances: single-instance
