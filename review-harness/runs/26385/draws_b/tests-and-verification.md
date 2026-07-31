### F1 — test_hdbscan_precomputed_non_brute exercises non-existent algorithm names
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` uses `"prims_kdtree"` / `"prims_balltree"`, which are NOT in `_parameter_constraints["algorithm"] = StrOptions({"auto", "brute", "kdtree", "balltree"})` in `sklearn/cluster/_hdbscan/hdbscan.py:629-638`.
scenario: "user calls `HDBSCAN(metric='precomputed', algorithm='kdtree').fit(X)` → the test asserts nothing about that combination; the actual test passes only because `_validate_params()` rejects the invalid algorithm string, so the precomputed-vs-tree logic (`hdbscan.py:772-783`) is never exercised."
contract: Rename to the valid values `"kdtree"` / `"balltree"` so the test actually verifies the precomputed-with-tree rejection its docstring claims.
instances: single-instance

### F2 — test_dbscan_clustering_outlier_data fails to exclude the infinite index from clean_idx
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:206-215 — `missing_labels_idx = np.flatnonzero(...)` and `infinite_labels_idx = np.flatnonzero(...)` return numpy arrays; `missing_labels_idx + infinite_labels_idx` performs numpy broadcasting (shape (2,) + shape (1,) → shape (2,) = `[2,5]`), NOT list concatenation, so `set(...)` yields `{2, 5}` and `clean_idx` still contains index 0 (the infinite-outlier row).
scenario: "The infinite row `[np.inf, 1]` remains in `X_outlier[clean_idx]` → the 'clean' model is not actually clean; the test still passes but stops verifying that outlier removal + fit is equivalent to labels[clean_idx]."
contract: Use `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or convert to lists first) so the set difference removes both missing AND infinite outlier indices.
instances: single-instance

### F3 — test_hdbscan_centers uses rtol=1, making the centroid/medoid check meaningless for the (3,3) center
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:318-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)`; for `center=(3,3)` the tolerance evaluates to `atol + rtol*|3| = 0.05 + 3 = 3.05` per component, i.e. the centroid could be anywhere in `[-0.05, 6.05]` and still pass.
scenario: "A regression that shifts the (3,3) centroid to (0.1, 0.1) or (5.9, 5.9) → the assertion still passes; the centroid computation could be arbitrarily broken for non-origin clusters and this test would not detect it."
contract: Use `rtol=0.05` (or `atol` only), matching the intent of "centroid accurate to ~5%".
instances: [sklearn/cluster/tests/test_hdbscan.py:319, sklearn/cluster/tests/test_hdbscan.py:320]

### F4 — `cluster_selection_method="leaf"` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — grep for `cluster_selection_method` returns only `"eom"` (lines 340, 354); the `leaf` branch in `sklearn/cluster/_hdbscan/_tree.pyx:761-782` (including the empty-`leaves` fallback at :763-766) has no test coverage.
scenario: "Any regression in the `leaf` branch of `_get_clusters` (e.g. `is_cluster[condensed_tree['parent'].min()] = True` fallback, or `epsilon_search` interaction) ships silently → users passing `cluster_selection_method='leaf'` get wrong labels with no test failure."
contract: Add at least one parametrized test that runs `HDBSCAN(cluster_selection_method="leaf")` on `X` and verifies both cluster count and label sanity (including with `cluster_selection_epsilon > 0`).
instances: single-instance

### F5 — `max_cluster_size` parameter has no test coverage
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — grep for `max_cluster_size` returns zero matches; the parameter is declared in `sklearn/cluster/_hdbscan/hdbscan.py:622-625` and consumed at `_tree.pyx:733` (`cluster_sizes[node] > max_cluster_size`), controlling whether an EOM cluster is subdivided.
scenario: "A regression in the `cluster_sizes[node] > max_cluster_size` short-circuit (e.g. off-by-one on the comparison, wrong sentinel when `None`) ships undetected → users passing `max_cluster_size=N` get subdivision behavior different from the documented contract."
contract: Add a test that fits `HDBSCAN(max_cluster_size=k)` on data with a known dominant cluster and verifies no returned cluster exceeds size `k`.
instances: single-instance

### F6 — `mutual_reachability_graph(max_distance=...)` branch is untested
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:209-212 — `elif max_distance > 0: data[i] = max_distance` is triggered only for sparse matrices with non-finite mutual-reachability entries; sklearn/cluster/_hdbscan/tests/test_reachibility.py has four tests, none of which pass a `max_distance` > 0, and none construct a sparse matrix where `further_neighbor_idx >= row_data.size` (the branch at _reachability.pyx:199-200 that produces `INFINITY`).
scenario: "A regression in the max_distance substitution (e.g. flipping the sign, applying it to finite values, or the INFINITY sentinel path) ships silently → the documented behavior of `max_distance` truncating infinite distances in sparse graphs is unverified."
contract: Add a test that builds a sparse CSR with rows having fewer than `min_samples` non-zeros, calls `mutual_reachability_graph(X, min_samples=k, max_distance=M)`, and asserts the resulting entries equal `M` where the true value would be infinite.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:199-200, sklearn/cluster/_hdbscan/_reachability.pyx:211-212]

### F7 — test_labelling_thresholding asserts only noise COUNT, not which samples are noise
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:519-533 — both assertions are `assert sum(num_noise) == sum(labels == -1)`, comparing scalar counts derived from `condensed_tree["value"] < threshold` against `labels == -1`.
scenario: "A regression that mislabels which specific samples are noise (e.g. swaps noise assignment between siblings) but preserves the total count → assertion holds; the `_do_labelling` per-sample thresholding logic is only verified in aggregate."
contract: Assert element-wise `np.array_equal(condensed_tree["value"] < threshold, labels == -1)` (or the equivalent index-set equality) so the test verifies WHICH samples are noise, not merely how many.
instances: [sklearn/cluster/tests/test_hdbscan.py:520, sklearn/cluster/tests/test_hdbscan.py:533]

### F8 — test_hdbscan_centers parametrizes `algorithm` but the first assertion ignores it
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:308-320 — `@pytest.mark.parametrize("algorithm", ALGORITHMS)` runs the test 4×, but the first fit at :316 is `HDBSCAN(store_centers="both").fit(H)` with no `algorithm=` — only the second fit at :322-325 uses the parametrized `algorithm`. The centroid/medoid accuracy check therefore runs 4× against the same "auto" algorithm.
scenario: "The centroid/medoid computation path for `algorithm='brute'` / `'kdtree'` / `'balltree'` is never covered by the accuracy assertion → a regression that affects only non-auto algorithms passes the tolerance check because it's always executed under `algorithm='auto'`."
contract: Pass `algorithm=algorithm` to the first HDBSCAN construction at :316 so the centroid/medoid accuracy is verified once per algorithm.
instances: single-instance

### F9 — test_hdbscan_sparse does not verify np.nan rows receive the missing-outlier label
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:297-301 — after `sparse_X_nan[0, 0] = np.nan` and `labels = HDBSCAN().fit(sparse_X_nan).labels_`, the test only checks `n_clusters = len(set(labels) - OUTLIER_SET) == 3`; it never asserts `labels[0] == _OUTLIER_ENCODING["missing"]["label"]`.
scenario: "A regression in the `_get_finite_row_indices` / `_OUTLIER_ENCODING['missing']` remapping for sparse inputs (`hdbscan.py:714-744`, `830-852`) that fails to label the nan row as `-3` → test still passes because `-3` ∈ OUTLIER_SET is subtracted before counting."
contract: Add `assert labels[0] == _OUTLIER_ENCODING["missing"]["label"]` (and that the remaining rows are not `-3`).
instances: single-instance

### F10 — test_hdbscan_min_cluster_size silently skips assertions when all points are labeled noise
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:259-263 — `if len(true_labels) != 0: assert np.min(np.bincount(true_labels)) >= min_cluster_size`; when every label is `-1` (which will happen at large `min_cluster_size`), the body is skipped and the loop iteration passes with no assertion.
scenario: "For any `min_cluster_size` value large enough that the algorithm returns all-noise, the test silently passes → a regression that produces all-noise for ordinary `min_cluster_size` (e.g. off-by-one in cluster acceptance) would go undetected."
contract: Remove the `if len(true_labels) != 0` guard, or add an assertion that at least some values of `min_cluster_size` in the swept range must produce a non-empty cluster set.
instances: single-instance
