### F1 — Precomputed-non-brute test uses invalid algorithm names, exercises the wrong code path
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` combined with `pytest.raises(ValueError)`. The `algorithm` constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-638 accepts only `{"auto","brute","kdtree","balltree"}`, so `"prims_kdtree"`/`"prims_balltree"` are rejected by `_validate_params()` before any precomputed-vs-tree logic runs. `InvalidParameterError` is a `ValueError` subclass (sklearn/utils/_param_validation.py:20), so the test passes for the wrong reason.
scenario: "Someone removes the `metric not in KDTree.valid_metrics()` guard at hdbscan.py:772-782 → test still passes because the invalid algorithm name still triggers `InvalidParameterError` → the actual precomputed+tree rejection is silently broken."
contract: Rewrite the test to use the valid names `"kdtree"`/`"balltree"` so the invalid `precomputed`-with-tree combination is actually reached, and match the specific error message about the metric not being valid for the KDTree/BallTree-based algorithm.
instances: single-instance

### F2 — `test_hdbscan_centers` uses `rtol=1`, permitting near-100% relative deviation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:314-320 — `centers = [(0.0, 0.0), (3.0, 3.0)]`; `assert_allclose(center, centroid, rtol=1, atol=0.05)` and same for `medoid`. `assert_allclose` uses `|actual-desired| <= atol + rtol*|desired|`, so for the `(3.0, 3.0)` blob the bound is `0.05 + 1*3.0 = 3.05` — the centroid may fall anywhere in roughly `[-0.05, 6.05]` and still pass.
scenario: "A regression that dramatically shifts centroids of any non-zero-centered cluster (e.g. weighting bug in `_weighted_cluster_center` at hdbscan.py:912) → the test still passes because the tolerance swallows the drift → silent centroid/medoid regression."
contract: Use an absolute tolerance calibrated to `cluster_std=0.5` (e.g. `atol=0.1, rtol=0`) so the assertion actually constrains the centroid/medoid location for non-zero centers.
instances: single-instance

### F3 — `max_cluster_size` parameter is completely untested
severity: medium
evidence: `max_cluster_size` is a documented public parameter (hdbscan.py:443-446), listed in `_parameter_constraints` (hdbscan.py:622-625), threaded through `tree_to_labels` (hdbscan.py:828) and gates the `cluster_sizes[node] > max_cluster_size` branch in `_get_clusters` at _tree.pyx:733. `grep max_cluster_size sklearn/cluster/tests/test_hdbscan.py` returns nothing.
scenario: "Someone breaks the `max_cluster_size` split logic in `_get_clusters` at _tree.pyx:733 (e.g. flipping the comparison) → every test in the suite passes → users of `max_cluster_size` receive silently wrong clusterings."
contract: Add a test that constructs data with a large parent cluster, fits with a `max_cluster_size` chosen to force the EOM re-selection branch, and asserts that no returned cluster exceeds `max_cluster_size` and that clustering differs from the `max_cluster_size=None` result.
instances: single-instance

### F4 — `test_dbscan_clustering` self-declared weak; only asserts cluster count
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:178-188 — docstring: `"This test is more of a sanity check than a rigorous evaluation. TODO: Improve and strengthen this test if at all possible."`; body only asserts `n_clusters == n_clusters_true`. No comparison against a reference DBSCAN output or expected labels; any implementation that returns three groups (even wrong ones) passes.
scenario: "A bug in `dbscan_clustering` at hdbscan.py:923-970 that permutes labels or misassigns points but still yields 3 groups → test still passes → `dbscan_clustering` returns silently wrong labels."
contract: Compare the returned labels against `cluster.DBSCAN(eps=0.3, min_samples=...).fit_predict(X)` (up to label permutation via `fowlkes_mallows_score >= 0.9`), so the test asserts the actual clustering, not just its cardinality.
instances: single-instance

### F5 — `test_hdbscan_algorithms` success branch asserts nothing about results
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:174-175 — `else: hdb.fit(X)`. On the metric/algo combinations that should succeed, the test merely calls `fit` with no assertion on `labels_`, `probabilities_`, or cluster count. Only the initial `HDBSCAN(algorithm=algo).fit_predict(X)` at line 143 asserts anything, and that uses the default euclidean metric — none of the parametrized `metric` values are actually checked to produce correct output.
scenario: "A regression that makes `_hdbscan_prims` silently return degenerate labels for metric X → `fit(X)` still completes without raising → the parametrized coverage across `_VALID_METRICS × ALGORITHMS` fails to detect it."
contract: In the success branch, assert `len(set(hdb.labels_) - OUTLIER_SET) == n_clusters_true` (or an equivalent quality check like `fowlkes_mallows_score >= 0.9`) so each parametrized combination validates the produced clustering, not merely that `fit` did not raise.
instances: single-instance
