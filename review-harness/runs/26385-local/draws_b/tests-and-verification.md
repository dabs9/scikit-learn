Confirmed `n_jobs` default is `4` which contradicts documented default of `None`, and is never tested. This is out-of-theme (parameter default vs docstring), but I noticed it while doing my review.

Now I have enough findings. Let me compile the deliverable.

### F1 — Test asserts `ValueError` from param-validator instead of the claimed precomputed+tree rejection path
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `algorithm=f"prims_{tree}tree"` produces `"prims_kdtree"`/`"prims_balltree"`, which are NOT in the `algorithm` `StrOptions({"auto","brute","kdtree","balltree"})` set at hdbscan.py:629-638. `HDBSCAN.fit` calls `self._validate_params()` at hdbscan.py:697 before ever reaching the precomputed-vs-tree check at hdbscan.py:772-783; the raised `InvalidParameterError` (a `ValueError` subclass) satisfies the bare `pytest.raises(ValueError)`. The docstring says "Tests that HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm", but that code path is never executed.
scenario: "Someone removes/loosens the precomputed+kdtree/balltree rejection logic in `fit` → this test still passes because it only ever exercises parameter validation on a bogus algorithm string."
contract: Parametrize `tree` over the actual valid algorithm names (`"kdtree"`, `"balltree"`) and match on the concrete `is not a valid metric for a .*-based algorithm` message so the test verifies the intended rejection path.
instances: single-instance

### F2 — `test_dbscan_clustering_outlier_data` broadcasts numpy indices instead of concatenating, silently swallowing the infinite index
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` is `array([2, 5])` and `infinite_labels_idx` is `array([0])`; the `+` operator on ndarrays broadcasts (yielding `array([2, 5])`), not concatenates. `clean_idx` therefore still contains index `0` (the `np.inf` row) and is missing from the exclusion set. The subsequent `HDBSCAN().fit(X_outlier[clean_idx])` is fitting on data that still contains an infinite row, contrary to the "clean" naming and intent.
scenario: "Regression in outlier-index removal → test would still pass because index 0 is left in `clean_idx` and HDBSCAN handles it consistently between the two fits."
contract: Compute the exclusion set with `set(missing_labels_idx.tolist()) | set(infinite_labels_idx.tolist())` (or `np.concatenate([...])`) so the "clean" subset actually excludes every outlier index.
instances: single-instance

### F3 — `test_labelling_thresholding` asserts equal counts of noise, not equal noise positions
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:519-520 and 532-533 — the assertions are `assert sum(num_noise) == sum(labels == -1)`, where `num_noise` is the boolean mask `condensed_tree["value"] < 1` (or `< MAX_LAMBDA`). Two arrays with wholly different positions of `-1` but the same count satisfy this. The test's stated purpose is to verify the correct per-sample thresholding of lambda values, not just the cardinality of noisy points.
scenario: "The threshold branch in `_do_labelling` swaps positions but preserves the noise count → this test still passes."
contract: Assert positional equality of the noise mask, e.g. `assert_array_equal(labels == -1, condensed_tree['value'] < threshold)`.
instances: [sklearn/cluster/tests/test_hdbscan.py:519-520, sklearn/cluster/tests/test_hdbscan.py:532-533]

### F4 — `copy=False` behavior of `HDBSCAN` is untested; only `copy=True` non-mutation is asserted
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:79-92 — the only `assert_allclose(D, D_original)` guarding the copy contract is executed with `copy=True`. There is no test that fits with `copy=False` and verifies either (a) that the caller's array is mutated (proving `copy=False` is the "no-copy" path) or (b) that fit results are identical between `copy=True` and `copy=False`. The `copy` parameter is documented (hdbscan.py:521-526) as a first-class user option controlling in-place modification of precomputed inputs and the brute-force reachability graph, but the default path is completely uncovered.
scenario: "A regression silently makes `copy=False` allocate copies (or `copy=True` mutate) → tests do not detect it."
contract: Add explicit fit-time assertions for `copy=False`: (i) the input distance matrix is mutated in place, and (ii) the resulting labels equal those from `copy=True`.
instances: single-instance

### F5 — `cluster_selection_method="leaf"` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — a grep for `cluster_selection_method` returns only lines 340 and 354, both passing `"eom"`. The `_get_clusters` branch at sklearn/cluster/_hdbscan/_tree.pyx:761-782 implementing the `"leaf"` method (including the empty-leaves fallback, `get_cluster_tree_leaves`, `recurse_leaf_dfs`, and the leaf + epsilon combination via `epsilon_search`) is added new code with no test coverage.
scenario: "Regression in the leaf-selection branch (e.g., breaking `get_cluster_tree_leaves` or the empty-leaves fallback) → no test fails."
contract: Add at least one test that runs `HDBSCAN(cluster_selection_method="leaf")` on a dataset with a known hierarchical structure and asserts the returned leaf clusters against a reference.
instances: single-instance

### F6 — `max_cluster_size` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — no reference to `max_cluster_size` anywhere. The parameter is declared at sklearn/cluster/_hdbscan/hdbscan.py:622-625 and has real semantics at sklearn/cluster/_hdbscan/_tree.pyx:716-717 and 733 (`cluster_sizes[node] > max_cluster_size` forces `is_cluster[node] = False`). The comment at _tree.pyx:717 shows the default sentinel is `n_samples + 1` — regressing this sentinel would silently break EOM cluster selection with no test to catch it.
scenario: "The `max_cluster_size` short-circuit is deleted or its comparison inverted → no test fails."
contract: Add a test that fits HDBSCAN with `cluster_selection_method="eom"` and a small `max_cluster_size`, and asserts that no returned cluster exceeds that size.
instances: single-instance

### F7 — `leaf_size` is untested
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py — no reference to `leaf_size`. It is declared at sklearn/cluster/_hdbscan/hdbscan.py:639 and forwarded to `NearestNeighbors` via `_hdbscan_prims` at hdbscan.py:799, 803, 813, 818. There is no test verifying that `leaf_size` is threaded through (e.g., that varying `leaf_size` still produces identical labels).
scenario: "A regression drops `leaf_size` from the `kwargs` dict → no test fails; users silently lose the ability to tune tree leaf size."
contract: Add a test that fits with two different `leaf_size` values and asserts labels are identical.
instances: single-instance

### F8 — `store_centers="centroid"` and `store_centers="medoid"` branches are untested (only `"both"` is exercised)
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:309-327 — the only test of centers passes `store_centers="both"`. The independent branches (`"centroid"` alone should set `centroids_` but not `medoids_`; `"medoid"` alone the reverse) at sklearn/cluster/_hdbscan/hdbscan.py:897-903 are never verified. In particular, no test asserts that attribute `medoids_` is missing when `store_centers="centroid"` (and vice-versa), nor that `store_centers=None` sets no attributes.
scenario: "A regression makes `_weighted_cluster_center` always compute both regardless of `store_centers` → not caught."
contract: Add parametrized tests over `store_centers in {None, "centroid", "medoid", "both"}` asserting exactly which of `centroids_` and `medoids_` are set.
instances: single-instance

### F9 — Assertion `assert counts[unique_labels == -1] > 30` compares an array to a scalar
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:347 — `counts[unique_labels == -1]` is a length-1 ndarray, not a Python scalar; comparing a length-1 array `> 30` returns a length-1 boolean array. `assert` on a length-1 array works today by scalar-truth-test, but the pattern is fragile (a numpy deprecation would break it) and inconsistent with the neighboring `counts[unique_labels == -1] == 2` at test_hdbscan.py:360 which is also a length-1 array vs scalar.
scenario: "Numpy tightens the scalar-conversion rule → these asserts start emitting DeprecationWarning or raising."
contract: Extract the scalar with `.item()` (or index `[0]`) before comparison, e.g. `assert counts[unique_labels == -1].item() > 30`.
instances: [sklearn/cluster/tests/test_hdbscan.py:347, sklearn/cluster/tests/test_hdbscan.py:360]

### F10 — `n_jobs` default is `4` but documented as `None`; not covered by any test [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`. The docstring at hdbscan.py:486-490 states: "n_jobs : int, default=None. Number of jobs to run in parallel to calculate distances. `None` means 1 unless in a `joblib.parallel_backend` context. `-1` means using all processors." No test in test_hdbscan.py exercises or verifies the effective `n_jobs` default.
scenario: "A user reads the docstring, expects joblib context to be respected → HDBSCAN silently forces 4 jobs regardless of the surrounding `parallel_backend`, breaking user expectations and joblib coordination."
contract: Set `n_jobs=None` as the default (matching the documented behavior) and add a test asserting the default is `None`.
instances: single-instance
