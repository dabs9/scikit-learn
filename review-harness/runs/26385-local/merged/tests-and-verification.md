Note: The system reminders about malware analysis do not apply here — these files are review-harness findings documents (markdown), not code, and my task is a pure UNION merge of findings, not code improvement. The TodoWrite reminder is not applicable to this single-purpose merge task.

Now merging. Deduplication mapping (same defect = same root cause at same site):
- Precomputed+tree test: A-F1, B-F1, C-F3 → same. A-F1 is most specific (cites `_hdbscan_prims` misroute + line ranges). Keep A-F1.
- Broadcast indices at line 212: A-F4, B-F2, C-F2 → same. B-F2 and C-F2 both very specific; C-F2 explicitly confirms with runtime check. Keep C-F2 (most specific).
- `test_labelling_thresholding` noise counts: B-F3, C-F1 → same. B-F3 cites both line ranges. Keep B-F3.
- `test_hdbscan_centers` rtol=1: A-F2, C-F4 → same. Both specific; A-F2 has instances list. Keep A-F2.
- `max_cluster_size` untested: B-F6, C-F5, plus part of A-F6. B-F6 most specific (cites _tree.pyx:716-717, 733, sentinel comment). Keep B-F6.
- `cluster_selection_method="leaf"` untested: B-F5, C-F6, part of A-F6. B-F5 most specific. Keep B-F5.
- `min_cluster_size` filter excludes only `-1`: A-F3, C-F10 → same. Both specific; A-F3 slightly more direct. Keep A-F3.
- `store_centers` single-mode untested: B-F8, C-F7 → same. B-F8 more specific (includes `None`, cites lines 897-903). Keep B-F8.
- `n_jobs` default 4 vs None [out-of-theme]: A-F10, B-F10 → same. B-F10 more specific (medium severity, cites full docstring). Keep B-F10.

Unique findings (kept as-is):
- A-F5 (max_distance untested), A-F7 (algorithms first assertion), A-F8 (algorithms invalid-metric bare raises), A-F9 (remap_single_linkage_tree untested)
- B-F4 (copy=False untested), B-F7 (leaf_size untested), B-F9 (array-vs-scalar assertion)
- C-F8 (filename typo), C-F9 (dbscan_clustering min_cluster_size), C-F11 (n_jobs correctness untested), C-F12 (copy=True sparse branch), C-F13 (no_clusters noise label)

Note A-F6 is a compound of leaf + max_cluster_size — both aspects survive via B-F5 and B-F6; no separate defect remains.

Ordering: high, then medium, then low.

### F1 — `test_hdbscan_precomputed_non_brute` uses invalid algorithm strings, passes for the wrong reason
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — the test constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` where the algorithm string is `"prims_kdtree"`/`"prims_balltree"`, but the estimator's `_parameter_constraints` (hdbscan.py:629-638) only accepts `{"auto", "brute", "kdtree", "balltree"}`. The `ValueError` raised is `InvalidParameterError` (from `_validate_params`, which extends `ValueError`), not the intended "precomputed data incompatible with tree" error. In fact, hdbscan.py:772-818 contains no code path that rejects `metric="precomputed"` when `algorithm="kdtree"`/`"balltree"`; the branch at line 785 silently dispatches to `_hdbscan_prims`.
scenario: "user passes metric='precomputed' with algorithm='kdtree'/'balltree' → test claims coverage but code path is unexercised; real behavior is untested (silent misroute to `_hdbscan_prims`)"
contract: The test must use the actually-supported strings (`algorithm="kdtree"`/`"balltree"`) and match against the specific error message the codebase intends to raise; if no such rejection exists in the estimator, the missing validation must be added and then covered by a matched-message test.
instances: single-instance

### F2 — `test_hdbscan_centers` uses `rtol=1`, effectively disabling the centroid/medoid accuracy check for non-zero centers
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:319-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)` and same for `medoid`. With `rtol=1`, the tolerance for `center=(3.0, 3.0)` becomes `atol + rtol*|3| = 3.05`, i.e. any value in `[-0.05, 6.05]` passes. Only the `(0, 0)` center is effectively constrained (to `atol=0.05`). The stated intent in the docstring is that centers "are accurate to the data".
scenario: "centroid computation regresses for any non-zero-centered cluster → test still passes because rtol=1 permits ~100% relative error"
contract: Drop `rtol=1` and use only a tight `atol` (or `rtol` ≪ 1 like `rtol=0.05`) so the check reflects the "accurate to the data" contract.
instances: [sklearn/cluster/tests/test_hdbscan.py:319, sklearn/cluster/tests/test_hdbscan.py:320]

### F3 — `test_hdbscan_min_cluster_size` filter excludes only `-1`, ignoring `-2`/`-3` outlier labels
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:261-263 — `true_labels = [label for label in labels if label != -1]` then `np.bincount(true_labels)`. `np.bincount` requires non-negative integers and will raise `ValueError` if `-2`/`-3` (defined in `_OUTLIER_ENCODING`) are present. The correct filter would use `OUTLIER_SET` (defined at line 37) which the file already imports/constructs for exactly this purpose.
scenario: "input triggers infinite/missing outlier labels (-2/-3) → test crashes with numpy bincount error instead of validating minimum cluster size; today it passes only because the `X` fixture is finite and no outlier labels appear"
contract: Filter with `label not in OUTLIER_SET` (or `label >= 0`) before `bincount`.
instances: single-instance

### F4 — `set(missing_labels_idx + infinite_labels_idx)` uses numpy broadcasting, not concatenation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))` where both operands are `np.flatnonzero(...)` results (ndarrays). With `missing_labels_idx = [2, 5]` and `infinite_labels_idx = [0]`, the `+` broadcasts the length-1 array over the length-2 array, producing `[2, 5]` instead of the intended concatenation `[2, 5, 0]`. Confirmed by running `np.array([2,5]) + np.array([0])` → `[2, 5]`.
scenario: "Infinite-outlier index 0 is never removed from `clean_idx` → `clean_model.fit(X_outlier[clean_idx])` is fit on data that still contains an `np.inf` row → the “clean vs full” equivalence assertion (`assert_array_equal(clean_labels, labels[clean_idx])`) does not actually verify what the test docstring claims (that removing outliers upfront yields the same labels for the finite points). The test will silently miss regressions in the finite-only path any time infinite outliers exist."
contract: Concatenate with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or use lists and `+`) before wrapping in `set(...)`.
instances: single-instance

### F5 — `max_distance` parameter of `mutual_reachability_graph` is untested
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:44-46, 209-212 — `max_distance` gates the replacement value for infinite mutual-reachability distances in the sparse path: `elif max_distance > 0: data[i] = max_distance`. sklearn/cluster/_hdbscan/tests/test_reachibility.py has no test that exercises non-zero `max_distance` or that verifies infinite entries are replaced correctly. The plan explicitly relies on this behavior via `_hdbscan_brute` (hdbscan.py:243 `max_distance = metric_params.get("max_distance", 0.0)`).
scenario: "regression that stops replacing infinite entries when `max_distance > 0` → silently returns infinite mutual-reachability distances, corrupting MST construction; no test catches this"
contract: Add a sparse-input test with infinite-would-be mutual-reachability values that asserts they are replaced by `max_distance` when it is set (and left as `INFINITY` when it is 0).
instances: single-instance

### F6 — `max_cluster_size` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — no reference to `max_cluster_size` anywhere. The parameter is declared at sklearn/cluster/_hdbscan/hdbscan.py:622-625 and has real semantics at sklearn/cluster/_hdbscan/_tree.pyx:716-717 and 733 (`cluster_sizes[node] > max_cluster_size` forces `is_cluster[node] = False`). The comment at _tree.pyx:717 shows the default sentinel is `n_samples + 1` — regressing this sentinel would silently break EOM cluster selection with no test to catch it.
scenario: "The `max_cluster_size` short-circuit is deleted or its comparison inverted → no test fails."
contract: Add a test that fits HDBSCAN with `cluster_selection_method="eom"` and a small `max_cluster_size`, and asserts that no returned cluster exceeds that size.
instances: single-instance

### F7 — `cluster_selection_method="leaf"` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — a grep for `cluster_selection_method` returns only lines 340 and 354, both passing `"eom"`. The `_get_clusters` branch at sklearn/cluster/_hdbscan/_tree.pyx:761-782 implementing the `"leaf"` method (including the empty-leaves fallback, `get_cluster_tree_leaves`, `recurse_leaf_dfs`, and the leaf + epsilon combination via `epsilon_search`) is added new code with no test coverage.
scenario: "Regression in the leaf-selection branch (e.g., breaking `get_cluster_tree_leaves` or the empty-leaves fallback) → no test fails."
contract: Add at least one test that runs `HDBSCAN(cluster_selection_method="leaf")` on a dataset with a known hierarchical structure and asserts the returned leaf clusters against a reference.
instances: single-instance

### F8 — `test_labelling_thresholding` asserts equal counts of noise, not equal noise positions
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:519-520 and 532-533 — the assertions are `assert sum(num_noise) == sum(labels == -1)`, where `num_noise` is the boolean mask `condensed_tree["value"] < 1` (or `< MAX_LAMBDA`). Two arrays with wholly different positions of `-1` but the same count satisfy this. The test's stated purpose is to verify the correct per-sample thresholding of lambda values, not just the cardinality of noisy points.
scenario: "The threshold branch in `_do_labelling` swaps positions but preserves the noise count → this test still passes."
contract: Assert positional equality of the noise mask, e.g. `assert_array_equal(labels == -1, condensed_tree['value'] < threshold)`.
instances: [sklearn/cluster/tests/test_hdbscan.py:519-520, sklearn/cluster/tests/test_hdbscan.py:532-533]

### F9 — `copy=False` behavior of `HDBSCAN` is untested; only `copy=True` non-mutation is asserted
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:79-92 — the only `assert_allclose(D, D_original)` guarding the copy contract is executed with `copy=True`. There is no test that fits with `copy=False` and verifies either (a) that the caller's array is mutated (proving `copy=False` is the "no-copy" path) or (b) that fit results are identical between `copy=True` and `copy=False`. The `copy` parameter is documented (hdbscan.py:521-526) as a first-class user option controlling in-place modification of precomputed inputs and the brute-force reachability graph, but the default path is completely uncovered.
scenario: "A regression silently makes `copy=False` allocate copies (or `copy=True` mutate) → tests do not detect it."
contract: Add explicit fit-time assertions for `copy=False`: (i) the input distance matrix is mutated in place, and (ii) the resulting labels equal those from `copy=True`.
instances: single-instance

### F10 — `n_jobs` default is `4` but documented as `None`; not covered by any test [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`. The docstring at hdbscan.py:486-490 states: "n_jobs : int, default=None. Number of jobs to run in parallel to calculate distances. `None` means 1 unless in a `joblib.parallel_backend` context. `-1` means using all processors." No test in test_hdbscan.py exercises or verifies the effective `n_jobs` default.
scenario: "A user reads the docstring, expects joblib context to be respected → HDBSCAN silently forces 4 jobs regardless of the surrounding `parallel_backend`, breaking user expectations and joblib coordination."
contract: Set `n_jobs=None` as the default (matching the documented behavior) and add a test asserting the default is `None`.
instances: single-instance

### F11 — `test_hdbscan_algorithms` first assertion ignores the parametrized `metric`
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:136-145 — the test is `@pytest.mark.parametrize("metric", _VALID_METRICS)` yet line 143 calls `HDBSCAN(algorithm=algo).fit_predict(X)` with the default `metric="euclidean"`, then asserts `n_clusters == n_clusters_true`. This assertion runs identically for every `metric` value, so the parametrization does not verify anything about the metric for that portion of the test — the metric is only used after the early return on line 149.
scenario: "regression in a non-euclidean metric path → early per-metric assertion still passes because it silently uses euclidean"
contract: Move the `n_clusters == n_clusters_true` block after the `metric` is applied so it actually exercises the parametrized metric.
instances: single-instance

### F12 — `test_hdbscan_algorithms` invalid-metric branch relies on bare `pytest.raises(ValueError)` and cannot distinguish parameter validation from algorithm-metric mismatch
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:168-170 — `with pytest.raises(ValueError): hdb.fit(X)` has no `match=` argument. HDBSCAN's `_parameter_constraints` (hdbscan.py:626) restricts `metric` to `FAST_METRICS | {"precomputed"}` (plus callable), so many `_VALID_METRICS` entries raise `InvalidParameterError` (a `ValueError` subclass) at `_validate_params`, before the intended KDTree/BallTree metric-validity check at hdbscan.py:772-783 ever runs. The test conflates two different code paths.
scenario: "the specific 'not a valid metric for a KDTree-based algorithm' rejection breaks or moves → test still passes because InvalidParameterError from `_validate_params` masks the change"
contract: Add `match=` matching the specific "is not a valid metric for a .*-based algorithm" message from hdbscan.py:774/781.
instances: single-instance

### F13 — `remap_single_linkage_tree` and `_get_finite_row_indices` have no direct tests
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-407 defines both helpers; grep across `sklearn/cluster/tests/test_hdbscan.py` and `sklearn/cluster/_hdbscan/tests/test_reachibility.py` shows neither symbol is imported or referenced. Their behavior is only exercised transitively through outlier fitting tests, so an off-by-one or index-remap regression could silently pass end-to-end tests that only count clusters.
scenario: "logic change in remap_single_linkage_tree that mis-indexes a single point → transitive tests still pass counts/labels-of-outlier checks; the internal `_single_linkage_tree_` structure is not verified"
contract: Add unit tests that construct a small tree with known non-finite indices and assert the exact structure of the remapped tree.
instances: single-instance

### F14 — `leaf_size` is untested
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py — no reference to `leaf_size`. It is declared at sklearn/cluster/_hdbscan/hdbscan.py:639 and forwarded to `NearestNeighbors` via `_hdbscan_prims` at hdbscan.py:799, 803, 813, 818. There is no test verifying that `leaf_size` is threaded through (e.g., that varying `leaf_size` still produces identical labels).
scenario: "A regression drops `leaf_size` from the `kwargs` dict → no test fails; users silently lose the ability to tune tree leaf size."
contract: Add a test that fits with two different `leaf_size` values and asserts labels are identical.
instances: single-instance

### F15 — `store_centers="centroid"` and `store_centers="medoid"` branches are untested (only `"both"` is exercised)
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:309-327 — the only test of centers passes `store_centers="both"`. The independent branches (`"centroid"` alone should set `centroids_` but not `medoids_`; `"medoid"` alone the reverse) at sklearn/cluster/_hdbscan/hdbscan.py:897-903 are never verified. In particular, no test asserts that attribute `medoids_` is missing when `store_centers="centroid"` (and vice-versa), nor that `store_centers=None` sets no attributes.
scenario: "A regression makes `_weighted_cluster_center` always compute both regardless of `store_centers` → not caught."
contract: Add parametrized tests over `store_centers in {None, "centroid", "medoid", "both"}` asserting exactly which of `centroids_` and `medoids_` are set.
instances: single-instance

### F16 — Assertion `assert counts[unique_labels == -1] > 30` compares an array to a scalar
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:347 — `counts[unique_labels == -1]` is a length-1 ndarray, not a Python scalar; comparing a length-1 array `> 30` returns a length-1 boolean array. `assert` on a length-1 array works today by scalar-truth-test, but the pattern is fragile (a numpy deprecation would break it) and inconsistent with the neighboring `counts[unique_labels == -1] == 2` at test_hdbscan.py:360 which is also a length-1 array vs scalar.
scenario: "Numpy tightens the scalar-conversion rule → these asserts start emitting DeprecationWarning or raising."
contract: Extract the scalar with `.item()` (or index `[0]`) before comparison, e.g. `assert counts[unique_labels == -1].item() > 30`.
instances: [sklearn/cluster/tests/test_hdbscan.py:347, sklearn/cluster/tests/test_hdbscan.py:360]

### F17 — Test filename typo: `test_reachibility.py` for module `_reachability.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py — filename is `reachibility` (misspelled), but it tests functions from `_reachability.pyx` (correctly spelled). The docstring at line 30 also spells `mutual_reachibility_distance`, and _reachability.pyx:127,144,182,206 use `mutual_reachibility_distance` — the misspelling propagates into implementation variable names too, but the test file's name is the surfaced defect for tests-and-verification.
scenario: "Grep/discoverability workflows that search by module basename (`reachability`) miss the test file entirely → contributors may add duplicate tests elsewhere or believe the module is untested."
contract: Rename `test_reachibility.py` to `test_reachability.py` so the test file name matches the module it tests.
instances: single-instance

### F18 — `dbscan_clustering(min_cluster_size=…)` parameter is untested
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-970 defines `dbscan_clustering(self, cut_distance, min_cluster_size=5)`. In sklearn/cluster/tests/test_hdbscan.py:186 and 204, `dbscan_clustering` is only ever called with `cut_distance=...` — no test varies `min_cluster_size`, so the internal filtering `if cluster_size[cluster] < min_cluster_size:` in `labelling_at_cut` (sklearn/cluster/_hdbscan/_tree.pyx:419-420) is not exercised through this public entry point.
scenario: "A regression to the `min_cluster_size` handling in `labelling_at_cut` (e.g., wrong comparison, dropped filter) → the `dbscan_clustering` output silently returns clusters below the requested minimum size, uncaught by tests."
contract: Add a test that calls `dbscan_clustering(cut_distance=..., min_cluster_size=k)` and asserts every non-noise cluster in the result has size ≥ k.
instances: single-instance

### F19 — `n_jobs` parameter is set but never verified to affect parallelism or correctness
severity: low
evidence: `n_jobs` defaults to `4` in `HDBSCAN.__init__` (sklearn/cluster/_hdbscan/hdbscan.py:658). No test in sklearn/cluster/tests/test_hdbscan.py sets `n_jobs` (grep count: 0). The parameter is forwarded into `pairwise_distances` and `NearestNeighbors` calls but its behavior (including e.g., `n_jobs=-1`, `n_jobs=1`) is untested for label-equivalence.
scenario: "A regression in the `n_jobs` handoff (e.g., wrong forwarding, non-deterministic tie-breaking under parallelism) → not caught."
contract: Add a parametrized test asserting `HDBSCAN(n_jobs=1).fit_predict(X) == HDBSCAN(n_jobs=-1).fit_predict(X)`.
instances: single-instance

### F20 — `copy=True` behavior is only verified against `metric="precomputed"`, not against the brute/sparse branches
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:79-88 — `HDBSCAN(metric="precomputed", copy=True).fit_predict(D); assert_allclose(D, D_original)`. The docstring for `copy` (hdbscan.py:521-526) states it applies to precomputed dense/CSR-sparse inputs when `algorithm="brute"`. The sparse `copy=True` path (which triggers a distinct code path via `_brute_mst` and in-place `_sparse_mutual_reachability_graph`) is not tested for immutability of the caller's data.
scenario: "A regression that mutates a caller's CSR precomputed sparse matrix even under `copy=True` → uncaught, silent data corruption for users."
contract: Extend the copy test to cover sparse CSR precomputed inputs, asserting the caller's sparse array remains unchanged after `fit_predict`.
instances: single-instance

### F21 — `test_hdbscan_no_clusters` only checks the cluster count, not that all points are noise-labeled
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:249-251 — `labels = HDBSCAN(min_cluster_size=len(X) - 1).fit_predict(X); n_clusters = len(set(labels) - OUTLIER_SET); assert n_clusters == 0`. This passes as long as no non-outlier label is produced, but says nothing about whether points receive the expected noise label (`-1`) versus, e.g., all being mislabeled as `-2`/`-3` (which are in OUTLIER_SET and thus subtracted).
scenario: "A regression that mislabels all points as `-2` (infinite) or `-3` (missing) on finite data → the \"no clusters\" test still passes; users get incorrect outlier semantics."
contract: Additionally assert `np.all(labels == -1)` for finite input when the min_cluster_size guarantees no cluster.
instances: single-instance
