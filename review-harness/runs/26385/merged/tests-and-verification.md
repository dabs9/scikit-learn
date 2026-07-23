### F1 — test_hdbscan_precomputed_non_brute uses non-existent algorithm names, so the assertion never exercises the tested behavior
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `@pytest.mark.parametrize("tree", ["kd", "ball"])` then `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` inside a `with pytest.raises(ValueError):`, but `_parameter_constraints["algorithm"]` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only allows `{"auto", "brute", "kdtree", "balltree"}`.
scenario: "test runs → parameter validator raises `InvalidParameterError` (subclass of `ValueError`) for the unknown value `prims_kdtree` before the precomputed+tree code path is entered → test passes for the wrong reason and gives no coverage of the precomputed-with-tree rejection its docstring advertises"
contract: Fix the parameter values to real algorithm strings (`"kdtree"`, `"balltree"`) and tighten the assertion with a `match=` regex that pins the specific precomputed/tree-metric error message emitted at hdbscan.py:772-783 (`"is not a valid metric for a KDTree-based algorithm"` / `"BallTree-based algorithm"`).
instances: single-instance

### F2 — test_hdbscan_centers uses `rtol=1`, dwarfing the reference and letting wildly wrong centers pass
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:314-320 — `centers = [(0.0, 0.0), (3.0, 3.0)]` compared with `assert_allclose(center, centroid, rtol=1, atol=0.05)` and `assert_allclose(center, medoid, rtol=1, atol=0.05)`. For `center=(3, 3)` the effective tolerance is `atol + rtol*|desired| = 0.05 + 1*3 = 3.05`, so any centroid inside `[-0.05, 6.05]` passes.
scenario: "regression pushes centroid/medoid computation off by up to 3.0 units (e.g., wrong weighting, wrong mask, wrong cluster) → test still passes because rtol=1 swallows the error → silent breakage of `store_centers` slips through CI"
contract: Use a tight absolute tolerance (e.g., `rtol=0, atol=0.05` — the blobs have `cluster_std=0.5` and the test compares to blob centers) and, since HDBSCAN's cluster label order is not guaranteed to align with `centers`, match centroids to centers by nearest-neighbour before comparing rather than by `zip`.
instances: single-instance

### F3 — Outlier-removal test builds `clean_idx` with numpy broadcasting instead of list concatenation, so an outlier row is retained in the "clean" model
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:206-215 — `missing_labels_idx = np.flatnonzero(...)` and `infinite_labels_idx = np.flatnonzero(...)` are `ndarray`s; `missing_labels_idx + infinite_labels_idx` performs broadcast arithmetic (e.g. `np.array([2,5]) + np.array([0]) == np.array([2,5])`), not list concatenation, so `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))` retains index 0, and `X_outlier[clean_idx]` still contains the infinite outlier row that was supposed to be filtered out.
scenario: "The 'clean' model is refit on data still containing an infinite outlier → the test does NOT verify that HDBSCAN produces identical labels on truly-outlier-free data; the assertion passes only because both sides re-detect the retained outlier and label it -2"
contract: Compute `clean_idx = list(set(range(200)) - set(missing_labels_idx.tolist()) - set(infinite_labels_idx.tolist()))` so the removal set is the union of index sets, not the elementwise sum.
instances: single-instance

### F4 — test_labelling_distinct passes even when every sample collapses to a single label
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:489-492 — `first_with_label = {_y: np.where(y == _y)[0][0] for _y in list(set(y))}`, `y_to_labels = {_y: labels[first_with_label[_y]] for _y in list(set(y))}`, `aligned_target = np.vectorize(y_to_labels.get)(y)`, `assert_array_equal(labels, aligned_target)`. If `_do_labelling` returns constant labels (e.g., all zeros), `y_to_labels` maps every distinct `_y` to the same value, so `aligned_target == labels` trivially.
scenario: "`_do_labelling` regresses to emit the same label for every sample → aligned_target degenerates to the same constant → assertion passes → the 'correctly assigns labels' test claim is not defended"
contract: After the alignment check, additionally assert that the number of distinct non-noise labels matches the number of distinct classes in `y` (three blobs → at least three distinct predicted labels excluding `-1`).
instances: single-instance

### F5 — test_hdbscan_algorithms asserts `pytest.raises(ValueError)` with no `match=`, so unrelated errors also pass
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:168-170 — inside the parametrised metric/algo loop, `if metric not in ALGOS_TREES[algo].valid_metrics(): with pytest.raises(ValueError): hdb.fit(X)`. The specific error string at sklearn/cluster/_hdbscan/hdbscan.py:773-776 / 780-783 (`"is not a valid metric for a KDTree-based algorithm. Please select a different metric."`) is never pinned.
scenario: "an unrelated `ValueError` (e.g., input validation, metric_params, shape) is raised for the same metric → test still passes → the intended KDTree/BallTree metric-rejection guard is not actually exercised"
contract: Add `match=r"is not a valid metric for a (KDTree|BallTree)-based algorithm\. Please select a different metric\."` to the `pytest.raises` call so the assertion pins the specific error emitted from hdbscan.py:772-783.
instances: single-instance

### F6 — `test_hdbscan_algorithms` ignores the parametrized `metric` for its main assertion and asserts nothing at all in the primary happy-path branch
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:136-175 — the first fit `labels = HDBSCAN(algorithm=algo).fit_predict(X)` at line 143 does not pass `metric`, so the outer `@pytest.mark.parametrize("metric", _VALID_METRICS)` has no effect on that call (default `"euclidean"` is used every time). Then `if algo in ("brute", "auto"): return` returns early, so half of the parametrization combinations run only the metric-agnostic assertion. In the final `else: hdb.fit(X)` branch (valid tree metric) there is no assertion at all — the fit executing without raising is the entire "test".
scenario: "A regression that produces silently wrong (but not error-raising) labels for e.g. `algorithm='balltree', metric='canberra'` → the test still passes"
contract: Thread `metric=metric` (and `metric_params`) into the first `HDBSCAN(...)` call, and add an assertion on the resulting `labels_` in the final `else` branch (e.g. `n_clusters` sanity check or an equivalence check against a reference algorithm).
instances: single-instance

### F7 — New public-API surface has no dedicated tests: `max_cluster_size` and `cluster_selection_method="leaf"`
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — grep for `max_cluster_size` returns no matches; grep for `cluster_selection_method` shows only `"eom"` (lines 340, 354). Meanwhile `sklearn/cluster/_hdbscan/hdbscan.py:622-625` declares `max_cluster_size` as a parameter and `hdbscan.py:641` declares `"leaf"` in the StrOptions; both are wired through `tree_to_labels` (`_tree.pyx:56-77`) and `_get_clusters` (`_tree.pyx:642-782`).
scenario: "A regression in the `max_cluster_size` truncation branch (`_tree.pyx:733`) or the `cluster_selection_method == 'leaf'` branch (`_tree.pyx:761`) → CI is silent"
contract: Add tests that (a) verify `HDBSCAN(max_cluster_size=k)` yields no cluster of size > k on a dataset that would otherwise produce a larger cluster, and (b) verify that `cluster_selection_method="leaf"` returns more (and finer-grained) clusters than `"eom"` on a hierarchical test dataset.
instances: single-instance

### F8 — test_hdbscan_precomputed_dense_nan feeds a non-square array, coupling the nan assertion to check ordering
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:449-454 — `X_nan = X.copy()` where module-level `X` is shape `(200, 2)` (feature array, not a distance matrix); then `HDBSCAN(metric="precomputed").fit(X_nan)` is expected to raise `"np.nan values found in precomputed-dense"`. The nan check at sklearn/cluster/_hdbscan/hdbscan.py:748-751 happens before the symmetry/shape check at sklearn/cluster/_hdbscan/hdbscan.py:222-234, so the test only works because of the current ordering; if a future change hoists the shape check earlier the message would change to `"has shape"` and this test would fail for reasons unrelated to nan-detection.
scenario: "shape/symmetry validation moves ahead of the nan check → test raises with the shape error instead → assertion fails not because nan-detection broke but because a different (still-correct) validator won the race"
contract: Use a symmetric square input with a nan entry (e.g., `D = euclidean_distances(X); D[0, 0] = np.nan`) so the test isolates nan-detection independent of check ordering.
instances: single-instance

### F9 — test_hdbscan_usable_inputs is a raise-no-error smoke test with no observable assertion
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:389-394 — `def test_hdbscan_usable_inputs(X, kwargs): "..." HDBSCAN(min_samples=1, **kwargs).fit(X)`. There is no assertion on labels, probabilities, `n_features_in_`, or the returned estimator. Any silent regression that produces bogus but non-raising output on these tiny inputs passes.
scenario: "`.fit` silently returns garbage labels for the 2-sample precomputed-inf case → test still passes because no output is inspected → coverage over the non-finite/degenerate-input path is illusory"
contract: Add at least one concrete post-fit assertion per input (e.g., `assert model.labels_.shape == (n_samples,)` and, for the inf-precomputed case, that the sample containing `np.inf` receives the `_OUTLIER_ENCODING["infinite"]["label"]`).
instances: single-instance

### F10 — `test_dbscan_clustering` self-declared weak; only asserts cluster count
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:178-188 — docstring: `"This test is more of a sanity check than a rigorous evaluation. TODO: Improve and strengthen this test if at all possible."`; body only asserts `n_clusters == n_clusters_true`. No comparison against a reference DBSCAN output or expected labels; any implementation that returns three groups (even wrong ones) passes.
scenario: "A bug in `dbscan_clustering` at hdbscan.py:923-970 that permutes labels or misassigns points but still yields 3 groups → test still passes → `dbscan_clustering` returns silently wrong labels."
contract: Compare the returned labels against `cluster.DBSCAN(eps=0.3, min_samples=...).fit_predict(X)` (up to label permutation via `fowlkes_mallows_score >= 0.9`), so the test asserts the actual clustering, not just its cardinality.
instances: single-instance

### F11 — `test_hdbscan_min_cluster_size` skips its assertion when no non-noise labels exist, weakening the loop to a no-op for large `min_cluster_size`
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:259-263 — `for min_cluster_size in range(2, len(X), 1): ... if len(true_labels) != 0: assert np.min(np.bincount(true_labels)) >= min_cluster_size`. When `min_cluster_size` grows large enough that every point is noise, the assertion is skipped entirely, so the tail of the iteration space (which is precisely where the invariant is most interesting/hardest) contributes nothing.
scenario: "A regression that emits an under-sized cluster only in the high-`min_cluster_size` regime where noise is common → this test's guard silently swallows it"
contract: Replace the truthy guard with `assert len(true_labels) == 0 or np.min(np.bincount(true_labels)) >= min_cluster_size`, so the loop iteration still records a positive assertion, and add a coverage assertion that at least one iteration in the loop produced a non-empty `true_labels` (guarding against the whole loop degenerating into skips).
instances: single-instance
