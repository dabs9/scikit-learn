### F1 — `_weighted_cluster_center` miscounts `n_clusters` when `-3` (missing) outliers exist
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` excludes only `-1` and `-2`, but `_OUTLIER_ENCODING["missing"]["label"] = -3` is documented (and applied at hdbscan.py:849) as a possible label.
scenario: "user sets `store_centers` and X contains any `np.nan` rows → `labels_` contains `-3` → set-diff yields `n_clusters + 1` entries → loop iterates one extra `idx` for which `mask` is all-False → `np.average` on empty data with zero-sum weights raises `ZeroDivisionError`, and the medoid branch's `np.argmin` on an empty array raises `ValueError`."
contract: exclude the full outlier label set (e.g. `set(self.labels_) - {v['label'] for v in _OUTLIER_ENCODING.values()} - {-1}`) so the loop only visits real cluster indices; the docstrings at hdbscan.py:567/578 already promise this behavior.
instances: single-instance

### F2 — Redundant `births` allocation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed, then immediately overwritten by an identical `births = np.full(...)` on the next line before the fill loop.
scenario: "each call to `_compute_stability` wastes one full-size allocation + fill of a NaN array → measurable in tight loops on large trees, and is confusing to readers who assume the first line does something distinct."
contract: delete the first allocation at line 252 so `births` is allocated exactly once.
instances: single-instance

### F3 — `HDBSCAN` default `n_jobs=4` breaks scikit-learn's `n_jobs=None` convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — signature has `n_jobs=4`; the docstring at hdbscan.py:486-489 states "`None` means 1 unless in a `joblib.parallel_backend` context. `-1` means using all processors." (i.e. describes the standard convention, not the actual default 4).
scenario: "users who instantiate `HDBSCAN()` unknowingly consume 4 worker threads/processes for pairwise-distance work → contention on shared hosts, unexpected memory, and behavior that contradicts every other estimator in `sklearn.cluster`; users following the documented `None` semantics from the docstring get surprising parallelism."
contract: change the default to `n_jobs=None` to match the documented and codebase-wide convention.
instances: single-instance

### F4 — `test_dbscan_clustering_outlier_data` uses numpy `+` where set concatenation was intended
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:218 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`; `missing_labels_idx` and `infinite_labels_idx` are 1-D numpy arrays produced by `np.flatnonzero(...)`. Numpy broadcasts the addition instead of concatenating, so with `missing_labels_idx == [2, 5]` and `infinite_labels_idx == [0]`, the subtracted set is `{2, 5}`, not `{0, 2, 5}`.
scenario: "the infinite-outlier row (index 0) is NOT excluded from `clean_idx` → `clean_model` is fit on data that still contains an `np.inf` row → the test happens to pass because both models label that row `-2`, but the test no longer verifies the property it claims (equivalence of clean vs. outlier-removed fits)."
contract: build the exclusion set from the concatenation, e.g. `set(missing_labels_idx.tolist()) | set(infinite_labels_idx.tolist())` (or `np.concatenate([...])`), so all documented outlier indices are actually removed.
instances: single-instance

### F5 — `test_hdbscan_precomputed_non_brute` passes an invalid algorithm and asserts nothing about precomputed inputs
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282-284 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` with `tree` in `("kd", "ball")`; but the `algorithm` parameter constraint at hdbscan.py:629-638 only accepts `{"auto", "brute", "kdtree", "balltree"}`, so `_validate_params()` raises `InvalidParameterError` (subclass of `ValueError`) before any precomputed handling is exercised.
scenario: "the test docstring says 'HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm', but the raise actually comes from unknown-algorithm validation and would trigger regardless of `metric` → a future regression that silently accepts precomputed + tree algorithms would not be caught."
contract: use a valid tree algorithm name (`algorithm="kdtree"` / `"balltree"`) and assert on the precomputed-error message emitted deeper in `fit`, so the test actually pins the intended contract.
instances: single-instance

### F6 — `_hdbscan_brute` and `_hdbscan_prims` docstrings misdocument defaults and list an unused parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:163-172 (signature: `min_samples=5, alpha=None`) vs. docstring at 185/189 (`min_samples : int, default=None`, `alpha : float, default=1.0`); sklearn/cluster/_hdbscan/hdbscan.py:275-283 (signature: `min_samples=5`, no `copy` parameter) vs. docstring at 296 (`min_samples : int, default=None`) and 319-324 (documents `copy : bool, default=False` that does not exist in the signature).
scenario: "readers of the internal API get the wrong defaults from the docstrings → anyone extending or calling these internals passes the docstring's `None`/`copy=False` and gets silently different behavior; the phantom `copy` parameter also misleads anyone trying to add copy semantics to the tree algorithms."
contract: sync each `default=` and parameter list to the actual signatures — `min_samples : int, default=5`; `alpha : float, default=None` in `_hdbscan_brute` (or change the signature to `alpha=1.0` if the default was the intent); delete the `copy` docstring block from `_hdbscan_prims`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:185, sklearn/cluster/_hdbscan/hdbscan.py:189, sklearn/cluster/_hdbscan/hdbscan.py:296, sklearn/cluster/_hdbscan/hdbscan.py:319]

### F7 — User-guide typos and stray colons in the new HDBSCAN section
severity: low
evidence: doc/modules/clustering.rst:1009-1011 — "removing any edges with value greater than :math:`\varepsilon`:\nfrom the original graph. Any points whose core distance is less than :math:`\varepsilon`:\nare at this staged marked as noise." Both math roles carry a trailing colon that renders literally, and "at this staged marked" should be "at this stage marked". doc/modules/clustering.rst:1025-1026 — "the fully\n-connected mutual reachability graph" renders with a stray hyphen/space split.
scenario: "Sphinx build renders the trailing `:` literally after each formula and prints 'at this staged marked' → published user guide has broken punctuation and a grammar error immediately after the algorithm's central formulas, which readers arriving at the new section see first."
contract: remove the trailing `:` after each `:math:` role, fix "staged" → "stage", and rejoin "fully-connected" onto one line.
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011, doc/modules/clustering.rst:1025]

### F8 — `plot_hdbscan.py` "Scale Invariance" loop never rescales the fitted data
severity: low
evidence: examples/cluster/plot_hdbscan.py:112-116 — the scale-invariance demonstration is `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ...)`. `X` (not `X * scale`) is fit and plotted every iteration, so the three panels are identical and do not demonstrate scale-invariance at all.
scenario: "example is rendered into the gallery with `scale` unused → the page shows three identical plots labeled with different `scale` values, contradicting the surrounding prose that promises 'HDBSCAN is scale-invariant' via a visual comparison."
contract: fit and plot on the scaled data — `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` — mirroring the DBSCAN loop just above.
instances: single-instance
