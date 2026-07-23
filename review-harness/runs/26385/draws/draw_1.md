### F1 — `HDBSCAN.__init__` default `n_jobs=4` contradicts docstring and sklearn convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in the `__init__` signature, while the docstring at :486 documents `n_jobs : int, default=None`. Sklearn's convention (and every other estimator listed in the whats_new entry) is `n_jobs=None`, meaning "1 unless in a `joblib.parallel_backend` context". A hard-coded `4` silently spawns 4 worker threads regardless of the user's `parallel_backend`.
scenario: "user creates `HDBSCAN()` inside `with joblib.parallel_backend('threading', n_jobs=1):` → sklearn spawns 4 threads anyway, contradicting the documented behavior and the context manager the user set"
contract: Change the default in `__init__` to `n_jobs=None` so behavior matches the docstring and sklearn glossary semantics.
instances: single-instance

### F2 — `_weighted_cluster_center` uses raw-length mask against filtered `X` when non-finite samples exist
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 — `if self.store_centers: self._weighted_cluster_center(X)`. In the non-finite branch (line 733 `X = X[finite_index]`) `X` is the filtered array of length `len(finite_index)`, whereas by line 844 `self.labels_ = new_labels` was replaced with an array of length `self._raw_data.shape[0]`. Inside `_weighted_cluster_center` at :908-909 `mask = self.labels_ == idx` therefore has raw length N, but `X[mask]` indexes into the shorter filtered X — numpy raises `IndexError: boolean index did not match indexed array along dimension 0` (raw N ≠ filtered N). Additionally at :895 `n_clusters = len(set(self.labels_) - {-1, -2})` fails to also exclude the `-3` missing label, so with `store_centers` and missing samples `n_clusters` is off-by-one and later `mask = self.labels_ == idx` for `idx == n_clusters-1` selects zero rows or the wrong rows.
scenario: "user fits `HDBSCAN(store_centers='centroid').fit(X)` where `X` contains any `np.inf`/`np.nan` row → `fit` raises `IndexError` instead of returning cluster centroids"
contract: In the non-finite branch pass the pre-filter raw data (or apply `finite_index` to both `X` and the mask) so that mask and data share the same length, and exclude `-3` from the `n_clusters` computation together with `-1`/`-2`. There is no test covering `store_centers` combined with non-finite input, so this path is uncaught.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:854, sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/_hdbscan/hdbscan.py:908]

### F3 — Duplicated `births` allocation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and :254 — two identical `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` statements back-to-back. The first is immediately overwritten by the second and does nothing.
scenario: "dead line allocates and discards an array on every call → wasted allocation and misleading code for readers"
contract: Delete line 252; keep only the assignment at line 254.
instances: single-instance

### F4 — `test_hdbscan_precomputed_non_brute` triggers `InvalidParameterError`, not the error under test
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`. The valid `algorithm` values in `_parameter_constraints` (sklearn/cluster/_hdbscan/hdbscan.py:629-638) are `{"auto","brute","kdtree","balltree"}`, so `"prims_kdtree"`/`"prims_balltree"` fail parameter validation before any precomputed-vs-tree logic runs. The assertion `pytest.raises(ValueError)` still passes because `InvalidParameterError` is a subclass of `ValueError`, but the test never exercises the code path it names in its docstring ("correctly raises an error when passing precomputed data while requesting a tree-based algorithm").
scenario: "test claims to guard the precomputed-vs-tree combination but actually only guards the algorithm StrOptions validator → the intended incompatibility check becomes uncovered and any regression there ships unnoticed"
contract: Use the actual algorithm strings supported by this estimator (`"kdtree"`, `"balltree"`) and assert with `match=` on the "Sparse data matrices only support algorithm `brute`" style message so the test fails if that guard is removed. Also verify a corresponding guard exists for the precomputed dense case.
instances: single-instance

### F5 — `test_dbscan_clustering_outlier_data` computes `clean_idx` with numpy elementwise addition
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` and `infinite_labels_idx` are `np.ndarray` results from `np.flatnonzero` (:206, :209), so `+` broadcasts elementwise, not list-concatenation. With `missing_labels_idx=[2,5]` and `infinite_labels_idx=[0]`, `missing_labels_idx + infinite_labels_idx == np.array([2,5])`, so index `0` (the infinite sample) is not excluded from `clean_idx`. `X_outlier[clean_idx]` therefore still contains `np.inf`, so the "clean_model" is not actually clean — the assertion still holds only because `HDBSCAN` reassigns the same `-2`/`-3` labels on both runs.
scenario: "test appears to compare a clean fit against an outlier-marked fit but actually keeps the infinite sample in both → the intended sanity check ('clean subset yields identical labels') is not exercised"
contract: Concatenate the index arrays explicitly, e.g. `np.concatenate([missing_labels_idx, infinite_labels_idx])`, before constructing the set.
instances: single-instance

### F6 — `remap_single_linkage_tree` docstring conflicts with actual call-site type
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 documents `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`, but the call site at :838 passes `non_finite=set(infinite_index + missing_index)` — a Python set of integer raw-data indices. The function's body relies on that set behaviour (`len(non_finite)` at :369 and `for i, outlier in enumerate(non_finite)` at :388 where `outlier` is used as a raw-data index at :389), which would not work with a boolean array. Iterating a `set` also gives non-deterministic order across Python invocations, so `self._single_linkage_tree_` for the non-finite path is nondeterministic between runs.
scenario: "reader relies on the docstring → passes a boolean mask and gets wrong results or an exception; separately, `_single_linkage_tree_` values differ across process invocations for otherwise identical inputs, complicating debugging and downstream serialization"
contract: Rewrite the docstring to state that `non_finite` is a collection of raw-data indices of non-finite rows, and pass a `sorted(...)` sequence (e.g. `sorted(set(infinite_index + missing_index))`) at :838 so ordering is deterministic.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:364, sklearn/cluster/_hdbscan/hdbscan.py:388, sklearn/cluster/_hdbscan/hdbscan.py:838]

### F7 — Cross-reference `:ref:\`User Guide <HDBSCAN>\`` in example uses wrong-case anchor
severity: low
evidence: examples/cluster/plot_hdbscan.py:110 — `see :ref:\`User Guide <HDBSCAN>\``. The corresponding label defined in `doc/modules/clustering.rst:51` is `.. _hdbscan:` (lowercase). Sphinx label lookups are case-sensitive, so this cross-reference fails to resolve and the rendered example will emit a Sphinx warning and produce a broken link.
scenario: "docs build → Sphinx warns about unknown label `HDBSCAN`; user clicking the User Guide link in the rendered example hits a dead reference"
contract: Change the reference to `:ref:\`User Guide <hdbscan>\`` to match the declared label.
instances: single-instance
