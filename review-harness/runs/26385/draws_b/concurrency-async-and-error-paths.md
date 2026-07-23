Based on my careful review of the changed files through the concurrency-async-and-error-paths lens, here are my findings:

### F1 — `store_centers` with non-finite input crashes: reduced-X vs. raw-length labels
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855 — at line 733 `X = X[finite_index]` reassigns `X` to only the finite rows; at line 844 `self.labels_ = new_labels` (of shape `self._raw_data.shape[0]`); then at line 855 `self._weighted_cluster_center(X)` is called with the *reduced* `X`, while inside (line 908) `mask = self.labels_ == idx` has raw length. `X[mask]` mixes a boolean mask of length `n_raw` with an array of length `n_finite`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data that contains any np.nan or +/-np.inf row → numpy raises `IndexError: boolean index did not match indexed array along dimension 0` during `fit`, so the estimator never returns and no attributes are set"
contract: pass `self._raw_data` (not the finite-subset X) into `_weighted_cluster_center`, since `self.labels_`/`self.probabilities_` are now raw-length after the outlier remap.
instances: single-instance

### F2 — `_weighted_cluster_center` cluster count omits the `-3` missing label
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the outlier encoding at lines 65-79 defines `missing` with `label=-3`, which is not excluded here.
scenario: "input contains np.nan rows so `self.labels_` contains -3 → `n_clusters` is one higher than the number of real clusters → the loop at line 907 runs one extra iteration for `idx == n_clusters-1` matching no samples → `np.average(data, weights=strength, axis=0)` receives empty arrays and raises `ZeroDivisionError`/`ValueError`"
contract: exclude every outlier label — subtract `{-1, -2, -3}` (or the values from `_OUTLIER_ENCODING`) so `n_clusters` matches the real (non-outlier) clusters.
instances: single-instance

### F3 — `n_jobs` default `4` contradicts the documented `None`/joblib-aware default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4` in `__init__`; the docstring at lines 486-490 says `n_jobs : int, default=None` and "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context."
scenario: "user constructs `HDBSCAN()` inside a `joblib.parallel_backend('threading', n_jobs=1)` context expecting per-doc single-threaded behavior → the estimator instead unconditionally spawns 4-way parallelism in `pairwise_distances`/`NearestNeighbors`, causing thread over-subscription and non-reproducible performance"
contract: set the constructor default to `n_jobs=None` so the joblib parallel-backend contract holds, matching the docstring.
instances: single-instance

### F4 — `remap_single_linkage_tree` iterates a set for `non_finite`, docstring promises an ndarray
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)`; the function docstring at line 364-365 declares `non_finite : ndarray, Boolean array of which entries in the raw data are non-finite`; line 388 then does `for i, outlier in enumerate(non_finite):` — enumerating a set yields Python-hash-order elements, so the `outlier_tree[i]` rows encode `left_node = <raw index>` in a non-contract order.
scenario: "future refactor swaps the caller to pass the boolean ndarray promised by the docstring (or PYTHONHASHSEED randomization changes set iteration) → `self._single_linkage_tree_` outlier rows differ in ordering between runs even with fixed seeds, breaking any downstream consumer that assumes canonical tree ordering"
contract: pass and iterate a sorted sequence (e.g. `np.sort(np.fromiter(non_finite, dtype=int))`) so the outlier rows have a deterministic, documented ordering.
instances: single-instance
