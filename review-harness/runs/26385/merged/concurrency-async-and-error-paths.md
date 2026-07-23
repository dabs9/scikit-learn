### F1 — store_centers with non-finite data crashes on boolean-mask/X shape mismatch
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,896,908-909 — line 733 reduces `X = X[finite_index]` (shape `(finite_count, n_features)`), lines 840-844 replace `self.labels_` with a new array of length `self._raw_data.shape[0]` (`raw_count`), then line 855 calls `self._weighted_cluster_center(X)` with the reduced `X`. Inside, line 908 computes `mask = self.labels_ == idx` (shape `(raw_count,)`) and line 909 does `data = X[mask]` where `X.shape[0] == finite_count != raw_count`.
scenario: "user calls `HDBSCAN(store_centers=...).fit(X)` with `metric!='precomputed'` and any `np.nan`/`np.inf` in `X` → NumPy `IndexError: boolean index did not match indexed array along dimension 0`, and the estimator's `centroids_`/`medoids_` attributes remain unset even though `fit` was expected to succeed under the plan's outlier-encoding contract"
contract: When `store_centers` is set and non-finite rows were removed, `_weighted_cluster_center` must be called with the same-sized data that `self.labels_` is indexed against — pass `self._raw_data` (and mask on the full-length labels) or restrict masks to the `finite_index` sub-population.
instances: single-instance

### F2 — `_weighted_cluster_center` excludes only {-1,-2} from cluster count, missing the -3 (missing) outlier label
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. Yet `_OUTLIER_ENCODING["missing"]["label"] = -3` (line 74) and this label is written into `self.labels_` at line 843 for NaN rows. The `for idx in range(n_clusters)` loop at line 907 therefore iterates one past the last real cluster id when any `-3` label is present, producing an empty `data = X[mask]` slice on the extra iteration; `np.average(data, weights=strength, axis=0)` on empty inputs raises `ZeroDivisionError` (weights sum to 0), and the medoid branch calls `np.argmin(dist_mat.sum(axis=1))` on a zero-length axis, which is undefined.
scenario: "if F1 is fixed by feeding the full raw X, a subsequent fit with `store_centers` set and NaN samples → cluster loop runs one extra iteration on a non-existent cluster and crashes with ZeroDivisionError / ValueError from empty-slice reductions"
contract: The exclusion set at line 895 must cover every non-cluster label in `_OUTLIER_ENCODING`. Compute it as `set(self.labels_) - ({-1} | {v["label"] for v in _OUTLIER_ENCODING.values()})` (or equivalently, count labels `>= 0`).
instances: single-instance

### F3 — `n_jobs` default `4` contradicts the documented `None`/joblib-aware default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4` in `__init__`; the docstring at lines 486-490 says `n_jobs : int, default=None` and "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context."
scenario: "user constructs `HDBSCAN()` inside a `joblib.parallel_backend('threading', n_jobs=1)` context expecting per-doc single-threaded behavior → the estimator instead unconditionally spawns 4-way parallelism in `pairwise_distances`/`NearestNeighbors`, causing thread over-subscription and non-reproducible performance"
contract: set the constructor default to `n_jobs=None` so the joblib parallel-backend contract holds, matching the docstring.
instances: single-instance

### F4 — sparse mutual reachability silently drops infinite MR back to the original stored distance when `max_distance <= 0`
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:206-212 — inside `with nogil`, the code writes `data[i] = mutual_reachibility_distance` only when finite, and writes `max_distance` only when `max_distance > 0`; when the true mutual-reachability distance is `INFINITY` **and** `max_distance <= 0` (the default), `data[i]` is left at its original (finite) stored value. Downstream, `_brute_mst` builds the MST from those finite values, so the sanity warning at sklearn/cluster/_hdbscan/hdbscan.py:256-265 ("The minimum spanning tree contains edge weights with value infinity...") is never triggered even though the user actually had rows with fewer than `min_samples` reachable neighbors (`core_distances[i] = INFINITY` was assigned at line 200).
scenario: "user passes a sparse distance matrix without setting `metric_params['max_distance']` and some rows have fewer than `min_samples` stored neighbors → those infinite mutual reachabilities are silently replaced with the original (finite) pairwise distance, the MST is built as if the graph were healthy, no warning is emitted, and the clustering silently degrades to whatever the residual distances imply"
contract: When `isfinite(mutual_reachibility_distance)` is false and no positive `max_distance` was supplied, propagate the infinity into `data[i]` (rather than leaving the pre-transform distance) so the downstream infinite-edge warning at hdbscan.py:256 can fire and users are told to increase neighbor count or set `max_distance`.
instances: single-instance

### F5 — `remap_single_linkage_tree` iterates a set for `non_finite`, docstring promises an ndarray
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)`; the function docstring at line 364-365 declares `non_finite : ndarray, Boolean array of which entries in the raw data are non-finite`; line 388 then does `for i, outlier in enumerate(non_finite):` — enumerating a set yields Python-hash-order elements, so the `outlier_tree[i]` rows encode `left_node = <raw index>` in a non-contract order.
scenario: "future refactor swaps the caller to pass the boolean ndarray promised by the docstring (or PYTHONHASHSEED randomization changes set iteration) → `self._single_linkage_tree_` outlier rows differ in ordering between runs even with fixed seeds, breaking any downstream consumer that assumes canonical tree ordering"
contract: pass and iterate a sorted sequence (e.g. `np.sort(np.fromiter(non_finite, dtype=int))`) so the outlier rows have a deterministic, documented ordering.
instances: single-instance
