### F1 — Length mismatch between `X` and `self.labels_` crashes `_weighted_cluster_center` for non-finite inputs
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 (`X = X[finite_index]` shrinks `X` to only finite rows), sklearn/cluster/_hdbscan/hdbscan.py:840-844 (`self.labels_` is rebuilt at `self._raw_data.shape[0]`, i.e. FULL length), sklearn/cluster/_hdbscan/hdbscan.py:854-855 (`self._weighted_cluster_center(X)` is invoked with the SHRUNKEN `X`), sklearn/cluster/_hdbscan/hdbscan.py:908-910 (`mask = self.labels_ == idx` has full length, `X[mask]` then indexes the shrunken `X`).
scenario: "user calls `HDBSCAN(store_centers=\"centroid\").fit(X)` where `X` contains any `np.nan`/`np.inf` row → `_validate_data(force_all_finite=False)` accepts it, `all_finite=False`, `X` is reduced to finite rows only while `self.labels_` is stretched back to `self._raw_data.shape[0]`; `X[mask]` in `_weighted_cluster_center` raises `IndexError: boolean index did not match indexed array along dimension 0` and `fit` aborts with `centroids_`/`medoids_` never set"
contract: pass `self._raw_data` (or another full-length reference to the original data) into `_weighted_cluster_center` in the non-finite branch, so that `mask` and the data it indexes have identical lengths.
instances: single-instance

### F2 — HDBSCAN default `n_jobs=4` diverges from documented default and defeats joblib.parallel_backend
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature has `n_jobs=4,`, while the same file's docstring at lines 486–490 states `n_jobs : int, default=None. Number of jobs to run in parallel to calculate distances. None means 1 unless in a joblib.parallel_backend context.` The value propagates verbatim through `kwargs["n_jobs"] = self.n_jobs` (line 769–775) to `pairwise_distances(..., n_jobs=n_jobs, ...)` (line 245) and `NearestNeighbors(..., n_jobs=n_jobs, ...)` (line 341).
scenario: "user does `with joblib.parallel_backend('loky', n_jobs=1): HDBSCAN().fit(X)` → 4 worker processes still spawn because the concrete `n_jobs=4` is passed through rather than sentinel `None` → resource-oversubscription and behavior contrary to what the docstring promises"
contract: Change the `__init__` default to `n_jobs=None` so both `pairwise_distances` and `NearestNeighbors` see `None` and respect the joblib parallel backend context, matching the documented contract.
instances: single-instance

### F3 — `_sparse_mutual_reachability_graph` silently retains original edge weights when core distance is infinite and `max_distance <= 0`
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:199-200 (`core_distances[i] = INFINITY` when the row has fewer than `further_neighbor_idx+1` explicit non-zeros), sklearn/cluster/_hdbscan/_reachability.pyx:206-212 (in the `nogil` loop, if `mutual_reachibility_distance` is not finite AND `max_distance <= 0`, `data[i]` is left at its ORIGINAL value — no branch updates or errors). The downstream guard in `_brute_mst` at sklearn/cluster/_hdbscan/hdbscan.py:111-123 only detects disconnected components, not per-row "fewer than min_samples neighbors" for still-connected rows.
scenario: "sparse CSR distance matrix where some row has ≥1 but <`min_samples` non-zeros and the whole graph stays connected, `metric_params` omits `max_distance` (default 0.0) → `core_distances[row]=INFINITY`, every edge from that row satisfies `!isfinite(max(...))` and skips both branches, so the returned mutual-reachability graph has the caller's original edge weights for those rows; `csgraph.connected_components` still returns 1 (no error), the MST/clustering are built from stale values, and the user gets a silently wrong result"
contract: when `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, explicitly set `data[i] = INFINITY` so `_brute_mst`'s connected-components check will fire instead of leaving stale finite values in place.
instances: single-instance

### F4 — `_weighted_cluster_center` silently produces meaningless centers when `metric="precomputed"` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 — `fit` unconditionally calls `self._weighted_cluster_center(X)` when `self.store_centers`, regardless of `self.metric`. The docstring at lines 882-883 explicitly warns "This requires `X` to be a raw feature array, not precomputed distances", but no guard exists. For `metric="precomputed"`, `X` at that point is the distance matrix passed by the user, so `np.average(data, weights=strength, axis=0)` at line 912 averages distance-vector rows and `pairwise_distances(data, metric=self.metric, ...)` at 915-917 requests pairwise-distance-of-distance-vectors.
scenario: "user fits `HDBSCAN(metric='precomputed', store_centers='centroid').fit(D)` → `self.centroids_` is populated with averages of rows of the distance matrix (garbage) with no warning or error; the estimator reports success but the stored attribute is nonsensical"
contract: In `fit`, raise `ValueError` if `store_centers is not None and self.metric == "precomputed"` before dispatching to `_weighted_cluster_center`.
instances: single-instance

### F5 — `_weighted_cluster_center` counts label `-3` (missing) as a cluster, producing empty-mask `np.average` crash [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 (`n_clusters = len(set(self.labels_) - {-1, -2})` excludes noise (-1) and infinite-outlier (-2) but NOT missing (-3, `_OUTLIER_ENCODING["missing"]["label"]`) as defined at hdbscan.py:74). The subsequent loop at hdbscan.py:907-920 iterates `range(n_clusters)`; when a `-3` label is present the count is inflated and one `idx` value never matches any real cluster, giving an empty `data`/`strength` slice passed to `np.average(data, weights=strength, axis=0)` (line 912) which raises `ZeroDivisionError: Weights sum to zero`.
scenario: "hypothetical fit path with a `-3` label surviving into `self.labels_` combined with `store_centers != None` → `n_clusters` overcounts by 1, the extra iteration hits `data=X[mask]` with an all-False mask, `np.average` raises ZeroDivisionError. Currently unreachable because F1 raises first, but the bug is intrinsic to this function and will surface as soon as F1 is fixed."
contract: subtract every outlier label defined in `_OUTLIER_ENCODING` (i.e. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}` → `{-1, -2, -3}`) from `set(self.labels_)` when computing `n_clusters`.
instances: single-instance
