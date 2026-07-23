# PR 26385 Review Report

**Commit range:** `86541f2b3bc8a96264e265cb810cf79858544340..04c8b6954e3e4b8f8af086cea9d386f954b76bfe`

**Findings by severity:** high=19, medium=81, low=148 (total=248)

## HIGH severity (19)

### [HIGH] `plot_hdbscan.py` scale-invariance demo never scales the data

User runs the published example → all three panes show the identical HDBSCAN result on the same unscaled data, so the section titled 'Scale Invariance' visually 'proves' scale invariance by never varying scale; the demonstration is meaningless and misleading Use `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` inside the loop, mirroring the DBSCAN loop that immediately precedes it.

<details><summary>verbatim finding</summary>

```
### F116 — `plot_hdbscan.py` scale-invariance demo never scales the data
severity: high
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ...)` uses `scale` only in the plot title; `X` is neither multiplied by `scale` on `fit` nor on `plot`, unlike the DBSCAN loop above (line 93-95) which does `dbs.fit(X * scale)` and `plot(X * scale, ...)`.
scenario: "User runs the published example → all three panes show the identical HDBSCAN result on the same unscaled data, so the section titled 'Scale Invariance' visually 'proves' scale invariance by never varying scale; the demonstration is meaningless and misleading"
contract: Use `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` inside the loop, mirroring the DBSCAN loop that immediately precedes it.
instances: single-instance
```
</details>

### [HIGH] `_weighted_cluster_center` shape mismatch when non-finite data + `store_centers` set

user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan/np.inf rows → boolean-mask indexing raises `IndexError: boolean index did not match indexed array along dimension 0` pass `self._raw_data` (and update the mask/probabilities selection to align with raw data indexing) so `X` and the boolean mask derived from `self.labels_` share the same length.

<details><summary>verbatim finding</summary>

```
### F1 — `_weighted_cluster_center` shape mismatch when non-finite data + `store_centers` set
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 passes the finite-filtered local `X` (from line 733: `X = X[finite_index]`) into `_weighted_cluster_center`, but by that point `self.labels_` has been re-expanded at lines 840-844 to shape `(self._raw_data.shape[0],)`. Inside `_weighted_cluster_center` at line 908, `mask = self.labels_ == idx` has length `n_raw`, then line 909 executes `data = X[mask]` where `X` has length `n_finite < n_raw`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan/np.inf rows → boolean-mask indexing raises `IndexError: boolean index did not match indexed array along dimension 0`"
contract: pass `self._raw_data` (and update the mask/probabilities selection to align with raw data indexing) so `X` and the boolean mask derived from `self.labels_` share the same length.
instances: single-instance
```
</details>

### [HIGH] `_weighted_cluster_center` treats `-3` (missing) as a valid cluster label

user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan rows → inflated `n_clusters`, empty-cluster iteration, ZeroDivisionError from `np.average` subtract all outlier labels — `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.

<details><summary>verbatim finding</summary>

```
### F2 — `_weighted_cluster_center` treats `-3` (missing) as a valid cluster label
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 computes `n_clusters = len(set(self.labels_) - {-1, -2})`, omitting `-3`. `_OUTLIER_ENCODING["missing"]["label"] = -3` (sklearn/cluster/_hdbscan/hdbscan.py:74) is a documented outlier label. When missing-data outliers are present, `-3` remains in the label set and inflates `n_clusters` by one; the loop at line 907 iterates `range(n_clusters)` looking for `self.labels_ == idx` and one of those idx values will have no matching rows, so `np.average(data, weights=strength, axis=0)` at line 912 receives an empty `data`/`strength` and raises `ZeroDivisionError: Weights sum to zero, can't be normalized`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan rows → inflated `n_clusters`, empty-cluster iteration, ZeroDivisionError from `np.average`"
contract: subtract all outlier labels — `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.
instances: single-instance
```
</details>

### [HIGH] `_hdbscan_prims` forwards `max_distance` to `DistanceMetric.get_metric`, causing an unexpected-keyword failure

User follows the error message from `_brute_mst` and passes `metric_params={'max_distance': 10}` → later calls with a non-precomputed metric route through `_hdbscan_prims`, which forwards the key into `DistanceMetric.get_metric(...)` and dies with a `TypeError: __init__() got an unexpected keyword argument 'max_distance'`, leaking an internal-looking error from user-driven input rather than a validated boundary check. Whitelist/strip `metric_params` keys that are HDBSCAN-only sentinels (currently `max_distance`) before forwarding them into `DistanceMetric.get_metric` and `NearestNeighbors`, so misplaced-trust in caller-supplied kwargs cannot produce cryptic downstream TypeErrors from third-party metric constructors.

<details><summary>verbatim finding</summary>

```
### F20 — `_hdbscan_prims` forwards `max_distance` to `DistanceMetric.get_metric`, causing an unexpected-keyword failure
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:344 — `dist_metric = DistanceMetric.get_metric(metric, **metric_params)` is invoked with the raw user-supplied `metric_params` dict; that same dict is documented (hdbscan.py:120–122, 174–178 of `_reachability` docstring) to carry a `max_distance` key used by the sparse mutual-reachability path. Unlike `_hdbscan_brute` (line 243), the prims path never pops/filters `max_distance` before splatting.
scenario: "User follows the error message from `_brute_mst` and passes `metric_params={'max_distance': 10}` → later calls with a non-precomputed metric route through `_hdbscan_prims`, which forwards the key into `DistanceMetric.get_metric(...)` and dies with a `TypeError: __init__() got an unexpected keyword argument 'max_distance'`, leaking an internal-looking error from user-driven input rather than a validated boundary check."
contract: Whitelist/strip `metric_params` keys that are HDBSCAN-only sentinels (currently `max_distance`) before forwarding them into `DistanceMetric.get_metric` and `NearestNeighbors`, so misplaced-trust in caller-supplied kwargs cannot produce cryptic downstream TypeErrors from third-party metric constructors.
instances: single-instance
```
</details>

### [HIGH] Shape mismatch in `_weighted_cluster_center` after non-finite pruning

User fits HDBSCAN with `store_centers=\"centroid\"|\"medoid\"|\"both\"` on data containing any `np.nan`/`np.inf` row → boolean-mask/array shape mismatch (or silently wrong-indexed rows) when computing centers, raising `IndexError`/producing garbage centroids. When `all_finite` is False, pass `self._raw_data` (or an aligned finite subset paired with re-indexed labels/probabilities) into `_weighted_cluster_center` so that `mask.shape == X.shape[0]`.

<details><summary>verbatim finding</summary>

```
### F34 — Shape mismatch in `_weighted_cluster_center` after non-finite pruning
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 pares `X = X[finite_index]`; sklearn/cluster/_hdbscan/hdbscan.py:840-844 rebuilds `self.labels_` to `self._raw_data.shape[0]` (raw length); sklearn/cluster/_hdbscan/hdbscan.py:854-855 then calls `self._weighted_cluster_center(X)` passing the pruned `X`; sklearn/cluster/_hdbscan/hdbscan.py:908 does `mask = self.labels_ == idx` (raw length) and sklearn/cluster/_hdbscan/hdbscan.py:909 does `data = X[mask]` on the pruned X.
scenario: "User fits HDBSCAN with `store_centers=\"centroid\"|\"medoid\"|\"both\"` on data containing any `np.nan`/`np.inf` row → boolean-mask/array shape mismatch (or silently wrong-indexed rows) when computing centers, raising `IndexError`/producing garbage centroids."
contract: When `all_finite` is False, pass `self._raw_data` (or an aligned finite subset paired with re-indexed labels/probabilities) into `_weighted_cluster_center` so that `mask.shape == X.shape[0]`.
instances: single-instance
```
</details>

### [HIGH] `remap_single_linkage_tree` `outlier_count` shift under-adjusts internal ids

User fits with a row that is *both* `np.inf` and `np.nan` in different columns (produces an entry in both `infinite_index` and `missing_index`) → `outlier_count` is smaller than the actual gap between finite indices and the id space needed, causing internal-node ids in the remapped hierarchy to collide with raw sample ids, silently mislabeling points in `_single_linkage_tree_`. Set `outlier_count = self._raw_data.shape[0] - finite_count` (or equivalently, count the union of infinite_index and missing_index without a set-of-tuples de-dup that under-counts) when remapping.

<details><summary>verbatim finding</summary>

```
### F35 — `remap_single_linkage_tree` `outlier_count` shift under-adjusts internal ids
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:369 sets `outlier_count = len(non_finite)`; sklearn/cluster/_hdbscan/hdbscan.py:377/381 shift internal cluster ids by `outlier_count`. `non_finite` is a `set` produced from `set(infinite_index + missing_index)` (sklearn/cluster/_hdbscan/hdbscan.py:838), so overlapping indices collapse; the correct number of "raw" sample slots that need to be reserved is the total raw sample count minus the finite count, not the size of the deduplicated set.
scenario: "User fits with a row that is *both* `np.inf` and `np.nan` in different columns (produces an entry in both `infinite_index` and `missing_index`) → `outlier_count` is smaller than the actual gap between finite indices and the id space needed, causing internal-node ids in the remapped hierarchy to collide with raw sample ids, silently mislabeling points in `_single_linkage_tree_`."
contract: Set `outlier_count = self._raw_data.shape[0] - finite_count` (or equivalently, count the union of infinite_index and missing_index without a set-of-tuples de-dup that under-counts) when remapping.
instances: single-instance
```
</details>

### [HIGH] `n_jobs` default is `4` while `_parameter_constraints` and docstring both say the default is `None` [out-of-theme]

User relying on the documented `n_jobs=None` (single-thread, joblib backend-controlled) default → HDBSCAN silently uses 4 processes at fit time, producing different resource consumption and, when combined with joblib contexts, unexpected parallelism. The `__init__` default must be `n_jobs=None` to match the documented and constrained default.

<details><summary>verbatim finding</summary>

```
### F75 — `n_jobs` default is `4` while `_parameter_constraints` and docstring both say the default is `None` [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None`; :640 declares the constraint as `"n_jobs": [Integral, None]`; :658 sets the actual constructor default to `n_jobs=4`.
scenario: "User relying on the documented `n_jobs=None` (single-thread, joblib backend-controlled) default → HDBSCAN silently uses 4 processes at fit time, producing different resource consumption and, when combined with joblib contexts, unexpected parallelism."
contract: The `__init__` default must be `n_jobs=None` to match the documented and constrained default.
instances: single-instance
```
</details>

### [HIGH] `store_centers` with non-finite data raises IndexError due to mask/data length mismatch

user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan or np.inf → boolean index of length `n_raw` applied to X of length `n_finite` raises IndexError, `fit` fails. When `store_centers` is set and non-finite data is present, `_weighted_cluster_center` must operate on labels restricted to the finite subset (or X must be re-indexed to raw shape with matching label lengths); the two arrays passed through the mask must have identical length.

<details><summary>verbatim finding</summary>

```
### F77 — `store_centers` with non-finite data raises IndexError due to mask/data length mismatch
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,908-909 — line 733 sets `X = X[finite_index]` (filtered), lines 840-844 remap `self.labels_` to shape `self._raw_data.shape[0]` (full), then line 855 calls `self._weighted_cluster_center(X)` with filtered `X`, and inside line 908 does `mask = self.labels_ == idx` (full length) then `data = X[mask]` (indexing filtered X with full-length mask).
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan or np.inf → boolean index of length `n_raw` applied to X of length `n_finite` raises IndexError, `fit` fails."
contract: When `store_centers` is set and non-finite data is present, `_weighted_cluster_center` must operate on labels restricted to the finite subset (or X must be re-indexed to raw shape with matching label lengths); the two arrays passed through the mask must have identical length.
instances: single-instance
```
</details>

### [HIGH] `_hdbscan_brute` accepts `alpha=None` default but divides by it

any caller not passing `alpha` (only the class currently does at line 767) → `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`. The API contract of the helper is broken and the docstring is a lying default. Change the signature to `alpha=1.0` to match the documented default and the actual class default.

<details><summary>verbatim finding</summary>

```
### F130 — `_hdbscan_brute` accepts `alpha=None` default but divides by it
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None,`; line 241 `distance_matrix /= alpha` will raise `TypeError` when `alpha=None`. Docstring at line 183 lies about the default: "alpha : float, default=1.0".
scenario: "any caller not passing `alpha` (only the class currently does at line 767) → `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`. The API contract of the helper is broken and the docstring is a lying default."
contract: Change the signature to `alpha=1.0` to match the documented default and the actual class default.
instances: single-instance
```
</details>

### [HIGH] `_weighted_cluster_center` computes `n_clusters` ignoring the `-3` (missing) label but not consistently

Fit on data with `np.nan` rows and `store_centers='centroid'` → `self.centroids_` has an extra uninitialized row, silently exposing garbage values as a valid centroid Change to `n_clusters = len(set(self.labels_) - set(out['label'] for out in _OUTLIER_ENCODING.values()) - {-1})` (or equivalently subtract `{-1, -2, -3}`).

<details><summary>verbatim finding</summary>

```
### F131 — `_weighted_cluster_center` computes `n_clusters` ignoring the `-3` (missing) label but not consistently
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. But `_OUTLIER_ENCODING["missing"]["label"] = -3` (line 74) is a documented outlier label that can appear in `self.labels_` (see lines 843, 957). Because `-3` is not subtracted, `n_clusters` is incorrectly inflated by one when any missing samples are present, producing `centroids_`/`medoids_` arrays that have a spurious last row that is never written to (loop `for idx in range(n_clusters)` iterates 0..n_clusters−1, none of which equal `-3`).
scenario: "Fit on data with `np.nan` rows and `store_centers='centroid'` → `self.centroids_` has an extra uninitialized row, silently exposing garbage values as a valid centroid"
contract: Change to `n_clusters = len(set(self.labels_) - set(out['label'] for out in _OUTLIER_ENCODING.values()) - {-1})` (or equivalently subtract `{-1, -2, -3}`).
instances: single-instance
```
</details>

### [HIGH] `n_jobs` default hard-codes 4 workers, contradicting docstring

user constructs `HDBSCAN()` expecting sklearn convention (`n_jobs=None`) → estimator silently spawns 4 workers regardless of the surrounding `joblib.parallel_backend`; docstring is a lying default and the class violates sklearn's `n_jobs` convention. Change the signature default to `n_jobs=None` to match the documented behavior and sklearn convention.

<details><summary>verbatim finding</summary>

```
### F132 — `n_jobs` default hard-codes 4 workers, contradicting docstring
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,`; docstring at lines 486-490 says `n_jobs : int, default=None` and "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors."
scenario: "user constructs `HDBSCAN()` expecting sklearn convention (`n_jobs=None`) → estimator silently spawns 4 workers regardless of the surrounding `joblib.parallel_backend`; docstring is a lying default and the class violates sklearn's `n_jobs` convention."
contract: Change the signature default to `n_jobs=None` to match the documented behavior and sklearn convention.
instances: single-instance
```
</details>

### [HIGH] HDBSCAN `n_jobs` docstring contradicts the actual default (`None` vs `4`)

User reads docstring expecting single-thread default → HDBSCAN silently spawns 4 workers, oversubscribing CPUs in shared or nested-parallel contexts The `n_jobs` docstring in the `HDBSCAN` class must state `default=4` and remove the "None means 1" boilerplate (or the constructor default must be changed to `None`); one of the two must move so signature and prose agree.

<details><summary>verbatim finding</summary>

```
### F166 — HDBSCAN `n_jobs` docstring contradicts the actual default (`None` vs `4`)
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring reads `n_jobs : int, default=None` and states "`None` means 1 unless in a joblib.parallel_backend context", but the constructor signature at line 658 sets `n_jobs=4`. Users following the documented default will get a completely different concurrency setting than the code enforces.
scenario: "User reads docstring expecting single-thread default → HDBSCAN silently spawns 4 workers, oversubscribing CPUs in shared or nested-parallel contexts"
contract: The `n_jobs` docstring in the `HDBSCAN` class must state `default=4` and remove the "None means 1" boilerplate (or the constructor default must be changed to `None`); one of the two must move so signature and prose agree.
instances: single-instance
```
</details>

### [HIGH] `_weighted_cluster_center` excludes only `{-1, -2}` while docstring/comment intent is "non-outlier / non-noise"

User calls `fit` with data containing missing rows and `store_centers` set → `-3` label leaks into the set, `n_clusters` is off by one, and the pre-allocated `centroids_`/`medoids_` arrays are one row too large (the last row is never written and stays as uninitialized `np.empty` values) [out-of-theme] The set must be `{-1, -2, -3}` (or better, derived from `_OUTLIER_ENCODING`), matching the documented invariant.

<details><summary>verbatim finding</summary>

```
### F167 — `_weighted_cluster_center` excludes only `{-1, -2}` while docstring/comment intent is "non-outlier / non-noise"
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:894-895 — `# Number of non-noise clusters` immediately followed by `n_clusters = len(set(self.labels_) - {-1, -2})`; the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:561-562 explicitly says `n_clusters` "only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded"; the missing-data label -3 is defined in sklearn/cluster/_hdbscan/hdbscan.py:73-78
scenario: "User calls `fit` with data containing missing rows and `store_centers` set → `-3` label leaks into the set, `n_clusters` is off by one, and the pre-allocated `centroids_`/`medoids_` arrays are one row too large (the last row is never written and stays as uninitialized `np.empty` values) [out-of-theme]"
contract: The set must be `{-1, -2, -3}` (or better, derived from `_OUTLIER_ENCODING`), matching the documented invariant.
instances: single-instance
```
</details>

### [HIGH] HDBSCAN-specific `max_distance` smuggled through generic `metric_params` leaks into `pairwise_distances`, `NearestNeighbors`, and `DistanceMetric.get_metric`

User calls `HDBSCAN(metric='euclidean', metric_params={'max_distance': 1.0}).fit(X)` on a raw feature matrix → `pairwise_distances(..., max_distance=1.0)` raises `TypeError` because `max_distance` is not a valid pairwise-metric kwarg; alternatively `DistanceMetric.get_metric('euclidean', max_distance=1.0)` in the prims path rejects the kwarg. The HDBSCAN-specific `max_distance` option must be a distinct estimator parameter (or popped from a copied dict before forwarding) rather than smuggled through the neighbor/metric layer's generic `metric_params`.

<details><summary>verbatim finding</summary>

```
### F219 — HDBSCAN-specific `max_distance` smuggled through generic `metric_params` leaks into `pairwise_distances`, `NearestNeighbors`, and `DistanceMetric.get_metric`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:238-243 — `_hdbscan_brute` calls `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` BEFORE fetching `max_distance = metric_params.get("max_distance", 0.0)` at line 243 without popping the key. sklearn/cluster/_hdbscan/hdbscan.py:337 forwards the same dict verbatim as `NearestNeighbors(..., metric_params=metric_params, ...)`, and line 344 does `DistanceMetric.get_metric(metric, **metric_params)`. The error message at hdbscan.py:121 explicitly instructs users to "specify a `max_distance` in `metric_params`".
scenario: "User calls `HDBSCAN(metric='euclidean', metric_params={'max_distance': 1.0}).fit(X)` on a raw feature matrix → `pairwise_distances(..., max_distance=1.0)` raises `TypeError` because `max_distance` is not a valid pairwise-metric kwarg; alternatively `DistanceMetric.get_metric('euclidean', max_distance=1.0)` in the prims path rejects the kwarg."
contract: The HDBSCAN-specific `max_distance` option must be a distinct estimator parameter (or popped from a copied dict before forwarding) rather than smuggled through the neighbor/metric layer's generic `metric_params`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:238-243, sklearn/cluster/_hdbscan/hdbscan.py:337, sklearn/cluster/_hdbscan/hdbscan.py:344, sklearn/cluster/_hdbscan/hdbscan.py:770]
```
</details>

### [HIGH] HDBSCAN `n_jobs` default is `4`, contradicting docstring and scikit-learn convention

User relies on documented default `None` behaviour (single-threaded unless in a `joblib.parallel_backend` context) → HDBSCAN silently oversubscribes 4 workers, breaking sandboxed/CI environments, container CPU quotas and reproducibility, and inflating memory pressure for `pairwise_distances` Set `n_jobs=None` in `HDBSCAN.__init__` so that the runtime default matches the documented default and the scikit-learn-wide convention.

<details><summary>verbatim finding</summary>

```
### F234 — HDBSCAN `n_jobs` default is `4`, contradicting docstring and scikit-learn convention
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in `__init__`, but the docstring at sklearn/cluster/_hdbscan/hdbscan.py:486 states `n_jobs : int, default=None`; all sibling estimators (DBSCAN at sklearn/cluster/_dbscan.py:96 and OPTICS at sklearn/cluster/_optics.py:157) use `default=None`
scenario: "User relies on documented default `None` behaviour (single-threaded unless in a `joblib.parallel_backend` context) → HDBSCAN silently oversubscribes 4 workers, breaking sandboxed/CI environments, container CPU quotas and reproducibility, and inflating memory pressure for `pairwise_distances`"
contract: Set `n_jobs=None` in `HDBSCAN.__init__` so that the runtime default matches the documented default and the scikit-learn-wide convention.
instances: single-instance
```
</details>

### [HIGH] `UnionFind.union`/`fast_find` `noexcept` declared in .pxd but not in .pyx implementation

Cython 3+ compilation of the module → signature-mismatch error or, in permissive versions, an unintended exception-propagation contract that silently diverges from the header's stated no-exception guarantee, breaking `nogil` callers. Declarations in `_hierarchical_fast.pxd` and implementations in `_hierarchical_fast.pyx` must carry the identical `noexcept` annotation for both `union` and `fast_find`.

<details><summary>verbatim finding</summary>

```
### F76 — `UnionFind.union`/`fast_find` `noexcept` declared in .pxd but not in .pyx implementation
severity: high
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declare `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 and :339 implement them without `noexcept` (`cdef void union(self, intp_t m, intp_t n):` and `cdef intp_t fast_find(self, intp_t n):`).
scenario: "Cython 3+ compilation of the module → signature-mismatch error or, in permissive versions, an unintended exception-propagation contract that silently diverges from the header's stated no-exception guarantee, breaking `nogil` callers."
contract: Declarations in `_hierarchical_fast.pxd` and implementations in `_hierarchical_fast.pyx` must carry the identical `noexcept` annotation for both `union` and `fast_find`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
```
</details>

### [HIGH] `test_hdbscan_precomputed_non_brute` uses invalid algorithm strings, passes for the wrong reason

user passes metric='precomputed' with algorithm='kdtree'/'balltree' → test claims coverage but code path is unexercised; real behavior is untested (silent misroute to `_hdbscan_prims`) The test must use the actually-supported strings (`algorithm="kdtree"`/`"balltree"`) and match against the specific error message the codebase intends to raise; if no such rejection exists in the estimator, the missing validation must be added and then covered by a matched-message test.

<details><summary>verbatim finding</summary>

```
### F54 — `test_hdbscan_precomputed_non_brute` uses invalid algorithm strings, passes for the wrong reason
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — the test constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` where the algorithm string is `"prims_kdtree"`/`"prims_balltree"`, but the estimator's `_parameter_constraints` (hdbscan.py:629-638) only accepts `{"auto", "brute", "kdtree", "balltree"}`. The `ValueError` raised is `InvalidParameterError` (from `_validate_params`, which extends `ValueError`), not the intended "precomputed data incompatible with tree" error. In fact, hdbscan.py:772-818 contains no code path that rejects `metric="precomputed"` when `algorithm="kdtree"`/`"balltree"`; the branch at line 785 silently dispatches to `_hdbscan_prims`.
scenario: "user passes metric='precomputed' with algorithm='kdtree'/'balltree' → test claims coverage but code path is unexercised; real behavior is untested (silent misroute to `_hdbscan_prims`)"
contract: The test must use the actually-supported strings (`algorithm="kdtree"`/`"balltree"`) and match against the specific error message the codebase intends to raise; if no such rejection exists in the estimator, the missing validation must be added and then covered by a matched-message test.
instances: single-instance
```
</details>

### [HIGH] `test_hdbscan_precomputed_non_brute` exercises invalid algorithm names, not the intended path [out-of-theme]

Someone regresses the precomputed+kdtree / precomputed+balltree guard in HDBSCAN.fit → this test still passes because any invalid string will raise ValueError during `_validate_params`; the regression ships Replace the algorithm name with a valid value (`"kdtree"` / `"balltree"`) so the test drives the actual code path the docstring claims to cover, and match the specific error message from the fit-time guard.

<details><summary>verbatim finding</summary>

```
### F117 — `test_hdbscan_precomputed_non_brute` exercises invalid algorithm names, not the intended path [out-of-theme]
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` constructs `"prims_kdtree"` / `"prims_balltree"`, but the estimator's `_parameter_constraints["algorithm"]` in sklearn/cluster/_hdbscan/hdbscan.py:635-644 is `StrOptions({"auto", "brute", "kdtree", "balltree"})`. The test asserts `pytest.raises(ValueError)`, so it passes — but on the parameter-validation error, never reaching the precomputed+tree combination the test docstring claims to guard.
scenario: "Someone regresses the precomputed+kdtree / precomputed+balltree guard in HDBSCAN.fit → this test still passes because any invalid string will raise ValueError during `_validate_params`; the regression ships"
contract: Replace the algorithm name with a valid value (`"kdtree"` / `"balltree"`) so the test drives the actual code path the docstring claims to cover, and match the specific error message from the fit-time guard.
instances: single-instance
```
</details>

### [HIGH] `test_dbscan_clustering_outlier_data` uses numpy `+` where set-union of indices is intended [out-of-theme]

The test runs with these fixed arrays → index 0 (the sample set to `[np.inf, 1]`) remains in `clean_idx`, so `X_outlier[clean_idx]` still contains a non-finite row and `clean_model.fit` follows a different code path than intended; the assertion may still pass by coincidence but the test does not verify what it claims Concatenate the indices with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or convert each to a list before `+`), then build the set of exclusions.

<details><summary>verbatim finding</summary>

```
### F118 — `test_dbscan_clustering_outlier_data` uses numpy `+` where set-union of indices is intended [out-of-theme]
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` is a `np.ndarray` (e.g. `[2, 5]` from `np.flatnonzero`) and `infinite_labels_idx` is an ndarray (e.g. `[0]`). `+` on two ndarrays broadcasts elementwise (giving `[2, 5]` here), it does not concatenate, so index `0` is silently dropped from the exclusion set.
scenario: "The test runs with these fixed arrays → index 0 (the sample set to `[np.inf, 1]`) remains in `clean_idx`, so `X_outlier[clean_idx]` still contains a non-finite row and `clean_model.fit` follows a different code path than intended; the assertion may still pass by coincidence but the test does not verify what it claims"
contract: Concatenate the indices with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or convert each to a list before `+`), then build the set of exclusions.
instances: single-instance
```
</details>

## MEDIUM severity (81)

### [MEDIUM] Scale-invariance loop never applies the scale [out-of-theme]

user reads the HDBSCAN scale-invariance demo → all three subplots show identical clustering of un-scaled `X` (misleadingly implying scale invariance from a degenerate experiment), instead of the intended comparison of clustering the scaled data. Fit and plot on `X * scale` (mirroring the DBSCAN block): `hdb.fit(X * scale); plot(X * scale, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`.

<details><summary>verbatim finding</summary>

```
### F134 — Scale-invariance loop never applies the scale [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ...)`; the DBSCAN analogue immediately above at lines 87-89 correctly uses `dbs.fit(X * scale)` and `plot(X * scale, ...)`.
scenario: "user reads the HDBSCAN scale-invariance demo → all three subplots show identical clustering of un-scaled `X` (misleadingly implying scale invariance from a degenerate experiment), instead of the intended comparison of clustering the scaled data."
contract: Fit and plot on `X * scale` (mirroring the DBSCAN block): `hdb.fit(X * scale); plot(X * scale, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`.
instances: single-instance
```
</details>

### [MEDIUM] Docstring anchor mismatch: `User Guide <HDBSCAN>` reference does not exist

Sphinx build of gallery example → broken/undefined :ref: link labeled 'User Guide' either warns or renders as raw text for readers Change the reference target to lowercase `hdbscan` to match the RST anchor.

<details><summary>verbatim finding</summary>

```
### F171 — Docstring anchor mismatch: `User Guide <HDBSCAN>` reference does not exist
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104 — `see :ref:\`User Guide <HDBSCAN>\``. The actual anchor label in doc/modules/clustering.rst is `.. _hdbscan:` (lowercase), so this cross-reference fails to resolve in the rendered docs.
scenario: "Sphinx build of gallery example → broken/undefined :ref: link labeled 'User Guide' either warns or renders as raw text for readers"
contract: Change the reference target to lowercase `hdbscan` to match the RST anchor.
instances: single-instance
```
</details>

### [MEDIUM] Parallel `MST_edge_dtype` / `MST_edge_t` and `HIERARCHY_dtype` / `HIERARCHY_t` / `CONDENSED_dtype` / `CONDENSED_t` dual definitions

A future edit widens `left_node` from `intp` to `int64` in the numpy `HIERARCHY_dtype` but forgets to update the `HIERARCHY_t` struct in `_tree.pxd` (or vice versa) → Cython memory-view accesses into the array read/write the wrong bytes for every element, producing silently corrupted trees rather than a type error. Move both the numpy dtype and the Cython struct for each pair to a single canonical `.pxd` with a runtime `sizeof(MST_edge_t) == MST_edge_dtype.itemsize` assertion at module import so the two representations cannot drift.

<details><summary>verbatim finding</summary>

```
### F101 — Parallel `MST_edge_dtype` / `MST_edge_t` and `HIERARCHY_dtype` / `HIERARCHY_t` / `CONDENSED_dtype` / `CONDENSED_t` dual definitions
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:48-58 defines `MST_edge_dtype` (numpy structured dtype with `np.int64, np.int64, np.float64`) and separately `MST_edge_t` (Cython packed struct with `int64_t, int64_t, float64_t`). Same pattern for `HIERARCHY_dtype` (sklearn/cluster/_hdbscan/_tree.pyx:42-47, fields `intp, intp, float64, intp`) vs `HIERARCHY_t` (sklearn/cluster/_hdbscan/_tree.pxd:34-38, fields `intp_t, intp_t, float64_t, intp_t`), and `CONDENSED_dtype` (sklearn/cluster/_hdbscan/_tree.pyx:49-54) vs `CONDENSED_t` (sklearn/cluster/_hdbscan/_tree.pxd:42-46). Each pair encodes the exact same record layout in two independent locations that must be manually kept in sync — nothing in the code checks (e.g. via `sizeof` asserts) that the Cython struct matches the numpy dtype.
scenario: "A future edit widens `left_node` from `intp` to `int64` in the numpy `HIERARCHY_dtype` but forgets to update the `HIERARCHY_t` struct in `_tree.pxd` (or vice versa) → Cython memory-view accesses into the array read/write the wrong bytes for every element, producing silently corrupted trees rather than a type error."
contract: Move both the numpy dtype and the Cython struct for each pair to a single canonical `.pxd` with a runtime `sizeof(MST_edge_t) == MST_edge_dtype.itemsize` assertion at module import so the two representations cannot drift.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:48, sklearn/cluster/_hdbscan/_linkage.pyx:56, sklearn/cluster/_hdbscan/_tree.pyx:42, sklearn/cluster/_hdbscan/_tree.pyx:42-47, sklearn/cluster/_hdbscan/_tree.pyx:49, sklearn/cluster/_hdbscan/_tree.pyx:49-54, sklearn/cluster/_hdbscan/_tree.pxd:34, sklearn/cluster/_hdbscan/_tree.pxd:34-38, sklearn/cluster/_hdbscan/_tree.pxd:42, sklearn/cluster/_hdbscan/_tree.pxd:42-46]
```
</details>

### [MEDIUM] `mst_from_mutual_reachability` allocates and reslices arrays every iteration → O(n²) allocations

Dense brute MST on n samples → ~n heap allocations of arrays sized O(n), plus a full O(n) copy of mutual_reachability's row through fancy indexing (rather than a contiguous row view), inflating memory bandwidth and GC/allocator pressure quadratically instead of a nogil in-place scan. Perform Prim's expansion in-place with a Cython typed-memoryview scan that indexes an `in_tree` mask (see `mst_from_data_matrix` for the pattern) and updates `min_reachability`/`current_sources` without reallocating per iteration.

<details><summary>verbatim finding</summary>

```
### F199 — `mst_from_mutual_reachability` allocates and reslices arrays every iteration → O(n²) allocations
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:94-106 — inside the Prim's main loop, `label_filter = current_labels != current_node`, `current_labels = current_labels[label_filter]`, `left = min_reachability[label_filter]`, `right = mutual_reachability[current_node][current_labels]`, and `min_reachability = np.minimum(left, right)` are all re-executed via NumPy fancy indexing, allocating fresh arrays on every iteration.
scenario: "Dense brute MST on n samples → ~n heap allocations of arrays sized O(n), plus a full O(n) copy of mutual_reachability's row through fancy indexing (rather than a contiguous row view), inflating memory bandwidth and GC/allocator pressure quadratically instead of a nogil in-place scan."
contract: Perform Prim's expansion in-place with a Cython typed-memoryview scan that indexes an `in_tree` mask (see `mst_from_data_matrix` for the pattern) and updates `min_reachability`/`current_sources` without reallocating per iteration.
instances: single-instance
```
</details>

### [MEDIUM] Sparse mutual-reachability leaves stale value in place when result is infinite and `max_distance <= 0` [out-of-theme]

User supplies a sparse precomputed distance matrix with a row that has fewer than `min_samples` stored neighbors (so `core_distances[row_ind] = INFINITY`), and does not specify `max_distance` → mutual reachability along that row is infinite but `data[i]` stays at the original finite distance → `_brute_mst` produces an MST that silently uses those stale distances instead of raising, and the 'contains edge weights with value infinity' warning at hdbscan.py:256 does not fire because the infinity was suppressed. When `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, `data[i]` must be set to `INFINITY` (or the entry otherwise flagged) so that downstream MST / connectivity checks see the true value; silently retaining the pre-existing entry hides a data condition that the code explicitly warns about elsewhere.

<details><summary>verbatim finding</summary>

```
### F38 — Sparse mutual-reachability leaves stale value in place when result is infinite and `max_distance <= 0` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:206-212 — after computing `mutual_reachibility_distance = max(core_distances[row_ind], core_distances[col_ind], data[i])`, the code writes back only when the result is finite, or when `max_distance > 0`. When the mutual reachability is infinite and `max_distance == 0` (the default at hdbscan.py:243), `data[i]` retains its original value (not the newly-computed infinite one). Downstream `_brute_mst` then runs `csgraph.connected_components` and `minimum_spanning_tree` on a graph whose weights do NOT reflect the mutual-reachability semantics for the affected edges, producing an MST whose weights understate the true infinite mutual reachability.
scenario: "User supplies a sparse precomputed distance matrix with a row that has fewer than `min_samples` stored neighbors (so `core_distances[row_ind] = INFINITY`), and does not specify `max_distance` → mutual reachability along that row is infinite but `data[i]` stays at the original finite distance → `_brute_mst` produces an MST that silently uses those stale distances instead of raising, and the 'contains edge weights with value infinity' warning at hdbscan.py:256 does not fire because the infinity was suppressed."
contract: When `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, `data[i]` must be set to `INFINITY` (or the entry otherwise flagged) so that downstream MST / connectivity checks see the true value; silently retaining the pre-existing entry hides a data condition that the code explicitly warns about elsewhere.
instances: single-instance
```
</details>

### [MEDIUM] `max_distance` parameter of `mutual_reachability_graph` is untested

regression that stops replacing infinite entries when `max_distance > 0` → silently returns infinite mutual-reachability distances, corrupting MST construction; no test catches this Add a sparse-input test with infinite-would-be mutual-reachability values that asserts they are replaced by `max_distance` when it is set (and left as `INFINITY` when it is 0).

<details><summary>verbatim finding</summary>

```
### F58 — `max_distance` parameter of `mutual_reachability_graph` is untested
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:44-46, 209-212 — `max_distance` gates the replacement value for infinite mutual-reachability distances in the sparse path: `elif max_distance > 0: data[i] = max_distance`. sklearn/cluster/_hdbscan/tests/test_reachibility.py has no test that exercises non-zero `max_distance` or that verifies infinite entries are replaced correctly. The plan explicitly relies on this behavior via `_hdbscan_brute` (hdbscan.py:243 `max_distance = metric_params.get("max_distance", 0.0)`).
scenario: "regression that stops replacing infinite entries when `max_distance > 0` → silently returns infinite mutual-reachability distances, corrupting MST construction; no test catches this"
contract: Add a sparse-input test with infinite-would-be mutual-reachability values that asserts they are replaced by `max_distance` when it is set (and left as `INFINITY` when it is 0).
instances: single-instance
```
</details>

### [MEDIUM] `_sparse_mutual_reachability_graph` docstring documents a nonexistent `distance_matrix` parameter and omits real ones

Reader/maintainer must reverse-engineer meaning of positional args from callers rather than the docstring → mis-uses the function or introduces bugs when refactoring call sites Rewrite the Parameters block to reflect the real CSR-split arguments (`data`, `indices`, `indptr`, `n_samples`) and drop `distance_matrix`.

<details><summary>verbatim finding</summary>

```
### F173 — `_sparse_mutual_reachability_graph` docstring documents a nonexistent `distance_matrix` parameter and omits real ones
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:152-179 — signature is `(data, indices, indptr, n_samples, further_neighbor_idx, max_distance)`; docstring only documents `distance_matrix`, `further_neighbor_idx`, `max_distance`, none of which is the actual `data`/`indices`/`indptr`/`n_samples` triple; the phrase "the sparse format should be `CSR`" describes an argument that is not passed
scenario: "Reader/maintainer must reverse-engineer meaning of positional args from callers rather than the docstring → mis-uses the function or introduces bugs when refactoring call sites"
contract: Rewrite the Parameters block to reflect the real CSR-split arguments (`data`, `indices`, `indptr`, `n_samples`) and drop `distance_matrix`.
instances: single-instance
```
</details>

### [MEDIUM] Dense mutual reachability graph iterates full n×n instead of exploiting symmetry

user calls HDBSCAN with `algorithm='brute'` / precomputed dense distance matrix → O(n²) cells are visited twice, doubling the hot-path cost and doubling memory-traffic vs. iterating only `j >= i` and mirroring. Iterate the upper triangle only (`for j in range(i, n_samples)`) and mirror the assignment to `distance_matrix[j, i]`; the loop must not do redundant work on a matrix its own docstring declares symmetric.

<details><summary>verbatim finding</summary>

```
### F208 — Dense mutual reachability graph iterates full n×n instead of exploiting symmetry
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:142-149 — the nested loop is `for i in range(n_samples): for j in range(n_samples): ... distance_matrix[i, j] = mutual_reachibility_distance`. The docstring at line 130 states "We assume that the distance matrix is symmetric." Both `[i,j]` and `[j,i]` receive the same computed `max(core[i], core[j], d[i,j])`, so every cell's value is computed twice.
scenario: "user calls HDBSCAN with `algorithm='brute'` / precomputed dense distance matrix → O(n²) cells are visited twice, doubling the hot-path cost and doubling memory-traffic vs. iterating only `j >= i` and mirroring."
contract: Iterate the upper triangle only (`for j in range(i, n_samples)`) and mirror the assignment to `distance_matrix[j, i]`; the loop must not do redundant work on a matrix its own docstring declares symmetric.
instances: single-instance
```
</details>

### [MEDIUM] `HIERARCHY_t.left_node`/`right_node` are `intp_t` but MST edges are `int64_t`, narrowing on 32-bit

Build on a 32-bit platform where `intp_t == int32_t` and `n_samples > 2**31` → silent truncation when storing MST node indices into the HIERARCHY struct. `HIERARCHY_t.left_node` and `HIERARCHY_t.right_node` must be widened to `int64_t` (with `HIERARCHY_dtype` updated to `np.int64` correspondingly) to match `MST_edge_t.current_node`/`next_node`.

<details><summary>verbatim finding</summary>

```
### F83 — `HIERARCHY_t.left_node`/`right_node` are `intp_t` but MST edges are `int64_t`, narrowing on 32-bit
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pxd:34-38 declares `HIERARCHY_t` with `intp_t left_node`/`right_node`; sklearn/cluster/_hdbscan/_linkage.pyx:56-59 declares `MST_edge_t` with `int64_t current_node`/`next_node`; :263-264 assigns `single_linkage[i].left_node = current_node_cluster` and `.right_node = next_node_cluster` where the cluster IDs came from `U.fast_find(current_node)` on a `int64_t current_node`.
scenario: "Build on a 32-bit platform where `intp_t == int32_t` and `n_samples > 2**31` → silent truncation when storing MST node indices into the HIERARCHY struct."
contract: `HIERARCHY_t.left_node` and `HIERARCHY_t.right_node` must be widened to `int64_t` (with `HIERARCHY_dtype` updated to `np.int64` correspondingly) to match `MST_edge_t.current_node`/`next_node`.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:34-38, sklearn/cluster/_hdbscan/_linkage.pyx:56-59, sklearn/cluster/_hdbscan/_linkage.pyx:263-264]
```
</details>

### [MEDIUM] Novel change #1 ("Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`") only partially satisfied; `_tree.pyx` still contains 107 `cnp.` occurrences

The plan asserts a completed refactor of `cnp.*_t` → typedef `*_t` typing across the added Cython sources. The refactor is present in `_linkage.pyx` and `_reachability.pyx` (typedef imports added), but `_tree.pyx` — the largest Cython file (+797 lines) — was added without importing from `_typedefs.pxd` and continues to reference `cnp.*` extensively. The residual `cnp.` names may be `cnp.ndarray` (legitimately still needed) rather than `cnp.*_t` typedefs, but the audit cannot confirm that from the manifest alone without the reader inspecting the file. A 'replaced' claim in the plan should hold uniformly across all added Cython files. Uneven application weakens the inverse-conformance mapping and may indicate the claim describes a partial migration.

<details><summary>verbatim finding</summary>

```
### F2 — Novel change #1 ("Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`") only partially satisfied; `_tree.pyx` still contains 107 `cnp.` occurrences
severity: medium
scenario: "The plan asserts a completed refactor of `cnp.*_t` → typedef `*_t` typing across the added Cython sources. The refactor is present in `_linkage.pyx` and `_reachability.pyx` (typedef imports added), but `_tree.pyx` — the largest Cython file (+797 lines) — was added without importing from `_typedefs.pxd` and continues to reference `cnp.*` extensively. The residual `cnp.` names may be `cnp.ndarray` (legitimately still needed) rather than `cnp.*_t` typedefs, but the audit cannot confirm that from the manifest alone without the reader inspecting the file."
contract: "A 'replaced' claim in the plan should hold uniformly across all added Cython files. Uneven application weakens the inverse-conformance mapping and may indicate the claim describes a partial migration."
evidence:
- Plan promise: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`" (PLAN.md:15).
- Verification of the cimport landing:
  - `hunks/sklearn_cluster__hdbscan__linkage.pyx.diff:48` — `+from ...utils._typedefs cimport intp_t, float64_t, int64_t, uint8_t`
  - `hunks/sklearn_cluster__hdbscan__reachability.pyx.diff:46` — `+from ...utils._typedefs cimport intp_t`
- Residual `cnp.` occurrences (post-change) counted in the hunk diffs:
  - `_linkage.pyx.diff`: 12
  - `_reachability.pyx.diff`: 4
  - `_tree.pyx.diff`: 107
  - `_tree.pxd.diff`: 1
- `sklearn/cluster/_hdbscan/_tree.pyx` (newly added, +797 lines per DIFF_MANIFEST.md:23) contains no `_typedefs` cimport (grep for `_typedefs` in that hunk returns no match).

instances: [sklearn/cluster/_hdbscan/_tree.pyx:1, sklearn/cluster/_hdbscan/_tree.pxd:1]
```
</details>

### [MEDIUM] `_do_labelling` reads `parent_lambda` and `threshold` per-sample instead of comparing scalars, causing potential wrong labels for `allow_single_cluster` epsilon path

User calls `HDBSCAN(allow_single_cluster=True, cluster_selection_epsilon=0.0).fit(X)` on data that produces exactly one cluster where every child of the root is a leaf-cluster → `_do_labelling` raises `ValueError: zero-size array` when computing per-sample thresholds The threshold and per-sample lambda arithmetic must be scalar-safe; if `parent_array == cluster` is empty, use the max child lambda over `condensed_tree` for that parent, and coerce `parent_lambda` to a scalar with an explicit `[0]` index.

<details><summary>verbatim finding</summary>

```
### F4 — `_do_labelling` reads `parent_lambda` and `threshold` per-sample instead of comparing scalars, causing potential wrong labels for `allow_single_cluster` epsilon path
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498-506 — `parent_lambda = lambda_array[child_array == n]` is an ndarray (length 1 in the intended case), and `threshold = lambda_array[parent_array == cluster].max()` uses `cluster` which is the union-find root that at this branch equals `root_cluster` (not a leaf/point), so `parent_array == cluster` may match zero entries because in `allow_single_cluster` mode all non-cluster edges were unioned; the min-parent root itself may have no child edges left after masking, producing `max()` on an empty array which raises `ValueError` at runtime.
scenario: "User calls `HDBSCAN(allow_single_cluster=True, cluster_selection_epsilon=0.0).fit(X)` on data that produces exactly one cluster where every child of the root is a leaf-cluster → `_do_labelling` raises `ValueError: zero-size array` when computing per-sample thresholds"
contract: The threshold and per-sample lambda arithmetic must be scalar-safe; if `parent_array == cluster` is empty, use the max child lambda over `condensed_tree` for that parent, and coerce `parent_lambda` to a scalar with an explicit `[0]` index.
instances: single-instance
```
</details>

### [MEDIUM] `cluster_selection_method="leaf"` is entirely untested

Regression in the leaf-selection branch (e.g., breaking `get_cluster_tree_leaves` or the empty-leaves fallback) → no test fails. Add at least one test that runs `HDBSCAN(cluster_selection_method="leaf")` on a dataset with a known hierarchical structure and asserts the returned leaf clusters against a reference.

<details><summary>verbatim finding</summary>

```
### F60 — `cluster_selection_method="leaf"` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — a grep for `cluster_selection_method` returns only lines 340 and 354, both passing `"eom"`. The `_get_clusters` branch at sklearn/cluster/_hdbscan/_tree.pyx:761-782 implementing the `"leaf"` method (including the empty-leaves fallback, `get_cluster_tree_leaves`, `recurse_leaf_dfs`, and the leaf + epsilon combination via `epsilon_search`) is added new code with no test coverage.
scenario: "Regression in the leaf-selection branch (e.g., breaking `get_cluster_tree_leaves` or the empty-leaves fallback) → no test fails."
contract: Add at least one test that runs `HDBSCAN(cluster_selection_method="leaf")` on a dataset with a known hierarchical structure and asserts the returned leaf clusters against a reference.
instances: single-instance
```
</details>

### [MEDIUM] `_do_labelling` scalar-vs-array contract: `parent_lambda` compared as a scalar

A malformed condensed tree (test fixture, buggy caller, or duplicate child-row) in which more than one row has `child_array == n` → `if parent_lambda >= threshold` raises an ambiguous-truth-value ValueError inside cluster labelling Extract a scalar explicitly (e.g. `cdef cnp.float64_t parent_lambda = lambda_array[child_array == n][0]`) and rely on the invariant, not on an implicit 1-element-array coercion.

<details><summary>verbatim finding</summary>

```
### F80 — `_do_labelling` scalar-vs-array contract: `parent_lambda` compared as a scalar
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 assigns `parent_lambda = lambda_array[child_array == n]` — an untyped Python object holding a 1-D `float64` ndarray. Line 505 does `if parent_lambda >= threshold:`, which relies on the array having exactly one element (otherwise NumPy raises `ValueError: The truth value of an array with more than one element is ambiguous`). The comment at line 496 asserts this without any runtime type/shape guard, and `parent_lambda` is not `cdef`-typed as a scalar.
scenario: "A malformed condensed tree (test fixture, buggy caller, or duplicate child-row) in which more than one row has `child_array == n` → `if parent_lambda >= threshold` raises an ambiguous-truth-value ValueError inside cluster labelling"
contract: Extract a scalar explicitly (e.g. `cdef cnp.float64_t parent_lambda = lambda_array[child_array == n][0]`) and rely on the invariant, not on an implicit 1-element-array coercion.
instances: single-instance
```
</details>

### [MEDIUM] `allow_single_cluster` parameter drifts across three C types on the same call chain

A future change that stores non-{0,1} in this argument → silent truthiness drift (e.g. -1 remains truthy through intp_t but wraps under uint8_t to 255, still truthy; but any refactor to an enum or tri-state will diverge silently across layers). A single canonical type — `bint` — must be used for `allow_single_cluster` at every `cdef` boundary in `_tree.pyx`.

<details><summary>verbatim finding</summary>

```
### F82 — `allow_single_cluster` parameter drifts across three C types on the same call chain
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:60 declares `bint allow_single_cluster=False` in `tree_to_labels`; :435 declares `cnp.intp_t allow_single_cluster` in `_do_labelling`; :580 and :608 declare `cnp.intp_t allow_single_cluster` in `traverse_upwards`/`epsilon_search`; :646 declares `cnp.uint8_t allow_single_cluster=False` in `_get_clusters`. All are called with the same Python boolean and represent the same conceptual flag.
scenario: "A future change that stores non-{0,1} in this argument → silent truthiness drift (e.g. -1 remains truthy through intp_t but wraps under uint8_t to 255, still truthy; but any refactor to an enum or tri-state will diverge silently across layers)."
contract: A single canonical type — `bint` — must be used for `allow_single_cluster` at every `cdef` boundary in `_tree.pyx`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:60, sklearn/cluster/_hdbscan/_tree.pyx:435, sklearn/cluster/_hdbscan/_tree.pyx:580, sklearn/cluster/_hdbscan/_tree.pyx:608, sklearn/cluster/_hdbscan/_tree.pyx:646]
```
</details>

### [MEDIUM] `traverse_upwards` assigns numpy arrays to C scalar–typed locals `parent` and `parent_eps`

A cluster tree in which `cluster_tree['child'] == leaf` selects zero rows (leaf not present) → coercion raises `TypeError: only size-1 arrays can be converted to Python scalars` instead of a clear domain error; more than one row → same failure. Additionally, `parent` is then passed recursively as a `cnp.intp_t` argument, silently discarding array shape. The mask must be reduced to a scalar explicitly (e.g. `int(arr[0])` / `float(arr[0])`) with a guard that exactly one row was selected, before assignment to the C scalar locals.

<details><summary>verbatim finding</summary>

```
### F85 — `traverse_upwards` assigns numpy arrays to C scalar–typed locals `parent` and `parent_eps`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:582 declares `cdef cnp.intp_t root, parent` and `cdef cnp.float64_t parent_eps`; :586 assigns `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` (a 1-D ndarray); :593 assigns `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']` (a 1-D ndarray). Both rely on numpy's implicit length-1-array-to-scalar coercion.
scenario: "A cluster tree in which `cluster_tree['child'] == leaf` selects zero rows (leaf not present) → coercion raises `TypeError: only size-1 arrays can be converted to Python scalars` instead of a clear domain error; more than one row → same failure. Additionally, `parent` is then passed recursively as a `cnp.intp_t` argument, silently discarding array shape."
contract: The mask must be reduced to a scalar explicitly (e.g. `int(arr[0])` / `float(arr[0])`) with a guard that exactly one row was selected, before assignment to the C scalar locals.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593]
```
</details>

### [MEDIUM] `_get_clusters.child_selection` typed as `uint8_t[::1]` receives an object-dtype ndarray from `==`

Cython buffer acquisition on newer numpy/Cython that strictly checks buffer format `?` vs `B` → `ValueError: Buffer dtype mismatch, expected 'unsigned char' but got 'bool'` at runtime on every `fit` that hits the `eom` branch. `child_selection` must be typed as a boolean-compatible memoryview (e.g. `cnp.uint8_t[::1]` assigned from `(cluster_tree['parent'] == node).view(np.uint8)`) or declared as `cnp.npy_bool[::1]`, with the assignment made explicit.

<details><summary>verbatim finding</summary>

```
### F86 — `_get_clusters.child_selection` typed as `uint8_t[::1]` receives an object-dtype ndarray from `==`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:695 declares `cnp.uint8_t[::1] child_selection`; :729 assigns `child_selection = (cluster_tree['parent'] == node)` — the result of `==` on a structured-field intp array is a numpy `bool_` array (dtype `np.bool_`), not `uint8`, and typed-memoryview assignment from bool dtype to `uint8_t[::1]` is not guaranteed by the buffer protocol.
scenario: "Cython buffer acquisition on newer numpy/Cython that strictly checks buffer format `?` vs `B` → `ValueError: Buffer dtype mismatch, expected 'unsigned char' but got 'bool'` at runtime on every `fit` that hits the `eom` branch."
contract: `child_selection` must be typed as a boolean-compatible memoryview (e.g. `cnp.uint8_t[::1]` assigned from `(cluster_tree['parent'] == node).view(np.uint8)`) or declared as `cnp.npy_bool[::1]`, with the assignment made explicit.
instances: single-instance
```
</details>

### [MEDIUM] `cnp.*_t` typing pervasive in `_tree.pyx` contradicts plan bullet #1 and diverges from `_tree.pxd`

Plan requires single-sourcing typedefs on `_typedefs.pxd`; leaving `cnp.*_t` in `_tree.pyx` → a future rename in `_typedefs.pxd` (e.g. widening `intp_t`) silently produces mismatched types across the pxd/pyx boundary in this one file, and any subsequent developer trying to grep-audit uses of the shared typedefs will miss `_tree.pyx` entirely. Replace every `cnp.intp_t` / `cnp.float64_t` / `cnp.uint8_t` in `_tree.pyx` with the corresponding `intp_t` / `float64_t` / `uint8_t` cimported from `...utils._typedefs`, matching what `_tree.pxd`, `_linkage.pyx`, and `_reachability.pyx` do.

<details><summary>verbatim finding</summary>

```
### F100 — `cnp.*_t` typing pervasive in `_tree.pyx` contradicts plan bullet #1 and diverges from `_tree.pxd`
severity: medium
evidence: The PR plan explicitly lists "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`" as a novel change. `sklearn/cluster/_hdbscan/_tree.pxd:36` already imports and uses `intp_t, float64_t, uint8_t` from `...utils._typedefs`. However `sklearn/cluster/_hdbscan/_tree.pyx` still declares 90 occurrences of `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` (verified via grep) — e.g. lines 39-40 (`cdef cnp.float64_t INFTY = np.inf` / `cdef cnp.intp_t NOISE = -1`), 58-73 (function signature), 145-155, 240-249, 281-290, 305-311, 335-343, 355-368, 388-400, 429-441, 471-483, 513-522, 555-565, 573-580, 605-621, 641-654, 693-706. Sister files `_linkage.pyx` and `_reachability.pyx` were converted (they cimport from `_typedefs` at lines 47 and 46 respectively), but `_tree.pyx` was not converted alongside them — the file uses two type systems in parallel (its own `cnp.*_t` locals interoperating with `_tree.pxd`-declared `intp_t`/`float64_t` struct fields).
scenario: "Plan requires single-sourcing typedefs on `_typedefs.pxd`; leaving `cnp.*_t` in `_tree.pyx` → a future rename in `_typedefs.pxd` (e.g. widening `intp_t`) silently produces mismatched types across the pxd/pyx boundary in this one file, and any subsequent developer trying to grep-audit uses of the shared typedefs will miss `_tree.pyx` entirely."
contract: Replace every `cnp.intp_t` / `cnp.float64_t` / `cnp.uint8_t` in `_tree.pyx` with the corresponding `intp_t` / `float64_t` / `uint8_t` cimported from `...utils._typedefs`, matching what `_tree.pxd`, `_linkage.pyx`, and `_reachability.pyx` do.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:61, sklearn/cluster/_hdbscan/_tree.pyx:66, sklearn/cluster/_hdbscan/_tree.pyx:66-67, sklearn/cluster/_hdbscan/_tree.pyx:67, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:90-91, sklearn/cluster/_hdbscan/_tree.pyx:91, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:145-155, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:147, sklearn/cluster/_hdbscan/_tree.pyx:150, sklearn/cluster/_hdbscan/_tree.pyx:151, sklearn/cluster/_hdbscan/_tree.pyx:153, sklearn/cluster/_hdbscan/_tree.pyx:154, sklearn/cluster/_hdbscan/_tree.pyx:155, sklearn/cluster/_hdbscan/_tree.pyx:182, sklearn/cluster/_hdbscan/_tree.pyx:240, sklearn/cluster/_hdbscan/_tree.pyx:240-249, sklearn/cluster/_hdbscan/_tree.pyx:241, sklearn/cluster/_hdbscan/_tree.pyx:243, sklearn/cluster/_hdbscan/_tree.pyx:244, sklearn/cluster/_hdbscan/_tree.pyx:246, sklearn/cluster/_hdbscan/_tree.pyx:247, sklearn/cluster/_hdbscan/_tree.pyx:248, sklearn/cluster/_hdbscan/_tree.pyx:249, sklearn/cluster/_hdbscan/_tree.pyx:280-290, sklearn/cluster/_hdbscan/_tree.pyx:281, sklearn/cluster/_hdbscan/_tree.pyx:289, sklearn/cluster/_hdbscan/_tree.pyx:290, sklearn/cluster/_hdbscan/_tree.pyx:299, sklearn/cluster/_hdbscan/_tree.pyx:299-329, sklearn/cluster/_hdbscan/_tree.pyx:305, sklearn/cluster/_hdbscan/_tree.pyx:306, sklearn/cluster/_hdbscan/_tree.pyx:307, sklearn/cluster/_hdbscan/_tree.pyx:311, sklearn/cluster/_hdbscan/_tree.pyx:335, sklearn/cluster/_hdbscan/_tree.pyx:336, sklearn/cluster/_hdbscan/_tree.pyx:340, sklearn/cluster/_hdbscan/_tree.pyx:343, sklearn/cluster/_hdbscan/_tree.pyx:355, sklearn/cluster/_hdbscan/_tree.pyx:358, sklearn/cluster/_hdbscan/_tree.pyx:359-475, sklearn/cluster/_hdbscan/_tree.pyx:360, sklearn/cluster/_hdbscan/_tree.pyx:361, sklearn/cluster/_hdbscan/_tree.pyx:362, sklearn/cluster/_hdbscan/_tree.pyx:363, sklearn/cluster/_hdbscan/_tree.pyx:388, sklearn/cluster/_hdbscan/_tree.pyx:389, sklearn/cluster/_hdbscan/_tree.pyx:395, sklearn/cluster/_hdbscan/_tree.pyx:396, sklearn/cluster/_hdbscan/_tree.pyx:400, sklearn/cluster/_hdbscan/_tree.pyx:429, sklearn/cluster/_hdbscan/_tree.pyx:432, sklearn/cluster/_hdbscan/_tree.pyx:434, sklearn/cluster/_hdbscan/_tree.pyx:435, sklearn/cluster/_hdbscan/_tree.pyx:436, sklearn/cluster/_hdbscan/_tree.pyx:437, sklearn/cluster/_hdbscan/_tree.pyx:441, sklearn/cluster/_hdbscan/_tree.pyx:471, sklearn/cluster/_hdbscan/_tree.pyx:472, sklearn/cluster/_hdbscan/_tree.pyx:473, sklearn/cluster/_hdbscan/_tree.pyx:483, sklearn/cluster/_hdbscan/_tree.pyx:513, sklearn/cluster/_hdbscan/_tree.pyx:513-559, sklearn/cluster/_hdbscan/_tree.pyx:514, sklearn/cluster/_hdbscan/_tree.pyx:517, sklearn/cluster/_hdbscan/_tree.pyx:518, sklearn/cluster/_hdbscan/_tree.pyx:519, sklearn/cluster/_hdbscan/_tree.pyx:520, sklearn/cluster/_hdbscan/_tree.pyx:522, sklearn/cluster/_hdbscan/_tree.pyx:535, sklearn/cluster/_hdbscan/_tree.pyx:555, sklearn/cluster/_hdbscan/_tree.pyx:557, sklearn/cluster/_hdbscan/_tree.pyx:565, sklearn/cluster/_hdbscan/_tree.pyx:569, sklearn/cluster/_hdbscan/_tree.pyx:573, sklearn/cluster/_hdbscan/_tree.pyx:576, sklearn/cluster/_hdbscan/_tree.pyx:582, sklearn/cluster/_hdbscan/_tree.pyx:605, sklearn/cluster/_hdbscan/_tree.pyx:606, sklearn/cluster/_hdbscan/_tree.pyx:606-698, sklearn/cluster/_hdbscan/_tree.pyx:610, sklearn/cluster/_hdbscan/_tree.pyx:641, sklearn/cluster/_hdbscan/_tree.pyx:642, sklearn/cluster/_hdbscan/_tree.pyx:646, sklearn/cluster/_hdbscan/_tree.pyx:647, sklearn/cluster/_hdbscan/_tree.pyx:693, sklearn/cluster/_hdbscan/_tree.pyx:694, sklearn/cluster/_hdbscan/_tree.pyx:695, sklearn/cluster/_hdbscan/_tree.pyx:696, sklearn/cluster/_hdbscan/_tree.pyx:697, sklearn/cluster/_hdbscan/_tree.pyx:698, sklearn/cluster/_hdbscan/_tree.pyx:700]
```
</details>

### [MEDIUM] `_do_labelling` sizes `result` by root cluster id, not sample count

someone changes the condensed-tree id offset (e.g. renumbering conventions) → silent wrong allocation size; the invariant 'smallest parent id == n_samples' is nowhere stated in code or comments and is not defensively asserted. Name/derive `n_samples` explicitly (as done elsewhere, e.g. line 90) and allocate `result = np.empty(n_samples, dtype=np.intp)`, with an assertion or comment pinning the invariant.

<details><summary>verbatim finding</summary>

```
### F135 — `_do_labelling` sizes `result` by root cluster id, not sample count
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-481 — `root_cluster = np.min(parent_array); result = np.empty(root_cluster, dtype=np.intp)`. `root_cluster` is the smallest parent id (== `n_samples`) by convention in the condensed-tree encoding, so this is a surprising cross-reference that reads as "allocate a cluster-id-many array" but actually happens to equal `n_samples`.
scenario: "someone changes the condensed-tree id offset (e.g. renumbering conventions) → silent wrong allocation size; the invariant 'smallest parent id == n_samples' is nowhere stated in code or comments and is not defensively asserted."
contract: Name/derive `n_samples` explicitly (as done elsewhere, e.g. line 90) and allocate `result = np.empty(n_samples, dtype=np.intp)`, with an assertion or comment pinning the invariant.
instances: single-instance
```
</details>

### [MEDIUM] `traverse_upwards` compares scalar to array from boolean-indexing without `.item()`

any condensed_tree encoding where `child == leaf` matches 0 or >1 rows (e.g. malformed input, or a future encoding change) → silent wrong comparison or crash; behavior is not documented at the function boundary. Extract the scalar explicitly (e.g. `.item()` on a `[0]`-indexed sub-array) and document/assert the "exactly one row" invariant at the call site.

<details><summary>verbatim finding</summary>

```
### F137 — `traverse_upwards` compares scalar to array from boolean-indexing without `.item()`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586-593 — `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` yields a 1-D ndarray, then compared `if parent == root:` (elementwise, returns an ndarray) and later `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']`. Cython typing on line 582 declares `cnp.intp_t parent` but the RHS is an array; Cython will attempt scalar coercion (works only when the array happens to have length 1). Similar pattern in `_do_labelling` at line 498 (`parent_lambda = lambda_array[child_array == n]`) and in `_get_clusters` (line 745 `eom_clusters[0]`). This is speculative abstraction: leans on hidden "always length 1" invariants.
scenario: "any condensed_tree encoding where `child == leaf` matches 0 or >1 rows (e.g. malformed input, or a future encoding change) → silent wrong comparison or crash; behavior is not documented at the function boundary."
contract: Extract the scalar explicitly (e.g. `.item()` on a `[0]`-indexed sub-array) and document/assert the "exactly one row" invariant at the call site.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593, sklearn/cluster/_hdbscan/_tree.pyx:498]
```
</details>

### [MEDIUM] `bfs_from_hierarchy` invariant comment references stale 2D indexing that no longer exists

Reader tries to reproduce the described indexing → hits AttributeError/IndexError; the load-bearing invariant is described in a schema that no longer exists Rewrite the comment to describe the structured-array fields actually used (`hierarchy[i - n_samples].left_node` / `.right_node`).

<details><summary>verbatim finding</summary>

```
### F174 — `bfs_from_hierarchy` invariant comment references stale 2D indexing that no longer exists
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:97-98 — comment reads "By construction, node i is formed by the union of nodes hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]"; the hierarchy is now a 1D structured array of `HIERARCHY_t` accessed via `.left_node`/`.right_node` (see sklearn/cluster/_hdbscan/_tree.pxd:34-38 and the usage at sklearn/cluster/_hdbscan/_tree.pyx:109-110)
scenario: "Reader tries to reproduce the described indexing → hits AttributeError/IndexError; the load-bearing invariant is described in a schema that no longer exists"
contract: Rewrite the comment to describe the structured-array fields actually used (`hierarchy[i - n_samples].left_node` / `.right_node`).
instances: single-instance
```
</details>

### [MEDIUM] `_get_clusters` docstring documents a `stabilities` return value that is not returned

Maintainer trusts docstring, writes downstream code unpacking three values → ValueError at runtime, or silently binds the probability array to `stabilities` Remove the `stabilities` entry from the Returns section (or add and return the value); pin the fix as removal to match current behavior.

<details><summary>verbatim finding</summary>

```
### F175 — `_get_clusters` docstring documents a `stabilities` return value that is not returned
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:681-690 — Returns section documents three items: `labels`, `probabilities`, and `stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster.`; sklearn/cluster/_hdbscan/_tree.pyx:797 — `return (labels, probs)` returns only two items
scenario: "Maintainer trusts docstring, writes downstream code unpacking three values → ValueError at runtime, or silently binds the probability array to `stabilities`"
contract: Remove the `stabilities` entry from the Returns section (or add and return the value); pin the fix as removal to match current behavior.
instances: single-instance
```
</details>

### [MEDIUM] `bfs_from_cluster_tree` uses `np.isin` on unsorted parents per BFS level

Cluster tree with many nodes and non-trivial depth → repeated full-array `np.isin` scans dominate leaf/eom cluster selection Build a `parent -> [child_indices]` adjacency map once at the top of `_get_clusters`/`epsilon_search` and iterate that map for BFS in O(reachable_nodes).

<details><summary>verbatim finding</summary>

```
### F201 — `bfs_from_cluster_tree` uses `np.isin` on unsorted parents per BFS level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-294 — `while len(process_queue) > 0: result.extend(process_queue.tolist()); process_queue = children[np.isin(parents, process_queue)]`. `np.isin` over the entire `parents` array (size O(n)) is executed once per BFS depth level; combined with callers (`epsilon_search`, `_get_clusters`) invoking BFS per node/leaf, aggregate cost is O(n²·depth) or worse.
scenario: "Cluster tree with many nodes and non-trivial depth → repeated full-array `np.isin` scans dominate leaf/eom cluster selection"
contract: Build a `parent -> [child_indices]` adjacency map once at the top of `_get_clusters`/`epsilon_search` and iterate that map for BFS in O(reachable_nodes).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:279-296, sklearn/cluster/_hdbscan/_tree.pyx:292, sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]
```
</details>

### [MEDIUM] `epsilon_search` uses O(n²) `list not-in` membership

`cluster_selection_method='leaf'` (or EOM with `cluster_selection_epsilon>0`) on a tree with many leaves → epsilon search quadratic in the number of processed nodes Use a `set()` for `processed` so containment/insertion are O(1).

<details><summary>verbatim finding</summary>

```
### F202 — `epsilon_search` uses O(n²) `list not-in` membership
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:611-634 — `processed = list()` then `if leaf not in processed:` and `processed.append(sub_node)` inside the loop over leaves; membership test on a Python list is O(len(processed)), giving O(L²) behavior in the number of leaves plus subtree sizes.
scenario: "`cluster_selection_method='leaf'` (or EOM with `cluster_selection_epsilon>0`) on a tree with many leaves → epsilon search quadratic in the number of processed nodes"
contract: Use a `set()` for `processed` so containment/insertion are O(1).
instances: single-instance
```
</details>

### [MEDIUM] `traverse_upwards` recomputes full-array boolean masks per recursion

Deep condensed-tree parent chain plus many leaves under `cluster_selection_epsilon>0` → O(L·d·n) post-processing Precompute `child -> (parent, value)` map once outside the recursion (or a `child -> row_index` lookup via `np.argsort(children)` + `np.searchsorted`), and index the map instead of masking the full array each call.

<details><summary>verbatim finding</summary>

```
### F203 — `traverse_upwards` recomputes full-array boolean masks per recursion
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:576-602 — each recursive call executes `cluster_tree['child'] == leaf` and `cluster_tree[cluster_tree['child'] == parent]['value']` (both O(n) scans of the full cluster_tree), then recurses on `parent`. A parent chain of length d gives O(d·n) work per starting leaf; called from `epsilon_search` once per unprocessed leaf.
scenario: "Deep condensed-tree parent chain plus many leaves under `cluster_selection_epsilon>0` → O(L·d·n) post-processing"
contract: Precompute `child -> (parent, value)` map once outside the recursion (or a `child -> row_index` lookup via `np.argsort(children)` + `np.searchsorted`), and index the map instead of masking the full array each call.
instances: single-instance
```
</details>

### [MEDIUM] `recurse_leaf_dfs` flattens with `sum([...], [])` (quadratic list concat)

`cluster_selection_method='leaf'` on a large condensed tree → leaf enumeration becomes quadratic in the number of leaves, and each recursion pays a full-array scan on top Replace the recursion with an iterative DFS using an explicit stack and `list.extend`/`list.append`; precompute a `parent -> children` adjacency map once for the tree walk.

<details><summary>verbatim finding</summary>

```
### F204 — `recurse_leaf_dfs` flattens with `sum([...], [])` (quadratic list concat)
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555-566 — `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`; Python's `sum` on lists is O(total_length²) due to repeated list copying. Additionally, at every recursion `cluster_tree[cluster_tree['parent'] == current_node]['child']` scans the whole tree.
scenario: "`cluster_selection_method='leaf'` on a large condensed tree → leaf enumeration becomes quadratic in the number of leaves, and each recursion pays a full-array scan on top"
contract: Replace the recursion with an iterative DFS using an explicit stack and `list.extend`/`list.append`; precompute a `parent -> children` adjacency map once for the tree walk.
instances: single-instance
```
</details>

### [MEDIUM] `_condense_tree` re-BFSes hierarchy from within its own BFS loop

Data producing a chain-like single-linkage tree with many small merges → condense step degrades to O(n²) Reuse the already-computed traversal order (or memoize subtree membership) instead of re-invoking `bfs_from_hierarchy` for each pruned subtree.

<details><summary>verbatim finding</summary>

```
### F205 — `_condense_tree` re-BFSes hierarchy from within its own BFS loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148-230 — the function first calls `bfs_from_hierarchy(hierarchy, root)` once to build `node_list` in linear time, then in the pruning branches (lines 200, 207, 216, 225) calls `bfs_from_hierarchy(hierarchy, left/right)` again per pruned subtree, walking the same descendants that will subsequently be marked in `ignore`. For skewed hierarchies where most children fall under `min_cluster_size` this is O(n²).
scenario: "Data producing a chain-like single-linkage tree with many small merges → condense step degrades to O(n²)"
contract: Reuse the already-computed traversal order (or memoize subtree membership) instead of re-invoking `bfs_from_hierarchy` for each pruned subtree.
instances: single-instance
```
</details>

### [MEDIUM] `_get_clusters` performs O(n²) boolean scans over cluster_tree per node

Fit HDBSCAN on a dataset that yields a large condensed cluster tree (deep hierarchy of small merges) → EOM cluster selection becomes O(n²) even after MST construction is done, dominating post-processing time Precompute a mapping `parent -> child_indices` once (e.g., using `np.argsort` on the parent column plus `np.searchsorted`, or a Python dict grouping), then look up children per node in O(k) instead of scanning the full array each time.

<details><summary>verbatim finding</summary>

```
### F206 — `_get_clusters` performs O(n²) boolean scans over cluster_tree per node
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-732 — `for node in node_list:` loop, each iteration computes `child_selection = (cluster_tree['parent'] == node)` a full-array boolean mask, then `np.sum([...])` over `cluster_tree['child'][child_selection]`; `node_list` has O(n) entries and `cluster_tree` has O(n) rows, giving O(n²) total work in the EOM path.
scenario: "Fit HDBSCAN on a dataset that yields a large condensed cluster tree (deep hierarchy of small merges) → EOM cluster selection becomes O(n²) even after MST construction is done, dominating post-processing time"
contract: Precompute a mapping `parent -> child_indices` once (e.g., using `np.argsort` on the parent column plus `np.searchsorted`, or a Python dict grouping), then look up children per node in O(k) instead of scanning the full array each time.
instances: single-instance
```
</details>

### [MEDIUM] Default `n_jobs=4` contradicts the documented default

User instantiates `HDBSCAN()` expecting the documented `None` default → 4 worker jobs are always spawned inside `pairwise_distances`, oversubscribing CPUs (especially inside outer parallel loops) and diverging from every other sklearn estimator's documented and actual default of `None`. Change the default to `n_jobs=None` so the runtime behavior matches the documented default.

<details><summary>verbatim finding</summary>

```
### F3 — Default `n_jobs=4` contradicts the documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature `n_jobs=4`; the docstring at lines 486–490 says `"None means 1 unless in a :obj:joblib.parallel_backend context. -1 means using all processors."` and the "default=None" convention is stated at line 486 (`n_jobs : int, default=None`).
scenario: "User instantiates `HDBSCAN()` expecting the documented `None` default → 4 worker jobs are always spawned inside `pairwise_distances`, oversubscribing CPUs (especially inside outer parallel loops) and diverging from every other sklearn estimator's documented and actual default of `None`."
contract: Change the default to `n_jobs=None` so the runtime behavior matches the documented default.
instances: single-instance
```
</details>

### [MEDIUM] `_hdbscan_brute` default `alpha=None` will crash on `distance_matrix /= alpha`

any direct call to `_hdbscan_brute(X)` without an explicit `alpha` → immediate `TypeError` at line 241 default `alpha=1.0` (matching `_hdbscan_prims` and the docstring at line 183).

<details><summary>verbatim finding</summary>

```
### F7 — `_hdbscan_brute` default `alpha=None` will crash on `distance_matrix /= alpha`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None` in the signature, but sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally runs `distance_matrix /= alpha` (no None check). `ndarray /= None` raises `TypeError`. The companion `_hdbscan_prims` at line 273 correctly defaults `alpha=1.0`.
scenario: "any direct call to `_hdbscan_brute(X)` without an explicit `alpha` → immediate `TypeError` at line 241"
contract: default `alpha=1.0` (matching `_hdbscan_prims` and the docstring at line 183).
instances: single-instance
```
</details>

### [MEDIUM] `remap_single_linkage_tree` iterates over a Python `set`, producing non-deterministic outlier tree

user fits HDBSCAN twice on identical data with non-finite rows in separate processes and inspects `self._single_linkage_tree_` → sees different orderings of the appended outlier subtree between runs pass and iterate a sorted, ordered container (e.g. `np.sort(np.unique(np.concatenate([infinite_index, missing_index])))`) so the appended outlier subtree is deterministic.

<details><summary>verbatim finding</summary>

```
### F8 — `remap_single_linkage_tree` iterates over a Python `set`, producing non-deterministic outlier tree
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` (a `set`) into `remap_single_linkage_tree`. sklearn/cluster/_hdbscan/hdbscan.py:388 `for i, outlier in enumerate(non_finite):` iterates a `set` and writes `outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`; `set` iteration order is not guaranteed reproducible across processes / PYTHONHASHSEED values, so `self._single_linkage_tree_` ordering of the appended outlier edges is non-deterministic.
scenario: "user fits HDBSCAN twice on identical data with non-finite rows in separate processes and inspects `self._single_linkage_tree_` → sees different orderings of the appended outlier subtree between runs"
contract: pass and iterate a sorted, ordered container (e.g. `np.sort(np.unique(np.concatenate([infinite_index, missing_index])))`) so the appended outlier subtree is deterministic.
instances: single-instance
```
</details>

### [MEDIUM] `_hdbscan_brute` calls `distance_matrix /= alpha` even in the `metric="precomputed"` + `copy=False` path, mutating user-owned data

User calls `HDBSCAN(metric='precomputed', alpha=2.0, copy=False).fit(D)` → `D` is silently divided by 2 in place; subsequent code using `D` gets wrong distances When `alpha != 1.0` and `metric == "precomputed"` and `copy=False`, force a copy before the in-place `/=`.

<details><summary>verbatim finding</summary>

```
### F9 — `_hdbscan_brute` calls `distance_matrix /= alpha` even in the `metric="precomputed"` + `copy=False` path, mutating user-owned data
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:236-241 — when `metric == "precomputed"`, `distance_matrix = X.copy() if copy else X`; unconditionally after this, `distance_matrix /= alpha` runs in-place. With `copy=False` (the default) and `alpha != 1.0`, the user's precomputed distance matrix `X` is silently divided by `alpha` and mutated. The docstring at lines 207-212 states "it only applies when `metric='precomputed'`, when passing a dense array or a CSR sparse array/matrix", implying the user's matrix should be preserved when `copy=True` but does not warn that `copy=False` will mutate it.
scenario: "User calls `HDBSCAN(metric='precomputed', alpha=2.0, copy=False).fit(D)` → `D` is silently divided by 2 in place; subsequent code using `D` gets wrong distances"
contract: When `alpha != 1.0` and `metric == "precomputed"` and `copy=False`, force a copy before the in-place `/=`.
instances: single-instance
```
</details>

### [MEDIUM] `n_jobs=4` default silently spawns worker processes without user opt-in

A user (or a sandboxed/containerized service) instantiates `HDBSCAN()` expecting the documented default → the estimator silently spins up up to four parallel workers via `pairwise_distances`, exceeding the caller's implicit CPU/memory boundary and contradicting the documented behavior; in constrained environments (containers with `cpuset=1`, CI runners) this yields resource exhaustion the caller never authorized. Change the default to `n_jobs=None` to match the documented contract and the rest of scikit-learn's clustering estimators, so that spawning parallel workers requires explicit caller consent.

<details><summary>verbatim finding</summary>

```
### F21 — `n_jobs=4` default silently spawns worker processes without user opt-in
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in `HDBSCAN.__init__`; the docstring at line 486–490 states "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors." — implying a `None` default per scikit-learn convention, but the actual default is a hard-coded `4`.
scenario: "A user (or a sandboxed/containerized service) instantiates `HDBSCAN()` expecting the documented default → the estimator silently spins up up to four parallel workers via `pairwise_distances`, exceeding the caller's implicit CPU/memory boundary and contradicting the documented behavior; in constrained environments (containers with `cpuset=1`, CI runners) this yields resource exhaustion the caller never authorized."
contract: Change the default to `n_jobs=None` to match the documented contract and the rest of scikit-learn's clustering estimators, so that spawning parallel workers requires explicit caller consent.
instances: single-instance
```
</details>

### [MEDIUM] `_get_finite_row_indices` misclassifies rows via `X.sum(axis=1)` producing NaN/Inf boundary confusion

User passes a feature array whose per-row values overflow when summed → rows containing only finite data are silently classified as infinite outliers and excluded from clustering, producing misleading labels that the caller trusts as reflecting the input data. Detect non-finite samples per-element (`np.any(np.isnan(X), axis=1)` and `np.any(np.isinf(X), axis=1)` for dense, equivalent sparse-safe iteration for sparse) rather than by row-summation.

<details><summary>verbatim finding</summary>

```
### F22 — `_get_finite_row_indices` misclassifies rows via `X.sum(axis=1)` producing NaN/Inf boundary confusion
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:721-728 — `reduced_X = X.sum(axis=1)`; `missing_index = list(np.isnan(reduced_X).nonzero()[0])`; `infinite_index = list(np.isinf(reduced_X).nonzero()[0])`. Detection of missing/infinite samples is driven by the row *sum*, not per-cell checks: a row containing both `+inf` and `-inf` yields `nan` from the sum and is (mis)labeled as `missing` (label -3), and a row whose finite values happen to overflow to `±inf` on summation is (mis)labeled as `infinite` (label -2). `_get_finite_row_indices` on line 406 uses the same fragile pattern `np.isfinite(matrix.sum(axis=1)).nonzero()` for the dense branch.
scenario: "User passes a feature array whose per-row values overflow when summed → rows containing only finite data are silently classified as infinite outliers and excluded from clustering, producing misleading labels that the caller trusts as reflecting the input data."
contract: Detect non-finite samples per-element (`np.any(np.isnan(X), axis=1)` and `np.any(np.isinf(X), axis=1)` for dense, equivalent sparse-safe iteration for sparse) rather than by row-summation.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:721, sklearn/cluster/_hdbscan/hdbscan.py:725, sklearn/cluster/_hdbscan/hdbscan.py:728, sklearn/cluster/_hdbscan/hdbscan.py:406]
```
</details>

### [MEDIUM] `_weighted_cluster_center` filters only `{-1, -2}` from labels, missing `-3` (missing-data label)

User fits HDBSCAN with `store_centers='centroid'` on data containing NaN → the returned `n_clusters` count includes a phantom slot for the -3 outlier label, and the corresponding row of `centroids_` receives `np.average` over an empty selection (RuntimeWarning: mean of empty slice) or is filled with garbage from `np.empty`. Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})`, matching the documented contract and the constants defined in `_OUTLIER_ENCODING`.

<details><summary>verbatim finding</summary>

```
### F23 — `_weighted_cluster_center` filters only `{-1, -2}` from labels, missing `-3` (missing-data label)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The class docstring (sklearn/cluster/_hdbscan/hdbscan.py:560-573) states "`n_clusters` only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded." Since `-3` is the `missing` outlier label defined in `_OUTLIER_ENCODING` (line 74), missing-data rows can leak into `n_clusters`, and the loop `for idx in range(n_clusters)` may attempt to access uninitialized `centroids_[idx]`/`medoids_[idx]` entries or, conversely, the last "cluster" index may correspond to no rows at all in `mask`.
scenario: "User fits HDBSCAN with `store_centers='centroid'` on data containing NaN → the returned `n_clusters` count includes a phantom slot for the -3 outlier label, and the corresponding row of `centroids_` receives `np.average` over an empty selection (RuntimeWarning: mean of empty slice) or is filled with garbage from `np.empty`."
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})`, matching the documented contract and the constants defined in `_OUTLIER_ENCODING`.
instances: single-instance
```
</details>

### [MEDIUM] `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default and sklearn convention

User instantiates `HDBSCAN()` inside a `joblib.parallel_backend` context expecting to inherit the outer backend → HDBSCAN silently forks 4 concurrent processes/threads inside `pairwise_distances`/`NearestNeighbors`, over-subscribing the machine and defeating the parent's parallel policy The default MUST be `n_jobs=None` to honor `joblib.parallel_backend` contexts consistent with the docstring and all other sklearn estimators

<details><summary>verbatim finding</summary>

```
### F36 — `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default and sklearn convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in `__init__` signature; docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents "`None` means 1 unless in a `joblib.parallel_backend` context" implying `n_jobs=None` as default, matching sklearn convention throughout the file (e.g. `_hdbscan_brute` at line 163 uses `n_jobs=None`, `_hdbscan_prims` at line 276 uses `n_jobs=None`)
scenario: "User instantiates `HDBSCAN()` inside a `joblib.parallel_backend` context expecting to inherit the outer backend → HDBSCAN silently forks 4 concurrent processes/threads inside `pairwise_distances`/`NearestNeighbors`, over-subscribing the machine and defeating the parent's parallel policy"
contract: The default MUST be `n_jobs=None` to honor `joblib.parallel_backend` contexts consistent with the docstring and all other sklearn estimators
instances: single-instance
```
</details>

### [MEDIUM] `remap_single_linkage_tree` iterates a `set`, producing non-deterministic `_single_linkage_tree_`

User fits `HDBSCAN` twice on the same data containing both `np.inf` and `np.nan` outliers → `self._single_linkage_tree_` differs across runs (rows reordered) because Python `set` iteration order is not stable across processes, breaking any code that hashes or diffs the tree attribute `non_finite` MUST be passed as a sorted, deterministic sequence (e.g. `np.unique(np.concatenate([infinite_index, missing_index]))`) so `_single_linkage_tree_` is reproducible

<details><summary>verbatim finding</summary>

```
### F37 — `remap_single_linkage_tree` iterates a `set`, producing non-deterministic `_single_linkage_tree_`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` into `remap_single_linkage_tree`; sklearn/cluster/_hdbscan/hdbscan.py:388 iterates that set with `for i, outlier in enumerate(non_finite)` and writes `outlier_tree[i] = (outlier, ...)` in set-iteration order
scenario: "User fits `HDBSCAN` twice on the same data containing both `np.inf` and `np.nan` outliers → `self._single_linkage_tree_` differs across runs (rows reordered) because Python `set` iteration order is not stable across processes, breaking any code that hashes or diffs the tree attribute"
contract: `non_finite` MUST be passed as a sorted, deterministic sequence (e.g. `np.unique(np.concatenate([infinite_index, missing_index]))`) so `_single_linkage_tree_` is reproducible
instances: single-instance
```
</details>

### [MEDIUM] `_weighted_cluster_center` counts missing-label (-3) samples as a cluster

User calls fit with `store_centers` set and X contains `np.nan` rows → `-3` is included in `n_clusters`, sizing `self.centroids_`/`self.medoids_` one row too large; the loop `for idx in range(n_clusters)` then produces an all-NaN or empty-mean row (empty slice warning + NaN centroid) and never touches the -3 mask, corrupting the exported centers. Exclude all three outlier encodings when counting non-noise clusters: `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.

<details><summary>verbatim finding</summary>

```
### F39 — `_weighted_cluster_center` counts missing-label (-3) samples as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the class-level `_OUTLIER_ENCODING` (sklearn/cluster/_hdbscan/hdbscan.py:65-79) defines `missing` label `-3`, which is excluded from `labels_` by sklearn/cluster/_hdbscan/hdbscan.py:843.
scenario: "User calls fit with `store_centers` set and X contains `np.nan` rows → `-3` is included in `n_clusters`, sizing `self.centroids_`/`self.medoids_` one row too large; the loop `for idx in range(n_clusters)` then produces an all-NaN or empty-mean row (empty slice warning + NaN centroid) and never touches the -3 mask, corrupting the exported centers."
contract: Exclude all three outlier encodings when counting non-noise clusters: `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.
instances: single-instance
```
</details>

### [MEDIUM] `_assert_all_finite(X.data)` on LIL sparse input misclassifies finiteness

User passes a LIL sparse feature matrix containing `np.nan`/`np.inf` → `_assert_all_finite` receives an object array and either (a) raises an unrelated dtype error, or (b) silently returns True because the object-dtype path does not descend into the per-row lists, sending non-finite data into `_hdbscan_brute` (fail-open path). Materialise a numeric buffer before finiteness checking for sparse formats other than CSR (e.g., convert with `X.tocsr()` before the `_assert_all_finite` call), rather than relying on `X.data` which is format-dependent.

<details><summary>verbatim finding</summary>

```
### F40 — `_assert_all_finite(X.data)` on LIL sparse input misclassifies finiteness
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:703 accepts `accept_sparse=["csr", "lil"]`; sklearn/cluster/_hdbscan/hdbscan.py:710 calls `_assert_all_finite(X.data if issparse(X) else X)`. For LIL sparse, `X.data` is an `object` ndarray of Python lists (one per row), not a numeric buffer.
scenario: "User passes a LIL sparse feature matrix containing `np.nan`/`np.inf` → `_assert_all_finite` receives an object array and either (a) raises an unrelated dtype error, or (b) silently returns True because the object-dtype path does not descend into the per-row lists, sending non-finite data into `_hdbscan_brute` (fail-open path)."
contract: Materialise a numeric buffer before finiteness checking for sparse formats other than CSR (e.g., convert with `X.tocsr()` before the `_assert_all_finite` call), rather than relying on `X.data` which is format-dependent.
instances: single-instance
```
</details>

### [MEDIUM] `_brute_mst` sparse path silently accepts disconnected graphs when `connected_components` count equals 1 but MST is short

User supplies a sparse precomputed distance matrix where the mutual-reachability graph is connected under the sparsity pattern but some MR distances collapse to zero (e.g., duplicate points) → returned MST has fewer than n-1 edges, `make_single_linkage` produces an incomplete/malformed `HIERARCHY_dtype` array, and downstream `_condense_tree` yields wrong labels rather than surfacing the partial-failure state. After `csgraph.minimum_spanning_tree`, assert `len(rows) == n_samples - 1` and raise a clear `ValueError` describing the partial MST rather than continuing with a truncated tree.

<details><summary>verbatim finding</summary>

```
### F41 — `_brute_mst` sparse path silently accepts disconnected graphs when `connected_components` count equals 1 but MST is short
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:111-123 raises only when `csgraph.connected_components(...) > 1`; sklearn/cluster/_hdbscan/hdbscan.py:126-131 builds `mst` via `np.core.records.fromarrays([rows, cols, ...])` of whatever nonzeros scipy returns. `scipy.sparse.csgraph.minimum_spanning_tree` on an already-symmetrized MR graph with zero-valued edges (which are dropped as "no edge") can return fewer than `n_samples - 1` edges.
scenario: "User supplies a sparse precomputed distance matrix where the mutual-reachability graph is connected under the sparsity pattern but some MR distances collapse to zero (e.g., duplicate points) → returned MST has fewer than n-1 edges, `make_single_linkage` produces an incomplete/malformed `HIERARCHY_dtype` array, and downstream `_condense_tree` yields wrong labels rather than surfacing the partial-failure state."
contract: After `csgraph.minimum_spanning_tree`, assert `len(rows) == n_samples - 1` and raise a clear `ValueError` describing the partial MST rather than continuing with a truncated tree.
instances: single-instance
```
</details>

### [MEDIUM] Precomputed sparse path can mutate user's input array in place

User passes a sparse CSR precomputed distance matrix with `HDBSCAN(metric='precomputed', copy=True).fit(D)` → `D.data` is mutated by `distance_matrix /= alpha` and by `_sparse_mutual_reachability_graph`, silently overwriting the caller's matrix; the `copy=True` documented guarantee is fail-open. In `_hdbscan_brute`, when `metric == 'precomputed'` and the input is sparse, honor `copy` by materialising a new CSR (e.g., `distance_matrix = X.copy()`) before any in-place mutation.

<details><summary>verbatim finding</summary>

```
### F42 — Precomputed sparse path can mutate user's input array in place
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:734-740 (`elif issparse(X):` — no `copy=` argument to `_validate_data`); sklearn/cluster/_hdbscan/hdbscan.py:241 in `_hdbscan_brute`: `distance_matrix /= alpha` and sklearn/cluster/_hdbscan/_reachability.pyx:56 documents "Note that all computations are done in-place." The CSR branch in `_reachability.pyx:202-212` writes into `data`. The `copy` kwarg is threaded only for `algorithm=="brute"` (sklearn/cluster/_hdbscan/hdbscan.py:795) and `_hdbscan_brute` only honors it in the dense precomputed branch (sklearn/cluster/_hdbscan/hdbscan.py:222-236); the sparse precomputed branch ignores `copy` entirely.
scenario: "User passes a sparse CSR precomputed distance matrix with `HDBSCAN(metric='precomputed', copy=True).fit(D)` → `D.data` is mutated by `distance_matrix /= alpha` and by `_sparse_mutual_reachability_graph`, silently overwriting the caller's matrix; the `copy=True` documented guarantee is fail-open."
contract: In `_hdbscan_brute`, when `metric == 'precomputed'` and the input is sparse, honor `copy` by materialising a new CSR (e.g., `distance_matrix = X.copy()`) before any in-place mutation.
instances: single-instance
```
</details>

### [MEDIUM] `algorithm="auto"` silently drops `copy=True` for precomputed inputs

User calls `HDBSCAN(metric='precomputed', copy=True).fit_predict(D)` (leaving algorithm at default `'auto'`) → the auto path picks `_hdbscan_brute` without forwarding `copy` → D is mutated in-place (divided by alpha, overwritten with mutual-reachability values) → user's `D` is silently corrupted. The `algorithm="auto"` branch must set `kwargs["copy"] = self.copy` whenever it dispatches to `_hdbscan_brute`, matching the explicit `algorithm="brute"` branch.

<details><summary>verbatim finding</summary>

```
### F43 — `algorithm="auto"` silently drops `copy=True` for precomputed inputs
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:804-819 — the "auto" branch dispatches to `_hdbscan_brute` at line 807 (`if issparse(X) or self.metric not in FAST_METRICS`) but never sets `kwargs["copy"] = self.copy`. In the explicit `algorithm="brute"` branch at line 795 the copy flag is honored. `_hdbscan_brute` defaults `copy=False` at line 164, so with `metric="precomputed"` and `algorithm="auto"` the user-provided distance matrix is modified in place at lines 241/251 despite `copy=True`.
scenario: "User calls `HDBSCAN(metric='precomputed', copy=True).fit_predict(D)` (leaving algorithm at default `'auto'`) → the auto path picks `_hdbscan_brute` without forwarding `copy` → D is mutated in-place (divided by alpha, overwritten with mutual-reachability values) → user's `D` is silently corrupted."
contract: The `algorithm="auto"` branch must set `kwargs["copy"] = self.copy` whenever it dispatches to `_hdbscan_brute`, matching the explicit `algorithm="brute"` branch.
instances: single-instance
```
</details>

### [MEDIUM] `max_cluster_size` is entirely untested

The `max_cluster_size` short-circuit is deleted or its comparison inverted → no test fails. Add a test that fits HDBSCAN with `cluster_selection_method="eom"` and a small `max_cluster_size`, and asserts that no returned cluster exceeds that size.

<details><summary>verbatim finding</summary>

```
### F59 — `max_cluster_size` is entirely untested
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py — no reference to `max_cluster_size` anywhere. The parameter is declared at sklearn/cluster/_hdbscan/hdbscan.py:622-625 and has real semantics at sklearn/cluster/_hdbscan/_tree.pyx:716-717 and 733 (`cluster_sizes[node] > max_cluster_size` forces `is_cluster[node] = False`). The comment at _tree.pyx:717 shows the default sentinel is `n_samples + 1` — regressing this sentinel would silently break EOM cluster selection with no test to catch it.
scenario: "The `max_cluster_size` short-circuit is deleted or its comparison inverted → no test fails."
contract: Add a test that fits HDBSCAN with `cluster_selection_method="eom"` and a small `max_cluster_size`, and asserts that no returned cluster exceeds that size.
instances: single-instance
```
</details>

### [MEDIUM] `n_jobs` default is `4` but documented as `None`; not covered by any test [out-of-theme]

A user reads the docstring, expects joblib context to be respected → HDBSCAN silently forces 4 jobs regardless of the surrounding `parallel_backend`, breaking user expectations and joblib coordination. Set `n_jobs=None` as the default (matching the documented behavior) and add a test asserting the default is `None`.

<details><summary>verbatim finding</summary>

```
### F63 — `n_jobs` default is `4` but documented as `None`; not covered by any test [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`. The docstring at hdbscan.py:486-490 states: "n_jobs : int, default=None. Number of jobs to run in parallel to calculate distances. `None` means 1 unless in a `joblib.parallel_backend` context. `-1` means using all processors." No test in test_hdbscan.py exercises or verifies the effective `n_jobs` default.
scenario: "A user reads the docstring, expects joblib context to be respected → HDBSCAN silently forces 4 jobs regardless of the surrounding `parallel_backend`, breaking user expectations and joblib coordination."
contract: Set `n_jobs=None` as the default (matching the documented behavior) and add a test asserting the default is `None`.
instances: single-instance
```
</details>

### [MEDIUM] `remap_single_linkage_tree` receives a set of indices where docstring declares a boolean array

Any code path that trusts the docstring and passes a boolean mask (the documented contract) → silently wrong tree construction, then arbitrary label assignment Change the parameter documentation to `non_finite : set of int / A set of raw indices corresponding to non-finite samples` so the actual runtime contract matches what the sole caller provides.

<details><summary>verbatim finding</summary>

```
### F78 — `remap_single_linkage_tree` receives a set of indices where docstring declares a boolean array
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 documents `non_finite : ndarray / Boolean array of which entries in the raw data are non-finite`, but the sole caller at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` — a `set` of integer indices. Inside the function line 383 does `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)` and line 388 `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)`. If the argument were actually a boolean ndarray as documented, `len(non_finite)` would equal n_samples (not the outlier count) and `enumerate(non_finite)` would iterate `True/False` values, silently producing wrong outlier rows.
scenario: "Any code path that trusts the docstring and passes a boolean mask (the documented contract) → silently wrong tree construction, then arbitrary label assignment"
contract: Change the parameter documentation to `non_finite : set of int / A set of raw indices corresponding to non-finite samples` so the actual runtime contract matches what the sole caller provides.
instances: single-instance
```
</details>

### [MEDIUM] `_hdbscan_brute` `alpha=None` default violates its numeric contract

Any external / test caller invokes `_hdbscan_brute(X, metric='precomputed')` accepting documented defaults → immediate TypeError inside `distance_matrix /= alpha` The default MUST be `alpha=1.0`, matching both the docstring and the corresponding parameter in `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.

<details><summary>verbatim finding</summary>

```
### F79 — `_hdbscan_brute` `alpha=None` default violates its numeric contract
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`, sklearn/cluster/_hdbscan/hdbscan.py:183 documents `alpha : float, default=1.0`, and sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally executes `distance_matrix /= alpha`. Calling `_hdbscan_brute(X)` without `alpha` therefore raises `TypeError: unsupported operand type(s) for /=: … 'NoneType'` — the default value is not a legal value for the parameter's own contract.
scenario: "Any external / test caller invokes `_hdbscan_brute(X, metric='precomputed')` accepting documented defaults → immediate TypeError inside `distance_matrix /= alpha`"
contract: The default MUST be `alpha=1.0`, matching both the docstring and the corresponding parameter in `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.
instances: single-instance
```
</details>

### [MEDIUM] `labels_` dtype changes between finite and non-finite input paths

Downstream consumer (or `dbscan_clustering` comparison `self.labels_ == _OUTLIER_ENCODING[…]['label']`) that assumes a stable label dtype → different dtype (`intp`/`int64` vs `int32`) between clean and non-finite fits, producing inconsistent behavior when integrating with typed pipelines or on 32-bit platforms where truncation could occur. `HDBSCAN.labels_` must be `np.intp` on every code path.

<details><summary>verbatim finding</summary>

```
### F84 — `labels_` dtype changes between finite and non-finite input paths
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:822-829 assigns `self.labels_` from `tree_to_labels` (which returns `intp` via `_do_labelling`); sklearn/cluster/_hdbscan/hdbscan.py:840 rebinds `self.labels_ = new_labels` where `new_labels = np.empty(self._raw_data.shape[0], dtype=np.int32)` only in the non-finite branch.
scenario: "Downstream consumer (or `dbscan_clustering` comparison `self.labels_ == _OUTLIER_ENCODING[…]['label']`) that assumes a stable label dtype → different dtype (`intp`/`int64` vs `int32`) between clean and non-finite fits, producing inconsistent behavior when integrating with typed pipelines or on 32-bit platforms where truncation could occur."
contract: `HDBSCAN.labels_` must be `np.intp` on every code path.
instances: single-instance
```
</details>

### [MEDIUM] `_weighted_cluster_center` counts `-3` (missing) label as a cluster

`HDBSCAN(store_centers='both').fit(X)` with X containing np.nan → `-3` is treated as a valid cluster index; `mask = self.labels_ == -3` never becomes true for `idx in range(n_clusters)` (since range uses non-negative ids), producing an off-by-one and empty averaging/inf medoid computation for the phantom cluster. The set of "cluster-excluding" outlier labels used to compute `n_clusters` must include every label in `_OUTLIER_ENCODING`, i.e. `{-1, -2, -3}`.

<details><summary>verbatim finding</summary>

```
### F87 — `_weighted_cluster_center` counts `-3` (missing) label as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; `_OUTLIER_ENCODING["missing"]["label"]` is `-3` per lines 73-78, but `-3` is not excluded from the noise-set here even though centroids/medoids should not be computed for missing samples.
scenario: "`HDBSCAN(store_centers='both').fit(X)` with X containing np.nan → `-3` is treated as a valid cluster index; `mask = self.labels_ == -3` never becomes true for `idx in range(n_clusters)` (since range uses non-negative ids), producing an off-by-one and empty averaging/inf medoid computation for the phantom cluster." 
contract: The set of "cluster-excluding" outlier labels used to compute `n_clusters` must include every label in `_OUTLIER_ENCODING`, i.e. `{-1, -2, -3}`.
instances: single-instance
```
</details>

### [MEDIUM] `max_distance` remains in `metric_params` and is forwarded to `pairwise_distances`

user passes `metric_params={'max_distance': 5.0}` with a standard metric like 'euclidean' → `pairwise_distances` raises TypeError from unknown keyword forwarded to the metric callable/scipy backend. `max_distance` must be extracted and removed from `metric_params` (e.g. via `pop`) before `metric_params` is spread to `pairwise_distances`.

<details><summary>verbatim finding</summary>

```
### F88 — `max_distance` remains in `metric_params` and is forwarded to `pairwise_distances`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:239,243 — `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` on line 239 spreads `metric_params`; line 243 `max_distance = metric_params.get("max_distance", 0.0)` reads but does not pop. If the caller supplies `metric_params={"max_distance": ...}`, `pairwise_distances` receives the unknown keyword and propagates it to underlying metric functions.
scenario: "user passes `metric_params={'max_distance': 5.0}` with a standard metric like 'euclidean' → `pairwise_distances` raises TypeError from unknown keyword forwarded to the metric callable/scipy backend."
contract: `max_distance` must be extracted and removed from `metric_params` (e.g. via `pop`) before `metric_params` is spread to `pairwise_distances`.
instances: single-instance
```
</details>

### [MEDIUM] Hardcoded outlier labels bypass `_OUTLIER_ENCODING`, driving values apart

Outlier labels are re-encoded inline at `_weighted_cluster_center` and `_tree.pyx` `get_probabilities` → if `_OUTLIER_ENCODING`/`NOISE` are ever renumbered (e.g., adding a fourth outlier category or shifting `-3` handling), the hardcoded literals silently disagree; for the current tree, `_weighted_cluster_center` already excludes `-1,-2` but not `-3`, causing `medoids_`/`centroids_` to size incorrectly when missing-data rows exist and produce empty rows or an IndexError at line 908 (`mask = self.labels_ == idx`) where `idx` would iterate up to a count that includes `-3` as a cluster. Reference `_OUTLIER_ENCODING` (or a derived `_OUTLIER_LABELS = {v['label'] for v in _OUTLIER_ENCODING.values()} | {-1}`) instead of the literal set `{-1, -2}` in `_weighted_cluster_center`, and use the `NOISE` constant instead of `-1` in `_tree.pyx:541`.

<details><summary>verbatim finding</summary>

```
### F99 — Hardcoded outlier labels bypass `_OUTLIER_ENCODING`, driving values apart
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` hardcodes the noise (`-1`) and infinite-outlier (`-2`) labels; the missing-outlier label (`-3`) is silently omitted from the exclusion set. `_OUTLIER_ENCODING` is defined at lines 65-79 exactly to be the single source of truth for these labels, and is used elsewhere (lines 842-843, 850-851, 964-969). sklearn/cluster/_hdbscan/_tree.pyx:541 — `if cluster_num == -1: continue` hardcodes the noise label instead of the module-level `NOISE` constant defined on line 40 (`cdef cnp.intp_t NOISE = -1`) and used at lines 414/420/492.
scenario: "Outlier labels are re-encoded inline at `_weighted_cluster_center` and `_tree.pyx` `get_probabilities` → if `_OUTLIER_ENCODING`/`NOISE` are ever renumbered (e.g., adding a fourth outlier category or shifting `-3` handling), the hardcoded literals silently disagree; for the current tree, `_weighted_cluster_center` already excludes `-1,-2` but not `-3`, causing `medoids_`/`centroids_` to size incorrectly when missing-data rows exist and produce empty rows or an IndexError at line 908 (`mask = self.labels_ == idx`) where `idx` would iterate up to a count that includes `-3` as a cluster."
contract: Reference `_OUTLIER_ENCODING` (or a derived `_OUTLIER_LABELS = {v['label'] for v in _OUTLIER_ENCODING.values()} | {-1}`) instead of the literal set `{-1, -2}` in `_weighted_cluster_center`, and use the `NOISE` constant instead of `-1` in `_tree.pyx:541`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/_hdbscan/_tree.pyx:541]
```
</details>

### [MEDIUM] `n_jobs` default drifts between `__init__` and its docstring

user reads docstring, expects default single-thread joblib behavior, but constructor silently uses 4 workers → unexpected parallelism, memory usage, and non-reproducible resource consumption in constrained environments Pin the constructor default to `n_jobs=None` to match the documented single-sourced default across the file and other sklearn estimators.

<details><summary>verbatim finding</summary>

```
### F102 — `n_jobs` default drifts between `__init__` and its docstring
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring `n_jobs : int, default=None`; sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature uses `n_jobs=4`. The `_brute_mst`/`_hdbscan_brute`/`_hdbscan_prims` helper docstrings (lines 197, 303) also state `n_jobs : int, default=None`.
scenario: "user reads docstring, expects default single-thread joblib behavior, but constructor silently uses 4 workers → unexpected parallelism, memory usage, and non-reproducible resource consumption in constrained environments"
contract: Pin the constructor default to `n_jobs=None` to match the documented single-sourced default across the file and other sklearn estimators.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:486, sklearn/cluster/_hdbscan/hdbscan.py:658]
```
</details>

### [MEDIUM] `alpha=None` default in `_hdbscan_brute` signature drifts from its docstring `default=1.0` [out-of-theme]

Any refactor that calls `_hdbscan_brute(X)` without explicitly passing `alpha=1.0` (relying on the documented default) → immediate `TypeError` on line 241 `distance_matrix /= alpha` because `alpha` is `None`. Change `_hdbscan_brute(X, min_samples=5, alpha=None, ...)` to `alpha=1.0` so the signature matches its own docstring and its sibling `_hdbscan_prims`.

<details><summary>verbatim finding</summary>

```
### F104 — `alpha=None` default in `_hdbscan_brute` signature drifts from its docstring `default=1.0` [out-of-theme]
severity: medium
evidence: `sklearn/cluster/_hdbscan/hdbscan.py:161` declares `alpha=None` in the `_hdbscan_brute` signature, but the docstring at line 183 states `alpha : float, default=1.0`. When the caller path in `fit` (line 767) passes `self.alpha` (which is the constructor default `1.0`), `_hdbscan_brute` unconditionally runs `distance_matrix /= alpha` at line 241. If any code path ever invokes `_hdbscan_brute` relying on the documented default (as `_hdbscan_prims` in fact does, using `alpha=1.0`), the None default raises `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`.
scenario: "Any refactor that calls `_hdbscan_brute(X)` without explicitly passing `alpha=1.0` (relying on the documented default) → immediate `TypeError` on line 241 `distance_matrix /= alpha` because `alpha` is `None`."
contract: Change `_hdbscan_brute(X, min_samples=5, alpha=None, ...)` to `alpha=1.0` so the signature matches its own docstring and its sibling `_hdbscan_prims`.
instances: single-instance
```
</details>

### [MEDIUM] `_hdbscan_brute` default `alpha=None` triggers `TypeError` on `/=`

New caller invokes `_hdbscan_brute(X)` relying on the documented default `alpha=1.0` → `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'` (dense) or scipy `TypeError` (sparse) Change the default to `alpha=1.0` (matching both the docstring at line 183 and the `_hdbscan_prims` sibling default).

<details><summary>verbatim finding</summary>

```
### F119 — `_hdbscan_brute` default `alpha=None` triggers `TypeError` on `/=`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`; sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally does `distance_matrix /= alpha`. Called from `HDBSCAN.fit` at line 767 which always passes `alpha=self.alpha` (validated `> 0`), so today the default is unreachable, but the default value is a live foot-gun for any future caller.
scenario: "New caller invokes `_hdbscan_brute(X)` relying on the documented default `alpha=1.0` → `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'` (dense) or scipy `TypeError` (sparse)"
contract: Change the default to `alpha=1.0` (matching both the docstring at line 183 and the `_hdbscan_prims` sibling default).
instances: single-instance
```
</details>

### [MEDIUM] `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default

User instantiates `HDBSCAN()` expecting single-threaded behavior per the docs → gets 4-thread parallelism unpredictably; also inconsistent with every other sklearn clusterer's convention Set `n_jobs=None` in the `__init__` signature to match the documented default and sklearn-wide convention.

<details><summary>verbatim finding</summary>

```
### F120 — `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in the signature; sklearn/cluster/_hdbscan/hdbscan.py:492-496 docstring: "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context." A default of `4` is neither `None` nor matches any other sklearn clusterer (DBSCAN, OPTICS use `n_jobs=None`).
scenario: "User instantiates `HDBSCAN()` expecting single-threaded behavior per the docs → gets 4-thread parallelism unpredictably; also inconsistent with every other sklearn clusterer's convention"
contract: Set `n_jobs=None` in the `__init__` signature to match the documented default and sklearn-wide convention.
instances: single-instance
```
</details>

### [MEDIUM] `_get_finite_row_indices` sparse branch surprises on non-LIL input

sparse feature matrix (not distance matrix) with an explicit `nan` stored in a row but implicit zeros elsewhere → the row is dropped, but a row with sparsity-implied zeros that 'should' be non-finite is kept. Silent surprising side effect due to a name that overstates the semantics. Rename or scope the helper (e.g. `_get_finite_row_indices_of_stored_entries`) and document that only stored values are inspected for sparse inputs.

<details><summary>verbatim finding</summary>

```
### F136 — `_get_finite_row_indices` sparse branch surprises on non-LIL input
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 — for sparse input the function calls `matrix.tolil().data` and iterates rows; every call on a CSR matrix silently converts (LIL construction cost) and iterates the explicit stored values only. Rows without stored `nan`/`inf` in non-stored positions are treated as finite. The function name promises "purely finite rows" but implicit zeros are considered finite by construction (which is correct in a distance-matrix context but not stated). The function is called on both sparse feature matrices and sparse precomputed distance matrices (via `fit`).
scenario: "sparse feature matrix (not distance matrix) with an explicit `nan` stored in a row but implicit zeros elsewhere → the row is dropped, but a row with sparsity-implied zeros that 'should' be non-finite is kept. Silent surprising side effect due to a name that overstates the semantics."
contract: Rename or scope the helper (e.g. `_get_finite_row_indices_of_stored_entries`) and document that only stored values are inspected for sparse inputs.
instances: single-instance
```
</details>

### [MEDIUM] `_weighted_cluster_center` docstring name misrepresents medoid computation

User relies on the documented medoid semantics → gets a probability-tilted result that shifts toward points that are far from low-probability samples, without any documentation of that behavior Either drop `strength` from the medoid computation to match the documented "minimizes distance to all other points" definition, or clearly document the weighting scheme.

<details><summary>verbatim finding</summary>

```
### F139 — `_weighted_cluster_center` docstring name misrepresents medoid computation
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:915-919 — for medoid: `dist_mat = pairwise_distances(data, metric=..., ...); dist_mat = dist_mat * strength; medoid_index = np.argmin(dist_mat.sum(axis=1))`. Here `strength` (`probabilities_[mask]`, shape `(n,)`) broadcasts across columns of `dist_mat`, weighting each source-point's contribution rather than each candidate's — that is neither the traditional medoid nor a standard weighted-medoid definition, yet the method name is `_weighted_cluster_center` and the docstring at line 517 says "the point in the fitted data which minimizes the distance to all other points in the cluster."
scenario: "User relies on the documented medoid semantics → gets a probability-tilted result that shifts toward points that are far from low-probability samples, without any documentation of that behavior"
contract: Either drop `strength` from the medoid computation to match the documented "minimizes distance to all other points" definition, or clearly document the weighting scheme.
instances: single-instance
```
</details>

### [MEDIUM] `remap_single_linkage_tree` iterates a `set` for positional assignment

Two runs with `PYTHONHASHSEED` differing → `self._single_linkage_tree_` has the same rows but in different order, so any downstream code that indexes by row position produces different results Convert to a sorted list before iteration, e.g. `for i, outlier in enumerate(sorted(non_finite)):`, and change the parameter contract to accept an ordered container.

<details><summary>verbatim finding</summary>

```
### F140 — `remap_single_linkage_tree` iterates a `set` for positional assignment
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:388 — `for i, outlier in enumerate(non_finite):` where `non_finite` is passed as `set(infinite_index + missing_index)` (line 838). Sets have no defined iteration order across Python runs; the loop writes rows in that order and assigns `last_cluster_id += 1` / `last_cluster_size += 1` per row, so the encoded merge order and the row-to-sample mapping are non-deterministic across processes.
scenario: "Two runs with `PYTHONHASHSEED` differing → `self._single_linkage_tree_` has the same rows but in different order, so any downstream code that indexes by row position produces different results"
contract: Convert to a sorted list before iteration, e.g. `for i, outlier in enumerate(sorted(non_finite)):`, and change the parameter contract to accept an ordered container.
instances: single-instance
```
</details>

### [MEDIUM] `dbscan_clustering` docstring promises `-3` handling that the code never performs [out-of-theme]

User calls `fit` with `metric='precomputed'` then `dbscan_clustering(...)` → returned labels do not contain the `-3` value the docstring promises for missing samples Explicitly propagate missing-sample state independent of `self.labels_` so `dbscan_clustering` always emits `-3` where the docstring promises it.

<details><summary>verbatim finding</summary>

```
### F141 — `dbscan_clustering` docstring promises `-3` handling that the code never performs [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:955-958 documents that returned labels include "-3" for missing samples; lines 960-969 compute `labels = labelling_at_cut(self._single_linkage_tree_, cut_distance, min_cluster_size)`, whose result has length `n_samples` of the tree *including* any remapped outlier rows (via `remap_single_linkage_tree`), then overwrites `labels[missing_index]` with `-3` and `labels[infinite_index]` with `-2` using boolean masks derived from `self.labels_`. But `labels` from `labelling_at_cut` has length equal to the tree's `n_samples`, which after remapping equals `self._raw_data.shape[0]` — this coincidence is undocumented and unenforced; if `fit` was called with `metric='precomputed'`, no remap happens, `_raw_data` is undefined, and this method still runs — but `self.labels_` was never populated with `-3`, so the missing/infinite overwrites become no-ops silently.
scenario: "User calls `fit` with `metric='precomputed'` then `dbscan_clustering(...)` → returned labels do not contain the `-3` value the docstring promises for missing samples"
contract: Explicitly propagate missing-sample state independent of `self.labels_` so `dbscan_clustering` always emits `-3` where the docstring promises it.
instances: single-instance
```
</details>

### [MEDIUM] `_hdbscan_prims` docstring documents parameters that do not exist and omits ones that do

Developer maintaining the file reads the docstring and adds precomputed-handling logic gated by `copy` → introduces dead code path because `copy` is never accepted; or omits documenting `algo`/`leaf_size` in downstream references, breaking user understanding The docstring's Parameters block must exactly match the actual signature — drop the `copy` block, drop the "metric='precomputed'" phrasing, add entries for `algo` and `leaf_size`, and correct the `min_samples` default.

<details><summary>verbatim finding</summary>

```
### F168 — `_hdbscan_prims` docstring documents parameters that do not exist and omits ones that do
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-327 — signature is `(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)` but docstring at :313 documents a non-existent `copy` parameter (with "Currently, it only applies when `metric="precomputed"`" — which this function never handles), and never documents `algo` or `leaf_size`. Also `min_samples : int, default=None` at :290 mismatches the actual `min_samples=5`.
scenario: "Developer maintaining the file reads the docstring and adds precomputed-handling logic gated by `copy` → introduces dead code path because `copy` is never accepted; or omits documenting `algo`/`leaf_size` in downstream references, breaking user understanding"
contract: The docstring's Parameters block must exactly match the actual signature — drop the `copy` block, drop the "metric='precomputed'" phrasing, add entries for `algo` and `leaf_size`, and correct the `min_samples` default.
instances: single-instance
```
</details>

### [MEDIUM] `_brute_mst` docstring parameter name and `min_samples` default are wrong

Reader relies on documented parameter name/default → uses wrong keyword or assumes None is accepted, hitting TypeError at call time Rename the parameter in the docstring to `mutual_reachability` and remove the `default=None` annotation to match the real signature.

<details><summary>verbatim finding</summary>

```
### F169 — `_brute_mst` docstring parameter name and `min_samples` default are wrong
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82-97 — the parameter section names the array `mututal_reachability_graph:` (typo, and does not match the actual argument `mutual_reachability`), and declares `min_samples : int, default=None` while the signature `_brute_mst(mutual_reachability, min_samples)` has no default at all.
scenario: "Reader relies on documented parameter name/default → uses wrong keyword or assumes None is accepted, hitting TypeError at call time"
contract: Rename the parameter in the docstring to `mutual_reachability` and remove the `default=None` annotation to match the real signature.
instances: single-instance
```
</details>

### [MEDIUM] `_hdbscan_brute` documents `alpha : float, default=1.0` but signature default is `None`

User invokes _hdbscan_brute without alpha per docstring's stated default → TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType' Change the signature to `alpha=1.0` so it matches the documented default and the unconditional `distance_matrix /= alpha` in the body.

<details><summary>verbatim finding</summary>

```
### F170 — `_hdbscan_brute` documents `alpha : float, default=1.0` but signature default is `None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 sets `alpha=None`; docstring at line 183 asserts `alpha : float, default=1.0`. In the body at line 241 the code unconditionally runs `distance_matrix /= alpha`, so calling with the documented default `1.0` works, but calling with the actual signature default (`None`) will raise a TypeError.
scenario: "User invokes _hdbscan_brute without alpha per docstring's stated default → TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'"
contract: Change the signature to `alpha=1.0` so it matches the documented default and the unconditional `distance_matrix /= alpha` in the body.
instances: single-instance
```
</details>

### [MEDIUM] `HDBSCAN.fit` remap comment states wrong outlier labels

Developer reviewing the remap logic trusts the comment and later refactors labelling to keep the '-1/-2' convention → silently breaks the documented `-2`/`-3` contract seen by end users Update the comment to state that `np.inf` is mapped to `-2` and `np.nan` to `-3`, matching `_OUTLIER_ENCODING` and the class docstring.

<details><summary>verbatim finding</summary>

```
### F176 — `HDBSCAN.fit` remap comment states wrong outlier labels
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." but the code (using `_OUTLIER_ENCODING`) maps `np.inf → -2` (`"infinite"` label) and `np.nan → -3` (`"missing"` label), matching the class-level docstring at sklearn/cluster/_hdbscan/hdbscan.py:534-537.
scenario: "Developer reviewing the remap logic trusts the comment and later refactors labelling to keep the '-1/-2' convention → silently breaks the documented `-2`/`-3` contract seen by end users"
contract: Update the comment to state that `np.inf` is mapped to `-2` and `np.nan` to `-3`, matching `_OUTLIER_ENCODING` and the class docstring.
instances: single-instance
```
</details>

### [MEDIUM] `_get_finite_row_indices` densifies sparse matrix via `.tolil()` and a Python `for` loop

User passes a large CSR precomputed distance matrix with `metric != 'precomputed'` but containing non-finite entries → the code converts CSR→LIL (an O(nnz) allocation of Python lists per row) and pays a Python-level per-row loop, in addition to the earlier `X.sum(axis=1)` on line 721 which already yields non-finite row markers. Compute finite-row indices for sparse input directly from the already-computed `reduced_X = X.sum(axis=1)` (i.e., `~(np.isnan(reduced_X) | np.isinf(reduced_X))`) rather than densifying via LIL.

<details><summary>verbatim finding</summary>

```
### F200 — `_get_finite_row_indices` densifies sparse matrix via `.tolil()` and a Python `for` loop
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 — `row_indices = np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])`. `matrix.tolil()` builds a full LIL representation and the comprehension iterates row-by-row in Python.
scenario: "User passes a large CSR precomputed distance matrix with `metric != 'precomputed'` but containing non-finite entries → the code converts CSR→LIL (an O(nnz) allocation of Python lists per row) and pays a Python-level per-row loop, in addition to the earlier `X.sum(axis=1)` on line 721 which already yields non-finite row markers."
contract: Compute finite-row indices for sparse input directly from the already-computed `reduced_X = X.sum(axis=1)` (i.e., `~(np.isnan(reduced_X) | np.isinf(reduced_X))`) rather than densifying via LIL.
instances: single-instance
```
</details>

### [MEDIUM] `_weighted_cluster_center` recomputes full pairwise distance matrix per cluster (medoid path)

User sets `store_centers='medoid'` (or `'both'`) on a fit with many clusters → total medoid computation is O(∑|cluster|²) done serially, ignoring `self.n_jobs`, and each iteration re-imports/re-dispatches through pairwise_distances even though the metric and params are identical, missing the opportunity to pass `n_jobs=self.n_jobs`. Pass `n_jobs=self.n_jobs` to the `pairwise_distances` call in `_weighted_cluster_center`, and short-circuit for clusters small enough that the medoid can be computed trivially.

<details><summary>verbatim finding</summary>

```
### F207 — `_weighted_cluster_center` recomputes full pairwise distance matrix per cluster (medoid path)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:907-920 — inside `for idx in range(n_clusters)`: `dist_mat = pairwise_distances(data, metric=self.metric, **self._metric_params)` where `data = X[mask]`. For each cluster the full O(|cluster|²) pairwise-distance matrix is computed sequentially without any `n_jobs` parallelism.
scenario: "User sets `store_centers='medoid'` (or `'both'`) on a fit with many clusters → total medoid computation is O(∑|cluster|²) done serially, ignoring `self.n_jobs`, and each iteration re-imports/re-dispatches through pairwise_distances even though the metric and params are identical, missing the opportunity to pass `n_jobs=self.n_jobs`."
contract: Pass `n_jobs=self.n_jobs` to the `pairwise_distances` call in `_weighted_cluster_center`, and short-circuit for clusters small enough that the medoid can be computed trivially.
instances: single-instance
```
</details>

### [MEDIUM] `n_jobs` default is a hard-coded `4`, ignoring joblib parallel context and per-machine CPU count

user runs on a 32-core box or inside a `joblib.parallel_backend` → HDBSCAN silently uses 4 workers rather than respecting the documented default, wasting available cores and diverging from every other scikit-learn estimator's `n_jobs=None` convention. Set `n_jobs=None` as the default in `__init__`, matching the docstring and the sklearn-wide convention.

<details><summary>verbatim finding</summary>

```
### F209 — `n_jobs` default is a hard-coded `4`, ignoring joblib parallel context and per-machine CPU count
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`; docstring at lines 486-490 states "`None` means 1 unless in a :obj:`joblib.parallel_backend` context. `-1` means using all processors." The default therefore contradicts the documented `None` semantics and hard-caps concurrency at 4 regardless of environment.
scenario: "user runs on a 32-core box or inside a `joblib.parallel_backend` → HDBSCAN silently uses 4 workers rather than respecting the documented default, wasting available cores and diverging from every other scikit-learn estimator's `n_jobs=None` convention."
contract: Set `n_jobs=None` as the default in `__init__`, matching the docstring and the sklearn-wide convention.
instances: single-instance
```
</details>

### [MEDIUM] New test module misspelled `test_reachibility.py` prevents rediscovery by name and mirrors the misspelling in production code

Contributor greps for `test_reachability` or configures a CI filter such as `pytest -k reachability` → the new test module is silently skipped; naming inconsistency also complicates future refactors and packaging tooling that infer test names from the module under test Rename the file to `test_reachability.py` and rename all `mutual_reachibility_distance` identifiers in `_reachability.pyx` to `mutual_reachability_distance` so file, symbol and module names agree.

<details><summary>verbatim finding</summary>

```
### F236 — New test module misspelled `test_reachibility.py` prevents rediscovery by name and mirrors the misspelling in production code
severity: medium
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 (file name); the module it tests is spelled correctly `_reachability.py`. The internal identifier `mutual_reachibility_distance` at sklearn/cluster/_hdbscan/_reachability.pyx:133,150,155,188,212,215 is also misspelled ("reachibility" vs "reachability")
scenario: "Contributor greps for `test_reachability` or configures a CI filter such as `pytest -k reachability` → the new test module is silently skipped; naming inconsistency also complicates future refactors and packaging tooling that infer test names from the module under test"
contract: Rename the file to `test_reachability.py` and rename all `mutual_reachibility_distance` identifiers in `_reachability.pyx` to `mutual_reachability_distance` so file, symbol and module names agree.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:150, sklearn/cluster/_hdbscan/_reachability.pyx:155, sklearn/cluster/_hdbscan/_reachability.pyx:188, sklearn/cluster/_hdbscan/_reachability.pyx:212, sklearn/cluster/_hdbscan/_reachability.pyx:215]
```
</details>

### [MEDIUM] `_hdbscan` subpackage cimports from parent package's private Cython (leaks internal UnionFind cross-package)

The `_hdbscan` subpackage needs a union-find data structure → author extracts a `.pxd` for `_hierarchical_fast` exposing UnionFind's private state (attributes and cdef methods) publicly to the whole tree, promoting an implementation-detail class of the `_agglomerative`/`_hierarchical` code path into a de facto library-wide Cython utility Move the shared `UnionFind` cdef class into a neutral utility module (e.g. `sklearn/utils/_union_find.pxd`) and have both `_hierarchical_fast.pyx` and `_hdbscan/_linkage.pyx` cimport from it, so the sibling subpackage does not depend on internals of another cluster module

<details><summary>verbatim finding</summary>

```
### F220 — `_hdbscan` subpackage cimports from parent package's private Cython (leaks internal UnionFind cross-package)
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39 — `from ...cluster._hierarchical_fast cimport UnionFind`; sklearn/cluster/_hierarchical_fast.pxd:3-9 — newly extracted `.pxd` exposes `UnionFind` (with `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` and cdef methods `union`/`fast_find`) at the module boundary specifically to satisfy the new subpackage's cimport
scenario: "The `_hdbscan` subpackage needs a union-find data structure → author extracts a `.pxd` for `_hierarchical_fast` exposing UnionFind's private state (attributes and cdef methods) publicly to the whole tree, promoting an implementation-detail class of the `_agglomerative`/`_hierarchical` code path into a de facto library-wide Cython utility"
contract: Move the shared `UnionFind` cdef class into a neutral utility module (e.g. `sklearn/utils/_union_find.pxd`) and have both `_hierarchical_fast.pyx` and `_hdbscan/_linkage.pyx` cimport from it, so the sibling subpackage does not depend on internals of another cluster module
instances: [sklearn/cluster/_hierarchical_fast.pxd:3-9, sklearn/cluster/_hdbscan/_linkage.pyx:39]
```
</details>

### [MEDIUM] `UnionFind.union` / `UnionFind.fast_find` pxd declaration is `noexcept` but pyx implementation omits the keyword

A Python exception is raised inside `union`/`fast_find` at runtime (e.g. IndexError from `self.parent[m] = self.next_label` when sizes mismatch) → the `noexcept` declaration causes the exception to be swallowed with only an unraisable warning rather than propagated to the caller. The pxd declaration and the pyx implementation must carry the identical `noexcept` qualifier on both `union` and `fast_find`.

<details><summary>verbatim finding</summary>

```
### F221 — `UnionFind.union` / `UnionFind.fast_find` pxd declaration is `noexcept` but pyx implementation omits the keyword
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`. sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` (no `noexcept`) and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` (no `noexcept`). The pxd was newly extracted in this PR (see hunks/sklearn_cluster__hierarchical_fast.pxd.diff), which lifts the field declarations out of the class body in the pyx (hunks/sklearn_cluster__hierarchical_fast.pyx.diff removes them) but silently strengthens the exception-handling contract with `noexcept` while the implementation still allows exceptions to propagate. [out-of-theme]
scenario: "A Python exception is raised inside `union`/`fast_find` at runtime (e.g. IndexError from `self.parent[m] = self.next_label` when sizes mismatch) → the `noexcept` declaration causes the exception to be swallowed with only an unraisable warning rather than propagated to the caller."
contract: The pxd declaration and the pyx implementation must carry the identical `noexcept` qualifier on both `union` and `fast_find`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
```
</details>

### [MEDIUM] `.pxd` declares `noexcept` but `.pyx` implementation omits it, causing Cython 3 signature mismatch

Building with Cython 3.x → 'Function signature does not match previous declaration' warning (or hard error if `-Werror` / cython-lint-strict is enforced), and the two functions carry different exception-checking semantics depending on which declaration Cython chose. The `.pyx` method signatures for `UnionFind.union` and `UnionFind.fast_find` must be updated to include the `noexcept` qualifier so they match the newly added `.pxd` declarations exactly.

<details><summary>verbatim finding</summary>

```
### F235 — `.pxd` declares `noexcept` but `.pyx` implementation omits it, causing Cython 3 signature mismatch
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, while sklearn/cluster/_hierarchical_fast.pyx:331 and :339 define them as `cdef void union(self, intp_t m, intp_t n):` and `cdef intp_t fast_find(self, intp_t n):` with no `noexcept` qualifier.
scenario: "Building with Cython 3.x → 'Function signature does not match previous declaration' warning (or hard error if `-Werror` / cython-lint-strict is enforced), and the two functions carry different exception-checking semantics depending on which declaration Cython chose."
contract: The `.pyx` method signatures for `UnionFind.union` and `UnionFind.fast_find` must be updated to include the `noexcept` qualifier so they match the newly added `.pxd` declarations exactly.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
```
</details>

### [MEDIUM] `test_dbscan_clustering_outlier_data` computes `clean_idx` by summing ndarrays instead of concatenating them, silently masking the wrong indices

Test run with any parametrized `cut_distance` → `clean_model.fit(X_outlier[clean_idx])` sees an infinite-valued row and behaves as another test case; the assertion at line 215 is comparing wrong slices Concatenate index arrays with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or use `.tolist()` and list concatenation) before wrapping in `set(...)`.

<details><summary>verbatim finding</summary>

```
### F5 — `test_dbscan_clustering_outlier_data` computes `clean_idx` by summing ndarrays instead of concatenating them, silently masking the wrong indices
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. With `missing_labels_idx == np.array([2,5])` and `infinite_labels_idx == np.array([0])`, NumPy broadcasts the `+` operator to element-wise addition producing `np.array([2, 5])`, so the "outlier" set becomes `{2, 5}` — the infinite index `0` is not excluded. `clean_idx` therefore contains index 0 (an `np.inf` row), and the subsequent `assert_array_equal(clean_labels, labels[clean_idx])` may fail or pass only by accident.
scenario: "Test run with any parametrized `cut_distance` → `clean_model.fit(X_outlier[clean_idx])` sees an infinite-valued row and behaves as another test case; the assertion at line 215 is comparing wrong slices"
contract: Concatenate index arrays with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or use `.tolist()` and list concatenation) before wrapping in `set(...)`.
instances: single-instance
```
</details>

### [MEDIUM] `test_hdbscan_precomputed_non_brute` never exercises the intended check because `algorithm="prims_kdtree"`/`"prims_balltree"` is rejected by parameter validation

Regression is introduced that silently allows `algorithm='kdtree'` with `metric='precomputed'` → this test still passes because it uses a bogus algorithm name and only tests parameter validation Change the parametrization to `["kdtree", "balltree"]` so the intended precomputed-vs-tree guard is actually exercised.

<details><summary>verbatim finding</summary>

```
### F6 — `test_hdbscan_precomputed_non_brute` never exercises the intended check because `algorithm="prims_kdtree"`/`"prims_balltree"` is rejected by parameter validation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` uses algorithm names that are not in the allowed StrOptions set `{"auto","brute","kdtree","balltree"}` (see sklearn/cluster/_hdbscan/hdbscan.py:629-638). `_validate_params()` raises `InvalidParameterError` (a `ValueError` subclass) before the "precomputed + tree" guard at hdbscan.py:772-783 is ever hit, so the test passes vacuously and does not validate what its docstring claims.
scenario: "Regression is introduced that silently allows `algorithm='kdtree'` with `metric='precomputed'` → this test still passes because it uses a bogus algorithm name and only tests parameter validation"
contract: Change the parametrization to `["kdtree", "balltree"]` so the intended precomputed-vs-tree guard is actually exercised.
instances: single-instance
```
</details>

### [MEDIUM] `test_hdbscan_centers` uses `rtol=1`, effectively disabling the centroid/medoid accuracy check for non-zero centers

centroid computation regresses for any non-zero-centered cluster → test still passes because rtol=1 permits ~100% relative error Drop `rtol=1` and use only a tight `atol` (or `rtol` ≪ 1 like `rtol=0.05`) so the check reflects the "accurate to the data" contract.

<details><summary>verbatim finding</summary>

```
### F55 — `test_hdbscan_centers` uses `rtol=1`, effectively disabling the centroid/medoid accuracy check for non-zero centers
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:319-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)` and same for `medoid`. With `rtol=1`, the tolerance for `center=(3.0, 3.0)` becomes `atol + rtol*|3| = 3.05`, i.e. any value in `[-0.05, 6.05]` passes. Only the `(0, 0)` center is effectively constrained (to `atol=0.05`). The stated intent in the docstring is that centers "are accurate to the data".
scenario: "centroid computation regresses for any non-zero-centered cluster → test still passes because rtol=1 permits ~100% relative error"
contract: Drop `rtol=1` and use only a tight `atol` (or `rtol` ≪ 1 like `rtol=0.05`) so the check reflects the "accurate to the data" contract.
instances: [sklearn/cluster/tests/test_hdbscan.py:319, sklearn/cluster/tests/test_hdbscan.py:320]
```
</details>

### [MEDIUM] `test_hdbscan_min_cluster_size` filter excludes only `-1`, ignoring `-2`/`-3` outlier labels

input triggers infinite/missing outlier labels (-2/-3) → test crashes with numpy bincount error instead of validating minimum cluster size; today it passes only because the `X` fixture is finite and no outlier labels appear Filter with `label not in OUTLIER_SET` (or `label >= 0`) before `bincount`.

<details><summary>verbatim finding</summary>

```
### F56 — `test_hdbscan_min_cluster_size` filter excludes only `-1`, ignoring `-2`/`-3` outlier labels
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:261-263 — `true_labels = [label for label in labels if label != -1]` then `np.bincount(true_labels)`. `np.bincount` requires non-negative integers and will raise `ValueError` if `-2`/`-3` (defined in `_OUTLIER_ENCODING`) are present. The correct filter would use `OUTLIER_SET` (defined at line 37) which the file already imports/constructs for exactly this purpose.
scenario: "input triggers infinite/missing outlier labels (-2/-3) → test crashes with numpy bincount error instead of validating minimum cluster size; today it passes only because the `X` fixture is finite and no outlier labels appear"
contract: Filter with `label not in OUTLIER_SET` (or `label >= 0`) before `bincount`.
instances: single-instance
```
</details>

### [MEDIUM] `set(missing_labels_idx + infinite_labels_idx)` uses numpy broadcasting, not concatenation

Infinite-outlier index 0 is never removed from `clean_idx` → `clean_model.fit(X_outlier[clean_idx])` is fit on data that still contains an `np.inf` row → the “clean vs full” equivalence assertion (`assert_array_equal(clean_labels, labels[clean_idx])`) does not actually verify what the test docstring claims (that removing outliers upfront yields the same labels for the finite points). The test will silently miss regressions in the finite-only path any time infinite outliers exist. Concatenate with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or use lists and `+`) before wrapping in `set(...)`.

<details><summary>verbatim finding</summary>

```
### F57 — `set(missing_labels_idx + infinite_labels_idx)` uses numpy broadcasting, not concatenation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))` where both operands are `np.flatnonzero(...)` results (ndarrays). With `missing_labels_idx = [2, 5]` and `infinite_labels_idx = [0]`, the `+` broadcasts the length-1 array over the length-2 array, producing `[2, 5]` instead of the intended concatenation `[2, 5, 0]`. Confirmed by running `np.array([2,5]) + np.array([0])` → `[2, 5]`.
scenario: "Infinite-outlier index 0 is never removed from `clean_idx` → `clean_model.fit(X_outlier[clean_idx])` is fit on data that still contains an `np.inf` row → the “clean vs full” equivalence assertion (`assert_array_equal(clean_labels, labels[clean_idx])`) does not actually verify what the test docstring claims (that removing outliers upfront yields the same labels for the finite points). The test will silently miss regressions in the finite-only path any time infinite outliers exist."
contract: Concatenate with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or use lists and `+`) before wrapping in `set(...)`.
instances: single-instance
```
</details>

### [MEDIUM] `test_labelling_thresholding` asserts equal counts of noise, not equal noise positions

The threshold branch in `_do_labelling` swaps positions but preserves the noise count → this test still passes. Assert positional equality of the noise mask, e.g. `assert_array_equal(labels == -1, condensed_tree['value'] < threshold)`.

<details><summary>verbatim finding</summary>

```
### F61 — `test_labelling_thresholding` asserts equal counts of noise, not equal noise positions
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:519-520 and 532-533 — the assertions are `assert sum(num_noise) == sum(labels == -1)`, where `num_noise` is the boolean mask `condensed_tree["value"] < 1` (or `< MAX_LAMBDA`). Two arrays with wholly different positions of `-1` but the same count satisfy this. The test's stated purpose is to verify the correct per-sample thresholding of lambda values, not just the cardinality of noisy points.
scenario: "The threshold branch in `_do_labelling` swaps positions but preserves the noise count → this test still passes."
contract: Assert positional equality of the noise mask, e.g. `assert_array_equal(labels == -1, condensed_tree['value'] < threshold)`.
instances: [sklearn/cluster/tests/test_hdbscan.py:519-520, sklearn/cluster/tests/test_hdbscan.py:532-533]
```
</details>

### [MEDIUM] `copy=False` behavior of `HDBSCAN` is untested; only `copy=True` non-mutation is asserted

A regression silently makes `copy=False` allocate copies (or `copy=True` mutate) → tests do not detect it. Add explicit fit-time assertions for `copy=False`: (i) the input distance matrix is mutated in place, and (ii) the resulting labels equal those from `copy=True`.

<details><summary>verbatim finding</summary>

```
### F62 — `copy=False` behavior of `HDBSCAN` is untested; only `copy=True` non-mutation is asserted
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:79-92 — the only `assert_allclose(D, D_original)` guarding the copy contract is executed with `copy=True`. There is no test that fits with `copy=False` and verifies either (a) that the caller's array is mutated (proving `copy=False` is the "no-copy" path) or (b) that fit results are identical between `copy=True` and `copy=False`. The `copy` parameter is documented (hdbscan.py:521-526) as a first-class user option controlling in-place modification of precomputed inputs and the brute-force reachability graph, but the default path is completely uncovered.
scenario: "A regression silently makes `copy=False` allocate copies (or `copy=True` mutate) → tests do not detect it."
contract: Add explicit fit-time assertions for `copy=False`: (i) the input distance matrix is mutated in place, and (ii) the resulting labels equal those from `copy=True`.
instances: single-instance
```
</details>

### [MEDIUM] `test_hdbscan_precomputed_non_brute` asserts on the wrong error path via an invalid `algorithm` string [out-of-theme]

A future refactor removes the precomputed-vs-tree ValueError branches at hdbscan.py:772-783 → this test still passes silently, giving false confidence in the coverage of that contract The test MUST use a valid algorithm from the StrOptions set (`"kdtree"` or `"balltree"`) so that the intended precomputed-vs-tree branch is exercised.

<details><summary>verbatim finding</summary>

```
### F81 — `test_hdbscan_precomputed_non_brute` asserts on the wrong error path via an invalid `algorithm` string [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 sets `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"`, `"prims_balltree"`), but the `algorithm` StrOptions at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only permits `{"auto", "brute", "kdtree", "balltree"}`. `_validate_params()` (called at hdbscan.py:697) therefore raises `InvalidParameterError` (a `ValueError` subclass) before the intended `"precomputed + tree"` check at lines 772-783 is ever reached. The test passes for a reason unrelated to what its docstring at lines 279-281 claims to verify.
scenario: "A future refactor removes the precomputed-vs-tree ValueError branches at hdbscan.py:772-783 → this test still passes silently, giving false confidence in the coverage of that contract"
contract: The test MUST use a valid algorithm from the StrOptions set (`"kdtree"` or `"balltree"`) so that the intended precomputed-vs-tree branch is exercised.
instances: single-instance
```
</details>

### [MEDIUM] `test_hdbscan_precomputed_non_brute` passes an invalid algorithm name and no longer tests the documented behavior [out-of-theme]

Test claims to verify 'HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm', but it triggers a parameter-validation error instead → the actual precomputed+kdtree/balltree code path is untested; a regression that silently accepts precomputed data with `algorithm='kdtree'` would pass CI. Change the test to pass a valid tree-based algorithm name (`f"{tree}tree"`, i.e. `"kdtree"`/`"balltree"`) so the ValueError from the precomputed+non-brute branch is what is asserted, and match against that branch's actual error message.

<details><summary>verbatim finding</summary>

```
### F103 — `test_hdbscan_precomputed_non_brute` passes an invalid algorithm name and no longer tests the documented behavior [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` with `tree in {"kd","ball"}` produces `algorithm="prims_kdtree"` / `"prims_balltree"`, but the estimator's parameter constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-638 restricts `algorithm` to `{"auto","brute","kdtree","balltree"}`. `fit` raises `InvalidParameterError` from `_validate_params()` (line 697) before any precomputed+non-brute logic is reached.
scenario: "Test claims to verify 'HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm', but it triggers a parameter-validation error instead → the actual precomputed+kdtree/balltree code path is untested; a regression that silently accepts precomputed data with `algorithm='kdtree'` would pass CI."
contract: Change the test to pass a valid tree-based algorithm name (`f"{tree}tree"`, i.e. `"kdtree"`/`"balltree"`) so the ValueError from the precomputed+non-brute branch is what is asserted, and match against that branch's actual error message.
instances: single-instance
```
</details>

### [MEDIUM] `test_hdbscan_precomputed_non_brute` asserts the wrong error path

developer removes/relaxes the algorithm-name enum → test still passes because it never reaches the metric-incompatibility check, so the actual invariant is silently unguarded Use the valid names `"kdtree"`/`"balltree"` in the parametrization and assert on the metric-incompatibility error message.

<details><summary>verbatim finding</summary>

```
### F133 — `test_hdbscan_precomputed_non_brute` asserts the wrong error path
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:277-284 — the test constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` where `algorithm` is `"prims_kdtree"`/`"prims_balltree"`. Those values are not in the `StrOptions({"auto","brute","kdtree","balltree"})` constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-638, so the test passes only because `_validate_params` raises `InvalidParameterError` (a `ValueError` subclass) — never exercising the intended `algorithm` vs `metric="precomputed"` incompatibility branch. The test's docstring says it verifies rejection of a tree-based algorithm with precomputed data; it actually verifies rejection of an unknown algorithm string.
scenario: "developer removes/relaxes the algorithm-name enum → test still passes because it never reaches the metric-incompatibility check, so the actual invariant is silently unguarded"
contract: Use the valid names `"kdtree"`/`"balltree"` in the parametrization and assert on the metric-incompatibility error message.
instances: single-instance
```
</details>

### [MEDIUM] `test_hdbscan_centers` accepts `rtol=1` which admits ~100% error

regression that pushes centroids drastically toward the wrong cluster → test still passes; the assertion name promises verification, but the tolerance renders it a lying check. Use a small `rtol` (e.g. `rtol=0.1`) that actually constrains centroid quality on the known blob geometry.

<details><summary>verbatim finding</summary>

```
### F138 — `test_hdbscan_centers` accepts `rtol=1` which admits ~100% error
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:319-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)` with `rtol=1` means a centroid up to `1.0 * |center| + 0.05` off passes; for `center=(3.0,3.0)` that permits a centroid anywhere in `[0, 6]` per coordinate.
scenario: "regression that pushes centroids drastically toward the wrong cluster → test still passes; the assertion name promises verification, but the tolerance renders it a lying check."
contract: Use a small `rtol` (e.g. `rtol=0.1`) that actually constrains centroid quality on the known blob geometry.
instances: single-instance
```
</details>

### [MEDIUM] Test uses ndarray `+` for concatenation, silently exercising broadcasting

future edit changes seed or outlier layout so `missing_labels_idx` and `infinite_labels_idx` have incompatible non-broadcastable shapes → test crashes with a broadcasting error unrelated to the tested behavior; today it silently uses wrong indices in the 'clean' baseline Use `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or `.tolist() + .tolist()`) for index-set arithmetic.

<details><summary>verbatim finding</summary>

```
### F142 — Test uses ndarray `+` for concatenation, silently exercising broadcasting
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` and `infinite_labels_idx` come from `np.flatnonzero(...)` (lines 206/209) and are ndarrays. `ndarray + ndarray` broadcasts elementwise; with shapes `(2,)` and `(1,)` here it happens to return a length-2 array of sums, not the concatenation the reader assumes.
scenario: "future edit changes seed or outlier layout so `missing_labels_idx` and `infinite_labels_idx` have incompatible non-broadcastable shapes → test crashes with a broadcasting error unrelated to the tested behavior; today it silently uses wrong indices in the 'clean' baseline"
contract: Use `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or `.tolist() + .tolist()`) for index-set arithmetic.
instances: single-instance
```
</details>

### [MEDIUM] Test-narration docstring lies about what is being tested (`test_hdbscan_precomputed_non_brute`)

Behavior change removes the precomputed-vs-tree check in fit → test still passes on unrelated grounds, giving false coverage confidence The test must call `HDBSCAN(metric='precomputed', algorithm=f'{tree}tree')` with a valid algorithm name, or the docstring must be rewritten to state that invalid algorithm names raise.

<details><summary>verbatim finding</summary>

```
### F172 — Test-narration docstring lies about what is being tested (`test_hdbscan_precomputed_non_brute`)
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — docstring claims the test checks that HDBSCAN "correctly raises an error when passing precomputed data while requesting a tree-based algorithm", but the algorithm string passed is `prims_kdtree` / `prims_balltree`, and `_parameter_constraints["algorithm"]` in `hdbscan.py:629-638` only accepts `{"auto", "brute", "kdtree", "balltree"}`. The `ValueError` therefore comes from parameter-name validation, not from the precomputed+tree combination the narration promises. The intended combination is never actually exercised.
scenario: "Behavior change removes the precomputed-vs-tree check in fit → test still passes on unrelated grounds, giving false coverage confidence"
contract: The test must call `HDBSCAN(metric='precomputed', algorithm=f'{tree}tree')` with a valid algorithm name, or the docstring must be rewritten to state that invalid algorithm names raise.
instances: single-instance
```
</details>

## LOW severity (148)

### [LOW] Reconciliation summary

Full inventory reconciliation performed across all 19 manifest paths and all plan items → every manifest file is accounted for by either an explicit plan claim, the plan's umbrella scope, or an unsanctioned-change finding above, leaving no orphan paths. Every manifest file must be accounted for by an explicit plan claim, the plan's umbrella scope, or an unsanctioned-change finding above.

<details><summary>verbatim finding</summary>

```
### F6 — Reconciliation summary
severity: low
scenario: "Full inventory reconciliation performed across all 19 manifest paths and all plan items → every manifest file is accounted for by either an explicit plan claim, the plan's umbrella scope, or an unsanctioned-change finding above, leaving no orphan paths."
contract: "Every manifest file must be accounted for by an explicit plan claim, the plan's umbrella scope, or an unsanctioned-change finding above."
evidence: All 19 manifest paths (DIFF_MANIFEST.md:6–30) accounted for as follows:
- Sanctioned by plan umbrella title (F1): `examples/cluster/plot_hdbscan.py`, `sklearn/cluster/_hdbscan/__init__.py`, `sklearn/cluster/_hdbscan/hdbscan.py`, `sklearn/cluster/_hdbscan/tests/__init__.py`, `sklearn/cluster/_hdbscan/tests/test_reachibility.py`, `sklearn/cluster/tests/test_hdbscan.py`, `sklearn/cluster/__init__.py`, `sklearn/utils/estimator_checks.py`, `setup.py`, `examples/cluster/plot_cluster_comparison.py`, `doc/modules/classes.rst`, `doc/modules/clustering.rst`, `doc/whats_new/v1.3.rst`.
- Sanctioned by plan novel changes #1/#2 (F2): `sklearn/cluster/_hdbscan/_linkage.pyx`, `sklearn/cluster/_hdbscan/_reachability.pyx`, `sklearn/cluster/_hdbscan/_tree.pyx`, `sklearn/cluster/_hdbscan/_tree.pxd`.
- Unsanctioned (F4): `sklearn/cluster/_hierarchical_fast.pyx`, `sklearn/cluster/_hierarchical_fast.pxd`.

Unmatched plan requirements: none beyond the historical checklist noted in F5.

instances: [DIFF_MANIFEST.md:6-30]
```
</details>

### [LOW] Test file name misspelled ("reachibility") [out-of-theme]

Developer searches for `test_reachability` → does not find the test file; the misspelling propagates via cargo-cult Rename the file to `test_reachability.py` before further work builds on the misspelling.

<details><summary>verbatim finding</summary>

```
### F125 — Test file name misspelled ("reachibility") [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py (new file per DIFF_MANIFEST.md:26) — filename misspells "reachability" as "reachibility". Note the module under test is `_reachability.pyx` (spelled correctly), and the docstring inside the file references "mutual reachability graph". The misspelling also appears inside `_reachability.pyx:132` (`mutual_reachibility_distance`) and `_reachability.pyx:187`, but the file-name itself is the change-hygiene defect committed by this PR.
scenario: "Developer searches for `test_reachability` → does not find the test file; the misspelling propagates via cargo-cult"
contract: Rename the file to `test_reachability.py` before further work builds on the misspelling.
instances: single-instance
```
</details>

### [LOW] Plan lacks explicit sanctioning for the entire HDBSCAN implementation body

Inverse audit — Inventory A (plan requirements). The plan states three explicit 'novel changes' that were injected within this PR → all other content (the estimator implementation itself, the Cython extension modules, tests, docs, examples, `what's new`, `setup.py` wiring) is only implicitly sanctioned by the umbrella title and by reference to external merged PRs whose content the plan does not enumerate at file granularity, so the audit cannot map individual files in B back to itemized promises in A. A plan-of-record for a 'novel-changes-only' delta PR relies on the reviewer accepting the umbrella statement ('Add HDBSCAN as a new estimator') as blanket sanction for all supporting infrastructure. That is acceptable but should be flagged: the audit cannot map individual files in B back to itemized promises in A.

<details><summary>verbatim finding</summary>

```
### F1 — Plan lacks explicit sanctioning for the entire HDBSCAN implementation body
severity: low
scenario: "Inverse audit — Inventory A (plan requirements). The plan states three explicit 'novel changes' that were injected within this PR → all other content (the estimator implementation itself, the Cython extension modules, tests, docs, examples, `what's new`, `setup.py` wiring) is only implicitly sanctioned by the umbrella title and by reference to external merged PRs whose content the plan does not enumerate at file granularity, so the audit cannot map individual files in B back to itemized promises in A."
contract: "A plan-of-record for a 'novel-changes-only' delta PR relies on the reviewer accepting the umbrella statement ('Add HDBSCAN as a new estimator') as blanket sanction for all supporting infrastructure. That is acceptable but should be flagged: the audit cannot map individual files in B back to itemized promises in A."
evidence:
- Plan PR title: "ENH Add `HDBSCAN` as a new estimator in `sklearn.cluster`" (PLAN.md:4).
- Plan "What does this implement/fix?" section defers to the linked issue #24686 and lists only three *novel* changes (PLAN.md:11–17).
- The mandatory-work checklist in PLAN.md:36–48 references separately-merged sub-PRs (#24857, #24701, #25768, #25826, #25827, #26011, #26096, #26101, #24698, #25538, #25134) — none of which are quoted as file-level requirements the reviewer can match against the manifest here.
- The manifest lists 12 net-new (`A`) source/test/example files totalling >3,000 lines (DIFF_MANIFEST.md:15, 19–26, 29) that implement the estimator body itself.

instances: single-instance

All these fall under class (b) in the audit spec: "changes in B with NO sanctioning requirement in A" — but the plan's umbrella title functions as sanction, so they are informational rather than defective.
```
</details>

### [LOW] Novel change #3 ("Trimmed unused variables") corroborated only in a peripheral file, not in the newly-added HDBSCAN sources

Inverse audit — the plan's third novel change cannot be independently verified from the manifest because it applies only to added files (nothing to diff against) → this is a limitation of 'novel-changes' bookkeeping in a rebase-heavy PR, not a defect, and the plan's checklist entry (#25538 in PLAN.md:47) purportedly covers the work in a prior PR. For added files, 'trimmed' is unverifiable from the diff alone; the plan's checklist entry (#25538 in PLAN.md:47) purportedly covers this work in a prior PR.

<details><summary>verbatim finding</summary>

```
### F3 — Novel change #3 ("Trimmed unused variables") corroborated only in a peripheral file, not in the newly-added HDBSCAN sources
severity: low
scenario: "Inverse audit — the plan's third novel change cannot be independently verified from the manifest because it applies only to added files (nothing to diff against) → this is a limitation of 'novel-changes' bookkeeping in a rebase-heavy PR, not a defect, and the plan's checklist entry (#25538 in PLAN.md:47) purportedly covers the work in a prior PR."
contract: "For added files, 'trimmed' is unverifiable from the diff alone; the plan's checklist entry (#25538 in PLAN.md:47) purportedly covers this work in a prior PR."
evidence:
- Plan promise: "Trimmed unused variables (thanks to Cython linting pre-commit)" (PLAN.md:17).
- The only *deletions* in the manifest touching Cython live in `sklearn/cluster/_hierarchical_fast.pyx` (`+0/-5`, DIFF_MANIFEST.md:28) and consist of moving `cdef` field declarations from the `.pyx` into a new `.pxd` (`sklearn/cluster/_hierarchical_fast.pxd`, +9, DIFF_MANIFEST.md:27), plus removing one blank line — this is a header-split refactor, not variable trimming.
- All HDBSCAN-specific new files (`_linkage.pyx`, `_reachability.pyx`, `_tree.pyx`, `_tree.pxd`) are `A` (added), so any "trimming" happened pre-add and is not visible in the diff.

instances: single-instance
```
</details>

### [LOW] Plan checklist items reference already-merged upstream PRs and cannot be matched to any file in the current manifest

Inverse audit class (a): requirements in A with no implementing change in B → however, these items are explicitly marked complete in prior PRs, so absence from *this* diff is expected and the finding is raised only to acknowledge the audit examined and dismissed them. A checklist of already-merged sub-PRs is not a set of requirements for the current diff; it is provenance. No action.

<details><summary>verbatim finding</summary>

```
### F5 — Plan checklist items reference already-merged upstream PRs and cannot be matched to any file in the current manifest
severity: low
scenario: "Inverse audit class (a): requirements in A with no implementing change in B → however, these items are explicitly marked complete in prior PRs, so absence from *this* diff is expected and the finding is raised only to acknowledge the audit examined and dismissed them."
contract: "A checklist of already-merged sub-PRs is not a set of requirements for the current diff; it is provenance. No action."
evidence:
- Plan checklist items (PLAN.md:36–48) are external PR/issue links: `#24857`, `PR#24701`, `PR#25768`, `PR#25826`, `PR#25827`, `PR#26011`, `PR#26096`, `PR#26101`, `PR#24698`, `PR#25538`, `#25134`.
- All are marked `[x]` (completed) and describe work already merged in prior PRs; none map to a file listed in DIFF_MANIFEST.md.

instances: [PLAN.md:36, PLAN.md:37, PLAN.md:38, PLAN.md:39, PLAN.md:40, PLAN.md:41, PLAN.md:42, PLAN.md:43, PLAN.md:44, PLAN.md:47, PLAN.md:48]
```
</details>

### [LOW] Test filename typo: `test_reachibility.py` for module `_reachability.pyx`

Grep/discoverability workflows that search by module basename (`reachability`) miss the test file entirely → contributors may add duplicate tests elsewhere or believe the module is untested. Rename `test_reachibility.py` to `test_reachability.py` so the test file name matches the module it tests.

<details><summary>verbatim finding</summary>

```
### F70 — Test filename typo: `test_reachibility.py` for module `_reachability.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py — filename is `reachibility` (misspelled), but it tests functions from `_reachability.pyx` (correctly spelled). The docstring at line 30 also spells `mutual_reachibility_distance`, and _reachability.pyx:127,144,182,206 use `mutual_reachibility_distance` — the misspelling propagates into implementation variable names too, but the test file's name is the surfaced defect for tests-and-verification.
scenario: "Grep/discoverability workflows that search by module basename (`reachability`) miss the test file entirely → contributors may add duplicate tests elsewhere or believe the module is untested."
contract: Rename `test_reachibility.py` to `test_reachability.py` so the test file name matches the module it tests.
instances: single-instance
```
</details>

### [LOW] `cluster.HDBSCAN` inserted out of alphabetical order in `doc/modules/classes.rst` [out-of-theme]

Autogenerated API index shows HDBSCAN out of order, breaking the alphabetisation contract that the rest of the block relies on → future maintainers add new estimators next to their alphabetical neighbours and cannot easily locate HDBSCAN in the class index Move the `cluster.HDBSCAN` entry so it appears after `cluster.FeatureAgglomeration` and before `cluster.KMeans` to preserve alphabetical ordering.

<details><summary>verbatim finding</summary>

```
### F242 — `cluster.HDBSCAN` inserted out of alphabetical order in `doc/modules/classes.rst` [out-of-theme]
severity: low
evidence: doc/modules/classes.rst:107 inserts `cluster.HDBSCAN` between `cluster.DBSCAN` and `cluster.FeatureAgglomeration`; the surrounding block is alphabetised (`AffinityPropagation`, `AgglomerativeClustering`, `Birch`, `DBSCAN`, `FeatureAgglomeration`, `KMeans`, ...), and `HDBSCAN` sorts after `FeatureAgglomeration`, not before it
scenario: "Autogenerated API index shows HDBSCAN out of order, breaking the alphabetisation contract that the rest of the block relies on → future maintainers add new estimators next to their alphabetical neighbours and cannot easily locate HDBSCAN in the class index"
contract: Move the `cluster.HDBSCAN` entry so it appears after `cluster.FeatureAgglomeration` and before `cluster.KMeans` to preserve alphabetical ordering.
instances: single-instance
```
</details>

### [LOW] Unrelated trailing-whitespace cleanup bundled with HDBSCAN PR

Blame/history archaeology for Mean Shift docs → unrelated commit surfaces as owning these lines; PR review scope inflated Revert the whitespace-only edits to the Mean Shift documentation sections; keep only the HDBSCAN additions.

<details><summary>verbatim finding</summary>

```
### F124 — Unrelated trailing-whitespace cleanup bundled with HDBSCAN PR
severity: low
evidence: doc/modules/clustering.rst hunks at lines 392-425 (see hunks/doc_modules_clustering.rst.diff:22-45) — three edits that strip trailing spaces from Mean Shift prose ("hill climbing", "density estimation", "small enough and is") unrelated to HDBSCAN. The PR description enumerates three novel changes (typedef swap, `shape[0]→len`, unused-var trim); trailing-whitespace fixes to Mean Shift docs are not sanctioned by the plan.
scenario: "Blame/history archaeology for Mean Shift docs → unrelated commit surfaces as owning these lines; PR review scope inflated"
contract: Revert the whitespace-only edits to the Mean Shift documentation sections; keep only the HDBSCAN additions.
instances: [doc/modules/clustering.rst:399-400, doc/modules/clustering.rst:422, doc/modules/clustering.rst:429]
```
</details>

### [LOW] User-guide prose contains stale "at this staged" grammar and describes wrong noise criterion

Reader learning HDBSCAN from user guide → internalizes an incorrect definition of DBSCAN* noise assignment Fix the typo to "at this stage" and correct the direction of the inequality: "Any points whose core distance is greater than ε are marked as noise.

<details><summary>verbatim finding</summary>

```
### F179 — User-guide prose contains stale "at this staged" grammar and describes wrong noise criterion
severity: low
evidence: doc/modules/clustering.rst:1023-1027 — "Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." Two problems verified: (a) typo "at this staged" (should be "at this stage"); (b) the invariant is inverted — points with core distance *greater than* ε are the ones DBSCAN* treats as noise, not those less than ε. Also stray trailing colons after `\varepsilon`: appear twice on those lines.
scenario: "Reader learning HDBSCAN from user guide → internalizes an incorrect definition of DBSCAN* noise assignment"
contract: Fix the typo to "at this stage" and correct the direction of the inequality: "Any points whose core distance is greater than ε are marked as noise."
instances: single-instance
```
</details>

### [LOW] User-guide prose uses undefined variable name `minimum_cluster_size` in place of `min_cluster_size`

User copies the guide's example `HDBSCAN(minimum_cluster_size=...)` → TypeError / warning about unknown parameter Replace both `minimum_cluster_size` occurrences with `min_cluster_size` to match the real API.

<details><summary>verbatim finding</summary>

```
### F180 — User-guide prose uses undefined variable name `minimum_cluster_size` in place of `min_cluster_size`
severity: low
evidence: doc/modules/clustering.rst:1060-1064 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples`". No parameter named `minimum_cluster_size` exists on `HDBSCAN`; the actual parameter is `min_cluster_size`.
scenario: "User copies the guide's example `HDBSCAN(minimum_cluster_size=...)` → TypeError / warning about unknown parameter"
contract: Replace both `minimum_cluster_size` occurrences with `min_cluster_size` to match the real API.
instances: single-instance
```
</details>

### [LOW] Stale/misleading narration comment in `plot_hdbscan.py`

Reader trying to understand the color scheme searches for the 'removed' color logic → wastes time hunting for palette manipulation code that does not exist Replace the comment with an accurate description, e.g. "Noise points (label -1) are drawn in black; other clusters use the Spectral palette.

<details><summary>verbatim finding</summary>

```
### F196 — Stale/misleading narration comment in `plot_hdbscan.py`
severity: low
evidence: examples/cluster/plot_hdbscan.py:28 — comment reads "# Black removed and is used for noise instead.", copy-pasted from the DBSCAN example where a specific color was removed from the palette; in this example nothing is removed — the Spectral palette is used unchanged and `col` is only overridden to black inside the `k == -1` branch (:36-37).
scenario: "Reader trying to understand the color scheme searches for the 'removed' color logic → wastes time hunting for palette manipulation code that does not exist"
contract: Replace the comment with an accurate description, e.g. "Noise points (label -1) are drawn in black; other clusters use the Spectral palette."
instances: single-instance
```
</details>

### [LOW] Docstring example loop uses `scale` variable but calls `hdb.fit(X)` unscaled [out-of-theme]

Reader runs the rendered example intending to see HDBSCAN's scale invariance → all three subplots are identical because the data is never scaled, undermining the tutorial's argument. Fit and plot on `X * scale` instead of `X` inside the HDBSCAN scale-invariance loop, matching the DBSCAN loop above it.

<details><summary>verbatim finding</summary>

```
### F240 — Docstring example loop uses `scale` variable but calls `hdb.fit(X)` unscaled [out-of-theme]
severity: low
evidence: examples/cluster/plot_hdbscan.py:112-116 iterates `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ..., parameters={"scale": scale})`. Compared to the immediately preceding DBSCAN loop at lines 93-95 which correctly does `dbs.fit(X * scale); plot(X * scale, ...)`, the HDBSCAN "scale invariance" demonstration never scales `X`, so it cannot actually demonstrate the property it claims to and simply plots the same clustering three times with different subtitle labels.
scenario: "Reader runs the rendered example intending to see HDBSCAN's scale invariance → all three subplots are identical because the data is never scaled, undermining the tutorial's argument."
contract: Fit and plot on `X * scale` instead of `X` inside the HDBSCAN scale-invariance loop, matching the DBSCAN loop above it.
instances: single-instance
```
</details>

### [LOW] Empty `_hdbscan/__init__.py` forces all imports to reach across a two-level private path

A second Cython-backed cluster algorithm follows this pattern → deep private import paths spread throughout the tree and each internal consumer becomes coupled to the concrete filename `hdbscan.py`. The `_hdbscan/__init__.py` must re-export the estimator (`from .hdbscan import HDBSCAN`) so importers can depend on `sklearn.cluster._hdbscan` rather than `sklearn.cluster._hdbscan.hdbscan`.

<details><summary>verbatim finding</summary>

```
### F228 — Empty `_hdbscan/__init__.py` forces all imports to reach across a two-level private path
severity: low
evidence: sklearn/cluster/_hdbscan/__init__.py is 0 bytes (Read reports "shorter than the provided offset (1). The file has 1 lines" and no content). Consumers must therefore write `from sklearn.cluster._hdbscan.hdbscan import HDBSCAN` (sklearn/cluster/__init__.py:26) and `from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING` (sklearn/cluster/tests/test_hdbscan.py:18), reaching into a nested private module rather than re-exporting `HDBSCAN` and any needed internals from the package's own `__init__`.
scenario: "A second Cython-backed cluster algorithm follows this pattern → deep private import paths spread throughout the tree and each internal consumer becomes coupled to the concrete filename `hdbscan.py`."
contract: The `_hdbscan/__init__.py` must re-export the estimator (`from .hdbscan import HDBSCAN`) so importers can depend on `sklearn.cluster._hdbscan` rather than `sklearn.cluster._hdbscan.hdbscan`.
instances: single-instance
```
</details>

### [LOW] `min_reachability = np.full(n_samples, ...)` never re-shrunk after first iteration in `mst_from_mutual_reachability`

Future refactor changes the initial `current_node` value → `label_filter` on the first pass has size != `min_reachability.size`, producing `IndexError` at runtime rather than a compile-time check. Initialise `min_reachability` explicitly as `np.full(n_samples, np.infty)` and after the first iteration select via `min_reachability[label_filter]` only when sizes are guaranteed to match; add a length assertion at the top of the loop.

<details><summary>verbatim finding</summary>

```
### F47 — `min_reachability = np.full(n_samples, ...)` never re-shrunk after first iteration in `mst_from_mutual_reachability`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:93-99 — `min_reachability` starts length `n_samples`, but is re-assigned via `min_reachability = np.minimum(left, right)` where `left = min_reachability[label_filter]` and `label_filter` was applied only to `current_labels`. On the first iteration, `label_filter.size == n_samples` matches `min_reachability.size == n_samples`, so it works, but the intent implicitly relies on lockstep shrinking; if the first-iteration invariant ever broke (e.g., `current_node != 0`), the boolean-index would raise a length-mismatch.
scenario: "Future refactor changes the initial `current_node` value → `label_filter` on the first pass has size != `min_reachability.size`, producing `IndexError` at runtime rather than a compile-time check."
contract: Initialise `min_reachability` explicitly as `np.full(n_samples, np.infty)` and after the first iteration select via `min_reachability[label_filter]` only when sizes are guaranteed to match; add a length assertion at the top of the loop.
instances: single-instance
```
</details>

### [LOW] `mst_from_mutual_reachability`'s `mutual_reachability` typed without `mode='c'` while callee assumes row-major access

A caller that passes a Fortran-ordered or strided distance matrix (e.g. from `pairwise_distances` with certain metric backends) → correct results but silent memory layout drift; combined with `PyArray_SHAPE(...)[0]` at :87 being used as the "rows" count, a Fortran-ordered input would compute the MST over the transpose axis. The parameter must be declared `cnp.ndarray[float64_t, ndim=2, mode='c']` so the C-contiguity is enforced at the API boundary.

<details><summary>verbatim finding</summary>

```
### F95 — `mst_from_mutual_reachability`'s `mutual_reachability` typed without `mode='c'` while callee assumes row-major access
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:62 declares `cnp.ndarray[float64_t, ndim=2] mutual_reachability`; the body uses `mutual_reachability[current_node][current_labels]` at :98 — a fancy-indexing row access that only produces a contiguous 1-D view when the source is C-contiguous.
scenario: "A caller that passes a Fortran-ordered or strided distance matrix (e.g. from `pairwise_distances` with certain metric backends) → correct results but silent memory layout drift; combined with `PyArray_SHAPE(...)[0]` at :87 being used as the "rows" count, a Fortran-ordered input would compute the MST over the transpose axis."
contract: The parameter must be declared `cnp.ndarray[float64_t, ndim=2, mode='c']` so the C-contiguity is enforced at the API boundary.
instances: single-instance
```
</details>

### [LOW] Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`

One package-local header (`_tree.pxd`) already provides the `PyArray_SHAPE` extern; `_linkage.pyx` re-declares it → two sources of truth for the same C signature. If the extern is ever adjusted (e.g., signature or header), the copies drift silently. Delete the extern block in `_linkage.pyx` and rely on the declaration in `_tree.pxd` (already cimported).

<details><summary>verbatim finding</summary>

```
### F105 — Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`, but the same extern is already declared in `sklearn/cluster/_hdbscan/_tree.pxd:48-49` and `_linkage.pyx` already `cimport`s from `_tree` (line 40).
scenario: "One package-local header (`_tree.pxd`) already provides the `PyArray_SHAPE` extern; `_linkage.pyx` re-declares it → two sources of truth for the same C signature. If the extern is ever adjusted (e.g., signature or header), the copies drift silently."
contract: Delete the extern block in `_linkage.pyx` and rely on the declaration in `_tree.pxd` (already cimported).
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44, sklearn/cluster/_hdbscan/_linkage.pyx:44-45, sklearn/cluster/_hdbscan/_tree.pxd:48-49]
```
</details>

### [LOW] `INFTY`/`np.infty`/`np.inf` used inconsistently across sibling files

`np.infty` was deprecated in NumPy 1.20 and removed in NumPy 2.0 → the `_linkage.pyx` sites will break with newer NumPy while sibling files that use `np.inf`/`INFTY` continue to work; the drift is invisible until CI hits a supported NumPy version that removed the alias. Standardise on one spelling for the infinity sentinel in this package. Use `np.inf` in Python-visible code (Cython allocations) and `INFTY`/`from libc.math cimport INFINITY` in Cython nogil contexts; remove all `np.infty` usages.

<details><summary>verbatim finding</summary>

```
### F107 — `INFTY`/`np.infty`/`np.inf` used inconsistently across sibling files
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:39 defines `cdef cnp.float64_t INFTY = np.inf` used at line 174; sklearn/cluster/_hdbscan/_linkage.pyx:93 and 159 use `np.infty` directly (`np.full(n_samples, fill_value=np.infty, ...)`); sklearn/cluster/_hdbscan/hdbscan.py:389 uses `np.inf` (`outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`). Three names for the same infinity value across three files in the same package.
scenario: "`np.infty` was deprecated in NumPy 1.20 and removed in NumPy 2.0 → the `_linkage.pyx` sites will break with newer NumPy while sibling files that use `np.inf`/`INFTY` continue to work; the drift is invisible until CI hits a supported NumPy version that removed the alias."
contract: Standardise on one spelling for the infinity sentinel in this package. Use `np.inf` in Python-visible code (Cython allocations) and `INFTY`/`from libc.math cimport INFINITY` in Cython nogil contexts; remove all `np.infty` usages.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:93, sklearn/cluster/_hdbscan/_linkage.pyx:159]
```
</details>

### [LOW] Copy-pasted docstring typos ("mutual-reahability", "collecteion", "single-linkage tree tree") duplicated across HDBSCAN files

Reader searches the API docs for 'mutual-reachability' → finds no hits, must guess the misspelling; fixing one instance leaves the others visible in generated Sphinx output. Fix all five instances of each phrase in a single edit; ideally extract the common `Returns`/`Parameters` blocks into a shared docstring template so future edits happen exactly once.

<details><summary>verbatim finding</summary>

```
### F114 — Copy-pasted docstring typos ("mutual-reahability", "collecteion", "single-linkage tree tree") duplicated across HDBSCAN files
severity: low
evidence: The misspellings "mutual-reahability" (missing 'c') and "collecteion" appear 5 times as a matched pair — sklearn/cluster/_hdbscan/_linkage.pyx:75-76, :137-138, :228-229 and sklearn/cluster/_hdbscan/hdbscan.py:102-103, :143-144. The redundant "single-linkage tree tree" (repeated word) appears 5 times — _linkage.pyx:234, hdbscan.py:149, :220, :326, :361. Each of these is one docstring template copy-pasted across multiple functions rather than being extracted or corrected in one place.
scenario: "Reader searches the API docs for 'mutual-reachability' → finds no hits, must guess the misspelling; fixing one instance leaves the others visible in generated Sphinx output."
contract: Fix all five instances of each phrase in a single edit; ideally extract the common `Returns`/`Parameters` blocks into a shared docstring template so future edits happen exactly once.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361]
```
</details>

### [LOW] Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`

Refactor to change the PyArray_SHAPE signature → must be updated in two places rather than one; risk of divergent declarations Remove the local `cdef extern` block in `_linkage.pyx` and rely on the declaration exported by `_tree.pxd`.

<details><summary>verbatim finding</summary>

```
### F123 — Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` locally, but the same extern is already declared in `sklearn/cluster/_hdbscan/_tree.pxd:48-49` which `_linkage.pyx` already cimports from (see line 40 `from ...cluster._hdbscan._tree cimport HIERARCHY_t`). The local re-declaration is superseded/leftover code.
scenario: "Refactor to change the PyArray_SHAPE signature → must be updated in two places rather than one; risk of divergent declarations"
contract: Remove the local `cdef extern` block in `_linkage.pyx` and rely on the declaration exported by `_tree.pxd`.
instances: single-instance
```
</details>

### [LOW] `mst_from_data_matrix` initializes `current_sources` to `1` (misleading sentinel)

reader auditing correctness on the first outer iteration (i=0) reads `current_sources[j]` before any write → cannot distinguish 'source=1 because we chose it' from 'source=1 because it was the fill value', obscuring the read-before-write invariant Initialize `current_sources` to `0` (matching the initial `current_node=0`) and document the invariant that reads only happen after a write.

<details><summary>verbatim finding</summary>

```
### F162 — `mst_from_data_matrix` initializes `current_sources` to `1` (misleading sentinel)
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:160 — `current_sources = np.ones(n_samples, dtype=np.int64)`. Nothing about "1" is meaningful as a source-of-source-node sentinel; before line 204 (`current_sources[j] = current_node`) writes to a slot, any read at line 179 (`next_node_source = current_sources[j]`) returns a bogus `1`. This value can then be assigned into `mst[i].current_node` via lines 197-199 if `next_node_min_reach` remains `INFTY`/other stale value. A lying name/initialization: it pretends every node's source is node 1.
scenario: "reader auditing correctness on the first outer iteration (i=0) reads `current_sources[j]` before any write → cannot distinguish 'source=1 because we chose it' from 'source=1 because it was the fill value', obscuring the read-before-write invariant"
contract: Initialize `current_sources` to `0` (matching the initial `current_node=0`) and document the invariant that reads only happen after a write.
instances: single-instance
```
</details>

### [LOW] `mst_from_mutual_reachability` starts from a hard-coded `current_node = 0` with no rationale

reader trying to understand seed-node semantics finds a bare magic 0 → wonders whether cluster ordering depends on it Add a single-line comment stating "Prim's is invariant to the starting node; we pick 0" or make the seed a named constant.

<details><summary>verbatim finding</summary>

```
### F165 — `mst_from_mutual_reachability` starts from a hard-coded `current_node = 0` with no rationale
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:92 — Prim's is seeded at node 0 unconditionally; same in `mst_from_data_matrix` at line 162. No comment states that Prim's MST is invariant to the starting node (which is true for connected graphs but non-obvious to Cython readers), and there is no assertion that the graph is connected.
scenario: "reader trying to understand seed-node semantics finds a bare magic 0 → wonders whether cluster ordering depends on it"
contract: Add a single-line comment stating "Prim's is invariant to the starting node; we pick 0" or make the seed a named constant.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:92, sklearn/cluster/_hdbscan/_linkage.pyx:162]
```
</details>

### [LOW] Pervasive typographical errors in docstrings and comments

Users grepping code/docs for canonical spellings ('mutual_reachability', 'sibling', etc.) miss all these locations → obscures searchability of the algorithm's terminology across the codebase and appears unprofessional in generated Sphinx docs Fix each misspelling to its canonical form ("reachability", "collection", "mutual", "smaller", "sibling") and remove the duplicated "tree tree"; rename `test_reachibility.py` to `test_reachability.py`.

<details><summary>verbatim finding</summary>

```
### F181 — Pervasive typographical errors in docstrings and comments
severity: low
evidence: recurring misspellings across new HDBSCAN code — "reahability" (sklearn/cluster/_hdbscan/_linkage.pyx:75, :137, :229; sklearn/cluster/_hdbscan/hdbscan.py:102, :143), "collecteion" (sklearn/cluster/_hdbscan/_linkage.pyx:76, :138, :229; sklearn/cluster/_hdbscan/hdbscan.py:103, :144), "mututal_reachability_graph" (sklearn/cluster/_hdbscan/hdbscan.py:91; sklearn/cluster/_hdbscan/_reachability.pyx:76), "mutual_reachibility_distance" (sklearn/cluster/_hdbscan/_reachability.pyx:127, :144, :149, :182, :206, :209), "smaler" (sklearn/cluster/_hdbscan/_tree.pyx:133), "simbling" (sklearn/cluster/_hdbscan/_tree.pyx:503; sklearn/cluster/tests/test_hdbscan.py:530), "single-linkage tree tree" duplicate word (sklearn/cluster/_hdbscan/_linkage.pyx:234; sklearn/cluster/_hdbscan/hdbscan.py:149, :220, :326, :361), test filename "test_reachibility.py".
scenario: "Users grepping code/docs for canonical spellings ('mutual_reachability', 'sibling', etc.) miss all these locations → obscures searchability of the algorithm's terminology across the codebase and appears unprofessional in generated Sphinx docs"
contract: Fix each misspelling to its canonical form ("reachability", "collection", "mutual", "smaller", "sibling") and remove the duplicated "tree tree"; rename `test_reachibility.py` to `test_reachability.py`.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:233, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:148, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:219, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/hdbscan.py:906, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/_reachability.pyx:76, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/tests/test_hdbscan.py:530, sklearn/cluster/_hdbscan/tests/test_reachibility.py]
```
</details>

### [LOW] `mst_from_data_matrix` initializes `current_sources` to 1 instead of 0 [out-of-theme]

In early Prim iterations, for a candidate `j` whose `min_reachability[j]` was updated in an earlier step and whose branch on line 195-200 reads `next_node_source = current_sources[j]` = 1 (never actually updated on the first pass) → whenever `next_node_min_reach < new_reachability`, the MST edge source is recorded as node 1 rather than the true source, producing incorrect edges when node 1 was not in fact the source. Initialize `current_sources` to `np.zeros(n_samples, dtype=np.int64)` (matching the initial `current_node = 0`) or, better, only read `current_sources[j]` after verifying `min_reachability[j] < INFTY` (i.e., a real update has occurred).

<details><summary>verbatim finding</summary>

```
### F216 — `mst_from_data_matrix` initializes `current_sources` to 1 instead of 0 [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:160 — `current_sources = np.ones(n_samples, dtype=np.int64)`. `current_sources[j]` is read on line 179 as `next_node_source` when `mutual_reachability_distance > next_node_min_reach` but before any real source has been recorded for `j`, defaulting to sample 1.
scenario: "In early Prim iterations, for a candidate `j` whose `min_reachability[j]` was updated in an earlier step and whose branch on line 195-200 reads `next_node_source = current_sources[j]` = 1 (never actually updated on the first pass) → whenever `next_node_min_reach < new_reachability`, the MST edge source is recorded as node 1 rather than the true source, producing incorrect edges when node 1 was not in fact the source."
contract: Initialize `current_sources` to `np.zeros(n_samples, dtype=np.int64)` (matching the initial `current_node = 0`) or, better, only read `current_sources[j]` after verifying `min_reachability[j] < INFTY` (i.e., a real update has occurred).
instances: single-instance
```
</details>

### [LOW] `_linkage.pyx` reaches `_tree` symbols via absolute cross-package path instead of local sibling import

Package rename or relocation of `_hdbscan` → the absolute path `...cluster._hdbscan._tree` breaks, while a local `from ._tree cimport ...` would remain valid; also obscures the sibling relationship at read time Import the sibling module locally: `from ._tree cimport HIERARCHY_t` and `from ._tree import HIERARCHY_dtype`, matching how `hdbscan.py` (L57-58) imports the same module

<details><summary>verbatim finding</summary>

```
### F224 — `_linkage.pyx` reaches `_tree` symbols via absolute cross-package path instead of local sibling import
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:40-41 — `from ...cluster._hdbscan._tree cimport HIERARCHY_t` and `from ...cluster._hdbscan._tree import HIERARCHY_dtype`; the same file imports `_typedefs` relatively but reroutes its own sibling `_tree` up three package levels and back down
scenario: "Package rename or relocation of `_hdbscan` → the absolute path `...cluster._hdbscan._tree` breaks, while a local `from ._tree cimport ...` would remain valid; also obscures the sibling relationship at read time"
contract: Import the sibling module locally: `from ._tree cimport HIERARCHY_t` and `from ._tree import HIERARCHY_dtype`, matching how `hdbscan.py` (L57-58) imports the same module
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]
```
</details>

### [LOW] `MST_edge_t` struct is unpublished from any pxd, blocking Cython-level cross-module reuse

A future HDBSCAN Cython routine needs the MST edge struct → it must either duplicate the ctypedef or bounce through Python, because `MST_edge_t` has no pxd home. `MST_edge_t` must be published from `_linkage.pxd` (a new pxd colocated with `_linkage.pyx`) so any dependent pyx can cimport it.

<details><summary>verbatim finding</summary>

```
### F229 — `MST_edge_t` struct is unpublished from any pxd, blocking Cython-level cross-module reuse
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:48-59 defines the `MST_edge_dtype` numpy dtype and the `MST_edge_t` packed struct at module scope with no corresponding pxd declaration (there is no `_linkage.pxd` in `sklearn/cluster/_hdbscan/`, verified via `ls sklearn/cluster/_hdbscan/`). `MST_edge_dtype` is imported at the Python level by sklearn/cluster/_hdbscan/hdbscan.py:55, but the Cython-visible `MST_edge_t` cannot be cimported by any other pyx. Meanwhile sklearn/cluster/_hdbscan/_linkage.pyx:190 inlines `mutual_reachability_distance = max(...)` inside `mst_from_data_matrix` while `_reachability.pyx` owns the mutual-reachability concept for the dense/sparse-precomputed paths.
scenario: "A future HDBSCAN Cython routine needs the MST edge struct → it must either duplicate the ctypedef or bounce through Python, because `MST_edge_t` has no pxd home."
contract: `MST_edge_t` must be published from `_linkage.pxd` (a new pxd colocated with `_linkage.pyx`) so any dependent pyx can cimport it.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:48-59]
```
</details>

### [LOW] `np.infty` is a deprecated NumPy alias used by newly added Cython code [out-of-theme]

Users on NumPy 2.0+ run HDBSCAN → AttributeError at import/first-call of `mst_from_mutual_reachability` and `mst_from_data_matrix`. Replace `np.infty` with `np.inf` at both call sites.

<details><summary>verbatim finding</summary>

```
### F233 — `np.infty` is a deprecated NumPy alias used by newly added Cython code [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:93 uses `np.full(n_samples, fill_value=np.infty, dtype=np.float64)` and line 159 uses `np.full(n_samples, fill_value=np.infty, dtype=np.float64)`. `np.infty` is a deprecated alias that NumPy 1.20+ warns about and NumPy 2.0 removes; canonical spelling is `np.inf`.
scenario: "Users on NumPy 2.0+ run HDBSCAN → AttributeError at import/first-call of `mst_from_mutual_reachability` and `mst_from_data_matrix`."
contract: Replace `np.infty` with `np.inf` at both call sites.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:93, sklearn/cluster/_hdbscan/_linkage.pyx:159]
```
</details>

### [LOW] Sparse core-distance semantics differ from dense (self-distance handling)

user supplies a precomputed CSR distance matrix with implicit zero diagonal → HDBSCAN produces a different clustering than the equivalent dense matrix with the same explicit distances set the dense diagonal to `np.inf` before partitioning so both paths exclude the self-entry consistently.

<details><summary>verbatim finding</summary>

```
### F17 — Sparse core-distance semantics differ from dense (self-distance handling)
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133-137 (dense) selects `partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx]` over the full row including the implicit `0` self-distance, i.e. the `(further_neighbor_idx)`-th smallest counting self at position 0. sklearn/cluster/_hdbscan/_reachability.pyx:193-200 (sparse) selects `partition(row_data, further_neighbor_idx)[further_neighbor_idx]` over `data[indptr[i]:indptr[i+1]]`, which excludes any zero self-entry that was `eliminate_zeros()`d (standard for CSR distance matrices). Result: for the same underlying distances, the sparse path effectively uses the k-th nearest neighbor while the dense path uses the (k-1)-th, an off-by-one.
scenario: "user supplies a precomputed CSR distance matrix with implicit zero diagonal → HDBSCAN produces a different clustering than the equivalent dense matrix with the same explicit distances"
contract: set the dense diagonal to `np.inf` before partitioning so both paths exclude the self-entry consistently.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:196]
```
</details>

### [LOW] `_dense_mutual_reachability_graph` writes to `distance_matrix` without a symmetric `nogil` guard on Python-object refcount

Maintainer replaces `range(n_samples)` with `prange(n_samples)` per the TODO → concurrent writes to symmetric entries `distance_matrix[i,j]` and `distance_matrix[j,i]` (which each write the same slot from opposite (i,j)) create a race, and the loop over `for j in range(n_samples)` (not `range(i, n_samples)`) redundantly writes each entry twice, doubling the race surface. Before adding `prange`, restrict the inner loop to `j in range(i, n_samples)` and mirror-write both symmetric entries, so parallelisation is safe.

<details><summary>verbatim finding</summary>

```
### F48 — `_dense_mutual_reachability_graph` writes to `distance_matrix` without a symmetric `nogil` guard on Python-object refcount
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — the `with nogil:` block reads/writes `distance_matrix[i, j]` via a typed memoryview; the `core_distances` array is a Python-managed numpy allocation created at line 133-137 outside `nogil`. There is no protection against GC/reallocation; because this is single-threaded there is no data race, but the `TODO: Update w/ prange` comment invites a future maintainer to enable OMP parallelism, which as written would create a race on `distance_matrix[i,j]`/`distance_matrix[j,i]` since both are written by different `i`.
scenario: "Maintainer replaces `range(n_samples)` with `prange(n_samples)` per the TODO → concurrent writes to symmetric entries `distance_matrix[i,j]` and `distance_matrix[j,i]` (which each write the same slot from opposite (i,j)) create a race, and the loop over `for j in range(n_samples)` (not `range(i, n_samples)`) redundantly writes each entry twice, doubling the race surface."
contract: Before adding `prange`, restrict the inner loop to `j in range(i, n_samples)` and mirror-write both symmetric entries, so parallelisation is safe.
instances: single-instance
```
</details>

### [LOW] `_dense_mutual_reachability_graph` is silently non-symmetric on non-symmetric input

internal caller passes a non-symmetric matrix (e.g. future refactor bypasses the `_hdbscan_brute` symmetry check) → silently wrong core distances and mutual reachabilities, no error surfaced. Move the symmetry precondition into the function docstring's Parameters section so the invariant is stated at the function boundary rather than relying on a hidden caller-established assumption.

<details><summary>verbatim finding</summary>

```
### F147 — `_dense_mutual_reachability_graph` is silently non-symmetric on non-symmetric input
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:130-149 — comment says "We assume that the distance matrix is symmetric" but the algorithm only computes `core_distances` row-wise via `np.partition` on each row. If the caller violates the unstated symmetry precondition, results are silently wrong; there is no assertion or docstring warning at the function boundary (in contrast to `_hdbscan_brute` which does check symmetry only when `metric="precomputed"`).
scenario: "internal caller passes a non-symmetric matrix (e.g. future refactor bypasses the `_hdbscan_brute` symmetry check) → silently wrong core distances and mutual reachabilities, no error surfaced."
contract: Move the symmetry precondition into the function docstring's Parameters section so the invariant is stated at the function boundary rather than relying on a hidden caller-established assumption.
instances: single-instance
```
</details>

### [LOW] `max_distance` docstring refers to a nonexistent identifier `max_dist`

Reader trying to enable this fallback searches for `max_dist` → finds nothing; passes `max_dist=` and gets a TypeError Replace `max_dist` with `max_distance` in both docstrings.

<details><summary>verbatim finding</summary>

```
### F182 — `max_distance` docstring refers to a nonexistent identifier `max_dist`
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:68-72 and 174-178 — both `max_distance` parameter blocks state "it is instead truncated to `max_dist`", but there is no `max_dist` argument or attribute anywhere; the correct name is `max_distance`.
scenario: "Reader trying to enable this fallback searches for `max_dist` → finds nothing; passes `max_dist=` and gets a TypeError"
contract: Replace `max_dist` with `max_distance` in both docstrings.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:71, sklearn/cluster/_hdbscan/_reachability.pyx:177]
```
</details>

### [LOW] Stale user-guide sentence claims "The HDBSCAN implementation is multithreaded"

Users choosing HDBSCAN over OPTICS on the basis of multithreading claim → surprised by wall-time single-thread performance Remove "multithreaded, and" (or reword) from the OPTICS-vs-HDBSCAN paragraph until `_dense_mutual_reachability_graph` is parallelized.

<details><summary>verbatim finding</summary>

```
### F184 — Stale user-guide sentence claims "The HDBSCAN implementation is multithreaded"
severity: low
evidence: doc/modules/clustering.rst hunk near line 1149 — "The HDBSCAN implementation is multithreaded, and has better algorithmic runtime complexity than OPTICS…" while sklearn/cluster/_hdbscan/_reachability.pyx:140-141 contains a TODO explicitly noting that the loop is **not** yet paralleled: "# TODO: Update w/ prange with thread count based on _openmp_effective_n_threads". The user-facing docs therefore promise multithreading that this implementation does not deliver.
scenario: "Users choosing HDBSCAN over OPTICS on the basis of multithreading claim → surprised by wall-time single-thread performance"
contract: Remove "multithreaded, and" (or reword) from the OPTICS-vs-HDBSCAN paragraph until `_dense_mutual_reachability_graph` is parallelized.
instances: single-instance
```
</details>

### [LOW] `_dense_mutual_reachability_graph` runs a single-threaded O(n²) loop despite the TODO

Brute-mode HDBSCAN with `n_jobs>1` on a large dense distance matrix → user pays full serial O(n²) cost building mutual reachability while pairwise_distances step used multiple cores Replace the serial `for i in range(n_samples)` with `prange` gated by `_openmp_effective_n_threads()`, matching the parallelism strategy used elsewhere in scikit-learn.

<details><summary>verbatim finding</summary>

```
### F212 — `_dense_mutual_reachability_graph` runs a single-threaded O(n²) loop despite the TODO
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — `with nogil:` wraps a serial double loop over `n_samples × n_samples`; a comment `# TODO: Update w/ prange with thread count based on _openmp_effective_n_threads` acknowledges the parallelization is missing. The Python-level `n_jobs` parameter is threaded through `_hdbscan_brute` but has no effect here.
scenario: "Brute-mode HDBSCAN with `n_jobs>1` on a large dense distance matrix → user pays full serial O(n²) cost building mutual reachability while pairwise_distances step used multiple cores"
contract: Replace the serial `for i in range(n_samples)` with `prange` gated by `_openmp_effective_n_threads()`, matching the parallelism strategy used elsewhere in scikit-learn.
instances: single-instance
```
</details>

### [LOW] Internal Cython variable `mutual_reachibility_distance` misspelled in `_reachability.pyx` [out-of-theme]

Grep-based navigation for `reachability_distance` in this module → does not match the loop-local variable, hindering search and code review. Rename every occurrence of `mutual_reachibility_distance` to `mutual_reachability_distance` in `_reachability.pyx`.

<details><summary>verbatim finding</summary>

```
### F237 — Internal Cython variable `mutual_reachibility_distance` misspelled in `_reachability.pyx` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133 declares `floating mutual_reachibility_distance`, and it is reused at lines 144, 149, 182, 206, 209, 210. The surrounding docstrings and public API name spell it correctly as "reachability" (e.g. line 47 "mutual reachability graph"). This is a straight misspelling propagated through both the dense and sparse implementations.
scenario: "Grep-based navigation for `reachability_distance` in this module → does not match the loop-local variable, hindering search and code review."
contract: Rename every occurrence of `mutual_reachibility_distance` to `mutual_reachability_distance` in `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]
```
</details>

### [LOW] Unused `uint8_t` cimport in `_tree.pxd`

Cython lint / pre-commit that the PR description explicitly credits for 'trimmed unused variables' should have flagged this dead cimport → the PR claims that hygiene pass; this slipped through Remove `uint8_t` from the cimport list in `_tree.pxd`.

<details><summary>verbatim finding</summary>

```
### F129 — Unused `uint8_t` cimport in `_tree.pxd`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:30 — `from ...utils._typedefs cimport intp_t, float64_t, uint8_t`; grep shows only `intp_t` and `float64_t` used in the file. No struct field or declaration uses `uint8_t`.
scenario: "Cython lint / pre-commit that the PR description explicitly credits for 'trimmed unused variables' should have flagged this dead cimport → the PR claims that hygiene pass; this slipped through"
contract: Remove `uint8_t` from the cimport list in `_tree.pxd`.
instances: single-instance
```
</details>

### [LOW] Duplicate `cdef extern PyArray_SHAPE` declared in both `_tree.pxd` and `_linkage.pyx`

Future maintainer changes signature/name in one place → the other silently diverges; extern declared in a pxd should be the single source, private redeclaration in the pyx violates single-owner rule. The extern declaration must live in one location — the pxd owned by the numpy-shape utility — and every dependent pyx cimports it from there, never re-declaring locally.

<details><summary>verbatim finding</summary>

```
### F226 — Duplicate `cdef extern PyArray_SHAPE` declared in both `_tree.pxd` and `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. `_linkage.pyx` already cimports from `_tree.pxd` (line 40 `from ...cluster._hdbscan._tree cimport HIERARCHY_t`) yet at sklearn/cluster/_hdbscan/_linkage.pyx:44-45 re-declares the identical `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`.
scenario: "Future maintainer changes signature/name in one place → the other silently diverges; extern declared in a pxd should be the single source, private redeclaration in the pyx violates single-owner rule."
contract: The extern declaration must live in one location — the pxd owned by the numpy-shape utility — and every dependent pyx cimports it from there, never re-declaring locally.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:48-49, sklearn/cluster/_hdbscan/_linkage.pyx:44-45]
```
</details>

### [LOW] `CONDENSED_t` struct leaked into `_tree.pxd` despite having zero external consumers

Publishing an internal-only type in the pxd invites external cimports → freezes the on-disk layout of `CONDENSED_dtype` as a cross-module contract when it is in fact `_tree.pyx`-private. Types with no cross-module consumers must be defined inside the `.pyx` (or a private module-local `.pxi`), not in the shared `.pxd`.

<details><summary>verbatim finding</summary>

```
### F227 — `CONDENSED_t` struct leaked into `_tree.pxd` despite having zero external consumers
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:42-46 defines `ctypedef packed struct CONDENSED_t` in the public pxd. Grep shows every use of `CONDENSED_t` is inside `_tree.pyx` itself (17 hits, all in `_tree.pyx`); no other translation unit cimports it. Only `HIERARCHY_t` is legitimately consumed externally (by `_linkage.pyx:40`).
scenario: "Publishing an internal-only type in the pxd invites external cimports → freezes the on-disk layout of `CONDENSED_dtype` as a cross-module contract when it is in fact `_tree.pyx`-private."
contract: Types with no cross-module consumers must be defined inside the `.pyx` (or a private module-local `.pxi`), not in the shared `.pxd`.
instances: single-instance
```
</details>

### [LOW] `_compute_stability` allocates `births` twice, hiding intent

Future maintenance edit that inserts logic between lines 252 and 254 → the inserted computation is silently discarded when line 254 overwrites the array, producing wrong stability values with no compile-time signal. Delete the duplicate `births = np.full(...)` at line 252 (or 254) so only a single, intentional allocation remains.

<details><summary>verbatim finding</summary>

```
### F10 — `_compute_stability` allocates `births` twice, hiding intent
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two identical `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` statements back-to-back; the first allocation is unconditionally overwritten by the second, so any read between them (there is none currently) would silently disappear if reintroduced.
scenario: "Future maintenance edit that inserts logic between lines 252 and 254 → the inserted computation is silently discarded when line 254 overwrites the array, producing wrong stability values with no compile-time signal."
contract: Delete the duplicate `births = np.full(...)` at line 252 (or 254) so only a single, intentional allocation remains.
instances: single-instance
```
</details>

### [LOW] `_condense_tree` builds `ignore` sized to `len(node_list)` but the `node_list` returned by `bfs_from_hierarchy` may not include indices used to write into `ignore[sub_node]`

A future edit reduces `bfs_from_hierarchy` output size (e.g., filters leaves) → `ignore[sub_node]` writes past the end of the array, silently corrupting adjacent memory when `boundscheck=False` Size `ignore` from a semantically explicit expression tied to the hierarchy (e.g., `2 * hierarchy.shape[0] + 1`) rather than `len(node_list)`.

<details><summary>verbatim finding</summary>

```
### F12 — `_condense_tree` builds `ignore` sized to `len(node_list)` but the `node_list` returned by `bfs_from_hierarchy` may not include indices used to write into `ignore[sub_node]`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:161 — `ignore = np.zeros(len(node_list), dtype=bool)`. In the "both children small" and single-side branches, indices written are `ignore[sub_node]` where `sub_node` iterates results of recursive `bfs_from_hierarchy(hierarchy, left/right)`. For a well-formed hierarchy of `n_samples` samples, the initial BFS from `root = 2*(n_samples-1)` visits all `2*n_samples - 1` nodes, so `len(node_list) == 2*n_samples - 1` and indices in `[0, 2*n_samples-2]` are valid. However, `ignore[node]` at line 164 uses `node`, which can be up to `root = 2*(n_samples-1) = 2*n_samples - 2` — the array size `2*n_samples - 1` is exactly `root + 1`, so within bounds. The invariant that BFS visits every node is not enforced or asserted; any future refactor that skips duplicate nodes in BFS breaks `ignore` sizing without a bounds error under Cython `boundscheck=False`.
scenario: "A future edit reduces `bfs_from_hierarchy` output size (e.g., filters leaves) → `ignore[sub_node]` writes past the end of the array, silently corrupting adjacent memory when `boundscheck=False`"
contract: Size `ignore` from a semantically explicit expression tied to the hierarchy (e.g., `2 * hierarchy.shape[0] + 1`) rather than `len(node_list)`.
instances: single-instance
```
</details>

### [LOW] `_do_labelling`: `result = np.empty(root_cluster, dtype=np.intp)` conflates the value of `root_cluster` (min parent id) with the number of samples

Refactor of `_condense_tree` shifts the root label → `_do_labelling` returns an array of the wrong length; label assignments silently corrupted Pass `n_samples` explicitly to `_do_labelling` (or derive it from `condensed_tree[condensed_tree['cluster_size']==1]['child'].max()+1`) rather than relying on `min(parent_array)` accidentally equalling `n_samples`.

<details><summary>verbatim finding</summary>

```
### F15 — `_do_labelling`: `result = np.empty(root_cluster, dtype=np.intp)` conflates the value of `root_cluster` (min parent id) with the number of samples
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-481 — `root_cluster = np.min(parent_array); result = np.empty(root_cluster, dtype=np.intp)`. This relies on `root_cluster == n_samples` which is only guaranteed by `_condense_tree`'s numbering scheme (relabel starts at `n_samples + 1`, but the root itself is relabeled to `n_samples`). Any change to the relabel scheme in `_condense_tree` breaks `_do_labelling` silently: `result` will be sized wrong, and the subsequent `for n in range(root_cluster)` will over- or under-scan samples with no error.
scenario: "Refactor of `_condense_tree` shifts the root label → `_do_labelling` returns an array of the wrong length; label assignments silently corrupted"
contract: Pass `n_samples` explicitly to `_do_labelling` (or derive it from `condensed_tree[condensed_tree['cluster_size']==1]['child'].max()+1`) rather than relying on `min(parent_array)` accidentally equalling `n_samples`.
instances: single-instance
```
</details>

### [LOW] `_do_labelling` single-cluster branch depends on `parent_lambda` being a scalar-like 1-element array

any hierarchy pathology (duplicated child row, or invocation with n that matches multiple entries) → cryptic ValueError from ambiguous-truth-value evaluation instead of a domain-specific error extract the scalar explicitly (`parent_lambda = lambda_array[child_array == n][0]`) and assert the single-edge invariant before comparison.

<details><summary>verbatim finding</summary>

```
### F19 — `_do_labelling` single-cluster branch depends on `parent_lambda` being a scalar-like 1-element array
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 `parent_lambda = lambda_array[child_array == n]` returns an ndarray; sklearn/cluster/_hdbscan/_tree.pyx:505 `if parent_lambda >= threshold:` implicitly converts to scalar bool only when the array has exactly one element. The inline comment at lines 496-497 asserts "There can only be one edge with this particular child" — but there is no assertion or guard enforcing it. If a malformed condensed tree ever has duplicate child rows (or `n` is inadvertently referenced as a cluster), `if` on a multi-element boolean array raises `ValueError: The truth value of an array with more than one element is ambiguous`.
scenario: "any hierarchy pathology (duplicated child row, or invocation with n that matches multiple entries) → cryptic ValueError from ambiguous-truth-value evaluation instead of a domain-specific error"
contract: extract the scalar explicitly (`parent_lambda = lambda_array[child_array == n][0]`) and assert the single-edge invariant before comparison.
instances: single-instance
```
</details>

### [LOW] Recursion in `recurse_leaf_dfs` on caller-controlled tree depth risks stack overflow

A malicious or pathological input dataset that produces a highly unbalanced condensed hierarchy (chain-of-clusters shape with depth ~ n_samples) → Python recursion limit is exceeded inside a `cpdef` Cython routine, raising `RecursionError` mid-fit and aborting the estimator with no graceful degradation. Reimplement `recurse_leaf_dfs` iteratively (explicit stack/deque) so tree-descent depth is bounded by heap memory, not by Python's recursion limit.

<details><summary>verbatim finding</summary>

```
### F25 — Recursion in `recurse_leaf_dfs` on caller-controlled tree depth risks stack overflow
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555–566 — `recurse_leaf_dfs` recursively descends the condensed cluster tree with no depth cap: `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`. The tree depth is a function of the input data shape, which is caller-supplied.
scenario: "A malicious or pathological input dataset that produces a highly unbalanced condensed hierarchy (chain-of-clusters shape with depth ~ n_samples) → Python recursion limit is exceeded inside a `cpdef` Cython routine, raising `RecursionError` mid-fit and aborting the estimator with no graceful degradation."
contract: Reimplement `recurse_leaf_dfs` iteratively (explicit stack/deque) so tree-descent depth is bounded by heap memory, not by Python's recursion limit.
instances: single-instance
```
</details>

### [LOW] `TreeUnionFind.find` uses unbounded recursion for path compression

Adversarially crafted input producing a deep union chain (before path compression kicks in) → `find` recurses `O(n_samples)` deep, tripping Python/C-stack limits or CPython's recursion cap when invoked from `labelling_at_cut`/`_do_labelling`, terminating the fit with a stack overflow rather than a bounded error. Convert `TreeUnionFind.find` to an iterative two-pass path-compression implementation so union-find operations are safe on adversarial input sizes.

<details><summary>verbatim finding</summary>

```
### F26 — `TreeUnionFind.find` uses unbounded recursion for path compression
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:352–356 — `cdef cnp.intp_t find(self, cnp.intp_t x): if self.data[x, 0] != x: self.data[x, 0] = self.find(self.data[x, 0]); ...` recurses through a union-find chain whose length is bounded by the number of samples.
scenario: "Adversarially crafted input producing a deep union chain (before path compression kicks in) → `find` recurses `O(n_samples)` deep, tripping Python/C-stack limits or CPython's recursion cap when invoked from `labelling_at_cut`/`_do_labelling`, terminating the fit with a stack overflow rather than a bounded error."
contract: Convert `TreeUnionFind.find` to an iterative two-pass path-compression implementation so union-find operations are safe on adversarial input sizes.
instances: single-instance
```
</details>

### [LOW] `cluster_selection_method` argument to `tree_to_labels` reaches `_get_clusters` without validation of untrusted string, and is not typed

A caller uses `sklearn.cluster._hdbscan._tree.tree_to_labels` directly with `cluster_selection_method='Eom'` (case typo) → the function silently returns labels derived from an unfiltered `is_cluster` dict rather than raising, giving the caller silently wrong labels. Validate `cluster_selection_method` at the top of `_get_clusters` and raise `ValueError` on any value outside `{'eom', 'leaf'}`.

<details><summary>verbatim finding</summary>

```
### F29 — `cluster_selection_method` argument to `tree_to_labels` reaches `_get_clusters` without validation of untrusted string, and is not typed
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:56-77 — `cpdef tuple tree_to_labels(..., cluster_selection_method="eom", ...)` accepts an untyped Python object as `cluster_selection_method`, forwards it unchanged to `_get_clusters`, where at line 727/761 it is compared to the string literals `'eom'` and `'leaf'`. If neither branch matches (e.g., typo `"EOM"`, or a callable), the function silently falls through and returns `labels` computed with `is_cluster` unmodified from its initial value (`{cluster: True for cluster in node_list}`), producing an incoherent labelling with no error raised. The Python `HDBSCAN` class validates its own `cluster_selection_method` at the estimator boundary (line 641), but the `cpdef` function is publicly importable (see tests importing `_do_labelling`, `_condense_tree` from the compiled module) and offers no defense-in-depth.
scenario: "A caller uses `sklearn.cluster._hdbscan._tree.tree_to_labels` directly with `cluster_selection_method='Eom'` (case typo) → the function silently returns labels derived from an unfiltered `is_cluster` dict rather than raising, giving the caller silently wrong labels."
contract: Validate `cluster_selection_method` at the top of `_get_clusters` and raise `ValueError` on any value outside `{'eom', 'leaf'}`.
instances: single-instance
```
</details>

### [LOW] `cpdef` Cython entry points accept caller-supplied index arrays without bounds validation while running with `boundscheck=False`

A user calls the publicly-importable `sklearn.cluster._hdbscan._tree._condense_tree` (as the tests import `_do_labelling`, `_condense_tree` at line 19-23 of the test file) with a hand-crafted `HIERARCHY_dtype` array containing an out-of-range `left_node` → `relabel[left] = next_label` writes past the end of the allocated buffer without raising, corrupting adjacent Python heap memory. Validate that `left`, `right` (from `HIERARCHY_t.left_node/right_node`) and all `parent`/`child` values from caller-supplied structured arrays fall within `[0, 2*hierarchy.shape[0]]` before use, and raise `ValueError` at the top of each `cpdef` entry point on violation.

<details><summary>verbatim finding</summary>

```
### F31 — `cpdef` Cython entry points accept caller-supplied index arrays without bounds validation while running with `boundscheck=False`
severity: low
evidence: sklearn/_build_utils/__init__.py:71-78 shows `compiler_directives = {..., "boundscheck": cython_enable_debug_directives, "wraparound": False, ...}` — production builds compile with bounds checking disabled. sklearn/cluster/_hdbscan/_tree.pyx exposes `cpdef` functions that indirectly index memoryviews from caller-supplied data: `tree_to_labels` (line 56) → `_condense_tree` (line 117) sizes `relabel = np.empty(root + 1, ...)` from `hierarchy.shape[0]` (line 145) then writes `relabel[left] = next_label` (line 187) where `left` is read from `hierarchy[node - n_samples].left_node` without validating that `left <= root`; `_do_labelling` (line 431) computes `root_cluster = np.min(parent_array)` (line 480) and allocates `result = np.empty(root_cluster, dtype=np.intp)` (line 481) then writes `result[n] = label` for `n in range(root_cluster)` (line 508) — but `union_find` is sized by `np.max(parent_array) + 1` and `cluster_label_map[cluster]` (line 494) is a dict lookup that may raise `KeyError` when caller-supplied `condensed_tree` violates the invariants. With `boundscheck=False`, a malformed caller-supplied `HIERARCHY_dtype`/`CONDENSED_dtype` array (e.g., a `left_node` value of `-1` or `> root`, or a `parent_array` with unexpected values) causes an out-of-bounds write to `relabel`/`deaths`/memoryviews — silent memory corruption, not `IndexError`.
scenario: "A user calls the publicly-importable `sklearn.cluster._hdbscan._tree._condense_tree` (as the tests import `_do_labelling`, `_condense_tree` at line 19-23 of the test file) with a hand-crafted `HIERARCHY_dtype` array containing an out-of-range `left_node` → `relabel[left] = next_label` writes past the end of the allocated buffer without raising, corrupting adjacent Python heap memory."
contract: Validate that `left`, `right` (from `HIERARCHY_t.left_node/right_node`) and all `parent`/`child` values from caller-supplied structured arrays fall within `[0, 2*hierarchy.shape[0]]` before use, and raise `ValueError` at the top of each `cpdef` entry point on violation.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:117, sklearn/cluster/_hdbscan/_tree.pyx:431, sklearn/cluster/_hdbscan/_tree.pyx:56, sklearn/cluster/_hdbscan/_tree.pyx:359]
```
</details>

### [LOW] `_do_labelling` scalar/array truthiness on `parent_lambda >= threshold` [out-of-theme]

Single-cluster branch reached with a point `n` that appears zero times or more than once as a child in the condensed tree → `ValueError: The truth value of an array with more than one element is ambiguous` bubbles up from an internal Cython routine instead of a clear diagnostic; on numpy futures, even the size-1 path may warn. Extract a scalar explicitly (`parent_lambda = float(lambda_array[child_array == n][0])`) and defensively handle the zero-match case with a raised `ValueError` describing the tree inconsistency.

<details><summary>verbatim finding</summary>

```
### F50 — `_do_labelling` scalar/array truthiness on `parent_lambda >= threshold` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498-506 — `parent_lambda = lambda_array[child_array == n]` is a 1-D numpy slice (possibly length 0 or >1). The subsequent `if parent_lambda >= threshold:` is then a `DeprecationWarning`-eligible truth-value-of-array test (raises `ValueError` if size != 1).
scenario: "Single-cluster branch reached with a point `n` that appears zero times or more than once as a child in the condensed tree → `ValueError: The truth value of an array with more than one element is ambiguous` bubbles up from an internal Cython routine instead of a clear diagnostic; on numpy futures, even the size-1 path may warn."
contract: Extract a scalar explicitly (`parent_lambda = float(lambda_array[child_array == n][0])`) and defensively handle the zero-match case with a raised `ValueError` describing the tree inconsistency.
instances: single-instance
```
</details>

### [LOW] `_do_labelling` KeyError when a component's root ID is missing from `cluster_label_map` [out-of-theme]

A malformed condensed tree (e.g., empty `clusters` set from `_get_clusters` in a corner case with an all-noise leaf method result) → `_do_labelling` raises `KeyError` at line 494 → partial `self.labels_` never gets returned; the exception surfaces out of `fit` without cleanup of the intermediate `self._single_linkage_tree_`. `_do_labelling` must fall back to `NOISE` (or raise a well-typed ValueError) when the discovered cluster root is not present in `cluster_label_map`, so that labeling cannot leak a bare KeyError.

<details><summary>verbatim finding</summary>

```
### F51 — `_do_labelling` KeyError when a component's root ID is missing from `cluster_label_map` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-508 — for each point, `cluster = union_find.find(n)`; if `cluster != root_cluster`, the code unconditionally does `label = cluster_label_map[cluster]`. `cluster_label_map` is built at hdbscan.py caller side from `sorted(list(clusters))`, but nothing guarantees that every root a point can be unioned to is present in `clusters`. When a point ends up in a union-find component whose root is neither `root_cluster` nor a selected cluster, this KeyError aborts labeling mid-loop, leaving `self.labels_` unset.
scenario: "A malformed condensed tree (e.g., empty `clusters` set from `_get_clusters` in a corner case with an all-noise leaf method result) → `_do_labelling` raises `KeyError` at line 494 → partial `self.labels_` never gets returned; the exception surfaces out of `fit` without cleanup of the intermediate `self._single_linkage_tree_`."
contract: `_do_labelling` must fall back to `NOISE` (or raise a well-typed ValueError) when the discovered cluster root is not present in `cluster_label_map`, so that labeling cannot leak a bare KeyError.
instances: single-instance
```
</details>

### [LOW] `_condense_tree` `ignore` array can be indexed out of range

A hierarchy whose BFS from root does not visit every id in `[0, 2*n_samples)` (any real hierarchy with size < 2*n_samples nodes reachable, e.g., due to relabeling gaps) → `len(node_list) < 2*n_samples` → `ignore[sub_node]` writes past the end → IndexError inside `_condense_tree`, aborting `fit` with a partial estimator. Size `ignore` by the maximum node id that can appear in the hierarchy (`2 * hierarchy.shape[0] + 1`), not by `len(node_list)`.

<details><summary>verbatim finding</summary>

```
### F52 — `_condense_tree` `ignore` array can be indexed out of range
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148-231 — `node_list = bfs_from_hierarchy(hierarchy, root)`; `ignore = np.zeros(len(node_list), dtype=bool)`. Later `ignore[sub_node] = True` (lines 205, 212, 221, 230) writes at indices that come from `bfs_from_hierarchy` — which returns raw node ids in the range `[0, 2*n_samples)` — not offsets into `node_list`. Only when the BFS is a complete walk of every node id in that range does `len(node_list) == 2*n_samples`; otherwise writes at index >= `len(node_list)` raise IndexError. The `ignore[node]` read at line 164 has the same problem.
scenario: "A hierarchy whose BFS from root does not visit every id in `[0, 2*n_samples)` (any real hierarchy with size < 2*n_samples nodes reachable, e.g., due to relabeling gaps) → `len(node_list) < 2*n_samples` → `ignore[sub_node]` writes past the end → IndexError inside `_condense_tree`, aborting `fit` with a partial estimator."
contract: Size `ignore` by the maximum node id that can appear in the hierarchy (`2 * hierarchy.shape[0] + 1`), not by `len(node_list)`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:161, sklearn/cluster/_hdbscan/_tree.pyx:164, sklearn/cluster/_hdbscan/_tree.pyx:205, sklearn/cluster/_hdbscan/_tree.pyx:212, sklearn/cluster/_hdbscan/_tree.pyx:221, sklearn/cluster/_hdbscan/_tree.pyx:230]
```
</details>

### [LOW] `HIERARCHY_dtype` / `CONDENSED_dtype` array shapes documented as `(n_samples,)` instead of `(n_samples - 1,)`

Consumer of `_condense_tree`/`labelling_at_cut` sizes buffers based on the documented `(n_samples,)` shape → off-by-one buffer / reader mismatch Align the Cython docstrings to `(n_samples - 1,)` to match both the implementation and the Python-side docstring.

<details><summary>verbatim finding</summary>

```
### F89 — `HIERARCHY_dtype` / `CONDENSED_dtype` array shapes documented as `(n_samples,)` instead of `(n_samples - 1,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 documents `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`, sklearn/cluster/_hdbscan/_tree.pyx:138 documents `condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype`, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656 repeat the same claim. However, the actual construction in sklearn/cluster/_hdbscan/_linkage.pyx:252 uses `np.zeros(n_samples - 1, ...)` and callers such as sklearn/cluster/_hdbscan/hdbscan.py:219 correctly document shape `(n_samples - 1,)`. The Cython-side shape contract is off-by-one and disagrees with the Python-side contract.
scenario: "Consumer of `_condense_tree`/`labelling_at_cut` sizes buffers based on the documented `(n_samples,)` shape → off-by-one buffer / reader mismatch"
contract: Align the Cython docstrings to `(n_samples - 1,)` to match both the implementation and the Python-side docstring.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]
```
</details>

### [LOW] `_get_clusters` `stabilities` return contract broken (docstring promises 3-tuple, code returns 2-tuple)

External code (e.g. a follow-up feature or user script that calls `_get_clusters` directly) unpacks the documented 3-tuple → `ValueError: not enough values to unpack` Remove the `stabilities` entry from the `_get_clusters` Returns block so the documented contract matches the actual 2-tuple return.

<details><summary>verbatim finding</summary>

```
### F91 — `_get_clusters` `stabilities` return contract broken (docstring promises 3-tuple, code returns 2-tuple)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:681-690 documents three return values (`labels`, `probabilities`, `stabilities`), but the actual `return` at sklearn/cluster/_hdbscan/_tree.pyx:797 is `return (labels, probs)` — a 2-tuple. `tree_to_labels` at sklearn/cluster/_hdbscan/_tree.pyx:70 unpacks only two values, so the caller conforms to the implementation, but the docstring's third return (`stabilities`) is a broken contract for any external consumer of `_get_clusters`.
scenario: "External code (e.g. a follow-up feature or user script that calls `_get_clusters` directly) unpacks the documented 3-tuple → `ValueError: not enough values to unpack`"
contract: Remove the `stabilities` entry from the `_get_clusters` Returns block so the documented contract matches the actual 2-tuple return.
instances: single-instance
```
</details>

### [LOW] `_compute_stability.result_pre_dict` typed as memoryview but consumed by `dict()` requiring iterable of pairs

A future Cython version where iterating a 2-D memoryview yields 1-D memoryview rows rather than ndarray rows → `dict()` fails with `TypeError: cannot convert dictionary update sequence element #0 to a sequence`. `result_pre_dict` must remain a `cnp.ndarray[cnp.float64_t, ndim=2]` (not a memoryview) since it is passed to `dict()` which relies on numpy row iteration semantics.

<details><summary>verbatim finding</summary>

```
### F93 — `_compute_stability.result_pre_dict` typed as memoryview but consumed by `dict()` requiring iterable of pairs
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:246 declares `cnp.float64_t[:, :] result_pre_dict`; :269-274 assigns it via `np.vstack(...).T` and then :276 returns `dict(result_pre_dict)`. `dict()` on a typed memoryview iterates rows as memoryviews, not as 2-tuples, so the conversion depends on the underlying ndarray view being extractable — the memoryview type is documentation drift versus the actual use.
scenario: "A future Cython version where iterating a 2-D memoryview yields 1-D memoryview rows rather than ndarray rows → `dict()` fails with `TypeError: cannot convert dictionary update sequence element #0 to a sequence`."
contract: `result_pre_dict` must remain a `cnp.ndarray[cnp.float64_t, ndim=2]` (not a memoryview) since it is passed to `dict()` which relies on numpy row iteration semantics.
instances: single-instance
```
</details>

### [LOW] `bfs_from_cluster_tree.children` typed as `intp_t` but populated from a structured field of `intp_t` via boolean-indexed lookup

A caller relying on `process_queue` being a fresh C-contiguous array after the loop → the actual object is a numpy view into `children` with unknown contiguity, breaking any code that would forward it to another `cnp.ndarray[..., mode='c']`-typed parameter. The reassignment on :294 must explicitly materialise as `np.ascontiguousarray(...)` or the declared type at :286 must drop the C-contiguity implication.

<details><summary>verbatim finding</summary>

```
### F94 — `bfs_from_cluster_tree.children` typed as `intp_t` but populated from a structured field of `intp_t` via boolean-indexed lookup
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:289 declares `cnp.ndarray[cnp.intp_t, ndim=1] children = condensed_tree['child']` and :286 declares `cnp.ndarray[cnp.intp_t, ndim=1] process_queue = np.array([bfs_root], dtype=np.intp)`; :294 assigns `process_queue = children[np.isin(parents, process_queue)]`. The typed decl at :286 is `cnp.ndarray[cnp.intp_t, …]` but the reassignment does not enforce contiguity or ownership — the previous typed slot is silently rebound to a view.
scenario: "A caller relying on `process_queue` being a fresh C-contiguous array after the loop → the actual object is a numpy view into `children` with unknown contiguity, breaking any code that would forward it to another `cnp.ndarray[..., mode='c']`-typed parameter."
contract: The reassignment on :294 must explicitly materialise as `np.ascontiguousarray(...)` or the declared type at :286 must drop the C-contiguity implication.
instances: single-instance
```
</details>

### [LOW] Duplicate `births` allocation in `_compute_stability`

Cut-and-paste left two copies of the same allocation → wastes an allocation per call and is a maintenance trap: a future edit that changes one but not the other yields inconsistent init state. Delete the redundant `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252, keeping only the line 254 allocation.

<details><summary>verbatim finding</summary>

```
### F106 — Duplicate `births` allocation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is immediately overwritten by the identical statement on line 254 `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`. The first allocation is dead work (its buffer is orphaned before any read).
scenario: "Cut-and-paste left two copies of the same allocation → wastes an allocation per call and is a maintenance trap: a future edit that changes one but not the other yields inconsistent init state."
contract: Delete the redundant `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252, keeping only the line 254 allocation.
instances: single-instance
```
</details>

### [LOW] Cython entry-point defaults for `min_cluster_size` (10) drift from the `HDBSCAN` estimator default (5)

downstream user (or test) calls `_condense_tree(tree)` directly relying on the same 5 default → gets a silently different pruning threshold than the public estimator produces Align the Cython defaults to 5 (matching the public estimator) or drop the defaults entirely so callers must pass an explicit value from the single source (`HDBSCAN.min_cluster_size`).

<details><summary>verbatim finding</summary>

```
### F110 — Cython entry-point defaults for `min_cluster_size` (10) drift from the `HDBSCAN` estimator default (5)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:58 (`tree_to_labels(..., cnp.intp_t min_cluster_size=10, ...)`) and sklearn/cluster/_hdbscan/_tree.pyx:119 (`_condense_tree(..., cnp.intp_t min_cluster_size=10)`) both default to 10; sklearn/cluster/_hdbscan/hdbscan.py:649 has `min_cluster_size=5` for the `HDBSCAN.__init__` and sklearn/cluster/_hdbscan/hdbscan.py:923 has `dbscan_clustering(..., min_cluster_size=5)`.
scenario: "downstream user (or test) calls `_condense_tree(tree)` directly relying on the same 5 default → gets a silently different pruning threshold than the public estimator produces"
contract: Align the Cython defaults to 5 (matching the public estimator) or drop the defaults entirely so callers must pass an explicit value from the single source (`HDBSCAN.min_cluster_size`).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/hdbscan.py:649, sklearn/cluster/_hdbscan/hdbscan.py:923]
```
</details>

### [LOW] `.shape[0]` pattern retained in `_tree.pyx` and `_reachability.pyx` contradicts plan bullet #2

Reader trusts the plan and uses `git grep '\\.shape\\[0\\]'` to confirm no ndarray uses remain → discovers ~15 residual instances, does not know whether these are memoryviews (allowed) or ndarrays (expected to be `len(*)`), must audit each by hand → the plan's claim is falsified. Audit every `.shape[0]` in the added HDBSCAN files and rewrite the ndarray cases to `len(*)` so the code matches the plan claim.

<details><summary>verbatim finding</summary>

```
### F111 — `.shape[0]` pattern retained in `_tree.pyx` and `_reachability.pyx` contradicts plan bullet #2
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:90, :145, :146, :396, :531, :563 still use `.shape[0]` on ndarray/memoryview objects (e.g. `hierarchy.shape[0] + 1`, `2 * hierarchy.shape[0]`, `2 * linkage.shape[0]`, `labels.shape[0]`, `children.shape[0]`); sklearn/cluster/_hdbscan/_reachability.pyx:103 and :132 similarly use `distance_matrix.shape[0]`; and sklearn/cluster/_hdbscan/hdbscan.py:758, :764, :767, :840, :846, :852, :896, :902, :907, :909 still use `X.shape[0]` for ndarray. The PR plan explicitly lists "Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects" as a novel change of this PR, but the replacement was applied inconsistently — sister file `_linkage.pyx` similarly retains raw `.shape[0]` in cdef.
scenario: "Reader trusts the plan and uses `git grep '\\.shape\\[0\\]'` to confirm no ndarray uses remain → discovers ~15 residual instances, does not know whether these are memoryviews (allowed) or ndarrays (expected to be `len(*)`), must audit each by hand → the plan's claim is falsified."
contract: Audit every `.shape[0]` in the added HDBSCAN files and rewrite the ndarray cases to `len(*)` so the code matches the plan claim.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:396, sklearn/cluster/_hdbscan/_tree.pyx:531, sklearn/cluster/_hdbscan/_tree.pyx:563, sklearn/cluster/_hdbscan/_reachability.pyx:103, sklearn/cluster/_hdbscan/_reachability.pyx:132, sklearn/cluster/_hdbscan/hdbscan.py:385, sklearn/cluster/_hdbscan/hdbscan.py:387, sklearn/cluster/_hdbscan/hdbscan.py:752, sklearn/cluster/_hdbscan/hdbscan.py:758, sklearn/cluster/_hdbscan/hdbscan.py:761, sklearn/cluster/_hdbscan/hdbscan.py:764, sklearn/cluster/_hdbscan/hdbscan.py:767, sklearn/cluster/_hdbscan/hdbscan.py:840, sklearn/cluster/_hdbscan/hdbscan.py:846, sklearn/cluster/_hdbscan/hdbscan.py:852, sklearn/cluster/_hdbscan/hdbscan.py:896, sklearn/cluster/_hdbscan/hdbscan.py:902, sklearn/cluster/_hdbscan/hdbscan.py:907, sklearn/cluster/_hdbscan/hdbscan.py:909]
```
</details>

### [LOW] `NOISE = -1` sentinel not single-sourced across HDBSCAN module

A follow-up PR decides to shift the noise label from `-1` to another sentinel to disambiguate it from user labels → editor must find every hard-coded `-1` across `.py`, `.pyx`, tests, and the example, whereas `NOISE` gets renamed once; misses drift silently past unit tests that themselves hard-code `-1`. Extend `_OUTLIER_ENCODING` in `hdbscan.py` with a `"noise"` entry whose `label` is `-1`, and reference it from `_weighted_cluster_center`, `dbscan_clustering`, the tests, and the example so the noise sentinel has exactly one source of truth across the module.

<details><summary>verbatim finding</summary>

```
### F113 — `NOISE = -1` sentinel not single-sourced across HDBSCAN module
severity: low
evidence: `sklearn/cluster/_hdbscan/_tree.pyx:40` defines the sentinel `cdef cnp.intp_t NOISE = -1` and uses it at lines 420, 426, 492. However the Python-side `hdbscan.py` (line 63 comment refers to `-1 noise label`, line 895 uses hardcoded `{-1, -2}`, and `dbscan_clustering` docstring at line 955 refers to `-1`), tests (test_hdbscan.py line 43 `{-1}`, line 377 `int(-1 in hdb.labels_)`), and the example (examples/cluster/plot_hdbscan.py lines 41, 50, 53, 55) all hard-code the `-1` literal instead of referencing a single-sourced constant. The Cython constant is not exposed on the Python side and `_OUTLIER_ENCODING` intentionally starts at `-2` (see the comment at hdbscan.py:63 "extensions to the -1 noise label"), leaving the fundamental noise sentinel with no single source of truth across the Python and Cython halves of the module.
scenario: "A follow-up PR decides to shift the noise label from `-1` to another sentinel to disambiguate it from user labels → editor must find every hard-coded `-1` across `.py`, `.pyx`, tests, and the example, whereas `NOISE` gets renamed once; misses drift silently past unit tests that themselves hard-code `-1`."
contract: Extend `_OUTLIER_ENCODING` in `hdbscan.py` with a `"noise"` entry whose `label` is `-1`, and reference it from `_weighted_cluster_center`, `dbscan_clustering`, the tests, and the example so the noise sentinel has exactly one source of truth across the module.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:414, sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/tests/test_hdbscan.py:43, sklearn/cluster/tests/test_hdbscan.py:377, examples/cluster/plot_hdbscan.py:41, examples/cluster/plot_hdbscan.py:50, examples/cluster/plot_hdbscan.py:53, examples/cluster/plot_hdbscan.py:55]
```
</details>

### [LOW] Duplicate `births` allocation is dead code

Reader/maintainer inspects `_compute_stability` → sees a duplicated allocation and must reason whether the first line has a side effect (it does not) Remove the assignment at line 252 so `births` is allocated exactly once, on the line preceding the population loop.

<details><summary>verbatim finding</summary>

```
### F121 — Duplicate `births` allocation is dead code
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed on line 252 with the assigned array never read; the identical statement on line 254 immediately overwrites `births` with a fresh array.
scenario: "Reader/maintainer inspects `_compute_stability` → sees a duplicated allocation and must reason whether the first line has a side effect (it does not)"
contract: Remove the assignment at line 252 so `births` is allocated exactly once, on the line preceding the population loop.
instances: single-instance
```
</details>

### [LOW] Redundant `<cnp.intp_t>` cast asymmetric with sibling branch

Reader compares the left/right branches → asymmetry falsely suggests the two counts have different types or that the cast matters Drop the `<cnp.intp_t>` cast on line 182 so the two branches are symmetric.

<details><summary>verbatim finding</summary>

```
### F122 — Redundant `<cnp.intp_t>` cast asymmetric with sibling branch
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:181-182 — `right_count = <cnp.intp_t> hierarchy[right - n_samples].cluster_size` uses a redundant cast; the symmetric branch at line 177 `left_count = hierarchy[left - n_samples].cluster_size` correctly does not, and `cluster_size` is already declared `intp_t` in `_tree.pxd:38`. The cast is superseded/leftover code from an earlier iteration.
scenario: "Reader compares the left/right branches → asymmetry falsely suggests the two counts have different types or that the cast matters"
contract: Drop the `<cnp.intp_t>` cast on line 182 so the two branches are symmetric.
instances: single-instance
```
</details>

### [LOW] Overwritten `births` allocation is dead code

reader tries to understand initialization → confused by the duplication; small allocation cost happens twice on every clustering. Delete the duplicate line 252 (or 254).

<details><summary>verbatim finding</summary>

```
### F146 — Overwritten `births` allocation is dead code
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` followed immediately by `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`; the first allocation is unconditionally overwritten before use.
scenario: "reader tries to understand initialization → confused by the duplication; small allocation cost happens twice on every clustering."
contract: Delete the duplicate line 252 (or 254).
instances: single-instance
```
</details>

### [LOW] `_do_labelling` uses implicitly-typed Python objects `label` and `parent_lambda` inside a hot `cdef` loop

Under `allow_single_cluster=True` with a condensed tree where the same `child==n` appears twice → `parent_lambda` becomes length-2 and `if parent_lambda >= threshold` raises `ValueError: The truth value of an array with more than one element is ambiguous` Declare `cdef cnp.intp_t label` and extract a scalar for `parent_lambda` (e.g. `parent_lambda = lambda_array[child_array == n][0]`) with the length-1 assumption made explicit.

<details><summary>verbatim finding</summary>

```
### F149 — `_do_labelling` uses implicitly-typed Python objects `label` and `parent_lambda` inside a hot `cdef` loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:492-508 — `label = NOISE`, `parent_lambda = lambda_array[child_array == n]`, and `if parent_lambda >= threshold:` are all untyped in the `cdef` block at line 467-474. `label` therefore boxes each per-sample write; `parent_lambda` is a numpy array compared with a scalar via `>=` — this yields a numpy bool array whose truthiness in an `if` triggers a `DeprecationWarning`/future error for non-length-1 arrays.
scenario: "Under `allow_single_cluster=True` with a condensed tree where the same `child==n` appears twice → `parent_lambda` becomes length-2 and `if parent_lambda >= threshold` raises `ValueError: The truth value of an array with more than one element is ambiguous`"
contract: Declare `cdef cnp.intp_t label` and extract a scalar for `parent_lambda` (e.g. `parent_lambda = lambda_array[child_array == n][0]`) with the length-1 assumption made explicit.
instances: single-instance
```
</details>

### [LOW] `_condense_tree` writes bool dtype into a `uint8` view

future NumPy where `bool` dtype layout changes, or a maintainer refactoring the memoryview annotation → silent breakage of a bare-metal assumption; comment/name does not explain the intentional aliasing. Use `dtype=np.uint8` in the `np.zeros` call to match the declared memoryview type.

<details><summary>verbatim finding</summary>

```
### F150 — `_condense_tree` writes bool dtype into a `uint8` view
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:151, 161 — `cnp.uint8_t[::1] ignore` then `ignore = np.zeros(len(node_list), dtype=bool)`. `np.bool_` and `np.uint8` are distinct dtypes; the memoryview assignment relies on identical itemsize but the surface-level type mismatch obscures intent (why declare `uint8` and initialize with `bool`?).
scenario: "future NumPy where `bool` dtype layout changes, or a maintainer refactoring the memoryview annotation → silent breakage of a bare-metal assumption; comment/name does not explain the intentional aliasing."
contract: Use `dtype=np.uint8` in the `np.zeros` call to match the declared memoryview type.
instances: single-instance
```
</details>

### [LOW] `n_samples` docstring shape mismatches actual shape for tree/condensed arrays

user or maintainer following the `_tree.pyx` docstring → allocates or slices with wrong length; the array-shape claim is a lying contract at the function boundary. Update `_tree.pyx` docstrings to state `(n_samples - 1,)` for `HIERARCHY_dtype` arrays and clarify that `CONDENSED_dtype` arrays have variable length (edge count of the condensed tree).

<details><summary>verbatim finding</summary>

```
### F151 — `n_samples` docstring shape mismatches actual shape for tree/condensed arrays
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129, 138, 447, 656 — Parameters/Returns say `ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` (or `CONDENSED_dtype`), yet the actual `HIERARCHY_t` array is size `n_samples - 1` (single linkage tree encoding) and the condensed tree is size variable ≠ `n_samples`. `hdbscan.py:101, 148, 219, 325, 360` correctly document the linkage tree shape as `(n_samples - 1,)`, so this is inconsistent documentation, not a naming choice.
scenario: "user or maintainer following the `_tree.pyx` docstring → allocates or slices with wrong length; the array-shape claim is a lying contract at the function boundary."
contract: Update `_tree.pyx` docstrings to state `(n_samples - 1,)` for `HIERARCHY_dtype` arrays and clarify that `CONDENSED_dtype` arrays have variable length (edge count of the condensed tree).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]
```
</details>

### [LOW] `_get_clusters` reuses `n_samples` name for the largest per-point child id

condensed tree that (post-condensation) does not include every sample as a leaf → `n_samples` under-counted, downstream `max_cluster_size` sentinel at line 717 (`n_samples + 1`) becomes wrong and can silently trigger the `cluster_sizes[node] > max_cluster_size` branch at line 733. Pass or derive the true `n_samples` from the caller (e.g., single-linkage tree length + 1) instead of inferring it from a max-child computation.

<details><summary>verbatim finding</summary>

```
### F153 — `_get_clusters` reuses `n_samples` name for the largest per-point child id
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:714 — `n_samples = np.max(condensed_tree[condensed_tree['cluster_size'] == 1]['child']) + 1`. Elsewhere `n_samples` is derived from `hierarchy.shape[0] + 1` (line 90/146) or `root // 2 + 1` (line 397). Here it happens to equal the true sample count only if every sample appears as a leaf in the condensed tree — a fragile invariant that is never asserted.
scenario: "condensed tree that (post-condensation) does not include every sample as a leaf → `n_samples` under-counted, downstream `max_cluster_size` sentinel at line 717 (`n_samples + 1`) becomes wrong and can silently trigger the `cluster_sizes[node] > max_cluster_size` branch at line 733."
contract: Pass or derive the true `n_samples` from the caller (e.g., single-linkage tree length + 1) instead of inferring it from a max-child computation.
instances: single-instance
```
</details>

### [LOW] `TreeUnionFind.find` overwrites `is_component` with `False` for every non-root node but nothing reads it

Reader assumes `is_component` is load-bearing → wastes time tracing state that never influences output Remove the `is_component` field and its side-effect write in `find`.

<details><summary>verbatim finding</summary>

```
### F158 — `TreeUnionFind.find` overwrites `is_component` with `False` for every non-root node but nothing reads it
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:337 initializes `self.is_component = np.ones(size, dtype=np.uint8)`, and line 355 sets `self.is_component[x] = False` during path compression, yet `is_component` is never read anywhere in `_tree.pyx` (verified by grep of the file). This is speculative abstraction that also introduces a surprising side effect in an otherwise-pure `find`.
scenario: "Reader assumes `is_component` is load-bearing → wastes time tracing state that never influences output"
contract: Remove the `is_component` field and its side-effect write in `find`.
instances: single-instance
```
</details>

### [LOW] `bfs_from_hierarchy` returns internal nodes offset by `-n_samples` interleaved with raw sample ids

Maintainer relies on the return being uniform tree-space ids → indexes `hierarchy[node - n_samples]` for leaves and gets a negative index / silent misread Document (or explicitly convert) the mixed-coordinate return; e.g. "returned nodes are in tree-index space (leaf ids in `[0, n_samples)`, cluster ids in `[n_samples, 2*n_samples-1]`)".

<details><summary>verbatim finding</summary>

```
### F159 — `bfs_from_hierarchy` returns internal nodes offset by `-n_samples` interleaved with raw sample ids
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:99-103 — `process_queue = [x - n_samples for x in process_queue if x >= n_samples]` mutates `process_queue` from "tree indexes" to "hierarchy row indexes" mid-iteration, but the earlier `result.extend(process_queue)` at line 96 stored the un-offset values. Callers therefore receive a mixed list where entries `>= n_samples` are original tree ids and entries `< n_samples` are leaf sample ids; the "internal" indices used inside the loop are silently converted. No docstring warns of this dual coordinate system.
scenario: "Maintainer relies on the return being uniform tree-space ids → indexes `hierarchy[node - n_samples]` for leaves and gets a negative index / silent misread"
contract: Document (or explicitly convert) the mixed-coordinate return; e.g. "returned nodes are in tree-index space (leaf ids in `[0, n_samples)`, cluster ids in `[n_samples, 2*n_samples-1]`)".
instances: single-instance
```
</details>

### [LOW] `_get_clusters` uses `@cython.wraparound(True)` for the entire function, silently disabling a fast-path optimization

Future maintainer removes the `[-1]` use → the decorator survives as cargo-cult noise, silently preventing later `wraparound(False)` optimizations Replace the function-scope decorator with a `with cython.wraparound(True):` block around the single `node_list[-1]` access.

<details><summary>verbatim finding</summary>

```
### F161 — `_get_clusters` uses `@cython.wraparound(True)` for the entire function, silently disabling a fast-path optimization
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:641 — `@cython.wraparound(True)` applied at function scope while the file has no module-level `wraparound(False)`; the decorator is only necessary for the single `node_list[-1]` access at line 724. Applying it at function scope makes every array read pay the wraparound overhead and hides which line actually needs negative indexing.
scenario: "Future maintainer removes the `[-1]` use → the decorator survives as cargo-cult noise, silently preventing later `wraparound(False)` optimizations"
contract: Replace the function-scope decorator with a `with cython.wraparound(True):` block around the single `node_list[-1]` access.
instances: single-instance
```
</details>

### [LOW] `_condense_tree` and `_do_labelling` Returns claim shape `(n_samples,)` for structures whose length is not `n_samples`

Reader relying on the documented shape sizes downstream buffers as `n_samples` → truncates or overflows because the condensed tree is not that length Correct the shape annotations: `hierarchy` should read `(n_samples-1,)`; `condensed_tree` should be described as a variable-length edgelist (e.g. `(n_edges,)`), not `(n_samples,)`.

<details><summary>verbatim finding</summary>

```
### F183 — `_condense_tree` and `_do_labelling` Returns claim shape `(n_samples,)` for structures whose length is not `n_samples`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 documents `hierarchy : ndarray of shape (n_samples,)` for what is actually `(n_samples-1,)` (see `make_single_linkage` at sklearn/cluster/_hdbscan/_linkage.pyx:252 which returns `np.zeros(n_samples - 1, ...)`); sklearn/cluster/_hdbscan/_tree.pyx:138 documents the `condensed_tree` output as shape `(n_samples,)` although it is a variable-length edgelist; sklearn/cluster/_hdbscan/_tree.pyx:447 repeats the wrong `(n_samples,)` shape for `condensed_tree`.
scenario: "Reader relying on the documented shape sizes downstream buffers as `n_samples` → truncates or overflows because the condensed tree is not that length"
contract: Correct the shape annotations: `hierarchy` should read `(n_samples-1,)`; `condensed_tree` should be described as a variable-length edgelist (e.g. `(n_edges,)`), not `(n_samples,)`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]
```
</details>

### [LOW] `_compute_stability` contains a duplicated `births = np.full(...)` line signalling a stale copy-paste [out-of-theme]

Reader wonders which assignment is authoritative and whether there is a hidden side effect → wasted comprehension time, plus a redundant allocation Delete the first assignment (line 252) so a single, documented allocation remains.

<details><summary>verbatim finding</summary>

```
### F186 — `_compute_stability` contains a duplicated `births = np.full(...)` line signalling a stale copy-paste [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and 254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears twice with no intervening use; the first assignment is immediately overwritten
scenario: "Reader wonders which assignment is authoritative and whether there is a hidden side effect → wasted comprehension time, plus a redundant allocation"
contract: Delete the first assignment (line 252) so a single, documented allocation remains.
instances: single-instance
```
</details>

### [LOW] `_do_labelling` Returns docstring makes a claim that is not enforced by the code [out-of-theme]

clusters passed by _get_clusters miss a value present in labels → KeyError instead of the documented -1 Enforce the noise fall-through the docstring promises by defaulting to NOISE for unknown `cluster` keys in `_do_labelling`.

<details><summary>verbatim finding</summary>

```
### F187 — `_do_labelling` Returns docstring makes a claim that is not enforced by the code [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:462-464 — Returns says "a label of -1 denotes a noise assignment", but at lines 497-506 the `label` variable that is written into `result` may take the value of `cluster_label_map[cluster]` where `cluster` is the child index `n`; if `n` is not in `cluster_label_map`, this raises KeyError. That behavior isn't in the contract and there is no cross-check ensuring `cluster_label_map` covers every reachable value. This isn't a docs-only defect (the code lacks defensive handling for the contract the docstring promises), which is why I flag it out-of-theme.
scenario: "clusters passed by _get_clusters miss a value present in labels → KeyError instead of the documented -1"
contract: Enforce the noise fall-through the docstring promises by defaulting to NOISE for unknown `cluster` keys in `_do_labelling`.
instances: single-instance
```
</details>

### [LOW] `_get_clusters` `# (exclude root)` narration comment sits after the slice that already excluded it

Contributor refactors `_condense_tree`'s labelling to use a different id scheme (e.g. hashed ids) → silently breaks `_get_clusters` because the assumed topological ordering is no longer produced, and this comment does not tell them where to look Replace the vague comment with an explicit invariant: reference `_condense_tree` and its use of `next_label`, e.g. "This relies on `_condense_tree` assigning cluster ids via monotonically-increasing `next_label`, which guarantees parent id < child id (i.e. sort-by-id is a reverse topological order). If `_condense_tree`'s id scheme changes, replace this `sorted(...)` with an explicit topological sort.

<details><summary>verbatim finding</summary>

```
### F188 — `_get_clusters` `# (exclude root)` narration comment sits after the slice that already excluded it
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:702-705 — comment "Assume clusters are ordered by numeric id equivalent to a topological sort of the tree; This is valid given the current implementation above, so don't change that ... or if you do, change this accordingly!" is vague ("the current implementation above") and does not name which upstream function (`_condense_tree`) it relies on, nor what property (`next_label` monotonic increment ensuring children > parent) makes the sort valid.
scenario: "Contributor refactors `_condense_tree`'s labelling to use a different id scheme (e.g. hashed ids) → silently breaks `_get_clusters` because the assumed topological ordering is no longer produced, and this comment does not tell them where to look"
contract: Replace the vague comment with an explicit invariant: reference `_condense_tree` and its use of `next_label`, e.g. "This relies on `_condense_tree` assigning cluster ids via monotonically-increasing `next_label`, which guarantees parent id < child id (i.e. sort-by-id is a reverse topological order). If `_condense_tree`'s id scheme changes, replace this `sorted(...)` with an explicit topological sort."
instances: [sklearn/cluster/_hdbscan/_tree.pyx:702, sklearn/cluster/_hdbscan/_tree.pyx:710]
```
</details>

### [LOW] `_compute_stability` iterates the condensed_tree twice via `for condensed_node in condensed_tree`

Large condensed trees → stability computation runs in Python loop speed rather than numpy speed even though it is cpdef Cython Replace the two element-wise loops with vectorized numpy assignments/`np.add.at` on the extracted structured columns.

<details><summary>verbatim finding</summary>

```
### F210 — `_compute_stability` iterates the condensed_tree twice via `for condensed_node in condensed_tree`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:254-267 — two consecutive `for condensed_node in condensed_tree:` loops iterate the structured numpy array element-by-element from Python, defeating vectorization. `births[condensed_node.child] = condensed_node.value` and `result[result_index] += (lambda_val - births[parent]) * cluster_size` could be vectorized with `births[children_array] = values_array` and `np.add.at(result, parent_array - smallest_cluster, (values_array - births[parent_array]) * cluster_sizes_array)`.
scenario: "Large condensed trees → stability computation runs in Python loop speed rather than numpy speed even though it is cpdef Cython"
contract: Replace the two element-wise loops with vectorized numpy assignments/`np.add.at` on the extracted structured columns.
instances: single-instance
```
</details>

### [LOW] Duplicated `births = np.full(...)` allocation

Any HDBSCAN fit → one unused O(n) allocation per stability computation Delete the duplicate line so `births` is allocated exactly once.

<details><summary>verbatim finding</summary>

```
### F211 — Duplicated `births = np.full(...)` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice in a row with the same arguments; the first allocation is immediately discarded, wasting O(n) time and memory allocation per call.
scenario: "Any HDBSCAN fit → one unused O(n) allocation per stability computation"
contract: Delete the duplicate line so `births` is allocated exactly once.
instances: single-instance
```
</details>

### [LOW] `_do_labelling` allocates `result` sized by `root_cluster` instead of `n_samples`

someone changes the relabelling scheme in `_condense_tree` → `_do_labelling`'s output array silently mis-sizes, producing wrong-length `labels_`; performance-wise the code additionally scans `parent_array` for its min before doing any real work. Compute the result length as `n_samples` explicitly (e.g. from the hierarchy shape passed through) rather than via `np.min(parent_array)`, and document the invariant that `root_cluster == n_samples`.

<details><summary>verbatim finding</summary>

```
### F217 — `_do_labelling` allocates `result` sized by `root_cluster` instead of `n_samples`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-481 — `root_cluster = np.min(parent_array); result = np.empty(root_cluster, dtype=np.intp)`. The value `root_cluster` is the smallest parent id, which happens to equal `n_samples` because `_condense_tree` starts labeling internal nodes at `n_samples + 1` and roots at `n_samples`. Using `root_cluster` as a length is implicit coupling and hides the actual size (`n_samples`), and requires materializing `parent_array` and taking its min just to compute the output length.
scenario: "someone changes the relabelling scheme in `_condense_tree` → `_do_labelling`'s output array silently mis-sizes, producing wrong-length `labels_`; performance-wise the code additionally scans `parent_array` for its min before doing any real work."
contract: Compute the result length as `n_samples` explicitly (e.g. from the hierarchy shape passed through) rather than via `np.min(parent_array)`, and document the invariant that `root_cluster == n_samples`.
instances: single-instance
```
</details>

### [LOW] `bfs_from_hierarchy` is invoked repeatedly with the same subtree roots in `_condense_tree`

hierarchies where many merges have one branch smaller than `min_cluster_size` (typical for `min_cluster_size` > 2) → the condensation runs BFS from thousands of internal nodes, each walking a Python-list queue with per-iteration `[x - n_samples for x in process_queue if x >= n_samples]` comprehensions, making condensation super-linear in n. Replace `bfs_from_hierarchy` with a preallocated intp buffer + integer head/tail indices (or an iterative stack traversal in Cython) so each BFS visits O(subtree_size) with no Python list churn; ensure the top-level `node_list` traversal shares this scratch buffer.

<details><summary>verbatim finding</summary>

```
### F218 — `bfs_from_hierarchy` is invoked repeatedly with the same subtree roots in `_condense_tree`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148,200,207,216,225 — `_condense_tree` calls `bfs_from_hierarchy(hierarchy, root)` once up front to obtain `node_list`, then inside the per-node loop calls `bfs_from_hierarchy(hierarchy, left)` and/or `bfs_from_hierarchy(hierarchy, right)` for each sub-min-cluster branch, each of which redoes a Python-level BFS that walks Python lists and allocates fresh `process_queue`/`next_queue` lists on every call.
scenario: "hierarchies where many merges have one branch smaller than `min_cluster_size` (typical for `min_cluster_size` > 2) → the condensation runs BFS from thousands of internal nodes, each walking a Python-list queue with per-iteration `[x - n_samples for x in process_queue if x >= n_samples]` comprehensions, making condensation super-linear in n."
contract: Replace `bfs_from_hierarchy` with a preallocated intp buffer + integer head/tail indices (or an iterative stack traversal in Cython) so each BFS visits O(subtree_size) with no Python list churn; ensure the top-level `node_list` traversal shares this scratch buffer.
instances: single-instance
```
</details>

### [LOW] `min_samples` bound check runs after silently dropping non-finite rows, producing misleading validation

User passes `min_samples=200` on 200-row input where 5 rows are non-finite → error message reports `n_samples=195` even though the user supplied 200 rows, hindering diagnosis. Compute and check `min_samples > n_samples_original` against the pre-reduction sample count, or clearly qualify the error message as "the number of finite samples in X".

<details><summary>verbatim finding</summary>

```
### F11 — `min_samples` bound check runs after silently dropping non-finite rows, producing misleading validation
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:758-762 — `if self._min_samples > X.shape[0]: raise ValueError(f"min_samples ({self._min_samples}) must be at most the number of samples in X ({X.shape[0]})")` runs after `X = X[finite_index]` at line 733; the reported "number of samples in X" is the count of finite rows, not the user-supplied `X.shape[0]`.
scenario: "User passes `min_samples=200` on 200-row input where 5 rows are non-finite → error message reports `n_samples=195` even though the user supplied 200 rows, hindering diagnosis."
contract: Compute and check `min_samples > n_samples_original` against the pre-reduction sample count, or clearly qualify the error message as "the number of finite samples in X".
instances: single-instance
```
</details>

### [LOW] `csgraph.connected_components` return value is treated as a comparable scalar without unpacking, silently changing meaning if `return_labels` semantics change

Older/newer scipy returns tuple/other type → `TypeError` at runtime or a truthy-but-meaningless comparison Explicitly unpack with `n_components = csgraph.connected_components(...)` and compare `n_components > 1`.

<details><summary>verbatim finding</summary>

```
### F13 — `csgraph.connected_components` return value is treated as a comparable scalar without unpacking, silently changing meaning if `return_labels` semantics change
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:111-116 — `if (csgraph.connected_components(mutual_reachability, directed=False, return_labels=False) > 1):`. `return_labels=False` currently returns a single `int`, so `> 1` is well-defined. However, historical scipy versions have returned a tuple even with `return_labels=False` in some contexts; the code is not defensive against that and will silently do wrong comparisons (`tuple > 1` raises `TypeError` on Python 3 — which is at least loud — but the reliance on a scipy implementation detail is fragile).
scenario: "Older/newer scipy returns tuple/other type → `TypeError` at runtime or a truthy-but-meaningless comparison"
contract: Explicitly unpack with `n_components = csgraph.connected_components(...)` and compare `n_components > 1`.
instances: single-instance
```
</details>

### [LOW] `HIERARCHY_dtype` `cluster_size` field is typed `np.intp` but `remap_single_linkage_tree` writes `last_cluster_size + 1` values that grow monotonically without bounds check

Refactor passes a list (not set) with duplicates in `non_finite` → `cluster_size` values grow past the true total sample count silently Derive final `cluster_size` from `len(tree)+1` invariant rather than counting increments.

<details><summary>verbatim finding</summary>

```
### F14 — `HIERARCHY_dtype` `cluster_size` field is typed `np.intp` but `remap_single_linkage_tree` writes `last_cluster_size + 1` values that grow monotonically without bounds check
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:387-391 — `last_cluster_size = tree[tree.shape[0] - 1]["cluster_size"]` then in a loop `last_cluster_size += 1` and stores `last_cluster_size + 1` per outlier. For `n_samples > 2^{31}` on Windows (where `intp` is 32-bit), this can overflow, but more importantly the reported `cluster_size` for each outlier row is `last_cluster_size + 1`, incremented per iteration — this is claiming the outlier cluster grows one-by-one, which is consistent with them being sequentially merged, but adds `outlier_count` to the reported cluster_size on the last row rather than the true final total (`n_samples`) if there are duplicates in `non_finite`. Since `non_finite=set(...)` removes duplicates, this happens to be correct today.
scenario: "Refactor passes a list (not set) with duplicates in `non_finite` → `cluster_size` values grow past the true total sample count silently"
contract: Derive final `cluster_size` from `len(tree)+1` invariant rather than counting increments.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_prims` computes `NearestNeighbors(..., p=None)` which is incompatible with `metric='minkowski'` default

User calls `HDBSCAN(metric='minkowski', metric_params={'p': 3})` with `algorithm='kdtree'` → `p=None` overrides; nearest-neighbor step may error or silently use `p=2` Do not pass `p=None`; let `NearestNeighbors` use its own default and route `p` through `metric_params`.

<details><summary>verbatim finding</summary>

```
### F16 — `_hdbscan_prims` computes `NearestNeighbors(..., p=None)` which is incompatible with `metric='minkowski'` default
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:332-340 — passes `p=None` unconditionally. `NearestNeighbors.__init__` documents `p=2` default and accepts numeric `p`. Passing `p=None` circumvents the metric-specific `p` parameter and may cause distance computations to ignore user-provided `p` in `metric_params` for `minkowski`-family metrics.
scenario: "User calls `HDBSCAN(metric='minkowski', metric_params={'p': 3})` with `algorithm='kdtree'` → `p=None` overrides; nearest-neighbor step may error or silently use `p=2`"
contract: Do not pass `p=None`; let `NearestNeighbors` use its own default and route `p` through `metric_params`.
instances: single-instance
```
</details>

### [LOW] Sparse `_get_finite_row_indices` misses non-finite structural patterns

user constructs a sparse distance matrix that omits distances beyond some threshold (typical KNN graph) → `_get_finite_row_indices` reports every row as finite, so no rows are separated out for the non-finite handling path even when the caller expected that behavior derive per-row non-finiteness from the CSR triplet directly, e.g. `finite_rows = np.array([np.all(np.isfinite(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])) for i in range(matrix.shape[0])])`.

<details><summary>verbatim finding</summary>

```
### F18 — Sparse `_get_finite_row_indices` misses non-finite structural patterns
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 iterates `matrix.tolil().data` and calls `np.all(np.isfinite(row))`, but LIL `.data` only exposes explicitly stored non-zero values. Rows in which the sparse pattern encodes missing distances via structural absence (rather than explicit `np.inf` / `np.nan`) will always be classified as fully finite; conversely, rows containing only structural zeros (empty `row`) also pass `np.all(np.isfinite([])) == True`.
scenario: "user constructs a sparse distance matrix that omits distances beyond some threshold (typical KNN graph) → `_get_finite_row_indices` reports every row as finite, so no rows are separated out for the non-finite handling path even when the caller expected that behavior"
contract: derive per-row non-finiteness from the CSR triplet directly, e.g. `finite_rows = np.array([np.all(np.isfinite(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])) for i in range(matrix.shape[0])])`.
instances: single-instance
```
</details>

### [LOW] `metric` callable is not validated against the `precomputed`/tree paths, letting user code silently run under an incompatible algorithm

User accidentally passes a callable while intending `metric='precomputed'` (e.g., `HDBSCAN(metric=lambda X: X)`) → the symmetry / square-matrix guardrails on lines 223–234 are skipped and the callable is invoked as a pairwise-distance function on the user's precomputed matrix, silently producing wrong (or infinite-loop-inducing) distances instead of an early input-validation error. Explicitly reject callable `metric` when the intent is a precomputed matrix, and add a positive `metric == "precomputed"` guard in `_hdbscan_brute` that raises when a non-string callable slips through.

<details><summary>verbatim finding</summary>

```
### F24 — `metric` callable is not validated against the `precomputed`/tree paths, letting user code silently run under an incompatible algorithm
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:626 — `"metric": [StrOptions(FAST_METRICS | {"precomputed"}), callable]`; combined with lines 772–783 the callable metric is only rejected for `kdtree`/`balltree`, never for `precomputed`. In `_hdbscan_brute` at line 238–240, when `metric="precomputed"` is not selected the callable is passed through `pairwise_distances`, but the branch at line 222 uses object identity `metric == "precomputed"` — a truthy callable passed while user believes they set `precomputed` would silently bypass the symmetry check.
scenario: "User accidentally passes a callable while intending `metric='precomputed'` (e.g., `HDBSCAN(metric=lambda X: X)`) → the symmetry / square-matrix guardrails on lines 223–234 are skipped and the callable is invoked as a pairwise-distance function on the user's precomputed matrix, silently producing wrong (or infinite-loop-inducing) distances instead of an early input-validation error."
contract: Explicitly reject callable `metric` when the intent is a precomputed matrix, and add a positive `metric == "precomputed"` guard in `_hdbscan_brute` that raises when a non-string callable slips through.
instances: single-instance
```
</details>

### [LOW] `self._raw_data = X` retains a reference to caller-owned input, defeating the `copy=False` boundary

User trains `HDBSCAN(copy=False)` on a large array and then mutates or frees the underlying buffer expecting scikit-learn to be done with it → the estimator silently keeps a live reference (via `_raw_data`), so mutations reflect into the estimator's stored state, and any later pickling of the estimator serializes the entire raw training set — an unexpected data-leak channel when the estimator is persisted or shipped between processes. Drop the `_raw_data` attribute after `fit` and reconstruct only what downstream methods actually need, so the estimator does not silently retain a live reference to caller-owned training data.

<details><summary>verbatim finding</summary>

```
### F27 — `self._raw_data = X` retains a reference to caller-owned input, defeating the `copy=False` boundary
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:707 — `self._raw_data = X` stores the validated (but not copied) input array on the estimator instance whenever `metric != "precomputed"`, regardless of `self.copy`.
scenario: "User trains `HDBSCAN(copy=False)` on a large array and then mutates or frees the underlying buffer expecting scikit-learn to be done with it → the estimator silently keeps a live reference (via `_raw_data`), so mutations reflect into the estimator's stored state, and any later pickling of the estimator serializes the entire raw training set — an unexpected data-leak channel when the estimator is persisted or shipped between processes."
contract: Drop the `_raw_data` attribute after `fit` and reconstruct only what downstream methods actually need, so the estimator does not silently retain a live reference to caller-owned training data.
instances: single-instance
```
</details>

### [LOW] `alpha=None` sentinel bypasses parameter validation and crashes with `TypeError` inside `_hdbscan_brute`

A downstream caller (or a future refactor) invokes `_hdbscan_brute(X)` without passing `alpha` → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=`, and the misleading default in the signature contradicts both the docstring and the estimator's actual validation. Set the private helper's default to the documented value (`alpha=1.0`) so the signature matches the docstring and the value cannot silently reach the in-place division as `None`.

<details><summary>verbatim finding</summary>

```
### F28 — `alpha=None` sentinel bypasses parameter validation and crashes with `TypeError` inside `_hdbscan_brute`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-241 — `_hdbscan_brute(..., alpha=None, ...)` accepts `alpha=None` as default and unconditionally executes `distance_matrix /= alpha` at line 241. `HDBSCAN.__init__` sets `self.alpha = 1.0` and `_parameter_constraints` requires `alpha` to be a positive `Real`, so the call site (line 767) always supplies a float — but the helper's signature advertises `alpha=None` as accepted while the docstring says "float, default=1.0". A caller invoking the private helper directly (as several tests import from `_hdbscan.hdbscan`) sees an unsafe silent contract.
scenario: "A downstream caller (or a future refactor) invokes `_hdbscan_brute(X)` without passing `alpha` → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=`, and the misleading default in the signature contradicts both the docstring and the estimator's actual validation."
contract: Set the private helper's default to the documented value (`alpha=1.0`) so the signature matches the docstring and the value cannot silently reach the in-place division as `None`.
instances: single-instance
```
</details>

### [LOW] `metric_params` is unpacked into `_hdbscan_brute` as `**kwargs` and silently consumes reserved key `max_distance`

User discovers via source-reading that `metric_params={'max_distance': 5.0}` alters sparse-matrix behavior, and passes it → the value flows unvalidated (no type/range check), and simultaneously the value is forwarded to the underlying `pairwise_distances`/`DistanceMetric.get_metric` call as an unknown metric kwarg, potentially causing an error or being silently ignored depending on the metric backend. Extract `max_distance` from `metric_params` at the estimator boundary in `fit`, pass it explicitly to `_hdbscan_brute`, and only forward the residual dict to `pairwise_distances`; validate its type/range via `_parameter_constraints`.

<details><summary>verbatim finding</summary>

```
### F30 — `metric_params` is unpacked into `_hdbscan_brute` as `**kwargs` and silently consumes reserved key `max_distance`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:764-771 — `kwargs = dict(X=X, min_samples=..., alpha=..., metric=..., n_jobs=..., **self._metric_params)` then passed as `mst_func(**kwargs)`. In `_hdbscan_brute` (line 243) `max_distance = metric_params.get("max_distance", 0.0)` intentionally overloads `metric_params` with an HDBSCAN-specific control key. There is no validation that `metric_params` does not collide with reserved HDBSCAN keyword names (`X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy`, `algo`, `leaf_size`), and any user-supplied `metric_params={'X': ..., 'min_samples': ...}` will raise a confusing `TypeError: got multiple values for keyword argument` deep inside the private helper rather than at the estimator boundary. More importantly, `max_distance` — a documented user-facing tuning parameter — is smuggled through `metric_params` rather than exposed as a proper estimator parameter subject to `_parameter_constraints`.
scenario: "User discovers via source-reading that `metric_params={'max_distance': 5.0}` alters sparse-matrix behavior, and passes it → the value flows unvalidated (no type/range check), and simultaneously the value is forwarded to the underlying `pairwise_distances`/`DistanceMetric.get_metric` call as an unknown metric kwarg, potentially causing an error or being silently ignored depending on the metric backend."
contract: Extract `max_distance` from `metric_params` at the estimator boundary in `fit`, pass it explicitly to `_hdbscan_brute`, and only forward the residual dict to `pairwise_distances`; validate its type/range via `_parameter_constraints`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:243, sklearn/cluster/_hdbscan/hdbscan.py:764-771]
```
</details>

### [LOW] `dbscan_clustering` overwrites labels using stale `self.labels_` masks with no protection against shape mismatch

A downstream refactor changes the point at which `_single_linkage_tree_` is remapped, or a pickled model from a slightly different code path is loaded → `self.labels_ == label` produces a mask of length ≠ len(labels), causing either a shape-mismatch IndexError or silent misindexing when NumPy broadcasts. Assert `len(labels) == self.labels_.shape[0]` before applying the boolean masks in `dbscan_clustering`, or explicitly re-derive the outlier masks from `self._raw_data`/stored non-finite indices.

<details><summary>verbatim finding</summary>

```
### F32 — `dbscan_clustering` overwrites labels using stale `self.labels_` masks with no protection against shape mismatch
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:960-969 — `labels = labelling_at_cut(self._single_linkage_tree_, cut_distance, min_cluster_size)`; then `infinite_index = self.labels_ == _OUTLIER_ENCODING["infinite"]["label"]` and `labels[infinite_index] = ...`. `labelling_at_cut` returns an array of length `n_samples` derived from `_single_linkage_tree_.shape[0] + 1`, while `self.labels_` has length equal to the *original* raw data (`self._raw_data.shape[0]`). When `metric != "precomputed"` and non-finite rows were dropped during `fit`, `_single_linkage_tree_` has been remapped by `remap_single_linkage_tree` (line 834) to include outlier leaves, so the shapes happen to align — but there is no assertion of this invariant, and any future refactor that changes when/whether the remap runs will silently produce mismatched-shape boolean-indexing that either raises `IndexError` or, worse, silently indexes wrongly. Callers relying on `dbscan_clustering` immediately after loading a pickled model built by an older version could see arbitrary corruption.
scenario: "A downstream refactor changes the point at which `_single_linkage_tree_` is remapped, or a pickled model from a slightly different code path is loaded → `self.labels_ == label` produces a mask of length ≠ len(labels), causing either a shape-mismatch IndexError or silent misindexing when NumPy broadcasts."
contract: Assert `len(labels) == self.labels_.shape[0]` before applying the boolean masks in `dbscan_clustering`, or explicitly re-derive the outlier masks from `self._raw_data`/stored non-finite indices.
instances: single-instance
```
</details>

### [LOW] `dbscan_clustering` accepts caller-supplied parameters without validation [out-of-theme]

Caller invokes `est.dbscan_clustering(cut_distance=-1.0, min_cluster_size=-5)` → Cython coerces the negative `min_cluster_size` into an intp, `cluster_size[cluster] < min_cluster_size` (line 419) never triggers, producing silently-invalid labels while the estimator API contract of validated, positive integer parameters is broken. Validate `cut_distance` and `min_cluster_size` against numeric-interval constraints (e.g. `cut_distance>=0`, `min_cluster_size>=1`) before dispatching to the Cython routine, matching the fit-time parameter validation performed by `_validate_params`.

<details><summary>verbatim finding</summary>

```
### F33 — `dbscan_clustering` accepts caller-supplied parameters without validation [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-962 — `dbscan_clustering(self, cut_distance, min_cluster_size=5)` performs no `_validate_params` / range check and passes the values straight into the Cython routine `labelling_at_cut(const HIERARCHY_t[::1] linkage, cnp.float64_t cut, cnp.intp_t min_cluster_size)` at sklearn/cluster/_hdbscan/_tree.pyx:359-362.
scenario: "Caller invokes `est.dbscan_clustering(cut_distance=-1.0, min_cluster_size=-5)` → Cython coerces the negative `min_cluster_size` into an intp, `cluster_size[cluster] < min_cluster_size` (line 419) never triggers, producing silently-invalid labels while the estimator API contract of validated, positive integer parameters is broken."
contract: Validate `cut_distance` and `min_cluster_size` against numeric-interval constraints (e.g. `cut_distance>=0`, `min_cluster_size>=1`) before dispatching to the Cython routine, matching the fit-time parameter validation performed by `_validate_params`.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_brute` mutates caller's sparse input in place even when `copy=True` [out-of-theme]

User passes a CSR sparse precomputed matrix with `copy=False` (the default) → `X.data /= alpha` and `mutual_reachability_graph` mutate the user's sparse buffer in place, invalidating the input for any further use When `copy=False` with a sparse input, the alpha rescaling and the in-place reachability update MUST be behind a `UserWarning` at the API boundary, matching the pattern used elsewhere in the file

<details><summary>verbatim finding</summary>

```
### F45 — `_hdbscan_brute` mutates caller's sparse input in place even when `copy=True` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:236 — `distance_matrix = X.copy() if copy else X`; sklearn/cluster/_hdbscan/hdbscan.py:244-247 — if the (now-copied) distance_matrix has a non-CSR format, `distance_matrix = distance_matrix.tocsr()` allocates a new CSR; but `X.copy()` on a sparse matrix preserves format, so if the user passed CSR with `copy=True`, `X.copy()` is CSR and `tocsr()` on a CSR returns `self` (no copy) — then sklearn/cluster/_hdbscan/hdbscan.py:241 (`distance_matrix /= alpha`) and the in-place `mutual_reachability_graph` at line 251 mutate the copy fine. However for `copy=False` + CSR user input, both `/=` and `mutual_reachability_graph` silently mutate the caller's `.data` buffer with no warning even though it is documented
scenario: "User passes a CSR sparse precomputed matrix with `copy=False` (the default) → `X.data /= alpha` and `mutual_reachability_graph` mutate the user's sparse buffer in place, invalidating the input for any further use"
contract: When `copy=False` with a sparse input, the alpha rescaling and the in-place reachability update MUST be behind a `UserWarning` at the API boundary, matching the pattern used elsewhere in the file
instances: single-instance
```
</details>

### [LOW] `remap_single_linkage_tree` crashes when `tree` is empty

Degenerate edge case where `_single_linkage_tree_` is empty (e.g. all-outlier input where the finite-subset collapse yields a zero-row hierarchy) with `metric != 'precomputed'` and non-finite data → `remap_single_linkage_tree` raises an uncaught `IndexError` mid-`fit`, leaving `self.labels_` set to the pre-remap values (internal indices, not raw indices) and `self._single_linkage_tree_` never remapped — a partial-failure state on the estimator `remap_single_linkage_tree` MUST validate `len(tree) > 0` and raise a clear `ValueError` before any state on the estimator has been partially mutated by the calling `fit`

<details><summary>verbatim finding</summary>

```
### F46 — `remap_single_linkage_tree` crashes when `tree` is empty
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:384-387 — unconditionally indexes `tree[tree.shape[0] - 1]` to compute `last_cluster_id`/`last_cluster_size`; if the caller-provided `tree` is length 0 this reads index `-1` (numpy wraps but on a zero-length structured array raises `IndexError`)
scenario: "Degenerate edge case where `_single_linkage_tree_` is empty (e.g. all-outlier input where the finite-subset collapse yields a zero-row hierarchy) with `metric != 'precomputed'` and non-finite data → `remap_single_linkage_tree` raises an uncaught `IndexError` mid-`fit`, leaving `self.labels_` set to the pre-remap values (internal indices, not raw indices) and `self._single_linkage_tree_` never remapped — a partial-failure state on the estimator"
contract: `remap_single_linkage_tree` MUST validate `len(tree) > 0` and raise a clear `ValueError` before any state on the estimator has been partially mutated by the calling `fit`
instances: single-instance
```
</details>

### [LOW] `_hdbscan_brute` sparse-format normalisation happens after `distance_matrix /= alpha`

User passes a LIL precomputed distance matrix with `copy=False` → `distance_matrix /= alpha` may return a new array (scipy semantics for `__itruediv__` on non-CSR/CSC vary by version), so the `copy=False` contract silently becomes copy-semantics, and vice-versa for CSR. Convert `distance_matrix` to CSR (with explicit `.copy()` if `copy=True`) **before** the `/= alpha` step, so alpha scaling and downstream in-place mutual-reachability run on a known format and known ownership.

<details><summary>verbatim finding</summary>

```
### F49 — `_hdbscan_brute` sparse-format normalisation happens after `distance_matrix /= alpha`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:241 executes `distance_matrix /= alpha`; sklearn/cluster/_hdbscan/hdbscan.py:244-247 subsequently checks `if issparse(distance_matrix) and distance_matrix.format != "csr": distance_matrix = distance_matrix.tocsr()`. For a non-CSR sparse input (e.g., LIL, which is in `accept_sparse`), the in-place `/= alpha` runs on a LIL matrix (supported but slow, and it *does* copy into COO internally in some scipy versions), potentially producing a *new* object and then the subsequent `tocsr()` runs on that new object; the tocsr result is then not the same buffer as `distance_matrix` originally referenced.
scenario: "User passes a LIL precomputed distance matrix with `copy=False` → `distance_matrix /= alpha` may return a new array (scipy semantics for `__itruediv__` on non-CSR/CSC vary by version), so the `copy=False` contract silently becomes copy-semantics, and vice-versa for CSR."
contract: Convert `distance_matrix` to CSR (with explicit `.copy()` if `copy=True`) **before** the `/= alpha` step, so alpha scaling and downstream in-place mutual-reachability run on a known format and known ownership.
instances: single-instance
```
</details>

### [LOW] Dense-precomputed path with `algorithm="auto"` never validates symmetry

User passes a dense precomputed distance matrix with default `algorithm='auto'` and `copy=False` → the auto branch dispatches to `_hdbscan_brute` without `copy` in kwargs → `_hdbscan_brute` runs `distance_matrix /= alpha` on the original array, silently corrupting the caller's matrix on subsequent uses; no warning is emitted. When dispatching to `_hdbscan_brute` from any branch (auto or explicit) with `metric='precomputed'`, `copy` must be forwarded so the caller's promised `copy=True` is honored; if `copy=False`, the doc must acknowledge that dense precomputed inputs are mutated in-place (currently docs at lines 521-527 imply copy only applies when explicit brute is chosen).

<details><summary>verbatim finding</summary>

```
### F53 — Dense-precomputed path with `algorithm="auto"` never validates symmetry
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:741-751, 804-819 — for dense precomputed inputs, symmetry/square checks live inside `_hdbscan_brute` (lines 222-234). When `algorithm="auto"` chose `_hdbscan_brute` (which it does for precomputed since `"precomputed" not in FAST_METRICS`), the checks run — but the fit-level branch that handles dense precomputed at line 747 also permits `np.inf`. `_hdbscan_brute` then divides by alpha and passes to `mutual_reachability_graph`; the resulting `distance_matrix /= alpha` at line 241 is executed on the user's array when `copy=False` (which is the effective default in auto mode per F2), silently mutating the user's precomputed matrix even in the auto path. There is no error path that detects and reports this in-place mutation.
scenario: "User passes a dense precomputed distance matrix with default `algorithm='auto'` and `copy=False` → the auto branch dispatches to `_hdbscan_brute` without `copy` in kwargs → `_hdbscan_brute` runs `distance_matrix /= alpha` on the original array, silently corrupting the caller's matrix on subsequent uses; no warning is emitted."
contract: When dispatching to `_hdbscan_brute` from any branch (auto or explicit) with `metric='precomputed'`, `copy` must be forwarded so the caller's promised `copy=True` is honored; if `copy=False`, the doc must acknowledge that dense precomputed inputs are mutated in-place (currently docs at lines 521-527 imply copy only applies when explicit brute is chosen).
instances: single-instance
```
</details>

### [LOW] `remap_single_linkage_tree` and `_get_finite_row_indices` have no direct tests

logic change in remap_single_linkage_tree that mis-indexes a single point → transitive tests still pass counts/labels-of-outlier checks; the internal `_single_linkage_tree_` structure is not verified Add unit tests that construct a small tree with known non-finite indices and assert the exact structure of the remapped tree.

<details><summary>verbatim finding</summary>

```
### F66 — `remap_single_linkage_tree` and `_get_finite_row_indices` have no direct tests
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-407 defines both helpers; grep across `sklearn/cluster/tests/test_hdbscan.py` and `sklearn/cluster/_hdbscan/tests/test_reachibility.py` shows neither symbol is imported or referenced. Their behavior is only exercised transitively through outlier fitting tests, so an off-by-one or index-remap regression could silently pass end-to-end tests that only count clusters.
scenario: "logic change in remap_single_linkage_tree that mis-indexes a single point → transitive tests still pass counts/labels-of-outlier checks; the internal `_single_linkage_tree_` structure is not verified"
contract: Add unit tests that construct a small tree with known non-finite indices and assert the exact structure of the remapped tree.
instances: single-instance
```
</details>

### [LOW] `leaf_size` is untested

A regression drops `leaf_size` from the `kwargs` dict → no test fails; users silently lose the ability to tune tree leaf size. Add a test that fits with two different `leaf_size` values and asserts labels are identical.

<details><summary>verbatim finding</summary>

```
### F67 — `leaf_size` is untested
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py — no reference to `leaf_size`. It is declared at sklearn/cluster/_hdbscan/hdbscan.py:639 and forwarded to `NearestNeighbors` via `_hdbscan_prims` at hdbscan.py:799, 803, 813, 818. There is no test verifying that `leaf_size` is threaded through (e.g., that varying `leaf_size` still produces identical labels).
scenario: "A regression drops `leaf_size` from the `kwargs` dict → no test fails; users silently lose the ability to tune tree leaf size."
contract: Add a test that fits with two different `leaf_size` values and asserts labels are identical.
instances: single-instance
```
</details>

### [LOW] `dbscan_clustering(min_cluster_size=…)` parameter is untested

A regression to the `min_cluster_size` handling in `labelling_at_cut` (e.g., wrong comparison, dropped filter) → the `dbscan_clustering` output silently returns clusters below the requested minimum size, uncaught by tests. Add a test that calls `dbscan_clustering(cut_distance=..., min_cluster_size=k)` and asserts every non-noise cluster in the result has size ≥ k.

<details><summary>verbatim finding</summary>

```
### F71 — `dbscan_clustering(min_cluster_size=…)` parameter is untested
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-970 defines `dbscan_clustering(self, cut_distance, min_cluster_size=5)`. In sklearn/cluster/tests/test_hdbscan.py:186 and 204, `dbscan_clustering` is only ever called with `cut_distance=...` — no test varies `min_cluster_size`, so the internal filtering `if cluster_size[cluster] < min_cluster_size:` in `labelling_at_cut` (sklearn/cluster/_hdbscan/_tree.pyx:419-420) is not exercised through this public entry point.
scenario: "A regression to the `min_cluster_size` handling in `labelling_at_cut` (e.g., wrong comparison, dropped filter) → the `dbscan_clustering` output silently returns clusters below the requested minimum size, uncaught by tests."
contract: Add a test that calls `dbscan_clustering(cut_distance=..., min_cluster_size=k)` and asserts every non-noise cluster in the result has size ≥ k.
instances: single-instance
```
</details>

### [LOW] `n_jobs` parameter is set but never verified to affect parallelism or correctness

A regression in the `n_jobs` handoff (e.g., wrong forwarding, non-deterministic tie-breaking under parallelism) → not caught. Add a parametrized test asserting `HDBSCAN(n_jobs=1).fit_predict(X) == HDBSCAN(n_jobs=-1).fit_predict(X)`.

<details><summary>verbatim finding</summary>

```
### F72 — `n_jobs` parameter is set but never verified to affect parallelism or correctness
severity: low
evidence: `n_jobs` defaults to `4` in `HDBSCAN.__init__` (sklearn/cluster/_hdbscan/hdbscan.py:658). No test in sklearn/cluster/tests/test_hdbscan.py sets `n_jobs` (grep count: 0). The parameter is forwarded into `pairwise_distances` and `NearestNeighbors` calls but its behavior (including e.g., `n_jobs=-1`, `n_jobs=1`) is untested for label-equivalence.
scenario: "A regression in the `n_jobs` handoff (e.g., wrong forwarding, non-deterministic tie-breaking under parallelism) → not caught."
contract: Add a parametrized test asserting `HDBSCAN(n_jobs=1).fit_predict(X) == HDBSCAN(n_jobs=-1).fit_predict(X)`.
instances: single-instance
```
</details>

### [LOW] `HDBSCAN` `algorithm` StrOptions omit the historical `"kd_tree"` / `"ball_tree"` names still emitted by the routing code

A user sets `algorithm='kd_tree'` matching the well-known `NearestNeighbors` spelling → InvalidParameterError, despite this being the exact string internally forwarded to NearestNeighbors The parameter-constraints StrOptions and the internal dispatch strings MUST use a single, documented spelling; if `kdtree`/`balltree` is the chosen public spelling, translate exactly once at the boundary with a comment naming the sklearn API contract, and mention this translation in the docstring.

<details><summary>verbatim finding</summary>

```
### F92 — `HDBSCAN` `algorithm` StrOptions omit the historical `"kd_tree"` / `"ball_tree"` names still emitted by the routing code
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:629-638 validates `algorithm` against `{"auto", "brute", "kdtree", "balltree"}`, but the routing code at sklearn/cluster/_hdbscan/hdbscan.py:798, :802, :812, :817 forwards `algo="kd_tree"` / `algo="ball_tree"` (underscored) into `NearestNeighbors(algorithm=algo)`. The user-facing API contract accepts one spelling while the internal contract to `NearestNeighbors.algorithm` requires another, and there is no cast/mapping documented; a reader/user cannot round-trip the accepted string.
scenario: "A user sets `algorithm='kd_tree'` matching the well-known `NearestNeighbors` spelling → InvalidParameterError, despite this being the exact string internally forwarded to NearestNeighbors"
contract: The parameter-constraints StrOptions and the internal dispatch strings MUST use a single, documented spelling; if `kdtree`/`balltree` is the chosen public spelling, translate exactly once at the boundary with a comment naming the sklearn API contract, and mention this translation in the docstring.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_prims` / `_hdbscan_brute` signatures declare `min_samples=5` but docstring says `default=None`

docs/signature mismatch → downstream reviewers or users misread the contract for private helpers, and the documented `None` value would break `NearestNeighbors(n_neighbors=None)` inside `_hdbscan_prims`. The docstring `default` for `min_samples` must match the signature's actual default (`5`).

<details><summary>verbatim finding</summary>

```
### F96 — `_hdbscan_prims` / `_hdbscan_brute` signatures declare `min_samples=5` but docstring says `default=None`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160,180-181,272,290-291 — both functions have `min_samples=5` in the signature; both docstrings say `min_samples : int, default=None`.
scenario: "docs/signature mismatch → downstream reviewers or users misread the contract for private helpers, and the documented `None` value would break `NearestNeighbors(n_neighbors=None)` inside `_hdbscan_prims`."
contract: The docstring `default` for `min_samples` must match the signature's actual default (`5`).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:161, sklearn/cluster/_hdbscan/hdbscan.py:272]
```
</details>

### [LOW] `_hdbscan_prims` documents a `copy` parameter that its signature does not accept

reader/refactorer relies on the documented `copy` parameter → passes `copy=...` which lands in `metric_params` (and forwards to distance metric functions), silently changing behavior or causing a TypeError. The `Parameters` section must reflect the function's actual accepted arguments — the `copy` entry must be removed (or the signature must accept it).

<details><summary>verbatim finding</summary>

```
### F97 — `_hdbscan_prims` documents a `copy` parameter that its signature does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278,313-318 — signature lists `X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params` with no `copy` parameter, yet the docstring includes a `copy : bool, default=False` parameter block.
scenario: "reader/refactorer relies on the documented `copy` parameter → passes `copy=...` which lands in `metric_params` (and forwards to distance metric functions), silently changing behavior or causing a TypeError."
contract: The `Parameters` section must reflect the function's actual accepted arguments — the `copy` entry must be removed (or the signature must accept it).
instances: single-instance
```
</details>

### [LOW] `_more_tags` marks `allow_nan=True` when metric is a callable, but callables may not accept NaN

user relies on the advertised `allow_nan=True` and calls `HDBSCAN(metric='euclidean', algorithm='kdtree').fit(X_with_nan)` → NaN rows are filtered before reaching KDTree, so it works — but the underlying tag/contract makes no distinction, and `metric='precomputed'` with a sparse LIL matrix (line 738) never runs the NaN-filter path yet doesn't validate `force_all_finite=True` either. The `allow_nan` tag should be conditional on the actual runtime input path — sparse precomputed matrices do not filter NaN, so the tag should not universally return True for non-precomputed metrics.

<details><summary>verbatim finding</summary>

```
### F98 — `_more_tags` marks `allow_nan=True` when metric is a callable, but callables may not accept NaN
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:972-973 — `return {"allow_nan": self.metric != "precomputed"}`. For any non-precomputed metric — including user-supplied callables — the estimator advertises NaN acceptance. But NaN handling only works because `_get_finite_row_indices` filters non-finite rows before passing to the metric; if the metric is a callable that expects finite input, the tag alone does not guarantee correctness — this is fine, but the tag is asserted for all algorithms including `algorithm="kdtree"`/`"balltree"`, whose backing `NearestNeighbors` does not tolerate NaN in the query data.
scenario: "user relies on the advertised `allow_nan=True` and calls `HDBSCAN(metric='euclidean', algorithm='kdtree').fit(X_with_nan)` → NaN rows are filtered before reaching KDTree, so it works — but the underlying tag/contract makes no distinction, and `metric='precomputed'` with a sparse LIL matrix (line 738) never runs the NaN-filter path yet doesn't validate `force_all_finite=True` either." 
contract: The `allow_nan` tag should be conditional on the actual runtime input path — sparse precomputed matrices do not filter NaN, so the tag should not universally return True for non-precomputed metrics.
instances: single-instance
```
</details>

### [LOW] Precomputed symmetry check re-implements `sklearn.utils.validation.check_symmetric`

The bespoke check disagrees with the shared helper on tolerance (`_allclose_dense_sparse` defaults vs `check_symmetric`'s `tol=1e-10`) and on error wording → users of precomputed matrices see different behaviour from HDBSCAN than from every other precomputed-matrix estimator, and future tightening of `check_symmetric` won't propagate here. Replace the manual shape and `_allclose_dense_sparse(X, X.T)` block in `_hdbscan_brute` with `check_symmetric(X, raise_exception=True)`.

<details><summary>verbatim finding</summary>

```
### F108 — Precomputed symmetry check re-implements `sklearn.utils.validation.check_symmetric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:222-234 — inside `_hdbscan_brute`, precomputed symmetry is validated by hand: shape check + `_allclose_dense_sparse(X, X.T)` with a bespoke error message. sklearn/utils/validation.py:1309 provides `check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False)` which is the module-wide helper used by other precomputed-matrix estimators.
scenario: "The bespoke check disagrees with the shared helper on tolerance (`_allclose_dense_sparse` defaults vs `check_symmetric`'s `tol=1e-10`) and on error wording → users of precomputed matrices see different behaviour from HDBSCAN than from every other precomputed-matrix estimator, and future tightening of `check_symmetric` won't propagate here."
contract: Replace the manual shape and `_allclose_dense_sparse(X, X.T)` block in `_hdbscan_brute` with `check_symmetric(X, raise_exception=True)`.
instances: single-instance
```
</details>

### [LOW] Inline comment in `fit()` restates outlier labels incorrectly, driven by the same hardcoding pattern

Because outlier labels are documented in prose next to code that reads them from `_OUTLIER_ENCODING`, prose and dict drift → reader assumes noise label `-1` covers infinite samples and misinterprets `labels_`. Replace prose-hardcoded label values in this comment with references to `_OUTLIER_ENCODING["infinite"]["label"]` and `_OUTLIER_ENCODING["missing"]["label"]`.

<details><summary>verbatim finding</summary>

```
### F109 — Inline comment in `fit()` restates outlier labels incorrectly, driven by the same hardcoding pattern
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The actual encoding (lines 65-79 and lines 842-843 immediately below) maps `np.inf` → `-2` and `np.nan` → `-3`. The comment is a stale duplicate of the truth held in `_OUTLIER_ENCODING`.
scenario: "Because outlier labels are documented in prose next to code that reads them from `_OUTLIER_ENCODING`, prose and dict drift → reader assumes noise label `-1` covers infinite samples and misinterprets `labels_`."
contract: Replace prose-hardcoded label values in this comment with references to `_OUTLIER_ENCODING["infinite"]["label"]` and `_OUTLIER_ENCODING["missing"]["label"]`.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_prims` docstring copy-pasted from `_hdbscan_brute` documents a nonexistent `copy` parameter

Reader consults `_hdbscan_prims` docstring → assumes `copy=True` will protect their input data from in-place modification, but the argument silently vanishes into `**metric_params` and cannot influence behavior. Remove the `copy` docstring block from `_hdbscan_prims` (or delete the `_hdbscan_prims` docstring entirely and inherit from a shared helper) so it exactly matches its signature.

<details><summary>verbatim finding</summary>

```
### F112 — `_hdbscan_prims` docstring copy-pasted from `_hdbscan_brute` documents a nonexistent `copy` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — `_hdbscan_prims` docstring contains a `copy : bool, default=False` block copied verbatim from `_hdbscan_brute`, but the function signature at lines 269-278 does not accept a `copy` parameter (and unlike `_hdbscan_brute` it never uses `copy` semantics since it always calls `np.asarray(X, order="C")` unconditionally).
scenario: "Reader consults `_hdbscan_prims` docstring → assumes `copy=True` will protect their input data from in-place modification, but the argument silently vanishes into `**metric_params` and cannot influence behavior."
contract: Remove the `copy` docstring block from `_hdbscan_prims` (or delete the `_hdbscan_prims` docstring entirely and inherit from a shared helper) so it exactly matches its signature.
instances: single-instance
```
</details>

### [LOW] Dead `mst_func = None` initialization in `HDBSCAN.fit`

Reader assumes the None default is meaningful → wastes time verifying every path reassigns; or a future edit adds a branch that fails to assign `mst_func`, and the AttributeError becomes a confusing 'NoneType is not callable' at line 820 instead of a clear NameError Remove the `mst_func = None` line and let missing-branch assignments raise NameError, which is louder and more debuggable.

<details><summary>verbatim finding</summary>

```
### F127 — Dead `mst_func = None` initialization in `HDBSCAN.fit`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:763 — `mst_func = None`. Every branch below (algorithm in {"kdtree","balltree","brute"} explicit, and every branch of the `auto` else) unconditionally assigns `mst_func`; the initial `None` is never observed.
scenario: "Reader assumes the None default is meaningful → wastes time verifying every path reassigns; or a future edit adds a branch that fails to assign `mst_func`, and the AttributeError becomes a confusing 'NoneType is not callable' at line 820 instead of a clear NameError"
contract: Remove the `mst_func = None` line and let missing-branch assignments raise NameError, which is louder and more debuggable.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_prims` documents a `copy` parameter it does not accept

Contributor reads the docstring and passes `copy=True` → TypeError from `_hdbscan_prims`; or a maintainer copies the docstring elsewhere and propagates the ghost parameter Delete the `copy` paragraph from `_hdbscan_prims`'s docstring; it belongs only on `_hdbscan_brute`.

<details><summary>verbatim finding</summary>

```
### F128 — `_hdbscan_prims` documents a `copy` parameter it does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature has no `copy`; docstring at sklearn/cluster/_hdbscan/hdbscan.py:313-318 describes `copy : bool, default=False`. There is no `copy` in the signature or body of `_hdbscan_prims`.
scenario: "Contributor reads the docstring and passes `copy=True` → TypeError from `_hdbscan_prims`; or a maintainer copies the docstring elsewhere and propagates the ghost parameter"
contract: Delete the `copy` paragraph from `_hdbscan_prims`'s docstring; it belongs only on `_hdbscan_brute`.
instances: single-instance
```
</details>

### [LOW] Comment in `fit` misdocuments outlier label mapping

reader trusts the comment → reader believes np.inf → label -1 and np.nan → label -2, but `_OUTLIER_ENCODING` in the same file (lines 65-79) actually assigns np.inf → -2 and np.nan → -3, which is what the code at lines 842-843 does; comment is a lying name for the behavior. Update the comment to match `_OUTLIER_ENCODING`: np.inf → -2 (infinite), np.nan → -3 (missing).

<details><summary>verbatim finding</summary>

```
### F143 — Comment in `fit` misdocuments outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — "# Remap indices to align with original data in the case of / # non-finite entries. Samples with np.inf are mapped to -1 and / # those with np.nan are mapped to -2."
scenario: "reader trusts the comment → reader believes np.inf → label -1 and np.nan → label -2, but `_OUTLIER_ENCODING` in the same file (lines 65-79) actually assigns np.inf → -2 and np.nan → -3, which is what the code at lines 842-843 does; comment is a lying name for the behavior."
contract: Update the comment to match `_OUTLIER_ENCODING`: np.inf → -2 (infinite), np.nan → -3 (missing).
instances: single-instance
```
</details>

### [LOW] `remap_single_linkage_tree` docstring omits its return value

reader/API user reads only the docstring → assumes the mutating name means in-place, misses that a fresh, concatenated array is returned. Add a `Returns` section documenting the returned `ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` and clarify it is not an in-place update.

<details><summary>verbatim finding</summary>

```
### F144 — `remap_single_linkage_tree` docstring omits its return value
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-393 — docstring documents Parameters but no `Returns` section; the function does return `tree` (line 393) and callers rely on the return value (line 834 `self._single_linkage_tree_ = remap_single_linkage_tree(...)`).
scenario: "reader/API user reads only the docstring → assumes the mutating name means in-place, misses that a fresh, concatenated array is returned."
contract: Add a `Returns` section documenting the returned `ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` and clarify it is not an in-place update.
instances: single-instance
```
</details>

### [LOW] `_brute_mst` docstring `min_samples` default lies

reader trusts docstring default → believes calling `_brute_mst(mr)` is valid; actually raises `TypeError` for missing positional arg. Also parroted by `_hdbscan_brute` (line 179) and `_hdbscan_prims` (line 290) where the signatures use `min_samples=5`. Remove the "default=None" claim (or specify the correct defaults matching each function's signature).

<details><summary>verbatim finding</summary>

```
### F145 — `_brute_mst` docstring `min_samples` default lies
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:95 — "min_samples : int, default=None" but the signature at line 82 is `def _brute_mst(mutual_reachability, min_samples):` (required positional, no default).
scenario: "reader trusts docstring default → believes calling `_brute_mst(mr)` is valid; actually raises `TypeError` for missing positional arg. Also parroted by `_hdbscan_brute` (line 179) and `_hdbscan_prims` (line 290) where the signatures use `min_samples=5`."
contract: Remove the "default=None" claim (or specify the correct defaults matching each function's signature).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:95, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]
```
</details>

### [LOW] `mutual_reachability_` variable's trailing underscore is meaningless noise

reader scanning for estimator attributes → is briefly misled into thinking `mutual_reachability_` is set on `self`; churns cognitive load. Trivial but a false abstraction signal. Rename to `mutual_reachability` (no trailing underscore) since it is a local.

<details><summary>verbatim finding</summary>

```
### F148 — `mutual_reachability_` variable's trailing underscore is meaningless noise
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:251 — `mutual_reachability_ = mutual_reachability_graph(...)` — local variable named with trailing underscore normally reserved (in sklearn convention) for fitted-attribute names on estimators, not local variables; misleading nomenclature.
scenario: "reader scanning for estimator attributes → is briefly misled into thinking `mutual_reachability_` is set on `self`; churns cognitive load. Trivial but a false abstraction signal."
contract: Rename to `mutual_reachability` (no trailing underscore) since it is a local.
instances: single-instance
```
</details>

### [LOW] `_process_mst` docstring claims MST is sorted in-place; caller may not expect side effect

reader expecting in-place mutation (per the wording) → misuses the return value or the argument; no functional bug today, but the doc/code split is misleading. Reword the docstring to "A sorted copy of the MST edges is used internally" or similar.

<details><summary>verbatim finding</summary>

```
### F152 — `_process_mst` docstring claims MST is sorted in-place; caller may not expect side effect
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:151-155 — `row_order = np.argsort(min_spanning_tree["distance"]); min_spanning_tree = min_spanning_tree[row_order]`. Docstring at lines 136-149 says "The MST is first sorted then processed" but does not clarify whether the caller's MST array is mutated. `argsort` + fancy indexing produces a new array (safe here), so the docstring's phrasing "MST is first sorted" is a surprising in-place-sounding side-effect promise that is not actually delivered — a mild lying name for the operation.
scenario: "reader expecting in-place mutation (per the wording) → misuses the return value or the argument; no functional bug today, but the doc/code split is misleading."
contract: Reword the docstring to "A sorted copy of the MST edges is used internally" or similar.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_prims` docstring documents non-existent `copy` parameter

User reading the source docstring expects `_hdbscan_prims` to accept `copy` → passes `copy=True` and gets an unexpected TypeError Remove the `copy` parameter section from the `_hdbscan_prims` docstring.

<details><summary>verbatim finding</summary>

```
### F154 — `_hdbscan_prims` docstring documents non-existent `copy` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False`; the function signature at lines 269-278 has no `copy` parameter, and no `copy` usage inside the function.
scenario: "User reading the source docstring expects `_hdbscan_prims` to accept `copy` → passes `copy=True` and gets an unexpected TypeError"
contract: Remove the `copy` parameter section from the `_hdbscan_prims` docstring.
instances: single-instance
```
</details>

### [LOW] Pervasive misspellings in public docstrings ("mututal", "reahability", "collecteion", "smaler", "simbling", "reachibility")

User-visible Sphinx docs render the misspelled Returns label and prose → damages perceived quality of the new estimator Fix each misspelling to the correct English form ("mutual", "reachability", "collection", "smaller", "sibling").

<details><summary>verbatim finding</summary>

```
### F156 — Pervasive misspellings in public docstrings ("mututal", "reahability", "collecteion", "smaler", "simbling", "reachibility")
severity: low
evidence: multiple docstrings and identifiers, verified via grep:
- `mututal_reachability_graph` (Returns section label) at sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/_reachability.pyx:76
- `mutual-reahability` at sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:143; sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:228
- `collecteion` at sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:144; sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:229
- `smaler` at sklearn/cluster/_hdbscan/_tree.pyx:133
- `simbling` at sklearn/cluster/_hdbscan/_tree.pyx:503
- Cython identifier `mutual_reachibility_distance` at sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210
- misspelled test module filename at sklearn/cluster/_hdbscan/tests/test_reachibility.py:1
scenario: "User-visible Sphinx docs render the misspelled Returns label and prose → damages perceived quality of the new estimator"
contract: Fix each misspelling to the correct English form ("mutual", "reachability", "collection", "smaller", "sibling").
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_reachability.pyx:76, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/tests/test_reachibility.py:1]
```
</details>

### [LOW] `HDBSCAN.fit` sets `self._raw_data = X` before dropping non-finite rows, silently exposing pre-validation data as a private attribute

Third-party subclass that accesses `self._raw_data` after `fit(X, metric='precomputed')` → AttributeError Always set `self._raw_data` in `fit` (with a comment about scope) so the attribute is defined on every code path.

<details><summary>verbatim finding</summary>

```
### F157 — `HDBSCAN.fit` sets `self._raw_data = X` before dropping non-finite rows, silently exposing pre-validation data as a private attribute
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:707 stores the validated-but-still-non-finite X as `self._raw_data`; downstream code at lines 840, 846 uses `self._raw_data.shape[0]` to size relabelled outputs. Name suggests "raw" but it is post-`_validate_data`, and it is set only on the non-precomputed branch (undefined on the precomputed branch).
scenario: "Third-party subclass that accesses `self._raw_data` after `fit(X, metric='precomputed')` → AttributeError"
contract: Always set `self._raw_data` in `fit` (with a comment about scope) so the attribute is defined on every code path.
instances: single-instance
```
</details>

### [LOW] `_weighted_cluster_center` allocates `mask` before the loop, then immediately rebinds it

Reader wonders whether the pre-allocation is used as an out-parameter for a Cython routine → wasted analysis Delete line 896.

<details><summary>verbatim finding</summary>

```
### F160 — `_weighted_cluster_center` allocates `mask` before the loop, then immediately rebinds it
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 `mask = np.empty((X.shape[0],), dtype=np.bool_)` is never read; line 908 `mask = self.labels_ == idx` rebinds every iteration.
scenario: "Reader wonders whether the pre-allocation is used as an out-parameter for a Cython routine → wasted analysis"
contract: Delete line 896.
instances: single-instance
```
</details>

### [LOW] `p=None` passed to `NearestNeighbors` in `_hdbscan_prims`

future upgrade of `NearestNeighbors` tightens `p` validation to reject `None` → HDBSCAN breaks for every tree algorithm without a code change here Omit the `p=None` kwarg and let the default apply, or document the invariant tying `p` handling to `metric_params`.

<details><summary>verbatim finding</summary>

```
### F163 — `p=None` passed to `NearestNeighbors` in `_hdbscan_prims`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:339 — `NearestNeighbors(..., p=None)`. `NearestNeighbors.p` documents `float, default=2`; passing `None` is a surprising side-input that bypasses the sensible default. There is no comment explaining the intent (presumably "let `metric_params` supply p, don't double-count") — a speculative abstraction whose contract is undocumented.
scenario: "future upgrade of `NearestNeighbors` tightens `p` validation to reject `None` → HDBSCAN breaks for every tree algorithm without a code change here"
contract: Omit the `p=None` kwarg and let the default apply, or document the invariant tying `p` handling to `metric_params`.
instances: single-instance
```
</details>

### [LOW] Duplicated/near-identical `_hdbscan_brute` and `_hdbscan_prims` docstrings mis-describe parameters

user picks a metric valid for `pairwise_distances` but not for `KDTree`/`BallTree` → fails; user reads about `copy` behaviour but the parameter is silently absent Rewrite the `_hdbscan_prims` docstring to describe only its actual parameters and the actual valid-metric constraint.

<details><summary>verbatim finding</summary>

```
### F164 — Duplicated/near-identical `_hdbscan_brute` and `_hdbscan_prims` docstrings mis-describe parameters
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-327 — `_hdbscan_prims` docstring documents `metric` "must be one of the options allowed by pairwise_distances" (line 297-301), but the function actually uses `NearestNeighbors` + `DistanceMetric` (line 332-344) which have different valid-metric sets; the same docstring includes a `copy` parameter (line 313-318) that the function does not accept. Speculative-abstraction copy from `_hdbscan_brute`.
scenario: "user picks a metric valid for `pairwise_distances` but not for `KDTree`/`BallTree` → fails; user reads about `copy` behaviour but the parameter is silently absent"
contract: Rewrite the `_hdbscan_prims` docstring to describe only its actual parameters and the actual valid-metric constraint.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_brute` / `_hdbscan_prims` docstrings state `min_samples : int, default=None` while signatures default to 5

Reader relies on `default=None` semantics (falling back elsewhere) → in reality the fallback is a hard-coded `5` Update each docstring to `default=5` to match the signature.

<details><summary>verbatim finding</summary>

```
### F177 — `_hdbscan_brute` / `_hdbscan_prims` docstrings state `min_samples : int, default=None` while signatures default to 5
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160 and 272 — both signatures declare `min_samples=5`; sklearn/cluster/_hdbscan/hdbscan.py:179-181 and 290-292 — both docstrings say `min_samples : int, default=None`
scenario: "Reader relies on `default=None` semantics (falling back elsewhere) → in reality the fallback is a hard-coded `5`"
contract: Update each docstring to `default=5` to match the signature.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]
```
</details>

### [LOW] `dbscan_clustering` docstring omits the removal of border points invariant and mislabels outputs

User calls `HDBSCAN().dbscan_clustering(0.3)` before fit per docstring reading → AttributeError on `self._single_linkage_tree_` Add an explicit precondition line stating that `fit` must be called first, and clarify that the -2/-3 labels only appear when non-finite samples were present during `fit`.

<details><summary>verbatim finding</summary>

```
### F178 — `dbscan_clustering` docstring omits the removal of border points invariant and mislabels outputs
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-958 — Returns section lists label rules "-1 noise", "-2 infinite", "-3 missing", but the implementation at lines 960-969 only reassigns `_OUTLIER_ENCODING["infinite"]` and `_OUTLIER_ENCODING["missing"]`; there is no `-1` produced by the underlying `labelling_at_cut` unless a cluster is smaller than `min_cluster_size`. More importantly, the "See Also" invariant that `dbscan_clustering` returns labels only for the data seen during `fit` (relies on `self.labels_`) is left implicit — calling it before `fit` raises AttributeError but nothing in the docstring warns of the `fit`-required precondition.
scenario: "User calls `HDBSCAN().dbscan_clustering(0.3)` before fit per docstring reading → AttributeError on `self._single_linkage_tree_`"
contract: Add an explicit precondition line stating that `fit` must be called first, and clarify that the -2/-3 labels only appear when non-finite samples were present during `fit`.
instances: single-instance
```
</details>

### [LOW] `HDBSCAN.labels_` docstring's outlier bullets contradict actual behavior for `dbscan_clustering`/precomputed

User with a precomputed distance matrix containing np.inf reads docstring → expects -2 for those points; gets an ordinary cluster id or ordinary -1 noise The `labels_` docstring must add a caveat that `-2`/`-3` labels are only produced when `metric != 'precomputed'`; for precomputed distance matrices, non-finite entries are left untouched (inf) or raised on (nan).

<details><summary>verbatim finding</summary>

```
### F185 — `HDBSCAN.labels_` docstring's outlier bullets contradict actual behavior for `dbscan_clustering`/precomputed
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:530-538 — the Attributes list unconditionally asserts label `-2` for infinite samples and `-3` for missing. But in `fit` (lines 730-753) the -2/-3 remapping is skipped when `metric == "precomputed"` (only `_get_finite_row_indices` is not invoked, no remap block runs), so a precomputed dense matrix containing `np.inf` never yields a `-2` label — those rows silently participate in clustering. The docstring gives no such caveat.
scenario: "User with a precomputed distance matrix containing np.inf reads docstring → expects -2 for those points; gets an ordinary cluster id or ordinary -1 noise"
contract: The `labels_` docstring must add a caveat that `-2`/`-3` labels are only produced when `metric != 'precomputed'`; for precomputed distance matrices, non-finite entries are left untouched (inf) or raised on (nan).
instances: single-instance
```
</details>

### [LOW] HDBSCAN docstring claims `centroids_`/`medoids_` are always-present attributes

User consults the Attributes section, expecting `centroids_` after `fit` → attribute is missing when `store_centers=None` (default), causing AttributeError Document these two attributes as conditional on `store_centers`, mirroring the sibling documentation style for optional attributes elsewhere in sklearn.

<details><summary>verbatim finding</summary>

```
### F190 — HDBSCAN docstring claims `centroids_`/`medoids_` are always-present attributes
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:556-573 — `centroids_` and `medoids_` are listed under `Attributes` with no note that they are only computed when `store_centers` is set; sklearn/cluster/_hdbscan/hdbscan.py:854-855 and 897-903 — these attributes are only created if `self.store_centers` is truthy (and only the matching kind is written)
scenario: "User consults the Attributes section, expecting `centroids_` after `fit` → attribute is missing when `store_centers=None` (default), causing AttributeError"
contract: Document these two attributes as conditional on `store_centers`, mirroring the sibling documentation style for optional attributes elsewhere in sklearn.
instances: single-instance
```
</details>

### [LOW] HDBSCAN `algorithm` docstring uses inconsistent value casing ("KDTree"/"BallTree" vs. accepted "kdtree"/"balltree")

User copies `algorithm="KDTree"` from the prose → parameter validation raises `InvalidParameterError` Refer to the accepted lowercase strings `"kdtree"` / `"balltree"` in the prose (keep class links separate).

<details><summary>verbatim finding</summary>

```
### F191 — HDBSCAN `algorithm` docstring uses inconsistent value casing ("KDTree"/"BallTree" vs. accepted "kdtree"/"balltree")
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:466-472 — parameter enumerates `{"auto", "brute", "kdtree", "balltree"}` (matching the `_parameter_constraints` StrOptions at 629-638) but the prose body says "Both `"KDTree"` and `"BallTree"` algorithms use the `NearestNeighbors` estimator."
scenario: "User copies `algorithm="KDTree"` from the prose → parameter validation raises `InvalidParameterError`"
contract: Refer to the accepted lowercase strings `"kdtree"` / `"balltree"` in the prose (keep class links separate).
instances: single-instance
```
</details>

### [LOW] `_weighted_cluster_center` inline comment is ungrammatical/misleading

Reader trying to understand why iteration over clusters is necessary → gets no useful information and must reverse-engineer the invariant from the code Replace with a comment explaining the actual invariant (clusters have varying sizes so results cannot be stacked into a homogeneous 3-D array).

<details><summary>verbatim finding</summary>

```
### F192 — `_weighted_cluster_center` inline comment is ungrammatical/misleading
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 — `# Need to handle iteratively seen each cluster may have a different` reads as a truncated/garbled sentence and does not describe the iteration invariant it precedes
scenario: "Reader trying to understand why iteration over clusters is necessary → gets no useful information and must reverse-engineer the invariant from the code"
contract: Replace with a comment explaining the actual invariant (clusters have varying sizes so results cannot be stacked into a homogeneous 3-D array).
instances: single-instance
```
</details>

### [LOW] Note about `dbscan_clustering` returning `-3` labels only applies when `metric != "precomputed"` but docstring omits that

User with `metric='precomputed'` expects the `-3` code and writes downstream logic checking for it → the code path never emits `-3` Note in the Returns block that the `-3` code is only produced when `metric != "precomputed"`.

<details><summary>verbatim finding</summary>

```
### F193 — Note about `dbscan_clustering` returning `-3` labels only applies when `metric != "precomputed"` but docstring omits that
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:952-958 — Returns section unconditionally lists `-3` for "Samples with missing data"; but per sklearn/cluster/_hdbscan/hdbscan.py:734-751, missing-data (`np.nan`) rejection differs between `metric="precomputed"` (raises ValueError before any label is assigned) and the non-precomputed path (uses `-3`), so `-3` never occurs under precomputed
scenario: "User with `metric='precomputed'` expects the `-3` code and writes downstream logic checking for it → the code path never emits `-3`"
contract: Note in the Returns block that the `-3` code is only produced when `metric != "precomputed"`.
instances: single-instance
```
</details>

### [LOW] `remap_single_linkage_tree` public helper missing a `Returns` section

Consumer of `remap_single_linkage_tree` expects an in-place mutation per the docstring's silent Returns → drops the return value and works on a stale tree lacking the concatenated outlier rows Add a `Returns` block documenting the returned `tree : ndarray of shape (n_samples - 1 + len(non_finite),), dtype=HIERARCHY_dtype` and explicitly note that the input is also mutated.

<details><summary>verbatim finding</summary>

```
### F194 — `remap_single_linkage_tree` public helper missing a `Returns` section
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-366 — docstring documents `tree`, `internal_to_raw`, `non_finite` but never documents the returned `tree` (which is a newly concatenated ndarray, per the function's `return tree` at :393).
scenario: "Consumer of `remap_single_linkage_tree` expects an in-place mutation per the docstring's silent Returns → drops the return value and works on a stale tree lacking the concatenated outlier rows"
contract: Add a `Returns` block documenting the returned `tree : ndarray of shape (n_samples - 1 + len(non_finite),), dtype=HIERARCHY_dtype` and explicitly note that the input is also mutated.
instances: single-instance
```
</details>

### [LOW] `_hdbscan_brute`/`_hdbscan_prims` docstring misdescribes `metric_params` as a dict with default None

Caller passes `metric_params={'V': cov}` per docstring → the helper receives a single kwarg named `metric_params` inside `**metric_params`, then `metric_params.get('max_distance', 0.0)` at sklearn/cluster/_hdbscan/hdbscan.py:243 silently misbehaves because the real metric params were never unpacked Rewrite the docstring to reflect the `**metric_params` API ("Additional keyword arguments passed to the distance metric.") and drop the fictitious `default=None`.

<details><summary>verbatim finding</summary>

```
### F195 — `_hdbscan_brute`/`_hdbscan_prims` docstring misdescribes `metric_params` as a dict with default None
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:214-215 — documents `metric_params : dict, default=None` but the signature captures it via `**metric_params` (i.e. it is not a dict-typed keyword and has no default; individual keyword arguments are unpacked). The same misdescription exists in `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:320-321.
scenario: "Caller passes `metric_params={'V': cov}` per docstring → the helper receives a single kwarg named `metric_params` inside `**metric_params`, then `metric_params.get('max_distance', 0.0)` at sklearn/cluster/_hdbscan/hdbscan.py:243 silently misbehaves because the real metric params were never unpacked"
contract: Rewrite the docstring to reflect the `**metric_params` API ("Additional keyword arguments passed to the distance metric.") and drop the fictitious `default=None`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:214, sklearn/cluster/_hdbscan/hdbscan.py:320]
```
</details>

### [LOW] `dbscan_clustering` docstring uses first-person narration and duplicated tautology instead of a precise invariant

Reader parsing the docstring is unsure whether the method depends on prior `fit` or on a fresh single-linkage build → the three narrations blur the actual precondition (that `self._single_linkage_tree_` must already exist) Collapse the three narrations into a single sentence stating the precondition ("Requires the estimator to be fitted first.") and the operation ("Returns the flat clustering obtained by cutting the fitted single-linkage tree at `cut_distance`, discarding clusters smaller than `min_cluster_size` as noise."), and fix the missing period at line 928.

<details><summary>verbatim finding</summary>

```
### F197 — `dbscan_clustering` docstring uses first-person narration and duplicated tautology instead of a precise invariant
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:926-937 — "Return clustering that would be equivalent to running DBSCAN* for a particular cut_distance (or epsilon) DBSCAN* can be thought of as DBSCAN without the border points." (missing punctuation between two sentences), followed by "This can also be thought of as a flat clustering derived from constant height cut through the single linkage tree." and "This represents the result of selecting a cut value for robust single linkage clustering." — three overlapping narrations of the same idea, and the missing period at ":928" creates a run-on sentence.
scenario: "Reader parsing the docstring is unsure whether the method depends on prior `fit` or on a fresh single-linkage build → the three narrations blur the actual precondition (that `self._single_linkage_tree_` must already exist)"
contract: Collapse the three narrations into a single sentence stating the precondition ("Requires the estimator to be fitted first.") and the operation ("Returns the flat clustering obtained by cutting the fitted single-linkage tree at `cut_distance`, discarding clusters smaller than `min_cluster_size` as noise."), and fix the missing period at line 928.
instances: single-instance
```
</details>

### [LOW] `HDBSCAN` docstring's `store_centers="centroid"` description omits that it is only meaningful for Euclidean-compatible metrics [out-of-theme]

User with `metric='precomputed'` sets `store_centers='centroid'` → silently receives meaningless 'centroids' that are averages of distance-matrix rows, because the docstring does not warn nor does the code guard The docstring must explicitly state that `store_centers` is only valid when `X` is a feature matrix (not a precomputed distance matrix), and mention that centroids are computed under the Euclidean metric regardless of the estimator's `metric` argument.

<details><summary>verbatim finding</summary>

```
### F198 — `HDBSCAN` docstring's `store_centers="centroid"` description omits that it is only meaningful for Euclidean-compatible metrics [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:508-511 — "`"centroid"` which calculates the center by taking the weighted average of their positions. Note that the algorithm uses the euclidean metric and does not guarantee that the output will be an observed data point." However, `_weighted_cluster_center` at sklearn/cluster/_hdbscan/hdbscan.py:912 uses `np.average` regardless of the user's `metric`, and there is no validation preventing `metric="precomputed"` combined with `store_centers="centroid"` (which would attempt to average distance-matrix rows as if they were coordinates).
scenario: "User with `metric='precomputed'` sets `store_centers='centroid'` → silently receives meaningless 'centroids' that are averages of distance-matrix rows, because the docstring does not warn nor does the code guard"
contract: The docstring must explicitly state that `store_centers` is only valid when `X` is a feature matrix (not a precomputed distance matrix), and mention that centroids are computed under the Euclidean metric regardless of the estimator's `metric` argument.
instances: single-instance
```
</details>

### [LOW] `n_jobs` default of 4 is unusual and can silently overfetch cores [out-of-theme]

User instantiates `HDBSCAN()` on an 8-core machine expecting the documented single-thread default → four parallel worker processes/threads spin up unexpectedly, contending with other work; docstring/default mismatch also violates common sklearn parameter conventions Set the constructor default to `n_jobs=None` to match the documented behavior and the sklearn-wide convention.

<details><summary>verbatim finding</summary>

```
### F213 — `n_jobs` default of 4 is unusual and can silently overfetch cores [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` is the constructor default. Elsewhere in scikit-learn the convention is `n_jobs=None` (meaning 1 unless in a joblib parallel_backend context); the class docstring at lines 486-490 explicitly documents "None means 1 unless in a joblib.parallel_backend context" — contradicting the actual default of 4.
scenario: "User instantiates `HDBSCAN()` on an 8-core machine expecting the documented single-thread default → four parallel worker processes/threads spin up unexpectedly, contending with other work; docstring/default mismatch also violates common sklearn parameter conventions"
contract: Set the constructor default to `n_jobs=None` to match the documented behavior and the sklearn-wide convention.
instances: single-instance
```
</details>

### [LOW] `remap_single_linkage_tree` iterates in Python instead of vectorizing

Fit with `force_all_finite=False` and many non-finite rows → post-processing remap runs in Python-loop time on an array of size (n_finite - 1) Rewrite both loops with vectorized numpy operations (boolean masking with `np.where` for the remap, arithmetic-progression assignment for `outlier_tree`).

<details><summary>verbatim finding</summary>

```
### F214 — `remap_single_linkage_tree` iterates in Python instead of vectorizing
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370-392 — `for i, _ in enumerate(tree):` accesses `tree[i]["left_node"]`/`tree[i]["right_node"]` and writes them back one element at a time, followed by another Python `for i, outlier in enumerate(non_finite):` loop to build `outlier_tree`. Both loops could be vectorized: mask + `np.where` for the remap and pure numpy assignment for `outlier_tree`.
scenario: "Fit with `force_all_finite=False` and many non-finite rows → post-processing remap runs in Python-loop time on an array of size (n_finite - 1)"
contract: Rewrite both loops with vectorized numpy operations (boolean masking with `np.where` for the remap, arithmetic-progression assignment for `outlier_tree`).
instances: single-instance
```
</details>

### [LOW] `_hdbscan_prims` requests `min_samples` neighbors then discards all but the k-th

Fit with large `min_samples` → NearestNeighbors returns an O(n × min_samples) distance array of which only the last column is used, wasting memory and copy bandwidth (`np.ascontiguousarray` allocates a fresh contiguous slice). For n=10⁶, min_samples=50 that is 400 MB of transient allocation to obtain 8 MB of core distances. Fetch only the k-th neighbor distance via `nbrs.kneighbors(X, min_samples)[0][:, -1]` combined with an explicit contiguous output buffer, or add a code path that avoids returning the full neighbor distance matrix when only the k-th column is required.

<details><summary>verbatim finding</summary>

```
### F215 — `_hdbscan_prims` requests `min_samples` neighbors then discards all but the k-th
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:342-343 — `neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True); core_distances = np.ascontiguousarray(neighbors_distances[:, -1])`. All but the last column of the returned `(n_samples, min_samples)` array is thrown away.
scenario: "Fit with large `min_samples` → NearestNeighbors returns an O(n × min_samples) distance array of which only the last column is used, wasting memory and copy bandwidth (`np.ascontiguousarray` allocates a fresh contiguous slice). For n=10⁶, min_samples=50 that is 400 MB of transient allocation to obtain 8 MB of core distances."
contract: Fetch only the k-th neighbor distance via `nbrs.kneighbors(X, min_samples)[0][:, -1]` combined with an explicit contiguous output buffer, or add a code path that avoids returning the full neighbor distance matrix when only the k-th column is required.
instances: single-instance
```
</details>

### [LOW] Module-level function `remap_single_linkage_tree` exposed without underscore prefix from the internal `_hdbscan.hdbscan` module, and placed in the estimator file rather than with the tree data it manipulates

Downstream code imports `sklearn.cluster._hdbscan.hdbscan.remap_single_linkage_tree` treating the un-prefixed name as public → subsequent refactor breaks it; meanwhile logic operating on `HIERARCHY_dtype` sits far from its type definition, forcing a Python-layer touch every time the tree layout changes. A helper that operates exclusively on the private `HIERARCHY_dtype` must (a) be underscore-prefixed to mark it internal and (b) live in `_tree.pyx` (or a private `_tree`-adjacent Python module) beside the dtype it mutates.

<details><summary>verbatim finding</summary>

```
### F223 — Module-level function `remap_single_linkage_tree` exposed without underscore prefix from the internal `_hdbscan.hdbscan` module, and placed in the estimator file rather than with the tree data it manipulates
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 defines `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):` at module top level with no underscore. It manipulates the `HIERARCHY_dtype` structured array (data owned by `_tree.pyx`) — indexing `tree[i]["left_node"]`, `tree[i]["right_node"]`, `tree[i]["cluster_size"]` — but lives in a Python file colocated with the estimator class rather than the module that owns the dtype. It is not re-exported through `sklearn.cluster.__init__` (see sklearn/cluster/__init__.py:26) yet its name suggests public API.
scenario: "Downstream code imports `sklearn.cluster._hdbscan.hdbscan.remap_single_linkage_tree` treating the un-prefixed name as public → subsequent refactor breaks it; meanwhile logic operating on `HIERARCHY_dtype` sits far from its type definition, forcing a Python-layer touch every time the tree layout changes."
contract: A helper that operates exclusively on the private `HIERARCHY_dtype` must (a) be underscore-prefixed to mark it internal and (b) live in `_tree.pyx` (or a private `_tree`-adjacent Python module) beside the dtype it mutates.
instances: single-instance
```
</details>

### [LOW] `_get_finite_row_indices` is a generic ndarray/sparse utility misplaced in the estimator module

Another estimator that needs finite-row filtering for sparse inputs → developer duplicates the same helper because it is buried in `_hdbscan.hdbscan`. Move `_get_finite_row_indices` to `sklearn/utils/validation.py` (or a nearby validation helper module) and cimport/import it from there.

<details><summary>verbatim finding</summary>

```
### F230 — `_get_finite_row_indices` is a generic ndarray/sparse utility misplaced in the estimator module
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 defines `_get_finite_row_indices(matrix)` — a pure array/sparse helper that has no HDBSCAN-specific knowledge (it only computes finite-row indices for dense arrays or LIL/sparse matrices). It is used only once at line 731, but its logic (finite-row detection for sparse+dense) is exactly the kind of thing that belongs alongside `_assert_all_finite` and `_allclose_dense_sparse` in `sklearn/utils/validation.py`.
scenario: "Another estimator that needs finite-row filtering for sparse inputs → developer duplicates the same helper because it is buried in `_hdbscan.hdbscan`."
contract: Move `_get_finite_row_indices` to `sklearn/utils/validation.py` (or a nearby validation helper module) and cimport/import it from there.
instances: single-instance
```
</details>

### [LOW] `_brute_mst` constructs raw `MST_edge_dtype` records in Python module using private `np.core.records`

NumPy removes access to `np.core.records` → HDBSCAN's sparse-`precomputed` code path breaks at import/call time even though `_linkage.pyx` (owner of the dtype) uses a fully supported constructor. Add a helper (e.g., `_mst_from_sparse_edges(rows, cols, data)`) to `_hdbscan/_linkage.pyx` that returns an `MST_edge_dtype` ndarray via `np.rec.fromarrays` or a plain structured `np.empty(...) + assignment`, and call that from `hdbscan.py`. Do not touch `np.core.*` from Python code.

<details><summary>verbatim finding</summary>

```
### F231 — `_brute_mst` constructs raw `MST_edge_dtype` records in Python module using private `np.core.records`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:128-131 builds MST records via `mst = np.core.records.fromarrays([rows, cols, sparse_min_spanning_tree.data], dtype=MST_edge_dtype)`. The dtype and layout are owned by `_linkage.pyx` (which defines both the `MST_edge_dtype` numpy dtype and the packed struct `MST_edge_t`), yet the construction path for the sparse case lives outside that module — logic far from its data. Additionally, `np.core.records` is a private numpy submodule (`np.core` is scheduled to become fully private in NumPy 2.x; already emits deprecation guidance).
scenario: "NumPy removes access to `np.core.records` → HDBSCAN's sparse-`precomputed` code path breaks at import/call time even though `_linkage.pyx` (owner of the dtype) uses a fully supported constructor."
contract: Add a helper (e.g., `_mst_from_sparse_edges(rows, cols, data)`) to `_hdbscan/_linkage.pyx` that returns an `MST_edge_dtype` ndarray via `np.rec.fromarrays` or a plain structured `np.empty(...) + assignment`, and call that from `hdbscan.py`. Do not touch `np.core.*` from Python code.
instances: single-instance
```
</details>

### [LOW] `HDBSCAN` docstring `Examples` block will fail doctest when run without a network / with a differently-shipped `load_digits` [out-of-theme]

sklearn docs/doctest CI runs the example under a different BLAS thread configuration → labels differ slightly and the doctest fails. Pass `n_jobs=1` explicitly in the doctest invocation so the pairwise computation is single-threaded and the hard-coded labels remain deterministic across environments.

<details><summary>verbatim finding</summary>

```
### F238 — `HDBSCAN` docstring `Examples` block will fail doctest when run without a network / with a differently-shipped `load_digits` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:604-613 shows a doctest that hard-codes exact cluster labels (`array([ 2,  6, -1, ..., -1, -1, -1])`) from `HDBSCAN(min_cluster_size=20).fit(load_digits(...))`, but the default constructor also sets `n_jobs=4` (F2) which triggers parallel pairwise-distance computation whose outputs are not deterministic across BLAS thread counts.
scenario: "sklearn docs/doctest CI runs the example under a different BLAS thread configuration → labels differ slightly and the doctest fails."
contract: Pass `n_jobs=1` explicitly in the doctest invocation so the pairwise computation is single-threaded and the hard-coded labels remain deterministic across environments.
instances: single-instance
```
</details>

### [LOW] `HDBSCAN.algorithm` option strings diverge from sklearn-wide `kd_tree`/`ball_tree` convention

User consulting sklearn documentation for `algorithm='ball_tree'` translates that call to `HDBSCAN(algorithm='ball_tree')` → InvalidParameterError because the parameter validator rejects underscored variants unique to the rest of the package. Accept the same string form (`"kd_tree"`, `"ball_tree"`) that every other sklearn estimator uses for the tree-based algorithm selectors.

<details><summary>verbatim finding</summary>

```
### F239 — `HDBSCAN.algorithm` option strings diverge from sklearn-wide `kd_tree`/`ball_tree` convention
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:616-645 registers `algorithm` allowed values as `{"auto", "brute", "kdtree", "balltree"}` (no underscore). The rest of scikit-learn (e.g. sklearn/neighbors/_base.py:392 `StrOptions({"auto", "ball_tree", "kd_tree", "brute"})`) uses `"kd_tree"`/`"ball_tree"`. Internally HDBSCAN then translates `"kdtree"`→`"kd_tree"` (line 798) and `"balltree"`→`"ball_tree"` (line 802) before passing to `NearestNeighbors`, confirming the underscore form is the sklearn convention it is silently mapping to.
scenario: "User consulting sklearn documentation for `algorithm='ball_tree'` translates that call to `HDBSCAN(algorithm='ball_tree')` → InvalidParameterError because the parameter validator rejects underscored variants unique to the rest of the package."
contract: Accept the same string form (`"kd_tree"`, `"ball_tree"`) that every other sklearn estimator uses for the tree-based algorithm selectors.
instances: single-instance
```
</details>

### [LOW] Two separate HDBSCAN test locations split coverage without a shared helper

Contributor changes the outlier semantics and updates test_hdbscan.py, missing test_reachibility.py → inconsistent test coverage across the two test directories, and the private-submodule tests may not be discovered by contributors who only look under `sklearn/cluster/tests/`. Consolidate all HDBSCAN tests under `sklearn/cluster/tests/test_hdbscan*.py` (splitting by aspect if desired, e.g. `test_hdbscan_reachability.py`) so the location convention matches every other sklearn cluster algorithm.

<details><summary>verbatim finding</summary>

```
### F115 — Two separate HDBSCAN test locations split coverage without a shared helper
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 and sklearn/cluster/tests/test_hdbscan.py:1 — the PR adds tests in two locations: `sklearn/cluster/tests/test_hdbscan.py` (the main sklearn cluster tests directory) and `sklearn/cluster/_hdbscan/tests/test_reachibility.py` (a private submodule tests directory that only tests one Cython file). The two files share utility import patterns (`sklearn.utils._testing`) and both need to construct outlier fixtures, but each file re-creates its own random data setup rather than sharing a fixture module. This duplicates the "location where a reader looks for HDBSCAN tests" and permits fixture drift.
scenario: "Contributor changes the outlier semantics and updates test_hdbscan.py, missing test_reachibility.py → inconsistent test coverage across the two test directories, and the private-submodule tests may not be discovered by contributors who only look under `sklearn/cluster/tests/`."
contract: Consolidate all HDBSCAN tests under `sklearn/cluster/tests/test_hdbscan*.py` (splitting by aspect if desired, e.g. `test_hdbscan_reachability.py`) so the location convention matches every other sklearn cluster algorithm.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/tests/test_hdbscan.py:1]
```
</details>

### [LOW] Misspelled test module filename `test_reachibility.py`

Developer greps for `test_reachability` → finds nothing, believes reachability code is untested Rename the file to `test_reachability.py`.

<details><summary>verbatim finding</summary>

```
### F155 — Misspelled test module filename `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — file name misspells "reachability" as "reachibility"; the module under test is `_reachability.pyx`.
scenario: "Developer greps for `test_reachability` → finds nothing, believes reachability code is untested"
contract: Rename the file to `test_reachability.py`.
instances: single-instance
```
</details>

### [LOW] Tests split across two locations for the same estimator

Contributor updates HDBSCAN reachability logic → searches `sklearn/cluster/tests/test_hdbscan.py` for related tests → does not find the reachability tests, which live in the nested `_hdbscan/tests/test_reachibility.py`, so related tests are easily missed / duplicated Consolidate all HDBSCAN tests under `sklearn/cluster/tests/` (e.g. merge `test_reachibility.py` content into a `test_reachability.py` there) following the convention used by every other estimator in `sklearn.cluster`

<details><summary>verbatim finding</summary>

```
### F222 — Tests split across two locations for the same estimator
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 exists (a whole new `tests/` folder inside the private subpackage) while sklearn/cluster/tests/test_hdbscan.py:1 holds all other HDBSCAN tests; every other cluster estimator keeps ALL its tests under `sklearn/cluster/tests/` (see `test_dbscan.py`, `test_hierarchical.py`, `test_optics.py`, etc.) and no other cluster estimator has an internal `tests/` folder
scenario: "Contributor updates HDBSCAN reachability logic → searches `sklearn/cluster/tests/test_hdbscan.py` for related tests → does not find the reachability tests, which live in the nested `_hdbscan/tests/test_reachibility.py`, so related tests are easily missed / duplicated"
contract: Consolidate all HDBSCAN tests under `sklearn/cluster/tests/` (e.g. merge `test_reachibility.py` content into a `test_reachability.py` there) following the convention used by every other estimator in `sklearn.cluster`
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/tests/__init__.py:1, sklearn/cluster/tests/test_hdbscan.py:1]
```
</details>

### [LOW] Test file name typo `test_reachibility.py` misspells "reachability"

Developer runs `pytest -k reachability` looking for the tests of `_reachability.py` → the misspelt file is not matched by the intuitive query and its tests are skipped/undiscovered Rename the file to `test_reachability.py` to match the module it tests

<details><summary>verbatim finding</summary>

```
### F225 — Test file name typo `test_reachibility.py` misspells "reachability"
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename `test_reachibility.py` while the module under test is `_reachability.py` (also see docstring at line 12: "mutual reachability graph")
scenario: "Developer runs `pytest -k reachability` looking for the tests of `_reachability.py` → the misspelt file is not matched by the intuitive query and its tests are skipped/undiscovered"
contract: Rename the file to `test_reachability.py` to match the module it tests
instances: single-instance
```
</details>

### [LOW] Manifest change to `sklearn/cluster/_hierarchical_fast.{pyx,pxd}` sanctioned by no plan item

Inverse audit class (b): a change in B without any sanctioning requirement in A → the refactor is plausibly needed as an enabler for HDBSCAN's `_linkage.pyx` (which is likely to cimport `UnionFind`) but the plan does not disclose this cross-module coupling change as one of its 'novel changes', so a reviewer cannot map the `_hierarchical_fast` edits to any itemized promise. The plan's 'novel changes' list should enumerate any modification to *unrelated* pre-existing modules made purely to enable the new estimator. This one is missing.

<details><summary>verbatim finding</summary>

```
### F4 — Manifest change to `sklearn/cluster/_hierarchical_fast.{pyx,pxd}` sanctioned by no plan item
severity: low
scenario: "Inverse audit class (b): a change in B without any sanctioning requirement in A → the refactor is plausibly needed as an enabler for HDBSCAN's `_linkage.pyx` (which is likely to cimport `UnionFind`) but the plan does not disclose this cross-module coupling change as one of its 'novel changes', so a reviewer cannot map the `_hierarchical_fast` edits to any itemized promise."
contract: "The plan's 'novel changes' list should enumerate any modification to *unrelated* pre-existing modules made purely to enable the new estimator. This one is missing."
evidence:
- Manifest entries: `M sklearn/cluster/_hierarchical_fast.pyx +0/-5` (DIFF_MANIFEST.md:28) and `A sklearn/cluster/_hierarchical_fast.pxd +9/-0` (DIFF_MANIFEST.md:27).
- Diff content (from `hunks/sklearn_cluster__hierarchical_fast.pyx.diff` and `hunks/sklearn_cluster__hierarchical_fast.pxd.diff`): moves `UnionFind`'s `cdef` field declarations (`next_label`, `parent`, `size`) plus `union`/`fast_find` method signatures out of the `.pyx` and into a new `.pxd` header, presumably to allow other modules (e.g., `_hdbscan/_linkage.pyx`) to `cimport UnionFind`.
- Plan (PLAN.md:14–17) enumerates three novel changes; none mention promoting `UnionFind` fields to a public `.pxd` header, nor does the mandatory-work checklist (PLAN.md:36–48) reference `_hierarchical_fast`.

instances: [sklearn/cluster/_hierarchical_fast.pyx:1, sklearn/cluster/_hierarchical_fast.pxd:1]
```
</details>

### [LOW] `UnionFind` `.pxd`/`.pyx` `noexcept` mismatch swallows exceptions raised in `union`/`fast_find`

Cython 3 treats the `.pxd` declaration as authoritative → any Python-level exception raised inside `union` or `fast_find` (e.g. `IndexError` from bounds checks, allocation failures within the memoryview writes) is silently swallowed with only a WriteUnraisable to stderr, and the calling `make_single_linkage` continues with a corrupted UnionFind, producing a garbage single-linkage tree The `.pyx` definitions MUST repeat the `noexcept` qualifier from the `.pxd` so the exception-propagation policy is unambiguous and reviewed

<details><summary>verbatim finding</summary>

```
### F44 — `UnionFind` `.pxd`/`.pyx` `noexcept` mismatch swallows exceptions raised in `union`/`fast_find`
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` — neither definition repeats the `noexcept` qualifier that the `.pxd` header pins
scenario: "Cython 3 treats the `.pxd` declaration as authoritative → any Python-level exception raised inside `union` or `fast_find` (e.g. `IndexError` from bounds checks, allocation failures within the memoryview writes) is silently swallowed with only a WriteUnraisable to stderr, and the calling `make_single_linkage` continues with a corrupted UnionFind, producing a garbage single-linkage tree"
contract: The `.pyx` definitions MUST repeat the `noexcept` qualifier from the `.pxd` so the exception-propagation policy is unambiguous and reviewed
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
```
</details>

### [LOW] `.pxd` declares `noexcept` but `.pyx` definitions omit it

Cython compiles with strict signature matching → mismatched exception-spec between `.pxd` and `.pyx` triggers warning/error, or exception propagation semantics differ from what the `.pxd` advertises to external cimporters Add `noexcept` to both method definitions in `_hierarchical_fast.pyx` so the pyx matches the pxd declaration exactly.

<details><summary>verbatim finding</summary>

```
### F126 — `.pxd` declares `noexcept` but `.pyx` definitions omit it
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:14-15 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, but the corresponding definitions in sklearn/cluster/_hierarchical_fast.pyx:331 (`cdef void union(self, intp_t m, intp_t n):`) and sklearn/cluster/_hierarchical_fast.pyx:339 (`cdef intp_t fast_find(self, intp_t n):`) omit `noexcept`. The `noexcept` in the pxd is either a lost-semantics attempt to declare no-exception behavior that the pyx does not honor, or dead annotation that will trigger Cython deprecation warnings when signatures mismatch.
scenario: "Cython compiles with strict signature matching → mismatched exception-spec between `.pxd` and `.pyx` triggers warning/error, or exception propagation semantics differ from what the `.pxd` advertises to external cimporters"
contract: Add `noexcept` to both method definitions in `_hierarchical_fast.pyx` so the pyx matches the pxd declaration exactly.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
```
</details>

### [LOW] `_hierarchical_fast.pyx` deletion of `cdef` block leaves no comment tying the fields to the new `.pxd` [out-of-theme]

Future contributor editing `_hierarchical_fast.pyx` reintroduces `cdef intp_t next_label` for clarity → Cython 'field defined twice' error, or removes the pxd thinking it is dead Add a one-line comment in `_hierarchical_fast.pyx` above `cdef class UnionFind(object):` pointing at `_hierarchical_fast.pxd` where the fields and cdef methods are now declared.

<details><summary>verbatim finding</summary>

```
### F189 — `_hierarchical_fast.pyx` deletion of `cdef` block leaves no comment tying the fields to the new `.pxd` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pyx:322-337 — the three `cdef intp_t next_label / intp_t[:] parent / intp_t[:] size` declarations were removed from the `UnionFind` class body and moved to the new `sklearn/cluster/_hierarchical_fast.pxd`. Neither file carries a comment noting that the declarations are now in the `.pxd` (nor that removing them again would break `_hdbscan/_linkage.pyx` which `cimport`s `UnionFind`). This is a lost-semantics documentation regression: the previous single-file view was self-describing; the current split has no cross-reference.
scenario: "Future contributor editing `_hierarchical_fast.pyx` reintroduces `cdef intp_t next_label` for clarity → Cython 'field defined twice' error, or removes the pxd thinking it is dead"
contract: Add a one-line comment in `_hierarchical_fast.pyx` above `cdef class UnionFind(object):` pointing at `_hierarchical_fast.pxd` where the fields and cdef methods are now declared.
instances: single-instance
```
</details>

### [LOW] `test_hdbscan_algorithms` first assertion ignores the parametrized `metric`

regression in a non-euclidean metric path → early per-metric assertion still passes because it silently uses euclidean Move the `n_clusters == n_clusters_true` block after the `metric` is applied so it actually exercises the parametrized metric.

<details><summary>verbatim finding</summary>

```
### F64 — `test_hdbscan_algorithms` first assertion ignores the parametrized `metric`
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:136-145 — the test is `@pytest.mark.parametrize("metric", _VALID_METRICS)` yet line 143 calls `HDBSCAN(algorithm=algo).fit_predict(X)` with the default `metric="euclidean"`, then asserts `n_clusters == n_clusters_true`. This assertion runs identically for every `metric` value, so the parametrization does not verify anything about the metric for that portion of the test — the metric is only used after the early return on line 149.
scenario: "regression in a non-euclidean metric path → early per-metric assertion still passes because it silently uses euclidean"
contract: Move the `n_clusters == n_clusters_true` block after the `metric` is applied so it actually exercises the parametrized metric.
instances: single-instance
```
</details>

### [LOW] `test_hdbscan_algorithms` invalid-metric branch relies on bare `pytest.raises(ValueError)` and cannot distinguish parameter validation from algorithm-metric mismatch

the specific 'not a valid metric for a KDTree-based algorithm' rejection breaks or moves → test still passes because InvalidParameterError from `_validate_params` masks the change Add `match=` matching the specific "is not a valid metric for a .*-based algorithm" message from hdbscan.py:774/781.

<details><summary>verbatim finding</summary>

```
### F65 — `test_hdbscan_algorithms` invalid-metric branch relies on bare `pytest.raises(ValueError)` and cannot distinguish parameter validation from algorithm-metric mismatch
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:168-170 — `with pytest.raises(ValueError): hdb.fit(X)` has no `match=` argument. HDBSCAN's `_parameter_constraints` (hdbscan.py:626) restricts `metric` to `FAST_METRICS | {"precomputed"}` (plus callable), so many `_VALID_METRICS` entries raise `InvalidParameterError` (a `ValueError` subclass) at `_validate_params`, before the intended KDTree/BallTree metric-validity check at hdbscan.py:772-783 ever runs. The test conflates two different code paths.
scenario: "the specific 'not a valid metric for a KDTree-based algorithm' rejection breaks or moves → test still passes because InvalidParameterError from `_validate_params` masks the change"
contract: Add `match=` matching the specific "is not a valid metric for a .*-based algorithm" message from hdbscan.py:774/781.
instances: single-instance
```
</details>

### [LOW] `store_centers="centroid"` and `store_centers="medoid"` branches are untested (only `"both"` is exercised)

A regression makes `_weighted_cluster_center` always compute both regardless of `store_centers` → not caught. Add parametrized tests over `store_centers in {None, "centroid", "medoid", "both"}` asserting exactly which of `centroids_` and `medoids_` are set.

<details><summary>verbatim finding</summary>

```
### F68 — `store_centers="centroid"` and `store_centers="medoid"` branches are untested (only `"both"` is exercised)
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:309-327 — the only test of centers passes `store_centers="both"`. The independent branches (`"centroid"` alone should set `centroids_` but not `medoids_`; `"medoid"` alone the reverse) at sklearn/cluster/_hdbscan/hdbscan.py:897-903 are never verified. In particular, no test asserts that attribute `medoids_` is missing when `store_centers="centroid"` (and vice-versa), nor that `store_centers=None` sets no attributes.
scenario: "A regression makes `_weighted_cluster_center` always compute both regardless of `store_centers` → not caught."
contract: Add parametrized tests over `store_centers in {None, "centroid", "medoid", "both"}` asserting exactly which of `centroids_` and `medoids_` are set.
instances: single-instance
```
</details>

### [LOW] Assertion `assert counts[unique_labels == -1] > 30` compares an array to a scalar

Numpy tightens the scalar-conversion rule → these asserts start emitting DeprecationWarning or raising. Extract the scalar with `.item()` (or index `[0]`) before comparison, e.g. `assert counts[unique_labels == -1].item() > 30`.

<details><summary>verbatim finding</summary>

```
### F69 — Assertion `assert counts[unique_labels == -1] > 30` compares an array to a scalar
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:347 — `counts[unique_labels == -1]` is a length-1 ndarray, not a Python scalar; comparing a length-1 array `> 30` returns a length-1 boolean array. `assert` on a length-1 array works today by scalar-truth-test, but the pattern is fragile (a numpy deprecation would break it) and inconsistent with the neighboring `counts[unique_labels == -1] == 2` at test_hdbscan.py:360 which is also a length-1 array vs scalar.
scenario: "Numpy tightens the scalar-conversion rule → these asserts start emitting DeprecationWarning or raising."
contract: Extract the scalar with `.item()` (or index `[0]`) before comparison, e.g. `assert counts[unique_labels == -1].item() > 30`.
instances: [sklearn/cluster/tests/test_hdbscan.py:347, sklearn/cluster/tests/test_hdbscan.py:360]
```
</details>

### [LOW] `copy=True` behavior is only verified against `metric="precomputed"`, not against the brute/sparse branches

A regression that mutates a caller's CSR precomputed sparse matrix even under `copy=True` → uncaught, silent data corruption for users. Extend the copy test to cover sparse CSR precomputed inputs, asserting the caller's sparse array remains unchanged after `fit_predict`.

<details><summary>verbatim finding</summary>

```
### F73 — `copy=True` behavior is only verified against `metric="precomputed"`, not against the brute/sparse branches
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:79-88 — `HDBSCAN(metric="precomputed", copy=True).fit_predict(D); assert_allclose(D, D_original)`. The docstring for `copy` (hdbscan.py:521-526) states it applies to precomputed dense/CSR-sparse inputs when `algorithm="brute"`. The sparse `copy=True` path (which triggers a distinct code path via `_brute_mst` and in-place `_sparse_mutual_reachability_graph`) is not tested for immutability of the caller's data.
scenario: "A regression that mutates a caller's CSR precomputed sparse matrix even under `copy=True` → uncaught, silent data corruption for users."
contract: Extend the copy test to cover sparse CSR precomputed inputs, asserting the caller's sparse array remains unchanged after `fit_predict`.
instances: single-instance
```
</details>

### [LOW] `test_hdbscan_no_clusters` only checks the cluster count, not that all points are noise-labeled

A regression that mislabels all points as `-2` (infinite) or `-3` (missing) on finite data → the \"no clusters\" test still passes; users get incorrect outlier semantics. Additionally assert `np.all(labels == -1)` for finite input when the min_cluster_size guarantees no cluster.

<details><summary>verbatim finding</summary>

```
### F74 — `test_hdbscan_no_clusters` only checks the cluster count, not that all points are noise-labeled
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:249-251 — `labels = HDBSCAN(min_cluster_size=len(X) - 1).fit_predict(X); n_clusters = len(set(labels) - OUTLIER_SET); assert n_clusters == 0`. This passes as long as no non-outlier label is produced, but says nothing about whether points receive the expected noise label (`-1`) versus, e.g., all being mislabeled as `-2`/`-3` (which are in OUTLIER_SET and thus subtracted).
scenario: "A regression that mislabels all points as `-2` (infinite) or `-3` (missing) on finite data → the \"no clusters\" test still passes; users get incorrect outlier semantics."
contract: Additionally assert `np.all(labels == -1)` for finite input when the min_cluster_size guarantees no cluster.
instances: single-instance
```
</details>

### [LOW] `test_dbscan_clustering_outlier_data` uses element-wise `+` on `np.ndarray` indices where set-union/concatenation is intended [out-of-theme]

Any future fixture change that puts a nonzero index into `infinite_labels_idx` → `clean_idx` is wrong, `clean_model` fits the wrong subset, and assertion at line 215 may compare different points Concatenate the two arrays explicitly, e.g. `set(np.concatenate([missing_labels_idx, infinite_labels_idx]).tolist())`.

<details><summary>verbatim finding</summary>

```
### F90 — `test_dbscan_clustering_outlier_data` uses element-wise `+` on `np.ndarray` indices where set-union/concatenation is intended [out-of-theme]
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:212 computes `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`, where `missing_labels_idx` and `infinite_labels_idx` are `np.ndarray`s returned by `np.flatnonzero` at lines 206 and 209. `missing_labels_idx + infinite_labels_idx` performs element-wise addition (with broadcasting from shape `(1,)` to `(2,)`), producing `[2, 5]` for the specific fixture, not the intended concatenation `[2, 5, 0]`. The test only passes because the broadcast happens to leave the "missing" indices unchanged; had `infinite_labels_idx` contained a nonzero value, `clean_idx` would silently omit a different row than intended.
scenario: "Any future fixture change that puts a nonzero index into `infinite_labels_idx` → `clean_idx` is wrong, `clean_model` fits the wrong subset, and assertion at line 215 may compare different points"
contract: Concatenate the two arrays explicitly, e.g. `set(np.concatenate([missing_labels_idx, infinite_labels_idx]).tolist())`.
instances: single-instance
```
</details>

### [LOW] `HDBSCAN` omitted from `test_f_contiguous_array_estimator` parametrisation alongside its density-based siblings

A future change to `NearestNeighbors` or `pairwise_distances` regresses F-contiguous handling → the check that already gates DBSCAN-family estimators fails to catch the regression for HDBSCAN, whose contiguity assumption is explicitly enforced by `X = np.asarray(X, order='C')` at sklearn/cluster/_hdbscan/hdbscan.py:328 Add `HDBSCAN` to the parametrised estimator list at sklearn/tests/test_common.py:522 so the F-contiguous non-regression coverage extends to it.

<details><summary>verbatim finding</summary>

```
### F241 — `HDBSCAN` omitted from `test_f_contiguous_array_estimator` parametrisation alongside its density-based siblings
severity: low
evidence: sklearn/tests/test_common.py:522 lists `OPTICS` (and other neighbour-based estimators) as the estimator set under the non-regression test for F-contiguous input; `HDBSCAN` — a density-based clusterer that also delegates to `NearestNeighbors`/`KDTree`/`BallTree` via `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:341-354 — is not registered in that list
scenario: "A future change to `NearestNeighbors` or `pairwise_distances` regresses F-contiguous handling → the check that already gates DBSCAN-family estimators fails to catch the regression for HDBSCAN, whose contiguity assumption is explicitly enforced by `X = np.asarray(X, order='C')` at sklearn/cluster/_hdbscan/hdbscan.py:328"
contract: Add `HDBSCAN` to the parametrised estimator list at sklearn/tests/test_common.py:522 so the F-contiguous non-regression coverage extends to it.
instances: single-instance
```
</details>

### [LOW] `estimator_checks.py` common-checks layer hard-codes knowledge of a specific concrete estimator

Every additional estimator that fails a common check adds another `if name == 'X'` branch → `_set_checking_parameters` becomes a scattered registry of estimator quirks rather than a generic helper. Estimator-specific check parameters should be advertised by the estimator itself (e.g., via a `_more_tags` entry such as `_xfail_checks` / `check_parameters`, or a class attribute), and `_set_checking_parameters` should read from that surface instead of switching on `name`.

<details><summary>verbatim finding</summary>

```
### F232 — `estimator_checks.py` common-checks layer hard-codes knowledge of a specific concrete estimator
severity: low
evidence: sklearn/utils/estimator_checks.py:777-778 adds `if name == "HDBSCAN": estimator.set_params(min_samples=1)`. `_set_checking_parameters` is a generic utility that is now growing a per-estimator special case (joining a small existing set such as `SpectralEmbedding`). The dependency direction is inverted: a utility in `sklearn.utils` knows about a specific estimator in `sklearn.cluster._hdbscan`.
scenario: "Every additional estimator that fails a common check adds another `if name == 'X'` branch → `_set_checking_parameters` becomes a scattered registry of estimator quirks rather than a generic helper."
contract: Estimator-specific check parameters should be advertised by the estimator itself (e.g., via a `_more_tags` entry such as `_xfail_checks` / `check_parameters`, or a class attribute), and `_set_checking_parameters` should read from that surface instead of switching on `name`.
instances: single-instance
```
</details>
