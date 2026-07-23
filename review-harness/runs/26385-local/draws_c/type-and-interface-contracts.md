### F1 — `store_centers` with non-finite data raises IndexError due to mask/data length mismatch
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,908-909 — line 733 sets `X = X[finite_index]` (filtered), lines 840-844 remap `self.labels_` to shape `self._raw_data.shape[0]` (full), then line 855 calls `self._weighted_cluster_center(X)` with filtered `X`, and inside line 908 does `mask = self.labels_ == idx` (full length) then `data = X[mask]` (indexing filtered X with full-length mask).
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan or np.inf → boolean index of length `n_raw` applied to X of length `n_finite` raises IndexError, `fit` fails."
contract: When `store_centers` is set and non-finite data is present, `_weighted_cluster_center` must operate on labels restricted to the finite subset (or X must be re-indexed to raw shape with matching label lengths); the two arrays passed through the mask must have identical length.
instances: single-instance

### F2 — `labels_` dtype is runtime-dependent (int32 vs intp)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:822,840,844 — `tree_to_labels` returns `cnp.intp_t[::1]` labels (int64 on 64-bit), assigned to `self.labels_`; when non-finite data is present, `new_labels = np.empty(self._raw_data.shape[0], dtype=np.int32)` is created and assigned back to `self.labels_`, silently narrowing the dtype and altering the public attribute's dtype based on input contents.
scenario: "downstream code that relies on `HDBSCAN().fit(X).labels_.dtype` sees intp (int64) for clean input but int32 for input containing nan/inf → dtype-sensitive indexing/serialization breaks unpredictably."
contract: `labels_` must have a single fixed dtype (`np.intp`) regardless of whether the input contains non-finite values.
instances: single-instance

### F3 — `_weighted_cluster_center` counts `-3` (missing) label as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; `_OUTLIER_ENCODING["missing"]["label"]` is `-3` per lines 73-78, but `-3` is not excluded from the noise-set here even though centroids/medoids should not be computed for missing samples.
scenario: "`HDBSCAN(store_centers='both').fit(X)` with X containing np.nan → `-3` is treated as a valid cluster index; `mask = self.labels_ == -3` never becomes true for `idx in range(n_clusters)` (since range uses non-negative ids), producing an off-by-one and empty averaging/inf medoid computation for the phantom cluster." 
contract: The set of "cluster-excluding" outlier labels used to compute `n_clusters` must include every label in `_OUTLIER_ENCODING`, i.e. `{-1, -2, -3}`.
instances: single-instance

### F4 — `max_distance` remains in `metric_params` and is forwarded to `pairwise_distances`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:239,243 — `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` on line 239 spreads `metric_params`; line 243 `max_distance = metric_params.get("max_distance", 0.0)` reads but does not pop. If the caller supplies `metric_params={"max_distance": ...}`, `pairwise_distances` receives the unknown keyword and propagates it to underlying metric functions.
scenario: "user passes `metric_params={'max_distance': 5.0}` with a standard metric like 'euclidean' → `pairwise_distances` raises TypeError from unknown keyword forwarded to the metric callable/scipy backend."
contract: `max_distance` must be extracted and removed from `metric_params` (e.g. via `pop`) before `metric_params` is spread to `pairwise_distances`.
instances: single-instance

### F5 — `_hdbscan_brute` default `alpha=None` contradicts docstring and would crash division
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161,183-184,241 — signature has `alpha=None`, docstring says `alpha : float, default=1.0`, and line 241 does `distance_matrix /= alpha`. If called at its default, `None / ndarray` raises TypeError.
scenario: "any direct caller (or future refactor) invoking `_hdbscan_brute(X)` without an explicit `alpha` → TypeError on line 241."
contract: The signature default for `alpha` must be `1.0`, matching the documented contract and the arithmetic use.
instances: single-instance

### F6 — `_hdbscan_prims` / `_hdbscan_brute` signatures declare `min_samples=5` but docstring says `default=None`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160,180-181,272,290-291 — both functions have `min_samples=5` in the signature; both docstrings say `min_samples : int, default=None`.
scenario: "docs/signature mismatch → downstream reviewers or users misread the contract for private helpers, and the documented `None` value would break `NearestNeighbors(n_neighbors=None)` inside `_hdbscan_prims`."
contract: The docstring `default` for `min_samples` must match the signature's actual default (`5`).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:161, sklearn/cluster/_hdbscan/hdbscan.py:272]

### F7 — `_hdbscan_prims` documents a `copy` parameter that its signature does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278,313-318 — signature lists `X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params` with no `copy` parameter, yet the docstring includes a `copy : bool, default=False` parameter block.
scenario: "reader/refactorer relies on the documented `copy` parameter → passes `copy=...` which lands in `metric_params` (and forwards to distance metric functions), silently changing behavior or causing a TypeError."
contract: The `Parameters` section must reflect the function's actual accepted arguments — the `copy` entry must be removed (or the signature must accept it).
instances: single-instance

### F8 — `HDBSCAN.__init__` `n_jobs=4` contradicts documented default behavior
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658,486-490 — signature has `n_jobs=4`, docstring says "`None` means 1 unless in a :obj:`joblib.parallel_backend` context", which is the sklearn convention that requires the default to be `None`.
scenario: "user relies on the documented contract that `n_jobs=None` is default and integrates with a `joblib.parallel_backend` context → actual default `4` bypasses the context, using 4 threads regardless."
contract: `n_jobs` default must be `None` to match the documented `n_jobs` glossary convention.
instances: single-instance

### F9 — `remap_single_linkage_tree` type contract: `non_finite` documented as boolean ndarray but called with a `set`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:362-366,383,388-389,838 — docstring says `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`; caller at line 838 passes `non_finite=set(infinite_index + missing_index)` (a `set`), and line 383 uses `np.zeros(len(non_finite), ...)` plus `for i, outlier in enumerate(non_finite):` (iterating a set, so `outlier` is an integer index, not a boolean). The parameter's runtime role is a set of raw-data indices, not a boolean array.
scenario: "reader trusting the docstring passes an actual boolean ndarray → `outlier` becomes 0/1, and `outlier_tree[i] = (outlier, ...)` seeds the tree with 0/1 indices instead of the intended sample indices, corrupting the returned tree."
contract: The `non_finite` parameter must be documented and typed as `set[int]` of raw-data indices to match its actual use.
instances: single-instance

### F10 — `_do_labelling` untyped `label` and reliance on length-1 boolean-array truthiness
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:492,498,505 — `label` is never declared in the `cdef` block, `parent_lambda = lambda_array[child_array == n]` yields an ndarray, and `if parent_lambda >= threshold:` uses numpy array truthiness that only works when the mask selects exactly one element (relying on the comment invariant "one edge with this particular child").
scenario: "future refactor of the condensed-tree layout violates the one-edge-per-child invariant → `if parent_lambda >= threshold` raises `ValueError: The truth value of an array with more than one element is ambiguous`."
contract: `label` must be `cdef cnp.intp_t label`, and `parent_lambda` must be reduced to a scalar (`.item()` or explicit indexing) before the boolean test — do not rely on implicit 1-element ndarray truthiness.
instances: single-instance

### F11 — `traverse_upwards` implicitly narrows length-1 ndarrays to `intp_t` scalars
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586,593 — `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` assigns a numpy ndarray into `cdef cnp.intp_t parent`; `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']` assigns an ndarray into `cdef cnp.float64_t parent_eps`. Both rely on the mask returning exactly one row.
scenario: "malformed condensed_tree with a duplicate child row is passed in → Cython's length-1 coercion fails, raising a hard-to-diagnose `TypeError: only length-1 arrays can be converted to Python scalars` deep inside a Cython call."
contract: The mask must be resolved to a scalar explicitly (`[0]` indexing) before assigning into scalar Cython locals.
instances: single-instance

### F12 — `_compute_stability` returns a `dict` with `np.float64` keys where callers use integer keys
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:269-276 — `result_pre_dict = np.vstack((np.arange(smallest_cluster, np.max(parents) + 1), result)).T` (upcasts integer cluster IDs to `float64`); `return dict(result_pre_dict)`. Callers use integer keys (e.g., `_get_clusters` line 731 `stability[child]` with `child` from `cluster_tree['child']` which is `intp`).
scenario: "the dict is serialized/deserialized to a plain-Python dict (or a numpy scalar with different equality/hash semantics is substituted) → integer-keyed lookups silently miss because `1.0`-vs-`1` hash equivalence is not preserved."
contract: `_compute_stability` must return a dict whose keys are `intp`/`int` cluster IDs, not float; construct the mapping via `dict(zip(np.arange(...), result))` where the arange dtype is `np.intp`.
instances: single-instance

### F13 — `_more_tags` marks `allow_nan=True` when metric is a callable, but callables may not accept NaN
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:972-973 — `return {"allow_nan": self.metric != "precomputed"}`. For any non-precomputed metric — including user-supplied callables — the estimator advertises NaN acceptance. But NaN handling only works because `_get_finite_row_indices` filters non-finite rows before passing to the metric; if the metric is a callable that expects finite input, the tag alone does not guarantee correctness — this is fine, but the tag is asserted for all algorithms including `algorithm="kdtree"`/`"balltree"`, whose backing `NearestNeighbors` does not tolerate NaN in the query data.
scenario: "user relies on the advertised `allow_nan=True` and calls `HDBSCAN(metric='euclidean', algorithm='kdtree').fit(X_with_nan)` → NaN rows are filtered before reaching KDTree, so it works — but the underlying tag/contract makes no distinction, and `metric='precomputed'` with a sparse LIL matrix (line 738) never runs the NaN-filter path yet doesn't validate `force_all_finite=True` either." 
contract: The `allow_nan` tag should be conditional on the actual runtime input path — sparse precomputed matrices do not filter NaN, so the tag should not universally return True for non-precomputed metrics.
instances: single-instance
