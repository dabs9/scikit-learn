I'll note the system reminder: this content is scikit-learn HDBSCAN code (not malware). I'm merging review findings, not modifying code. Now I'll produce the UNION merge.

Matching findings across draws:
- Draw A F1 (max_distance in _hdbscan_prims) and Draw B F5 (metric_params/max_distance smuggling) — related but distinct sites/root causes: A is prims-path forward, B is estimator-boundary smuggling. Both survive.
- Draw B F2 and Draw C F1 both cover `{-1,-2}` missing `-3` at hdbscan.py:895 — same defect. Keep B's version (more specific evidence about `_OUTLIER_ENCODING` and centroids/medoids allocation, with `_do_labelling` reference structure); merge instances (both single-instance).
- Draw C F2 (dbscan_clustering param validation) — distinct from Draw B F7 (dbscan_clustering shape mismatch). Both survive.

Ordering: high → medium → low.

### F1 — `_hdbscan_prims` forwards `max_distance` to `DistanceMetric.get_metric`, causing an unexpected-keyword failure
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:344 — `dist_metric = DistanceMetric.get_metric(metric, **metric_params)` is invoked with the raw user-supplied `metric_params` dict; that same dict is documented (hdbscan.py:120–122, 174–178 of `_reachability` docstring) to carry a `max_distance` key used by the sparse mutual-reachability path. Unlike `_hdbscan_brute` (line 243), the prims path never pops/filters `max_distance` before splatting.
scenario: "User follows the error message from `_brute_mst` and passes `metric_params={'max_distance': 10}` → later calls with a non-precomputed metric route through `_hdbscan_prims`, which forwards the key into `DistanceMetric.get_metric(...)` and dies with a `TypeError: __init__() got an unexpected keyword argument 'max_distance'`, leaking an internal-looking error from user-driven input rather than a validated boundary check."
contract: Whitelist/strip `metric_params` keys that are HDBSCAN-only sentinels (currently `max_distance`) before forwarding them into `DistanceMetric.get_metric` and `NearestNeighbors`, so misplaced-trust in caller-supplied kwargs cannot produce cryptic downstream TypeErrors from third-party metric constructors.
instances: single-instance

### F2 — `n_jobs=4` default silently spawns worker processes without user opt-in
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in `HDBSCAN.__init__`; the docstring at line 486–490 states "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors." — implying a `None` default per scikit-learn convention, but the actual default is a hard-coded `4`.
scenario: "A user (or a sandboxed/containerized service) instantiates `HDBSCAN()` expecting the documented default → the estimator silently spins up up to four parallel workers via `pairwise_distances`, exceeding the caller's implicit CPU/memory boundary and contradicting the documented behavior; in constrained environments (containers with `cpuset=1`, CI runners) this yields resource exhaustion the caller never authorized."
contract: Change the default to `n_jobs=None` to match the documented contract and the rest of scikit-learn's clustering estimators, so that spawning parallel workers requires explicit caller consent.
instances: single-instance

### F3 — `_get_finite_row_indices` misclassifies rows via `X.sum(axis=1)` producing NaN/Inf boundary confusion
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:721-728 — `reduced_X = X.sum(axis=1)`; `missing_index = list(np.isnan(reduced_X).nonzero()[0])`; `infinite_index = list(np.isinf(reduced_X).nonzero()[0])`. Detection of missing/infinite samples is driven by the row *sum*, not per-cell checks: a row containing both `+inf` and `-inf` yields `nan` from the sum and is (mis)labeled as `missing` (label -3), and a row whose finite values happen to overflow to `±inf` on summation is (mis)labeled as `infinite` (label -2). `_get_finite_row_indices` on line 406 uses the same fragile pattern `np.isfinite(matrix.sum(axis=1)).nonzero()` for the dense branch.
scenario: "User passes a feature array whose per-row values overflow when summed → rows containing only finite data are silently classified as infinite outliers and excluded from clustering, producing misleading labels that the caller trusts as reflecting the input data."
contract: Detect non-finite samples per-element (`np.any(np.isnan(X), axis=1)` and `np.any(np.isinf(X), axis=1)` for dense, equivalent sparse-safe iteration for sparse) rather than by row-summation.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:721, sklearn/cluster/_hdbscan/hdbscan.py:725, sklearn/cluster/_hdbscan/hdbscan.py:728, sklearn/cluster/_hdbscan/hdbscan.py:406]

### F4 — `_weighted_cluster_center` filters only `{-1, -2}` from labels, missing `-3` (missing-data label)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The class docstring (sklearn/cluster/_hdbscan/hdbscan.py:560-573) states "`n_clusters` only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded." Since `-3` is the `missing` outlier label defined in `_OUTLIER_ENCODING` (line 74), missing-data rows can leak into `n_clusters`, and the loop `for idx in range(n_clusters)` may attempt to access uninitialized `centroids_[idx]`/`medoids_[idx]` entries or, conversely, the last "cluster" index may correspond to no rows at all in `mask`.
scenario: "User fits HDBSCAN with `store_centers='centroid'` on data containing NaN → the returned `n_clusters` count includes a phantom slot for the -3 outlier label, and the corresponding row of `centroids_` receives `np.average` over an empty selection (RuntimeWarning: mean of empty slice) or is filled with garbage from `np.empty`."
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})`, matching the documented contract and the constants defined in `_OUTLIER_ENCODING`.
instances: single-instance

### F5 — `metric` callable is not validated against the `precomputed`/tree paths, letting user code silently run under an incompatible algorithm
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:626 — `"metric": [StrOptions(FAST_METRICS | {"precomputed"}), callable]`; combined with lines 772–783 the callable metric is only rejected for `kdtree`/`balltree`, never for `precomputed`. In `_hdbscan_brute` at line 238–240, when `metric="precomputed"` is not selected the callable is passed through `pairwise_distances`, but the branch at line 222 uses object identity `metric == "precomputed"` — a truthy callable passed while user believes they set `precomputed` would silently bypass the symmetry check.
scenario: "User accidentally passes a callable while intending `metric='precomputed'` (e.g., `HDBSCAN(metric=lambda X: X)`) → the symmetry / square-matrix guardrails on lines 223–234 are skipped and the callable is invoked as a pairwise-distance function on the user's precomputed matrix, silently producing wrong (or infinite-loop-inducing) distances instead of an early input-validation error."
contract: Explicitly reject callable `metric` when the intent is a precomputed matrix, and add a positive `metric == "precomputed"` guard in `_hdbscan_brute` that raises when a non-string callable slips through.
instances: single-instance

### F6 — Recursion in `recurse_leaf_dfs` on caller-controlled tree depth risks stack overflow
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555–566 — `recurse_leaf_dfs` recursively descends the condensed cluster tree with no depth cap: `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`. The tree depth is a function of the input data shape, which is caller-supplied.
scenario: "A malicious or pathological input dataset that produces a highly unbalanced condensed hierarchy (chain-of-clusters shape with depth ~ n_samples) → Python recursion limit is exceeded inside a `cpdef` Cython routine, raising `RecursionError` mid-fit and aborting the estimator with no graceful degradation."
contract: Reimplement `recurse_leaf_dfs` iteratively (explicit stack/deque) so tree-descent depth is bounded by heap memory, not by Python's recursion limit.
instances: single-instance

### F7 — `TreeUnionFind.find` uses unbounded recursion for path compression
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:352–356 — `cdef cnp.intp_t find(self, cnp.intp_t x): if self.data[x, 0] != x: self.data[x, 0] = self.find(self.data[x, 0]); ...` recurses through a union-find chain whose length is bounded by the number of samples.
scenario: "Adversarially crafted input producing a deep union chain (before path compression kicks in) → `find` recurses `O(n_samples)` deep, tripping Python/C-stack limits or CPython's recursion cap when invoked from `labelling_at_cut`/`_do_labelling`, terminating the fit with a stack overflow rather than a bounded error."
contract: Convert `TreeUnionFind.find` to an iterative two-pass path-compression implementation so union-find operations are safe on adversarial input sizes.
instances: single-instance

### F8 — `self._raw_data = X` retains a reference to caller-owned input, defeating the `copy=False` boundary
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:707 — `self._raw_data = X` stores the validated (but not copied) input array on the estimator instance whenever `metric != "precomputed"`, regardless of `self.copy`.
scenario: "User trains `HDBSCAN(copy=False)` on a large array and then mutates or frees the underlying buffer expecting scikit-learn to be done with it → the estimator silently keeps a live reference (via `_raw_data`), so mutations reflect into the estimator's stored state, and any later pickling of the estimator serializes the entire raw training set — an unexpected data-leak channel when the estimator is persisted or shipped between processes."
contract: Drop the `_raw_data` attribute after `fit` and reconstruct only what downstream methods actually need, so the estimator does not silently retain a live reference to caller-owned training data.
instances: single-instance

### F9 — `alpha=None` sentinel bypasses parameter validation and crashes with `TypeError` inside `_hdbscan_brute`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-241 — `_hdbscan_brute(..., alpha=None, ...)` accepts `alpha=None` as default and unconditionally executes `distance_matrix /= alpha` at line 241. `HDBSCAN.__init__` sets `self.alpha = 1.0` and `_parameter_constraints` requires `alpha` to be a positive `Real`, so the call site (line 767) always supplies a float — but the helper's signature advertises `alpha=None` as accepted while the docstring says "float, default=1.0". A caller invoking the private helper directly (as several tests import from `_hdbscan.hdbscan`) sees an unsafe silent contract.
scenario: "A downstream caller (or a future refactor) invokes `_hdbscan_brute(X)` without passing `alpha` → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=`, and the misleading default in the signature contradicts both the docstring and the estimator's actual validation."
contract: Set the private helper's default to the documented value (`alpha=1.0`) so the signature matches the docstring and the value cannot silently reach the in-place division as `None`.
instances: single-instance

### F10 — `cluster_selection_method` argument to `tree_to_labels` reaches `_get_clusters` without validation of untrusted string, and is not typed
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:56-77 — `cpdef tuple tree_to_labels(..., cluster_selection_method="eom", ...)` accepts an untyped Python object as `cluster_selection_method`, forwards it unchanged to `_get_clusters`, where at line 727/761 it is compared to the string literals `'eom'` and `'leaf'`. If neither branch matches (e.g., typo `"EOM"`, or a callable), the function silently falls through and returns `labels` computed with `is_cluster` unmodified from its initial value (`{cluster: True for cluster in node_list}`), producing an incoherent labelling with no error raised. The Python `HDBSCAN` class validates its own `cluster_selection_method` at the estimator boundary (line 641), but the `cpdef` function is publicly importable (see tests importing `_do_labelling`, `_condense_tree` from the compiled module) and offers no defense-in-depth.
scenario: "A caller uses `sklearn.cluster._hdbscan._tree.tree_to_labels` directly with `cluster_selection_method='Eom'` (case typo) → the function silently returns labels derived from an unfiltered `is_cluster` dict rather than raising, giving the caller silently wrong labels."
contract: Validate `cluster_selection_method` at the top of `_get_clusters` and raise `ValueError` on any value outside `{'eom', 'leaf'}`.
instances: single-instance

### F11 — `metric_params` is unpacked into `_hdbscan_brute` as `**kwargs` and silently consumes reserved key `max_distance`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:764-771 — `kwargs = dict(X=X, min_samples=..., alpha=..., metric=..., n_jobs=..., **self._metric_params)` then passed as `mst_func(**kwargs)`. In `_hdbscan_brute` (line 243) `max_distance = metric_params.get("max_distance", 0.0)` intentionally overloads `metric_params` with an HDBSCAN-specific control key. There is no validation that `metric_params` does not collide with reserved HDBSCAN keyword names (`X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy`, `algo`, `leaf_size`), and any user-supplied `metric_params={'X': ..., 'min_samples': ...}` will raise a confusing `TypeError: got multiple values for keyword argument` deep inside the private helper rather than at the estimator boundary. More importantly, `max_distance` — a documented user-facing tuning parameter — is smuggled through `metric_params` rather than exposed as a proper estimator parameter subject to `_parameter_constraints`.
scenario: "User discovers via source-reading that `metric_params={'max_distance': 5.0}` alters sparse-matrix behavior, and passes it → the value flows unvalidated (no type/range check), and simultaneously the value is forwarded to the underlying `pairwise_distances`/`DistanceMetric.get_metric` call as an unknown metric kwarg, potentially causing an error or being silently ignored depending on the metric backend."
contract: Extract `max_distance` from `metric_params` at the estimator boundary in `fit`, pass it explicitly to `_hdbscan_brute`, and only forward the residual dict to `pairwise_distances`; validate its type/range via `_parameter_constraints`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:243, sklearn/cluster/_hdbscan/hdbscan.py:764-771]

### F12 — `cpdef` Cython entry points accept caller-supplied index arrays without bounds validation while running with `boundscheck=False`
severity: low
evidence: sklearn/_build_utils/__init__.py:71-78 shows `compiler_directives = {..., "boundscheck": cython_enable_debug_directives, "wraparound": False, ...}` — production builds compile with bounds checking disabled. sklearn/cluster/_hdbscan/_tree.pyx exposes `cpdef` functions that indirectly index memoryviews from caller-supplied data: `tree_to_labels` (line 56) → `_condense_tree` (line 117) sizes `relabel = np.empty(root + 1, ...)` from `hierarchy.shape[0]` (line 145) then writes `relabel[left] = next_label` (line 187) where `left` is read from `hierarchy[node - n_samples].left_node` without validating that `left <= root`; `_do_labelling` (line 431) computes `root_cluster = np.min(parent_array)` (line 480) and allocates `result = np.empty(root_cluster, dtype=np.intp)` (line 481) then writes `result[n] = label` for `n in range(root_cluster)` (line 508) — but `union_find` is sized by `np.max(parent_array) + 1` and `cluster_label_map[cluster]` (line 494) is a dict lookup that may raise `KeyError` when caller-supplied `condensed_tree` violates the invariants. With `boundscheck=False`, a malformed caller-supplied `HIERARCHY_dtype`/`CONDENSED_dtype` array (e.g., a `left_node` value of `-1` or `> root`, or a `parent_array` with unexpected values) causes an out-of-bounds write to `relabel`/`deaths`/memoryviews — silent memory corruption, not `IndexError`.
scenario: "A user calls the publicly-importable `sklearn.cluster._hdbscan._tree._condense_tree` (as the tests import `_do_labelling`, `_condense_tree` at line 19-23 of the test file) with a hand-crafted `HIERARCHY_dtype` array containing an out-of-range `left_node` → `relabel[left] = next_label` writes past the end of the allocated buffer without raising, corrupting adjacent Python heap memory."
contract: Validate that `left`, `right` (from `HIERARCHY_t.left_node/right_node`) and all `parent`/`child` values from caller-supplied structured arrays fall within `[0, 2*hierarchy.shape[0]]` before use, and raise `ValueError` at the top of each `cpdef` entry point on violation.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:117, sklearn/cluster/_hdbscan/_tree.pyx:431, sklearn/cluster/_hdbscan/_tree.pyx:56, sklearn/cluster/_hdbscan/_tree.pyx:359]

### F13 — `dbscan_clustering` overwrites labels using stale `self.labels_` masks with no protection against shape mismatch
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:960-969 — `labels = labelling_at_cut(self._single_linkage_tree_, cut_distance, min_cluster_size)`; then `infinite_index = self.labels_ == _OUTLIER_ENCODING["infinite"]["label"]` and `labels[infinite_index] = ...`. `labelling_at_cut` returns an array of length `n_samples` derived from `_single_linkage_tree_.shape[0] + 1`, while `self.labels_` has length equal to the *original* raw data (`self._raw_data.shape[0]`). When `metric != "precomputed"` and non-finite rows were dropped during `fit`, `_single_linkage_tree_` has been remapped by `remap_single_linkage_tree` (line 834) to include outlier leaves, so the shapes happen to align — but there is no assertion of this invariant, and any future refactor that changes when/whether the remap runs will silently produce mismatched-shape boolean-indexing that either raises `IndexError` or, worse, silently indexes wrongly. Callers relying on `dbscan_clustering` immediately after loading a pickled model built by an older version could see arbitrary corruption.
scenario: "A downstream refactor changes the point at which `_single_linkage_tree_` is remapped, or a pickled model from a slightly different code path is loaded → `self.labels_ == label` produces a mask of length ≠ len(labels), causing either a shape-mismatch IndexError or silent misindexing when NumPy broadcasts."
contract: Assert `len(labels) == self.labels_.shape[0]` before applying the boolean masks in `dbscan_clustering`, or explicitly re-derive the outlier masks from `self._raw_data`/stored non-finite indices.
instances: single-instance

### F14 — `dbscan_clustering` accepts caller-supplied parameters without validation [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-962 — `dbscan_clustering(self, cut_distance, min_cluster_size=5)` performs no `_validate_params` / range check and passes the values straight into the Cython routine `labelling_at_cut(const HIERARCHY_t[::1] linkage, cnp.float64_t cut, cnp.intp_t min_cluster_size)` at sklearn/cluster/_hdbscan/_tree.pyx:359-362.
scenario: "Caller invokes `est.dbscan_clustering(cut_distance=-1.0, min_cluster_size=-5)` → Cython coerces the negative `min_cluster_size` into an intp, `cluster_size[cluster] < min_cluster_size` (line 419) never triggers, producing silently-invalid labels while the estimator API contract of validated, positive integer parameters is broken."
contract: Validate `cut_distance` and `min_cluster_size` against numeric-interval constraints (e.g. `cut_distance>=0`, `min_cluster_size>=1`) before dispatching to the Cython routine, matching the fit-time parameter validation performed by `_validate_params`.
instances: single-instance
