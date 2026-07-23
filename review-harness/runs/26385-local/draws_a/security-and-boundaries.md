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

### F3 — `metric` callable is not validated against the `precomputed`/tree paths, letting user code silently run under an incompatible algorithm
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:626 — `"metric": [StrOptions(FAST_METRICS | {"precomputed"}), callable]`; combined with lines 772–783 the callable metric is only rejected for `kdtree`/`balltree`, never for `precomputed`. In `_hdbscan_brute` at line 238–240, when `metric="precomputed"` is not selected the callable is passed through `pairwise_distances`, but the branch at line 222 uses object identity `metric == "precomputed"` — a truthy callable passed while user believes they set `precomputed` would silently bypass the symmetry check.
scenario: "User accidentally passes a callable while intending `metric='precomputed'` (e.g., `HDBSCAN(metric=lambda X: X)`) → the symmetry / square-matrix guardrails on lines 223–234 are skipped and the callable is invoked as a pairwise-distance function on the user's precomputed matrix, silently producing wrong (or infinite-loop-inducing) distances instead of an early input-validation error."
contract: Explicitly reject callable `metric` when the intent is a precomputed matrix, and add a positive `metric == "precomputed"` guard in `_hdbscan_brute` that raises when a non-string callable slips through.
instances: single-instance

### F4 — Recursion in `recurse_leaf_dfs` on caller-controlled tree depth risks stack overflow
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555–566 — `recurse_leaf_dfs` recursively descends the condensed cluster tree with no depth cap: `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`. The tree depth is a function of the input data shape, which is caller-supplied.
scenario: "A malicious or pathological input dataset that produces a highly unbalanced condensed hierarchy (chain-of-clusters shape with depth ~ n_samples) → Python recursion limit is exceeded inside a `cpdef` Cython routine, raising `RecursionError` mid-fit and aborting the estimator with no graceful degradation."
contract: Reimplement `recurse_leaf_dfs` iteratively (explicit stack/deque) so tree-descent depth is bounded by heap memory, not by Python's recursion limit.
instances: single-instance

### F5 — `TreeUnionFind.find` uses unbounded recursion for path compression
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:352–356 — `cdef cnp.intp_t find(self, cnp.intp_t x): if self.data[x, 0] != x: self.data[x, 0] = self.find(self.data[x, 0]); ...` recurses through a union-find chain whose length is bounded by the number of samples.
scenario: "Adversarially crafted input producing a deep union chain (before path compression kicks in) → `find` recurses `O(n_samples)` deep, tripping Python/C-stack limits or CPython's recursion cap when invoked from `labelling_at_cut`/`_do_labelling`, terminating the fit with a stack overflow rather than a bounded error."
contract: Convert `TreeUnionFind.find` to an iterative two-pass path-compression implementation so union-find operations are safe on adversarial input sizes.
instances: single-instance

### F6 — `self._raw_data = X` retains a reference to caller-owned input, defeating the `copy=False` boundary
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:707 — `self._raw_data = X` stores the validated (but not copied) input array on the estimator instance whenever `metric != "precomputed"`, regardless of `self.copy`.
scenario: "User trains `HDBSCAN(copy=False)` on a large array and then mutates or frees the underlying buffer expecting scikit-learn to be done with it → the estimator silently keeps a live reference (via `_raw_data`), so mutations reflect into the estimator's stored state, and any later pickling of the estimator serializes the entire raw training set — an unexpected data-leak channel when the estimator is persisted or shipped between processes."
contract: Drop the `_raw_data` attribute after `fit` and reconstruct only what downstream methods actually need, so the estimator does not silently retain a live reference to caller-owned training data.
instances: single-instance
