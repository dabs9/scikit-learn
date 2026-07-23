Now I have enough to write the closure findings. Let me finalize the output:

### F1 — rule of D:F5 recurs in `test_hdbscan_algorithms` — `pytest.raises(ValueError)` used with no `match=`, so validator-level errors can shadow the guard under test
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:168-170 — the branch `if metric not in ALGOS_TREES[algo].valid_metrics(): with pytest.raises(ValueError): hdb.fit(X)`. The intended guard is `hdbscan.py:772-783` (`f"{self.metric} is not a valid metric for a KDTree-based algorithm."`). But `_parameter_constraints["metric"]` at hdbscan.py:626 is `StrOptions(FAST_METRICS | {"precomputed"}), callable` — any metric in `_VALID_METRICS` that is not in `FAST_METRICS` (e.g. `"correlation"`, `"cosine"` when the current `FAST_METRICS` set doesn't include it, or `"jaccard"`/`"matching"`/etc.) raises `InvalidParameterError` (a subclass of `ValueError`) *before* the tree-vs-metric guard runs. Since `pytest.raises(ValueError)` has no `match=`, the outer test passes whether the tripped code path was the parameter validator or the intended tree-metric guard — mirrors the exact defect F5 flagged at test_hdbscan.py:282.
scenario: "regression removes the KDTree/BallTree metric guard in `HDBSCAN.fit` → `test_hdbscan_algorithms` still passes for many `metric` values because the parameter validator's `InvalidParameterError` (a `ValueError`) satisfies the raise-assertion instead"
contract: Add `match=r"is not a valid metric for a .*-based algorithm"` to the `pytest.raises(ValueError, ...)` at line 169 so the assertion actually pins the intended guard.
instances: single-instance

### F2 — rule of D:F14 recurs in `_hdbscan_prims` — `metric_params` including `max_distance` forwarded to `NearestNeighbors`/`DistanceMetric.get_metric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:332-344 — `_hdbscan_prims` receives `**metric_params` and forwards them via `NearestNeighbors(..., metric_params=metric_params, ...)` and `DistanceMetric.get_metric(metric, **metric_params)`. Neither call site strips the `max_distance` key documented for sparse precomputed inputs (see F14's discussion of `_hdbscan_brute`). Setting `HDBSCAN(algorithm='kdtree', metric_params={'max_distance': 5.0})` yields `DistanceMetric.get_metric("euclidean", max_distance=5.0)` → `TypeError` from Cython dispatch. F14's contract ("pop `max_distance` out of `metric_params` before forwarding") therefore must be applied in both `_hdbscan_brute` and `_hdbscan_prims`.
scenario: "user copies the sparse-precomputed `metric_params={'max_distance': ...}` idiom into a tree-based HDBSCAN call → TypeError from DistanceMetric.get_metric or NearestNeighbors metric constructor"
contract: In `_hdbscan_prims`, pop `max_distance` (and any other reserved runtime keys) out of `metric_params` before it reaches `NearestNeighbors` or `DistanceMetric.get_metric`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:337, sklearn/cluster/_hdbscan/hdbscan.py:344]

### F3 — rule of D:F16 recurs in `_brute_mst` — docstring documents a `min_samples` default that the signature does not provide
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82,95 — signature is `def _brute_mst(mutual_reachability, min_samples):` (no default), but the docstring states `min_samples : int, default=None`. Same defect pattern as F16 for `_hdbscan_brute`/`_hdbscan_prims`: the documented default contradicts the runtime signature, misleading anyone who calls the helper with defaults.
scenario: "internal caller invokes `_brute_mst(matrix)` as the docstring suggests → `TypeError: _brute_mst() missing 1 required positional argument: 'min_samples'`; the documented `default=None` is a phantom"
contract: Edit the docstring at `hdbscan.py:95` to remove the `default=None` clause so the parameter is documented as `min_samples : int` matching the signature; sync with F16's fix so all three helpers' documented defaults are consistent.
instances: single-instance

### F4 — rule of D:F20 recurs in `bfs_from_cluster_tree` — Python-object BFS queue in a Cython hot loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:279-296 — `bfs_from_cluster_tree` is declared `cdef list` and its main loop performs `np.isin(parents, process_queue)` plus `.tolist()` on Python `list` objects, which allocates new ndarrays and Python lists at each iteration. It is invoked from `epsilon_search` (line 632) and `_get_clusters` (line 737), both in the cluster-selection hot path. Same defect pattern as F20 for `bfs_from_hierarchy`: the "Cython lint cleanup" advertised in the PR description missed this sibling routine.
scenario: "user runs `HDBSCAN(cluster_selection_method='leaf', cluster_selection_epsilon=…).fit(X)` or `HDBSCAN(cluster_selection_method='eom').fit(X)` → the flat-clustering step allocates a new `np.isin` boolean array and a fresh Python list on every BFS layer, dominating post-condensation runtime"
contract: Replace `process_queue` with a preallocated `intp_t[::1]` buffer and front/back indices, and use a boolean membership array indexed by `parent` instead of `np.isin` at each iteration — mirroring F20's fix.
instances: single-instance

### F5 — rule of D:F32 recurs — "homogenous" typo in the plot example
severity: low
evidence: examples/cluster/plot_hdbscan.py:116 — "Traditional DBSCAN assumes that any potential clusters are homogenous in density." F32's evidence flagged the identical typo in `hdbscan.py:906` but its `instances` list did not include the example file. `homogenous` is not standard English (the correct term is `homogeneous`, as used throughout the rest of the scikit-learn tree — see `doc/modules/clustering.rst:978`).
scenario: "Sphinx renders the auto-generated HDBSCAN gallery example → the reader-facing narrative displays 'homogenous', an obvious misspelling directly beneath the algorithm's headline feature comparison with DBSCAN"
contract: Change `homogenous` to `homogeneous` at `examples/cluster/plot_hdbscan.py:116`, alongside F32's fix at `hdbscan.py:906`.
instances: single-instance

### F6 — rule of D:F32 recurs — "reachibility" typo pervasively in `_reachability.pyx` code + test filename
severity: low
evidence: F32 flagged `reahability` (missing `c`) in docstrings of `_linkage.pyx` and `hdbscan.py`, but a distinct misspelling — `reachibility` (missing `a`) — is used consistently in `_reachability.pyx` **as an identifier**: `mutual_reachibility_distance` at sklearn/cluster/_hdbscan/_reachability.pyx:127, 144, 149, 182, 206, 209, 210, and the test module filename itself is `sklearn/cluster/_hdbscan/tests/test_reachibility.py`. Because these are identifiers/paths (not free-text prose), fixing them requires renames rather than a docstring edit — a scope F32 did not address.
scenario: "reviewers grep for `mutual_reachability_distance` (the correct spelling used in prose everywhere else) → get zero hits in the actual computation code, wasting review cycles; the misspelled test module name will render as `test_reachibility` in any test-report or CI summary, cementing the typo in perpetuity"
contract: Rename the local variable `mutual_reachibility_distance` → `mutual_reachability_distance` at every occurrence in `sklearn/cluster/_hdbscan/_reachability.pyx`; rename the test file to `test_reachability.py` and update any imports/CI configs that reference it.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/tests/test_reachibility.py]

### F7 — rule of D:F40 recurs — additional docstrings claim `shape (n_samples,)` for variable-length edge-list / linkage arrays
severity: low
evidence: F40 flagged `_tree.pyx:138, 447, 656`. Two more docstrings in the same file assert the same wrong shape for arrays that are actually of length `n_samples - 1`:
  - sklearn/cluster/_hdbscan/_tree.pyx:129 — `_condense_tree`'s input `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` (actual shape is `(n_samples - 1,)` — see `_linkage.pyx:252` `single_linkage = np.zeros(n_samples - 1, ...)`).
  - sklearn/cluster/_hdbscan/_tree.pyx:371 — `labelling_at_cut`'s input `linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` (again `(n_samples - 1,)`; `root = 2 * linkage.shape[0]` on line 396 depends on this and would break if the shape were truly `(n_samples,)`).
Same defect pattern as F40: shape docstrings copied wholesale across all consumers without matching the actual array size.
scenario: "reviewer or downstream user reads `_condense_tree`/`labelling_at_cut` docstrings → expects `hierarchy`/`linkage` to have length `n_samples` and slices/indexes accordingly, silently corrupting the linkage tree or triggering `IndexError` on the last row"
contract: Correct the two docstrings to state `shape (n_samples - 1,), dtype=HIERARCHY_dtype`, mirroring the correct shape declared in `_linkage.pyx:74, 136, 227, 233`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:371]
