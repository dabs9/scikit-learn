### F1 — `mst_from_mutual_reachability` allocates and reslices arrays every iteration → O(n²) allocations
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:94-106 — inside the Prim's main loop, `label_filter = current_labels != current_node`, `current_labels = current_labels[label_filter]`, `left = min_reachability[label_filter]`, `right = mutual_reachability[current_node][current_labels]`, and `min_reachability = np.minimum(left, right)` are all re-executed via NumPy fancy indexing, allocating fresh arrays on every iteration.
scenario: "Dense brute MST on n samples → ~n heap allocations of arrays sized O(n), plus a full O(n) copy of mutual_reachability's row through fancy indexing (rather than a contiguous row view), inflating memory bandwidth and GC/allocator pressure quadratically instead of a nogil in-place scan."
contract: Perform Prim's expansion in-place with a Cython typed-memoryview scan that indexes an `in_tree` mask (see `mst_from_data_matrix` for the pattern) and updates `min_reachability`/`current_sources` without reallocating per iteration.
instances: single-instance

### F2 — `_get_finite_row_indices` densifies sparse matrix via `.tolil()` and a Python `for` loop
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 — `row_indices = np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])`. `matrix.tolil()` builds a full LIL representation and the comprehension iterates row-by-row in Python.
scenario: "User passes a large CSR precomputed distance matrix with `metric != 'precomputed'` but containing non-finite entries → the code converts CSR→LIL (an O(nnz) allocation of Python lists per row) and pays a Python-level per-row loop, in addition to the earlier `X.sum(axis=1)` on line 721 which already yields non-finite row markers."
contract: Compute finite-row indices for sparse input directly from the already-computed `reduced_X = X.sum(axis=1)` (i.e., `~(np.isnan(reduced_X) | np.isinf(reduced_X))`) rather than densifying via LIL.
instances: single-instance

### F3 — `_dense_mutual_reachability_graph` inner loop is single-threaded despite `nogil`
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — comment `# TODO: Update w/ prange with thread count based on _openmp_effective_n_threads` and the loop `for i in range(n_samples): for j in range(n_samples): ...` runs serially, though the operation is embarrassingly parallel and the docstring/plan advertises `n_jobs` support.
scenario: "User selects `algorithm='brute'` with `n_jobs>1` expecting the O(n²) mutual-reachability rewrite to use multiple cores → only the pairwise distance step honors `n_jobs`; the mutual-reachability rewrite remains single-threaded, silently ignoring `n_jobs` for the hottest O(n²) block."
contract: Replace the outer `for i in range(n_samples)` with `prange(n_samples, nogil=True, num_threads=_openmp_effective_n_threads())`.
instances: single-instance

### F4 — `bfs_from_cluster_tree` runs `np.isin` over the entire condensed tree per BFS level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-294 — `while len(process_queue) > 0: result.extend(process_queue.tolist()); process_queue = children[np.isin(parents, process_queue)]`. Each iteration scans all `parents` (length = size of condensed tree) against all queued nodes.
scenario: "Deep condensed trees (e.g., long thin cluster hierarchies) → BFS runs in O(depth × n_condensed × |queue|) using `np.isin`, quadratic in tree size rather than the O(n_condensed) achievable with a parent→children adjacency list built once and reused across BFS calls (called repeatedly from `_get_clusters` and `epsilon_search`)."
contract: Build a parent→children index once (e.g., a dict or CSR of the condensed tree) and reuse it across all BFS invocations from `_get_clusters`/`epsilon_search`, replacing the per-level `np.isin` scan.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:292, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]

### F5 — `epsilon_search` uses Python `list` membership tests → O(k²) per invocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:611-634 — `processed = list()` and later `if leaf not in processed:` followed by `processed.append(sub_node)`. `list.__contains__` is O(k).
scenario: "Trees with many leaves flowing through the epsilon-cluster promotion path → linear scans of the growing `processed` list make cluster selection O(k²) where k is the number of processed sub-nodes, when a `set` would be O(1) per test."
contract: Use `processed = set()` and `processed.add(sub_node)`.
instances: single-instance

### F6 — `_condense_tree` calls `bfs_from_hierarchy` recursively for every pruned subtree → O(n²) worst case
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:200-230 — for each internal node whose children fall below `min_cluster_size`, the code calls `bfs_from_hierarchy(hierarchy, left)` (and/or `right`), and each such call itself walks the entire subtree, then marks `ignore[sub_node] = True`. On line 163 the outer loop already iterates over `node_list = bfs_from_hierarchy(hierarchy, root)` once, so the per-node sub-BFS repeats work.
scenario: "Hierarchies where many internal nodes are pruned (large `min_cluster_size` or highly imbalanced trees) → total BFS work becomes O(n²) because the same descendants are traversed inside every ancestor's `bfs_from_hierarchy` call, defeating the `ignore` mask (which prevents processing at the outer loop level but not the inner BFS)."
contract: Compute each node's descendant list once (bottom-up cache) or short-circuit the inner `bfs_from_hierarchy` on the `ignore` mask so already-marked subtrees are not re-walked.
instances: single-instance

### F7 — `_compute_stability` allocates `births` twice back-to-back
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` immediately followed by `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` (duplicated allocation of the same array; the first is dead).
scenario: "Every HDBSCAN fit executes `_compute_stability` → pays a redundant O(largest_child) allocation and fill that is immediately overwritten by an identical call, wasting a hot-path allocation on every fit."
contract: Delete the duplicate `births = np.full(...)` on line 252.
instances: single-instance

### F8 — `recurse_leaf_dfs` uses `sum([...], [])` list concatenation → O(k²) building leaves
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:566 — `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`. Repeated `list + list` via `sum` is O(k²) in total leaf count.
scenario: "Cluster trees with many leaves (leaf selection method or deep hierarchies) → building the leaves list scales quadratically with tree size instead of linearly, and the Python-level recursion also re-runs `cluster_tree['parent'] == current_node` (line 562) per node, an O(n_condensed) mask scan per recursive call."
contract: Replace the `sum([...], [])` fold with `list(itertools.chain.from_iterable(...))` or an iterative DFS with `list.extend`, and precompute a parent→children adjacency map to avoid rescanning the condensed tree at every node.
instances: single-instance

### F9 — `traverse_upwards` re-scans the condensed tree with boolean masks on every recursion
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586-602 — each recursive call executes `cluster_tree[cluster_tree['child'] == leaf]['parent']` and `cluster_tree[cluster_tree['child'] == parent]['value']`, i.e., two O(n_condensed) linear scans per level.
scenario: "Deep condensed trees + non-zero `cluster_selection_epsilon` → walking each leaf up to its epsilon-cluster costs O(depth × n_condensed) instead of O(depth) with a precomputed child→(parent, value) map, potentially dominating fit time for large trees."
contract: Precompute `child_to_parent` and `child_to_value` arrays/dicts once in `_get_clusters`/`epsilon_search` and pass them into `traverse_upwards` for O(1) lookups.
instances: single-instance

### F10 — `_hdbscan_prims` requests `min_samples` neighbors then discards all but the k-th
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:342-343 — `neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True); core_distances = np.ascontiguousarray(neighbors_distances[:, -1])`. All but the last column of the returned `(n_samples, min_samples)` array is thrown away.
scenario: "Fit with large `min_samples` → NearestNeighbors returns an O(n × min_samples) distance array of which only the last column is used, wasting memory and copy bandwidth (`np.ascontiguousarray` allocates a fresh contiguous slice). For n=10⁶, min_samples=50 that is 400 MB of transient allocation to obtain 8 MB of core distances."
contract: Fetch only the k-th neighbor distance via `nbrs.kneighbors(X, min_samples)[0][:, -1]` combined with an explicit contiguous output buffer, or add a code path that avoids returning the full neighbor distance matrix when only the k-th column is required.
instances: single-instance

### F11 — `_weighted_cluster_center` recomputes full pairwise distance matrix per cluster (medoid path)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:907-920 — inside `for idx in range(n_clusters)`: `dist_mat = pairwise_distances(data, metric=self.metric, **self._metric_params)` where `data = X[mask]`. For each cluster the full O(|cluster|²) pairwise-distance matrix is computed sequentially without any `n_jobs` parallelism.
scenario: "User sets `store_centers='medoid'` (or `'both'`) on a fit with many clusters → total medoid computation is O(∑|cluster|²) done serially, ignoring `self.n_jobs`, and each iteration re-imports/re-dispatches through pairwise_distances even though the metric and params are identical, missing the opportunity to pass `n_jobs=self.n_jobs`."
contract: Pass `n_jobs=self.n_jobs` to the `pairwise_distances` call in `_weighted_cluster_center`, and short-circuit for clusters small enough that the medoid can be computed trivially.
instances: single-instance

### F12 — `remap_single_linkage_tree` iterates the tree in pure Python per-row
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370-381 — Python `for i, _ in enumerate(tree):` with per-row branching on `left < finite_count`/`right < finite_count` and per-row dict lookups `internal_to_raw[left]`.
scenario: "Non-finite data path (any NaN/Inf in X) → this remap runs at Python speed over an O(n) structured array with dict lookups per row, when it could be vectorized by materializing `internal_to_raw` as an ndarray and using boolean masks on `tree['left_node']`/`tree['right_node']`."
contract: Convert `internal_to_raw` to a `np.ndarray` (`remap = np.asarray([internal_to_raw[i] for i in range(finite_count)])`) and vectorize the remap with `np.where`/boolean indexing over the structured `tree` fields.
instances: single-instance

### F13 — `mst_from_data_matrix` initializes `current_sources` to 1 instead of 0 [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:160 — `current_sources = np.ones(n_samples, dtype=np.int64)`. `current_sources[j]` is read on line 179 as `next_node_source` when `mutual_reachability_distance > next_node_min_reach` but before any real source has been recorded for `j`, defaulting to sample 1.
scenario: "In early Prim iterations, for a candidate `j` whose `min_reachability[j]` was updated in an earlier step and whose branch on line 195-200 reads `next_node_source = current_sources[j]` = 1 (never actually updated on the first pass) → whenever `next_node_min_reach < new_reachability`, the MST edge source is recorded as node 1 rather than the true source, producing incorrect edges when node 1 was not in fact the source."
contract: Initialize `current_sources` to `np.zeros(n_samples, dtype=np.int64)` (matching the initial `current_node = 0`) or, better, only read `current_sources[j]` after verifying `min_reachability[j] < INFTY` (i.e., a real update has occurred).
instances: single-instance
