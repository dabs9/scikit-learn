I'll perform the union merge. The code being reviewed is legitimate scikit-learn code, not malware — I am only merging existing findings without modifying or augmenting any code.

Now producing the merged findings document:

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

### F3 — `bfs_from_cluster_tree` uses `np.isin` on unsorted parents per BFS level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-294 — `while len(process_queue) > 0: result.extend(process_queue.tolist()); process_queue = children[np.isin(parents, process_queue)]`. `np.isin` over the entire `parents` array (size O(n)) is executed once per BFS depth level; combined with callers (`epsilon_search`, `_get_clusters`) invoking BFS per node/leaf, aggregate cost is O(n²·depth) or worse.
scenario: "Cluster tree with many nodes and non-trivial depth → repeated full-array `np.isin` scans dominate leaf/eom cluster selection"
contract: Build a `parent -> [child_indices]` adjacency map once at the top of `_get_clusters`/`epsilon_search` and iterate that map for BFS in O(reachable_nodes).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:279-296, sklearn/cluster/_hdbscan/_tree.pyx:292, sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]

### F4 — `epsilon_search` uses O(n²) `list not-in` membership
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:611-634 — `processed = list()` then `if leaf not in processed:` and `processed.append(sub_node)` inside the loop over leaves; membership test on a Python list is O(len(processed)), giving O(L²) behavior in the number of leaves plus subtree sizes.
scenario: "`cluster_selection_method='leaf'` (or EOM with `cluster_selection_epsilon>0`) on a tree with many leaves → epsilon search quadratic in the number of processed nodes"
contract: Use a `set()` for `processed` so containment/insertion are O(1).
instances: single-instance

### F5 — `traverse_upwards` recomputes full-array boolean masks per recursion
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:576-602 — each recursive call executes `cluster_tree['child'] == leaf` and `cluster_tree[cluster_tree['child'] == parent]['value']` (both O(n) scans of the full cluster_tree), then recurses on `parent`. A parent chain of length d gives O(d·n) work per starting leaf; called from `epsilon_search` once per unprocessed leaf.
scenario: "Deep condensed-tree parent chain plus many leaves under `cluster_selection_epsilon>0` → O(L·d·n) post-processing"
contract: Precompute `child -> (parent, value)` map once outside the recursion (or a `child -> row_index` lookup via `np.argsort(children)` + `np.searchsorted`), and index the map instead of masking the full array each call.
instances: single-instance

### F6 — `recurse_leaf_dfs` flattens with `sum([...], [])` (quadratic list concat)
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555-566 — `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`; Python's `sum` on lists is O(total_length²) due to repeated list copying. Additionally, at every recursion `cluster_tree[cluster_tree['parent'] == current_node]['child']` scans the whole tree.
scenario: "`cluster_selection_method='leaf'` on a large condensed tree → leaf enumeration becomes quadratic in the number of leaves, and each recursion pays a full-array scan on top"
contract: Replace the recursion with an iterative DFS using an explicit stack and `list.extend`/`list.append`; precompute a `parent -> children` adjacency map once for the tree walk.
instances: single-instance

### F7 — `_condense_tree` re-BFSes hierarchy from within its own BFS loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148-230 — the function first calls `bfs_from_hierarchy(hierarchy, root)` once to build `node_list` in linear time, then in the pruning branches (lines 200, 207, 216, 225) calls `bfs_from_hierarchy(hierarchy, left/right)` again per pruned subtree, walking the same descendants that will subsequently be marked in `ignore`. For skewed hierarchies where most children fall under `min_cluster_size` this is O(n²).
scenario: "Data producing a chain-like single-linkage tree with many small merges → condense step degrades to O(n²)"
contract: Reuse the already-computed traversal order (or memoize subtree membership) instead of re-invoking `bfs_from_hierarchy` for each pruned subtree.
instances: single-instance

### F8 — `_get_clusters` performs O(n²) boolean scans over cluster_tree per node
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-732 — `for node in node_list:` loop, each iteration computes `child_selection = (cluster_tree['parent'] == node)` a full-array boolean mask, then `np.sum([...])` over `cluster_tree['child'][child_selection]`; `node_list` has O(n) entries and `cluster_tree` has O(n) rows, giving O(n²) total work in the EOM path.
scenario: "Fit HDBSCAN on a dataset that yields a large condensed cluster tree (deep hierarchy of small merges) → EOM cluster selection becomes O(n²) even after MST construction is done, dominating post-processing time"
contract: Precompute a mapping `parent -> child_indices` once (e.g., using `np.argsort` on the parent column plus `np.searchsorted`, or a Python dict grouping), then look up children per node in O(k) instead of scanning the full array each time.
instances: single-instance

### F9 — `_weighted_cluster_center` recomputes full pairwise distance matrix per cluster (medoid path)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:907-920 — inside `for idx in range(n_clusters)`: `dist_mat = pairwise_distances(data, metric=self.metric, **self._metric_params)` where `data = X[mask]`. For each cluster the full O(|cluster|²) pairwise-distance matrix is computed sequentially without any `n_jobs` parallelism.
scenario: "User sets `store_centers='medoid'` (or `'both'`) on a fit with many clusters → total medoid computation is O(∑|cluster|²) done serially, ignoring `self.n_jobs`, and each iteration re-imports/re-dispatches through pairwise_distances even though the metric and params are identical, missing the opportunity to pass `n_jobs=self.n_jobs`."
contract: Pass `n_jobs=self.n_jobs` to the `pairwise_distances` call in `_weighted_cluster_center`, and short-circuit for clusters small enough that the medoid can be computed trivially.
instances: single-instance

### F10 — Dense mutual reachability graph iterates full n×n instead of exploiting symmetry
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:142-149 — the nested loop is `for i in range(n_samples): for j in range(n_samples): ... distance_matrix[i, j] = mutual_reachibility_distance`. The docstring at line 130 states "We assume that the distance matrix is symmetric." Both `[i,j]` and `[j,i]` receive the same computed `max(core[i], core[j], d[i,j])`, so every cell's value is computed twice.
scenario: "user calls HDBSCAN with `algorithm='brute'` / precomputed dense distance matrix → O(n²) cells are visited twice, doubling the hot-path cost and doubling memory-traffic vs. iterating only `j >= i` and mirroring."
contract: Iterate the upper triangle only (`for j in range(i, n_samples)`) and mirror the assignment to `distance_matrix[j, i]`; the loop must not do redundant work on a matrix its own docstring declares symmetric.
instances: single-instance

### F11 — `n_jobs` default is a hard-coded `4`, ignoring joblib parallel context and per-machine CPU count
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`; docstring at lines 486-490 states "`None` means 1 unless in a :obj:`joblib.parallel_backend` context. `-1` means using all processors." The default therefore contradicts the documented `None` semantics and hard-caps concurrency at 4 regardless of environment.
scenario: "user runs on a 32-core box or inside a `joblib.parallel_backend` → HDBSCAN silently uses 4 workers rather than respecting the documented default, wasting available cores and diverging from every other scikit-learn estimator's `n_jobs=None` convention."
contract: Set `n_jobs=None` as the default in `__init__`, matching the docstring and the sklearn-wide convention.
instances: single-instance

### F12 — `_compute_stability` iterates the condensed_tree twice via `for condensed_node in condensed_tree`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:254-267 — two consecutive `for condensed_node in condensed_tree:` loops iterate the structured numpy array element-by-element from Python, defeating vectorization. `births[condensed_node.child] = condensed_node.value` and `result[result_index] += (lambda_val - births[parent]) * cluster_size` could be vectorized with `births[children_array] = values_array` and `np.add.at(result, parent_array - smallest_cluster, (values_array - births[parent_array]) * cluster_sizes_array)`.
scenario: "Large condensed trees → stability computation runs in Python loop speed rather than numpy speed even though it is cpdef Cython"
contract: Replace the two element-wise loops with vectorized numpy assignments/`np.add.at` on the extracted structured columns.
instances: single-instance

### F13 — Duplicated `births = np.full(...)` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice in a row with the same arguments; the first allocation is immediately discarded, wasting O(n) time and memory allocation per call.
scenario: "Any HDBSCAN fit → one unused O(n) allocation per stability computation"
contract: Delete the duplicate line so `births` is allocated exactly once.
instances: single-instance

### F14 — `_dense_mutual_reachability_graph` runs a single-threaded O(n²) loop despite the TODO
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — `with nogil:` wraps a serial double loop over `n_samples × n_samples`; a comment `# TODO: Update w/ prange with thread count based on _openmp_effective_n_threads` acknowledges the parallelization is missing. The Python-level `n_jobs` parameter is threaded through `_hdbscan_brute` but has no effect here.
scenario: "Brute-mode HDBSCAN with `n_jobs>1` on a large dense distance matrix → user pays full serial O(n²) cost building mutual reachability while pairwise_distances step used multiple cores"
contract: Replace the serial `for i in range(n_samples)` with `prange` gated by `_openmp_effective_n_threads()`, matching the parallelism strategy used elsewhere in scikit-learn.
instances: single-instance

### F15 — `n_jobs` default of 4 is unusual and can silently overfetch cores [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` is the constructor default. Elsewhere in scikit-learn the convention is `n_jobs=None` (meaning 1 unless in a joblib parallel_backend context); the class docstring at lines 486-490 explicitly documents "None means 1 unless in a joblib.parallel_backend context" — contradicting the actual default of 4.
scenario: "User instantiates `HDBSCAN()` on an 8-core machine expecting the documented single-thread default → four parallel worker processes/threads spin up unexpectedly, contending with other work; docstring/default mismatch also violates common sklearn parameter conventions"
contract: Set the constructor default to `n_jobs=None` to match the documented behavior and the sklearn-wide convention.
instances: single-instance

### F16 — `remap_single_linkage_tree` iterates in Python instead of vectorizing
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370-392 — `for i, _ in enumerate(tree):` accesses `tree[i]["left_node"]`/`tree[i]["right_node"]` and writes them back one element at a time, followed by another Python `for i, outlier in enumerate(non_finite):` loop to build `outlier_tree`. Both loops could be vectorized: mask + `np.where` for the remap and pure numpy assignment for `outlier_tree`.
scenario: "Fit with `force_all_finite=False` and many non-finite rows → post-processing remap runs in Python-loop time on an array of size (n_finite - 1)"
contract: Rewrite both loops with vectorized numpy operations (boolean masking with `np.where` for the remap, arithmetic-progression assignment for `outlier_tree`).
instances: single-instance

### F17 — `_hdbscan_prims` requests `min_samples` neighbors then discards all but the k-th
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:342-343 — `neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True); core_distances = np.ascontiguousarray(neighbors_distances[:, -1])`. All but the last column of the returned `(n_samples, min_samples)` array is thrown away.
scenario: "Fit with large `min_samples` → NearestNeighbors returns an O(n × min_samples) distance array of which only the last column is used, wasting memory and copy bandwidth (`np.ascontiguousarray` allocates a fresh contiguous slice). For n=10⁶, min_samples=50 that is 400 MB of transient allocation to obtain 8 MB of core distances."
contract: Fetch only the k-th neighbor distance via `nbrs.kneighbors(X, min_samples)[0][:, -1]` combined with an explicit contiguous output buffer, or add a code path that avoids returning the full neighbor distance matrix when only the k-th column is required.
instances: single-instance

### F18 — `mst_from_data_matrix` initializes `current_sources` to 1 instead of 0 [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:160 — `current_sources = np.ones(n_samples, dtype=np.int64)`. `current_sources[j]` is read on line 179 as `next_node_source` when `mutual_reachability_distance > next_node_min_reach` but before any real source has been recorded for `j`, defaulting to sample 1.
scenario: "In early Prim iterations, for a candidate `j` whose `min_reachability[j]` was updated in an earlier step and whose branch on line 195-200 reads `next_node_source = current_sources[j]` = 1 (never actually updated on the first pass) → whenever `next_node_min_reach < new_reachability`, the MST edge source is recorded as node 1 rather than the true source, producing incorrect edges when node 1 was not in fact the source."
contract: Initialize `current_sources` to `np.zeros(n_samples, dtype=np.int64)` (matching the initial `current_node = 0`) or, better, only read `current_sources[j]` after verifying `min_reachability[j] < INFTY` (i.e., a real update has occurred).
instances: single-instance

### F19 — `_do_labelling` allocates `result` sized by `root_cluster` instead of `n_samples`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-481 — `root_cluster = np.min(parent_array); result = np.empty(root_cluster, dtype=np.intp)`. The value `root_cluster` is the smallest parent id, which happens to equal `n_samples` because `_condense_tree` starts labeling internal nodes at `n_samples + 1` and roots at `n_samples`. Using `root_cluster` as a length is implicit coupling and hides the actual size (`n_samples`), and requires materializing `parent_array` and taking its min just to compute the output length.
scenario: "someone changes the relabelling scheme in `_condense_tree` → `_do_labelling`'s output array silently mis-sizes, producing wrong-length `labels_`; performance-wise the code additionally scans `parent_array` for its min before doing any real work."
contract: Compute the result length as `n_samples` explicitly (e.g. from the hierarchy shape passed through) rather than via `np.min(parent_array)`, and document the invariant that `root_cluster == n_samples`.
instances: single-instance

### F20 — `bfs_from_hierarchy` is invoked repeatedly with the same subtree roots in `_condense_tree`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148,200,207,216,225 — `_condense_tree` calls `bfs_from_hierarchy(hierarchy, root)` once up front to obtain `node_list`, then inside the per-node loop calls `bfs_from_hierarchy(hierarchy, left)` and/or `bfs_from_hierarchy(hierarchy, right)` for each sub-min-cluster branch, each of which redoes a Python-level BFS that walks Python lists and allocates fresh `process_queue`/`next_queue` lists on every call.
scenario: "hierarchies where many merges have one branch smaller than `min_cluster_size` (typical for `min_cluster_size` > 2) → the condensation runs BFS from thousands of internal nodes, each walking a Python-list queue with per-iteration `[x - n_samples for x in process_queue if x >= n_samples]` comprehensions, making condensation super-linear in n."
contract: Replace `bfs_from_hierarchy` with a preallocated intp buffer + integer head/tail indices (or an iterative stack traversal in Cython) so each BFS visits O(subtree_size) with no Python list churn; ensure the top-level `node_list` traversal shares this scratch buffer.
instances: single-instance
