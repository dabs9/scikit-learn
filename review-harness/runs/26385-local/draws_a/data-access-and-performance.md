Not malware — legitimate scikit-learn code.

Compiling my findings:

### F1 — `_get_clusters` performs O(n²) boolean scans over cluster_tree per node
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-732 — `for node in node_list:` loop, each iteration computes `child_selection = (cluster_tree['parent'] == node)` a full-array boolean mask, then `np.sum([...])` over `cluster_tree['child'][child_selection]`; `node_list` has O(n) entries and `cluster_tree` has O(n) rows, giving O(n²) total work in the EOM path.
scenario: "Fit HDBSCAN on a dataset that yields a large condensed cluster tree (deep hierarchy of small merges) → EOM cluster selection becomes O(n²) even after MST construction is done, dominating post-processing time"
contract: Precompute a mapping `parent -> child_indices` once (e.g., using `np.argsort` on the parent column plus `np.searchsorted`, or a Python dict grouping), then look up children per node in O(k) instead of scanning the full array each time.
instances: single-instance

### F2 — `bfs_from_cluster_tree` uses `np.isin` on unsorted parents per BFS level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-294 — `while len(process_queue) > 0: result.extend(process_queue.tolist()); process_queue = children[np.isin(parents, process_queue)]`. `np.isin` over the entire `parents` array (size O(n)) is executed once per BFS depth level; combined with callers (`epsilon_search`, `_get_clusters`) invoking BFS per node/leaf, aggregate cost is O(n²·depth) or worse.
scenario: "Cluster tree with many nodes and non-trivial depth → repeated full-array `np.isin` scans dominate leaf/eom cluster selection"
contract: Build a `parent -> [child_indices]` adjacency map once at the top of `_get_clusters`/`epsilon_search` and iterate that map for BFS in O(reachable_nodes).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:279-296, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]

### F3 — `epsilon_search` uses O(n²) `list not-in` membership
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:611-634 — `processed = list()` then `if leaf not in processed:` and `processed.append(sub_node)` inside the loop over leaves; membership test on a Python list is O(len(processed)), giving O(L²) behavior in the number of leaves plus subtree sizes.
scenario: "`cluster_selection_method='leaf'` (or EOM with `cluster_selection_epsilon>0`) on a tree with many leaves → epsilon search quadratic in the number of processed nodes"
contract: Use a `set()` for `processed` so containment/insertion are O(1).
instances: single-instance

### F4 — `traverse_upwards` recomputes full-array boolean masks per recursion
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:576-602 — each recursive call executes `cluster_tree['child'] == leaf` and `cluster_tree[cluster_tree['child'] == parent]['value']` (both O(n) scans of the full cluster_tree), then recurses on `parent`. A parent chain of length d gives O(d·n) work per starting leaf; called from `epsilon_search` once per unprocessed leaf.
scenario: "Deep condensed-tree parent chain plus many leaves under `cluster_selection_epsilon>0` → O(L·d·n) post-processing"
contract: Precompute `child -> (parent, value)` map once outside the recursion (or a `child -> row_index` lookup via `np.argsort(children)` + `np.searchsorted`), and index the map instead of masking the full array each call.
instances: single-instance

### F5 — `recurse_leaf_dfs` flattens with `sum([...], [])` (quadratic list concat)
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555-566 — `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`; Python's `sum` on lists is O(total_length²) due to repeated list copying. Additionally, at every recursion `cluster_tree[cluster_tree['parent'] == current_node]['child']` scans the whole tree.
scenario: "`cluster_selection_method='leaf'` on a large condensed tree → leaf enumeration becomes quadratic in the number of leaves, and each recursion pays a full-array scan on top"
contract: Replace the recursion with an iterative DFS using an explicit stack and `list.extend`/`list.append`; precompute a `parent -> children` adjacency map once for the tree walk.
instances: single-instance

### F6 — `_condense_tree` re-BFSes hierarchy from within its own BFS loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148-230 — the function first calls `bfs_from_hierarchy(hierarchy, root)` once to build `node_list` in linear time, then in the pruning branches (lines 200, 207, 216, 225) calls `bfs_from_hierarchy(hierarchy, left/right)` again per pruned subtree, walking the same descendants that will subsequently be marked in `ignore`. For skewed hierarchies where most children fall under `min_cluster_size` this is O(n²).
scenario: "Data producing a chain-like single-linkage tree with many small merges → condense step degrades to O(n²)"
contract: Reuse the already-computed traversal order (or memoize subtree membership) instead of re-invoking `bfs_from_hierarchy` for each pruned subtree.
instances: single-instance

### F7 — `mst_from_mutual_reachability` allocates full-size temporary arrays each iteration
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:94-106 — inside the Prim loop of `n_samples - 1` iterations, each iteration executes `label_filter = current_labels != current_node`, `current_labels = current_labels[label_filter]`, `left = min_reachability[label_filter]`, `right = mutual_reachability[current_node][current_labels]`, `min_reachability = np.minimum(left, right)`, `np.argmin(min_reachability)`. Each of these allocates a fresh numpy array of size O(n − i). This is chatty allocation of O(n²) memory traffic and defeats the `cdef` typing; the comparable dense Prim in `mst_from_data_matrix` runs entirely in nogil in-place.
scenario: "Dense brute-mode HDBSCAN on moderately large n → MST-from-mutual-reachability spends most time allocating/freeing NumPy arrays rather than doing math"
contract: Convert this loop to nogil Cython operating in-place on typed memoryviews (analogous to `mst_from_data_matrix`), avoiding per-iteration numpy fancy-indexing allocations.
instances: single-instance

### F8 — `_dense_mutual_reachability_graph` runs a single-threaded O(n²) loop despite the TODO
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — `with nogil:` wraps a serial double loop over `n_samples × n_samples`; a comment `# TODO: Update w/ prange with thread count based on _openmp_effective_n_threads` acknowledges the parallelization is missing. The Python-level `n_jobs` parameter is threaded through `_hdbscan_brute` but has no effect here.
scenario: "Brute-mode HDBSCAN with `n_jobs>1` on a large dense distance matrix → user pays full serial O(n²) cost building mutual reachability while pairwise_distances step used multiple cores"
contract: Replace the serial `for i in range(n_samples)` with `prange` gated by `_openmp_effective_n_threads()`, matching the parallelism strategy used elsewhere in scikit-learn.
instances: single-instance

### F9 — `_get_finite_row_indices` materializes sparse matrix as LIL and iterates in Python
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 — `row_indices = np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])`. Converting CSR→LIL forces a full copy into a Python list-of-lists, then a Python-level list comprehension iterates rows; both are O(n²) memory-and-time worst case and lose vectorization.
scenario: "Sparse input with non-finite entries and moderate n_samples → finite-row detection becomes the dominant cost, and LIL materialization can OOM on very large matrices"
contract: Use `np.add.reduceat` on `matrix.data` plus `np.isfinite` (or `np.diff(indptr)` with `np.isfinite(matrix.data)` and `np.logical_and.reduceat`) to compute per-row finiteness in vectorized C code without ever materializing LIL.
instances: single-instance

### F10 — `_compute_stability` iterates the condensed_tree twice via `for condensed_node in condensed_tree`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:254-267 — two consecutive `for condensed_node in condensed_tree:` loops iterate the structured numpy array element-by-element from Python, defeating vectorization. `births[condensed_node.child] = condensed_node.value` and `result[result_index] += (lambda_val - births[parent]) * cluster_size` could be vectorized with `births[children_array] = values_array` and `np.add.at(result, parent_array - smallest_cluster, (values_array - births[parent_array]) * cluster_sizes_array)`.
scenario: "Large condensed trees → stability computation runs in Python loop speed rather than numpy speed even though it is cpdef Cython"
contract: Replace the two element-wise loops with vectorized numpy assignments/`np.add.at` on the extracted structured columns.
instances: single-instance

### F11 — Duplicated `births = np.full(...)` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice in a row with the same arguments; the first allocation is immediately discarded, wasting O(n) time and memory allocation per call.
scenario: "Any HDBSCAN fit → one unused O(n) allocation per stability computation"
contract: Delete the duplicate line so `births` is allocated exactly once.
instances: single-instance

### F12 — `_weighted_cluster_center` recomputes full pairwise distances per cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:907-920 — for each of `n_clusters`, `dist_mat = pairwise_distances(data, metric=self.metric, **self._metric_params)` is invoked. When `store_centers="medoid"` or `"both"`, this is O(k · c_i²) with no reuse of any distance already computed by the brute path (which produced a full pairwise matrix). A user-facing TODO acknowledges the inefficient backend.
scenario: "`store_centers` in {'medoid','both'} with brute algorithm on a dataset where pairwise distances were already computed → distances are recomputed cluster-by-cluster instead of subselecting the already-materialized distance matrix"
contract: When `algorithm='brute'` (and the full pairwise matrix is available on the estimator), subselect it via `distance_matrix[np.ix_(cluster_indices, cluster_indices)]` instead of calling `pairwise_distances` per cluster.
instances: single-instance

### F13 — `n_jobs` default of 4 is unusual and can silently overfetch cores [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` is the constructor default. Elsewhere in scikit-learn the convention is `n_jobs=None` (meaning 1 unless in a joblib parallel_backend context); the class docstring at lines 486-490 explicitly documents "None means 1 unless in a joblib.parallel_backend context" — contradicting the actual default of 4.
scenario: "User instantiates `HDBSCAN()` on an 8-core machine expecting the documented single-thread default → four parallel worker processes/threads spin up unexpectedly, contending with other work; docstring/default mismatch also violates common sklearn parameter conventions"
contract: Set the constructor default to `n_jobs=None` to match the documented behavior and the sklearn-wide convention.
instances: single-instance

### F14 — `remap_single_linkage_tree` iterates in Python instead of vectorizing
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370-392 — `for i, _ in enumerate(tree):` accesses `tree[i]["left_node"]`/`tree[i]["right_node"]` and writes them back one element at a time, followed by another Python `for i, outlier in enumerate(non_finite):` loop to build `outlier_tree`. Both loops could be vectorized: mask + `np.where` for the remap and pure numpy assignment for `outlier_tree`.
scenario: "Fit with `force_all_finite=False` and many non-finite rows → post-processing remap runs in Python-loop time on an array of size (n_finite - 1)"
contract: Rewrite both loops with vectorized numpy operations (boolean masking with `np.where` for the remap, arithmetic-progression assignment for `outlier_tree`).
instances: single-instance
