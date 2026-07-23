### F1 — Dense mutual-reachability walks the full n×n square despite symmetric input
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — inner nogil loop iterates `for i in range(n_samples): for j in range(n_samples):` and writes `distance_matrix[i, j] = mutual_reachibility_distance` for every pair, even though the block comment at :130-132 explicitly states "We assume that the distance matrix is symmetric." The adjacent TODO ("Update w/ prange with thread count based on `_openmp_effective_n_threads`") is likewise unresolved, so the loop is also single-threaded regardless of the user-supplied `n_jobs`.
scenario: "`HDBSCAN(algorithm=\"brute\").fit(X)` with dense pairwise input → the O(n²) mutual-reachability computation performs 2× the necessary `max()` operations (each symmetric pair `(i,j)`/`(j,i)` recomputed and rewritten), all on a single core; wall-clock on large dense fits is roughly double what a triangle-only or `prange` implementation would deliver."
contract: Iterate only the upper triangle (`for j in range(i, n_samples)`) and mirror the result to `distance_matrix[j, i]`, and dispatch the outer loop through `prange` gated by `_openmp_effective_n_threads(n_jobs)`.
instances: single-instance

### F2 — `_get_clusters` EOM loop is quadratic in condensed-tree size
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — inside `for node in node_list:` (which iterates every cluster in the stability dict), each iteration performs `child_selection = (cluster_tree['parent'] == node)` — a full linear scan over the parent array — followed by `np.sum([stability[child] for child in cluster_tree['child'][child_selection]])`, a Python list comprehension boxed into `np.sum`. Nothing precomputes a `parent → children` adjacency. The same pattern recurs at :562, :586, :593, :725, :729.
scenario: "`HDBSCAN(...).fit(X)` on data with many clusters → EOM cluster selection is O(|node_list| × |cluster_tree|), which for a fanned-out condensed tree becomes quadratic in tree size with a Python-level constant factor per outer iteration."
contract: Precompute a `parent → children` mapping (e.g. sort cluster_tree by parent once and build an index array, or a Python `defaultdict(list)`) before the EOM loop and index into it per node; likewise reuse it for `traverse_upwards`, `recurse_leaf_dfs`, and the `cluster_sizes` root sum.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:562, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593, sklearn/cluster/_hdbscan/_tree.pyx:725, sklearn/cluster/_hdbscan/_tree.pyx:729]

### F3 — `mst_from_mutual_reachability` allocates fresh numpy arrays inside its O(n²) hot loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:94-108 — every one of `n_samples - 1` iterations creates: `label_filter = current_labels != current_node`, `current_labels[label_filter]` (fancy-index copy), `min_reachability[label_filter]` (fancy-index copy), `mutual_reachability[current_node][current_labels]` (integer-index copy), and `np.minimum(left, right)` (new allocation), plus `np.argmin(...)`. All are Python-orchestrated with the GIL held. The docstring at :79-80 concedes ndarrays are chosen "to make use of numpy binary indexing and sub-selection", but there is no `nogil`/Cython buffer path.
scenario: "`HDBSCAN(algorithm=\"brute\").fit(dense_X)` with large n → Prim's MST performs ≥5 heap allocations per outer iteration for a total of O(n) short-lived ndarrays and O(n²) copied elements, GIL held throughout; the sister routine `mst_from_data_matrix` (:141-220) is fully nogil for comparison."
contract: Rewrite the loop as a `nogil` Cython pass using pre-allocated typed memoryviews for `in_tree`, `min_reachability`, and per-iteration argmin, matching `mst_from_data_matrix`'s pattern; drop the ndarray-based fancy-index chain.
instances: single-instance

### F4 — `HDBSCAN.__init__` defaults `n_jobs=4`, contradicting its own docstring and scikit-learn convention [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:647-658 — the constructor signature reads `n_jobs=4`, while the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None` with "`None` means 1 unless in a :obj:`joblib.parallel_backend` context." The value flows straight into `pairwise_distances(..., n_jobs=n_jobs, ...)` at line 239 and `NearestNeighbors(..., n_jobs=n_jobs, ...)` at line 338.
scenario: "User instantiates `HDBSCAN()` on a small input → 4 worker threads/processes are silently spun up in `pairwise_distances`/`NearestNeighbors` for every fit, causing thread-startup overhead to dominate on small problems and producing behavior the documented default (`None`, meaning 1) forbids."
contract: Change the constructor default to `n_jobs=None` so runtime behavior matches the documented default and the standard scikit-learn convention for `n_jobs`.
instances: single-instance

### F5 — `bfs_from_cluster_tree` scans the entire condensed tree on every BFS layer
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:279-296 — `while len(process_queue) > 0: … process_queue = children[np.isin(parents, process_queue)]`. Each iteration performs `np.isin` over the whole `parents` array (length = |condensed_tree|).
scenario: "EOM cluster selection with many clusters → `_get_clusters` calls `bfs_from_cluster_tree(cluster_tree, node)` inside the `for node in node_list` loop (line 737), so every non-selected cluster triggers a fresh full-array `np.isin` sweep, giving overall O(|node_list| · depth · |cluster_tree|) hot-path work on the condensed tree."
contract: Precompute a `parent → list-of-children` adjacency (single O(|cluster_tree|) pass), then walk that adjacency in `bfs_from_cluster_tree` instead of `np.isin` scans.
instances: single-instance

### F6 — `_do_labelling` recomputes a loop-invariant argmax per sample
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-506 — in the loop `for n in range(root_cluster): … elif len(clusters) == 1 and allow_single_cluster: … threshold = lambda_array[parent_array == cluster].max()`. In this branch `cluster == root_cluster` (the `if cluster != root_cluster` failed), so `parent_array == cluster` is identical for every `n`, yet a full-array boolean scan + `.max()` is executed inside the per-sample loop.
scenario: "`allow_single_cluster=True` on a dataset that yields a single detected cluster → threshold recomputed n_samples times → O(n_samples · |condensed_tree|) instead of a single hoisted O(|condensed_tree|) computation."
contract: Compute `threshold = lambda_array[parent_array == root_cluster].max()` once before the loop when `cluster_selection_epsilon == 0.0`, and reuse it.
instances: single-instance

### F7 — `recurse_leaf_dfs` concatenates children lists with `sum([...], [])`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555-566 — `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])` recursively flattens with `sum(list_of_lists, [])`, which is O(k²) in the total number of leaves due to repeated list rebuilding on each `+`.
scenario: "Deep or wide condensed cluster trees on `cluster_selection_method=\"leaf\"` (or EOM followed by `get_cluster_tree_leaves`) → leaf DFS on n-leaf trees costs O(n²) instead of O(n)."
contract: Accumulate results into a single mutable list (e.g., `result = []; ... result.extend(recurse_leaf_dfs(...))`) or convert to an explicit stack-based DFS.
instances: single-instance

### F8 — `epsilon_search` performs O(|processed|) membership tests via a list
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:611-634 — `processed` is declared `list processed = list()` and later probed with `if leaf not in processed:` (line 623) and appended via `processed.append(sub_node)` (line 634). `in` on a Python list is O(|processed|).
scenario: "Cluster-selection with a non-trivial `cluster_selection_epsilon` on trees with many leaves/sub-nodes → total cost of the guard is O(leaves²) instead of O(leaves)."
contract: Change `processed` to a `set` (initialize `processed = set()`, use `.add(sub_node)`).
instances: single-instance

### F9 — `traverse_upwards` scans full cluster tree per recursion step
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:576-602 — each recursive call executes `cluster_tree[cluster_tree['child'] == leaf]['parent']` (line 586) and `cluster_tree[cluster_tree['child'] == parent]['value']` (line 593), scanning the whole condensed tree twice per upward step.
scenario: "Called from `epsilon_search` (line 624) for every low-epsilon leaf → total cost O(leaves · depth · |cluster_tree|); on tall cluster hierarchies this dominates cluster selection."
contract: Build a `child → (parent, value)` dict once at entry to `_get_clusters` and look up parents/values in O(1) inside `traverse_upwards`.
instances: single-instance

### F10 — `_compute_stability` allocates and populates `births` twice
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-256 — two consecutive statements: `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252, then again `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 254 with no intervening use. The first allocation is dead.
scenario: "Every call to `_compute_stability` (once per `fit`) → wastes an O(largest_child) allocation + fill; small in absolute terms but a straight duplicate."
contract: Delete line 252 so `births` is allocated exactly once before the population loop.
instances: single-instance

### F11 — `_get_finite_row_indices` densifies sparse input via `.tolil()` and iterates rows in Python
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 — for sparse input the function does `[i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]`, which allocates a LIL representation (list-of-lists per row) and then loops in Python with a per-row `np.isfinite` call.
scenario: "Sparse feature matrix with any non-finite entry (triggers `all_finite=False`) → the finite-row filter walks the sparse matrix row-by-row in the Python interpreter with a full LIL conversion, scaling as O(n_rows) Python iterations plus an intermediate LIL copy of the entire matrix, when a single vectorized `np.isfinite(matrix.data)` combined with `np.add.reduceat(..., indptr[:-1])` (or equivalent on CSR structure) would give the same result in a single C-level pass."
contract: On the sparse branch, work directly on the CSR `data`/`indptr` arrays with a vectorised `np.isfinite`+`reduceat` (or `matrix.getnnz(axis=1)` + row-wise all-finite test) rather than converting to LIL and iterating rows in Python.
instances: single-instance

### F12 — Medoid computation calls `pairwise_distances` per cluster, ignoring `n_jobs`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:907-920 — inside `for idx in range(n_clusters):` the medoid branch invokes `pairwise_distances(data, metric=self.metric, **self._metric_params)` with no `n_jobs` argument (so joblib parallelism defaults to 1), and `mask = self.labels_ == idx` scans the full `labels_` array on every iteration. The estimator carries `self.n_jobs` (default `4`, :658) but does not forward it here. The in-code `TODO` ("Implement weighted argmin PWD backend") is left unresolved.
scenario: "`HDBSCAN(store_centers=\"medoid\", n_jobs=-1).fit(X)` on data with many clusters → each cluster's medoid distance matrix is built serially on one core; per-cluster mask scans give an O(n_samples × n_clusters) label-scan cost on top of the pairwise work."
contract: Pass `n_jobs=self.n_jobs` into `pairwise_distances` and precompute `cluster_indices = {idx: np.flatnonzero(self.labels_ == idx) for idx in range(n_clusters)}` in a single pass before the loop; index `X` and `probabilities_` from the cached indices.
instances: single-instance
