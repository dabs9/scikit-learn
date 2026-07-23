### F1 — Dense mutual-reachability walks the full n×n square despite symmetric input
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — inner nogil loop iterates `for i in range(n_samples): for j in range(n_samples):` and writes `distance_matrix[i, j] = mutual_reachibility_distance` for every pair, even though the block comment at :130-132 explicitly states "We assume that the distance matrix is symmetric." The adjacent TODO ("Update w/ prange with thread count based on `_openmp_effective_n_threads`") is likewise unresolved, so the loop is also single-threaded regardless of the user-supplied `n_jobs`.
scenario: "`HDBSCAN(algorithm=\"brute\").fit(X)` with dense pairwise input → the O(n²) mutual-reachability computation performs 2× the necessary `max()` operations (each symmetric pair `(i,j)`/`(j,i)` recomputed and rewritten), all on a single core; wall-clock on large dense fits is roughly double what a triangle-only or `prange` implementation would deliver."
contract: Iterate only the upper triangle (`for j in range(i, n_samples)`) and mirror the result to `distance_matrix[j, i]`, and dispatch the outer loop through `prange` gated by `_openmp_effective_n_threads(n_jobs)`.
instances: single-instance

### F2 — `_compute_stability` allocates the `births` array twice
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two back-to-back statements `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` with no intervening use of the first result; the first allocation is immediately discarded by the reassignment on the next line.
scenario: "Every HDBSCAN fit → `_compute_stability` runs once per fit, allocating `largest_child + 1` float64s and abandoning them; on large inputs `largest_child` grows with tree size, so a per-fit O(n) allocation is thrown away."
contract: Delete the duplicated allocation at line 252 and keep the single `births = np.full(...)` on line 254.
instances: single-instance

### F3 — `_get_clusters` EOM loop is quadratic in condensed-tree size
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — inside `for node in node_list:` (which iterates every cluster in the stability dict), each iteration performs `child_selection = (cluster_tree['parent'] == node)` — a full linear scan over the parent array — followed by `np.sum([stability[child] for child in cluster_tree['child'][child_selection]])`, a Python list comprehension boxed into `np.sum`. Nothing precomputes a `parent → children` adjacency. The same pattern recurs at :562, :586, :593, :725, :729.
scenario: "`HDBSCAN(...).fit(X)` on data with many clusters → EOM cluster selection is O(|node_list| × |cluster_tree|), which for a fanned-out condensed tree becomes quadratic in tree size with a Python-level constant factor per outer iteration."
contract: Precompute a `parent → children` mapping (e.g. sort cluster_tree by parent once and build an index array, or a Python `defaultdict(list)`) before the EOM loop and index into it per node; likewise reuse it for `traverse_upwards`, `recurse_leaf_dfs`, and the `cluster_sizes` root sum.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:562, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593, sklearn/cluster/_hdbscan/_tree.pyx:725, sklearn/cluster/_hdbscan/_tree.pyx:729]

### F4 — Sparse `_get_finite_row_indices` converts CSR→LIL and walks a Python list
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 — `row_indices = np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])`. `matrix.tolil()` materializes an object array of Python lists (one per row, size ≈ number of rows), the comprehension then iterates row-by-row in Python and calls `np.isfinite` on each list.
scenario: "`HDBSCAN().fit(sparse_X_with_nans)` on a large CSR matrix → an unnecessary O(n_rows) Python-level pass plus a LIL conversion that walks all nnz values into row-lists, when the same predicate is derivable directly from the CSR `.data`/`.indptr` in one vectorized pass."
contract: Compute a boolean `finite_data = np.isfinite(matrix.data)` and use `matrix.indptr` (via `np.add.reduceat` or per-row all-check) to build `row_indices` without leaving the CSR layout.
instances: single-instance

### F5 — `mst_from_mutual_reachability` allocates fresh numpy arrays inside its O(n²) hot loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:94-108 — every one of `n_samples - 1` iterations creates: `label_filter = current_labels != current_node`, `current_labels[label_filter]` (fancy-index copy), `min_reachability[label_filter]` (fancy-index copy), `mutual_reachability[current_node][current_labels]` (integer-index copy), and `np.minimum(left, right)` (new allocation), plus `np.argmin(...)`. All are Python-orchestrated with the GIL held. The docstring at :79-80 concedes ndarrays are chosen "to make use of numpy binary indexing and sub-selection", but there is no `nogil`/Cython buffer path.
scenario: "`HDBSCAN(algorithm=\"brute\").fit(dense_X)` with large n → Prim's MST performs ≥5 heap allocations per outer iteration for a total of O(n) short-lived ndarrays and O(n²) copied elements, GIL held throughout; the sister routine `mst_from_data_matrix` (:141-220) is fully nogil for comparison."
contract: Rewrite the loop as a `nogil` Cython pass using pre-allocated typed memoryviews for `in_tree`, `min_reachability`, and per-iteration argmin, matching `mst_from_data_matrix`'s pattern; drop the ndarray-based fancy-index chain.
instances: single-instance

### F6 — Medoid computation calls `pairwise_distances` per cluster, ignoring `n_jobs`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:907-920 — inside `for idx in range(n_clusters):` the medoid branch invokes `pairwise_distances(data, metric=self.metric, **self._metric_params)` with no `n_jobs` argument (so joblib parallelism defaults to 1), and `mask = self.labels_ == idx` scans the full `labels_` array on every iteration. The estimator carries `self.n_jobs` (default `4`, :658) but does not forward it here. The in-code `TODO` ("Implement weighted argmin PWD backend") is left unresolved.
scenario: "`HDBSCAN(store_centers=\"medoid\", n_jobs=-1).fit(X)` on data with many clusters → each cluster's medoid distance matrix is built serially on one core; per-cluster mask scans give an O(n_samples × n_clusters) label-scan cost on top of the pairwise work."
contract: Pass `n_jobs=self.n_jobs` into `pairwise_distances` and precompute `cluster_indices = {idx: np.flatnonzero(self.labels_ == idx) for idx in range(n_clusters)}` in a single pass before the loop; index `X` and `probabilities_` from the cached indices.
instances: single-instance
