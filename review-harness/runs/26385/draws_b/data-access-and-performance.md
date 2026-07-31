### F1 — `_dense_mutual_reachability_graph` recomputes each cell twice
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:142-155 — nested `for i in range(n_samples): for j in range(n_samples):` computes `max(core_distances[i], core_distances[j], distance_matrix[i, j])` and writes `distance_matrix[i, j]`. The expression is symmetric in `(i, j)` and the input matrix is required to be symmetric (`pairwise_distances` output or the `_allclose_dense_sparse(X, X.T)` check in `_hdbscan_brute`), so cell `(i, j)` and `(j, i)` are recomputed and rewritten with the same value.
scenario: "brute algorithm on any dense input of size n → the mutual-reachability rewrite performs 2× the necessary comparisons and writes on the hot path"
contract: Iterate only the upper (or lower) triangle (`for j in range(i, n_samples)`) and mirror the assignment to `distance_matrix[j, i]`, halving loop work.
instances: single-instance

### F2 — `_do_labelling` recomputes constant `parent_array == cluster` scan inside the per-sample loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-506 — inside `for n in range(root_cluster):`, the `allow_single_cluster` + single-cluster branch executes `threshold = lambda_array[parent_array == cluster].max()` at line 504, but that branch is only entered when `cluster == root_cluster` (guaranteed by the enclosing `elif` after `if cluster != root_cluster:`), so the max is invariant across iterations. Similarly `1 / cluster_selection_epsilon` at line 500 is constant.
scenario: "`allow_single_cluster=True`, `cluster_selection_epsilon=0.0`, dataset resolves to one root cluster with N samples → the O(len(condensed_tree)) boolean scan+max runs up to N times, yielding O(N·|tree|) work where O(|tree|) suffices"
contract: Hoist both `threshold` computations out of the `for n in range(root_cluster):` loop; compute once before the loop starts.
instances: single-instance

### F3 — `_do_labelling` per-sample `child_array == n` scan is O(N) per iteration
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 — `parent_lambda = lambda_array[child_array == n]` scans the entire condensed tree once for every `n` in `range(root_cluster)`. The comment above the line acknowledges the child value is unique, so a precomputed mapping would give O(1) lookup.
scenario: "`allow_single_cluster=True` single-cluster path with N samples → O(N²) child-array scans instead of the O(N + |tree|) achievable with a `dict` built once from `zip(child_array, lambda_array)`"
contract: Build a `child → lambda` dict once before the loop (e.g., `child_to_lambda = dict(zip(child_array, lambda_array))`) and replace `lambda_array[child_array == n]` with a direct lookup.
instances: single-instance

### F4 — `X.sum(axis=1)` recomputed twice when input contains non-finite values
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:721 (`reduced_X = X.sum(axis=1)`) followed by `_get_finite_row_indices(X)` at line 731, whose dense branch at hdbscan.py:406 (`np.isfinite(matrix.sum(axis=1)).nonzero()`) recomputes the same `sum(axis=1)` over the full matrix. For dense X that is an extra O(n·d) pass; for sparse X the call additionally forces a CSR→LIL conversion.
scenario: "fit called on a large dense X containing any NaN/inf → the (n, d) matrix is summed row-wise twice back-to-back for a single logical check"
contract: Compute `reduced_X = X.sum(axis=1)` once and derive `finite_index = np.isfinite(reduced_X).nonzero()[0]` from it in the non-precomputed branch, dropping the separate `_get_finite_row_indices(X)` call for this path.
instances: single-instance

### F5 — Sparse `_get_finite_row_indices` forces CSR→LIL conversion and iterates row-by-row in Python
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 — `[i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]` materializes a whole LIL copy of the matrix and does a Python-level per-row `np.isfinite` on a Python list of values, discarding the (already available) CSR `indptr`/`data` slabs.
scenario: "sparse CSR input with any non-finite entries → an extra O(nnz) LIL rebuild plus O(n_rows) Python-loop calls to np.isfinite, versus one vectorized pass over `matrix.data`"
contract: Use the CSR arrays directly, e.g. compute `bad_rows = np.add.reduceat(~np.isfinite(matrix.data), matrix.indptr[:-1])` then `finite = np.flatnonzero(bad_rows == 0)`, avoiding the LIL conversion and Python-level per-row loop.
instances: single-instance

### F6 — `bfs_from_cluster_tree` rescans the full `parents` array on every BFS layer
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-296 — the loop body is `process_queue = children[np.isin(parents, process_queue)]`, which walks the entire `parents` array once per BFS layer. Called from `_get_clusters` at line 737 (`for sub_node in bfs_from_cluster_tree(cluster_tree, node)`) inside `for node in node_list`, so overlapping subtrees are re-walked from each retained cluster.
scenario: "EOM selection on a hierarchy where many candidate clusters are retained → each retained cluster triggers a fresh `np.isin(parents, queue)` scan per BFS layer, giving O(retained · depth · |tree|) work on the cluster-selection hot path"
contract: Build a `parent → list-of-children` adjacency (e.g. via `np.argsort(parents)` + `np.searchsorted`) once at the top of `_get_clusters` and reuse it inside `bfs_from_cluster_tree`, dropping the per-layer full-array scan.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]
