### F1 — Dense NaN/Inf detection recomputes `X.sum(axis=1)` twice per fit
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:721 computes `reduced_X = X.sum(axis=1)` for identifying NaN/Inf rows, and immediately after at hdbscan.py:731 calls `_get_finite_row_indices(X)` which (for the dense branch, hdbscan.py:406) runs `np.isfinite(matrix.sum(axis=1)).nonzero()` — a second full O(n·d) sum over the same array.
scenario: "fit(X) with any non-finite entry in a large dense X → two full-array reductions where one would suffice, doubling the memory-bandwidth cost of finite-row bookkeeping"
contract: Compute `reduced_X = X.sum(axis=1)` once, then derive `finite_index` from that same array (e.g. `np.isfinite(reduced_X).nonzero()[0]`) instead of re-summing inside `_get_finite_row_indices`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:721, sklearn/cluster/_hdbscan/hdbscan.py:406]

### F2 — Repeated `cluster_tree['parent' | 'child'] == v` full scans build an O(N²) cluster-selection loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:728-739 iterates `for node in node_list` and at line 729 evaluates `child_selection = (cluster_tree['parent'] == node)` — a full O(|cluster_tree|) linear scan per node; and at 737 calls `bfs_from_cluster_tree` whose step (line 294) is `np.isin(parents, process_queue)`, another O(|parents|·|queue|) scan.  `recurse_leaf_dfs` (line 562) and `traverse_upwards` (lines 586, 593) exhibit the same "filter by field equality" pattern once per recursion.
scenario: "large dataset producing an O(n) condensed cluster tree → EOM cluster selection performs Θ(n²) structured-array scans in `_get_clusters`, and epsilon-search on a deep tree multiplies that by additional per-leaf recursive scans"
contract: Group `cluster_tree` once by parent (e.g. build a `defaultdict(list)` of `parent → (child, value, size)` or a sorted-by-parent index with searchsorted boundaries) at entry to `_get_clusters` / `epsilon_search`, and consult that index inside the hot loops instead of re-scanning the structured array on every iteration.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:729, sklearn/cluster/_hdbscan/_tree.pyx:737, sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:562, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593, sklearn/cluster/_hdbscan/_tree.pyx:620, sklearn/cluster/_hdbscan/_tree.pyx:725]

### F3 — Sparse `_get_finite_row_indices` densifies via LIL and iterates in Python
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 uses `matrix.tolil().data`, which materializes a Python object array of per-row Python lists, then walks it with a Python list-comp calling `np.all(np.isfinite(row))` per row.
scenario: "large sparse precomputed distance matrix containing any non-finite entry → conversion to LIL and per-row Python iteration allocates one Python list per sample and forces per-row NumPy dispatch, giving O(nnz) work with a per-row Python-level constant instead of the vectorised O(nnz) `np.isfinite` over `matrix.data`"
contract: Do a single vectorised pass on the CSR representation: compute `bad = ~np.isfinite(matrix.data)` and use `np.add.reduceat(bad, matrix.indptr[:-1])` (or `np.diff(np.searchsorted(...))` over bad indices) to derive the fully-finite row mask, without going through LIL.
instances: single-instance

### F4 — `_dense_mutual_reachability_graph` allocates a full n×n temporary just to extract n core distances
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133-137 — `core_distances = np.ascontiguousarray(np.partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx])`.  `np.partition(..., axis=1)` allocates a full n×n copy of the distance matrix before the `[:, k]` column slice throws all of it away.
scenario: "brute-force HDBSCAN on a dense n=20 000 pairwise-distance matrix → an additional ~3 GB float64 temporary is allocated for a computation whose output is n floats, doubling peak memory of the brute path"
contract: Partition row-by-row into an O(n) scratch buffer (e.g. `for i in range(n_samples): row = distance_matrix[i].copy(); row.partition(k); core_distances[i] = row[k]`) so the temporary is O(n), not O(n²).
instances: single-instance

### F5 — `_dense_mutual_reachability_graph` recomputes each cell twice
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:142-155 — nested `for i in range(n_samples): for j in range(n_samples):` computes `max(core_distances[i], core_distances[j], distance_matrix[i, j])` and writes `distance_matrix[i, j]`. The expression is symmetric in `(i, j)` and the input matrix is required to be symmetric (`pairwise_distances` output or the `_allclose_dense_sparse(X, X.T)` check in `_hdbscan_brute`), so cell `(i, j)` and `(j, i)` are recomputed and rewritten with the same value.
scenario: "brute algorithm on any dense input of size n → the mutual-reachability rewrite performs 2× the necessary comparisons and writes on the hot path"
contract: Iterate only the upper (or lower) triangle (`for j in range(i, n_samples)`) and mirror the assignment to `distance_matrix[j, i]`, halving loop work.
instances: single-instance

### F6 — `_do_labelling` recomputes constant `parent_array == cluster` scan inside the per-sample loop
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-506 — inside `for n in range(root_cluster):`, the `allow_single_cluster` + single-cluster branch executes `threshold = lambda_array[parent_array == cluster].max()` at line 504, but that branch is only entered when `cluster == root_cluster` (guaranteed by the enclosing `elif` after `if cluster != root_cluster:`), so the max is invariant across iterations. Similarly `1 / cluster_selection_epsilon` at line 500 is constant.
scenario: "`allow_single_cluster=True`, `cluster_selection_epsilon=0.0`, dataset resolves to one root cluster with N samples → the O(len(condensed_tree)) boolean scan+max runs up to N times, yielding O(N·|tree|) work where O(|tree|) suffices"
contract: Hoist both `threshold` computations out of the `for n in range(root_cluster):` loop; compute once before the loop starts.
instances: single-instance

### F7 — `_do_labelling` per-sample `child_array == n` scan is O(N) per iteration
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 — `parent_lambda = lambda_array[child_array == n]` scans the entire condensed tree once for every `n` in `range(root_cluster)`. The comment above the line acknowledges the child value is unique, so a precomputed mapping would give O(1) lookup.
scenario: "`allow_single_cluster=True` single-cluster path with N samples → O(N²) child-array scans instead of the O(N + |tree|) achievable with a `dict` built once from `zip(child_array, lambda_array)`"
contract: Build a `child → lambda` dict once before the loop (e.g., `child_to_lambda = dict(zip(child_array, lambda_array))`) and replace `lambda_array[child_array == n]` with a direct lookup.
instances: single-instance

### F8 — `_get_clusters` EOM loop: per-node full mask scan of `cluster_tree`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — `for node in node_list: child_selection = (cluster_tree['parent'] == node); subtree_stability = np.sum([stability[child] for child in cluster_tree['child'][child_selection]])` and, in the else-branch, `for sub_node in bfs_from_cluster_tree(cluster_tree, node): ...`
scenario: "large tree passed to `HDBSCAN.fit` with `cluster_selection_method='eom'` (default) → per-node boolean mask over the whole `cluster_tree` for every internal cluster gives O(|node_list| × |cluster_tree|) work in the mainline hot path; a single precomputed `parent → children` adjacency dict makes the same sweep O(|cluster_tree|)"
contract: Build a `parent → list-of-children` (or `parent → list-of-(child, stability)`) dict once up-front from `cluster_tree` and index it in the EOM loop (and in `bfs_from_cluster_tree`) rather than re-masking the array per node.
instances: single-instance

### F9 — `traverse_upwards` scans full `cluster_tree` twice per recursion level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586-602 — `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` then `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']`, followed by tail-recursive call
scenario: "`cluster_selection_epsilon > 0` with many leaves near the root → for each leaf, every level of ancestor traversal performs two full O(|cluster_tree|) mask scans of `cluster_tree['child']`, giving O(leaves × depth × |cluster_tree|) worst-case work in `epsilon_search`"
contract: Precompute a `child → (parent, value)` dict once and follow parent pointers in O(1) per step; the recursive scans must not remain in the per-leaf loop.
instances: single-instance

### F10 — `recurse_leaf_dfs` combines per-recursion full scan with O(n²) `sum([...], [])`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:562-566 — `children = cluster_tree[cluster_tree['parent'] == current_node]['child']` then `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`
scenario: "`cluster_selection_method='leaf'` on a tree with many leaves → each recursion masks all of `cluster_tree` again, and `sum([...], [])` reallocates the growing leaf list on every element, so leaf-finding is O(nodes × |cluster_tree|) with an extra O(leaves²) from the list concatenation"
contract: Build a `parent → children` dict once, and replace `sum([...], [])` with `itertools.chain.from_iterable(...)` or an in-place-appending accumulator so the concat is O(leaves).
instances: single-instance

### F11 — `_do_labelling` single-cluster branch: per-iteration full mask scans and non-hoisted constant threshold
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-508 — inside `for n in range(root_cluster):`, the `elif len(clusters) == 1 and allow_single_cluster:` branch runs `parent_lambda = lambda_array[child_array == n]` and `threshold = lambda_array[parent_array == cluster].max()` (with `cluster == root_cluster` fixed) per iteration
scenario: "`allow_single_cluster=True` on data producing one cluster with n samples → each iteration scans `child_array` and `parent_array` (each size = |condensed_tree|) again; `threshold` is invariant across the loop (`cluster` is constant here and `cluster_selection_epsilon` is a scalar) but is still recomputed n times, giving O(n × |condensed_tree|) work in what could be O(n) after hoisting"
contract: Precompute `threshold` and a `child → lambda` dict once before the `for n` loop; use dict lookups inside.
instances: single-instance

### F12 — `epsilon_search` uses a `list` for the "already processed" set
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:612 declares `list processed = list()`, line 623 tests `if leaf not in processed` and line 634 does `processed.append(sub_node)`.  Membership on a list is O(len(processed)).
scenario: "cluster tree with many leaves under a common epsilon-ancestor → `leaf not in processed` becomes an O(k²) accumulator over the leaves already merged into ancestor subtrees"
contract: Declare `processed` as a `set` and use `add` / `in` for O(1) membership.
instances: single-instance

### F13 — `bfs_from_cluster_tree` rescans the full `parents` array on every BFS layer
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-296 — the loop body is `process_queue = children[np.isin(parents, process_queue)]`, which walks the entire `parents` array once per BFS layer. Called from `_get_clusters` at line 737 (`for sub_node in bfs_from_cluster_tree(cluster_tree, node)`) inside `for node in node_list`, so overlapping subtrees are re-walked from each retained cluster.
scenario: "EOM selection on a hierarchy where many candidate clusters are retained → each retained cluster triggers a fresh `np.isin(parents, queue)` scan per BFS layer, giving O(retained · depth · |tree|) work on the cluster-selection hot path"
contract: Build a `parent → list-of-children` adjacency (e.g. via `np.argsort(parents)` + `np.searchsorted`) once at the top of `_get_clusters` and reuse it inside `bfs_from_cluster_tree`, dropping the per-layer full-array scan.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:294, sklearn/cluster/_hdbscan/_tree.pyx:632, sklearn/cluster/_hdbscan/_tree.pyx:737]

### F14 — `_hdbscan_brute` unconditionally executes `distance_matrix /= alpha` even when `alpha == 1.0`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:241 — `distance_matrix /= alpha` runs regardless of `alpha`; `HDBSCAN.__init__` sets `alpha=1.0` by default (hdbscan.py:655) and passes it through in `kwargs` (hdbscan.py:764-770)
scenario: "default `HDBSCAN(...)` with `algorithm='brute'` (or brute chosen by dispatch) on n=20 000 → n² = 4×10⁸ float divisions-by-1.0 are executed and n² cells are re-written, dirtying the whole distance matrix for a numerical no-op"
contract: Guard the scaling with `if alpha != 1.0: distance_matrix /= alpha`.
instances: single-instance

### F15 — `_compute_stability` allocates `births` twice
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252 is immediately overwritten by the same expression on line 254 with no interposed use
scenario: "any `HDBSCAN.fit` → the first `np.full` allocation is unused (an obvious merge artifact) and its buffer is garbage-collected without ever being read; wastes an O(largest_child) allocation per call"
contract: Delete the first `births = np.full(...)` on line 252.
instances: single-instance
