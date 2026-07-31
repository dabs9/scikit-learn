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

### F5 — Loop-invariant threshold recomputed per sample in `_do_labelling`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-506 — inside `for n in range(root_cluster)`, the `elif` branch is only entered when `cluster == root_cluster`, yet `threshold = 1 / cluster_selection_epsilon` (line 500) and `threshold = lambda_array[parent_array == cluster].max()` (line 504, plus its Θ(|condensed_tree|) mask) are recomputed on every iteration despite depending on nothing that varies with `n`.
scenario: "single-cluster dataset with `allow_single_cluster=True` on a large sample → an extra Θ(n·|condensed_tree|) of parent-mask scans plus n identical divisions, purely wasted"
contract: Hoist the epsilon check and the `lambda_array[parent_array == root_cluster].max()` above the `for n` loop; reuse the scalar `threshold` inside the loop.
instances: single-instance

### F6 — `births` allocated twice in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 assigns `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` and line 254 immediately re-assigns the identical expression; the first allocation is unused.
scenario: "every call to `_compute_stability` (once per `fit`) → an extra `largest_child + 1` float64 array is allocated and discarded on entry"
contract: Delete the line-252 allocation; keep only the one at line 254.
instances: single-instance

### F7 — `epsilon_search` uses a `list` for the "already processed" set
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:612 declares `list processed = list()`, line 623 tests `if leaf not in processed` and line 634 does `processed.append(sub_node)`.  Membership on a list is O(len(processed)).
scenario: "cluster tree with many leaves under a common epsilon-ancestor → `leaf not in processed` becomes an O(k²) accumulator over the leaves already merged into ancestor subtrees"
contract: Declare `processed` as a `set` and use `add` / `in` for O(1) membership.
instances: single-instance
