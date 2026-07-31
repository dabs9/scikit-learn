Based on my exhaustive review of the data-access and performance surface across the new HDBSCAN modules, here are my verified findings.

### F1 — `_get_clusters` EOM loop: per-node full mask scan of `cluster_tree`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — `for node in node_list: child_selection = (cluster_tree['parent'] == node); subtree_stability = np.sum([stability[child] for child in cluster_tree['child'][child_selection]])` and, in the else-branch, `for sub_node in bfs_from_cluster_tree(cluster_tree, node): ...`
scenario: "large tree passed to `HDBSCAN.fit` with `cluster_selection_method='eom'` (default) → per-node boolean mask over the whole `cluster_tree` for every internal cluster gives O(|node_list| × |cluster_tree|) work in the mainline hot path; a single precomputed `parent → children` adjacency dict makes the same sweep O(|cluster_tree|)"
contract: Build a `parent → list-of-children` (or `parent → list-of-(child, stability)`) dict once up-front from `cluster_tree` and index it in the EOM loop (and in `bfs_from_cluster_tree`) rather than re-masking the array per node.
instances: single-instance

### F2 — `traverse_upwards` scans full `cluster_tree` twice per recursion level
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586-602 — `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` then `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']`, followed by tail-recursive call
scenario: "`cluster_selection_epsilon > 0` with many leaves near the root → for each leaf, every level of ancestor traversal performs two full O(|cluster_tree|) mask scans of `cluster_tree['child']`, giving O(leaves × depth × |cluster_tree|) worst-case work in `epsilon_search`"
contract: Precompute a `child → (parent, value)` dict once and follow parent pointers in O(1) per step; the recursive scans must not remain in the per-leaf loop.
instances: single-instance

### F3 — `recurse_leaf_dfs` combines per-recursion full scan with O(n²) `sum([...], [])`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:562-566 — `children = cluster_tree[cluster_tree['parent'] == current_node]['child']` then `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])`
scenario: "`cluster_selection_method='leaf'` on a tree with many leaves → each recursion masks all of `cluster_tree` again, and `sum([...], [])` reallocates the growing leaf list on every element, so leaf-finding is O(nodes × |cluster_tree|) with an extra O(leaves²) from the list concatenation"
contract: Build a `parent → children` dict once, and replace `sum([...], [])` with `itertools.chain.from_iterable(...)` or an in-place-appending accumulator so the concat is O(leaves).
instances: single-instance

### F4 — `_do_labelling` single-cluster branch: per-iteration full mask scans and non-hoisted constant threshold
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-508 — inside `for n in range(root_cluster):`, the `elif len(clusters) == 1 and allow_single_cluster:` branch runs `parent_lambda = lambda_array[child_array == n]` and `threshold = lambda_array[parent_array == cluster].max()` (with `cluster == root_cluster` fixed) per iteration
scenario: "`allow_single_cluster=True` on data producing one cluster with n samples → each iteration scans `child_array` and `parent_array` (each size = |condensed_tree|) again; `threshold` is invariant across the loop (`cluster` is constant here and `cluster_selection_epsilon` is a scalar) but is still recomputed n times, giving O(n × |condensed_tree|) work in what could be O(n) after hoisting"
contract: Precompute `threshold` and a `child → lambda` dict once before the `for n` loop; use dict lookups inside.
instances: single-instance

### F5 — `_dense_mutual_reachability_graph` iterates the full symmetric matrix, doubling the work
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — `for i in range(n_samples): for j in range(n_samples): mutual_reachibility_distance = max(core_distances[i], core_distances[j], distance_matrix[i, j]); distance_matrix[i, j] = mutual_reachibility_distance`
scenario: "`_hdbscan_brute` with a dense precomputed / pairwise distance matrix on n=10 000 → the nogil loop computes `max(...)` and writes n² = 10⁸ cells even though the input and output are both symmetric; iterating `j in range(i, n_samples)` and mirroring the write halves the work"
contract: Loop `j` from `i` to `n_samples`, compute the mrd once, and write both `distance_matrix[i, j]` and `distance_matrix[j, i]`.
instances: single-instance

### F6 — `_hdbscan_brute` unconditionally executes `distance_matrix /= alpha` even when `alpha == 1.0`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:241 — `distance_matrix /= alpha` runs regardless of `alpha`; `HDBSCAN.__init__` sets `alpha=1.0` by default (hdbscan.py:655) and passes it through in `kwargs` (hdbscan.py:764-770)
scenario: "default `HDBSCAN(...)` with `algorithm='brute'` (or brute chosen by dispatch) on n=20 000 → n² = 4×10⁸ float divisions-by-1.0 are executed and n² cells are re-written, dirtying the whole distance matrix for a numerical no-op"
contract: Guard the scaling with `if alpha != 1.0: distance_matrix /= alpha`.
instances: single-instance

### F7 — `bfs_from_cluster_tree` walks a full `np.isin` over `parents` at every BFS level
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:292-294 — `while len(process_queue) > 0: result.extend(process_queue.tolist()); process_queue = children[np.isin(parents, process_queue)]`
scenario: "invoked from the EOM loop (`_tree.pyx:737`) and `epsilon_search` (`_tree.pyx:632`) on a deep tree → each BFS level re-scans the whole `parents` column of `cluster_tree` via `np.isin`, giving O(depth × |cluster_tree|); with the `parent → children` dict from F1 this is O(nodes-visited)"
contract: Share the `parent → children` dict introduced for F1 and walk it directly instead of scanning `parents` per level.
instances: single-instance

### F8 — `_compute_stability` allocates `births` twice
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252 is immediately overwritten by the same expression on line 254 with no interposed use
scenario: "any `HDBSCAN.fit` → the first `np.full` allocation is unused (an obvious merge artifact) and its buffer is garbage-collected without ever being read; wastes an O(largest_child) allocation per call"
contract: Delete the first `births = np.full(...)` on line 252.
instances: single-instance

### F9 — `_get_finite_row_indices` converts CSR to LIL just to walk rows
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:400-404 — `if issparse(matrix): row_indices = np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])`
scenario: "`HDBSCAN.fit` on a sparse `X` with non-finite entries → `.tolil()` allocates two Python lists-of-lists of length n_samples (rows / data) and copies every non-zero into Python-list cells purely so the comprehension can iterate; iterating CSR's `indptr` and calling `np.isfinite` on each `data[indptr[i]:indptr[i+1]]` slice avoids the full-format conversion"
contract: Walk CSR directly via `indptr` slices of `matrix.data`; drop the `.tolil()` conversion.
instances: single-instance
