### F1 — `bfs_from_cluster_tree` scans the entire condensed tree on every BFS layer
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:279-296 — `while len(process_queue) > 0: … process_queue = children[np.isin(parents, process_queue)]`. Each iteration performs `np.isin` over the whole `parents` array (length = |condensed_tree|).
scenario: "EOM cluster selection with many clusters → `_get_clusters` calls `bfs_from_cluster_tree(cluster_tree, node)` inside the `for node in node_list` loop (line 737), so every non-selected cluster triggers a fresh full-array `np.isin` sweep, giving overall O(|node_list| · depth · |cluster_tree|) hot-path work on the condensed tree."
contract: Precompute a `parent → list-of-children` adjacency (single O(|cluster_tree|) pass), then walk that adjacency in `bfs_from_cluster_tree` instead of `np.isin` scans.
instances: single-instance

### F2 — `_do_labelling` recomputes a loop-invariant argmax per sample
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-506 — in the loop `for n in range(root_cluster): … elif len(clusters) == 1 and allow_single_cluster: … threshold = lambda_array[parent_array == cluster].max()`. In this branch `cluster == root_cluster` (the `if cluster != root_cluster` failed), so `parent_array == cluster` is identical for every `n`, yet a full-array boolean scan + `.max()` is executed inside the per-sample loop.
scenario: "`allow_single_cluster=True` on a dataset that yields a single detected cluster → threshold recomputed n_samples times → O(n_samples · |condensed_tree|) instead of a single hoisted O(|condensed_tree|) computation."
contract: Compute `threshold = lambda_array[parent_array == root_cluster].max()` once before the loop when `cluster_selection_epsilon == 0.0`, and reuse it.
instances: single-instance

### F3 — Dense mutual-reachability writes both triangles of a symmetric matrix
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:130-149 — comment states "We assume that the distance matrix is symmetric", yet `for i in range(n_samples): for j in range(n_samples): … distance_matrix[i, j] = mutual_reachibility_distance` iterates the full n×n grid and stores each cell (each pair written twice via (i,j) and (j,i)).
scenario: "`_hdbscan_brute` on n samples → this dense pass costs ≈ 2× the necessary computations and stores, doubling wall-time of the O(n²) hot path."
contract: Iterate only `for j in range(i, n_samples)` (or `i+1..n`), compute the value once, mirror-assign `distance_matrix[i, j] = distance_matrix[j, i] = mutual_reachibility_distance`.
instances: single-instance

### F4 — `recurse_leaf_dfs` concatenates children lists with `sum([...], [])`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:555-566 — `return sum([recurse_leaf_dfs(cluster_tree, child) for child in children], [])` recursively flattens with `sum(list_of_lists, [])`, which is O(k²) in the total number of leaves due to repeated list rebuilding on each `+`.
scenario: "Deep or wide condensed cluster trees on `cluster_selection_method=\"leaf\"` (or EOM followed by `get_cluster_tree_leaves`) → leaf DFS on n-leaf trees costs O(n²) instead of O(n)."
contract: Accumulate results into a single mutable list (e.g., `result = []; ... result.extend(recurse_leaf_dfs(...))`) or convert to an explicit stack-based DFS.
instances: single-instance

### F5 — `epsilon_search` performs O(|processed|) membership tests via a list
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:611-634 — `processed` is declared `list processed = list()` and later probed with `if leaf not in processed:` (line 623) and appended via `processed.append(sub_node)` (line 634). `in` on a Python list is O(|processed|).
scenario: "Cluster-selection with a non-trivial `cluster_selection_epsilon` on trees with many leaves/sub-nodes → total cost of the guard is O(leaves²) instead of O(leaves)."
contract: Change `processed` to a `set` (initialize `processed = set()`, use `.add(sub_node)`).
instances: single-instance

### F6 — `_compute_stability` allocates and populates `births` twice
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-256 — two consecutive statements: `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252, then again `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 254 with no intervening use. The first allocation is dead.
scenario: "Every call to `_compute_stability` (once per `fit`) → wastes an O(largest_child) allocation + fill; small in absolute terms but a straight duplicate."
contract: Delete line 252 so `births` is allocated exactly once before the population loop.
instances: single-instance

### F7 — `_get_clusters` EOM loop rescans `cluster_tree['parent'] == node` per node
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — `for node in node_list: child_selection = (cluster_tree['parent'] == node) …` performs a full boolean scan of the parent column of `cluster_tree` for every node in `node_list`.
scenario: "Cluster selection with a condensed tree of C nodes → O(C · |cluster_tree|) parent-scan work, plus the nested `bfs_from_cluster_tree` scan noted in F1, giving quadratic behavior on the EOM hot path for large trees."
contract: Precompute a `parent → child_indices` mapping (single O(|cluster_tree|) pass) and reuse it for both the `child_selection` gathers and the BFS in F1.
instances: single-instance

### F8 — `traverse_upwards` scans full cluster tree per recursion step
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:576-602 — each recursive call executes `cluster_tree[cluster_tree['child'] == leaf]['parent']` (line 586) and `cluster_tree[cluster_tree['child'] == parent]['value']` (line 593), scanning the whole condensed tree twice per upward step.
scenario: "Called from `epsilon_search` (line 624) for every low-epsilon leaf → total cost O(leaves · depth · |cluster_tree|); on tall cluster hierarchies this dominates cluster selection."
contract: Build a `child → (parent, value)` dict once at entry to `_get_clusters` and look up parents/values in O(1) inside `traverse_upwards`.
instances: single-instance

### F9 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented `None` default [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — signature declares `n_jobs=4`, but the class docstring (lines 486-490) states "`None` means 1 unless in a `joblib.parallel_backend` context" and lists `n_jobs : int, default=None`. Every call without an explicit `n_jobs` silently oversubscribes to 4 threads inside `pairwise_distances`.
scenario: "A user relying on the documented default gets 4-way parallelism (thread oversubscription in nested parallel contexts, unexpected CPU usage) → surprises users and contradicts sklearn's estimator convention."
contract: Change the default to `n_jobs=None` in `__init__` so behavior matches the docstring and sklearn convention.
instances: single-instance
