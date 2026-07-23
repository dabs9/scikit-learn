Acknowledged. `ignore` sizing: `len(node_list)` in the BFS traversal. `len(node_list) = 2n-1` (full tree). `node` values can be up to `2n-2` = root. `ignore[2n-2]` for size `2n-1` array — OK.

But wait, `ignore[sub_node] = True` at lines 205, 212, 221, 230 — sub_node is from `bfs_from_hierarchy(hierarchy, left/right)`, which returns nodes reachable from left/right. sub_node values can include internal nodes (up to root) and leaves. So `sub_node` up to root = 2n-2 → `ignore[2n-2]` for size 2n-1. OK.

Actually wait — sub_node could equal `right` (the sibling) if there's some weird tree walk? No, `bfs_from_hierarchy(hierarchy, left)` traverses subtree rooted at `left`, which excludes `right` in normal binary tree structure. OK.

But there's actually a subtle issue: `ignore = np.zeros(len(node_list), dtype=bool)`. `len(node_list)` = number of nodes visited in BFS from root — which for a full binary tree with n leaves is 2n-1. But `node` in the outer loop takes values from `node_list`. It's possible that a value like `2n-2` (root) is accessed in `ignore[2n-2]`. Size 2n-1, so max index is 2n-2. Fits exactly. OK.

Now, finalizing findings.

Let me add one more careful check. In `_get_clusters`, `condensed_tree` might be empty in edge cases where `_condense_tree` returns empty (e.g., very small dataset with high min_cluster_size). Then line 714: `n_samples = np.max(condensed_tree[condensed_tree['cluster_size'] == 1]['child']) + 1` fails.

Let me trace: for n=2 samples, hierarchy has 1 merge. Root = node 2. `_condense_tree` processes node 2 with left=0, right=1 (both leaves), left_count=right_count=1. If `min_cluster_size=5` (default) → both < 5 → elif branch: BFS from 0 gives [0], BFS from 1 gives [1]. Appends (n_samples, 0, lambda, 1) and (n_samples, 1, lambda, 1). Non-empty. OK.

For n=2 samples with min_cluster_size=2: 1 < 2 for both. Same elif branch. Non-empty.

Now `test_hdbscan_no_clusters` line 249: `min_cluster_size=len(X)-1=199`. Only the very final merge (root) has counts both>=199. Every other merge has one or both counts<199. So many elif/else branches. Non-empty result_list. OK.

I'll finalize now.

### F1 — `_weighted_cluster_center` omits `-3` (missing) label when counting clusters
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` only removes the noise (`-1`) and infinite-outlier (`-2`) labels, but not the missing-data label `-3` produced by `_OUTLIER_ENCODING["missing"]["label"]` (defined at hdbscan.py:74) when `X` contains `np.nan` rows.
scenario: "User calls `HDBSCAN(store_centers='centroid').fit(X)` where `X` contains any `np.nan` row → `-3` is counted as one of the clusters, so `n_clusters` is one greater than the number of real clusters; the subsequent `for idx in range(n_clusters)` reaches an `idx` with an empty mask, and `np.average(data, weights=strength, axis=0)` raises ZeroDivisionError (or `self.centroids_[idx]` remains uninitialized `np.empty`)."
contract: Compute `n_clusters` by removing all outlier labels defined in `_OUTLIER_ENCODING`, e.g. `n_clusters = len(set(self.labels_) - {-1} - {v['label'] for v in _OUTLIER_ENCODING.values()})`.
instances: single-instance

### F2 — `_weighted_cluster_center` masks reduced `X` with full-length `self.labels_`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:855 passes the reduced `X` (set at line 733 by `X = X[finite_index]`) into `_weighted_cluster_center`, while at line 908 the method computes `mask = self.labels_ == idx` using `self.labels_` which was expanded to raw size at line 840–844 (`new_labels = np.empty(self._raw_data.shape[0], ...)`, `self.labels_ = new_labels`). `X[mask]` then indexes a length-`n_finite` array with a length-`n_raw` boolean mask.
scenario: "User calls `HDBSCAN(store_centers='centroid').fit(X)` with any non-finite rows in `X` → `X.shape[0] != mask.shape[0]`, raising IndexError from the boolean indexing (or, if lengths coincidentally match, aligning centroid computations to the wrong rows)."
contract: Pass `self._raw_data` (or equivalently the raw-sized feature matrix) into `_weighted_cluster_center` so that `X` and `self.labels_` share the same length; the reduced `X` must not be used with the remapped `self.labels_`.
instances: single-instance

### F3 — Default `n_jobs=4` contradicts the documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature `n_jobs=4`; the docstring at lines 486–490 says `"None means 1 unless in a :obj:joblib.parallel_backend context. -1 means using all processors."` and the "default=None" convention is stated at line 486 (`n_jobs : int, default=None`).
scenario: "User instantiates `HDBSCAN()` expecting the documented `None` default → 4 worker jobs are always spawned inside `pairwise_distances`, oversubscribing CPUs (especially inside outer parallel loops) and diverging from every other sklearn estimator's documented and actual default of `None`."
contract: Change the default to `n_jobs=None` so the runtime behavior matches the documented default.
instances: single-instance

### F4 — `_hdbscan_brute` default `alpha=None` divides by `None`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None` as the default, but line 241 unconditionally executes `distance_matrix /= alpha`.
scenario: "Any direct caller of `_hdbscan_brute` that relies on the documented default (line 183 says `alpha : float, default=1.0`) → `TypeError: unsupported operand type(s) for /=: '...' and 'NoneType'` from line 241."
contract: Set the signature default to `alpha=1.0` to match the docstring, or guard the divide (`if alpha is not None`).
instances: single-instance

### F5 — `_compute_stability` allocates `births` twice, hiding intent
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two identical `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` statements back-to-back; the first allocation is unconditionally overwritten by the second, so any read between them (there is none currently) would silently disappear if reintroduced.
scenario: "Future maintenance edit that inserts logic between lines 252 and 254 → the inserted computation is silently discarded when line 254 overwrites the array, producing wrong stability values with no compile-time signal."
contract: Delete the duplicate `births = np.full(...)` at line 252 (or 254) so only a single, intentional allocation remains.
instances: single-instance

### F6 — `min_samples` bound check runs after silently dropping non-finite rows, producing misleading validation
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:758-762 — `if self._min_samples > X.shape[0]: raise ValueError(f"min_samples ({self._min_samples}) must be at most the number of samples in X ({X.shape[0]})")` runs after `X = X[finite_index]` at line 733; the reported "number of samples in X" is the count of finite rows, not the user-supplied `X.shape[0]`.
scenario: "User passes `min_samples=200` on 200-row input where 5 rows are non-finite → error message reports `n_samples=195` even though the user supplied 200 rows, hindering diagnosis."
contract: Compute and check `min_samples > n_samples_original` against the pre-reduction sample count, or clearly qualify the error message as "the number of finite samples in X".
instances: single-instance

NO FINDINGS beyond the above.
