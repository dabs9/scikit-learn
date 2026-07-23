I note the system reminders about malware analysis. This is a merge task for review findings about scikit-learn's HDBSCAN implementation — no malware is present, just code review findings. I will produce only the merged findings document as instructed.

Now I'll merge. Identifying same-defect groups by root cause + site:

- **weighted_cluster_center shape mismatch (X vs labels_)**: A/F2, C/F1 — same site (hdbscan.py:855). Use C/F1 (most specific with line ranges 854-855, 908-909).
- **weighted_cluster_center omits -3**: A/F1, C/F2 — same site (hdbscan.py:895). Use C/F2 (more specific with line 907, 912, ZeroDivisionError).
- **n_jobs default mismatch**: A/F3, B/F5, C/F3 — same site (hdbscan.py:658). A/F3 has most specific evidence.
- **_hdbscan_brute alpha=None**: A/F4, B/F4, C/F5 — same site (hdbscan.py:161/241). C/F5 is most specific (mentions companion _hdbscan_prims).
- **remap_single_linkage_tree set iteration**: B/F6, C/F4 — same site (hdbscan.py:388). C/F4 is more specific.
- **_do_labelling parent_lambda scalar**: B/F1 (allow_single_cluster path, threshold empty max) vs C/F8 (ambiguous truth value). Different aspects but same site/related. B/F1 focuses on threshold max() empty; C/F8 focuses on scalar-vs-array in `if`. These are related but distinct concerns — keep both.

Others are unique.

Now producing the merged output (renumbered by severity: high, medium, low):

### F1 — `_weighted_cluster_center` shape mismatch when non-finite data + `store_centers` set
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-855 passes the finite-filtered local `X` (from line 733: `X = X[finite_index]`) into `_weighted_cluster_center`, but by that point `self.labels_` has been re-expanded at lines 840-844 to shape `(self._raw_data.shape[0],)`. Inside `_weighted_cluster_center` at line 908, `mask = self.labels_ == idx` has length `n_raw`, then line 909 executes `data = X[mask]` where `X` has length `n_finite < n_raw`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan/np.inf rows → boolean-mask indexing raises `IndexError: boolean index did not match indexed array along dimension 0`"
contract: pass `self._raw_data` (and update the mask/probabilities selection to align with raw data indexing) so `X` and the boolean mask derived from `self.labels_` share the same length.
instances: single-instance

### F2 — `_weighted_cluster_center` treats `-3` (missing) as a valid cluster label
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 computes `n_clusters = len(set(self.labels_) - {-1, -2})`, omitting `-3`. `_OUTLIER_ENCODING["missing"]["label"] = -3` (sklearn/cluster/_hdbscan/hdbscan.py:74) is a documented outlier label. When missing-data outliers are present, `-3` remains in the label set and inflates `n_clusters` by one; the loop at line 907 iterates `range(n_clusters)` looking for `self.labels_ == idx` and one of those idx values will have no matching rows, so `np.average(data, weights=strength, axis=0)` at line 912 receives an empty `data`/`strength` and raises `ZeroDivisionError: Weights sum to zero, can't be normalized`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan rows → inflated `n_clusters`, empty-cluster iteration, ZeroDivisionError from `np.average`"
contract: subtract all outlier labels — `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.
instances: single-instance

### F3 — Default `n_jobs=4` contradicts the documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature `n_jobs=4`; the docstring at lines 486–490 says `"None means 1 unless in a :obj:joblib.parallel_backend context. -1 means using all processors."` and the "default=None" convention is stated at line 486 (`n_jobs : int, default=None`).
scenario: "User instantiates `HDBSCAN()` expecting the documented `None` default → 4 worker jobs are always spawned inside `pairwise_distances`, oversubscribing CPUs (especially inside outer parallel loops) and diverging from every other sklearn estimator's documented and actual default of `None`."
contract: Change the default to `n_jobs=None` so the runtime behavior matches the documented default.
instances: single-instance

### F4 — `_do_labelling` reads `parent_lambda` and `threshold` per-sample instead of comparing scalars, causing potential wrong labels for `allow_single_cluster` epsilon path
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498-506 — `parent_lambda = lambda_array[child_array == n]` is an ndarray (length 1 in the intended case), and `threshold = lambda_array[parent_array == cluster].max()` uses `cluster` which is the union-find root that at this branch equals `root_cluster` (not a leaf/point), so `parent_array == cluster` may match zero entries because in `allow_single_cluster` mode all non-cluster edges were unioned; the min-parent root itself may have no child edges left after masking, producing `max()` on an empty array which raises `ValueError` at runtime.
scenario: "User calls `HDBSCAN(allow_single_cluster=True, cluster_selection_epsilon=0.0).fit(X)` on data that produces exactly one cluster where every child of the root is a leaf-cluster → `_do_labelling` raises `ValueError: zero-size array` when computing per-sample thresholds"
contract: The threshold and per-sample lambda arithmetic must be scalar-safe; if `parent_array == cluster` is empty, use the max child lambda over `condensed_tree` for that parent, and coerce `parent_lambda` to a scalar with an explicit `[0]` index.
instances: single-instance

### F5 — `test_dbscan_clustering_outlier_data` computes `clean_idx` by summing ndarrays instead of concatenating them, silently masking the wrong indices
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. With `missing_labels_idx == np.array([2,5])` and `infinite_labels_idx == np.array([0])`, NumPy broadcasts the `+` operator to element-wise addition producing `np.array([2, 5])`, so the "outlier" set becomes `{2, 5}` — the infinite index `0` is not excluded. `clean_idx` therefore contains index 0 (an `np.inf` row), and the subsequent `assert_array_equal(clean_labels, labels[clean_idx])` may fail or pass only by accident.
scenario: "Test run with any parametrized `cut_distance` → `clean_model.fit(X_outlier[clean_idx])` sees an infinite-valued row and behaves as another test case; the assertion at line 215 is comparing wrong slices"
contract: Concatenate index arrays with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or use `.tolist()` and list concatenation) before wrapping in `set(...)`.
instances: single-instance

### F6 — `test_hdbscan_precomputed_non_brute` never exercises the intended check because `algorithm="prims_kdtree"`/`"prims_balltree"` is rejected by parameter validation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` uses algorithm names that are not in the allowed StrOptions set `{"auto","brute","kdtree","balltree"}` (see sklearn/cluster/_hdbscan/hdbscan.py:629-638). `_validate_params()` raises `InvalidParameterError` (a `ValueError` subclass) before the "precomputed + tree" guard at hdbscan.py:772-783 is ever hit, so the test passes vacuously and does not validate what its docstring claims.
scenario: "Regression is introduced that silently allows `algorithm='kdtree'` with `metric='precomputed'` → this test still passes because it uses a bogus algorithm name and only tests parameter validation"
contract: Change the parametrization to `["kdtree", "balltree"]` so the intended precomputed-vs-tree guard is actually exercised.
instances: single-instance

### F7 — `_hdbscan_brute` default `alpha=None` will crash on `distance_matrix /= alpha`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None` in the signature, but sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally runs `distance_matrix /= alpha` (no None check). `ndarray /= None` raises `TypeError`. The companion `_hdbscan_prims` at line 273 correctly defaults `alpha=1.0`.
scenario: "any direct call to `_hdbscan_brute(X)` without an explicit `alpha` → immediate `TypeError` at line 241"
contract: default `alpha=1.0` (matching `_hdbscan_prims` and the docstring at line 183).
instances: single-instance

### F8 — `remap_single_linkage_tree` iterates over a Python `set`, producing non-deterministic outlier tree
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` (a `set`) into `remap_single_linkage_tree`. sklearn/cluster/_hdbscan/hdbscan.py:388 `for i, outlier in enumerate(non_finite):` iterates a `set` and writes `outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`; `set` iteration order is not guaranteed reproducible across processes / PYTHONHASHSEED values, so `self._single_linkage_tree_` ordering of the appended outlier edges is non-deterministic.
scenario: "user fits HDBSCAN twice on identical data with non-finite rows in separate processes and inspects `self._single_linkage_tree_` → sees different orderings of the appended outlier subtree between runs"
contract: pass and iterate a sorted, ordered container (e.g. `np.sort(np.unique(np.concatenate([infinite_index, missing_index])))`) so the appended outlier subtree is deterministic.
instances: single-instance

### F9 — `_hdbscan_brute` calls `distance_matrix /= alpha` even in the `metric="precomputed"` + `copy=False` path, mutating user-owned data
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:236-241 — when `metric == "precomputed"`, `distance_matrix = X.copy() if copy else X`; unconditionally after this, `distance_matrix /= alpha` runs in-place. With `copy=False` (the default) and `alpha != 1.0`, the user's precomputed distance matrix `X` is silently divided by `alpha` and mutated. The docstring at lines 207-212 states "it only applies when `metric='precomputed'`, when passing a dense array or a CSR sparse array/matrix", implying the user's matrix should be preserved when `copy=True` but does not warn that `copy=False` will mutate it.
scenario: "User calls `HDBSCAN(metric='precomputed', alpha=2.0, copy=False).fit(D)` → `D` is silently divided by 2 in place; subsequent code using `D` gets wrong distances"
contract: When `alpha != 1.0` and `metric == "precomputed"` and `copy=False`, force a copy before the in-place `/=`.
instances: single-instance

### F10 — `_compute_stability` allocates `births` twice, hiding intent
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two identical `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` statements back-to-back; the first allocation is unconditionally overwritten by the second, so any read between them (there is none currently) would silently disappear if reintroduced.
scenario: "Future maintenance edit that inserts logic between lines 252 and 254 → the inserted computation is silently discarded when line 254 overwrites the array, producing wrong stability values with no compile-time signal."
contract: Delete the duplicate `births = np.full(...)` at line 252 (or 254) so only a single, intentional allocation remains.
instances: single-instance

### F11 — `min_samples` bound check runs after silently dropping non-finite rows, producing misleading validation
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:758-762 — `if self._min_samples > X.shape[0]: raise ValueError(f"min_samples ({self._min_samples}) must be at most the number of samples in X ({X.shape[0]})")` runs after `X = X[finite_index]` at line 733; the reported "number of samples in X" is the count of finite rows, not the user-supplied `X.shape[0]`.
scenario: "User passes `min_samples=200` on 200-row input where 5 rows are non-finite → error message reports `n_samples=195` even though the user supplied 200 rows, hindering diagnosis."
contract: Compute and check `min_samples > n_samples_original` against the pre-reduction sample count, or clearly qualify the error message as "the number of finite samples in X".
instances: single-instance

### F12 — `_condense_tree` builds `ignore` sized to `len(node_list)` but the `node_list` returned by `bfs_from_hierarchy` may not include indices used to write into `ignore[sub_node]`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:161 — `ignore = np.zeros(len(node_list), dtype=bool)`. In the "both children small" and single-side branches, indices written are `ignore[sub_node]` where `sub_node` iterates results of recursive `bfs_from_hierarchy(hierarchy, left/right)`. For a well-formed hierarchy of `n_samples` samples, the initial BFS from `root = 2*(n_samples-1)` visits all `2*n_samples - 1` nodes, so `len(node_list) == 2*n_samples - 1` and indices in `[0, 2*n_samples-2]` are valid. However, `ignore[node]` at line 164 uses `node`, which can be up to `root = 2*(n_samples-1) = 2*n_samples - 2` — the array size `2*n_samples - 1` is exactly `root + 1`, so within bounds. The invariant that BFS visits every node is not enforced or asserted; any future refactor that skips duplicate nodes in BFS breaks `ignore` sizing without a bounds error under Cython `boundscheck=False`.
scenario: "A future edit reduces `bfs_from_hierarchy` output size (e.g., filters leaves) → `ignore[sub_node]` writes past the end of the array, silently corrupting adjacent memory when `boundscheck=False`"
contract: Size `ignore` from a semantically explicit expression tied to the hierarchy (e.g., `2 * hierarchy.shape[0] + 1`) rather than `len(node_list)`.
instances: single-instance

### F13 — `csgraph.connected_components` return value is treated as a comparable scalar without unpacking, silently changing meaning if `return_labels` semantics change
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:111-116 — `if (csgraph.connected_components(mutual_reachability, directed=False, return_labels=False) > 1):`. `return_labels=False` currently returns a single `int`, so `> 1` is well-defined. However, historical scipy versions have returned a tuple even with `return_labels=False` in some contexts; the code is not defensive against that and will silently do wrong comparisons (`tuple > 1` raises `TypeError` on Python 3 — which is at least loud — but the reliance on a scipy implementation detail is fragile).
scenario: "Older/newer scipy returns tuple/other type → `TypeError` at runtime or a truthy-but-meaningless comparison"
contract: Explicitly unpack with `n_components = csgraph.connected_components(...)` and compare `n_components > 1`.
instances: single-instance

### F14 — `HIERARCHY_dtype` `cluster_size` field is typed `np.intp` but `remap_single_linkage_tree` writes `last_cluster_size + 1` values that grow monotonically without bounds check
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:387-391 — `last_cluster_size = tree[tree.shape[0] - 1]["cluster_size"]` then in a loop `last_cluster_size += 1` and stores `last_cluster_size + 1` per outlier. For `n_samples > 2^{31}` on Windows (where `intp` is 32-bit), this can overflow, but more importantly the reported `cluster_size` for each outlier row is `last_cluster_size + 1`, incremented per iteration — this is claiming the outlier cluster grows one-by-one, which is consistent with them being sequentially merged, but adds `outlier_count` to the reported cluster_size on the last row rather than the true final total (`n_samples`) if there are duplicates in `non_finite`. Since `non_finite=set(...)` removes duplicates, this happens to be correct today.
scenario: "Refactor passes a list (not set) with duplicates in `non_finite` → `cluster_size` values grow past the true total sample count silently"
contract: Derive final `cluster_size` from `len(tree)+1` invariant rather than counting increments.
instances: single-instance

### F15 — `_do_labelling`: `result = np.empty(root_cluster, dtype=np.intp)` conflates the value of `root_cluster` (min parent id) with the number of samples
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-481 — `root_cluster = np.min(parent_array); result = np.empty(root_cluster, dtype=np.intp)`. This relies on `root_cluster == n_samples` which is only guaranteed by `_condense_tree`'s numbering scheme (relabel starts at `n_samples + 1`, but the root itself is relabeled to `n_samples`). Any change to the relabel scheme in `_condense_tree` breaks `_do_labelling` silently: `result` will be sized wrong, and the subsequent `for n in range(root_cluster)` will over- or under-scan samples with no error.
scenario: "Refactor of `_condense_tree` shifts the root label → `_do_labelling` returns an array of the wrong length; label assignments silently corrupted"
contract: Pass `n_samples` explicitly to `_do_labelling` (or derive it from `condensed_tree[condensed_tree['cluster_size']==1]['child'].max()+1`) rather than relying on `min(parent_array)` accidentally equalling `n_samples`.
instances: single-instance

### F16 — `_hdbscan_prims` computes `NearestNeighbors(..., p=None)` which is incompatible with `metric='minkowski'` default
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:332-340 — passes `p=None` unconditionally. `NearestNeighbors.__init__` documents `p=2` default and accepts numeric `p`. Passing `p=None` circumvents the metric-specific `p` parameter and may cause distance computations to ignore user-provided `p` in `metric_params` for `minkowski`-family metrics.
scenario: "User calls `HDBSCAN(metric='minkowski', metric_params={'p': 3})` with `algorithm='kdtree'` → `p=None` overrides; nearest-neighbor step may error or silently use `p=2`"
contract: Do not pass `p=None`; let `NearestNeighbors` use its own default and route `p` through `metric_params`.
instances: single-instance

### F17 — Sparse core-distance semantics differ from dense (self-distance handling)
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133-137 (dense) selects `partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx]` over the full row including the implicit `0` self-distance, i.e. the `(further_neighbor_idx)`-th smallest counting self at position 0. sklearn/cluster/_hdbscan/_reachability.pyx:193-200 (sparse) selects `partition(row_data, further_neighbor_idx)[further_neighbor_idx]` over `data[indptr[i]:indptr[i+1]]`, which excludes any zero self-entry that was `eliminate_zeros()`d (standard for CSR distance matrices). Result: for the same underlying distances, the sparse path effectively uses the k-th nearest neighbor while the dense path uses the (k-1)-th, an off-by-one.
scenario: "user supplies a precomputed CSR distance matrix with implicit zero diagonal → HDBSCAN produces a different clustering than the equivalent dense matrix with the same explicit distances"
contract: set the dense diagonal to `np.inf` before partitioning so both paths exclude the self-entry consistently.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:196]

### F18 — Sparse `_get_finite_row_indices` misses non-finite structural patterns
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 iterates `matrix.tolil().data` and calls `np.all(np.isfinite(row))`, but LIL `.data` only exposes explicitly stored non-zero values. Rows in which the sparse pattern encodes missing distances via structural absence (rather than explicit `np.inf` / `np.nan`) will always be classified as fully finite; conversely, rows containing only structural zeros (empty `row`) also pass `np.all(np.isfinite([])) == True`.
scenario: "user constructs a sparse distance matrix that omits distances beyond some threshold (typical KNN graph) → `_get_finite_row_indices` reports every row as finite, so no rows are separated out for the non-finite handling path even when the caller expected that behavior"
contract: derive per-row non-finiteness from the CSR triplet directly, e.g. `finite_rows = np.array([np.all(np.isfinite(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])) for i in range(matrix.shape[0])])`.
instances: single-instance

### F19 — `_do_labelling` single-cluster branch depends on `parent_lambda` being a scalar-like 1-element array
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 `parent_lambda = lambda_array[child_array == n]` returns an ndarray; sklearn/cluster/_hdbscan/_tree.pyx:505 `if parent_lambda >= threshold:` implicitly converts to scalar bool only when the array has exactly one element. The inline comment at lines 496-497 asserts "There can only be one edge with this particular child" — but there is no assertion or guard enforcing it. If a malformed condensed tree ever has duplicate child rows (or `n` is inadvertently referenced as a cluster), `if` on a multi-element boolean array raises `ValueError: The truth value of an array with more than one element is ambiguous`.
scenario: "any hierarchy pathology (duplicated child row, or invocation with n that matches multiple entries) → cryptic ValueError from ambiguous-truth-value evaluation instead of a domain-specific error"
contract: extract the scalar explicitly (`parent_lambda = lambda_array[child_array == n][0]`) and assert the single-edge invariant before comparison.
instances: single-instance
