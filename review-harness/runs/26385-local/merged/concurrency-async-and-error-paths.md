I note the system reminders about malware analysis policy. These files are review findings documents (markdown), not executable code, and my task is purely a merge operation — no code improvement or augmentation is involved.

I will not use TodoWrite for this simple merge task. Now performing the union merge:

Analysis of duplicates across draws:
- A.F2 = B.F4 (non_finite set iteration nondeterminism, hdbscan.py:388/838) → keep A.F2 (more specific with concrete fix)
- A.F3 = B.F10 = C.F3 (sparse mutual-reachability stale value when max_distance<=0, _reachability.pyx:209-212) → C.F3 is most specific (mentions the warning at hdbscan.py:256); keep C.F3
- B.F6 vs C.F2: B.F6 covers sparse precomputed copy issue in general; C.F2 pinpoints the `algorithm="auto"` branch not forwarding copy — different sites/root causes → keep both
- C.F6 relates to B.F6/C.F2 but focuses on dense precomputed path with auto and lack of copy forwarding + doc gap → distinct enough, keep

Sorting: high → medium → low.

### F1 — Shape mismatch in `_weighted_cluster_center` after non-finite pruning
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 pares `X = X[finite_index]`; sklearn/cluster/_hdbscan/hdbscan.py:840-844 rebuilds `self.labels_` to `self._raw_data.shape[0]` (raw length); sklearn/cluster/_hdbscan/hdbscan.py:854-855 then calls `self._weighted_cluster_center(X)` passing the pruned `X`; sklearn/cluster/_hdbscan/hdbscan.py:908 does `mask = self.labels_ == idx` (raw length) and sklearn/cluster/_hdbscan/hdbscan.py:909 does `data = X[mask]` on the pruned X.
scenario: "User fits HDBSCAN with `store_centers=\"centroid\"|\"medoid\"|\"both\"` on data containing any `np.nan`/`np.inf` row → boolean-mask/array shape mismatch (or silently wrong-indexed rows) when computing centers, raising `IndexError`/producing garbage centroids."
contract: When `all_finite` is False, pass `self._raw_data` (or an aligned finite subset paired with re-indexed labels/probabilities) into `_weighted_cluster_center` so that `mask.shape == X.shape[0]`.
instances: single-instance

### F2 — `remap_single_linkage_tree` `outlier_count` shift under-adjusts internal ids
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:369 sets `outlier_count = len(non_finite)`; sklearn/cluster/_hdbscan/hdbscan.py:377/381 shift internal cluster ids by `outlier_count`. `non_finite` is a `set` produced from `set(infinite_index + missing_index)` (sklearn/cluster/_hdbscan/hdbscan.py:838), so overlapping indices collapse; the correct number of "raw" sample slots that need to be reserved is the total raw sample count minus the finite count, not the size of the deduplicated set.
scenario: "User fits with a row that is *both* `np.inf` and `np.nan` in different columns (produces an entry in both `infinite_index` and `missing_index`) → `outlier_count` is smaller than the actual gap between finite indices and the id space needed, causing internal-node ids in the remapped hierarchy to collide with raw sample ids, silently mislabeling points in `_single_linkage_tree_`."
contract: Set `outlier_count = self._raw_data.shape[0] - finite_count` (or equivalently, count the union of infinite_index and missing_index without a set-of-tuples de-dup that under-counts) when remapping.
instances: single-instance

### F3 — `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default and sklearn convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in `__init__` signature; docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents "`None` means 1 unless in a `joblib.parallel_backend` context" implying `n_jobs=None` as default, matching sklearn convention throughout the file (e.g. `_hdbscan_brute` at line 163 uses `n_jobs=None`, `_hdbscan_prims` at line 276 uses `n_jobs=None`)
scenario: "User instantiates `HDBSCAN()` inside a `joblib.parallel_backend` context expecting to inherit the outer backend → HDBSCAN silently forks 4 concurrent processes/threads inside `pairwise_distances`/`NearestNeighbors`, over-subscribing the machine and defeating the parent's parallel policy"
contract: The default MUST be `n_jobs=None` to honor `joblib.parallel_backend` contexts consistent with the docstring and all other sklearn estimators
instances: single-instance

### F4 — `remap_single_linkage_tree` iterates a `set`, producing non-deterministic `_single_linkage_tree_`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` into `remap_single_linkage_tree`; sklearn/cluster/_hdbscan/hdbscan.py:388 iterates that set with `for i, outlier in enumerate(non_finite)` and writes `outlier_tree[i] = (outlier, ...)` in set-iteration order
scenario: "User fits `HDBSCAN` twice on the same data containing both `np.inf` and `np.nan` outliers → `self._single_linkage_tree_` differs across runs (rows reordered) because Python `set` iteration order is not stable across processes, breaking any code that hashes or diffs the tree attribute"
contract: `non_finite` MUST be passed as a sorted, deterministic sequence (e.g. `np.unique(np.concatenate([infinite_index, missing_index]))`) so `_single_linkage_tree_` is reproducible
instances: single-instance

### F5 — Sparse mutual-reachability leaves stale value in place when result is infinite and `max_distance <= 0` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:206-212 — after computing `mutual_reachibility_distance = max(core_distances[row_ind], core_distances[col_ind], data[i])`, the code writes back only when the result is finite, or when `max_distance > 0`. When the mutual reachability is infinite and `max_distance == 0` (the default at hdbscan.py:243), `data[i]` retains its original value (not the newly-computed infinite one). Downstream `_brute_mst` then runs `csgraph.connected_components` and `minimum_spanning_tree` on a graph whose weights do NOT reflect the mutual-reachability semantics for the affected edges, producing an MST whose weights understate the true infinite mutual reachability.
scenario: "User supplies a sparse precomputed distance matrix with a row that has fewer than `min_samples` stored neighbors (so `core_distances[row_ind] = INFINITY`), and does not specify `max_distance` → mutual reachability along that row is infinite but `data[i]` stays at the original finite distance → `_brute_mst` produces an MST that silently uses those stale distances instead of raising, and the 'contains edge weights with value infinity' warning at hdbscan.py:256 does not fire because the infinity was suppressed."
contract: When `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, `data[i]` must be set to `INFINITY` (or the entry otherwise flagged) so that downstream MST / connectivity checks see the true value; silently retaining the pre-existing entry hides a data condition that the code explicitly warns about elsewhere.
instances: single-instance

### F6 — `_weighted_cluster_center` counts missing-label (-3) samples as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the class-level `_OUTLIER_ENCODING` (sklearn/cluster/_hdbscan/hdbscan.py:65-79) defines `missing` label `-3`, which is excluded from `labels_` by sklearn/cluster/_hdbscan/hdbscan.py:843.
scenario: "User calls fit with `store_centers` set and X contains `np.nan` rows → `-3` is included in `n_clusters`, sizing `self.centroids_`/`self.medoids_` one row too large; the loop `for idx in range(n_clusters)` then produces an all-NaN or empty-mean row (empty slice warning + NaN centroid) and never touches the -3 mask, corrupting the exported centers."
contract: Exclude all three outlier encodings when counting non-noise clusters: `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.
instances: single-instance

### F7 — `_assert_all_finite(X.data)` on LIL sparse input misclassifies finiteness
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:703 accepts `accept_sparse=["csr", "lil"]`; sklearn/cluster/_hdbscan/hdbscan.py:710 calls `_assert_all_finite(X.data if issparse(X) else X)`. For LIL sparse, `X.data` is an `object` ndarray of Python lists (one per row), not a numeric buffer.
scenario: "User passes a LIL sparse feature matrix containing `np.nan`/`np.inf` → `_assert_all_finite` receives an object array and either (a) raises an unrelated dtype error, or (b) silently returns True because the object-dtype path does not descend into the per-row lists, sending non-finite data into `_hdbscan_brute` (fail-open path)."
contract: Materialise a numeric buffer before finiteness checking for sparse formats other than CSR (e.g., convert with `X.tocsr()` before the `_assert_all_finite` call), rather than relying on `X.data` which is format-dependent.
instances: single-instance

### F8 — `_brute_mst` sparse path silently accepts disconnected graphs when `connected_components` count equals 1 but MST is short
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:111-123 raises only when `csgraph.connected_components(...) > 1`; sklearn/cluster/_hdbscan/hdbscan.py:126-131 builds `mst` via `np.core.records.fromarrays([rows, cols, ...])` of whatever nonzeros scipy returns. `scipy.sparse.csgraph.minimum_spanning_tree` on an already-symmetrized MR graph with zero-valued edges (which are dropped as "no edge") can return fewer than `n_samples - 1` edges.
scenario: "User supplies a sparse precomputed distance matrix where the mutual-reachability graph is connected under the sparsity pattern but some MR distances collapse to zero (e.g., duplicate points) → returned MST has fewer than n-1 edges, `make_single_linkage` produces an incomplete/malformed `HIERARCHY_dtype` array, and downstream `_condense_tree` yields wrong labels rather than surfacing the partial-failure state."
contract: After `csgraph.minimum_spanning_tree`, assert `len(rows) == n_samples - 1` and raise a clear `ValueError` describing the partial MST rather than continuing with a truncated tree.
instances: single-instance

### F9 — Precomputed sparse path can mutate user's input array in place
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:734-740 (`elif issparse(X):` — no `copy=` argument to `_validate_data`); sklearn/cluster/_hdbscan/hdbscan.py:241 in `_hdbscan_brute`: `distance_matrix /= alpha` and sklearn/cluster/_hdbscan/_reachability.pyx:56 documents "Note that all computations are done in-place." The CSR branch in `_reachability.pyx:202-212` writes into `data`. The `copy` kwarg is threaded only for `algorithm=="brute"` (sklearn/cluster/_hdbscan/hdbscan.py:795) and `_hdbscan_brute` only honors it in the dense precomputed branch (sklearn/cluster/_hdbscan/hdbscan.py:222-236); the sparse precomputed branch ignores `copy` entirely.
scenario: "User passes a sparse CSR precomputed distance matrix with `HDBSCAN(metric='precomputed', copy=True).fit(D)` → `D.data` is mutated by `distance_matrix /= alpha` and by `_sparse_mutual_reachability_graph`, silently overwriting the caller's matrix; the `copy=True` documented guarantee is fail-open."
contract: In `_hdbscan_brute`, when `metric == 'precomputed'` and the input is sparse, honor `copy` by materialising a new CSR (e.g., `distance_matrix = X.copy()`) before any in-place mutation.
instances: single-instance

### F10 — `algorithm="auto"` silently drops `copy=True` for precomputed inputs
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:804-819 — the "auto" branch dispatches to `_hdbscan_brute` at line 807 (`if issparse(X) or self.metric not in FAST_METRICS`) but never sets `kwargs["copy"] = self.copy`. In the explicit `algorithm="brute"` branch at line 795 the copy flag is honored. `_hdbscan_brute` defaults `copy=False` at line 164, so with `metric="precomputed"` and `algorithm="auto"` the user-provided distance matrix is modified in place at lines 241/251 despite `copy=True`.
scenario: "User calls `HDBSCAN(metric='precomputed', copy=True).fit_predict(D)` (leaving algorithm at default `'auto'`) → the auto path picks `_hdbscan_brute` without forwarding `copy` → D is mutated in-place (divided by alpha, overwritten with mutual-reachability values) → user's `D` is silently corrupted."
contract: The `algorithm="auto"` branch must set `kwargs["copy"] = self.copy` whenever it dispatches to `_hdbscan_brute`, matching the explicit `algorithm="brute"` branch.
instances: single-instance

### F11 — `UnionFind` `.pxd`/`.pyx` `noexcept` mismatch swallows exceptions raised in `union`/`fast_find`
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` — neither definition repeats the `noexcept` qualifier that the `.pxd` header pins
scenario: "Cython 3 treats the `.pxd` declaration as authoritative → any Python-level exception raised inside `union` or `fast_find` (e.g. `IndexError` from bounds checks, allocation failures within the memoryview writes) is silently swallowed with only a WriteUnraisable to stderr, and the calling `make_single_linkage` continues with a corrupted UnionFind, producing a garbage single-linkage tree"
contract: The `.pyx` definitions MUST repeat the `noexcept` qualifier from the `.pxd` so the exception-propagation policy is unambiguous and reviewed
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F12 — `_hdbscan_brute` mutates caller's sparse input in place even when `copy=True` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:236 — `distance_matrix = X.copy() if copy else X`; sklearn/cluster/_hdbscan/hdbscan.py:244-247 — if the (now-copied) distance_matrix has a non-CSR format, `distance_matrix = distance_matrix.tocsr()` allocates a new CSR; but `X.copy()` on a sparse matrix preserves format, so if the user passed CSR with `copy=True`, `X.copy()` is CSR and `tocsr()` on a CSR returns `self` (no copy) — then sklearn/cluster/_hdbscan/hdbscan.py:241 (`distance_matrix /= alpha`) and the in-place `mutual_reachability_graph` at line 251 mutate the copy fine. However for `copy=False` + CSR user input, both `/=` and `mutual_reachability_graph` silently mutate the caller's `.data` buffer with no warning even though it is documented
scenario: "User passes a CSR sparse precomputed matrix with `copy=False` (the default) → `X.data /= alpha` and `mutual_reachability_graph` mutate the user's sparse buffer in place, invalidating the input for any further use"
contract: When `copy=False` with a sparse input, the alpha rescaling and the in-place reachability update MUST be behind a `UserWarning` at the API boundary, matching the pattern used elsewhere in the file
instances: single-instance

### F13 — `remap_single_linkage_tree` crashes when `tree` is empty
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:384-387 — unconditionally indexes `tree[tree.shape[0] - 1]` to compute `last_cluster_id`/`last_cluster_size`; if the caller-provided `tree` is length 0 this reads index `-1` (numpy wraps but on a zero-length structured array raises `IndexError`)
scenario: "Degenerate edge case where `_single_linkage_tree_` is empty (e.g. all-outlier input where the finite-subset collapse yields a zero-row hierarchy) with `metric != 'precomputed'` and non-finite data → `remap_single_linkage_tree` raises an uncaught `IndexError` mid-`fit`, leaving `self.labels_` set to the pre-remap values (internal indices, not raw indices) and `self._single_linkage_tree_` never remapped — a partial-failure state on the estimator"
contract: `remap_single_linkage_tree` MUST validate `len(tree) > 0` and raise a clear `ValueError` before any state on the estimator has been partially mutated by the calling `fit`
instances: single-instance

### F14 — `min_reachability = np.full(n_samples, ...)` never re-shrunk after first iteration in `mst_from_mutual_reachability`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:93-99 — `min_reachability` starts length `n_samples`, but is re-assigned via `min_reachability = np.minimum(left, right)` where `left = min_reachability[label_filter]` and `label_filter` was applied only to `current_labels`. On the first iteration, `label_filter.size == n_samples` matches `min_reachability.size == n_samples`, so it works, but the intent implicitly relies on lockstep shrinking; if the first-iteration invariant ever broke (e.g., `current_node != 0`), the boolean-index would raise a length-mismatch.
scenario: "Future refactor changes the initial `current_node` value → `label_filter` on the first pass has size != `min_reachability.size`, producing `IndexError` at runtime rather than a compile-time check."
contract: Initialise `min_reachability` explicitly as `np.full(n_samples, np.infty)` and after the first iteration select via `min_reachability[label_filter]` only when sizes are guaranteed to match; add a length assertion at the top of the loop.
instances: single-instance

### F15 — `_dense_mutual_reachability_graph` writes to `distance_matrix` without a symmetric `nogil` guard on Python-object refcount
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — the `with nogil:` block reads/writes `distance_matrix[i, j]` via a typed memoryview; the `core_distances` array is a Python-managed numpy allocation created at line 133-137 outside `nogil`. There is no protection against GC/reallocation; because this is single-threaded there is no data race, but the `TODO: Update w/ prange` comment invites a future maintainer to enable OMP parallelism, which as written would create a race on `distance_matrix[i,j]`/`distance_matrix[j,i]` since both are written by different `i`.
scenario: "Maintainer replaces `range(n_samples)` with `prange(n_samples)` per the TODO → concurrent writes to symmetric entries `distance_matrix[i,j]` and `distance_matrix[j,i]` (which each write the same slot from opposite (i,j)) create a race, and the loop over `for j in range(n_samples)` (not `range(i, n_samples)`) redundantly writes each entry twice, doubling the race surface."
contract: Before adding `prange`, restrict the inner loop to `j in range(i, n_samples)` and mirror-write both symmetric entries, so parallelisation is safe.
instances: single-instance

### F16 — `_hdbscan_brute` sparse-format normalisation happens after `distance_matrix /= alpha`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:241 executes `distance_matrix /= alpha`; sklearn/cluster/_hdbscan/hdbscan.py:244-247 subsequently checks `if issparse(distance_matrix) and distance_matrix.format != "csr": distance_matrix = distance_matrix.tocsr()`. For a non-CSR sparse input (e.g., LIL, which is in `accept_sparse`), the in-place `/= alpha` runs on a LIL matrix (supported but slow, and it *does* copy into COO internally in some scipy versions), potentially producing a *new* object and then the subsequent `tocsr()` runs on that new object; the tocsr result is then not the same buffer as `distance_matrix` originally referenced.
scenario: "User passes a LIL precomputed distance matrix with `copy=False` → `distance_matrix /= alpha` may return a new array (scipy semantics for `__itruediv__` on non-CSR/CSC vary by version), so the `copy=False` contract silently becomes copy-semantics, and vice-versa for CSR."
contract: Convert `distance_matrix` to CSR (with explicit `.copy()` if `copy=True`) **before** the `/= alpha` step, so alpha scaling and downstream in-place mutual-reachability run on a known format and known ownership.
instances: single-instance

### F17 — `_do_labelling` scalar/array truthiness on `parent_lambda >= threshold` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498-506 — `parent_lambda = lambda_array[child_array == n]` is a 1-D numpy slice (possibly length 0 or >1). The subsequent `if parent_lambda >= threshold:` is then a `DeprecationWarning`-eligible truth-value-of-array test (raises `ValueError` if size != 1).
scenario: "Single-cluster branch reached with a point `n` that appears zero times or more than once as a child in the condensed tree → `ValueError: The truth value of an array with more than one element is ambiguous` bubbles up from an internal Cython routine instead of a clear diagnostic; on numpy futures, even the size-1 path may warn."
contract: Extract a scalar explicitly (`parent_lambda = float(lambda_array[child_array == n][0])`) and defensively handle the zero-match case with a raised `ValueError` describing the tree inconsistency.
instances: single-instance

### F18 — `_do_labelling` KeyError when a component's root ID is missing from `cluster_label_map` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-508 — for each point, `cluster = union_find.find(n)`; if `cluster != root_cluster`, the code unconditionally does `label = cluster_label_map[cluster]`. `cluster_label_map` is built at hdbscan.py caller side from `sorted(list(clusters))`, but nothing guarantees that every root a point can be unioned to is present in `clusters`. When a point ends up in a union-find component whose root is neither `root_cluster` nor a selected cluster, this KeyError aborts labeling mid-loop, leaving `self.labels_` unset.
scenario: "A malformed condensed tree (e.g., empty `clusters` set from `_get_clusters` in a corner case with an all-noise leaf method result) → `_do_labelling` raises `KeyError` at line 494 → partial `self.labels_` never gets returned; the exception surfaces out of `fit` without cleanup of the intermediate `self._single_linkage_tree_`."
contract: `_do_labelling` must fall back to `NOISE` (or raise a well-typed ValueError) when the discovered cluster root is not present in `cluster_label_map`, so that labeling cannot leak a bare KeyError.
instances: single-instance

### F19 — `_condense_tree` `ignore` array can be indexed out of range
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148-231 — `node_list = bfs_from_hierarchy(hierarchy, root)`; `ignore = np.zeros(len(node_list), dtype=bool)`. Later `ignore[sub_node] = True` (lines 205, 212, 221, 230) writes at indices that come from `bfs_from_hierarchy` — which returns raw node ids in the range `[0, 2*n_samples)` — not offsets into `node_list`. Only when the BFS is a complete walk of every node id in that range does `len(node_list) == 2*n_samples`; otherwise writes at index >= `len(node_list)` raise IndexError. The `ignore[node]` read at line 164 has the same problem.
scenario: "A hierarchy whose BFS from root does not visit every id in `[0, 2*n_samples)` (any real hierarchy with size < 2*n_samples nodes reachable, e.g., due to relabeling gaps) → `len(node_list) < 2*n_samples` → `ignore[sub_node]` writes past the end → IndexError inside `_condense_tree`, aborting `fit` with a partial estimator."
contract: Size `ignore` by the maximum node id that can appear in the hierarchy (`2 * hierarchy.shape[0] + 1`), not by `len(node_list)`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:161, sklearn/cluster/_hdbscan/_tree.pyx:164, sklearn/cluster/_hdbscan/_tree.pyx:205, sklearn/cluster/_hdbscan/_tree.pyx:212, sklearn/cluster/_hdbscan/_tree.pyx:221, sklearn/cluster/_hdbscan/_tree.pyx:230]

### F20 — Dense-precomputed path with `algorithm="auto"` never validates symmetry
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:741-751, 804-819 — for dense precomputed inputs, symmetry/square checks live inside `_hdbscan_brute` (lines 222-234). When `algorithm="auto"` chose `_hdbscan_brute` (which it does for precomputed since `"precomputed" not in FAST_METRICS`), the checks run — but the fit-level branch that handles dense precomputed at line 747 also permits `np.inf`. `_hdbscan_brute` then divides by alpha and passes to `mutual_reachability_graph`; the resulting `distance_matrix /= alpha` at line 241 is executed on the user's array when `copy=False` (which is the effective default in auto mode per F2), silently mutating the user's precomputed matrix even in the auto path. There is no error path that detects and reports this in-place mutation.
scenario: "User passes a dense precomputed distance matrix with default `algorithm='auto'` and `copy=False` → the auto branch dispatches to `_hdbscan_brute` without `copy` in kwargs → `_hdbscan_brute` runs `distance_matrix /= alpha` on the original array, silently corrupting the caller's matrix on subsequent uses; no warning is emitted."
contract: When dispatching to `_hdbscan_brute` from any branch (auto or explicit) with `metric='precomputed'`, `copy` must be forwarded so the caller's promised `copy=True` is honored; if `copy=False`, the doc must acknowledge that dense precomputed inputs are mutated in-place (currently docs at lines 521-527 imply copy only applies when explicit brute is chosen).
instances: single-instance
