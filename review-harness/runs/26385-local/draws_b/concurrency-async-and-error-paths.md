### F1 — Shape mismatch in `_weighted_cluster_center` after non-finite pruning
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 pares `X = X[finite_index]`; sklearn/cluster/_hdbscan/hdbscan.py:840-844 rebuilds `self.labels_` to `self._raw_data.shape[0]` (raw length); sklearn/cluster/_hdbscan/hdbscan.py:854-855 then calls `self._weighted_cluster_center(X)` passing the pruned `X`; sklearn/cluster/_hdbscan/hdbscan.py:908 does `mask = self.labels_ == idx` (raw length) and sklearn/cluster/_hdbscan/hdbscan.py:909 does `data = X[mask]` on the pruned X.
scenario: "User fits HDBSCAN with `store_centers=\"centroid\"|\"medoid\"|\"both\"` on data containing any `np.nan`/`np.inf` row → boolean-mask/array shape mismatch (or silently wrong-indexed rows) when computing centers, raising `IndexError`/producing garbage centroids."
contract: When `all_finite` is False, pass `self._raw_data` (or an aligned finite subset paired with re-indexed labels/probabilities) into `_weighted_cluster_center` so that `mask.shape == X.shape[0]`.
instances: single-instance

### F2 — `_weighted_cluster_center` counts missing-label (-3) samples as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the class-level `_OUTLIER_ENCODING` (sklearn/cluster/_hdbscan/hdbscan.py:65-79) defines `missing` label `-3`, which is excluded from `labels_` by sklearn/cluster/_hdbscan/hdbscan.py:843.
scenario: "User calls fit with `store_centers` set and X contains `np.nan` rows → `-3` is included in `n_clusters`, sizing `self.centroids_`/`self.medoids_` one row too large; the loop `for idx in range(n_clusters)` then produces an all-NaN or empty-mean row (empty slice warning + NaN centroid) and never touches the -3 mask, corrupting the exported centers."
contract: Exclude all three outlier encodings when counting non-noise clusters: `n_clusters = len(set(self.labels_) - {-1, -2, -3})`.
instances: single-instance

### F3 — `_assert_all_finite(X.data)` on LIL sparse input misclassifies finiteness
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:703 accepts `accept_sparse=["csr", "lil"]`; sklearn/cluster/_hdbscan/hdbscan.py:710 calls `_assert_all_finite(X.data if issparse(X) else X)`. For LIL sparse, `X.data` is an `object` ndarray of Python lists (one per row), not a numeric buffer.
scenario: "User passes a LIL sparse feature matrix containing `np.nan`/`np.inf` → `_assert_all_finite` receives an object array and either (a) raises an unrelated dtype error, or (b) silently returns True because the object-dtype path does not descend into the per-row lists, sending non-finite data into `_hdbscan_brute` (fail-open path)."
contract: Materialise a numeric buffer before finiteness checking for sparse formats other than CSR (e.g., convert with `X.tocsr()` before the `_assert_all_finite` call), rather than relying on `X.data` which is format-dependent.
instances: single-instance

### F4 — Ordering of `non_finite` iteration is nondeterministic
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 constructs `non_finite=set(infinite_index + missing_index)`; sklearn/cluster/_hdbscan/hdbscan.py:388 iterates `for i, outlier in enumerate(non_finite)` treating `non_finite` as an ordered sequence.
scenario: "Two runs with identical inputs containing multiple non-finite rows → `_single_linkage_tree_` layout of the appended outlier subtree (parent ids, cluster sizes) differs between runs due to Python set iteration order, breaking result reproducibility for consumers that inspect `_single_linkage_tree_` or call `dbscan_clustering` and compare across runs."
contract: Pass a stably-ordered container (e.g., `sorted(set(infinite_index + missing_index))`) into `remap_single_linkage_tree` so that outlier remap is deterministic.
instances: single-instance

### F5 — `_brute_mst` sparse path silently accepts disconnected graphs when `connected_components` count equals 1 but MST is short
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:111-123 raises only when `csgraph.connected_components(...) > 1`; sklearn/cluster/_hdbscan/hdbscan.py:126-131 builds `mst` via `np.core.records.fromarrays([rows, cols, ...])` of whatever nonzeros scipy returns. `scipy.sparse.csgraph.minimum_spanning_tree` on an already-symmetrized MR graph with zero-valued edges (which are dropped as "no edge") can return fewer than `n_samples - 1` edges.
scenario: "User supplies a sparse precomputed distance matrix where the mutual-reachability graph is connected under the sparsity pattern but some MR distances collapse to zero (e.g., duplicate points) → returned MST has fewer than n-1 edges, `make_single_linkage` produces an incomplete/malformed `HIERARCHY_dtype` array, and downstream `_condense_tree` yields wrong labels rather than surfacing the partial-failure state."
contract: After `csgraph.minimum_spanning_tree`, assert `len(rows) == n_samples - 1` and raise a clear `ValueError` describing the partial MST rather than continuing with a truncated tree.
instances: single-instance

### F6 — Precomputed sparse path can mutate user's input array in place
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:734-740 (`elif issparse(X):` — no `copy=` argument to `_validate_data`); sklearn/cluster/_hdbscan/hdbscan.py:241 in `_hdbscan_brute`: `distance_matrix /= alpha` and sklearn/cluster/_hdbscan/_reachability.pyx:56 documents "Note that all computations are done in-place." The CSR branch in `_reachability.pyx:202-212` writes into `data`. The `copy` kwarg is threaded only for `algorithm=="brute"` (sklearn/cluster/_hdbscan/hdbscan.py:795) and `_hdbscan_brute` only honors it in the dense precomputed branch (sklearn/cluster/_hdbscan/hdbscan.py:222-236); the sparse precomputed branch ignores `copy` entirely.
scenario: "User passes a sparse CSR precomputed distance matrix with `HDBSCAN(metric='precomputed', copy=True).fit(D)` → `D.data` is mutated by `distance_matrix /= alpha` and by `_sparse_mutual_reachability_graph`, silently overwriting the caller's matrix; the `copy=True` documented guarantee is fail-open."
contract: In `_hdbscan_brute`, when `metric == 'precomputed'` and the input is sparse, honor `copy` by materialising a new CSR (e.g., `distance_matrix = X.copy()`) before any in-place mutation.
instances: single-instance

### F7 — `remap_single_linkage_tree` `outlier_count` shift under-adjusts internal ids
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:369 sets `outlier_count = len(non_finite)`; sklearn/cluster/_hdbscan/hdbscan.py:377/381 shift internal cluster ids by `outlier_count`. `non_finite` is a `set` produced from `set(infinite_index + missing_index)` (sklearn/cluster/_hdbscan/hdbscan.py:838), so overlapping indices collapse; the correct number of "raw" sample slots that need to be reserved is the total raw sample count minus the finite count, not the size of the deduplicated set.
scenario: "User fits with a row that is *both* `np.inf` and `np.nan` in different columns (produces an entry in both `infinite_index` and `missing_index`) → `outlier_count` is smaller than the actual gap between finite indices and the id space needed, causing internal-node ids in the remapped hierarchy to collide with raw sample ids, silently mislabeling points in `_single_linkage_tree_`."
contract: Set `outlier_count = self._raw_data.shape[0] - finite_count` (or equivalently, count the union of infinite_index and missing_index without a set-of-tuples de-dup that under-counts) when remapping.
instances: single-instance

### F8 — `min_reachability = np.full(n_samples, ...)` never re-shrunk after first iteration in `mst_from_mutual_reachability`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:93-99 — `min_reachability` starts length `n_samples`, but is re-assigned via `min_reachability = np.minimum(left, right)` where `left = min_reachability[label_filter]` and `label_filter` was applied only to `current_labels`. On the first iteration, `label_filter.size == n_samples` matches `min_reachability.size == n_samples`, so it works, but the intent implicitly relies on lockstep shrinking; if the first-iteration invariant ever broke (e.g., `current_node != 0`), the boolean-index would raise a length-mismatch.
scenario: "Future refactor changes the initial `current_node` value → `label_filter` on the first pass has size != `min_reachability.size`, producing `IndexError` at runtime rather than a compile-time check."
contract: Initialise `min_reachability` explicitly as `np.full(n_samples, np.infty)` and after the first iteration select via `min_reachability[label_filter]` only when sizes are guaranteed to match; add a length assertion at the top of the loop.
instances: single-instance

### F9 — `_dense_mutual_reachability_graph` writes to `distance_matrix` without a symmetric `nogil` guard on Python-object refcount
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — the `with nogil:` block reads/writes `distance_matrix[i, j]` via a typed memoryview; the `core_distances` array is a Python-managed numpy allocation created at line 133-137 outside `nogil`. There is no protection against GC/reallocation; because this is single-threaded there is no data race, but the `TODO: Update w/ prange` comment invites a future maintainer to enable OMP parallelism, which as written would create a race on `distance_matrix[i,j]`/`distance_matrix[j,i]` since both are written by different `i`.
scenario: "Maintainer replaces `range(n_samples)` with `prange(n_samples)` per the TODO → concurrent writes to symmetric entries `distance_matrix[i,j]` and `distance_matrix[j,i]` (which each write the same slot from opposite (i,j)) create a race, and the loop over `for j in range(n_samples)` (not `range(i, n_samples)`) redundantly writes each entry twice, doubling the race surface."
contract: Before adding `prange`, restrict the inner loop to `j in range(i, n_samples)` and mirror-write both symmetric entries, so parallelisation is safe.
instances: single-instance

### F10 — Sparse `_sparse_mutual_reachability_graph` fail-open when `max_distance == 0` for infinite reachability
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:209-212 — after computing `mutual_reachibility_distance`, if it is not finite the code writes `max_distance` only when `max_distance > 0`. Otherwise the original (potentially finite) `data[i]` value is left untouched, so an infinite mutual-reachability distance is silently *not* recorded as infinite in the sparse output.
scenario: "User passes a sparse precomputed distance matrix such that some point has fewer than `min_samples` neighbours (giving `core_distances[i] = INFINITY` at line 200) and does not pass a `max_distance` in `metric_params` (default 0.0 at sklearn/cluster/_hdbscan/hdbscan.py:243) → the infinite MR value is silently dropped; the downstream `connected_components` check at hdbscan.py:111 may still see the graph as connected and proceed, producing a clustering that treats those points as if their reachability were the *original stored distance* rather than infinity, a fail-open failure mode."
contract: When `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, raise a `ValueError` mirroring the dense-path expectation rather than silently keeping the stale finite value.
instances: single-instance

### F11 — `_hdbscan_brute` sparse-format normalisation happens after `distance_matrix /= alpha`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:241 executes `distance_matrix /= alpha`; sklearn/cluster/_hdbscan/hdbscan.py:244-247 subsequently checks `if issparse(distance_matrix) and distance_matrix.format != "csr": distance_matrix = distance_matrix.tocsr()`. For a non-CSR sparse input (e.g., LIL, which is in `accept_sparse`), the in-place `/= alpha` runs on a LIL matrix (supported but slow, and it *does* copy into COO internally in some scipy versions), potentially producing a *new* object and then the subsequent `tocsr()` runs on that new object; the tocsr result is then not the same buffer as `distance_matrix` originally referenced.
scenario: "User passes a LIL precomputed distance matrix with `copy=False` → `distance_matrix /= alpha` may return a new array (scipy semantics for `__itruediv__` on non-CSR/CSC vary by version), so the `copy=False` contract silently becomes copy-semantics, and vice-versa for CSR."
contract: Convert `distance_matrix` to CSR (with explicit `.copy()` if `copy=True`) **before** the `/= alpha` step, so alpha scaling and downstream in-place mutual-reachability run on a known format and known ownership.
instances: single-instance

### F12 — `_do_labelling` scalar/array truthiness on `parent_lambda >= threshold` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498-506 — `parent_lambda = lambda_array[child_array == n]` is a 1-D numpy slice (possibly length 0 or >1). The subsequent `if parent_lambda >= threshold:` is then a `DeprecationWarning`-eligible truth-value-of-array test (raises `ValueError` if size != 1).
scenario: "Single-cluster branch reached with a point `n` that appears zero times or more than once as a child in the condensed tree → `ValueError: The truth value of an array with more than one element is ambiguous` bubbles up from an internal Cython routine instead of a clear diagnostic; on numpy futures, even the size-1 path may warn."
contract: Extract a scalar explicitly (`parent_lambda = float(lambda_array[child_array == n][0])`) and defensively handle the zero-match case with a raised `ValueError` describing the tree inconsistency.
instances: single-instance
