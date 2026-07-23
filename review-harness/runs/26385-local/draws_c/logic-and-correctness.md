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

### F3 — Docstring/default mismatch for `HDBSCAN.n_jobs`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` and "`None` means 1"; sklearn/cluster/_hdbscan/hdbscan.py:658 sets the actual `__init__` default to `n_jobs=4`.
scenario: "user reads the docstring and expects single-threaded behavior by default → HDBSCAN silently uses 4 workers, contradicting the documented default and sklearn convention"
contract: default `n_jobs=None` (matching the docstring and sklearn convention).
instances: single-instance

### F4 — `remap_single_linkage_tree` iterates over a Python `set`, producing non-deterministic outlier tree
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` (a `set`) into `remap_single_linkage_tree`. sklearn/cluster/_hdbscan/hdbscan.py:388 `for i, outlier in enumerate(non_finite):` iterates a `set` and writes `outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`; `set` iteration order is not guaranteed reproducible across processes / PYTHONHASHSEED values, so `self._single_linkage_tree_` ordering of the appended outlier edges is non-deterministic.
scenario: "user fits HDBSCAN twice on identical data with non-finite rows in separate processes and inspects `self._single_linkage_tree_` → sees different orderings of the appended outlier subtree between runs"
contract: pass and iterate a sorted, ordered container (e.g. `np.sort(np.unique(np.concatenate([infinite_index, missing_index])))`) so the appended outlier subtree is deterministic.
instances: single-instance

### F5 — `_hdbscan_brute` default `alpha=None` will crash on `distance_matrix /= alpha`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None` in the signature, but sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally runs `distance_matrix /= alpha` (no None check). `ndarray /= None` raises `TypeError`. The companion `_hdbscan_prims` at line 273 correctly defaults `alpha=1.0`.
scenario: "any direct call to `_hdbscan_brute(X)` without an explicit `alpha` → immediate `TypeError` at line 241"
contract: default `alpha=1.0` (matching `_hdbscan_prims` and the docstring at line 183).
instances: single-instance

### F6 — Sparse core-distance semantics differ from dense (self-distance handling)
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133-137 (dense) selects `partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx]` over the full row including the implicit `0` self-distance, i.e. the `(further_neighbor_idx)`-th smallest counting self at position 0. sklearn/cluster/_hdbscan/_reachability.pyx:193-200 (sparse) selects `partition(row_data, further_neighbor_idx)[further_neighbor_idx]` over `data[indptr[i]:indptr[i+1]]`, which excludes any zero self-entry that was `eliminate_zeros()`d (standard for CSR distance matrices). Result: for the same underlying distances, the sparse path effectively uses the k-th nearest neighbor while the dense path uses the (k-1)-th, an off-by-one.
scenario: "user supplies a precomputed CSR distance matrix with implicit zero diagonal → HDBSCAN produces a different clustering than the equivalent dense matrix with the same explicit distances"
contract: set the dense diagonal to `np.inf` before partitioning so both paths exclude the self-entry consistently.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:196]

### F7 — Sparse `_get_finite_row_indices` misses non-finite structural patterns
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:401-404 iterates `matrix.tolil().data` and calls `np.all(np.isfinite(row))`, but LIL `.data` only exposes explicitly stored non-zero values. Rows in which the sparse pattern encodes missing distances via structural absence (rather than explicit `np.inf` / `np.nan`) will always be classified as fully finite; conversely, rows containing only structural zeros (empty `row`) also pass `np.all(np.isfinite([])) == True`.
scenario: "user constructs a sparse distance matrix that omits distances beyond some threshold (typical KNN graph) → `_get_finite_row_indices` reports every row as finite, so no rows are separated out for the non-finite handling path even when the caller expected that behavior"
contract: derive per-row non-finiteness from the CSR triplet directly, e.g. `finite_rows = np.array([np.all(np.isfinite(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])) for i in range(matrix.shape[0])])`.
instances: single-instance

### F8 — `_do_labelling` single-cluster branch depends on `parent_lambda` being a scalar-like 1-element array
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 `parent_lambda = lambda_array[child_array == n]` returns an ndarray; sklearn/cluster/_hdbscan/_tree.pyx:505 `if parent_lambda >= threshold:` implicitly converts to scalar bool only when the array has exactly one element. The inline comment at lines 496-497 asserts "There can only be one edge with this particular child" — but there is no assertion or guard enforcing it. If a malformed condensed tree ever has duplicate child rows (or `n` is inadvertently referenced as a cluster), `if` on a multi-element boolean array raises `ValueError: The truth value of an array with more than one element is ambiguous`.
scenario: "any hierarchy pathology (duplicated child row, or invocation with n that matches multiple entries) → cryptic ValueError from ambiguous-truth-value evaluation instead of a domain-specific error"
contract: extract the scalar explicitly (`parent_lambda = lambda_array[child_array == n][0]`) and assert the single-edge invariant before comparison.
instances: single-instance
