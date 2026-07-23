Acknowledged. Confirming my finding: on line 855, when non-finite data exists, `X` is the reduced X (line 733) but `self.labels_` has been remapped to `_raw_data.shape[0]`. So `mask = self.labels_ == idx` (shape n_raw) doesn't match X (shape n_finite). This would raise IndexError or produce garbage.

Actually wait — `mask` shape is n_raw, but `X.shape[0]` is n_finite. `X[mask]` where mask.shape > X.shape[0] would raise IndexError. This is a real error path issue.

Also, `strength = self.probabilities_[mask]` — same issue.

Let me now check whether the connected_components check in `_brute_mst` swallows errors:

Also note: at `_brute_mst`, the return type from `csgraph.connected_components` with `return_labels=False` — this returns an int. Fine.

Now checking `remap_single_linkage_tree` — line 383: `outlier_tree = np.zeros(len(non_finite), dtype=HIERARCHY_dtype)`. `non_finite` is passed as a `set` (see line 838). `len(non_finite)` works. Then `for i, outlier in enumerate(non_finite):` iterates a set with unspecified order — outlier tree is built with non-deterministic outlier order but this may be fine.

Let me now compile my final findings.

### F1 — `store_centers` with non-finite data raises IndexError due to shape mismatch
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-909 — `if self.store_centers: self._weighted_cluster_center(X)` is invoked after `self.labels_` was replaced with a full-size array indexed by `_raw_data.shape[0]` (lines 840-844), yet the `X` passed in was reduced to the finite subset at line 733 (`X = X[finite_index]`). Inside `_weighted_cluster_center`, `mask = self.labels_ == idx` has shape `n_raw`, but `X.shape[0] == n_finite < n_raw`, so `data = X[mask]` triggers `IndexError: boolean index did not match indexed array` — or (if same length by coincidence) silently returns misaligned data.
scenario: "User calls `HDBSCAN(store_centers='centroid').fit(X_with_nan)` → `_weighted_cluster_center` executes `X[mask]` where mask length exceeds X length → IndexError raised late in fit; user's estimator is left with partial state (`labels_`, `probabilities_`, `_single_linkage_tree_` written but `centroids_` unset)."
contract: When any non-finite samples were removed, `_weighted_cluster_center` must be called with the full `self._raw_data` (or the labels must be projected back to the finite subset before the mask compare) so the mask and feature array agree in the first dimension.
instances: single-instance

### F2 — `algorithm="auto"` silently drops `copy=True` for precomputed inputs
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:804-819 — the "auto" branch dispatches to `_hdbscan_brute` at line 807 (`if issparse(X) or self.metric not in FAST_METRICS`) but never sets `kwargs["copy"] = self.copy`. In the explicit `algorithm="brute"` branch at line 795 the copy flag is honored. `_hdbscan_brute` defaults `copy=False` at line 164, so with `metric="precomputed"` and `algorithm="auto"` the user-provided distance matrix is modified in place at lines 241/251 despite `copy=True`.
scenario: "User calls `HDBSCAN(metric='precomputed', copy=True).fit_predict(D)` (leaving algorithm at default `'auto'`) → the auto path picks `_hdbscan_brute` without forwarding `copy` → D is mutated in-place (divided by alpha, overwritten with mutual-reachability values) → user's `D` is silently corrupted."
contract: The `algorithm="auto"` branch must set `kwargs["copy"] = self.copy` whenever it dispatches to `_hdbscan_brute`, matching the explicit `algorithm="brute"` branch.
instances: single-instance

### F3 — Sparse mutual-reachability leaves stale value in place when result is infinite and `max_distance <= 0` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:206-212 — after computing `mutual_reachibility_distance = max(core_distances[row_ind], core_distances[col_ind], data[i])`, the code writes back only when the result is finite, or when `max_distance > 0`. When the mutual reachability is infinite and `max_distance == 0` (the default at hdbscan.py:243), `data[i]` retains its original value (not the newly-computed infinite one). Downstream `_brute_mst` then runs `csgraph.connected_components` and `minimum_spanning_tree` on a graph whose weights do NOT reflect the mutual-reachability semantics for the affected edges, producing an MST whose weights understate the true infinite mutual reachability.
scenario: "User supplies a sparse precomputed distance matrix with a row that has fewer than `min_samples` stored neighbors (so `core_distances[row_ind] = INFINITY`), and does not specify `max_distance` → mutual reachability along that row is infinite but `data[i]` stays at the original finite distance → `_brute_mst` produces an MST that silently uses those stale distances instead of raising, and the 'contains edge weights with value infinity' warning at hdbscan.py:256 does not fire because the infinity was suppressed."
contract: When `mutual_reachibility_distance` is non-finite and `max_distance <= 0`, `data[i]` must be set to `INFINITY` (or the entry otherwise flagged) so that downstream MST / connectivity checks see the true value; silently retaining the pre-existing entry hides a data condition that the code explicitly warns about elsewhere.
instances: single-instance

### F4 — `_do_labelling` KeyError when a component's root ID is missing from `cluster_label_map` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:490-508 — for each point, `cluster = union_find.find(n)`; if `cluster != root_cluster`, the code unconditionally does `label = cluster_label_map[cluster]`. `cluster_label_map` is built at hdbscan.py caller side from `sorted(list(clusters))`, but nothing guarantees that every root a point can be unioned to is present in `clusters`. When a point ends up in a union-find component whose root is neither `root_cluster` nor a selected cluster, this KeyError aborts labeling mid-loop, leaving `self.labels_` unset.
scenario: "A malformed condensed tree (e.g., empty `clusters` set from `_get_clusters` in a corner case with an all-noise leaf method result) → `_do_labelling` raises `KeyError` at line 494 → partial `self.labels_` never gets returned; the exception surfaces out of `fit` without cleanup of the intermediate `self._single_linkage_tree_`."
contract: `_do_labelling` must fall back to `NOISE` (or raise a well-typed ValueError) when the discovered cluster root is not present in `cluster_label_map`, so that labeling cannot leak a bare KeyError.
instances: single-instance

### F5 — `_condense_tree` `ignore` array can be indexed out of range
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:148-231 — `node_list = bfs_from_hierarchy(hierarchy, root)`; `ignore = np.zeros(len(node_list), dtype=bool)`. Later `ignore[sub_node] = True` (lines 205, 212, 221, 230) writes at indices that come from `bfs_from_hierarchy` — which returns raw node ids in the range `[0, 2*n_samples)` — not offsets into `node_list`. Only when the BFS is a complete walk of every node id in that range does `len(node_list) == 2*n_samples`; otherwise writes at index >= `len(node_list)` raise IndexError. The `ignore[node]` read at line 164 has the same problem.
scenario: "A hierarchy whose BFS from root does not visit every id in `[0, 2*n_samples)` (any real hierarchy with size < 2*n_samples nodes reachable, e.g., due to relabeling gaps) → `len(node_list) < 2*n_samples` → `ignore[sub_node]` writes past the end → IndexError inside `_condense_tree`, aborting `fit` with a partial estimator."
contract: Size `ignore` by the maximum node id that can appear in the hierarchy (`2 * hierarchy.shape[0] + 1`), not by `len(node_list)`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:161, sklearn/cluster/_hdbscan/_tree.pyx:164, sklearn/cluster/_hdbscan/_tree.pyx:205, sklearn/cluster/_hdbscan/_tree.pyx:212, sklearn/cluster/_hdbscan/_tree.pyx:221, sklearn/cluster/_hdbscan/_tree.pyx:230]

### F6 — Dense-precomputed path with `algorithm="auto"` never validates symmetry
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:741-751, 804-819 — for dense precomputed inputs, symmetry/square checks live inside `_hdbscan_brute` (lines 222-234). When `algorithm="auto"` chose `_hdbscan_brute` (which it does for precomputed since `"precomputed" not in FAST_METRICS`), the checks run — but the fit-level branch that handles dense precomputed at line 747 also permits `np.inf`. `_hdbscan_brute` then divides by alpha and passes to `mutual_reachability_graph`; the resulting `distance_matrix /= alpha` at line 241 is executed on the user's array when `copy=False` (which is the effective default in auto mode per F2), silently mutating the user's precomputed matrix even in the auto path. There is no error path that detects and reports this in-place mutation.
scenario: "User passes a dense precomputed distance matrix with default `algorithm='auto'` and `copy=False` → the auto branch dispatches to `_hdbscan_brute` without `copy` in kwargs → `_hdbscan_brute` runs `distance_matrix /= alpha` on the original array, silently corrupting the caller's matrix on subsequent uses; no warning is emitted."
contract: When dispatching to `_hdbscan_brute` from any branch (auto or explicit) with `metric='precomputed'`, `copy` must be forwarded so the caller's promised `copy=True` is honored; if `copy=False`, the doc must acknowledge that dense precomputed inputs are mutated in-place (currently docs at lines 521-527 imply copy only applies when explicit brute is chosen).
instances: single-instance
