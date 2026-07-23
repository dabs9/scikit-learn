### F1 — `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default and sklearn convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in `__init__` signature; docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents "`None` means 1 unless in a `joblib.parallel_backend` context" implying `n_jobs=None` as default, matching sklearn convention throughout the file (e.g. `_hdbscan_brute` at line 163 uses `n_jobs=None`, `_hdbscan_prims` at line 276 uses `n_jobs=None`)
scenario: "User instantiates `HDBSCAN()` inside a `joblib.parallel_backend` context expecting to inherit the outer backend → HDBSCAN silently forks 4 concurrent processes/threads inside `pairwise_distances`/`NearestNeighbors`, over-subscribing the machine and defeating the parent's parallel policy"
contract: The default MUST be `n_jobs=None` to honor `joblib.parallel_backend` contexts consistent with the docstring and all other sklearn estimators
instances: single-instance

### F2 — `remap_single_linkage_tree` iterates a `set`, producing non-deterministic `_single_linkage_tree_`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` into `remap_single_linkage_tree`; sklearn/cluster/_hdbscan/hdbscan.py:388 iterates that set with `for i, outlier in enumerate(non_finite)` and writes `outlier_tree[i] = (outlier, ...)` in set-iteration order
scenario: "User fits `HDBSCAN` twice on the same data containing both `np.inf` and `np.nan` outliers → `self._single_linkage_tree_` differs across runs (rows reordered) because Python `set` iteration order is not stable across processes, breaking any code that hashes or diffs the tree attribute"
contract: `non_finite` MUST be passed as a sorted, deterministic sequence (e.g. `np.unique(np.concatenate([infinite_index, missing_index]))`) so `_single_linkage_tree_` is reproducible
instances: single-instance

### F3 — Sparse mutual-reachability silently leaves stale distances when `max_distance <= 0`
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:209-212 — `if isfinite(mutual_reachibility_distance): data[i] = mutual_reachibility_distance; elif max_distance > 0: data[i] = max_distance` — the `else` branch (non-finite mutual reachability AND `max_distance == 0` which is the module default set at sklearn/cluster/_hdbscan/hdbscan.py:243) falls through and leaves `data[i]` at its original (pre-update) distance value
scenario: "User supplies a sparse precomputed distance matrix where some row has fewer than `min_samples` non-zero entries but the overall graph is still connected → those rows get `core_distances[i] = INFINITY` (line 200), so `mutual_reachibility_distance` becomes `inf`, but `data[i]` is silently left at its original stale finite value; the subsequent MST built by `csgraph.minimum_spanning_tree` proceeds with corrupted mutual-reachability weights and returns wrong clusters with no warning"
contract: When the computed mutual-reachability distance is infinite and no `max_distance` truncation is configured, the entry MUST be replaced with `INFINITY` so the downstream MST/connected-components step observes the true topology
instances: single-instance

### F4 — `UnionFind` `.pxd`/`.pyx` `noexcept` mismatch swallows exceptions raised in `union`/`fast_find`
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` — neither definition repeats the `noexcept` qualifier that the `.pxd` header pins
scenario: "Cython 3 treats the `.pxd` declaration as authoritative → any Python-level exception raised inside `union` or `fast_find` (e.g. `IndexError` from bounds checks, allocation failures within the memoryview writes) is silently swallowed with only a WriteUnraisable to stderr, and the calling `make_single_linkage` continues with a corrupted UnionFind, producing a garbage single-linkage tree"
contract: The `.pyx` definitions MUST repeat the `noexcept` qualifier from the `.pxd` so the exception-propagation policy is unambiguous and reviewed
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F5 — `_hdbscan_brute` mutates caller's sparse input in place even when `copy=True` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:236 — `distance_matrix = X.copy() if copy else X`; sklearn/cluster/_hdbscan/hdbscan.py:244-247 — if the (now-copied) distance_matrix has a non-CSR format, `distance_matrix = distance_matrix.tocsr()` allocates a new CSR; but `X.copy()` on a sparse matrix preserves format, so if the user passed CSR with `copy=True`, `X.copy()` is CSR and `tocsr()` on a CSR returns `self` (no copy) — then sklearn/cluster/_hdbscan/hdbscan.py:241 (`distance_matrix /= alpha`) and the in-place `mutual_reachability_graph` at line 251 mutate the copy fine. However for `copy=False` + CSR user input, both `/=` and `mutual_reachability_graph` silently mutate the caller's `.data` buffer with no warning even though it is documented
scenario: "User passes a CSR sparse precomputed matrix with `copy=False` (the default) → `X.data /= alpha` and `mutual_reachability_graph` mutate the user's sparse buffer in place, invalidating the input for any further use"
contract: When `copy=False` with a sparse input, the alpha rescaling and the in-place reachability update MUST be behind a `UserWarning` at the API boundary, matching the pattern used elsewhere in the file
instances: single-instance

### F6 — `remap_single_linkage_tree` crashes when `tree` is empty
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:384-387 — unconditionally indexes `tree[tree.shape[0] - 1]` to compute `last_cluster_id`/`last_cluster_size`; if the caller-provided `tree` is length 0 this reads index `-1` (numpy wraps but on a zero-length structured array raises `IndexError`)
scenario: "Degenerate edge case where `_single_linkage_tree_` is empty (e.g. all-outlier input where the finite-subset collapse yields a zero-row hierarchy) with `metric != 'precomputed'` and non-finite data → `remap_single_linkage_tree` raises an uncaught `IndexError` mid-`fit`, leaving `self.labels_` set to the pre-remap values (internal indices, not raw indices) and `self._single_linkage_tree_` never remapped — a partial-failure state on the estimator"
contract: `remap_single_linkage_tree` MUST validate `len(tree) > 0` and raise a clear `ValueError` before any state on the estimator has been partially mutated by the calling `fit`
instances: single-instance
