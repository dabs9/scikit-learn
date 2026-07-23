### F1 — Duplicate `births` allocation wastes O(n) fill in condensed-tree stability
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-256 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed on two consecutive lines with no intervening use of the first array; the first allocation is discarded and immediately re-created before the write loop.
scenario: "Every call to `_compute_stability` (once per `fit`) → an extra `n`-sized `np.full` allocation and NaN-fill is performed and thrown away, doubling this bookkeeping cost on the hot fitting path."
contract: Delete the duplicate `births = np.full(...)` on line 252 so `_compute_stability` allocates and fills the `births` array exactly once.
instances: single-instance

### F2 — Dense mutual-reachability graph writes both halves of a symmetric matrix
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:139-149 — the `nogil` double loop iterates `for i in range(n_samples): for j in range(n_samples):` and writes `distance_matrix[i, j]` for every ordered pair, even though the reachability distance `max(core[i], core[j], d[i, j])` is symmetric in `(i, j)` and the input matrix is documented as symmetric (line 130-132).
scenario: "Brute-path fit with dense pairwise distances → the O(n²) update loop performs redundant work for the lower triangle, roughly doubling the hot-path cost of `mutual_reachability_graph` for every dense brute run."
contract: Iterate only `j in range(i, n_samples)` in the update loop and mirror the write to `distance_matrix[j, i]`, so each unordered pair is computed exactly once.
instances: single-instance

### F3 — `_get_finite_row_indices` densifies sparse input via `.tolil()` and iterates rows in Python
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 — for sparse input the function does `[i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]`, which allocates a LIL representation (list-of-lists per row) and then loops in Python with a per-row `np.isfinite` call.
scenario: "Sparse feature matrix with any non-finite entry (triggers `all_finite=False`) → the finite-row filter walks the sparse matrix row-by-row in the Python interpreter with a full LIL conversion, scaling as O(n_rows) Python iterations plus an intermediate LIL copy of the entire matrix, when a single vectorized `np.isfinite(matrix.data)` combined with `np.add.reduceat(..., indptr[:-1])` (or equivalent on CSR structure) would give the same result in a single C-level pass."
contract: On the sparse branch, work directly on the CSR `data`/`indptr` arrays with a vectorised `np.isfinite`+`reduceat` (or `matrix.getnnz(axis=1)` + row-wise all-finite test) rather than converting to LIL and iterating rows in Python.
instances: single-instance

### F4 — EOM cluster selection performs O(|cluster_tree|²) linear scans over the condensed tree [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:727-739 — inside `_get_clusters`, `for node in node_list:` iterates over every stability key and evaluates `child_selection = (cluster_tree['parent'] == node)` on each iteration; this is an O(|cluster_tree|) full boolean scan for each of the ~|cluster_tree| nodes, giving overall O(|cluster_tree|²) work.
scenario: "Fitting on a dataset producing a large condensed cluster tree (many merges, e.g. tens of thousands of internal nodes) → the EOM subtree-stability accumulation degenerates to quadratic time in cluster-tree size, dominating fit time even though a single O(|cluster_tree|) group-by over `cluster_tree['parent']` (e.g., `np.add.reduceat` on parents sorted by parent id, or an `np.bincount` over `parent - offset`) would produce the same per-node subtree stability."
contract: Precompute the per-parent stability sum in a single pass (`np.bincount(cluster_tree['parent'] - offset, weights=child_stabilities)` or an equivalent group-by) and index into it during the `node_list` loop instead of scanning `cluster_tree['parent'] == node` for every node.
instances: single-instance

### F5 — `HDBSCAN.__init__` defaults `n_jobs=4`, contradicting its own docstring and scikit-learn convention [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:647-658 — the constructor signature reads `n_jobs=4`, while the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None` with "`None` means 1 unless in a :obj:`joblib.parallel_backend` context." The value flows straight into `pairwise_distances(..., n_jobs=n_jobs, ...)` at line 239 and `NearestNeighbors(..., n_jobs=n_jobs, ...)` at line 338.
scenario: "User instantiates `HDBSCAN()` on a small input → 4 worker threads/processes are silently spun up in `pairwise_distances`/`NearestNeighbors` for every fit, causing thread-startup overhead to dominate on small problems and producing behavior the documented default (`None`, meaning 1) forbids."
contract: Change the constructor default to `n_jobs=None` so runtime behavior matches the documented default and the standard scikit-learn convention for `n_jobs`.
instances: single-instance
