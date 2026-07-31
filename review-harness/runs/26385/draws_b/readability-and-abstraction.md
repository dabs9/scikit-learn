### F1 — `_weighted_cluster_center` iterates a phantom cluster when `-3` (missing) labels are present, and boolean-indexes a mismatched-length X
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` excludes `-1` and `-2` but not `-3`, while the class docstring at lines 561-562 and 572-573 asserts "the `-1, -2, -3` labels for the outlier clusters are excluded". Additionally, at line 855 `_weighted_cluster_center(X)` is called with the reduced `X` (line 733 assigned `X = X[finite_index]`) while `self.labels_` was replaced at line 844 with a full-size `new_labels` sized to `self._raw_data.shape[0]`. Inside the helper, line 908 `mask = self.labels_ == idx` builds a mask sized to the full raw data, then line 909 does `data = X[mask]` on the reduced X.
scenario: "`HDBSCAN(store_centers='centroid').fit(X)` with any `np.nan` in `X` → `-3` inflates `n_clusters` so the loop reaches `idx` values that never label any sample, and `X[mask]` raises `IndexError: boolean index did not match indexed array along dimension 0` because `len(mask)==raw_n` while `X` has `finite_n` rows"
contract: Compute `n_clusters` from the label set with the full outlier encoding (`set(self.labels_) - {-1, -2, -3}`) and pass the finite-only X together with the finite-only labels_ into `_weighted_cluster_center`.
instances: single-instance

### F2 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented `default=None` and violates the sklearn convention the docstring cites
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — the class docstring reads "`n_jobs : int, default=None ... ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.``-1`` means using all processors.`" ; line 658 sets `n_jobs=4` in the signature, and line 673 stores that value on `self.n_jobs`, which is later passed straight into `NearestNeighbors`/`pairwise_distances`.
scenario: "user reads the docstring and expects the estimator to behave like every other sklearn estimator (respect `joblib.parallel_backend`); instead the constructor spawns 4 processes unconditionally → surprise CPU/memory usage and non-composition with parallel_backend contexts"
contract: Set the default to `n_jobs=None` to match the docstring and sklearn-wide convention.
instances: single-instance

### F3 — `_compute_stability` initializes `births` twice back-to-back (dead first assignment)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — the two adjacent statements `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` (line 252) and `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` (line 254) are identical; the first allocation is immediately overwritten by the second and never read.
scenario: "reader trying to understand the initialization of `births` wonders whether the two lines were meant to differ (e.g., a shape mismatch or a typo hiding a bug) → wasted debugging time and one wasted allocation per call"
contract: Delete line 252 so `births` is initialized exactly once.
instances: single-instance

### F4 — `_hdbscan_prims` docstring documents a `copy` parameter it does not accept and claims support for `metric="precomputed"` it does not implement
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-327 — the signature at line 269 is `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)` (no `copy`); the docstring header at lines 280-283 says "If `metric='precomputed'` then `X` must be a symmetric array of distances. Otherwise, the pairwise distances are calculated directly and passed to `mutual_reachability_graph`."; lines 313-318 document a `copy : bool, default=False` parameter.
scenario: "user reads the docstring, passes `copy=True` (or expects the function to switch on `metric='precomputed'`) → `copy` is silently swallowed into `metric_params` and forwarded into the distance metric, producing an opaque error or, worse, silent incorrect behavior; `metric='precomputed'` reaches this path via `algorithm='kdtree'/'balltree'` and produces a confusing NN failure rather than the promised behavior"
contract: Remove the `copy` parameter block and the `metric='precomputed'` sentence from the docstring so it reflects only what the function actually does (compute core distances via NearestNeighbors, then MST via `mst_from_data_matrix`).
instances: single-instance

### F5 — `remap_single_linkage_tree` docstring lies about `non_finite`'s type (`Boolean array`) and the function's callsite iterates a `set` non-deterministically
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-365 — docstring: "non_finite : ndarray - Boolean array of which entries in the raw data are non-finite"; callsite at line 838 passes `non_finite=set(infinite_index + missing_index)` (a Python set of integer indices); line 388 `for i, outlier in enumerate(non_finite):` iterates that set to write rows into `outlier_tree`, so the row ordering of the appended outlier rows in the linkage tree depends on Python set iteration order rather than a stable index order.
scenario: "reader inspecting the linkage tree or testing across Python versions sees the outlier rows in different orders for the same input, because `non_finite` is a set and the loop writes rows in iteration order → docstring lies about the parameter and the tree contents become surprisingly implementation-dependent"
contract: Change `non_finite` to a sorted 1-D integer array of outlier indices and update the docstring to describe that actual type.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:363, sklearn/cluster/_hdbscan/hdbscan.py:388, sklearn/cluster/_hdbscan/hdbscan.py:838]

### F6 — Multiple docstrings in `_tree.pyx` describe tree arrays as shape `(n_samples,)` when they are `(n_samples - 1,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 says `hierarchy : ndarray of shape (n_samples,)` but line 146 computes `n_samples = hierarchy.shape[0] + 1`; sklearn/cluster/_hdbscan/_tree.pyx:138 says the returned `condensed_tree` is shape `(n_samples,)` though its size is data-dependent; sklearn/cluster/_hdbscan/_tree.pyx:371 says `linkage : ndarray of shape (n_samples,)` but line 397 computes `n_samples = root // 2 + 1` with `root = 2 * linkage.shape[0]`; sklearn/cluster/_hdbscan/_tree.pyx:656 says `condensed_tree : ndarray of shape (n_samples,)`.
scenario: "developer sizing a buffer or writing a test based on the docstring uses `n_samples`-length arrays for the hierarchy/linkage → shape mismatch with the real `(n_samples - 1,)` layout, silent off-by-one bugs downstream"
contract: Correct these docstrings to `(n_samples - 1,)` for `hierarchy`/`linkage` and to a variable/`(n_edges,)` shape for `condensed_tree`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F7 — `_weighted_cluster_center` pre-allocates a `mask` buffer that is unconditionally overwritten (dead code implying reuse)
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)` is allocated once, but the loop body at line 908 rebinds `mask = self.labels_ == idx` on every iteration, so the pre-allocation is dead and misleadingly suggests a reused buffer.
scenario: "reader assumes the outer `mask` is being reused in-place for perf (e.g. via `np.equal(..., out=mask)`) → wastes time chasing an optimization that isn't there; also fails when `_weighted_cluster_center` is called with a mismatched-length `self.labels_` because the pre-allocation gives false confidence that shapes align (see F1)"
contract: Delete the line-896 pre-allocation; declare `mask` fresh inside the loop where it is actually used.
instances: single-instance

### F8 — `_weighted_cluster_center` ends with a stray bare `return` after a series of side-effect assignments
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:921 — the helper is documented (lines 882-887) to "instead stores them in the `self.{centroids, medoids}_` attributes" rather than returning anything, yet it ends with `return` on line 921 which returns `None`. Reads as if the helper were previously value-returning.
scenario: "reader reaches the trailing `return` on a documented mutator → wonders what value they missed and pays a small but real cognitive tax that is out of line with the surrounding style (`fit` uses `return self`; other helpers omit it)"
contract: Remove the bare `return` line.
instances: single-instance
