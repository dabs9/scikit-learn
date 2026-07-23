### F1 — `_weighted_cluster_center` crashes when input has non-finite rows and `store_centers` is set
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,895,908-909 — after the non-finite branch, `X = X[finite_index]` (shape `finite_count × n_features`) and `self.labels_` is rebuilt to shape `self._raw_data.shape[0]` (raw shape). Then `if self.store_centers: self._weighted_cluster_center(X)` is invoked with the reduced `X` while `_weighted_cluster_center` computes `mask = self.labels_ == idx` (raw-length boolean) and applies `data = X[mask]`, which raises `IndexError: boolean index did not match indexed array along dimension 0`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing any np.inf/np.nan row → fit() raises IndexError instead of storing centroids"
contract: Pass the reduced `X` together with `self.labels_[finite_index]` into `_weighted_cluster_center` so mask length matches data length.
instances: single-instance

### F2 — `n_clusters` in `_weighted_cluster_center` overcounts when missing-data label (-3) is present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. `_OUTLIER_ENCODING["missing"]["label"] == -3` (lines 74), so if any sample has np.nan the label -3 is present and is counted as a "cluster." The loop `for idx in range(n_clusters)` will iterate one extra index that no real cluster label matches, giving an all-False mask that either causes `np.average` on empty input to fail (`ZeroDivisionError`/`RuntimeWarning` with NaN result) or produces a spurious NaN row in `self.centroids_`/`self.medoids_`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing np.nan → an extra bogus centroid row is emitted or the call raises inside `np.average`"
contract: The noise-label filter must exclude every outlier encoding: `n_clusters = len(set(self.labels_) - {-1} - {out['label'] for out in _OUTLIER_ENCODING.values()})`.
instances: single-instance

### F3 — "Scale Invariance" demo never scales the data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — the loop `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})` fits and plots the unscaled `X` on every iteration; the parallel DBSCAN block above at lines 91-95 correctly uses `X * scale`. The rendered gallery therefore shows three identical clusterings labelled "scale=1/0.5/3", contradicting the section's stated demonstration of HDBSCAN's scale invariance.
scenario: "reader opens the HDBSCAN plotting example to see scale-invariance behavior → sees three identical plots that neither demonstrate nor even test the property"
contract: Fit and plot `X * scale` on each iteration, mirroring the DBSCAN block above.
instances: single-instance

### F4 — `n_jobs` docstring/default mismatch
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — the class docstring says `n_jobs : int, default=None`, and lines 654-658/669-673 in `__init__` use `n_jobs=4` as the default. Every other sklearn estimator uses `n_jobs=None` as the documented, joblib-context-honoring default; this estimator silently spawns 4 workers when the user does nothing.
scenario: "user calls `HDBSCAN()` inside a joblib backend context or on a small dataset → four subprocesses are spawned unexpectedly, contradicting both the docstring and sklearn convention"
contract: Change the `__init__` default to `n_jobs=None` to match the documented default and the sklearn convention.
instances: single-instance

### F5 — `_hdbscan_brute` default `alpha=None` will crash if the function is ever called without the keyword
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166,241 — signature is `def _hdbscan_brute(X, min_samples=5, alpha=None, ...)`, body performs `distance_matrix /= alpha`. `None` is not a valid divisor; `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`. This is masked today because `fit()` always injects `alpha=self.alpha` into `kwargs`, but the default in the helper is silently wrong and misleads readers/future callers. `_hdbscan_prims` correctly defaults `alpha=1.0` (line 273).
scenario: "future caller invokes `_hdbscan_brute(X)` without keyword or refactors `fit()` to omit alpha → TypeError instead of the documented `alpha=1.0` behavior"
contract: Set the helper default to `alpha=1.0`, mirroring `_hdbscan_prims`.
instances: single-instance

### F6 — `_hierarchical_fast.pxd` declares `noexcept` on methods whose `.pyx` implementations omit it
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:7-8 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331,339 defines them without the `noexcept` specifier. Under Cython 3.x this triggers a compile-time signature-mismatch warning; without `noexcept` the compiler must generate exception-propagation stubs on every call, defeating the point of the pxd declaration.
scenario: "Cython 3 compilation of `_hierarchical_fast.pyx` → emits `noexcept` mismatch warnings and inserts exception-check code around every call site, making `UnionFind.union` / `fast_find` slower and diverging from the header contract"
contract: Add `noexcept` to both `.pyx` method signatures so the definition matches the `.pxd` declaration.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F7 — `test_hdbscan_precomputed_non_brute` passes via parameter validation, not the code path it claims to test
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282-284 — the test builds `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` (i.e. `"prims_kdtree"` / `"prims_balltree"`). The class' `_parameter_constraints["algorithm"]` at sklearn/cluster/_hdbscan/hdbscan.py:629-644 restricts algorithm to `{"auto","brute","kdtree","balltree"}`, so `_validate_params()` raises `InvalidParameterError` (a `ValueError`) before any precomputed-vs-tree logic is exercised. The test docstring claims to check that "HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm" but the tree-based path is never entered.
scenario: "someone removes or renames the actual precomputed-vs-tree guard in `fit()` → this test still passes silently because it only exercises `_validate_params`"
contract: Rewrite the test to use `algorithm="kdtree"` / `"balltree"` (valid names) with `metric="precomputed"`, matching a real error raised by `fit()`, or otherwise assert against the specific expected message.
instances: single-instance

### F8 — `_hdbscan_prims` never applies `alpha` — silent parameter drop for tree-based algorithms
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-348 — the function receives `alpha=1.0` and passes it through as `mst_from_data_matrix(X, core_distances, dist_metric, alpha)`; `alpha` is only honored inside `mst_from_data_matrix` (sklearn/cluster/_hdbscan/_linkage.pyx:193 `pair_distance /= alpha`). But `core_distances` are computed via `NearestNeighbors(...).kneighbors(X)` at lines 332-343 with no `alpha` scaling. In `_hdbscan_brute` (line 241) `distance_matrix /= alpha` divides the *entire* distance matrix before core distances are derived from it, so core distances are also scaled by 1/alpha. Result: with `algorithm="brute"` core distances participate scaled; with tree algorithms they participate unscaled. The two backends produce different clusterings for the same non-default `alpha`.
scenario: "user sets `HDBSCAN(alpha=0.5)` and switches between `algorithm='brute'` and `algorithm='kdtree'` on identical data → the two backends return different labels because prims-based code paths silently ignore alpha when computing core distances"
contract: Divide `core_distances` by `alpha` in the prims path before they enter `mst_from_data_matrix`, matching the brute-force semantics.
instances: single-instance

### F9 — `remap_single_linkage_tree` writes `outlier + outlier_count` into left/right node columns that are meant to be raw sample indices, corrupting bookkeeping when both nan and inf are present
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:383-398 — `outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`; the `outlier` value used for the left node is the raw index of the non-finite point, but at line 838 the caller passes `non_finite=set(infinite_index + missing_index)`, which is a **set** whose iteration order is not deterministic. The subsequent `new_labels[infinite_index] = ...` and `new_labels[missing_index] = ...` at lines 842-843 are index-based writes and therefore correct, but any consumer that walks `self._single_linkage_tree_` (e.g. `dbscan_clustering` via `labelling_at_cut`) will see outlier merges appended in nondeterministic order, meaning `_single_linkage_tree_` — a public-through-`dbscan_clustering` attribute — is not reproducible across runs on the same data when non-finite entries are present.
scenario: "user calls `HDBSCAN().fit(X)` twice on identical non-finite data → `hdb._single_linkage_tree_` differs in edge order between runs; downstream `dbscan_clustering(cut_distance=...)` output is still correct but internal tree state is nondeterministic and breaks equality-based caching or serialization checks"
contract: Pass a deterministically ordered container (e.g. `sorted(set(infinite_index + missing_index))`) to `remap_single_linkage_tree`, and iterate that same ordering when constructing `outlier_tree`.
instances: single-instance

### F10 — Sparse feature path with non-finite entries never runs `_get_finite_row_indices`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:699-733 — the non-finite handling branch is guarded by `if self.metric != "precomputed":` and calls `_get_finite_row_indices(X)`. When `X` is a sparse (csr/lil) feature matrix (i.e. non-precomputed sparse), `_get_finite_row_indices` at line 401-407 handles sparse via `matrix.tolil().data`. But the diff test `test_hdbscan_sparse` at sklearn/cluster/tests/test_hdbscan.py:302-306 sets `sparse_X_nan[0, 0] = np.nan` on a `csr_matrix`, and `_assert_all_finite(X.data if issparse(X) else X)` at line 710 correctly detects it. Then `finite_index = _get_finite_row_indices(X)` runs `matrix.tolil().data` where each entry is a Python list of stored values — however `_get_finite_row_indices` treats rows with all-zero stored values as "finite" even when the un-stored zeros are irrelevant, but real bug: `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` at line 732 uses `finite_index` which is a `np.ndarray` for the dense case and a `np.array([...])` from list comprehension in the sparse case, whose dtype defaults to `int64` on most platforms — OK. However `reduced_X = X.sum(axis=1)` on a sparse matrix at line 721 returns a `np.matrix` (not ndarray); `np.isnan(reduced_X).nonzero()[0]` on a `np.matrix` returns 2-D indices whose semantics are then mishandled: `list(np.isnan(reduced_X).nonzero()[0])` yields row indices when the matrix is 1-column but for the general sparse case the `sum(axis=1)` yields `(n_samples, 1)` matrix, so `.nonzero()` returns row/col pairs and `.nonzero()[0]` gives rows — but the resulting `missing_index`/`infinite_index` still get passed to `new_labels[missing_index] = ...` at line 843 with dtype int64 — this happens to work, but is fragile and inconsistent with the dense path.
scenario: "sparse feature matrix with np.nan in stored data flows through the non-finite branch → `reduced_X = X.sum(axis=1)` returns np.matrix and subsequent scalar ops rely on np.matrix quirks; test coverage passes only because n_features>1 makes the shape workable"
contract: Convert `reduced_X` to an ndarray via `np.asarray(reduced_X).ravel()` before running `.nonzero()`; add an explicit test for np.inf (not just np.nan) in a sparse feature matrix to lock in behavior.
instances: single-instance

### F11 — `_do_labelling` returns `result` sized to `root_cluster` while callers assume it has length `n_samples`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:481 — `result = np.empty(root_cluster, dtype=np.intp)`, then `for n in range(root_cluster): ... result[n] = label`. `root_cluster = np.min(parent_array)`, which by construction equals `n_samples` (see `_condense_tree` at line 147 seeding `next_label = n_samples + 1` and relabel[root] = n_samples). So the assumption "root_cluster == n_samples" is invariant-dependent, undocumented, and if `_do_labelling` is ever called with a hand-crafted condensed tree whose parent ids don't start at n_samples (as the test `test_labelling_thresholding` does with `parent=5, n_samples=5` — matching the invariant) the array size will be wrong. In fact test_hdbscan.py:508-517 constructs a tree with parents=5 and n_samples=5 which fits the invariant only by coincidence.
scenario: "reviewer or downstream user calls `_do_labelling` with a valid condensed tree that has parent ids ≥ n_samples+1 (e.g. after external filtering) → `result` is too short and later `_get_clusters` / caller silently trims samples"
contract: Take `n_samples` as an explicit argument (or infer it from `child_array.max() + 1` where child < parents), and size `result = np.empty(n_samples, dtype=np.intp)`. Add a docstring assertion that `parent_array.min() == n_samples`.
instances: single-instance

### F12 — Deprecated `np.core.records` and `np.infty` usage
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:128 `mst = np.core.records.fromarrays(...)` — `np.core` is a private submodule that NumPy 1.25+ warns is on deprecation track (public spelling is `np.rec.fromarrays`). sklearn/cluster/_hdbscan/_linkage.pyx:99,165 use `np.infty`, which was deprecated in NumPy 1.20 in favor of `np.inf`.
scenario: "NumPy 2.x removes `np.infty` and further tightens `np.core` access → HDBSCAN imports emit DeprecationWarning and eventually fail"
contract: Replace `np.core.records.fromarrays` with `np.rec.fromarrays` and `np.infty` with `np.inf`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:128, sklearn/cluster/_hdbscan/_linkage.pyx:99, sklearn/cluster/_hdbscan/_linkage.pyx:165]

### F13 — `_compute_stability` duplicated `births` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice back-to-back with no intervening use. The first allocation is dead code (immediately overwritten). Per the PR description item 3 ("Trimmed unused variables (thanks to Cython linting pre-commit)"), this is exactly the sort of leftover Cython linting should have flagged.
scenario: "`_compute_stability` is invoked during every HDBSCAN fit → the redundant `np.full` allocation runs on every call, wasting memory and signaling that the advertised Cython cleanup was incomplete"
contract: Delete the duplicate line at :252 (keep :254).
instances: single-instance

### F14 — `bfs_from_hierarchy` list-of-Python-ints in a hot loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:87-114 — `bfs_from_hierarchy` is declared `cdef list ...` and uses Python-level list comprehensions/`extend`. It is called once per non-tiny internal node from `_condense_tree` at lines 200, 207, 216, 225. For a 10⁴-sample dataset this dominates runtime and is why HDBSCAN condensation is materially slower than the reference implementation. The plan of record's item "Clean `_hdbscan/_tree.pyx`" was expected to reach Cython-idiomatic code; this function is still a Python routine wrapped in `cdef`.
scenario: "user runs `HDBSCAN().fit(X)` on a large dataset → hot-path condensation is bottlenecked on Python object allocation inside `bfs_from_hierarchy`, and sub-cluster BFS iterations grow O(cluster_count²) worst-case"
contract: Replace the Python `list` queue with a preallocated `intp_t[::1]` buffer and index-tracked front/back pointers, and remove per-node Python allocations.
instances: single-instance

### F15 — Docstring says "distance matrix must be square" but validation checks only equal dims (redundant), missing "same value on transpose"-only checks for sparse
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:222-234 vs 734-740 — the brute path guards precomputed symmetry via `_allclose_dense_sparse(X, X.T)` and `X.shape[0] != X.shape[1]`; the sparse-precomputed branch at lines 734-740 skips both checks. A user passing a non-square or asymmetric sparse precomputed matrix will not receive the documented error; instead an obscure downstream failure in `mutual_reachability_graph` occurs.
scenario: "user calls `HDBSCAN(metric='precomputed').fit(sparse_matrix)` with a non-square sparse matrix → obscure downstream error instead of the documented ValueError"
contract: Move the shape/symmetry validation out of the brute branch (lines 222-234) into the shared entry point so it applies to sparse precomputed input as well; add tests covering sparse-precomputed shape and symmetry errors.
instances: single-instance

### F16 — `test_hdbscan_min_cluster_size` never actually asserts noise handling when `min_cluster_size` exceeds `n_samples/2`
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:260-269 — the loop `for min_cluster_size in range(2, len(X), 1):` iterates through 198 fit-predict runs, but the body only asserts `np.min(np.bincount(true_labels)) >= min_cluster_size` **inside** `if len(true_labels) != 0`. There is no assertion for the many cases where every point becomes noise and `true_labels` is empty; the loop silently accepts "no clusters" for large `min_cluster_size` values. Given that `_min_samples` defaults to `min_cluster_size` and the check at hdbscan.py:758 forbids `_min_samples > X.shape[0]`, the top of the range approaches that limit with no explicit expectation — a change that makes the classifier degrade silently (e.g. always emitting -1) would keep this test green.
scenario: "regression turns every HDBSCAN run into all-noise for large min_cluster_size → test still passes because the assertion is skipped when `true_labels` is empty"
contract: Assert an explicit condition for the "no clusters" case (e.g. bound the min_cluster_size range so at least one cluster is guaranteed, or assert that `len(true_labels) > 0` above some cutoff).
instances: single-instance
