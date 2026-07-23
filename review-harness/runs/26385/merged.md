### F1 — `_weighted_cluster_center` crashes when input has non-finite rows and `store_centers` is set
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,895,908-909 — after the non-finite branch, `X = X[finite_index]` (shape `finite_count × n_features`) and `self.labels_` is rebuilt to shape `self._raw_data.shape[0]` (raw shape). Then `if self.store_centers: self._weighted_cluster_center(X)` is invoked with the reduced `X` while `_weighted_cluster_center` computes `mask = self.labels_ == idx` (raw-length boolean) and applies `data = X[mask]`, which raises `IndexError: boolean index did not match indexed array along dimension 0`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing any np.inf/np.nan row → fit() raises IndexError instead of storing centroids"
contract: Pass the reduced `X` together with `self.labels_[finite_index]` into `_weighted_cluster_center` so mask length matches data length.
instances: single-instance

### F2 — `_weighted_cluster_center` miscounts clusters when missing-data label `-3` is present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the outlier-encoding at hdbscan.py:80 defines the missing label as `-3`, and hdbscan.py:849 assigns `-3` to samples with NaN rows
scenario: "User calls `HDBSCAN(store_centers='centroid').fit(X)` on data with any NaN rows → `-3` remains in the label set, `n_clusters` is inflated by 1, the final iteration `idx = n_clusters-1` selects an empty mask, and `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError` / returns nan-filled row (medoid path calls `pairwise_distances` on an empty array)"
contract: exclude every outlier label — use `n_clusters = len(set(self.labels_) - {-1, -2, -3})` (or `- set(v['label'] for v in _OUTLIER_ENCODING.values()) - {-1}`); mirror this same set everywhere non-outlier labels are enumerated
instances: single-instance

### F3 — `_hdbscan_prims` never applies `alpha` — silent parameter drop for tree-based algorithms
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-348 — the function receives `alpha=1.0` and passes it through as `mst_from_data_matrix(X, core_distances, dist_metric, alpha)`; `alpha` is only honored inside `mst_from_data_matrix` (sklearn/cluster/_hdbscan/_linkage.pyx:193 `pair_distance /= alpha`). But `core_distances` are computed via `NearestNeighbors(...).kneighbors(X)` at lines 332-343 with no `alpha` scaling. In `_hdbscan_brute` (line 241) `distance_matrix /= alpha` divides the *entire* distance matrix before core distances are derived from it, so core distances are also scaled by 1/alpha. Result: with `algorithm="brute"` core distances participate scaled; with tree algorithms they participate unscaled. The two backends produce different clusterings for the same non-default `alpha`.
scenario: "user sets `HDBSCAN(alpha=0.5)` and switches between `algorithm='brute'` and `algorithm='kdtree'` on identical data → the two backends return different labels because prims-based code paths silently ignore alpha when computing core distances"
contract: Divide `core_distances` by `alpha` in the prims path before they enter `mst_from_data_matrix`, matching the brute-force semantics.
instances: single-instance

### F4 — `HDBSCAN.__init__` default `n_jobs=4` contradicts docstring and sklearn convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in the `__init__` signature, while the docstring at :486 documents `n_jobs : int, default=None`. Sklearn's convention (and every other estimator listed in the whats_new entry) is `n_jobs=None`, meaning "1 unless in a `joblib.parallel_backend` context". A hard-coded `4` silently spawns 4 worker threads regardless of the user's `parallel_backend`.
scenario: "user creates `HDBSCAN()` inside `with joblib.parallel_backend('threading', n_jobs=1):` → sklearn spawns 4 threads anyway, contradicting the documented behavior and the context manager the user set"
contract: Change the default in `__init__` to `n_jobs=None` so behavior matches the docstring and sklearn glossary semantics.
instances: single-instance

### F5 — `test_hdbscan_precomputed_non_brute` triggers `InvalidParameterError`, not the error under test
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`. The valid `algorithm` values in `_parameter_constraints` (sklearn/cluster/_hdbscan/hdbscan.py:629-638) are `{"auto","brute","kdtree","balltree"}`, so `"prims_kdtree"`/`"prims_balltree"` fail parameter validation before any precomputed-vs-tree logic runs. The assertion `pytest.raises(ValueError)` still passes because `InvalidParameterError` is a subclass of `ValueError`, but the test never exercises the code path it names in its docstring ("correctly raises an error when passing precomputed data while requesting a tree-based algorithm").
scenario: "test claims to guard the precomputed-vs-tree combination but actually only guards the algorithm StrOptions validator → the intended incompatibility check becomes uncovered and any regression there ships unnoticed"
contract: Use the actual algorithm strings supported by this estimator (`"kdtree"`, `"balltree"`) and assert with `match=` on the "Sparse data matrices only support algorithm `brute`" style message so the test fails if that guard is removed. Also verify a corresponding guard exists for the precomputed dense case.
instances: single-instance

### F6 — Test computes `clean_idx` via array addition, not concatenation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212-221 — `missing_labels_idx = np.flatnonzero(...)` (line 212, returns ndarray `[2, 5]`), `infinite_labels_idx = np.flatnonzero(...)` (line 215, ndarray `[0]`), then line 218 `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx + infinite_labels_idx` is elementwise numpy addition, not concatenation — `[2, 5] + [0]` broadcasts to `[2, 5]`, so the sample at index `0` (an infinite outlier) is left in `clean_idx`. The subsequent `assert_array_equal(clean_labels, labels[clean_idx])` passes only because HDBSCAN still assigns the infinite sample `-2` when kept, matching the original labels at that index.
scenario: "test intends to compare a clean-refit against outlier-marked labels → actually keeps outlier row 0 in the 'clean' subset → the assertion succeeds by accident and any change that made the two labelings diverge at row 0 would silently be missed"
contract: Concatenate as lists — `clean_idx = list(set(range(200)) - set(missing_labels_idx.tolist() + infinite_labels_idx.tolist()))` (or `np.concatenate([...]).tolist()`) — so `clean_idx` truly excludes every outlier.
instances: single-instance

### F7 — Plot example claims scale-invariance but never scales `X`
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — inside `for idx, scale in enumerate((1, 0.5, 3))` the body calls `hdb.fit(X)` and `plot(X, hdb.labels_, ...)` with the unscaled `X`, while the surrounding narrative at examples/cluster/plot_hdbscan.py:104-105 says "HDBSCAN is scale-invariant" and the parallel DBSCAN block at examples/cluster/plot_hdbscan.py:92-95 uses `X * scale`. Every panel receives identical data, so the rendered figure demonstrates nothing about scale invariance.
scenario: "Reader looks at the auto-generated example expecting scaled inputs → sees three identical HDBSCAN plots labelled `scale=1`, `scale=0.5`, `scale=3`, which is trivially self-consistent and does not evidence the property being taught"
contract: Change the loop body to `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)`, mirroring the DBSCAN block.
instances: single-instance

### F8 — `remap_single_linkage_tree` writes `outlier + outlier_count` into left/right node columns that are meant to be raw sample indices, corrupting bookkeeping when both nan and inf are present
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:383-398 — `outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`; the `outlier` value used for the left node is the raw index of the non-finite point, but at line 838 the caller passes `non_finite=set(infinite_index + missing_index)`, which is a **set** whose iteration order is not deterministic. The subsequent `new_labels[infinite_index] = ...` and `new_labels[missing_index] = ...` at lines 842-843 are index-based writes and therefore correct, but any consumer that walks `self._single_linkage_tree_` (e.g. `dbscan_clustering` via `labelling_at_cut`) will see outlier merges appended in nondeterministic order, meaning `_single_linkage_tree_` — a public-through-`dbscan_clustering` attribute — is not reproducible across runs on the same data when non-finite entries are present.
scenario: "user calls `HDBSCAN().fit(X)` twice on identical non-finite data → `hdb._single_linkage_tree_` differs in edge order between runs; downstream `dbscan_clustering(cut_distance=...)` output is still correct but internal tree state is nondeterministic and breaks equality-based caching or serialization checks"
contract: Pass a deterministically ordered container (e.g. `sorted(set(infinite_index + missing_index))`) to `remap_single_linkage_tree`, and iterate that same ordering when constructing `outlier_tree`.
instances: single-instance

### F9 — Sparse feature path with non-finite entries never runs `_get_finite_row_indices`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:699-733 — the non-finite handling branch is guarded by `if self.metric != "precomputed":` and calls `_get_finite_row_indices(X)`. When `X` is a sparse (csr/lil) feature matrix (i.e. non-precomputed sparse), `_get_finite_row_indices` at line 401-407 handles sparse via `matrix.tolil().data`. But the diff test `test_hdbscan_sparse` at sklearn/cluster/tests/test_hdbscan.py:302-306 sets `sparse_X_nan[0, 0] = np.nan` on a `csr_matrix`, and `_assert_all_finite(X.data if issparse(X) else X)` at line 710 correctly detects it. Then `finite_index = _get_finite_row_indices(X)` runs `matrix.tolil().data` where each entry is a Python list of stored values — however `_get_finite_row_indices` treats rows with all-zero stored values as "finite" even when the un-stored zeros are irrelevant, but real bug: `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` at line 732 uses `finite_index` which is a `np.ndarray` for the dense case and a `np.array([...])` from list comprehension in the sparse case, whose dtype defaults to `int64` on most platforms — OK. However `reduced_X = X.sum(axis=1)` on a sparse matrix at line 721 returns a `np.matrix` (not ndarray); `np.isnan(reduced_X).nonzero()[0]` on a `np.matrix` returns 2-D indices whose semantics are then mishandled: `list(np.isnan(reduced_X).nonzero()[0])` yields row indices when the matrix is 1-column but for the general sparse case the `sum(axis=1)` yields `(n_samples, 1)` matrix, so `.nonzero()` returns row/col pairs and `.nonzero()[0]` gives rows — but the resulting `missing_index`/`infinite_index` still get passed to `new_labels[missing_index] = ...` at line 843 with dtype int64 — this happens to work, but is fragile and inconsistent with the dense path.
scenario: "sparse feature matrix with np.nan in stored data flows through the non-finite branch → `reduced_X = X.sum(axis=1)` returns np.matrix and subsequent scalar ops rely on np.matrix quirks; test coverage passes only because n_features>1 makes the shape workable"
contract: Convert `reduced_X` to an ndarray via `np.asarray(reduced_X).ravel()` before running `.nonzero()`; add an explicit test for np.inf (not just np.nan) in a sparse feature matrix to lock in behavior.
instances: single-instance

### F10 — `_do_labelling` returns `result` sized to `root_cluster` while callers assume it has length `n_samples`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:481 — `result = np.empty(root_cluster, dtype=np.intp)`, then `for n in range(root_cluster): ... result[n] = label`. `root_cluster = np.min(parent_array)`, which by construction equals `n_samples` (see `_condense_tree` at line 147 seeding `next_label = n_samples + 1` and relabel[root] = n_samples). So the assumption "root_cluster == n_samples" is invariant-dependent, undocumented, and if `_do_labelling` is ever called with a hand-crafted condensed tree whose parent ids don't start at n_samples (as the test `test_labelling_thresholding` does with `parent=5, n_samples=5` — matching the invariant) the array size will be wrong. In fact test_hdbscan.py:508-517 constructs a tree with parents=5 and n_samples=5 which fits the invariant only by coincidence.
scenario: "reviewer or downstream user calls `_do_labelling` with a valid condensed tree that has parent ids ≥ n_samples+1 (e.g. after external filtering) → `result` is too short and later `_get_clusters` / caller silently trims samples"
contract: Take `n_samples` as an explicit argument (or infer it from `child_array.max() + 1` where child < parents), and size `result = np.empty(n_samples, dtype=np.intp)`. Add a docstring assertion that `parent_array.min() == n_samples`.
instances: single-instance

### F11 — `_compute_stability` duplicated `births` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice back-to-back with no intervening use. The first allocation is dead code (immediately overwritten). Per the PR description item 3 ("Trimmed unused variables (thanks to Cython linting pre-commit)"), this is exactly the sort of leftover Cython linting should have flagged.
scenario: "`_compute_stability` is invoked during every HDBSCAN fit → the redundant `np.full` allocation runs on every call, wasting memory and signaling that the advertised Cython cleanup was incomplete"
contract: Delete the duplicate line at :252 (keep :254).
instances: single-instance

### F12 — `remap_single_linkage_tree` docstring conflicts with actual call-site type
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 documents `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`, but the call site at :838 passes `non_finite=set(infinite_index + missing_index)` — a Python set of integer raw-data indices. The function's body relies on that set behaviour (`len(non_finite)` at :369 and `for i, outlier in enumerate(non_finite)` at :388 where `outlier` is used as a raw-data index at :389), which would not work with a boolean array. Iterating a `set` also gives non-deterministic order across Python invocations, so `self._single_linkage_tree_` for the non-finite path is nondeterministic between runs.
scenario: "reader relies on the docstring → passes a boolean mask and gets wrong results or an exception; separately, `_single_linkage_tree_` values differ across process invocations for otherwise identical inputs, complicating debugging and downstream serialization"
contract: Rewrite the docstring to state that `non_finite` is a collection of raw-data indices of non-finite rows, and pass a `sorted(...)` sequence (e.g. `sorted(set(infinite_index + missing_index))`) at :838 so ordering is deterministic.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:364, sklearn/cluster/_hdbscan/hdbscan.py:388, sklearn/cluster/_hdbscan/hdbscan.py:838]

### F13 — Cross-reference `:ref:\`User Guide <HDBSCAN>\`` in example uses wrong-case anchor
severity: low
evidence: examples/cluster/plot_hdbscan.py:110 — `see :ref:\`User Guide <HDBSCAN>\``. The corresponding label defined in `doc/modules/clustering.rst:51` is `.. _hdbscan:` (lowercase). Sphinx label lookups are case-sensitive, so this cross-reference fails to resolve and the rendered example will emit a Sphinx warning and produce a broken link.
scenario: "docs build → Sphinx warns about unknown label `HDBSCAN`; user clicking the User Guide link in the rendered example hits a dead reference"
contract: Change the reference to `:ref:\`User Guide <hdbscan>\`` to match the declared label.
instances: single-instance

### F14 — `_hdbscan_brute` forwards `metric_params` to `pairwise_distances` including `max_distance`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:238-243 — the non-precomputed branch calls `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` and only afterwards extracts `max_distance = metric_params.get("max_distance", 0.0)`. `pairwise_distances` does not accept a `max_distance` keyword, so any user who sets `metric_params={"max_distance": ...}` with a non-precomputed metric gets `TypeError: ... got an unexpected keyword argument 'max_distance'`. The docstring at sklearn/cluster/_hdbscan/hdbscan.py:459-460 advertises `metric_params : dict` as generic "Arguments passed to the distance metric" without excluding this key.
scenario: "User sets `metric_params={'max_distance': 5.0}` (documented for sparse precomputed) but with `metric='euclidean'` on a dense array → `pairwise_distances` rejects the kwarg and raises TypeError instead of quietly ignoring it"
contract: Pop `max_distance` out of `metric_params` before forwarding to `pairwise_distances`, and treat it as a top-level runtime option.
instances: single-instance

### F15 — `whats_new` reference to `DBSCAN` missing module prefix
severity: low
evidence: doc/whats_new/v1.3.rst:194-196 — reads `Similarly to :class:`cluster.OPTICS`, it can be seen as a generalization of :class:`DBSCAN` by allowing for hierarchical instead of flat clustering`. The `:class:` role on `DBSCAN` omits the `cluster.` module prefix used everywhere else in the paragraph (and used for `cluster.HDBSCAN`, `cluster.OPTICS`), so Sphinx cannot resolve the cross-reference.
scenario: "Doc build for 1.3 whats-new runs with `nitpicky` mode → emits an unresolved-reference warning and renders `DBSCAN` as unlinked literal text instead of a link to the DBSCAN class"
contract: Change to `:class:`cluster.DBSCAN`` to match the sibling references.
instances: single-instance

### F16 — `_hdbscan_brute` and `_hdbscan_prims` docstrings misdocument defaults and list an unused parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:163-172 (signature: `min_samples=5, alpha=None`) vs. docstring at 185/189 (`min_samples : int, default=None`, `alpha : float, default=1.0`); sklearn/cluster/_hdbscan/hdbscan.py:275-283 (signature: `min_samples=5`, no `copy` parameter) vs. docstring at 296 (`min_samples : int, default=None`) and 319-324 (documents `copy : bool, default=False` that does not exist in the signature).
scenario: "readers of the internal API get the wrong defaults from the docstrings → anyone extending or calling these internals passes the docstring's `None`/`copy=False` and gets silently different behavior; the phantom `copy` parameter also misleads anyone trying to add copy semantics to the tree algorithms."
contract: sync each `default=` and parameter list to the actual signatures — `min_samples : int, default=5`; `alpha : float, default=None` in `_hdbscan_brute` (or change the signature to `alpha=1.0` if the default was the intent); delete the `copy` docstring block from `_hdbscan_prims`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:185, sklearn/cluster/_hdbscan/hdbscan.py:189, sklearn/cluster/_hdbscan/hdbscan.py:296, sklearn/cluster/_hdbscan/hdbscan.py:319]

### F17 — User-guide typos and stray colons in the new HDBSCAN section
severity: low
evidence: doc/modules/clustering.rst:1009-1011 — "removing any edges with value greater than :math:`\varepsilon`:\nfrom the original graph. Any points whose core distance is less than :math:`\varepsilon`:\nare at this staged marked as noise." Both math roles carry a trailing colon that renders literally, and "at this staged marked" should be "at this stage marked". doc/modules/clustering.rst:1025-1026 — "the fully\n-connected mutual reachability graph" renders with a stray hyphen/space split.
scenario: "Sphinx build renders the trailing `:` literally after each formula and prints 'at this staged marked' → published user guide has broken punctuation and a grammar error immediately after the algorithm's central formulas, which readers arriving at the new section see first."
contract: remove the trailing `:` after each `:math:` role, fix "staged" → "stage", and rejoin "fully-connected" onto one line.
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011, doc/modules/clustering.rst:1025]

### F18 — `_hierarchical_fast.pxd` declares `noexcept` on methods whose `.pyx` implementations omit it
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:7-8 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331,339 defines them without the `noexcept` specifier. Under Cython 3.x this triggers a compile-time signature-mismatch warning; without `noexcept` the compiler must generate exception-propagation stubs on every call, defeating the point of the pxd declaration.
scenario: "Cython 3 compilation of `_hierarchical_fast.pyx` → emits `noexcept` mismatch warnings and inserts exception-check code around every call site, making `UnionFind.union` / `fast_find` slower and diverging from the header contract"
contract: Add `noexcept` to both `.pyx` method signatures so the definition matches the `.pxd` declaration.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F19 — Deprecated `np.core.records` and `np.infty` usage
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:128 `mst = np.core.records.fromarrays(...)` — `np.core` is a private submodule that NumPy 1.25+ warns is on deprecation track (public spelling is `np.rec.fromarrays`). sklearn/cluster/_hdbscan/_linkage.pyx:99,165 use `np.infty`, which was deprecated in NumPy 1.20 in favor of `np.inf`.
scenario: "NumPy 2.x removes `np.infty` and further tightens `np.core` access → HDBSCAN imports emit DeprecationWarning and eventually fail"
contract: Replace `np.core.records.fromarrays` with `np.rec.fromarrays` and `np.infty` with `np.inf`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:128, sklearn/cluster/_hdbscan/_linkage.pyx:99, sklearn/cluster/_hdbscan/_linkage.pyx:165]

### F20 — `bfs_from_hierarchy` list-of-Python-ints in a hot loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:87-114 — `bfs_from_hierarchy` is declared `cdef list ...` and uses Python-level list comprehensions/`extend`. It is called once per non-tiny internal node from `_condense_tree` at lines 200, 207, 216, 225. For a 10⁴-sample dataset this dominates runtime and is why HDBSCAN condensation is materially slower than the reference implementation. The plan of record's item "Clean `_hdbscan/_tree.pyx`" was expected to reach Cython-idiomatic code; this function is still a Python routine wrapped in `cdef`.
scenario: "user runs `HDBSCAN().fit(X)` on a large dataset → hot-path condensation is bottlenecked on Python object allocation inside `bfs_from_hierarchy`, and sub-cluster BFS iterations grow O(cluster_count²) worst-case"
contract: Replace the Python `list` queue with a preallocated `intp_t[::1]` buffer and index-tracked front/back pointers, and remove per-node Python allocations.
instances: single-instance

### F21 — Docstring says "distance matrix must be square" but validation checks only equal dims (redundant), missing "same value on transpose"-only checks for sparse
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:222-234 vs 734-740 — the brute path guards precomputed symmetry via `_allclose_dense_sparse(X, X.T)` and `X.shape[0] != X.shape[1]`; the sparse-precomputed branch at lines 734-740 skips both checks. A user passing a non-square or asymmetric sparse precomputed matrix will not receive the documented error; instead an obscure downstream failure in `mutual_reachability_graph` occurs.
scenario: "user calls `HDBSCAN(metric='precomputed').fit(sparse_matrix)` with a non-square sparse matrix → obscure downstream error instead of the documented ValueError"
contract: Move the shape/symmetry validation out of the brute branch (lines 222-234) into the shared entry point so it applies to sparse precomputed input as well; add tests covering sparse-precomputed shape and symmetry errors.
instances: single-instance

### F22 — `test_hdbscan_min_cluster_size` never actually asserts noise handling when `min_cluster_size` exceeds `n_samples/2`
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:260-269 — the loop `for min_cluster_size in range(2, len(X), 1):` iterates through 198 fit-predict runs, but the body only asserts `np.min(np.bincount(true_labels)) >= min_cluster_size` **inside** `if len(true_labels) != 0`. There is no assertion for the many cases where every point becomes noise and `true_labels` is empty; the loop silently accepts "no clusters" for large `min_cluster_size` values. Given that `_min_samples` defaults to `min_cluster_size` and the check at hdbscan.py:758 forbids `_min_samples > X.shape[0]`, the top of the range approaches that limit with no explicit expectation — a change that makes the classifier degrade silently (e.g. always emitting -1) would keep this test green.
scenario: "regression turns every HDBSCAN run into all-noise for large min_cluster_size → test still passes because the assertion is skipped when `true_labels` is empty"
contract: Assert an explicit condition for the "no clusters" case (e.g. bound the min_cluster_size range so at least one cluster is guaranteed, or assert that `len(true_labels) > 0` above some cutoff).
instances: single-instance

### F23 — Stale comment describes wrong outlier labels
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The very next lines use `_OUTLIER_ENCODING["infinite"]["label"]` = `-2` and `_OUTLIER_ENCODING["missing"]["label"]` = `-3` (see the encoding definitions at lines 72-84 and the class docstring at lines 540-543 which correctly document `-2` / `-3`).
scenario: "a maintainer reads the comment to reason about the label semantics → mis-conflates infinite with -1 (which is generic noise) and missing with -2 (which is infinite) → wrong changes to downstream code"
contract: Update the comment to state "np.inf → -2 and np.nan → -3", matching the class docstring and `_OUTLIER_ENCODING`.
instances: single-instance

### F24 — Plan claim "cnp.*_t → *_t from _typedefs" not applied to _tree.pyx
severity: low
evidence: The PR description lists as a novel change: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`". `_linkage.pyx` and `_reachability.pyx` import scalar typedefs from `...utils._typedefs cimport ...` and use bare `intp_t`, `float64_t`, etc. However `sklearn/cluster/_hdbscan/_tree.pyx` still uses `cnp.intp_t`, `cnp.float64_t`, `cnp.uint8_t` etc. throughout — 90 total occurrences (see grep). Only the `_tree.pxd` header imports the new typedefs (`_tree.pxd:36`); the `.pyx` body itself was not converted.
scenario: "plan-of-record says every Cython file uses the new typedef import → _tree.pyx is inconsistent → next author touching this file has to guess which convention applies and may reintroduce mismatches"
contract: Convert `_tree.pyx` to use `intp_t`/`float64_t`/`uint8_t` from `...utils._typedefs cimport ...`, matching the sibling `_linkage.pyx`/`_reachability.pyx`.
instances: single-instance

### F25 — Docstring uses capitalized "KDTree"/"BallTree" as if they were the parameter values
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:472-478 — "algorithm : {"auto", "brute", "kdtree", "balltree"}, default="auto" ... Both `"KDTree"` and `"BallTree"` algorithms use the :class:`~sklearn.neighbors.NearestNeighbors` estimator." The accepted parameter values are lowercase `"kdtree"`/`"balltree"` (matches `_parameter_constraints` and `StrOptions`).
scenario: "user copy-pastes `algorithm=\"KDTree\"` from the description prose → `_validate_params()` raises `InvalidParameterError` because the value does not match the lowercase StrOptions set"
contract: Refer to the algorithm strings only in their true lowercase form throughout the docstring — `\"kdtree\"` / `\"balltree\"`.
instances: single-instance

### F26 — User-guide references undefined parameter name `minimum_cluster_size`
severity: low
evidence: doc/modules/clustering.rst:138-141 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples` to couple the parameters"; the actual estimator parameter is `min_cluster_size` (hdbscan.py:649, 435)
scenario: "User copies the recommendation verbatim → `HDBSCAN(minimum_cluster_size=min_samples)` raises `TypeError: unexpected keyword argument` (or the parameter is silently ignored, depending on estimator plumbing)"
contract: rename both occurrences to the true parameter, `min_cluster_size`
instances: [doc/modules/clustering.rst:139, doc/modules/clustering.rst:140]

### F27 — `internal_to_raw` built as a dict for what is a plain index lookup
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:732 — `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` where `finite_index` is already an ndarray of positional integers; the callee at hdbscan.py:375/379 only ever does `internal_to_raw[left]` / `[right]` where `left`/`right` are `< finite_count`
scenario: "`HDBSCAN.fit` runs on data with any non-finite rows → the dict comprehension allocates one Python-int mapping per finite sample (O(n) hashmap entries) purely to reproduce ndarray indexing, dominating the outlier-handling path for large `n_samples` with a small non-finite fraction"
contract: pass the `finite_index` ndarray directly and use ndarray indexing in `remap_single_linkage_tree` (`internal_to_raw[left]` becomes exactly the same expression); update the docstring accordingly
instances: single-instance
