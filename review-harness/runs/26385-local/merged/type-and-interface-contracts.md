I note the system reminders about malware analysis. The files I read are review findings documents (markdown), not code. They contain analysis of scikit-learn code but are not themselves malware and contain no executable code that I'm being asked to improve. I'm performing my assigned UNION-merge task on review findings. I will not use TodoWrite as this is a single-step merge operation.

Now merging. Let me identify duplicates across draws:

- **n_jobs=4 vs None**: A/F2 (high), B/F12 (high, out-of-theme), C/F8 (low). Same defect (hdbscan.py:658 vs docstring). B is most specific (cites constraint + docstring + signature). Keep B/F12 verbatim.
- **remap_single_linkage_tree non_finite**: A/F1 (medium), B/F5 (medium), C/F9 (low). Same defect. A is most specific (explains len/enumerate consequences in detail). Keep A/F1.
- **_do_labelling parent_lambda scalar**: A/F4 (medium), B/F6 (medium), C/F10 (low). Same defect. C adds `label` untyped observation but that's a distinct addition — actually C/F10 covers both `label` untyped AND the parent_lambda issue. A/F4 is most specific on the parent_lambda root cause. Keep A/F4 for parent_lambda. But C/F10 mentions `label` untyped which is a distinct sub-defect. Reviewing more carefully: C/F10 title is "untyped `label` and reliance on length-1", so it's the same root cause site (line 498/505) plus an added observation about `label`. Since the root cause is same (implicit truthiness on line 505), keep A/F4 as most specific and don't add C/F10 separately.
- **_hdbscan_brute alpha=None**: A/F3 (medium), C/F5 (medium). Same defect. A is most specific (mentions parallel _hdbscan_prims). Keep A/F3.
- **allow_single_cluster type drift**: A/F5 (low), B/F2 (medium). Same defect. B is most specific (5 locations vs 3). Keep B/F2.
- **labels_ dtype int32 vs intp**: B/F4 (medium), C/F2 (medium). Same defect. Both similar; B slightly more specific about downstream. Keep B/F4.
- **traverse_upwards scalar coercion**: B/F7 (medium), C/F11 (low). Same defect. B is more specific. Keep B/F7.

Unique findings:
- A/F6 (Cython dtype shape docs), A/F7 (test out-of-theme), A/F8 (test out-of-theme), A/F9 (_get_clusters stabilities), A/F10 (kd_tree spelling)
- B/F1 (noexcept .pxd/.pyx), B/F3 (HIERARCHY_t intp vs int64), B/F8 (child_selection uint8/bool), B/F9 (result_pre_dict memoryview), B/F10 (bfs_from_cluster_tree contiguity), B/F11 (mutual_reachability mode='c')
- C/F1 (store_centers IndexError - high), C/F3 (-3 label), C/F4 (max_distance in metric_params), C/F6 (min_samples docstring), C/F7 (copy param docstring), C/F13 (allow_nan tag)

Ordering: high, then medium, then low.

**High:** B/F12, B/F1, C/F1
**Medium:** A/F1, A/F3, A/F4, A/F7, B/F2, B/F3, B/F4, B/F7, B/F8, C/F3, C/F4
**Low:** A/F6, A/F8, A/F9, A/F10, B/F9, B/F10, B/F11, C/F6, C/F7, C/F13

### F1 — `n_jobs` default is `4` while `_parameter_constraints` and docstring both say the default is `None` [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None`; :640 declares the constraint as `"n_jobs": [Integral, None]`; :658 sets the actual constructor default to `n_jobs=4`.
scenario: "User relying on the documented `n_jobs=None` (single-thread, joblib backend-controlled) default → HDBSCAN silently uses 4 processes at fit time, producing different resource consumption and, when combined with joblib contexts, unexpected parallelism."
contract: The `__init__` default must be `n_jobs=None` to match the documented and constrained default.
instances: single-instance

### F2 — `UnionFind.union`/`fast_find` `noexcept` declared in .pxd but not in .pyx implementation
severity: high
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declare `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 and :339 implement them without `noexcept` (`cdef void union(self, intp_t m, intp_t n):` and `cdef intp_t fast_find(self, intp_t n):`).
scenario: "Cython 3+ compilation of the module → signature-mismatch error or, in permissive versions, an unintended exception-propagation contract that silently diverges from the header's stated no-exception guarantee, breaking `nogil` callers."
contract: Declarations in `_hierarchical_fast.pxd` and implementations in `_hierarchical_fast.pyx` must carry the identical `noexcept` annotation for both `union` and `fast_find`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F3 — `store_centers` with non-finite data raises IndexError due to mask/data length mismatch
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,908-909 — line 733 sets `X = X[finite_index]` (filtered), lines 840-844 remap `self.labels_` to shape `self._raw_data.shape[0]` (full), then line 855 calls `self._weighted_cluster_center(X)` with filtered `X`, and inside line 908 does `mask = self.labels_ == idx` (full length) then `data = X[mask]` (indexing filtered X with full-length mask).
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` where X contains np.nan or np.inf → boolean index of length `n_raw` applied to X of length `n_finite` raises IndexError, `fit` fails."
contract: When `store_centers` is set and non-finite data is present, `_weighted_cluster_center` must operate on labels restricted to the finite subset (or X must be re-indexed to raw shape with matching label lengths); the two arrays passed through the mask must have identical length.
instances: single-instance

### F4 — `remap_single_linkage_tree` receives a set of indices where docstring declares a boolean array
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 documents `non_finite : ndarray / Boolean array of which entries in the raw data are non-finite`, but the sole caller at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` — a `set` of integer indices. Inside the function line 383 does `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)` and line 388 `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)`. If the argument were actually a boolean ndarray as documented, `len(non_finite)` would equal n_samples (not the outlier count) and `enumerate(non_finite)` would iterate `True/False` values, silently producing wrong outlier rows.
scenario: "Any code path that trusts the docstring and passes a boolean mask (the documented contract) → silently wrong tree construction, then arbitrary label assignment"
contract: Change the parameter documentation to `non_finite : set of int / A set of raw indices corresponding to non-finite samples` so the actual runtime contract matches what the sole caller provides.
instances: single-instance

### F5 — `_hdbscan_brute` `alpha=None` default violates its numeric contract
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`, sklearn/cluster/_hdbscan/hdbscan.py:183 documents `alpha : float, default=1.0`, and sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally executes `distance_matrix /= alpha`. Calling `_hdbscan_brute(X)` without `alpha` therefore raises `TypeError: unsupported operand type(s) for /=: … 'NoneType'` — the default value is not a legal value for the parameter's own contract.
scenario: "Any external / test caller invokes `_hdbscan_brute(X, metric='precomputed')` accepting documented defaults → immediate TypeError inside `distance_matrix /= alpha`"
contract: The default MUST be `alpha=1.0`, matching both the docstring and the corresponding parameter in `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.
instances: single-instance

### F6 — `_do_labelling` scalar-vs-array contract: `parent_lambda` compared as a scalar
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 assigns `parent_lambda = lambda_array[child_array == n]` — an untyped Python object holding a 1-D `float64` ndarray. Line 505 does `if parent_lambda >= threshold:`, which relies on the array having exactly one element (otherwise NumPy raises `ValueError: The truth value of an array with more than one element is ambiguous`). The comment at line 496 asserts this without any runtime type/shape guard, and `parent_lambda` is not `cdef`-typed as a scalar.
scenario: "A malformed condensed tree (test fixture, buggy caller, or duplicate child-row) in which more than one row has `child_array == n` → `if parent_lambda >= threshold` raises an ambiguous-truth-value ValueError inside cluster labelling"
contract: Extract a scalar explicitly (e.g. `cdef cnp.float64_t parent_lambda = lambda_array[child_array == n][0]`) and rely on the invariant, not on an implicit 1-element-array coercion.
instances: single-instance

### F7 — `test_hdbscan_precomputed_non_brute` asserts on the wrong error path via an invalid `algorithm` string [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 sets `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"`, `"prims_balltree"`), but the `algorithm` StrOptions at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only permits `{"auto", "brute", "kdtree", "balltree"}`. `_validate_params()` (called at hdbscan.py:697) therefore raises `InvalidParameterError` (a `ValueError` subclass) before the intended `"precomputed + tree"` check at lines 772-783 is ever reached. The test passes for a reason unrelated to what its docstring at lines 279-281 claims to verify.
scenario: "A future refactor removes the precomputed-vs-tree ValueError branches at hdbscan.py:772-783 → this test still passes silently, giving false confidence in the coverage of that contract"
contract: The test MUST use a valid algorithm from the StrOptions set (`"kdtree"` or `"balltree"`) so that the intended precomputed-vs-tree branch is exercised.
instances: single-instance

### F8 — `allow_single_cluster` parameter drifts across three C types on the same call chain
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:60 declares `bint allow_single_cluster=False` in `tree_to_labels`; :435 declares `cnp.intp_t allow_single_cluster` in `_do_labelling`; :580 and :608 declare `cnp.intp_t allow_single_cluster` in `traverse_upwards`/`epsilon_search`; :646 declares `cnp.uint8_t allow_single_cluster=False` in `_get_clusters`. All are called with the same Python boolean and represent the same conceptual flag.
scenario: "A future change that stores non-{0,1} in this argument → silent truthiness drift (e.g. -1 remains truthy through intp_t but wraps under uint8_t to 255, still truthy; but any refactor to an enum or tri-state will diverge silently across layers)."
contract: A single canonical type — `bint` — must be used for `allow_single_cluster` at every `cdef` boundary in `_tree.pyx`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:60, sklearn/cluster/_hdbscan/_tree.pyx:435, sklearn/cluster/_hdbscan/_tree.pyx:580, sklearn/cluster/_hdbscan/_tree.pyx:608, sklearn/cluster/_hdbscan/_tree.pyx:646]

### F9 — `HIERARCHY_t.left_node`/`right_node` are `intp_t` but MST edges are `int64_t`, narrowing on 32-bit
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pxd:34-38 declares `HIERARCHY_t` with `intp_t left_node`/`right_node`; sklearn/cluster/_hdbscan/_linkage.pyx:56-59 declares `MST_edge_t` with `int64_t current_node`/`next_node`; :263-264 assigns `single_linkage[i].left_node = current_node_cluster` and `.right_node = next_node_cluster` where the cluster IDs came from `U.fast_find(current_node)` on a `int64_t current_node`.
scenario: "Build on a 32-bit platform where `intp_t == int32_t` and `n_samples > 2**31` → silent truncation when storing MST node indices into the HIERARCHY struct."
contract: `HIERARCHY_t.left_node` and `HIERARCHY_t.right_node` must be widened to `int64_t` (with `HIERARCHY_dtype` updated to `np.int64` correspondingly) to match `MST_edge_t.current_node`/`next_node`.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:34-38, sklearn/cluster/_hdbscan/_linkage.pyx:56-59, sklearn/cluster/_hdbscan/_linkage.pyx:263-264]

### F10 — `labels_` dtype changes between finite and non-finite input paths
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:822-829 assigns `self.labels_` from `tree_to_labels` (which returns `intp` via `_do_labelling`); sklearn/cluster/_hdbscan/hdbscan.py:840 rebinds `self.labels_ = new_labels` where `new_labels = np.empty(self._raw_data.shape[0], dtype=np.int32)` only in the non-finite branch.
scenario: "Downstream consumer (or `dbscan_clustering` comparison `self.labels_ == _OUTLIER_ENCODING[…]['label']`) that assumes a stable label dtype → different dtype (`intp`/`int64` vs `int32`) between clean and non-finite fits, producing inconsistent behavior when integrating with typed pipelines or on 32-bit platforms where truncation could occur."
contract: `HDBSCAN.labels_` must be `np.intp` on every code path.
instances: single-instance

### F11 — `traverse_upwards` assigns numpy arrays to C scalar–typed locals `parent` and `parent_eps`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:582 declares `cdef cnp.intp_t root, parent` and `cdef cnp.float64_t parent_eps`; :586 assigns `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` (a 1-D ndarray); :593 assigns `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']` (a 1-D ndarray). Both rely on numpy's implicit length-1-array-to-scalar coercion.
scenario: "A cluster tree in which `cluster_tree['child'] == leaf` selects zero rows (leaf not present) → coercion raises `TypeError: only size-1 arrays can be converted to Python scalars` instead of a clear domain error; more than one row → same failure. Additionally, `parent` is then passed recursively as a `cnp.intp_t` argument, silently discarding array shape."
contract: The mask must be reduced to a scalar explicitly (e.g. `int(arr[0])` / `float(arr[0])`) with a guard that exactly one row was selected, before assignment to the C scalar locals.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593]

### F12 — `_get_clusters.child_selection` typed as `uint8_t[::1]` receives an object-dtype ndarray from `==`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:695 declares `cnp.uint8_t[::1] child_selection`; :729 assigns `child_selection = (cluster_tree['parent'] == node)` — the result of `==` on a structured-field intp array is a numpy `bool_` array (dtype `np.bool_`), not `uint8`, and typed-memoryview assignment from bool dtype to `uint8_t[::1]` is not guaranteed by the buffer protocol.
scenario: "Cython buffer acquisition on newer numpy/Cython that strictly checks buffer format `?` vs `B` → `ValueError: Buffer dtype mismatch, expected 'unsigned char' but got 'bool'` at runtime on every `fit` that hits the `eom` branch."
contract: `child_selection` must be typed as a boolean-compatible memoryview (e.g. `cnp.uint8_t[::1]` assigned from `(cluster_tree['parent'] == node).view(np.uint8)`) or declared as `cnp.npy_bool[::1]`, with the assignment made explicit.
instances: single-instance

### F13 — `_weighted_cluster_center` counts `-3` (missing) label as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; `_OUTLIER_ENCODING["missing"]["label"]` is `-3` per lines 73-78, but `-3` is not excluded from the noise-set here even though centroids/medoids should not be computed for missing samples.
scenario: "`HDBSCAN(store_centers='both').fit(X)` with X containing np.nan → `-3` is treated as a valid cluster index; `mask = self.labels_ == -3` never becomes true for `idx in range(n_clusters)` (since range uses non-negative ids), producing an off-by-one and empty averaging/inf medoid computation for the phantom cluster." 
contract: The set of "cluster-excluding" outlier labels used to compute `n_clusters` must include every label in `_OUTLIER_ENCODING`, i.e. `{-1, -2, -3}`.
instances: single-instance

### F14 — `max_distance` remains in `metric_params` and is forwarded to `pairwise_distances`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:239,243 — `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` on line 239 spreads `metric_params`; line 243 `max_distance = metric_params.get("max_distance", 0.0)` reads but does not pop. If the caller supplies `metric_params={"max_distance": ...}`, `pairwise_distances` receives the unknown keyword and propagates it to underlying metric functions.
scenario: "user passes `metric_params={'max_distance': 5.0}` with a standard metric like 'euclidean' → `pairwise_distances` raises TypeError from unknown keyword forwarded to the metric callable/scipy backend."
contract: `max_distance` must be extracted and removed from `metric_params` (e.g. via `pop`) before `metric_params` is spread to `pairwise_distances`.
instances: single-instance

### F15 — `HIERARCHY_dtype` / `CONDENSED_dtype` array shapes documented as `(n_samples,)` instead of `(n_samples - 1,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 documents `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`, sklearn/cluster/_hdbscan/_tree.pyx:138 documents `condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype`, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656 repeat the same claim. However, the actual construction in sklearn/cluster/_hdbscan/_linkage.pyx:252 uses `np.zeros(n_samples - 1, ...)` and callers such as sklearn/cluster/_hdbscan/hdbscan.py:219 correctly document shape `(n_samples - 1,)`. The Cython-side shape contract is off-by-one and disagrees with the Python-side contract.
scenario: "Consumer of `_condense_tree`/`labelling_at_cut` sizes buffers based on the documented `(n_samples,)` shape → off-by-one buffer / reader mismatch"
contract: Align the Cython docstrings to `(n_samples - 1,)` to match both the implementation and the Python-side docstring.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F16 — `test_dbscan_clustering_outlier_data` uses element-wise `+` on `np.ndarray` indices where set-union/concatenation is intended [out-of-theme]
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:212 computes `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`, where `missing_labels_idx` and `infinite_labels_idx` are `np.ndarray`s returned by `np.flatnonzero` at lines 206 and 209. `missing_labels_idx + infinite_labels_idx` performs element-wise addition (with broadcasting from shape `(1,)` to `(2,)`), producing `[2, 5]` for the specific fixture, not the intended concatenation `[2, 5, 0]`. The test only passes because the broadcast happens to leave the "missing" indices unchanged; had `infinite_labels_idx` contained a nonzero value, `clean_idx` would silently omit a different row than intended.
scenario: "Any future fixture change that puts a nonzero index into `infinite_labels_idx` → `clean_idx` is wrong, `clean_model` fits the wrong subset, and assertion at line 215 may compare different points"
contract: Concatenate the two arrays explicitly, e.g. `set(np.concatenate([missing_labels_idx, infinite_labels_idx]).tolist())`.
instances: single-instance

### F17 — `_get_clusters` `stabilities` return contract broken (docstring promises 3-tuple, code returns 2-tuple)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:681-690 documents three return values (`labels`, `probabilities`, `stabilities`), but the actual `return` at sklearn/cluster/_hdbscan/_tree.pyx:797 is `return (labels, probs)` — a 2-tuple. `tree_to_labels` at sklearn/cluster/_hdbscan/_tree.pyx:70 unpacks only two values, so the caller conforms to the implementation, but the docstring's third return (`stabilities`) is a broken contract for any external consumer of `_get_clusters`.
scenario: "External code (e.g. a follow-up feature or user script that calls `_get_clusters` directly) unpacks the documented 3-tuple → `ValueError: not enough values to unpack`"
contract: Remove the `stabilities` entry from the `_get_clusters` Returns block so the documented contract matches the actual 2-tuple return.
instances: single-instance

### F18 — `HDBSCAN` `algorithm` StrOptions omit the historical `"kd_tree"` / `"ball_tree"` names still emitted by the routing code
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:629-638 validates `algorithm` against `{"auto", "brute", "kdtree", "balltree"}`, but the routing code at sklearn/cluster/_hdbscan/hdbscan.py:798, :802, :812, :817 forwards `algo="kd_tree"` / `algo="ball_tree"` (underscored) into `NearestNeighbors(algorithm=algo)`. The user-facing API contract accepts one spelling while the internal contract to `NearestNeighbors.algorithm` requires another, and there is no cast/mapping documented; a reader/user cannot round-trip the accepted string.
scenario: "A user sets `algorithm='kd_tree'` matching the well-known `NearestNeighbors` spelling → InvalidParameterError, despite this being the exact string internally forwarded to NearestNeighbors"
contract: The parameter-constraints StrOptions and the internal dispatch strings MUST use a single, documented spelling; if `kdtree`/`balltree` is the chosen public spelling, translate exactly once at the boundary with a comment naming the sklearn API contract, and mention this translation in the docstring.
instances: single-instance

### F19 — `_compute_stability.result_pre_dict` typed as memoryview but consumed by `dict()` requiring iterable of pairs
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:246 declares `cnp.float64_t[:, :] result_pre_dict`; :269-274 assigns it via `np.vstack(...).T` and then :276 returns `dict(result_pre_dict)`. `dict()` on a typed memoryview iterates rows as memoryviews, not as 2-tuples, so the conversion depends on the underlying ndarray view being extractable — the memoryview type is documentation drift versus the actual use.
scenario: "A future Cython version where iterating a 2-D memoryview yields 1-D memoryview rows rather than ndarray rows → `dict()` fails with `TypeError: cannot convert dictionary update sequence element #0 to a sequence`."
contract: `result_pre_dict` must remain a `cnp.ndarray[cnp.float64_t, ndim=2]` (not a memoryview) since it is passed to `dict()` which relies on numpy row iteration semantics.
instances: single-instance

### F20 — `bfs_from_cluster_tree.children` typed as `intp_t` but populated from a structured field of `intp_t` via boolean-indexed lookup
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:289 declares `cnp.ndarray[cnp.intp_t, ndim=1] children = condensed_tree['child']` and :286 declares `cnp.ndarray[cnp.intp_t, ndim=1] process_queue = np.array([bfs_root], dtype=np.intp)`; :294 assigns `process_queue = children[np.isin(parents, process_queue)]`. The typed decl at :286 is `cnp.ndarray[cnp.intp_t, …]` but the reassignment does not enforce contiguity or ownership — the previous typed slot is silently rebound to a view.
scenario: "A caller relying on `process_queue` being a fresh C-contiguous array after the loop → the actual object is a numpy view into `children` with unknown contiguity, breaking any code that would forward it to another `cnp.ndarray[..., mode='c']`-typed parameter."
contract: The reassignment on :294 must explicitly materialise as `np.ascontiguousarray(...)` or the declared type at :286 must drop the C-contiguity implication.
instances: single-instance

### F21 — `mst_from_mutual_reachability`'s `mutual_reachability` typed without `mode='c'` while callee assumes row-major access
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:62 declares `cnp.ndarray[float64_t, ndim=2] mutual_reachability`; the body uses `mutual_reachability[current_node][current_labels]` at :98 — a fancy-indexing row access that only produces a contiguous 1-D view when the source is C-contiguous.
scenario: "A caller that passes a Fortran-ordered or strided distance matrix (e.g. from `pairwise_distances` with certain metric backends) → correct results but silent memory layout drift; combined with `PyArray_SHAPE(...)[0]` at :87 being used as the "rows" count, a Fortran-ordered input would compute the MST over the transpose axis."
contract: The parameter must be declared `cnp.ndarray[float64_t, ndim=2, mode='c']` so the C-contiguity is enforced at the API boundary.
instances: single-instance

### F22 — `_hdbscan_prims` / `_hdbscan_brute` signatures declare `min_samples=5` but docstring says `default=None`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160,180-181,272,290-291 — both functions have `min_samples=5` in the signature; both docstrings say `min_samples : int, default=None`.
scenario: "docs/signature mismatch → downstream reviewers or users misread the contract for private helpers, and the documented `None` value would break `NearestNeighbors(n_neighbors=None)` inside `_hdbscan_prims`."
contract: The docstring `default` for `min_samples` must match the signature's actual default (`5`).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:161, sklearn/cluster/_hdbscan/hdbscan.py:272]

### F23 — `_hdbscan_prims` documents a `copy` parameter that its signature does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278,313-318 — signature lists `X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params` with no `copy` parameter, yet the docstring includes a `copy : bool, default=False` parameter block.
scenario: "reader/refactorer relies on the documented `copy` parameter → passes `copy=...` which lands in `metric_params` (and forwards to distance metric functions), silently changing behavior or causing a TypeError."
contract: The `Parameters` section must reflect the function's actual accepted arguments — the `copy` entry must be removed (or the signature must accept it).
instances: single-instance

### F24 — `_more_tags` marks `allow_nan=True` when metric is a callable, but callables may not accept NaN
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:972-973 — `return {"allow_nan": self.metric != "precomputed"}`. For any non-precomputed metric — including user-supplied callables — the estimator advertises NaN acceptance. But NaN handling only works because `_get_finite_row_indices` filters non-finite rows before passing to the metric; if the metric is a callable that expects finite input, the tag alone does not guarantee correctness — this is fine, but the tag is asserted for all algorithms including `algorithm="kdtree"`/`"balltree"`, whose backing `NearestNeighbors` does not tolerate NaN in the query data.
scenario: "user relies on the advertised `allow_nan=True` and calls `HDBSCAN(metric='euclidean', algorithm='kdtree').fit(X_with_nan)` → NaN rows are filtered before reaching KDTree, so it works — but the underlying tag/contract makes no distinction, and `metric='precomputed'` with a sparse LIL matrix (line 738) never runs the NaN-filter path yet doesn't validate `force_all_finite=True` either." 
contract: The `allow_nan` tag should be conditional on the actual runtime input path — sparse precomputed matrices do not filter NaN, so the tag should not universally return True for non-precomputed metrics.
instances: single-instance
