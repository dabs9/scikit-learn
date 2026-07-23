I've read the three draws. Regarding the system reminder about malware: these files are review findings documents (markdown reports about scikit-learn code quality); they contain no executable code and are not malware. My task here is a pure text-merge operation — I am not improving or augmenting any code, only unioning finding blocks verbatim as instructed.

Now I'll produce the merged UNION document. Grouping same-defect findings across draws:

- n_jobs=4 default: A-F11(high), B-F1(medium), C-F1(medium) → same defect; A most specific (highest severity, richest evidence)
- alpha=None default: A-F6(high), B-F2(medium) → A most specific
- _weighted_cluster_center n_clusters missing -3: A-F12(medium), B-F5(high), C-F3(high) → B most specific
- Comment misdocuments outlier mapping: A-F1, C-F2 → A slightly more specific
- Scale-invariance demo: A-F3, C-F13 → A more specific
- births duplicate allocation: A-F7, B-F6, C-F5 → A slightly more specific
- _do_labelling result sizing: A-F8, B-F9 → A more specific
- _do_labelling untyped label/parent_lambda: A-F15, B-F10, C-F6 → B most specific (includes runtime-crash scenario)
- traverse_upwards scalar/array: A-F14, B-F17 → A more specific (broader instances)
- test_hdbscan_precomputed_non_brute: A-F2, C-F7 → C more specific
- _get_finite_row_indices sparse: A-F10, B-F12 → A more specific
- Typos/misspellings: B-F8, C-F12 (B-F7 is a subset of B-F8; both preserved as separate blocks per rules since B-F7 is a distinct finding block in draw B). B and C both cover typos but B-F8 is more comprehensive → keep B-F8; C-F12 also more specific about identifier renaming; treat as same defect → B-F8 most specific. B-F7 (test_reachibility.py rename) is a subset but was reported as separate block by draw B; per rules "same defect only if root cause AND site match" — B-F7 targets the file rename specifically and is included within B-F8's instances too, so same root cause. Keep B-F8 (more specific, includes B-F7's site).

All others are unique.

Ordering: high, then medium, then low.

Highs: A-F6 (alpha), B-F5 (n_clusters -3)
Mediums: A-F11 (n_jobs), C-F7 (precomputed_non_brute test), A-F3 (scale-invariance out-of-theme), A-F8 (root_cluster sizing), A-F10 (finite_row_indices sparse), A-F14 (traverse_upwards), A-F16 (rtol=1), B-F4 (medoid weighting), B-F11 (set iteration), B-F19 (dbscan_clustering out-of-theme), C-F9 (ndarray + concat)
Lows: everything else

### F1 — `_hdbscan_brute` accepts `alpha=None` default but divides by it
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None,`; line 241 `distance_matrix /= alpha` will raise `TypeError` when `alpha=None`. Docstring at line 183 lies about the default: "alpha : float, default=1.0".
scenario: "any caller not passing `alpha` (only the class currently does at line 767) → `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`. The API contract of the helper is broken and the docstring is a lying default."
contract: Change the signature to `alpha=1.0` to match the documented default and the actual class default.
instances: single-instance

### F2 — `_weighted_cluster_center` computes `n_clusters` ignoring the `-3` (missing) label but not consistently
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. But `_OUTLIER_ENCODING["missing"]["label"] = -3` (line 74) is a documented outlier label that can appear in `self.labels_` (see lines 843, 957). Because `-3` is not subtracted, `n_clusters` is incorrectly inflated by one when any missing samples are present, producing `centroids_`/`medoids_` arrays that have a spurious last row that is never written to (loop `for idx in range(n_clusters)` iterates 0..n_clusters−1, none of which equal `-3`).
scenario: "Fit on data with `np.nan` rows and `store_centers='centroid'` → `self.centroids_` has an extra uninitialized row, silently exposing garbage values as a valid centroid"
contract: Change to `n_clusters = len(set(self.labels_) - set(out['label'] for out in _OUTLIER_ENCODING.values()) - {-1})` (or equivalently subtract `{-1, -2, -3}`).
instances: single-instance

### F3 — `n_jobs` default hard-codes 4 workers, contradicting docstring
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,`; docstring at lines 486-490 says `n_jobs : int, default=None` and "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all processors."
scenario: "user constructs `HDBSCAN()` expecting sklearn convention (`n_jobs=None`) → estimator silently spawns 4 workers regardless of the surrounding `joblib.parallel_backend`; docstring is a lying default and the class violates sklearn's `n_jobs` convention."
contract: Change the signature default to `n_jobs=None` to match the documented behavior and sklearn convention.
instances: single-instance

### F4 — `test_hdbscan_precomputed_non_brute` asserts the wrong error path
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:277-284 — the test constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` where `algorithm` is `"prims_kdtree"`/`"prims_balltree"`. Those values are not in the `StrOptions({"auto","brute","kdtree","balltree"})` constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-638, so the test passes only because `_validate_params` raises `InvalidParameterError` (a `ValueError` subclass) — never exercising the intended `algorithm` vs `metric="precomputed"` incompatibility branch. The test's docstring says it verifies rejection of a tree-based algorithm with precomputed data; it actually verifies rejection of an unknown algorithm string.
scenario: "developer removes/relaxes the algorithm-name enum → test still passes because it never reaches the metric-incompatibility check, so the actual invariant is silently unguarded"
contract: Use the valid names `"kdtree"`/`"balltree"` in the parametrization and assert on the metric-incompatibility error message.
instances: single-instance

### F5 — Scale-invariance loop never applies the scale [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ...)`; the DBSCAN analogue immediately above at lines 87-89 correctly uses `dbs.fit(X * scale)` and `plot(X * scale, ...)`.
scenario: "user reads the HDBSCAN scale-invariance demo → all three subplots show identical clustering of un-scaled `X` (misleadingly implying scale invariance from a degenerate experiment), instead of the intended comparison of clustering the scaled data."
contract: Fit and plot on `X * scale` (mirroring the DBSCAN block): `hdb.fit(X * scale); plot(X * scale, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`.
instances: single-instance

### F6 — `_do_labelling` sizes `result` by root cluster id, not sample count
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-481 — `root_cluster = np.min(parent_array); result = np.empty(root_cluster, dtype=np.intp)`. `root_cluster` is the smallest parent id (== `n_samples`) by convention in the condensed-tree encoding, so this is a surprising cross-reference that reads as "allocate a cluster-id-many array" but actually happens to equal `n_samples`.
scenario: "someone changes the condensed-tree id offset (e.g. renumbering conventions) → silent wrong allocation size; the invariant 'smallest parent id == n_samples' is nowhere stated in code or comments and is not defensively asserted."
contract: Name/derive `n_samples` explicitly (as done elsewhere, e.g. line 90) and allocate `result = np.empty(n_samples, dtype=np.intp)`, with an assertion or comment pinning the invariant.
instances: single-instance

### F7 — `_get_finite_row_indices` sparse branch surprises on non-LIL input
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 — for sparse input the function calls `matrix.tolil().data` and iterates rows; every call on a CSR matrix silently converts (LIL construction cost) and iterates the explicit stored values only. Rows without stored `nan`/`inf` in non-stored positions are treated as finite. The function name promises "purely finite rows" but implicit zeros are considered finite by construction (which is correct in a distance-matrix context but not stated). The function is called on both sparse feature matrices and sparse precomputed distance matrices (via `fit`).
scenario: "sparse feature matrix (not distance matrix) with an explicit `nan` stored in a row but implicit zeros elsewhere → the row is dropped, but a row with sparsity-implied zeros that 'should' be non-finite is kept. Silent surprising side effect due to a name that overstates the semantics."
contract: Rename or scope the helper (e.g. `_get_finite_row_indices_of_stored_entries`) and document that only stored values are inspected for sparse inputs.
instances: single-instance

### F8 — `traverse_upwards` compares scalar to array from boolean-indexing without `.item()`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:586-593 — `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` yields a 1-D ndarray, then compared `if parent == root:` (elementwise, returns an ndarray) and later `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']`. Cython typing on line 582 declares `cnp.intp_t parent` but the RHS is an array; Cython will attempt scalar coercion (works only when the array happens to have length 1). Similar pattern in `_do_labelling` at line 498 (`parent_lambda = lambda_array[child_array == n]`) and in `_get_clusters` (line 745 `eom_clusters[0]`). This is speculative abstraction: leans on hidden "always length 1" invariants.
scenario: "any condensed_tree encoding where `child == leaf` matches 0 or >1 rows (e.g. malformed input, or a future encoding change) → silent wrong comparison or crash; behavior is not documented at the function boundary."
contract: Extract the scalar explicitly (e.g. `.item()` on a `[0]`-indexed sub-array) and document/assert the "exactly one row" invariant at the call site.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593, sklearn/cluster/_hdbscan/_tree.pyx:498]

### F9 — `test_hdbscan_centers` accepts `rtol=1` which admits ~100% error
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:319-320 — `assert_allclose(center, centroid, rtol=1, atol=0.05)` with `rtol=1` means a centroid up to `1.0 * |center| + 0.05` off passes; for `center=(3.0,3.0)` that permits a centroid anywhere in `[0, 6]` per coordinate.
scenario: "regression that pushes centroids drastically toward the wrong cluster → test still passes; the assertion name promises verification, but the tolerance renders it a lying check."
contract: Use a small `rtol` (e.g. `rtol=0.1`) that actually constrains centroid quality on the known blob geometry.
instances: single-instance

### F10 — `_weighted_cluster_center` docstring name misrepresents medoid computation
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:915-919 — for medoid: `dist_mat = pairwise_distances(data, metric=..., ...); dist_mat = dist_mat * strength; medoid_index = np.argmin(dist_mat.sum(axis=1))`. Here `strength` (`probabilities_[mask]`, shape `(n,)`) broadcasts across columns of `dist_mat`, weighting each source-point's contribution rather than each candidate's — that is neither the traditional medoid nor a standard weighted-medoid definition, yet the method name is `_weighted_cluster_center` and the docstring at line 517 says "the point in the fitted data which minimizes the distance to all other points in the cluster."
scenario: "User relies on the documented medoid semantics → gets a probability-tilted result that shifts toward points that are far from low-probability samples, without any documentation of that behavior"
contract: Either drop `strength` from the medoid computation to match the documented "minimizes distance to all other points" definition, or clearly document the weighting scheme.
instances: single-instance

### F11 — `remap_single_linkage_tree` iterates a `set` for positional assignment
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:388 — `for i, outlier in enumerate(non_finite):` where `non_finite` is passed as `set(infinite_index + missing_index)` (line 838). Sets have no defined iteration order across Python runs; the loop writes rows in that order and assigns `last_cluster_id += 1` / `last_cluster_size += 1` per row, so the encoded merge order and the row-to-sample mapping are non-deterministic across processes.
scenario: "Two runs with `PYTHONHASHSEED` differing → `self._single_linkage_tree_` has the same rows but in different order, so any downstream code that indexes by row position produces different results"
contract: Convert to a sorted list before iteration, e.g. `for i, outlier in enumerate(sorted(non_finite)):`, and change the parameter contract to accept an ordered container.
instances: single-instance

### F12 — `dbscan_clustering` docstring promises `-3` handling that the code never performs [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:955-958 documents that returned labels include "-3" for missing samples; lines 960-969 compute `labels = labelling_at_cut(self._single_linkage_tree_, cut_distance, min_cluster_size)`, whose result has length `n_samples` of the tree *including* any remapped outlier rows (via `remap_single_linkage_tree`), then overwrites `labels[missing_index]` with `-3` and `labels[infinite_index]` with `-2` using boolean masks derived from `self.labels_`. But `labels` from `labelling_at_cut` has length equal to the tree's `n_samples`, which after remapping equals `self._raw_data.shape[0]` — this coincidence is undocumented and unenforced; if `fit` was called with `metric='precomputed'`, no remap happens, `_raw_data` is undefined, and this method still runs — but `self.labels_` was never populated with `-3`, so the missing/infinite overwrites become no-ops silently.
scenario: "User calls `fit` with `metric='precomputed'` then `dbscan_clustering(...)` → returned labels do not contain the `-3` value the docstring promises for missing samples"
contract: Explicitly propagate missing-sample state independent of `self.labels_` so `dbscan_clustering` always emits `-3` where the docstring promises it.
instances: single-instance

### F13 — Test uses ndarray `+` for concatenation, silently exercising broadcasting
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` and `infinite_labels_idx` come from `np.flatnonzero(...)` (lines 206/209) and are ndarrays. `ndarray + ndarray` broadcasts elementwise; with shapes `(2,)` and `(1,)` here it happens to return a length-2 array of sums, not the concatenation the reader assumes.
scenario: "future edit changes seed or outlier layout so `missing_labels_idx` and `infinite_labels_idx` have incompatible non-broadcastable shapes → test crashes with a broadcasting error unrelated to the tested behavior; today it silently uses wrong indices in the 'clean' baseline"
contract: Use `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or `.tolist() + .tolist()`) for index-set arithmetic.
instances: single-instance

### F14 — Comment in `fit` misdocuments outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — "# Remap indices to align with original data in the case of / # non-finite entries. Samples with np.inf are mapped to -1 and / # those with np.nan are mapped to -2."
scenario: "reader trusts the comment → reader believes np.inf → label -1 and np.nan → label -2, but `_OUTLIER_ENCODING` in the same file (lines 65-79) actually assigns np.inf → -2 and np.nan → -3, which is what the code at lines 842-843 does; comment is a lying name for the behavior."
contract: Update the comment to match `_OUTLIER_ENCODING`: np.inf → -2 (infinite), np.nan → -3 (missing).
instances: single-instance

### F15 — `remap_single_linkage_tree` docstring omits its return value
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-393 — docstring documents Parameters but no `Returns` section; the function does return `tree` (line 393) and callers rely on the return value (line 834 `self._single_linkage_tree_ = remap_single_linkage_tree(...)`).
scenario: "reader/API user reads only the docstring → assumes the mutating name means in-place, misses that a fresh, concatenated array is returned."
contract: Add a `Returns` section documenting the returned `ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` and clarify it is not an in-place update.
instances: single-instance

### F16 — `_brute_mst` docstring `min_samples` default lies
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:95 — "min_samples : int, default=None" but the signature at line 82 is `def _brute_mst(mutual_reachability, min_samples):` (required positional, no default).
scenario: "reader trusts docstring default → believes calling `_brute_mst(mr)` is valid; actually raises `TypeError` for missing positional arg. Also parroted by `_hdbscan_brute` (line 179) and `_hdbscan_prims` (line 290) where the signatures use `min_samples=5`."
contract: Remove the "default=None" claim (or specify the correct defaults matching each function's signature).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:95, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F17 — Overwritten `births` allocation is dead code
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` followed immediately by `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`; the first allocation is unconditionally overwritten before use.
scenario: "reader tries to understand initialization → confused by the duplication; small allocation cost happens twice on every clustering."
contract: Delete the duplicate line 252 (or 254).
instances: single-instance

### F18 — `_dense_mutual_reachability_graph` is silently non-symmetric on non-symmetric input
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:130-149 — comment says "We assume that the distance matrix is symmetric" but the algorithm only computes `core_distances` row-wise via `np.partition` on each row. If the caller violates the unstated symmetry precondition, results are silently wrong; there is no assertion or docstring warning at the function boundary (in contrast to `_hdbscan_brute` which does check symmetry only when `metric="precomputed"`).
scenario: "internal caller passes a non-symmetric matrix (e.g. future refactor bypasses the `_hdbscan_brute` symmetry check) → silently wrong core distances and mutual reachabilities, no error surfaced."
contract: Move the symmetry precondition into the function docstring's Parameters section so the invariant is stated at the function boundary rather than relying on a hidden caller-established assumption.
instances: single-instance

### F19 — `mutual_reachability_` variable's trailing underscore is meaningless noise
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:251 — `mutual_reachability_ = mutual_reachability_graph(...)` — local variable named with trailing underscore normally reserved (in sklearn convention) for fitted-attribute names on estimators, not local variables; misleading nomenclature.
scenario: "reader scanning for estimator attributes → is briefly misled into thinking `mutual_reachability_` is set on `self`; churns cognitive load. Trivial but a false abstraction signal."
contract: Rename to `mutual_reachability` (no trailing underscore) since it is a local.
instances: single-instance

### F20 — `_do_labelling` uses implicitly-typed Python objects `label` and `parent_lambda` inside a hot `cdef` loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:492-508 — `label = NOISE`, `parent_lambda = lambda_array[child_array == n]`, and `if parent_lambda >= threshold:` are all untyped in the `cdef` block at line 467-474. `label` therefore boxes each per-sample write; `parent_lambda` is a numpy array compared with a scalar via `>=` — this yields a numpy bool array whose truthiness in an `if` triggers a `DeprecationWarning`/future error for non-length-1 arrays.
scenario: "Under `allow_single_cluster=True` with a condensed tree where the same `child==n` appears twice → `parent_lambda` becomes length-2 and `if parent_lambda >= threshold` raises `ValueError: The truth value of an array with more than one element is ambiguous`"
contract: Declare `cdef cnp.intp_t label` and extract a scalar for `parent_lambda` (e.g. `parent_lambda = lambda_array[child_array == n][0]`) with the length-1 assumption made explicit.
instances: single-instance

### F21 — `_condense_tree` writes bool dtype into a `uint8` view
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:151, 161 — `cnp.uint8_t[::1] ignore` then `ignore = np.zeros(len(node_list), dtype=bool)`. `np.bool_` and `np.uint8` are distinct dtypes; the memoryview assignment relies on identical itemsize but the surface-level type mismatch obscures intent (why declare `uint8` and initialize with `bool`?).
scenario: "future NumPy where `bool` dtype layout changes, or a maintainer refactoring the memoryview annotation → silent breakage of a bare-metal assumption; comment/name does not explain the intentional aliasing."
contract: Use `dtype=np.uint8` in the `np.zeros` call to match the declared memoryview type.
instances: single-instance

### F22 — `n_samples` docstring shape mismatches actual shape for tree/condensed arrays
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129, 138, 447, 656 — Parameters/Returns say `ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` (or `CONDENSED_dtype`), yet the actual `HIERARCHY_t` array is size `n_samples - 1` (single linkage tree encoding) and the condensed tree is size variable ≠ `n_samples`. `hdbscan.py:101, 148, 219, 325, 360` correctly document the linkage tree shape as `(n_samples - 1,)`, so this is inconsistent documentation, not a naming choice.
scenario: "user or maintainer following the `_tree.pyx` docstring → allocates or slices with wrong length; the array-shape claim is a lying contract at the function boundary."
contract: Update `_tree.pyx` docstrings to state `(n_samples - 1,)` for `HIERARCHY_dtype` arrays and clarify that `CONDENSED_dtype` arrays have variable length (edge count of the condensed tree).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F23 — `_process_mst` docstring claims MST is sorted in-place; caller may not expect side effect
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:151-155 — `row_order = np.argsort(min_spanning_tree["distance"]); min_spanning_tree = min_spanning_tree[row_order]`. Docstring at lines 136-149 says "The MST is first sorted then processed" but does not clarify whether the caller's MST array is mutated. `argsort` + fancy indexing produces a new array (safe here), so the docstring's phrasing "MST is first sorted" is a surprising in-place-sounding side-effect promise that is not actually delivered — a mild lying name for the operation.
scenario: "reader expecting in-place mutation (per the wording) → misuses the return value or the argument; no functional bug today, but the doc/code split is misleading."
contract: Reword the docstring to "A sorted copy of the MST edges is used internally" or similar.
instances: single-instance

### F24 — `_get_clusters` reuses `n_samples` name for the largest per-point child id
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:714 — `n_samples = np.max(condensed_tree[condensed_tree['cluster_size'] == 1]['child']) + 1`. Elsewhere `n_samples` is derived from `hierarchy.shape[0] + 1` (line 90/146) or `root // 2 + 1` (line 397). Here it happens to equal the true sample count only if every sample appears as a leaf in the condensed tree — a fragile invariant that is never asserted.
scenario: "condensed tree that (post-condensation) does not include every sample as a leaf → `n_samples` under-counted, downstream `max_cluster_size` sentinel at line 717 (`n_samples + 1`) becomes wrong and can silently trigger the `cluster_sizes[node] > max_cluster_size` branch at line 733."
contract: Pass or derive the true `n_samples` from the caller (e.g., single-linkage tree length + 1) instead of inferring it from a max-child computation.
instances: single-instance

### F25 — `_hdbscan_prims` docstring documents non-existent `copy` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False`; the function signature at lines 269-278 has no `copy` parameter, and no `copy` usage inside the function.
scenario: "User reading the source docstring expects `_hdbscan_prims` to accept `copy` → passes `copy=True` and gets an unexpected TypeError"
contract: Remove the `copy` parameter section from the `_hdbscan_prims` docstring.
instances: single-instance

### F26 — Misspelled test module filename `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — file name misspells "reachability" as "reachibility"; the module under test is `_reachability.pyx`.
scenario: "Developer greps for `test_reachability` → finds nothing, believes reachability code is untested"
contract: Rename the file to `test_reachability.py`.
instances: single-instance

### F27 — Pervasive misspellings in public docstrings ("mututal", "reahability", "collecteion", "smaler", "simbling", "reachibility")
severity: low
evidence: multiple docstrings and identifiers, verified via grep:
- `mututal_reachability_graph` (Returns section label) at sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/_reachability.pyx:76
- `mutual-reahability` at sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:143; sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:228
- `collecteion` at sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:144; sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:229
- `smaler` at sklearn/cluster/_hdbscan/_tree.pyx:133
- `simbling` at sklearn/cluster/_hdbscan/_tree.pyx:503
- Cython identifier `mutual_reachibility_distance` at sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210
- misspelled test module filename at sklearn/cluster/_hdbscan/tests/test_reachibility.py:1
scenario: "User-visible Sphinx docs render the misspelled Returns label and prose → damages perceived quality of the new estimator"
contract: Fix each misspelling to the correct English form ("mutual", "reachability", "collection", "smaller", "sibling").
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_reachability.pyx:76, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/tests/test_reachibility.py:1]

### F28 — `HDBSCAN.fit` sets `self._raw_data = X` before dropping non-finite rows, silently exposing pre-validation data as a private attribute
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:707 stores the validated-but-still-non-finite X as `self._raw_data`; downstream code at lines 840, 846 uses `self._raw_data.shape[0]` to size relabelled outputs. Name suggests "raw" but it is post-`_validate_data`, and it is set only on the non-precomputed branch (undefined on the precomputed branch).
scenario: "Third-party subclass that accesses `self._raw_data` after `fit(X, metric='precomputed')` → AttributeError"
contract: Always set `self._raw_data` in `fit` (with a comment about scope) so the attribute is defined on every code path.
instances: single-instance

### F29 — `TreeUnionFind.find` overwrites `is_component` with `False` for every non-root node but nothing reads it
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:337 initializes `self.is_component = np.ones(size, dtype=np.uint8)`, and line 355 sets `self.is_component[x] = False` during path compression, yet `is_component` is never read anywhere in `_tree.pyx` (verified by grep of the file). This is speculative abstraction that also introduces a surprising side effect in an otherwise-pure `find`.
scenario: "Reader assumes `is_component` is load-bearing → wastes time tracing state that never influences output"
contract: Remove the `is_component` field and its side-effect write in `find`.
instances: single-instance

### F30 — `bfs_from_hierarchy` returns internal nodes offset by `-n_samples` interleaved with raw sample ids
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:99-103 — `process_queue = [x - n_samples for x in process_queue if x >= n_samples]` mutates `process_queue` from "tree indexes" to "hierarchy row indexes" mid-iteration, but the earlier `result.extend(process_queue)` at line 96 stored the un-offset values. Callers therefore receive a mixed list where entries `>= n_samples` are original tree ids and entries `< n_samples` are leaf sample ids; the "internal" indices used inside the loop are silently converted. No docstring warns of this dual coordinate system.
scenario: "Maintainer relies on the return being uniform tree-space ids → indexes `hierarchy[node - n_samples]` for leaves and gets a negative index / silent misread"
contract: Document (or explicitly convert) the mixed-coordinate return; e.g. "returned nodes are in tree-index space (leaf ids in `[0, n_samples)`, cluster ids in `[n_samples, 2*n_samples-1]`)".
instances: single-instance

### F31 — `_weighted_cluster_center` allocates `mask` before the loop, then immediately rebinds it
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 `mask = np.empty((X.shape[0],), dtype=np.bool_)` is never read; line 908 `mask = self.labels_ == idx` rebinds every iteration.
scenario: "Reader wonders whether the pre-allocation is used as an out-parameter for a Cython routine → wasted analysis"
contract: Delete line 896.
instances: single-instance

### F32 — `_get_clusters` uses `@cython.wraparound(True)` for the entire function, silently disabling a fast-path optimization
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:641 — `@cython.wraparound(True)` applied at function scope while the file has no module-level `wraparound(False)`; the decorator is only necessary for the single `node_list[-1]` access at line 724. Applying it at function scope makes every array read pay the wraparound overhead and hides which line actually needs negative indexing.
scenario: "Future maintainer removes the `[-1]` use → the decorator survives as cargo-cult noise, silently preventing later `wraparound(False)` optimizations"
contract: Replace the function-scope decorator with a `with cython.wraparound(True):` block around the single `node_list[-1]` access.
instances: single-instance

### F33 — `mst_from_data_matrix` initializes `current_sources` to `1` (misleading sentinel)
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:160 — `current_sources = np.ones(n_samples, dtype=np.int64)`. Nothing about "1" is meaningful as a source-of-source-node sentinel; before line 204 (`current_sources[j] = current_node`) writes to a slot, any read at line 179 (`next_node_source = current_sources[j]`) returns a bogus `1`. This value can then be assigned into `mst[i].current_node` via lines 197-199 if `next_node_min_reach` remains `INFTY`/other stale value. A lying name/initialization: it pretends every node's source is node 1.
scenario: "reader auditing correctness on the first outer iteration (i=0) reads `current_sources[j]` before any write → cannot distinguish 'source=1 because we chose it' from 'source=1 because it was the fill value', obscuring the read-before-write invariant"
contract: Initialize `current_sources` to `0` (matching the initial `current_node=0`) and document the invariant that reads only happen after a write.
instances: single-instance

### F34 — `p=None` passed to `NearestNeighbors` in `_hdbscan_prims`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:339 — `NearestNeighbors(..., p=None)`. `NearestNeighbors.p` documents `float, default=2`; passing `None` is a surprising side-input that bypasses the sensible default. There is no comment explaining the intent (presumably "let `metric_params` supply p, don't double-count") — a speculative abstraction whose contract is undocumented.
scenario: "future upgrade of `NearestNeighbors` tightens `p` validation to reject `None` → HDBSCAN breaks for every tree algorithm without a code change here"
contract: Omit the `p=None` kwarg and let the default apply, or document the invariant tying `p` handling to `metric_params`.
instances: single-instance

### F35 — Duplicated/near-identical `_hdbscan_brute` and `_hdbscan_prims` docstrings mis-describe parameters
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-327 — `_hdbscan_prims` docstring documents `metric` "must be one of the options allowed by pairwise_distances" (line 297-301), but the function actually uses `NearestNeighbors` + `DistanceMetric` (line 332-344) which have different valid-metric sets; the same docstring includes a `copy` parameter (line 313-318) that the function does not accept. Speculative-abstraction copy from `_hdbscan_brute`.
scenario: "user picks a metric valid for `pairwise_distances` but not for `KDTree`/`BallTree` → fails; user reads about `copy` behaviour but the parameter is silently absent"
contract: Rewrite the `_hdbscan_prims` docstring to describe only its actual parameters and the actual valid-metric constraint.
instances: single-instance

### F36 — `mst_from_mutual_reachability` starts from a hard-coded `current_node = 0` with no rationale
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:92 — Prim's is seeded at node 0 unconditionally; same in `mst_from_data_matrix` at line 162. No comment states that Prim's MST is invariant to the starting node (which is true for connected graphs but non-obvious to Cython readers), and there is no assertion that the graph is connected.
scenario: "reader trying to understand seed-node semantics finds a bare magic 0 → wonders whether cluster ordering depends on it"
contract: Add a single-line comment stating "Prim's is invariant to the starting node; we pick 0" or make the seed a named constant.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:92, sklearn/cluster/_hdbscan/_linkage.pyx:162]
