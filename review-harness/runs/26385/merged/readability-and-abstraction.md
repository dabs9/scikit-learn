### F1 — `_weighted_cluster_center` breaks on non-finite input with `store_centers` [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 reduces `X = X[finite_index]` to only finite rows; sklearn/cluster/_hdbscan/hdbscan.py:840-844 rebuilds `self.labels_` to `self._raw_data.shape[0]`; sklearn/cluster/_hdbscan/hdbscan.py:855 then calls `self._weighted_cluster_center(X)` with the *reduced* X but sklearn/cluster/_hdbscan/hdbscan.py:908-909 does `mask = self.labels_ == idx` (raw length) then `data = X[mask]` (reduced length) → boolean-index length mismatch → IndexError.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_rows)` → IndexError: boolean index did not match indexed array along dimension 0."
contract: index `self._raw_data` (or the equivalent finite-only labels) inside `_weighted_cluster_center` so mask and data share a shape.
instances: single-instance

### F2 — `remap_single_linkage_tree` docstring lies about `non_finite`: says "Boolean array" but caller passes a `set`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-366 — `non_finite : ndarray\n    Boolean array of which entries in the raw data are non-finite`; sklearn/cluster/_hdbscan/hdbscan.py:838 caller passes `non_finite=set(infinite_index + missing_index)` (a set of integer indices). Inside the function (line 388-389), `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)` — `outlier` is written into the tree's `left_node` field, which only makes sense if it is an integer index, not a boolean.
scenario: "A reader takes the docstring at its word and calls `remap_single_linkage_tree(tree, mapping, non_finite=np.isnan(X.sum(1)) | np.isinf(X.sum(1)))` → outlier_tree rows get `left_node ∈ {0,1}` (True/False cast), silently corrupting the extended tree."
contract: rewrite the parameter as `non_finite : iterable of int — raw-data indices of non-finite rows`.
instances: single-instance

### F3 — HDBSCAN `n_jobs` default: signature is `4`, docstring says `None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — the class docstring reads "`n_jobs : int, default=None ... ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.``-1`` means using all processors.`" ; line 658 sets `n_jobs=4` in the signature, and line 673 stores that value on `self.n_jobs`, which is later passed straight into `NearestNeighbors`/`pairwise_distances`.
scenario: "user reads the docstring and expects the estimator to behave like every other sklearn estimator (respect `joblib.parallel_backend`); instead the constructor spawns 4 processes unconditionally → surprise CPU/memory usage and non-composition with parallel_backend contexts"
contract: Set the default to `n_jobs=None` to match the docstring and sklearn-wide convention.
instances: single-instance

### F4 — Duplicated `births = np.full(...)` allocation is dead code
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — line 252 executes `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`, then line 254 (after a blank line) immediately overwrites it with the identical assignment. The first assignment is unconditionally discarded.
scenario: "Reader tries to understand `_compute_stability` → sees two identical allocations, hunts for a subtle difference, wastes review time; every fit pays for one throwaway numpy array allocation proportional to `largest_child + 1`"
contract: Delete the redundant `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 252.
instances: single-instance

### F5 — `n_clusters` in `_weighted_cluster_center` omits the `-3` (missing) label [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; sklearn/cluster/_hdbscan/hdbscan.py:843 shows -3 is written into `labels_` for missing samples; the outlier encoding at sklearn/cluster/_hdbscan/hdbscan.py:65-79 also names -3 as an outlier.
scenario: "Data with missing rows → labels_ contains `{-1, -3, 0, ..., k-1}` → `n_clusters = k+1` → `range(n_clusters)` iterates to `idx == k` for which no row matches → `np.average` over an empty slice raises."
contract: `n_clusters = len(set(self.labels_) - {-1, -2, -3})` to exclude every outlier label defined in `_OUTLIER_ENCODING`.
instances: single-instance

### F6 — `TreeUnionFind.is_component` is written but never read
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:330 declares `cdef cnp.uint8_t[::1] is_component`; line 337 initializes it; line 355 writes `self.is_component[x] = False` inside `find`; no reader anywhere in the codebase (`grep is_component sklearn/cluster` returns only these three write/init sites).
scenario: "Reader of `TreeUnionFind.find` sees a boolean-side-effect write and assumes the flag matters for correctness, spends time reasoning about it → the flag has zero observable effect; every path-compressing call also pays for a memoryview store that only exists as noise"
contract: Remove the `is_component` attribute, its allocation in `__init__`, and its write in `find`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:330, sklearn/cluster/_hdbscan/_tree.pyx:337, sklearn/cluster/_hdbscan/_tree.pyx:355]

### F7 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist and omits `algo` / `leaf_size`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature has params `(X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params)`; sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents a full `copy` block; there is no `copy=` in the signature, and `algo` / `leaf_size` (which are required/consumed) are undocumented.
scenario: "Caller reads the docstring and passes `_hdbscan_prims(X, copy=True, ...)` → TypeError: unexpected keyword; conversely a caller looking for `algo` finds no documentation and guesses."
contract: delete the fabricated `copy` block; document `algo` and `leaf_size`.
instances: single-instance

### F8 — `_hdbscan_prims` docstring claims `metric='precomputed'` support that the code does not provide
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-283 — `Builds a single-linkage tree (SLT) from the input data \`X\`. If \`metric="precomputed"\` then \`X\` must be a symmetric array of distances.`; sklearn/cluster/_hdbscan/hdbscan.py:328-347 uses `NearestNeighbors`/`DistanceMetric.get_metric`, neither of which accepts `"precomputed"`. The estimator itself routes precomputed only through `_hdbscan_brute` (hdbscan.py:793-803).
scenario: "Reader sees `_hdbscan_prims` docstring, believes prims supports precomputed, invokes it with `metric='precomputed'` → NearestNeighbors immediately rejects the metric."
contract: strip the precomputed sentence from `_hdbscan_prims`; describe it only as the tree-based path for feature arrays.
instances: single-instance

### F9 — `_hdbscan_brute` docstring `alpha : float, default=1.0` contradicts signature `alpha=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None,`; sklearn/cluster/_hdbscan/hdbscan.py:183-184 — `alpha : float, default=1.0`; hdbscan.py:241 does `distance_matrix /= alpha` which errors on the documented "default".
scenario: "Reader calls `_hdbscan_brute(X, metric='euclidean')` per the doc's default → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`."
contract: set the signature default to `alpha=1.0` (matching the docstring and the class default).
instances: single-instance

### F10 — `plot_hdbscan.py` "scale invariance" demonstration does not scale the data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ..., parameters={"scale": scale})`. Every iteration fits and plots the *same* `X` (compare with the DBSCAN loop at examples/cluster/plot_hdbscan.py:86-89 which correctly uses `X * scale`).
scenario: "Reader runs the example to see HDBSCAN's scale-invariance advantage → three identical subplots labelled `scale=1, 0.5, 3` → the ‘proof' is inert and misleads about what invariance means."
contract: fit and plot on `X * scale`, mirroring the DBSCAN loop just above.
instances: single-instance

### F11 — Inline comment lies about outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 says "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The code immediately below (lines 842-843) actually assigns `_OUTLIER_ENCODING["infinite"]["label"]` (= -2) for inf and `_OUTLIER_ENCODING["missing"]["label"]` (= -3) for nan. The comment values (-1, -2) do not match the code (-2, -3).
scenario: "Reader trusts the comment → uses `-1` to filter np.inf samples in downstream code and gets nothing, or `-2` to filter np.nan and misses them entirely"
contract: Update the comment to "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3."
instances: single-instance

### F12 — Misspelled `mutual_reachibility_distance` local diverges from `reachability` naming
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:127, :144, :149, :182, :206, :209, :210 all use the identifier `mutual_reachibility_distance` while the file/function/module all consistently use "reachability" (see line 44 `mutual_reachability_graph`, file name `_reachability.pyx`).
scenario: "Someone greps for `reachability` in the reachability code → misses the loop bodies that actually compute the reachability distance; a copy-paste of the variable name into new call-sites propagates the misspelling"
contract: Rename the local to `mutual_reachability_distance` at every occurrence (7 lines above), matching the module and function names.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F13 — Public docstrings render "single-linkage tree tree" and misspell "reahability"/"collecteion"
severity: low
evidence: The strings appear verbatim in user-visible NumPy-style docstrings that Sphinx will render: `single-linkage tree tree` at sklearn/cluster/_hdbscan/hdbscan.py:149, :220, :326, :361 and sklearn/cluster/_hdbscan/_linkage.pyx:234; `mutual-reahability graph` at sklearn/cluster/_hdbscan/hdbscan.py:102, :143 and sklearn/cluster/_hdbscan/_linkage.pyx:75, :137, :228; `collecteion of edges` at sklearn/cluster/_hdbscan/hdbscan.py:103, :144 and sklearn/cluster/_hdbscan/_linkage.pyx:76, :138, :229; `simbling` at sklearn/cluster/_hdbscan/_tree.pyx:503 and sklearn/cluster/tests/test_hdbscan.py:530; `smaler` at sklearn/cluster/_hdbscan/_tree.pyx:133.
scenario: "Docs build publishes broken prose in official scikit-learn documentation → user searches Sphinx for `reachability` and misses these entries; documentation credibility hit"
contract: Fix each misspelling to "reachability", "collection", "sibling", "smaller", and drop the doubled "tree" in "single-linkage tree tree" — a single sweep across the enumerated line-set is sufficient.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/tests/test_hdbscan.py:530]

### F14 — `_weighted_cluster_center` pre-allocates a `mask` buffer it never uses
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 allocates `mask = np.empty((X.shape[0],), dtype=np.bool_)`. Inside the loop at line 908 the very first statement rebinds `mask = self.labels_ == idx`, discarding the pre-allocated buffer. `mask` is never read between allocation and rebinding.
scenario: "Reader sees a pre-allocated typed buffer and assumes it's part of an in-place reuse optimization → wastes time proving no reuse exists; every fit with `store_centers` also allocates and immediately abandons the buffer"
contract: Delete line 896 — the loop's `mask = self.labels_ == idx` already produces the correct array.
instances: single-instance

### F15 — `enumerate(tree)` with `_` while indexing `tree[i]` misleads the reader
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370 uses `for i, _ in enumerate(tree):` and the loop body at lines 371-381 accesses `tree[i]["left_node"]`, `tree[i]["right_node"]`, and writes `tree[i]["left_node"] = ...`. The `_` implies "value unused", but each iteration re-fetches the same row that `enumerate` already produced.
scenario: "Reader parses `for i, _ in enumerate(tree)` as an index-only walk and misses that the loop mutates `tree[i]` in place → refactoring to `for entry in tree` is tempted and silently breaks because structured-array element access is view-vs-copy dependent"
contract: Replace with `for i in range(len(tree)):` — it accurately signals index-only iteration and matches the mutation intent.
instances: single-instance

### F16 — `_tree.pyx` docstrings pin single-linkage/condensed-tree arrays to shape `(n_samples,)` when they are `n_samples-1` long
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 (`hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`), :138 (condensed_tree output), :371 (`linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`), :447, :656. The construction at sklearn/cluster/_hdbscan/_linkage.pyx:252 (`single_linkage = np.zeros(n_samples - 1, dtype=HIERARCHY_dtype)`) is the authoritative shape.
scenario: "Downstream user sizes their preallocated linkage buffer using the documented `(n_samples,)` → off-by-one when they later pass it here."
contract: use `(n_samples - 1,)` (or an explicit `(n_edges,)`) in the docstrings for the SLT and condensed-tree parameters.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F17 — `_weighted_cluster_center` ends with a stray bare `return` after a series of side-effect assignments
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:921 — the helper is documented (lines 882-887) to "instead stores them in the `self.{centroids, medoids}_` attributes" rather than returning anything, yet it ends with `return` on line 921 which returns `None`. Reads as if the helper were previously value-returning.
scenario: "reader reaches the trailing `return` on a documented mutator → wonders what value they missed and pays a small but real cognitive tax that is out of line with the surrounding style (`fit` uses `return self`; other helpers omit it)"
contract: Remove the bare `return` line.
instances: single-instance

### F18 — `_hdbscan_brute` / `_hdbscan_prims` docstring `min_samples default=None` contradicts signature `min_samples=5`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160,272 signatures both use `min_samples=5`; sklearn/cluster/_hdbscan/hdbscan.py:179-181,290-292 docstrings both claim `default=None`.
scenario: "Reader believes omitting `min_samples` uses `None` → configures unrelated logic downstream (e.g. sets min_samples from external code only if not-None) → default of 5 silently applied."
contract: change both docstrings to `default=5`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F19 — `_do_labelling` names the sample count `root_cluster`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-490 — `root_cluster = np.min(parent_array)` then used as an array size (`result = np.empty(root_cluster, ...)`) and loop bound (`for n in range(root_cluster)`); the identifier reads as "id of the root cluster", but the semantics used here are "number of samples".
scenario: "A future editor changes the labelling convention so the smallest parent is no longer equal to `n_samples` → every use of `root_cluster` as a size silently breaks because the name concealed the invariant."
contract: rename to `n_samples` (or introduce `n_samples = root_cluster` and use it in the sizing/loop) so the size and the id are not overloaded on one identifier.
instances: single-instance

### F20 — `_brute_mst`: `mutual_reachability` parameter docstring header uses a misspelled name
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82 signature has `mutual_reachability`; sklearn/cluster/_hdbscan/hdbscan.py:91-93 docstring documents it as `mututal_reachability_graph:` (typo *mututal*, and different name). Also documents `min_samples : int, default=None` for a parameter that has no signature default (sklearn/cluster/_hdbscan/hdbscan.py:82).
scenario: "Reader searches the source for `mutual_reachability` and matches only the code / not the docstring → trusts the misspelled `mututal` when following a doc-generation warning and propagates the typo into downstream references."
contract: rename the docstring parameter to `mutual_reachability : {ndarray, sparse matrix}` and remove the phantom `default=None` for `min_samples`.
instances: single-instance

### F21 — `enumerate(non_finite)` over a `set` produces non-deterministic outlier ordering
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:388-391 iterates a Python set (constructed at sklearn/cluster/_hdbscan/hdbscan.py:838 via `set(infinite_index + missing_index)`); the loop then assigns `last_cluster_id`/`last_cluster_size` in that iteration order.
scenario: "Two processes with different `PYTHONHASHSEED` fit HDBSCAN on identical data with non-finite rows → resulting `_single_linkage_tree_` outlier-appended rows differ in id assignment → downstream reproducibility checks fail even though algorithmic output is equivalent."
contract: iterate `sorted(non_finite)` (or accept a sequence in the API) so the appended outlier tree is deterministic.
instances: single-instance
