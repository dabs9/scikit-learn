Not applicable. I have enough evidence to enumerate findings. Emitting the deliverable now.

### F1 — `remap_single_linkage_tree` docstring lies about `non_finite`: says "Boolean array" but caller passes a `set`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-366 — `non_finite : ndarray\n    Boolean array of which entries in the raw data are non-finite`; sklearn/cluster/_hdbscan/hdbscan.py:838 caller passes `non_finite=set(infinite_index + missing_index)` (a set of integer indices). Inside the function (line 388-389), `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)` — `outlier` is written into the tree's `left_node` field, which only makes sense if it is an integer index, not a boolean.
scenario: "A reader takes the docstring at its word and calls `remap_single_linkage_tree(tree, mapping, non_finite=np.isnan(X.sum(1)) | np.isinf(X.sum(1)))` → outlier_tree rows get `left_node ∈ {0,1}` (True/False cast), silently corrupting the extended tree."
contract: rewrite the parameter as `non_finite : iterable of int — raw-data indices of non-finite rows`.
instances: single-instance

### F2 — HDBSCAN `n_jobs` default: signature is `4`, docstring says `None`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,`; sklearn/cluster/_hdbscan/hdbscan.py:486-490 docstring — `n_jobs : int, default=None\n    ... ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.`
scenario: "User instantiates `HDBSCAN()` expecting sklearn's usual `n_jobs=None` (single-threaded outside joblib context) → estimator silently spawns 4 workers on every fit, contradicting docs and sklearn convention."
contract: set the signature default to `n_jobs=None` to match the documented behaviour.
instances: single-instance

### F3 — `_weighted_cluster_center` breaks on non-finite input with `store_centers` [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733 reduces `X = X[finite_index]` to only finite rows; sklearn/cluster/_hdbscan/hdbscan.py:840-844 rebuilds `self.labels_` to `self._raw_data.shape[0]`; sklearn/cluster/_hdbscan/hdbscan.py:855 then calls `self._weighted_cluster_center(X)` with the *reduced* X but sklearn/cluster/_hdbscan/hdbscan.py:908-909 does `mask = self.labels_ == idx` (raw length) then `data = X[mask]` (reduced length) → boolean-index length mismatch → IndexError.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_rows)` → IndexError: boolean index did not match indexed array along dimension 0."
contract: index `self._raw_data` (or the equivalent finite-only labels) inside `_weighted_cluster_center` so mask and data share a shape.
instances: single-instance

### F4 — `n_clusters` in `_weighted_cluster_center` omits the `-3` (missing) label [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; sklearn/cluster/_hdbscan/hdbscan.py:843 shows -3 is written into `labels_` for missing samples; the outlier encoding at sklearn/cluster/_hdbscan/hdbscan.py:65-79 also names -3 as an outlier.
scenario: "Data with missing rows → labels_ contains `{-1, -3, 0, ..., k-1}` → `n_clusters = k+1` → `range(n_clusters)` iterates to `idx == k` for which no row matches → `np.average` over an empty slice raises."
contract: `n_clusters = len(set(self.labels_) - {-1, -2, -3})` to exclude every outlier label defined in `_OUTLIER_ENCODING`.
instances: single-instance

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist and omits `algo` / `leaf_size`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature has params `(X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params)`; sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents a full `copy` block; there is no `copy=` in the signature, and `algo` / `leaf_size` (which are required/consumed) are undocumented.
scenario: "Caller reads the docstring and passes `_hdbscan_prims(X, copy=True, ...)` → TypeError: unexpected keyword; conversely a caller looking for `algo` finds no documentation and guesses."
contract: delete the fabricated `copy` block; document `algo` and `leaf_size`.
instances: single-instance

### F6 — `_hdbscan_prims` docstring claims `metric='precomputed'` support that the code does not provide
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-283 — `Builds a single-linkage tree (SLT) from the input data \`X\`. If \`metric="precomputed"\` then \`X\` must be a symmetric array of distances.`; sklearn/cluster/_hdbscan/hdbscan.py:328-347 uses `NearestNeighbors`/`DistanceMetric.get_metric`, neither of which accepts `"precomputed"`. The estimator itself routes precomputed only through `_hdbscan_brute` (hdbscan.py:793-803).
scenario: "Reader sees `_hdbscan_prims` docstring, believes prims supports precomputed, invokes it with `metric='precomputed'` → NearestNeighbors immediately rejects the metric."
contract: strip the precomputed sentence from `_hdbscan_prims`; describe it only as the tree-based path for feature arrays.
instances: single-instance

### F7 — `_hdbscan_brute` docstring `alpha : float, default=1.0` contradicts signature `alpha=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None,`; sklearn/cluster/_hdbscan/hdbscan.py:183-184 — `alpha : float, default=1.0`; hdbscan.py:241 does `distance_matrix /= alpha` which errors on the documented "default".
scenario: "Reader calls `_hdbscan_brute(X, metric='euclidean')` per the doc's default → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`."
contract: set the signature default to `alpha=1.0` (matching the docstring and the class default).
instances: single-instance

### F8 — `_hdbscan_brute` / `_hdbscan_prims` docstring `min_samples default=None` contradicts signature `min_samples=5`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160,272 signatures both use `min_samples=5`; sklearn/cluster/_hdbscan/hdbscan.py:179-181,290-292 docstrings both claim `default=None`.
scenario: "Reader believes omitting `min_samples` uses `None` → configures unrelated logic downstream (e.g. sets min_samples from external code only if not-None) → default of 5 silently applied."
contract: change both docstrings to `default=5`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F9 — `plot_hdbscan.py` "scale invariance" demonstration does not scale the data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ..., parameters={"scale": scale})`. Every iteration fits and plots the *same* `X` (compare with the DBSCAN loop at examples/cluster/plot_hdbscan.py:86-89 which correctly uses `X * scale`).
scenario: "Reader runs the example to see HDBSCAN's scale-invariance advantage → three identical subplots labelled `scale=1, 0.5, 3` → the ‘proof' is inert and misleads about what invariance means."
contract: fit and plot on `X * scale`, mirroring the DBSCAN loop just above.
instances: single-instance

### F10 — Dead `mask` allocation in `_weighted_cluster_center`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)` is never read; sklearn/cluster/_hdbscan/hdbscan.py:908 unconditionally rebinds `mask = self.labels_ == idx`.
scenario: "Reader assumes the pre-allocation is a memory optimisation feeding an in-place fill → hunts for the missing update → wasted attention on speculative code."
contract: delete line 896.
instances: single-instance

### F11 — Duplicated `births` initialisation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two consecutive `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` statements with a blank line between them.
scenario: "Reader wonders whether the two identical lines are hiding different intent (e.g. a mid-computation reset) → chases a phantom → the second is just leftover after a refactor."
contract: delete one of the two initialisations.
instances: single-instance

### F12 — `_do_labelling` names the sample count `root_cluster`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:480-490 — `root_cluster = np.min(parent_array)` then used as an array size (`result = np.empty(root_cluster, ...)`) and loop bound (`for n in range(root_cluster)`); the identifier reads as "id of the root cluster", but the semantics used here are "number of samples".
scenario: "A future editor changes the labelling convention so the smallest parent is no longer equal to `n_samples` → every use of `root_cluster` as a size silently breaks because the name concealed the invariant."
contract: rename to `n_samples` (or introduce `n_samples = root_cluster` and use it in the sizing/loop) so the size and the id are not overloaded on one identifier.
instances: single-instance

### F13 — `_tree.pyx` docstrings pin single-linkage/condensed-tree arrays to shape `(n_samples,)` when they are `n_samples-1` long
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 (`hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`), :138 (condensed_tree output), :371 (`linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`), :447, :656. The construction at sklearn/cluster/_hdbscan/_linkage.pyx:252 (`single_linkage = np.zeros(n_samples - 1, dtype=HIERARCHY_dtype)`) is the authoritative shape.
scenario: "Downstream user sizes their preallocated linkage buffer using the documented `(n_samples,)` → off-by-one when they later pass it here."
contract: use `(n_samples - 1,)` (or an explicit `(n_edges,)`) in the docstrings for the SLT and condensed-tree parameters.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F14 — `_brute_mst`: `mutual_reachability` parameter docstring header uses a misspelled name
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82 signature has `mutual_reachability`; sklearn/cluster/_hdbscan/hdbscan.py:91-93 docstring documents it as `mututal_reachability_graph:` (typo *mututal*, and different name). Also documents `min_samples : int, default=None` for a parameter that has no signature default (sklearn/cluster/_hdbscan/hdbscan.py:82).
scenario: "Reader searches the source for `mutual_reachability` and matches only the code / not the docstring → trusts the misspelled `mututal` when following a doc-generation warning and propagates the typo into downstream references."
contract: rename the docstring parameter to `mutual_reachability : {ndarray, sparse matrix}` and remove the phantom `default=None` for `min_samples`.
instances: single-instance

### F15 — `enumerate(non_finite)` over a `set` produces non-deterministic outlier ordering
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:388-391 iterates a Python set (constructed at sklearn/cluster/_hdbscan/hdbscan.py:838 via `set(infinite_index + missing_index)`); the loop then assigns `last_cluster_id`/`last_cluster_size` in that iteration order.
scenario: "Two processes with different `PYTHONHASHSEED` fit HDBSCAN on identical data with non-finite rows → resulting `_single_linkage_tree_` outlier-appended rows differ in id assignment → downstream reproducibility checks fail even though algorithmic output is equivalent."
contract: iterate `sorted(non_finite)` (or accept a sequence in the API) so the appended outlier tree is deterministic.
instances: single-instance
