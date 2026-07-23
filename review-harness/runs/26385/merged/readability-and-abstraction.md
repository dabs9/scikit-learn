### F1 — HDBSCAN class docstring lies about `n_jobs` default
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` for the public `HDBSCAN` class; the actual signature at sklearn/cluster/_hdbscan/hdbscan.py:658 is `n_jobs=4`.
scenario: "user reads the docstring or hovers help(HDBSCAN) and assumes default=None (i.e. serial unless in a joblib backend) → user gets a 4-thread `pairwise_distances` on every fit, silently pinning cores and yielding non-`None` behavior that diverges from every other sklearn estimator convention"
contract: Change the signature default to `n_jobs=None` to match the docstring and sklearn convention.
instances: single-instance

### F2 — `_weighted_cluster_center` counts and iterates clusters using a mis-specified outlier set [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:894-895 says `# Number of non-noise clusters / n_clusters = len(set(self.labels_) - {-1, -2})` — but the class docstring (sklearn/cluster/_hdbscan/hdbscan.py:562, 573) explicitly states that `n_clusters` "only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded". Additionally, when `metric != "precomputed"` and the input contains non-finite rows, sklearn/cluster/_hdbscan/hdbscan.py:854-855 calls `self._weighted_cluster_center(X)` with the finite-only `X` (shrunk at line 733) while `self.labels_` has already been re-inflated to raw-data length at line 844 — so `mask = self.labels_ == idx` in `_weighted_cluster_center` has length `n_raw`, but `X` has length `n_finite`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing `np.nan` rows → `n_clusters` inflates by 1 for every -3 label in `set(labels_)`, and `data = X[mask]` at line 909 raises `IndexError: boolean index did not match indexed array along dimension 0` (or silently returns wrong rows)"
contract: Subtract `{-1, -2, -3}` when counting `n_clusters`, and pass `self._raw_data` (not the finite-only `X`) to `_weighted_cluster_center` so the mask length matches the data length.
instances: single-instance

### F3 — Scale-invariance HDBSCAN example never scales `X`, contradicting the section's claim [out-of-theme]
severity: high
evidence: examples/cluster/plot_hdbscan.py:106-110 — the loop is `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, ...)`. `X` (and the fit input) is never multiplied by `scale`. Compare with the immediately preceding DBSCAN block at lines 87-89 which correctly does `dbs.fit(X * scale)` and `plot(X * scale, ...)`. The narrative above (lines 102-105) claims the example demonstrates that "HDBSCAN is scale-invariant" — but three identical calls on the same unscaled `X` demonstrate nothing about scale.
scenario: "User reads the gallery example to convince themselves HDBSCAN is scale invariant → sees three identical plots that of course match → misinterprets identical output as proof of scale invariance, when in fact the example is a no-op loop."
contract: Change the loop to `hdb.fit(X * scale)` and `plot(X * scale, ...)` (mirroring the DBSCAN block just above), so the three subplots actually show HDBSCAN fitted on 1×, 0.5×, and 3× rescalings.
instances: single-instance

### F4 — `_hdbscan_brute` docstring lies about `alpha` default (and the coded default is unusable)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:183-184 documents `alpha : float, default=1.0`; the signature at sklearn/cluster/_hdbscan/hdbscan.py:161 is `alpha=None`. Line 241 then does `distance_matrix /= alpha`, which would raise `TypeError` on the documented "default" call.
scenario: "any caller who invokes `_hdbscan_brute` relying on the documented `alpha=1.0` default → immediate TypeError from `ndarray /= None`; caller reading the docstring cannot detect this without reading the code"
contract: Change the signature default to `alpha=1.0`.
instances: single-instance

### F5 — `_hdbscan_prims` documents a `copy` parameter that does not exist
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 contains a full `copy : bool, default=False ...` block, but the function signature at lines 269-278 has no `copy` argument, and `_hdbscan_prims` is called at line 820 (via `mst_func(**kwargs)`) with `kwargs` that never contain `copy` (only the `_hdbscan_brute` branches inject it — lines 795, 808).
scenario: "A user reads the Prim's docstring, believes they can influence in-place behavior for KDTree/BallTree paths → passes `copy=True` and gets `TypeError: unexpected keyword argument`, or trusts that some in-place safeguard applies to the Prim's path when none does."
contract: Delete the `copy` parameter block from `_hdbscan_prims`'s docstring.
instances: single-instance

### F6 — `remap_single_linkage_tree` docstring type is a lie (and set iteration is non-deterministic)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 declares `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`, but the sole caller at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` — a `set` of integer indices — and the body iterates with `enumerate(non_finite)` (line 388) so the value written into `outlier_tree[i][0]` is an integer index, not a boolean.
scenario: "Someone maintaining or extending this helper reads the docstring, believes `non_finite` is a boolean mask and either passes one directly (writing `True`/`False` into `left_node`) or refactors the loop under that false assumption → silent index corruption in the reconstructed tree; separately, because a `set` is iterated, the order in which outliers appear in `outlier_tree` is non-deterministic across runs."
contract: Update the docstring to state `non_finite : sequence of int  Indices of non-finite rows in the raw data`, and change the caller (line 838) to pass a deterministically ordered container (e.g. `sorted(set(infinite_index + missing_index))`).
instances: single-instance

### F7 — Test name/docstring claim vs. test behavior: `test_hdbscan_precomputed_non_brute` exercises invalid algorithm strings, not the documented case [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — the docstring says "HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm", but the test parametrizes `algorithm=f"prims_{tree}tree"` — i.e. `"prims_kdtree"` and `"prims_balltree"`. Neither string is in the accepted set `{"auto", "brute", "kdtree", "balltree"}` (lines 630-637 of `hdbscan.py`). So the raised `ValueError` comes from parameter validation on the algorithm name itself, *not* from the precomputed+tree combination the test purports to cover. The intended combination (`metric="precomputed"`, `algorithm="kdtree"` or `"balltree"`) is never exercised.
scenario: "Someone regresses the actual precomputed-with-tree guard → this test still passes (because the algorithm-string check keeps raising) → the regression ships silently."
contract: Change the parametrization to `algorithm` in `("kdtree", "balltree")` so the test actually exercises the documented "precomputed data + tree-based algorithm" incompatibility path.
instances: single-instance

### F8 — `_hdbscan_brute` / `_hdbscan_prims` docstrings claim `min_samples` default is `None` while signatures use `5`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160 (`min_samples=5`) but line 179 documents `min_samples : int, default=None`; sklearn/cluster/_hdbscan/hdbscan.py:272 (`min_samples=5`) but line 290 documents `min_samples : int, default=None`.
scenario: "reader assumes calling `_hdbscan_brute(X)` yields the documented `None` semantics ('defaults to `min_cluster_size`') → they get a hard-coded `5` instead, silently diverging from the estate-level `HDBSCAN` behavior"
contract: Change both docstrings to `min_samples : int, default=5`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F9 — `remap_single_linkage_tree` docstring claims outliers "considered noise points"; code assigns -2/-3
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:355-357 — docstring "These points will all be merged into the final node at np.inf distance and considered noise points." But the callers at lines 842-843 assign `_OUTLIER_ENCODING["infinite"]["label"]` (-2) and `_OUTLIER_ENCODING["missing"]["label"]` (-3), not the noise label -1.
scenario: "Reader trusts the docstring while designing consumer code → filters labels==-1 to exclude outliers → misses -2 and -3 outliers entirely."
contract: Replace "considered noise points" with "assigned the corresponding `_OUTLIER_ENCODING` labels (-2 for infinite, -3 for missing) by the caller".
instances: single-instance

### F10 — `_hdbscan_prims` docstring omits actual `**metric_params` kwargs contract and mis-labels it as `dict`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:277 signature `**metric_params`; docstring line 320-321 says "metric_params : dict, default=None". Same shape in `_hdbscan_brute` at lines 165 & 214-215.
scenario: "Caller reads `metric_params : dict` → attempts `_hdbscan_prims(..., metric_params={'V': ...})` → TypeError because the function only accepts keyword-splatted entries, not a `metric_params=` kwarg."
contract: Document as "**metric_params : dict — keyword arguments passed to the distance metric" (matching the signature).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:214-215, sklearn/cluster/_hdbscan/hdbscan.py:320-321]

### F11 — `_brute_mst` docstring parameter name and description do not match the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82-104 — signature parameter is `mutual_reachability`; docstring block calls it `mututal_reachability_graph:` (misspelled and a different name). The returned-value description also has typos "mutual-reahability" and "collecteion".
scenario: "Sphinx-style tooling and readers parsing the numpydoc block cannot resolve `mututal_reachability_graph` to the actual argument → cross-references break and the typos surface in rendered docs, misleading users about the parameter contract."
contract: Rename the docstring parameter to `mutual_reachability` and fix the two typos ("mutual-reachability", "collection").
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102-103, sklearn/cluster/_hdbscan/hdbscan.py:143-144, sklearn/cluster/_hdbscan/_linkage.pyx:75-76, sklearn/cluster/_hdbscan/_linkage.pyx:137-138, sklearn/cluster/_hdbscan/_linkage.pyx:229]

### F12 — `mutual_reachibility_distance` local variable is a typo of the concept it names
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:127, 144, 149, 182, 206, 209, 210 — the local is spelled `mutual_reachibility_distance` (missing the second `a`) throughout both dense and sparse loops, while the surrounding docstrings, module title, and public function name all use the correct `mutual_reachability`.
scenario: "reader greps for `mutual_reachability_distance` (the term used everywhere else in the module) to locate the actual computation → the search returns no hits and the reader concludes the computation lives elsewhere, or later assumes the misspelled local names a subtly different quantity"
contract: Rename to `mutual_reachability_distance` in every occurrence.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F13 — `bfs_from_hierarchy` comment references a stale 2D indexing shape
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:97-98 — comment "By construction, node i is formed by the union of nodes hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]". But `hierarchy` is a 1-D `HIERARCHY_t[::1]` typed memoryview of structs; the actual code below indexes `hierarchy[node].left_node` and `hierarchy[node].right_node` (fields, not a second axis).
scenario: "Reader tracking down index arithmetic trusts the comment → assumes hierarchy is a 2D `(n-1, 4)` array (SciPy-style) → refactors nearby code expecting axis-1 indexing → build breaks or silently reads wrong fields."
contract: Rewrite the comment as "By construction, node `i` (for `i >= n_samples`) is formed by the union of `hierarchy[i - n_samples].left_node` and `hierarchy[i - n_samples].right_node`."
instances: single-instance

### F14 — `_compute_stability`: `births` is allocated twice back-to-back
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 contains
```
largest_child = max(largest_child, smallest_cluster)
births = np.full(largest_child + 1, np.nan, dtype=np.float64)

births = np.full(largest_child + 1, np.nan, dtype=np.float64)
```
The first assignment is immediately shadowed by the second; both allocate the identical NaN-filled array.
scenario: "Reader spots the duplicated line and wastes time hunting for a subtle reason (initial values that differ, side effects on a memoryview, etc.) that does not exist → a maintainer patches one of the two lines and leaves the other, producing genuinely confusing dead code."
contract: Delete the redundant duplicate at line 252 (retain only the assignment at line 254).
instances: single-instance

### F15 — Convoluted iteration idioms in `remap_single_linkage_tree`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370 (`for i, _ in enumerate(tree):` — the second binding is unused, the natural form is `for i in range(len(tree))`); sklearn/cluster/_hdbscan/hdbscan.py:385-387 uses `tree[tree.shape[0] - 1]["left_node"]` etc. three times where `tree[-1]` is the canonical form.
scenario: "reader parses each expression thinking there's a reason not to use `range` or `tree[-1]` → wasted attention; makes the loop and last-row extraction look weightier than they are"
contract: Replace `enumerate(tree)` with `range(len(tree))` and `tree[tree.shape[0] - 1]` with `tree[-1]`.
instances: single-instance

### F16 — Docstring/code mismatch: outlier labels comment claims -1 and -2, actual mapping is -2 and -3
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — inline comment "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." But `_OUTLIER_ENCODING` at lines 65-79 defines `infinite → -2` and `missing → -3`, and the subsequent code at lines 842-843 assigns exactly those (-2, -3) values.
scenario: "A reader trusts the comment while debugging label semantics → assumes -1/-2 for inf/nan → misinterprets which outlier class received which encoding, and any downstream comparison logic will be off by one class."
contract: Update the comment to state "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3, per `_OUTLIER_ENCODING`."
instances: single-instance

### F17 — Speculative-abstraction dead attribute `TreeUnionFind.is_component`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:330,337,355 — `is_component` is declared, initialized, and mutated inside `find`, but grep across the whole tree finds no reader. It is a purely write-only attribute whose upkeep still costs a store per path-compression step in a nogil-worthy hot path.
scenario: "Reader trying to understand `TreeUnionFind.find` sees `self.is_component[x] = False` and hunts for its consumer → concludes the writer is load-bearing for correctness → refuses to remove it during future cleanups, or writes new code that depends on this ghost invariant."
contract: Delete the `is_component` declaration, its initialization in `__init__`, and its assignment inside `find`.
instances: single-instance
