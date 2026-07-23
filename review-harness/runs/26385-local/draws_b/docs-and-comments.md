### F1 — HDBSCAN docstring `n_jobs` default contradicts actual default of 4
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — the class parameter section documents `n_jobs : int, default=None` and states "`None` means 1 unless in a joblib.parallel_backend context"; sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature is `n_jobs=4`
scenario: "User reads docs believing HDBSCAN defaults to a single-threaded joblib-context-aware behavior → in reality it spawns 4 workers, producing different resource usage and results the docs do not describe"
contract: Update the docstring to `default=4` and remove the "None means 1" description that no longer applies to the actual default.
instances: single-instance

### F2 — HDBSCAN docstring claims `centroids_`/`medoids_` are always-present attributes
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:556-573 — `centroids_` and `medoids_` are listed under `Attributes` with no note that they are only computed when `store_centers` is set; sklearn/cluster/_hdbscan/hdbscan.py:854-855 and 897-903 — these attributes are only created if `self.store_centers` is truthy (and only the matching kind is written)
scenario: "User consults the Attributes section, expecting `centroids_` after `fit` → attribute is missing when `store_centers=None` (default), causing AttributeError"
contract: Document these two attributes as conditional on `store_centers`, mirroring the sibling documentation style for optional attributes elsewhere in sklearn.
instances: single-instance

### F3 — `_weighted_cluster_center` excludes only `{-1, -2}` while docstring/comment intent is "non-outlier / non-noise"
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:894-895 — `# Number of non-noise clusters` immediately followed by `n_clusters = len(set(self.labels_) - {-1, -2})`; the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:561-562 explicitly says `n_clusters` "only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded"; the missing-data label -3 is defined in sklearn/cluster/_hdbscan/hdbscan.py:73-78
scenario: "User calls `fit` with data containing missing rows and `store_centers` set → `-3` label leaks into the set, `n_clusters` is off by one, and the pre-allocated `centroids_`/`medoids_` arrays are one row too large (the last row is never written and stays as uninitialized `np.empty` values) [out-of-theme]"
contract: The set must be `{-1, -2, -3}` (or better, derived from `_OUTLIER_ENCODING`), matching the documented invariant.
instances: single-instance

### F4 — `_get_clusters` docstring documents a `stabilities` return value that is not returned
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:681-690 — Returns section documents three items: `labels`, `probabilities`, and `stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster.`; sklearn/cluster/_hdbscan/_tree.pyx:797 — `return (labels, probs)` returns only two items
scenario: "Maintainer trusts docstring, writes downstream code unpacking three values → ValueError at runtime, or silently binds the probability array to `stabilities`"
contract: Remove the `stabilities` entry from the Returns section (or add and return the value); pin the fix as removal to match current behavior.
instances: single-instance

### F5 — `_brute_mst` docstring parameter name and default don't match the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82-97 — signature is `def _brute_mst(mutual_reachability, min_samples):` (no default), while docstring says `mututal_reachability_graph: {ndarray, sparse matrix}` (wrong name + typo) and `min_samples : int, default=None` (nonexistent default)
scenario: "Reader relies on the Parameters block → mis-names the argument in call sites and assumes an optional `min_samples`, both wrong"
contract: Rename the docstring parameter to `mutual_reachability` and drop the fictitious `default=None`.
instances: single-instance

### F6 — `_hdbscan_prims` docstring documents a `copy` parameter the function does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature has no `copy` parameter; sklearn/cluster/_hdbscan/hdbscan.py:313-318 — docstring nevertheless includes a full `copy : bool, default=False` block copied verbatim from `_hdbscan_brute`
scenario: "Reader/caller passes `copy=True` believing it is honored → silently ignored (or TypeError if forwarded as positional)"
contract: Remove the `copy` block from `_hdbscan_prims` docstring; it applies only to `_hdbscan_brute`.
instances: single-instance

### F7 — `_hdbscan_brute` / `_hdbscan_prims` docstrings state `min_samples : int, default=None` while signatures default to 5
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160 and 272 — both signatures declare `min_samples=5`; sklearn/cluster/_hdbscan/hdbscan.py:179-181 and 290-292 — both docstrings say `min_samples : int, default=None`
scenario: "Reader relies on `default=None` semantics (falling back elsewhere) → in reality the fallback is a hard-coded `5`"
contract: Update each docstring to `default=5` to match the signature.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F8 — `_sparse_mutual_reachability_graph` docstring documents a nonexistent `distance_matrix` parameter and omits real ones
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:152-179 — signature is `(data, indices, indptr, n_samples, further_neighbor_idx, max_distance)`; docstring only documents `distance_matrix`, `further_neighbor_idx`, `max_distance`, none of which is the actual `data`/`indices`/`indptr`/`n_samples` triple; the phrase "the sparse format should be `CSR`" describes an argument that is not passed
scenario: "Reader/maintainer must reverse-engineer meaning of positional args from callers rather than the docstring → mis-uses the function or introduces bugs when refactoring call sites"
contract: Rewrite the Parameters block to reflect the real CSR-split arguments (`data`, `indices`, `indptr`, `n_samples`) and drop `distance_matrix`.
instances: single-instance

### F9 — `mutual_reachability_graph` docstring references `max_dist`, which is not a parameter
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:68-72 — the `max_distance` parameter block says "truncated to `max_dist`"; same wording appears in `_sparse_mutual_reachability_graph` docstring at sklearn/cluster/_hdbscan/_reachability.pyx:174-178
scenario: "Reader searches for `max_dist` in the API → nothing exists; wording introduces a phantom name for `max_distance`"
contract: Replace `max_dist` with `max_distance` in both docstrings.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:71, sklearn/cluster/_hdbscan/_reachability.pyx:177]

### F10 — `bfs_from_hierarchy` invariant comment references stale 2D indexing that no longer exists
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:97-98 — comment reads "By construction, node i is formed by the union of nodes hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]"; the hierarchy is now a 1D structured array of `HIERARCHY_t` accessed via `.left_node`/`.right_node` (see sklearn/cluster/_hdbscan/_tree.pxd:34-38 and the usage at sklearn/cluster/_hdbscan/_tree.pyx:109-110)
scenario: "Reader tries to reproduce the described indexing → hits AttributeError/IndexError; the load-bearing invariant is described in a schema that no longer exists"
contract: Rewrite the comment to describe the structured-array fields actually used (`hierarchy[i - n_samples].left_node` / `.right_node`).
instances: single-instance

### F11 — Recurring "tree tree", "reahability", "collecteion", "smaler", "simbling" docstring typos on public/exported symbols
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:102-103, 148-149, 219-220 (`mutual-reahability`, `collecteion`, `single-linkage tree tree`); sklearn/cluster/_hdbscan/_linkage.pyx:75-76, 137-138, 233-234 (same three); sklearn/cluster/_hdbscan/_tree.pyx:133-134 (`smaler`), 502-504 (`simbling`); sklearn/cluster/_hdbscan/_reachability.pyx:126-127 (`mutual_reachibility_distance` in code and comments)
scenario: "These docstrings render into the public API documentation → typos ship user-facing"
contract: Correct spellings across the listed occurrences ("mutual-reachability", "collection", "single-linkage tree (dendrogram)", "smaller", "sibling", "mutual_reachability_distance").
instances: [sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:148, sklearn/cluster/_hdbscan/hdbscan.py:219, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:233, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209]

### F12 — `_condense_tree` / `_do_labelling` / `_get_clusters` docstrings state hierarchy/condensed_tree shape is `(n_samples,)` when it is `(n_samples - 1,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:128-130 states `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`; but the linkage output is `(n_samples - 1,)` as documented in sklearn/cluster/_hdbscan/_linkage.pyx:232-234 and confirmed by allocation at sklearn/cluster/_hdbscan/_linkage.pyx:252 (`np.zeros(n_samples - 1, dtype=HIERARCHY_dtype)`); similar wrong shape appears in `_do_labelling` docstring at sklearn/cluster/_hdbscan/_tree.pyx:447-450 and `_get_clusters` at sklearn/cluster/_hdbscan/_tree.pyx:656-659
scenario: "Reader trusts the shape → miscomputes buffer sizes when building test fixtures or bindings"
contract: Change these shapes to `(n_samples - 1,)` matching the actual producer.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:448, sklearn/cluster/_hdbscan/_tree.pyx:657]

### F13 — HDBSCAN `algorithm` docstring uses inconsistent value casing ("KDTree"/"BallTree" vs. accepted "kdtree"/"balltree")
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:466-472 — parameter enumerates `{"auto", "brute", "kdtree", "balltree"}` (matching the `_parameter_constraints` StrOptions at 629-638) but the prose body says "Both `"KDTree"` and `"BallTree"` algorithms use the `NearestNeighbors` estimator."
scenario: "User copies `algorithm="KDTree"` from the prose → parameter validation raises `InvalidParameterError`"
contract: Refer to the accepted lowercase strings `"kdtree"` / `"balltree"` in the prose (keep class links separate).
instances: single-instance

### F14 — Duplicated `births = np.full(...)` allocation is dead-code narration noise
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and 254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears twice with no intervening use; the first assignment is immediately overwritten
scenario: "Reader wonders which assignment is authoritative and whether there is a hidden side effect → wasted comprehension time, plus a redundant allocation"
contract: Delete the first assignment (line 252) so a single, documented allocation remains.
instances: single-instance

### F15 — `_weighted_cluster_center` inline comment is ungrammatical/misleading
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 — `# Need to handle iteratively seen each cluster may have a different` reads as a truncated/garbled sentence and does not describe the iteration invariant it precedes
scenario: "Reader trying to understand why iteration over clusters is necessary → gets no useful information and must reverse-engineer the invariant from the code"
contract: Replace with a comment explaining the actual invariant (clusters have varying sizes so results cannot be stacked into a homogeneous 3-D array).
instances: single-instance

### F16 — Note about `dbscan_clustering` returning `-3` labels only applies when `metric != "precomputed"` but docstring omits that
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:952-958 — Returns section unconditionally lists `-3` for "Samples with missing data"; but per sklearn/cluster/_hdbscan/hdbscan.py:734-751, missing-data (`np.nan`) rejection differs between `metric="precomputed"` (raises ValueError before any label is assigned) and the non-precomputed path (uses `-3`), so `-3` never occurs under precomputed
scenario: "User with `metric='precomputed'` expects the `-3` code and writes downstream logic checking for it → the code path never emits `-3`"
contract: Note in the Returns block that the `-3` code is only produced when `metric != "precomputed"`.
instances: single-instance
