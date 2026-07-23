I'll note the malware-consideration reminder: these files are review draws about scikit-learn's HDBSCAN documentation — they are analysis notes, not executable code, and contain no malicious content. I will perform the union-merge only, not modify or augment code.

Now producing the merged UNION output. Deduplication mapping across A/B/C:

- n_jobs default: A1 = B1 = C1 → keep most specific (A1, high severity, most detail)
- _hdbscan_prims copy/algo/leaf_size: A2 = B6 = C3 → C3 most specific
- _brute_mst param name/default: A3 = B5 = C4 → A3 most specific
- _hdbscan_brute min_samples default: A4 = B7 (covers both prims+brute) — B7 more specific covering both sites. C2 combines with alpha.
- _hdbscan_brute alpha default: A5 = C2 (partial) — A5 focused on alpha; C2 covers both min_samples+alpha. Keep A5 (alpha-specific) and B7 (min_samples-specific covering both sites). Note C2 combines both — will treat as same defects already covered by A5 + B7.
- _sparse_mutual_reachability_graph distance_matrix: A13 = B8 = C8 → B8/A13 same specificity, pick B8 (medium severity, mentions CSR format detail)
- max_dist/max_distance: A12 = B9 → A12 same specificity, both fine
- bfs_from_hierarchy comment: A14 = B10 → B10 more specific (references pxd)
- Typos: A11 = B11 = C11 → C11 most comprehensive (includes test_reachibility.py filename)
- User guide clustering.rst typos (staged/minimum_cluster_size): A9 + A10 vs C12 (combines both). C12 combines them; A9 and A10 are separate defects at different lines. Keep A9, A10 as separate blocks (distinct sites/defects) — C12 covers same two defects merged. Prefer keeping A9 and A10 separate since they're distinct root causes at distinct sites.
- _get_clusters stabilities: A20 = B4 = C7 → B4 most specific (medium severity)
- _condense_tree/hierarchy shape: A19 = B12 = C9 → C9 most specific (widest instance list)
- _weighted_cluster_center n_clusters exclusion: B3 = C6 → B3 high severity, most specific
- Duplicated births: A18 = B14 → B14 more specific (identifies line 252 as first)
- _get_clusters ordering comment: A21 = C16 → C16 much more specific
- dbscan_clustering docstring: A7 (missing precondition + -2/-3) vs C15 (narration/run-on). These are related but distinct defects at same site — A7 is about precondition + label conditions; C15 is about narration/punctuation. Keep both.
- dbscan_clustering -3 only non-precomputed: B16 — distinct from A7 (B16 focuses on -3 conditionality). Keep separate.
- HDBSCAN.labels_ precomputed caveat: A17 — distinct from B16 (labels_ vs dbscan_clustering docstring).

Now producing output ordered by severity (high, medium, low) with stable ordering within.

### F1 — HDBSCAN `n_jobs` docstring contradicts the actual default (`None` vs `4`)
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring reads `n_jobs : int, default=None` and states "`None` means 1 unless in a joblib.parallel_backend context", but the constructor signature at line 658 sets `n_jobs=4`. Users following the documented default will get a completely different concurrency setting than the code enforces.
scenario: "User reads docstring expecting single-thread default → HDBSCAN silently spawns 4 workers, oversubscribing CPUs in shared or nested-parallel contexts"
contract: The `n_jobs` docstring in the `HDBSCAN` class must state `default=4` and remove the "None means 1" boilerplate (or the constructor default must be changed to `None`); one of the two must move so signature and prose agree.
instances: single-instance

### F2 — `_weighted_cluster_center` excludes only `{-1, -2}` while docstring/comment intent is "non-outlier / non-noise"
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:894-895 — `# Number of non-noise clusters` immediately followed by `n_clusters = len(set(self.labels_) - {-1, -2})`; the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:561-562 explicitly says `n_clusters` "only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded"; the missing-data label -3 is defined in sklearn/cluster/_hdbscan/hdbscan.py:73-78
scenario: "User calls `fit` with data containing missing rows and `store_centers` set → `-3` label leaks into the set, `n_clusters` is off by one, and the pre-allocated `centroids_`/`medoids_` arrays are one row too large (the last row is never written and stays as uninitialized `np.empty` values) [out-of-theme]"
contract: The set must be `{-1, -2, -3}` (or better, derived from `_OUTLIER_ENCODING`), matching the documented invariant.
instances: single-instance

### F3 — `_hdbscan_prims` docstring documents parameters that do not exist and omits ones that do
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-327 — signature is `(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)` but docstring at :313 documents a non-existent `copy` parameter (with "Currently, it only applies when `metric="precomputed"`" — which this function never handles), and never documents `algo` or `leaf_size`. Also `min_samples : int, default=None` at :290 mismatches the actual `min_samples=5`.
scenario: "Developer maintaining the file reads the docstring and adds precomputed-handling logic gated by `copy` → introduces dead code path because `copy` is never accepted; or omits documenting `algo`/`leaf_size` in downstream references, breaking user understanding"
contract: The docstring's Parameters block must exactly match the actual signature — drop the `copy` block, drop the "metric='precomputed'" phrasing, add entries for `algo` and `leaf_size`, and correct the `min_samples` default.
instances: single-instance

### F4 — `_brute_mst` docstring parameter name and `min_samples` default are wrong
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82-97 — the parameter section names the array `mututal_reachability_graph:` (typo, and does not match the actual argument `mutual_reachability`), and declares `min_samples : int, default=None` while the signature `_brute_mst(mutual_reachability, min_samples)` has no default at all.
scenario: "Reader relies on documented parameter name/default → uses wrong keyword or assumes None is accepted, hitting TypeError at call time"
contract: Rename the parameter in the docstring to `mutual_reachability` and remove the `default=None` annotation to match the real signature.
instances: single-instance

### F5 — `_hdbscan_brute` documents `alpha : float, default=1.0` but signature default is `None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 sets `alpha=None`; docstring at line 183 asserts `alpha : float, default=1.0`. In the body at line 241 the code unconditionally runs `distance_matrix /= alpha`, so calling with the documented default `1.0` works, but calling with the actual signature default (`None`) will raise a TypeError.
scenario: "User invokes _hdbscan_brute without alpha per docstring's stated default → TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'"
contract: Change the signature to `alpha=1.0` so it matches the documented default and the unconditional `distance_matrix /= alpha` in the body.
instances: single-instance

### F6 — Docstring anchor mismatch: `User Guide <HDBSCAN>` reference does not exist
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104 — `see :ref:\`User Guide <HDBSCAN>\``. The actual anchor label in doc/modules/clustering.rst is `.. _hdbscan:` (lowercase), so this cross-reference fails to resolve in the rendered docs.
scenario: "Sphinx build of gallery example → broken/undefined :ref: link labeled 'User Guide' either warns or renders as raw text for readers"
contract: Change the reference target to lowercase `hdbscan` to match the RST anchor.
instances: single-instance

### F7 — Test-narration docstring lies about what is being tested (`test_hdbscan_precomputed_non_brute`)
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — docstring claims the test checks that HDBSCAN "correctly raises an error when passing precomputed data while requesting a tree-based algorithm", but the algorithm string passed is `prims_kdtree` / `prims_balltree`, and `_parameter_constraints["algorithm"]` in `hdbscan.py:629-638` only accepts `{"auto", "brute", "kdtree", "balltree"}`. The `ValueError` therefore comes from parameter-name validation, not from the precomputed+tree combination the narration promises. The intended combination is never actually exercised.
scenario: "Behavior change removes the precomputed-vs-tree check in fit → test still passes on unrelated grounds, giving false coverage confidence"
contract: The test must call `HDBSCAN(metric='precomputed', algorithm=f'{tree}tree')` with a valid algorithm name, or the docstring must be rewritten to state that invalid algorithm names raise.
instances: single-instance

### F8 — `_sparse_mutual_reachability_graph` docstring documents a nonexistent `distance_matrix` parameter and omits real ones
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:152-179 — signature is `(data, indices, indptr, n_samples, further_neighbor_idx, max_distance)`; docstring only documents `distance_matrix`, `further_neighbor_idx`, `max_distance`, none of which is the actual `data`/`indices`/`indptr`/`n_samples` triple; the phrase "the sparse format should be `CSR`" describes an argument that is not passed
scenario: "Reader/maintainer must reverse-engineer meaning of positional args from callers rather than the docstring → mis-uses the function or introduces bugs when refactoring call sites"
contract: Rewrite the Parameters block to reflect the real CSR-split arguments (`data`, `indices`, `indptr`, `n_samples`) and drop `distance_matrix`.
instances: single-instance

### F9 — `bfs_from_hierarchy` invariant comment references stale 2D indexing that no longer exists
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:97-98 — comment reads "By construction, node i is formed by the union of nodes hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]"; the hierarchy is now a 1D structured array of `HIERARCHY_t` accessed via `.left_node`/`.right_node` (see sklearn/cluster/_hdbscan/_tree.pxd:34-38 and the usage at sklearn/cluster/_hdbscan/_tree.pyx:109-110)
scenario: "Reader tries to reproduce the described indexing → hits AttributeError/IndexError; the load-bearing invariant is described in a schema that no longer exists"
contract: Rewrite the comment to describe the structured-array fields actually used (`hierarchy[i - n_samples].left_node` / `.right_node`).
instances: single-instance

### F10 — `_get_clusters` docstring documents a `stabilities` return value that is not returned
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:681-690 — Returns section documents three items: `labels`, `probabilities`, and `stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster.`; sklearn/cluster/_hdbscan/_tree.pyx:797 — `return (labels, probs)` returns only two items
scenario: "Maintainer trusts docstring, writes downstream code unpacking three values → ValueError at runtime, or silently binds the probability array to `stabilities`"
contract: Remove the `stabilities` entry from the Returns section (or add and return the value); pin the fix as removal to match current behavior.
instances: single-instance

### F11 — `HDBSCAN.fit` remap comment states wrong outlier labels
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." but the code (using `_OUTLIER_ENCODING`) maps `np.inf → -2` (`"infinite"` label) and `np.nan → -3` (`"missing"` label), matching the class-level docstring at sklearn/cluster/_hdbscan/hdbscan.py:534-537.
scenario: "Developer reviewing the remap logic trusts the comment and later refactors labelling to keep the '-1/-2' convention → silently breaks the documented `-2`/`-3` contract seen by end users"
contract: Update the comment to state that `np.inf` is mapped to `-2` and `np.nan` to `-3`, matching `_OUTLIER_ENCODING` and the class docstring.
instances: single-instance

### F12 — `_hdbscan_brute` / `_hdbscan_prims` docstrings state `min_samples : int, default=None` while signatures default to 5
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160 and 272 — both signatures declare `min_samples=5`; sklearn/cluster/_hdbscan/hdbscan.py:179-181 and 290-292 — both docstrings say `min_samples : int, default=None`
scenario: "Reader relies on `default=None` semantics (falling back elsewhere) → in reality the fallback is a hard-coded `5`"
contract: Update each docstring to `default=5` to match the signature.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F13 — `dbscan_clustering` docstring omits the removal of border points invariant and mislabels outputs
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-958 — Returns section lists label rules "-1 noise", "-2 infinite", "-3 missing", but the implementation at lines 960-969 only reassigns `_OUTLIER_ENCODING["infinite"]` and `_OUTLIER_ENCODING["missing"]`; there is no `-1` produced by the underlying `labelling_at_cut` unless a cluster is smaller than `min_cluster_size`. More importantly, the "See Also" invariant that `dbscan_clustering` returns labels only for the data seen during `fit` (relies on `self.labels_`) is left implicit — calling it before `fit` raises AttributeError but nothing in the docstring warns of the `fit`-required precondition.
scenario: "User calls `HDBSCAN().dbscan_clustering(0.3)` before fit per docstring reading → AttributeError on `self._single_linkage_tree_`"
contract: Add an explicit precondition line stating that `fit` must be called first, and clarify that the -2/-3 labels only appear when non-finite samples were present during `fit`.
instances: single-instance

### F14 — User-guide prose contains stale "at this staged" grammar and describes wrong noise criterion
severity: low
evidence: doc/modules/clustering.rst:1023-1027 — "Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." Two problems verified: (a) typo "at this staged" (should be "at this stage"); (b) the invariant is inverted — points with core distance *greater than* ε are the ones DBSCAN* treats as noise, not those less than ε. Also stray trailing colons after `\varepsilon`: appear twice on those lines.
scenario: "Reader learning HDBSCAN from user guide → internalizes an incorrect definition of DBSCAN* noise assignment"
contract: Fix the typo to "at this stage" and correct the direction of the inequality: "Any points whose core distance is greater than ε are marked as noise."
instances: single-instance

### F15 — User-guide prose uses undefined variable name `minimum_cluster_size` in place of `min_cluster_size`
severity: low
evidence: doc/modules/clustering.rst:1060-1064 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples`". No parameter named `minimum_cluster_size` exists on `HDBSCAN`; the actual parameter is `min_cluster_size`.
scenario: "User copies the guide's example `HDBSCAN(minimum_cluster_size=...)` → TypeError / warning about unknown parameter"
contract: Replace both `minimum_cluster_size` occurrences with `min_cluster_size` to match the real API.
instances: single-instance

### F16 — Pervasive typographical errors in docstrings and comments
severity: low
evidence: recurring misspellings across new HDBSCAN code — "reahability" (sklearn/cluster/_hdbscan/_linkage.pyx:75, :137, :229; sklearn/cluster/_hdbscan/hdbscan.py:102, :143), "collecteion" (sklearn/cluster/_hdbscan/_linkage.pyx:76, :138, :229; sklearn/cluster/_hdbscan/hdbscan.py:103, :144), "mututal_reachability_graph" (sklearn/cluster/_hdbscan/hdbscan.py:91; sklearn/cluster/_hdbscan/_reachability.pyx:76), "mutual_reachibility_distance" (sklearn/cluster/_hdbscan/_reachability.pyx:127, :144, :149, :182, :206, :209), "smaler" (sklearn/cluster/_hdbscan/_tree.pyx:133), "simbling" (sklearn/cluster/_hdbscan/_tree.pyx:503; sklearn/cluster/tests/test_hdbscan.py:530), "single-linkage tree tree" duplicate word (sklearn/cluster/_hdbscan/_linkage.pyx:234; sklearn/cluster/_hdbscan/hdbscan.py:149, :220, :326, :361), test filename "test_reachibility.py".
scenario: "Users grepping code/docs for canonical spellings ('mutual_reachability', 'sibling', etc.) miss all these locations → obscures searchability of the algorithm's terminology across the codebase and appears unprofessional in generated Sphinx docs"
contract: Fix each misspelling to its canonical form ("reachability", "collection", "mutual", "smaller", "sibling") and remove the duplicated "tree tree"; rename `test_reachibility.py` to `test_reachability.py`.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:233, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:148, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:219, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/hdbscan.py:906, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/_reachability.pyx:76, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/tests/test_hdbscan.py:530, sklearn/cluster/_hdbscan/tests/test_reachibility.py]

### F17 — `max_distance` docstring refers to a nonexistent identifier `max_dist`
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:68-72 and 174-178 — both `max_distance` parameter blocks state "it is instead truncated to `max_dist`", but there is no `max_dist` argument or attribute anywhere; the correct name is `max_distance`.
scenario: "Reader trying to enable this fallback searches for `max_dist` → finds nothing; passes `max_dist=` and gets a TypeError"
contract: Replace `max_dist` with `max_distance` in both docstrings.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:71, sklearn/cluster/_hdbscan/_reachability.pyx:177]

### F18 — `_condense_tree` and `_do_labelling` Returns claim shape `(n_samples,)` for structures whose length is not `n_samples`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 documents `hierarchy : ndarray of shape (n_samples,)` for what is actually `(n_samples-1,)` (see `make_single_linkage` at sklearn/cluster/_hdbscan/_linkage.pyx:252 which returns `np.zeros(n_samples - 1, ...)`); sklearn/cluster/_hdbscan/_tree.pyx:138 documents the `condensed_tree` output as shape `(n_samples,)` although it is a variable-length edgelist; sklearn/cluster/_hdbscan/_tree.pyx:447 repeats the wrong `(n_samples,)` shape for `condensed_tree`.
scenario: "Reader relying on the documented shape sizes downstream buffers as `n_samples` → truncates or overflows because the condensed tree is not that length"
contract: Correct the shape annotations: `hierarchy` should read `(n_samples-1,)`; `condensed_tree` should be described as a variable-length edgelist (e.g. `(n_edges,)`), not `(n_samples,)`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F19 — Stale user-guide sentence claims "The HDBSCAN implementation is multithreaded"
severity: low
evidence: doc/modules/clustering.rst hunk near line 1149 — "The HDBSCAN implementation is multithreaded, and has better algorithmic runtime complexity than OPTICS…" while sklearn/cluster/_hdbscan/_reachability.pyx:140-141 contains a TODO explicitly noting that the loop is **not** yet paralleled: "# TODO: Update w/ prange with thread count based on _openmp_effective_n_threads". The user-facing docs therefore promise multithreading that this implementation does not deliver.
scenario: "Users choosing HDBSCAN over OPTICS on the basis of multithreading claim → surprised by wall-time single-thread performance"
contract: Remove "multithreaded, and" (or reword) from the OPTICS-vs-HDBSCAN paragraph until `_dense_mutual_reachability_graph` is parallelized.
instances: single-instance

### F20 — `HDBSCAN.labels_` docstring's outlier bullets contradict actual behavior for `dbscan_clustering`/precomputed
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:530-538 — the Attributes list unconditionally asserts label `-2` for infinite samples and `-3` for missing. But in `fit` (lines 730-753) the -2/-3 remapping is skipped when `metric == "precomputed"` (only `_get_finite_row_indices` is not invoked, no remap block runs), so a precomputed dense matrix containing `np.inf` never yields a `-2` label — those rows silently participate in clustering. The docstring gives no such caveat.
scenario: "User with a precomputed distance matrix containing np.inf reads docstring → expects -2 for those points; gets an ordinary cluster id or ordinary -1 noise"
contract: The `labels_` docstring must add a caveat that `-2`/`-3` labels are only produced when `metric != 'precomputed'`; for precomputed distance matrices, non-finite entries are left untouched (inf) or raised on (nan).
instances: single-instance

### F21 — `_compute_stability` contains a duplicated `births = np.full(...)` line signalling a stale copy-paste [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and 254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears twice with no intervening use; the first assignment is immediately overwritten
scenario: "Reader wonders which assignment is authoritative and whether there is a hidden side effect → wasted comprehension time, plus a redundant allocation"
contract: Delete the first assignment (line 252) so a single, documented allocation remains.
instances: single-instance

### F22 — `_do_labelling` Returns docstring makes a claim that is not enforced by the code [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:462-464 — Returns says "a label of -1 denotes a noise assignment", but at lines 497-506 the `label` variable that is written into `result` may take the value of `cluster_label_map[cluster]` where `cluster` is the child index `n`; if `n` is not in `cluster_label_map`, this raises KeyError. That behavior isn't in the contract and there is no cross-check ensuring `cluster_label_map` covers every reachable value. This isn't a docs-only defect (the code lacks defensive handling for the contract the docstring promises), which is why I flag it out-of-theme.
scenario: "clusters passed by _get_clusters miss a value present in labels → KeyError instead of the documented -1"
contract: Enforce the noise fall-through the docstring promises by defaulting to NOISE for unknown `cluster` keys in `_do_labelling`.
instances: single-instance

### F23 — `_get_clusters` `# (exclude root)` narration comment sits after the slice that already excluded it
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:702-705 — comment "Assume clusters are ordered by numeric id equivalent to a topological sort of the tree; This is valid given the current implementation above, so don't change that ... or if you do, change this accordingly!" is vague ("the current implementation above") and does not name which upstream function (`_condense_tree`) it relies on, nor what property (`next_label` monotonic increment ensuring children > parent) makes the sort valid.
scenario: "Contributor refactors `_condense_tree`'s labelling to use a different id scheme (e.g. hashed ids) → silently breaks `_get_clusters` because the assumed topological ordering is no longer produced, and this comment does not tell them where to look"
contract: Replace the vague comment with an explicit invariant: reference `_condense_tree` and its use of `next_label`, e.g. "This relies on `_condense_tree` assigning cluster ids via monotonically-increasing `next_label`, which guarantees parent id < child id (i.e. sort-by-id is a reverse topological order). If `_condense_tree`'s id scheme changes, replace this `sorted(...)` with an explicit topological sort."
instances: [sklearn/cluster/_hdbscan/_tree.pyx:702, sklearn/cluster/_hdbscan/_tree.pyx:710]

### F24 — `_hierarchical_fast.pyx` deletion of `cdef` block leaves no comment tying the fields to the new `.pxd` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pyx:322-337 — the three `cdef intp_t next_label / intp_t[:] parent / intp_t[:] size` declarations were removed from the `UnionFind` class body and moved to the new `sklearn/cluster/_hierarchical_fast.pxd`. Neither file carries a comment noting that the declarations are now in the `.pxd` (nor that removing them again would break `_hdbscan/_linkage.pyx` which `cimport`s `UnionFind`). This is a lost-semantics documentation regression: the previous single-file view was self-describing; the current split has no cross-reference.
scenario: "Future contributor editing `_hierarchical_fast.pyx` reintroduces `cdef intp_t next_label` for clarity → Cython 'field defined twice' error, or removes the pxd thinking it is dead"
contract: Add a one-line comment in `_hierarchical_fast.pyx` above `cdef class UnionFind(object):` pointing at `_hierarchical_fast.pxd` where the fields and cdef methods are now declared.
instances: single-instance

### F25 — HDBSCAN docstring claims `centroids_`/`medoids_` are always-present attributes
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:556-573 — `centroids_` and `medoids_` are listed under `Attributes` with no note that they are only computed when `store_centers` is set; sklearn/cluster/_hdbscan/hdbscan.py:854-855 and 897-903 — these attributes are only created if `self.store_centers` is truthy (and only the matching kind is written)
scenario: "User consults the Attributes section, expecting `centroids_` after `fit` → attribute is missing when `store_centers=None` (default), causing AttributeError"
contract: Document these two attributes as conditional on `store_centers`, mirroring the sibling documentation style for optional attributes elsewhere in sklearn.
instances: single-instance

### F26 — HDBSCAN `algorithm` docstring uses inconsistent value casing ("KDTree"/"BallTree" vs. accepted "kdtree"/"balltree")
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:466-472 — parameter enumerates `{"auto", "brute", "kdtree", "balltree"}` (matching the `_parameter_constraints` StrOptions at 629-638) but the prose body says "Both `"KDTree"` and `"BallTree"` algorithms use the `NearestNeighbors` estimator."
scenario: "User copies `algorithm="KDTree"` from the prose → parameter validation raises `InvalidParameterError`"
contract: Refer to the accepted lowercase strings `"kdtree"` / `"balltree"` in the prose (keep class links separate).
instances: single-instance

### F27 — `_weighted_cluster_center` inline comment is ungrammatical/misleading
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 — `# Need to handle iteratively seen each cluster may have a different` reads as a truncated/garbled sentence and does not describe the iteration invariant it precedes
scenario: "Reader trying to understand why iteration over clusters is necessary → gets no useful information and must reverse-engineer the invariant from the code"
contract: Replace with a comment explaining the actual invariant (clusters have varying sizes so results cannot be stacked into a homogeneous 3-D array).
instances: single-instance

### F28 — Note about `dbscan_clustering` returning `-3` labels only applies when `metric != "precomputed"` but docstring omits that
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:952-958 — Returns section unconditionally lists `-3` for "Samples with missing data"; but per sklearn/cluster/_hdbscan/hdbscan.py:734-751, missing-data (`np.nan`) rejection differs between `metric="precomputed"` (raises ValueError before any label is assigned) and the non-precomputed path (uses `-3`), so `-3` never occurs under precomputed
scenario: "User with `metric='precomputed'` expects the `-3` code and writes downstream logic checking for it → the code path never emits `-3`"
contract: Note in the Returns block that the `-3` code is only produced when `metric != "precomputed"`.
instances: single-instance

### F29 — `remap_single_linkage_tree` public helper missing a `Returns` section
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-366 — docstring documents `tree`, `internal_to_raw`, `non_finite` but never documents the returned `tree` (which is a newly concatenated ndarray, per the function's `return tree` at :393).
scenario: "Consumer of `remap_single_linkage_tree` expects an in-place mutation per the docstring's silent Returns → drops the return value and works on a stale tree lacking the concatenated outlier rows"
contract: Add a `Returns` block documenting the returned `tree : ndarray of shape (n_samples - 1 + len(non_finite),), dtype=HIERARCHY_dtype` and explicitly note that the input is also mutated.
instances: single-instance

### F30 — `_hdbscan_brute`/`_hdbscan_prims` docstring misdescribes `metric_params` as a dict with default None
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:214-215 — documents `metric_params : dict, default=None` but the signature captures it via `**metric_params` (i.e. it is not a dict-typed keyword and has no default; individual keyword arguments are unpacked). The same misdescription exists in `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:320-321.
scenario: "Caller passes `metric_params={'V': cov}` per docstring → the helper receives a single kwarg named `metric_params` inside `**metric_params`, then `metric_params.get('max_distance', 0.0)` at sklearn/cluster/_hdbscan/hdbscan.py:243 silently misbehaves because the real metric params were never unpacked"
contract: Rewrite the docstring to reflect the `**metric_params` API ("Additional keyword arguments passed to the distance metric.") and drop the fictitious `default=None`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:214, sklearn/cluster/_hdbscan/hdbscan.py:320]

### F31 — Stale/misleading narration comment in `plot_hdbscan.py`
severity: low
evidence: examples/cluster/plot_hdbscan.py:28 — comment reads "# Black removed and is used for noise instead.", copy-pasted from the DBSCAN example where a specific color was removed from the palette; in this example nothing is removed — the Spectral palette is used unchanged and `col` is only overridden to black inside the `k == -1` branch (:36-37).
scenario: "Reader trying to understand the color scheme searches for the 'removed' color logic → wastes time hunting for palette manipulation code that does not exist"
contract: Replace the comment with an accurate description, e.g. "Noise points (label -1) are drawn in black; other clusters use the Spectral palette."
instances: single-instance

### F32 — `dbscan_clustering` docstring uses first-person narration and duplicated tautology instead of a precise invariant
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:926-937 — "Return clustering that would be equivalent to running DBSCAN* for a particular cut_distance (or epsilon) DBSCAN* can be thought of as DBSCAN without the border points." (missing punctuation between two sentences), followed by "This can also be thought of as a flat clustering derived from constant height cut through the single linkage tree." and "This represents the result of selecting a cut value for robust single linkage clustering." — three overlapping narrations of the same idea, and the missing period at ":928" creates a run-on sentence.
scenario: "Reader parsing the docstring is unsure whether the method depends on prior `fit` or on a fresh single-linkage build → the three narrations blur the actual precondition (that `self._single_linkage_tree_` must already exist)"
contract: Collapse the three narrations into a single sentence stating the precondition ("Requires the estimator to be fitted first.") and the operation ("Returns the flat clustering obtained by cutting the fitted single-linkage tree at `cut_distance`, discarding clusters smaller than `min_cluster_size` as noise."), and fix the missing period at line 928.
instances: single-instance

### F33 — `HDBSCAN` docstring's `store_centers="centroid"` description omits that it is only meaningful for Euclidean-compatible metrics [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:508-511 — "`"centroid"` which calculates the center by taking the weighted average of their positions. Note that the algorithm uses the euclidean metric and does not guarantee that the output will be an observed data point." However, `_weighted_cluster_center` at sklearn/cluster/_hdbscan/hdbscan.py:912 uses `np.average` regardless of the user's `metric`, and there is no validation preventing `metric="precomputed"` combined with `store_centers="centroid"` (which would attempt to average distance-matrix rows as if they were coordinates).
scenario: "User with `metric='precomputed'` sets `store_centers='centroid'` → silently receives meaningless 'centroids' that are averages of distance-matrix rows, because the docstring does not warn nor does the code guard"
contract: The docstring must explicitly state that `store_centers` is only valid when `X` is a feature matrix (not a precomputed distance matrix), and mention that centroids are computed under the Euclidean metric regardless of the estimator's `metric` argument.
instances: single-instance
