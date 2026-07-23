### F1 — HDBSCAN docstring lists `n_jobs` default as `None` but constructor defaults to `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 documents `n_jobs : int, default=None` while sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4` in `__init__`; the docstring further asserts "`None` means 1 unless in a :obj:`joblib.parallel_backend` context." at line 488, giving users a wrong picture of the built-in parallelism.
scenario: "user reads the public docstring / help(HDBSCAN) to decide whether to set n_jobs → assumes single-threaded default and does not realize the estimator silently uses 4 workers"
contract: Update the docstring to `n_jobs : int, default=4` and rewrite the description to match the actual constructor default, so the public signature and prose agree.
instances: single-instance

### F2 — Comment in `fit()` describes the wrong label mapping for non-finite samples
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 says "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2" — the actual `_OUTLIER_ENCODING` (line 65-79) maps `infinite → -2` and `missing → -3`, and the following lines 842-843 write those very values.
scenario: "maintainer reads the comment to understand outlier remapping → forms an inverted mental model of the label encoding and may 'fix' downstream code to the wrong convention"
contract: Update the comment to state that samples with `np.inf` are mapped to -2 and samples with `np.nan` are mapped to -3, matching `_OUTLIER_ENCODING`.
instances: single-instance

### F3 — User Guide inverts the core-distance/noise condition for DBSCAN*
severity: medium
evidence: doc/modules/clustering.rst:1010 reads "Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise" — for DBSCAN* with parameter :math:`\varepsilon`, a point is core (not noise) exactly when its core distance is ≤ :math:`\varepsilon`; noise corresponds to core distance **greater** than :math:`\varepsilon`.
scenario: "user reads the User Guide to learn HDBSCAN's noise rule → applies the opposite condition when reasoning about parameter choices"
contract: Change the sentence to say points whose core distance is **greater than** :math:`\varepsilon` are marked as noise, and drop the stray colons after `\varepsilon` and the typo "at this staged marked" (should be "at this stage marked").
instances: single-instance

### F4 — User Guide references a nonexistent parameter name `minimum_cluster_size`
severity: medium
evidence: doc/modules/clustering.rst:1057-1060 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples` …"; the estimator's parameter is `min_cluster_size` (see hdbscan.py:435 and the constructor signature).
scenario: "user copies the recommendation `HDBSCAN(minimum_cluster_size=min_samples)` → `TypeError: unexpected keyword argument`; or is confused into thinking two different knobs exist"
contract: Use `min_cluster_size` consistently in the User Guide narrative, matching the estimator parameter.
instances: [doc/modules/clustering.rst:1058, doc/modules/clustering.rst:1059, doc/modules/clustering.rst:1060]

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter and precomputed metric it does not support
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-283 states "If `metric='precomputed'` then `X` must be a symmetric array of distances" and lines 313-318 document a `copy : bool, default=False` parameter, but the signature at lines 269-277 accepts neither `copy` nor a precomputed path (it always calls `NearestNeighbors(...).fit(X)` and `DistanceMetric.get_metric(metric, ...)`, both of which reject `"precomputed"`).
scenario: "maintainer sees the copy/precomputed language and threads `copy=self.copy` into the prims path, or a user infers that KDTree/BallTree accept precomputed input → runtime error or silent broken assumption"
contract: Rewrite the `_hdbscan_prims` docstring so the summary describes only raw-data input, document `algo` and `leaf_size` (present in the signature), and drop the `copy` and precomputed language.
instances: single-instance

### F6 — Attribute docstring claims -3 is excluded from `n_clusters`, but the computation excludes only -1 and -2
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:562-563 and 578-579 state "n_clusters only counts non-outlier clusters. That is to say, the -1, -2, -3 labels for the outlier clusters are excluded." — the actual computation at line 895 is `n_clusters = len(set(self.labels_) - {-1, -2})`, which does not remove -3 (missing).
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X_with_nan)` → the docstring says `centroids_` will contain only real clusters; instead one row is allocated for the -3 group and either raises or produces a bogus centroid, contradicting the documented contract [out-of-theme]"
contract: Fix line 895 to compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` so the computation matches the documented `_OUTLIER_ENCODING` contract.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:562-563, sklearn/cluster/_hdbscan/hdbscan.py:578-579]

### F7 — Test uses stale `prims_kdtree`/`prims_balltree` names that are not valid `algorithm` values [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` for `tree in ("kd", "ball")`, but the parameter constraints (sklearn/cluster/_hdbscan/hdbscan.py:635-644) only accept `{"auto", "brute", "kdtree", "balltree"}` — so this test passes only because `_validate_params` raises `InvalidParameterError` (a `ValueError` subclass) for the wrong reason, not because the "precomputed + tree" combination raises the intended error at hdbscan.py:786-789. The test docstring claims to check the tree-vs-precomputed guard, but that code path is never exercised.
scenario: "someone tightens `_validate_params` error messages and adjusts this test → discovers the actual precomputed guard has been silently untested; conversely, if the `algorithm` value list changes, this test silently keeps passing without exercising anything"
contract: Change the parametrized values to the real algorithm names `"kdtree"` / `"balltree"` so the test actually exercises the "precomputed disallowed for tree algorithms" branch.
instances: single-instance

### F8 — Scale-invariance example plots identical data three times [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — "fig, axes = plt.subplots(3, 1, figsize=(10, 12)) / hdb = HDBSCAN() / for idx, scale in enumerate((1, 0.5, 3)): / hdb.fit(X) / plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={'scale': scale})"
scenario: "A user reads the surrounding narrative — 'HDBSCAN is scale-invariant. …One immediate advantage is that HDBSCAN is scale-invariant' — and expects the three subplots to demonstrate identical clusterings across three distinct scales → in fact all three subplots fit and plot the same un-scaled `X`, so the identical plots are a tautology (same input, same output) rather than a demonstration of scale invariance, while the axis title still displays scale=0.5, scale=3 as if the data had been rescaled."
contract: Change the loop body to `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={'scale': scale})`, so the demonstration actually varies the input by `scale`, mirroring the DBSCAN block immediately above (plot_hdbscan.py:85-89).
instances: single-instance

### F9 — `plot_hdbscan.py` cross-reference to the User Guide uses wrong-case label
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104 — `# clusters from all possible clusters (see :ref:\`User Guide <HDBSCAN>\`)`. The label defined in doc/modules/clustering.rst:971 is `.. _hdbscan:` (lowercase); Sphinx `:ref:` targets are case-sensitive
scenario: "docs build renders the example → the User-Guide cross-reference resolves as an undefined-label warning (or renders as literal text), instead of linking to the new HDBSCAN chapter"
contract: change `<HDBSCAN>` to `<hdbscan>` to match the label defined in `doc/modules/clustering.rst`
instances: single-instance

### F10 — `_get_clusters` docstring promises a `stabilities` return value that the function never emits
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:695-697 documents "stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster." but the function returns `(labels, probs)` at line 803 and `tree_to_labels` in the same file only unpacks two values (line 76-83).
scenario: "maintainer follows the docstring to consume `stabilities` → `ValueError: not enough values to unpack`"
contract: Remove the `stabilities` entry from the Returns section of `_get_clusters` so the docstring reflects the actual `(labels, probabilities)` tuple.
instances: single-instance

### F11 — `remap_single_linkage_tree` docstring calls the reinserted non-finite samples "noise points"
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:355-356 — "These points will all be merged into the final node at np.inf distance and considered noise points." Non-finite samples are labeled -2 (infinite) or -3 (missing) by `_OUTLIER_ENCODING`, not the noise label -1.
scenario: "reader following the label convention downstream trusts the docstring and treats -2/-3 samples as ordinary noise → misreports outlier accounting"
contract: Replace "considered noise points" with wording that says they are assigned the infinite/missing outlier labels defined by `_OUTLIER_ENCODING`.
instances: single-instance

### F12 — Parameter docstrings misspell the parameter name as `mututal_reachability_graph`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:91 and sklearn/cluster/_hdbscan/_reachability.pyx:82 use the identifier `mututal_reachability_graph:` in the Parameters/Returns block; the actual parameter of `_brute_mst` is `mutual_reachability` (line 82), and the returned name of `mutual_reachability_graph` in `_reachability.pyx` is spelled correctly at line 50.
scenario: "numpydoc lint / IDE hover cross-references the docstring identifier → resolves to a nonexistent name and readers second-guess whether `mututal_*` is a separate thing"
contract: Correct the spelling to `mutual_reachability` in `hdbscan.py:91` and `mutual_reachability_graph` in the Returns block of `_reachability.pyx:82` so the docstring identifiers match real names.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/_reachability.pyx:82, sklearn/cluster/_hdbscan/_reachability.pyx:76]

### F13 — `_sparse_mutual_reachability_graph` docstring lists a `distance_matrix` parameter that is not part of the signature
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:172-175 documents `distance_matrix : sparse matrix of shape (n_samples, n_samples)` under Parameters, but the signature (lines 158-165) accepts `data`, `indices`, `indptr`, `n_samples`, `further_neighbor_idx`, and `max_distance` — no `distance_matrix`. The three CSR component arrays are undocumented.
scenario: "maintainer reads the docstring to understand the sparse contract → forms a wrong picture of how the input arrives, or attempts to call with a keyword that does not exist"
contract: Rewrite the Parameters block to enumerate `data`, `indices`, `indptr`, `n_samples`, `further_neighbor_idx`, and `max_distance`, and drop the fictitious `distance_matrix` entry.
instances: single-instance

### F14 — Sparse docstring references `max_dist` in narration while the parameter is `max_distance`
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:71 and sklearn/cluster/_hdbscan/_reachability.pyx:184 both end with "it is instead truncated to `max_dist`" but the parameter is spelled `max_distance` (line 51 and 164) and this is the name used in `_brute_mst`/`fit`.
scenario: "user searches the codebase for `max_dist` to find the option → finds nothing; or infers the option is called `max_dist`"
contract: Rename the reference in prose to `max_distance` to match the parameter name.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:71, sklearn/cluster/_hdbscan/_reachability.pyx:184]

### F15 — `_hdbscan_brute` / `_hdbscan_prims` docstrings advertise defaults that contradict the signatures
severity: low
evidence: `_hdbscan_brute` signature (sklearn/cluster/_hdbscan/hdbscan.py:158-166) has `min_samples=5, alpha=None`, while the docstring says `min_samples : int, default=None` (line 179) and `alpha : float, default=1.0` (line 183); `_hdbscan_prims` signature (line 269-277) has `min_samples=5, alpha=1.0` and takes required `algo` plus `leaf_size=40`, while the docstring at lines 290-291 and 313-318 documents `min_samples : int, default=None` and a nonexistent `copy` parameter but never mentions `algo` or `leaf_size`.
scenario: "reader consults the helper docstrings to reason about defaults → picks the wrong default value, and a user experimenting via internal helpers passes `copy=...` and gets `TypeError`"
contract: Match documented defaults to the signatures (`min_samples=5`; `alpha=None` for `_hdbscan_brute` and `alpha=1.0` for `_hdbscan_prims`), add `algo` / `leaf_size` documentation to `_hdbscan_prims`, and remove the fake `copy` entry from `_hdbscan_prims`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:290, sklearn/cluster/_hdbscan/hdbscan.py:313-318]

### F16 — Persistent typos in narration across docstrings and comments
severity: low
evidence: The strings `mutual-reahability` and `collecteion` appear together in four "Returns" blocks (sklearn/cluster/_hdbscan/hdbscan.py:102-103, 143-144; sklearn/cluster/_hdbscan/_linkage.pyx:75-76, 137-138, 228-229); `mutual_reachibility_distance` is the local variable name used throughout `_reachability.pyx` (lines 127, 144, 149, 182, 206, 209, 210); `smaler` appears at sklearn/cluster/_hdbscan/_tree.pyx:133; `simbling` at sklearn/cluster/_hdbscan/_tree.pyx:503; and "single-linkage tree tree" (doubled "tree") appears at sklearn/cluster/_hdbscan/hdbscan.py:149, 220, 326, 361 and sklearn/cluster/_hdbscan/_linkage.pyx:234.
scenario: "developer greps for the correct spelling (`reachability`, `sibling`, `collection`, `smaller`) → misses these sites and doubts the accuracy of the surrounding text"
contract: Fix each misspelling to the intended English word and remove the duplicated "tree tree" so that the identifiers, prose, and cited noun phrases read consistently as "mutual-reachability", "sibling", "collection", "smaller", and "single-linkage tree (dendrogram)".
instances: [sklearn/cluster/_hdbscan/hdbscan.py:102-103, sklearn/cluster/_hdbscan/hdbscan.py:143-144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/_linkage.pyx:75-76, sklearn/cluster/_hdbscan/_linkage.pyx:137-138, sklearn/cluster/_hdbscan/_linkage.pyx:228-229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503]

### F17 — Narration comment in `_brute_mst` has broken grammar and conveys the wrong invariant
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:108-110: "Check connected component on mutual reachability / If more than one component, it means that even if the distance matrix X has one component, there exists with less than `min_samples` neighbors". The clause "there exists with less than" is ungrammatical, and the intended meaning ("some point has fewer than min_samples finite neighbors") is not conveyed.
scenario: "maintainer reading the guard tries to reason about when to relax it → cannot tell what the invariant is and has to reverse-engineer it from the error message"
contract: Rewrite the comment to state plainly: "if the mutual-reachability graph has more than one connected component, some points have fewer than min_samples finite neighbors — even when the raw distance matrix is connected — so an MST cannot be built."
instances: single-instance

### F18 — Comment in `_do_labelling` claims the expression returns "a unique, scalar lambda value" when it returns a 1-element ndarray
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:496-497 says "There can only be one edge with this particular child hence this expression extracts a unique, scalar lambda value." — the expression `parent_lambda = lambda_array[child_array == n]` at line 498 evaluates to a length-1 `ndarray`, not a Python scalar; the subsequent `parent_lambda >= threshold` (line 505) is a broadcasted comparison that returns another length-1 array, and the `if` implicitly unwraps that.
scenario: "maintainer changes the code to rely on the 'scalar' promise (e.g. `float(parent_lambda)` or direct arithmetic) → hits a shape mismatch when the child appears in an unexpected number of edges"
contract: Reword the comment to reflect reality: `parent_lambda` is a length-1 array (uniqueness is enforced by construction of the condensed tree) and the subsequent scalar-like comparison relies on that length-1 broadcast.
instances: single-instance

### F19 — Stray/garbled narration in `_weighted_cluster_center`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 reads "Need to handle iteratively seen each cluster may have a different number of samples, hence we can't create a homogenous 3D array." — the sentence is ungrammatical ("handle iteratively seen" is not a construction) and "homogenous" should be "homogeneous".
scenario: "reader tries to understand why the loop is per-cluster → gets distracted parsing the ungrammatical sentence instead of grasping the intent"
contract: Replace with a plain statement: "Iterate per cluster because each cluster may contain a different number of samples, so a single 3-D array of samples×features per cluster is not possible."
instances: single-instance

### F20 — Stray hyphen from an in-word line break in the User Guide narrative
severity: low
evidence: doc/modules/clustering.rst:1025-1026 reads "HDBSCAN first extracts a minimum spanning tree (MST) from the fully\n-connected mutual reachability graph" — the `-` at the start of line 1026 becomes a literal hyphen in the rendered output, producing "fully -connected".
scenario: "user reads the rendered User Guide → sees a dangling hyphen and momentarily doubts what 'fully -connected' means"
contract: Join the two lines so the phrase renders as "fully connected", with no stray hyphen.
instances: single-instance

### F21 — `_hdbscan_prims` docstring omits `algo` and `leaf_size`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature has `algo`, `leaf_size=40`; the docstring at hdbscan.py:279-326 documents `X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy` (which doesn't exist — see F5), and `metric_params` but never mentions `algo` or `leaf_size`.
scenario: "A maintainer reads the docstring to understand how to invoke `_hdbscan_prims` → they cannot tell how `algo` selects kd_tree vs. ball_tree or what `leaf_size` controls, forcing them to inspect the call site to reverse-engineer semantics."
contract: Add Parameter entries for `algo` (values `{'kd_tree', 'ball_tree'}` passed to `NearestNeighbors`) and `leaf_size` (mirrors the class parameter), matching the entries in the `HDBSCAN` class docstring.
instances: single-instance

### F22 — `mst_from_data_matrix` docstring omits `alpha`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:111-138 — signature includes `float64_t alpha=1.0`, but the Parameters block documents only `raw_data`, `core_distances`, `dist_metric` and never `alpha`.
scenario: "A caller wanting to know what `alpha` does in the prims path finds no documentation → they must read the body to discover that it scales `pair_distance /= alpha` at _linkage.pyx:187, wasting time and risking misuse."
contract: Add an `alpha : float, default=1.0` entry to the Parameters block, mirroring the description used in `HDBSCAN`'s class docstring.
instances: single-instance

### F23 — User Guide typo/broken sentence in the Mutual Reachability Graph section
severity: low
evidence: doc/modules/clustering.rst:1009-1011 — "by removing any edges with value greater than :math:`\varepsilon`: from the original graph. Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." Two spurious trailing colons after `\varepsilon` and the word "staged" (should be "stage")
scenario: "reader loads the rendered User Guide → sees a stray ':' after each ε and the ungrammatical phrase 'at this staged marked as noise' verbatim in the docs"
contract: drop the trailing `:` after each `\varepsilon` inline math and change "at this staged marked as noise" to "at this stage marked as noise"
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011]

### F24 — `bfs_from_hierarchy` comment still references the old 2-D scipy layout, not the current struct fields
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:97-98 — comment reads "By construction, node i is formed by the union of nodes hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]", but immediately below (:109-111) the code accesses `hierarchy[node].left_node` and `hierarchy[node].right_node` — the array is a struct-dtype (see `HIERARCHY_dtype` at _tree.pyx:48-53), not a 2-D `[n, 0/1]` array
scenario: "reader tries to follow the comment while modifying the BFS → is misled into thinking they can index into columns 0/1, which will fail with the packed struct dtype"
contract: rewrite the comment to reference the actual field names: "node i is formed by the union of nodes `hierarchy[i - n_samples].left_node` and `hierarchy[i - n_samples].right_node`"
instances: single-instance

### F25 — `_condense_tree` / `_do_labelling` / `_get_clusters` docstrings misstate condensed-tree shape as `(n_samples,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:138, :447, :656 — three docstrings claim `condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype`, and :129 claims `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`; the single-linkage `hierarchy` actually has length `n_samples - 1` (used explicitly at :90, :146, :151) and the condensed tree is an edgelist whose row count is not a function of `n_samples` in general
scenario: "developer relies on the documented shape to size buffers or write assertions → uses `n_samples` where the real length is `n_samples - 1` (hierarchy) or a data-dependent number (condensed tree), producing off-by-one bugs or over-allocations"
contract: change the documented shapes to accurately describe them — `hierarchy : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype` and `condensed_tree : ndarray of shape (n_edges,), dtype=CONDENSED_dtype`
instances: [sklearn/cluster/_hdbscan/_tree.pyx:63 area, sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:135, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:377, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656, sklearn/cluster/_hdbscan/_linkage.pyx:80 area]

### F26 — `_do_labelling` docstring header sentence is broken across two lines with wrong hyphenation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:440-443 — "Note that this is where points may be marked as noisy outliers. The determination of some points as noise is in large, single-cluster datasets is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters." The phrase "The determination of some points as noise is in large, single-cluster datasets is controlled" has a superfluous "is" and the hyphenated "single-cluster" is split with a line break
scenario: "reader opens rendered Sphinx docs → sees a doubled 'is' and a garbled sentence in the summary of the private helper"
contract: reflow to "The determination of some points as noise, in large single-cluster datasets, is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters."
instances: single-instance

### F27 — `_hdbscan_prims` and `_hdbscan_brute` docstrings describe the "mutual_reachability_graph" path but `_hdbscan_prims` never calls it
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-283 — `_hdbscan_prims` docstring copy-pasted from `_hdbscan_brute` still says "Otherwise, the pairwise distances are calculated directly and passed to `mutual_reachability_graph`." In reality `_hdbscan_prims` never calls `mutual_reachability_graph`; it uses `NearestNeighbors` + `mst_from_data_matrix` (see :332-347), and mutual reachability is computed implicitly inside `mst_from_data_matrix` (comment at :346)
scenario: "developer tracing where mutual reachability is computed in the KD/BallTree path follows the docstring → greps for `mutual_reachability_graph` inside `_hdbscan_prims`, finds nothing, and is misled about the algorithm's data flow"
contract: rewrite `_hdbscan_prims`'s summary to describe its actual behavior — computing k-nearest-neighbour core distances and calling `mst_from_data_matrix`, in which mutual reachability is implicit — and drop the reference to `mutual_reachability_graph`
instances: single-instance
