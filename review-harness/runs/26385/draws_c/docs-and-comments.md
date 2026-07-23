### F1 — Public docstrings litter Sphinx-rendered API pages with repeated typos
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:91,102,103,143,144 ("mututal_reachability_graph", "mutual-reahability", "collecteion"); sklearn/cluster/_hdbscan/_linkage.pyx:75,76,137,138,228,229 (same "mutual-reahability", "collecteion"); sklearn/cluster/_hdbscan/_reachability.pyx:76 ("mututal_reachability_graph") and :127,144,149,182,206,209,210 ("mutual_reachibility_distance"); sklearn/cluster/_hdbscan/_tree.pyx:133 ("smaler than this"), :503 ("largest lambda of any simbling node")
scenario: "user reads the rendered API docs / hovers docstrings in IDE → sees repeated misspellings that undermine credibility and complicate future grep/refactor"
contract: fix each typo to its correct spelling ("mutual", "reachability", "collection", "smaller", "sibling")
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_reachability.pyx:76, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503]

### F2 — "The single-linkage tree tree (dendrogram)" — doubled word in every docstring using it
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:149, :220, :326, :361 and sklearn/cluster/_hdbscan/_linkage.pyx:234 — verbatim phrase "The single-linkage tree tree (dendrogram) built from the MST"
scenario: "user reads Returns section of `_hdbscan_brute` / `_hdbscan_prims` / `_process_mst` / `remap_single_linkage_tree` / `make_single_linkage` → sees the same doubled 'tree tree' typo everywhere"
contract: replace "The single-linkage tree tree (dendrogram)" with "The single-linkage tree (dendrogram)" at each site
instances: [sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/_linkage.pyx:234]

### F3 — `_hdbscan_brute` docstring parameter block names a parameter the function does not have
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:88-104 — signature is `def _brute_mst(mutual_reachability, min_samples)`, but the docstring's Parameters block leads with `mututal_reachability_graph: {ndarray, sparse matrix} …`; the real parameter `mutual_reachability` is never documented, and `min_samples` is documented with `default=None` although the signature has no default and the caller passes an int
scenario: "developer looks at `_brute_mst` docstring to understand its contract → param name doesn't match the signature, so IDE tooltips and Sphinx-generated help are misleading"
contract: rename the documented parameter to `mutual_reachability` and drop the false `default=None` (the parameter is required)
instances: single-instance

### F4 — Stale/incorrect `default=` values in `_hdbscan_brute` / `_hdbscan_prims` docstrings
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:160-165 defines `def _hdbscan_brute(X, min_samples=5, alpha=None, ...)` yet the docstring at :179 says `min_samples : int, default=None` and at :183 says `alpha : float, default=1.0`; sklearn/cluster/_hdbscan/hdbscan.py:269-278 defines `_hdbscan_prims(..., min_samples=5, alpha=1.0, ...)` yet the docstring at :290 says `min_samples : int, default=None`
scenario: "developer reads the docstring for the private algorithm dispatchers → concludes `min_samples` defaults to None (matching public API), or that `alpha` defaults to 1.0 in `_hdbscan_brute` when in fact its default is None"
contract: update each `default=` in the docstrings to match the signature — `min_samples : int, default=5` in both functions, and remove the misleading `default=1.0` from `_hdbscan_brute`'s `alpha` documentation (the actual default is `None`)
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist on the function
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — `def _hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)`, no `copy` argument; yet its docstring at :313-318 contains a full "copy : bool, default=False" entry describing behavior that is only implemented in `_hdbscan_brute`
scenario: "reader trusts the docstring → concludes `copy` is honored on the KD/BallTree code path, but `_hdbscan_prims` silently ignores the setting and always operates on `X` as-is"
contract: delete the `copy` block from `_hdbscan_prims`'s docstring — the parameter does not exist on this function
instances: single-instance

### F6 — Comment describes outlier remapping with the wrong label numbers
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Remap indices to align with original data in the case of non-finite entries. Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." But `_OUTLIER_ENCODING` (:71-84) defines `infinite → -2` and `missing → -3`, and the code immediately below (:842-843) writes those exact -2/-3 labels
scenario: "reader trusts the comment while auditing outlier handling → concludes non-finite samples become -1 (noise) and NaNs become -2, but in reality np.inf ↦ -2 and np.nan ↦ -3, which changes downstream semantics"
contract: rewrite the comment to match the code and `_OUTLIER_ENCODING`: "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3."
instances: single-instance

### F7 — `plot_hdbscan.py` cross-reference to the User Guide uses wrong-case label
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104 — `# clusters from all possible clusters (see :ref:\`User Guide <HDBSCAN>\`)`. The label defined in doc/modules/clustering.rst:971 is `.. _hdbscan:` (lowercase); Sphinx `:ref:` targets are case-sensitive
scenario: "docs build renders the example → the User-Guide cross-reference resolves as an undefined-label warning (or renders as literal text), instead of linking to the new HDBSCAN chapter"
contract: change `<HDBSCAN>` to `<hdbscan>` to match the label defined in `doc/modules/clustering.rst`
instances: single-instance

### F8 — User Guide uses `minimum_cluster_size` for the parameter actually named `min_cluster_size`
severity: medium
evidence: doc/modules/clustering.rst:1057-1060 — "HDBSCAN can be smoothed with an additional hyperparameter `min_cluster_size` which specifies that during the hierarchical clustering, components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples` to couple the parameters"
scenario: "user reads the User Guide → tries `HDBSCAN(minimum_cluster_size=…)` → `TypeError: __init__() got an unexpected keyword argument`, because the public parameter is `min_cluster_size`"
contract: replace both occurrences of `minimum_cluster_size` in this paragraph with `min_cluster_size`
instances: [doc/modules/clustering.rst:1059, doc/modules/clustering.rst:1060]

### F9 — User Guide typo/broken sentence in the Mutual Reachability Graph section
severity: low
evidence: doc/modules/clustering.rst:1009-1011 — "by removing any edges with value greater than :math:`\varepsilon`: from the original graph. Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." Two spurious trailing colons after `\varepsilon` and the word "staged" (should be "stage")
scenario: "reader loads the rendered User Guide → sees a stray ':' after each ε and the ungrammatical phrase 'at this staged marked as noise' verbatim in the docs"
contract: drop the trailing `:` after each `\varepsilon` inline math and change "at this staged marked as noise" to "at this stage marked as noise"
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011]

### F10 — User Guide splits "fully-connected" across a line, breaking the compound word in the rendered output
severity: low
evidence: doc/modules/clustering.rst:1025-1026 — "HDBSCAN first extracts a minimum spanning tree (MST) from the fully\n-connected mutual reachability graph" (the hyphen sits at the start of the following line, so Sphinx renders it as "fully -connected")
scenario: "user reads the rendered HTML → sees 'fully -connected' with a leading-space hyphen instead of a hyphenated compound word"
contract: reflow so "fully-connected" sits on a single line (join the two source lines)
instances: single-instance

### F11 — `bfs_from_hierarchy` comment still references the old 2-D scipy layout, not the current struct fields
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:97-98 — comment reads "By construction, node i is formed by the union of nodes hierarchy[i - n_samples, 0] and hierarchy[i - n_samples, 1]", but immediately below (:109-111) the code accesses `hierarchy[node].left_node` and `hierarchy[node].right_node` — the array is a struct-dtype (see `HIERARCHY_dtype` at _tree.pyx:48-53), not a 2-D `[n, 0/1]` array
scenario: "reader tries to follow the comment while modifying the BFS → is misled into thinking they can index into columns 0/1, which will fail with the packed struct dtype"
contract: rewrite the comment to reference the actual field names: "node i is formed by the union of nodes `hierarchy[i - n_samples].left_node` and `hierarchy[i - n_samples].right_node`"
instances: single-instance

### F12 — `_condense_tree` / `_do_labelling` / `_get_clusters` docstrings misstate condensed-tree shape as `(n_samples,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:138, :447, :656 — three docstrings claim `condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype`, and :129 claims `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`; the single-linkage `hierarchy` actually has length `n_samples - 1` (used explicitly at :90, :146, :151) and the condensed tree is an edgelist whose row count is not a function of `n_samples` in general
scenario: "developer relies on the documented shape to size buffers or write assertions → uses `n_samples` where the real length is `n_samples - 1` (hierarchy) or a data-dependent number (condensed tree), producing off-by-one bugs or over-allocations"
contract: change the documented shapes to accurately describe them — `hierarchy : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype` and `condensed_tree : ndarray of shape (n_edges,), dtype=CONDENSED_dtype`
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F13 — `_brute_mst` sparse-branch comment is ungrammatical and drops the "points" noun
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:108-110 — "Check connected component on mutual reachability / If more than one component, it means that even if the distance matrix X has one component, there exists with less than `min_samples` neighbors"
scenario: "reader trying to understand why a sparse `min_samples` failure path exists parses the comment → grammatically broken sentence ('there exists with less than … neighbors') missing subject/noun ('points')"
contract: rewrite as "…it means points exist with fewer than `min_samples` neighbors" and drop the surrounding fragment so the comment is a complete sentence
instances: single-instance

### F14 — `_do_labelling` docstring header sentence is broken across two lines with wrong hyphenation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:440-443 — "Note that this is where points may be marked as noisy outliers. The determination of some points as noise is in large, single-cluster datasets is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters." The phrase "The determination of some points as noise is in large, single-cluster datasets is controlled" has a superfluous "is" and the hyphenated "single-cluster" is split with a line break
scenario: "reader opens rendered Sphinx docs → sees a doubled 'is' and a garbled sentence in the summary of the private helper"
contract: reflow to "The determination of some points as noise, in large single-cluster datasets, is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters."
instances: single-instance

### F15 — `_hdbscan_prims` and `_hdbscan_brute` docstrings describe the "mutual_reachability_graph" path but `_hdbscan_prims` never calls it
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-283 — `_hdbscan_prims` docstring copy-pasted from `_hdbscan_brute` still says "Otherwise, the pairwise distances are calculated directly and passed to `mutual_reachability_graph`." In reality `_hdbscan_prims` never calls `mutual_reachability_graph`; it uses `NearestNeighbors` + `mst_from_data_matrix` (see :332-347), and mutual reachability is computed implicitly inside `mst_from_data_matrix` (comment at :346)
scenario: "developer tracing where mutual reachability is computed in the KD/BallTree path follows the docstring → greps for `mutual_reachability_graph` inside `_hdbscan_prims`, finds nothing, and is misled about the algorithm's data flow"
contract: rewrite `_hdbscan_prims`'s summary to describe its actual behavior — computing k-nearest-neighbour core distances and calling `mst_from_data_matrix`, in which mutual reachability is implicit — and drop the reference to `mutual_reachability_graph`
instances: single-instance
