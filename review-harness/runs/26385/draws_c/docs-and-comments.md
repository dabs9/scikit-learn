### F1 — Stale comment in `fit()` misdescribes outlier label mapping
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment states "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2", but `_OUTLIER_ENCODING` at lines 66-78 assigns `infinite→-2` and `missing→-3`, and lines 842-843 actually use those `_OUTLIER_ENCODING` labels.
scenario: "A future maintainer reads the comment while debugging outlier handling → reasons about the code with the wrong label mapping (-1/-2 instead of -2/-3), leading to incorrect fixes or downstream assumptions."
contract: Update the comment to state that `np.inf` samples are mapped to `-2` and `np.nan` samples are mapped to `-3`, matching `_OUTLIER_ENCODING`.
instances: single-instance

### F2 — `_hdbscan_prims` docstring documents a non-existent `copy` parameter
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — docstring block "copy : bool, default=False If `copy=True` then any time an in-place modifications would be made ...". Function signature at lines 269-278 has no `copy` parameter (only `X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params`).
scenario: "A caller reads the docstring, passes `copy=True` as a positional/kwarg → because the function does not accept `copy`, it silently gets swallowed into `**metric_params` and forwarded to `NearestNeighbors` / `DistanceMetric.get_metric`, producing an obscure downstream failure."
contract: Delete the `copy` block from the `_hdbscan_prims` docstring (it was copy-pasted from `_hdbscan_brute`).
instances: single-instance

### F3 — Docstrings list wrong parameter defaults for helper functions
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:101 (`min_samples : int, default=None` for `_brute_mst` whose signature `def _brute_mst(mutual_reachability, min_samples)` at line 82 has no default), :179 (`min_samples : int, default=None` — but signature line 160 has `min_samples=5`), :183 (`alpha : float, default=1.0` — but signature line 161 has `alpha=None`), :290 (`min_samples : int, default=None` — but `_hdbscan_prims` signature line 272 has `min_samples=5`).
scenario: "A developer reads a docstring, believes the documented default matches the signature and omits the argument expecting `alpha=1.0` → at runtime line 241 executes `distance_matrix /= alpha` with `alpha=None`, raising `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`."
contract: Correct each parameter's default in the docstring to match the actual signature (`_brute_mst.min_samples`: no default, `_hdbscan_brute.min_samples`: 5, `_hdbscan_brute.alpha`: `None`, `_hdbscan_prims.min_samples`: 5).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:101, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F4 — `HDBSCAN.n_jobs` docstring says `default=None` but constructor uses `n_jobs=4`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 — `n_jobs : int, default=None`; sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature contains `n_jobs=4,`.
scenario: "A user reads the API doc, expects the estimator to default to a single job unless a joblib backend is set, but the constructor actually spawns 4 jobs by default → surprise parallelism / resource use, and reproducibility differences vs. the documented behavior."
contract: Change the docstring default to `default=4` (or change the constructor default to `None` to match the doc, but the user-facing docstring must not lie about the default).
instances: single-instance

### F5 — `_sparse_mutual_reachability_graph` docstring lists a `distance_matrix` parameter the function does not accept
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:154-159 (function signature takes `data, indices, indptr, n_samples, further_neighbor_idx, max_distance`) but the docstring parameter block at lines 172-183 describes `distance_matrix : sparse matrix of shape (n_samples, n_samples)` and never documents `data`, `indices`, `indptr`, or `n_samples`.
scenario: "A maintainer trying to understand the internal sparse path reads the docstring, believes the function receives a full CSR matrix, and misroutes callers or misdesigns changes to preprocessing → time wasted reconciling the doc with the actual CSR-arrays contract."
contract: Rewrite the `Parameters` section to describe `data`, `indices`, `indptr`, and `n_samples` (the actual CSR-array inputs), removing the copy-pasted `distance_matrix` block.
instances: single-instance

### F6 — `_get_clusters` docstring lists a third return value (`stabilities`) that the function never returns
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:687-696 (docstring `Returns` block lists `labels`, `probabilities`, and `stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster.`) but the function returns `(labels, probs)` at line 803, and there is no `stabilities` output anywhere in the body.
scenario: "A caller (e.g. someone extending HDBSCAN) reads the docstring and writes `labels, probs, stab = _get_clusters(...)` → `ValueError: not enough values to unpack` at runtime."
contract: Delete the `stabilities` entry from the `Returns` section; the documented outputs must be exactly `labels` and `probabilities`.
instances: single-instance

### F7 — User Guide references a non-existent parameter `minimum_cluster_size`
severity: medium
evidence: doc/modules/clustering.rst:1075-1078 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples` to couple the parameters ...". The actual estimator parameter is `min_cluster_size` (see `HDBSCAN.__init__` sklearn/cluster/_hdbscan/hdbscan.py:651 and `_parameter_constraints` line 617).
scenario: "A user reads the User Guide, tries `HDBSCAN(minimum_cluster_size=min_samples)` → `TypeError: __init__() got an unexpected keyword argument 'minimum_cluster_size'`."
contract: Replace both occurrences of `minimum_cluster_size` in the paragraph with `min_cluster_size`, matching the estimator's public parameter name.
instances: single-instance

### F8 — Malformed RST math role syntax in User Guide (stray trailing colons)
severity: medium
evidence: doc/modules/clustering.rst:1026-1027 — "removing any edges with value greater than :math:`\varepsilon`: from the original graph. Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." The colons immediately after the closing backtick of the `:math:` role are stray, and "at this staged" should be "at this stage".
scenario: "Sphinx renders the paragraph → math role is followed by a bare colon appearing as literal text (or breaks role parsing on some Sphinx versions), and the sentence reads with a typo and misplaced punctuation."
contract: Remove the two stray colons after ``:math:`\varepsilon` `` and fix "at this staged" to "at this stage".
instances: single-instance

### F9 — Repeated docstring typos: "mutual-reahability", "collecteion", "tree tree"
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:75-76, 137-138, 228-229, 234 (all three phrases); sklearn/cluster/_hdbscan/hdbscan.py:91 ("mututal_reachability_graph" spelling in `Returns` name), :102-103, :143-144, :149, :220, :326, :361; sklearn/cluster/_hdbscan/_reachability.pyx:76 ("mututal_reachability_graph"). All are misspellings/duplications in public and internal docstrings.
scenario: "Reader hits multiple visible misspellings in docstrings surfaced by IDE tooltips, `help()`, and the online API reference → decreased trust in the estimator's documentation quality; the duplicated word `tree tree` reads as a bug in the doc."
contract: Fix to "mutual-reachability", "collection", "mutual_reachability_graph", and remove the duplicated "tree" so the phrase reads "The single-linkage tree (dendrogram) built from the MST.".
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/_reachability.pyx:76]

### F10 — Wrong array shapes documented for hierarchy/linkage/condensed_tree in `_tree.pyx` docstrings
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 (docstring: `hierarchy : ndarray of shape (n_samples,)` — but line 146 of the same function derives `n_samples = hierarchy.shape[0] + 1`, so hierarchy has shape `(n_samples - 1,)`); :138 (`condensed_tree : ndarray of shape (n_samples,)` — condensed_tree length is not `n_samples`); :371 (`linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` — same off-by-one); :447 and :656 (both re-use the `(n_samples,)` shape for `condensed_tree`).
scenario: "A caller pre-allocates or reasons about the shape of `hierarchy`/`linkage` based on the docstring and passes an `(n_samples,)`-shaped array or asserts that shape in tests → runtime off-by-one errors or silently incorrect indexing."
contract: Correct the documented shapes so `hierarchy`/`linkage` are `(n_samples - 1,)` (matching `mst_from_*` and `make_single_linkage` outputs) and remove the misleading `(n_samples,)` shape claim for `condensed_tree` (its size depends on tree topology).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F11 — Example narrator claims scale-invariance demo but code never scales the fitted data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104-110 (narrative: "HDBSCAN is scale-invariant. ... One immediate advantage is that HDBSCAN is scale-invariant.") followed by lines 108-110 (`for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ..., parameters={"scale": scale})`). The `scale` loop variable is used only in the plot title — `hdb.fit(X)` and `plot(X, ...)` both use the unscaled `X`, so the three sub-plots are identical and do not demonstrate scale invariance.
scenario: "A user reads the example gallery to convince themselves HDBSCAN is scale-invariant → sees three identical panels labelled with different `scale` values but no evidence of actual rescaling; the narrative and code disagree and the intended pedagogical point is not conveyed."
contract: Fit and plot the rescaled data — `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` — so the code matches the "HDBSCAN is scale-invariant" narrative directly above it.
instances: single-instance

### F12 — Misspelling "simbling" in code comments
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:503 — "The threshold should be calculated per-sample based on the largest lambda of any simbling node."; sklearn/cluster/tests/test_hdbscan.py:530 — "The threshold should be calculated per-sample based on the largest lambda of any simbling node. In this case, all points are siblings ...".
scenario: "Reader searches the code for the invariant on siblings while debugging cluster-selection behavior → grep for `sibling` misses the comment; the misspelling reads as sloppy."
contract: Fix "simbling" → "sibling" in both comments so future greps for the invariant land on the right line.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/tests/test_hdbscan.py:530]

### F13 — "smaler" misspelling in `_condense_tree` docstring
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:133 — "Clusters smaler than this are pruned from the tree." in the `min_cluster_size` parameter description.
scenario: "A user reading `help(_condense_tree)` sees the misspelling in the parameter description → cosmetic loss of trust in the doc."
contract: Fix "smaler" → "smaller".
instances: single-instance
