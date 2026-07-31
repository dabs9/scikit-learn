### F1 — Comment misdocuments outlier label mapping, contradicting `_OUTLIER_ENCODING`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — the comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." Immediately below, `new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` (which is -2 at line 73) and `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` (which is -3 at line 80). The class docstring at line 535-537 correctly documents -2 for infinite and -3 for missing.
scenario: "Future maintainer reads block comment while investigating non-finite handling → uses -1/-2 assumption to write dependent logic (e.g. new outlier subclass) and ships a mismatch against the actual encoding."
contract: Rewrite the comment to state that samples with `np.inf` are relabelled to `_OUTLIER_ENCODING["infinite"]["label"]` (-2) and samples with `np.nan` to `_OUTLIER_ENCODING["missing"]["label"]` (-3), matching the class docstring.
instances: single-instance

### F2 — User Guide names parameter `minimum_cluster_size` that does not exist
severity: high
evidence: doc/modules/clustering.rst:1057-1060 — "HDBSCAN can be smoothed with an additional hyperparameter `min_cluster_size` … components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples`". The public constructor at sklearn/cluster/_hdbscan/hdbscan.py:649 exposes `min_cluster_size`, not `minimum_cluster_size`; the same paragraph even switches between the two names.
scenario: "Reader copies the recommended snippet `minimum_cluster_size = min_samples` into `HDBSCAN(minimum_cluster_size=…)` → `TypeError: __init__() got an unexpected keyword argument 'minimum_cluster_size'` (or the arg is silently absorbed and ignored)."
contract: Replace every occurrence of `minimum_cluster_size` in the paragraph with `min_cluster_size`.
instances: [doc/modules/clustering.rst:1059, doc/modules/clustering.rst:1060]

### F3 — Class docstring lists `n_jobs` default as `None` but constructor defaults to `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 — "n_jobs : int, default=None" followed by "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context." But `__init__` at line 658 defines `n_jobs=4`, so calling `HDBSCAN()` runs with 4 workers, not 1.
scenario: "User reads docstring expecting single-threaded default behaviour and deploys `HDBSCAN()` on a resource-constrained machine → the estimator runs with 4 workers, causing unexpected multi-core usage / oversubscription."
contract: Change the docstring to `n_jobs : int, default=4` and rewrite the description to reflect the actual default (4 parallel jobs), or change the constructor default to `None` to match the documented contract.
instances: single-instance

### F4 — `_hdbscan_prims` docstring documents non-existent `copy` param and omits `algo`, `leaf_size`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature is `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric='euclidean', leaf_size=40, n_jobs=None, **metric_params)`, but the docstring (lines 285-322) documents `X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy`, `metric_params` — `copy` does not exist on this function, and required `algo` / `leaf_size` are undocumented.
scenario: "developer maintaining this file reads the docstring and assumes `_hdbscan_prims` accepts `copy=` (mirroring `_hdbscan_brute`), calls `_hdbscan_prims(..., copy=True)` from a new code path → TypeError; conversely, no docstring guidance exists for the required `algo` positional argument"
contract: Docstring parameter list must be the exact set of the function signature: drop the `copy` block, add `algo` and `leaf_size` sections mirroring the constructor's descriptions.
instances: single-instance

### F5 — `_sparse_mutual_reachability_graph` docstring documents a `distance_matrix` param that does not exist
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:152-179 — function signature is `_sparse_mutual_reachability_graph(data, indices, indptr, n_samples, further_neighbor_idx, max_distance)`, but the Parameters block documents only `distance_matrix`, `further_neighbor_idx`, `max_distance`. Neither the actual positional CSR components (`data`, `indices`, `indptr`, `n_samples`) nor their contracts are described.
scenario: "Contributor extending the sparse path consults the docstring for how to call this helper → writes a call using the phantom `distance_matrix` argument that fails at runtime, and remains misled about how CSR components are passed."
contract: Rewrite the Parameters section to describe `data`, `indices`, `indptr`, and `n_samples` as the CSR components with their shapes/dtypes, and delete the phantom `distance_matrix` entry.
instances: single-instance

### F6 — `_get_clusters` docstring advertises a `stabilities` return that is not returned
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:686-690 — Returns section lists three arrays including "stabilities : ndarray (n_clusters,) / The cluster coherence strengths of each cluster." The function actually ends at line 796 with `return (labels, probs)` — a two-tuple; no stability array is ever produced.
scenario: "Caller does `labels, probs, stabilities = _get_clusters(...)` based on the docstring → `ValueError: not enough values to unpack (expected 3, got 2)`."
contract: Delete the `stabilities` entry from the Returns block so it describes only `(labels, probabilities)`.
instances: single-instance

### F7 — `_do_labelling` docstring omits two of its five parameters
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:437-465 — signature declares `condensed_tree, clusters, cluster_label_map, allow_single_cluster, cluster_selection_epsilon`, but the Parameters block only documents the first three. `allow_single_cluster` and `cluster_selection_epsilon` are undocumented, even though the docstring's own prose ("The determination of some points as noise … is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters.") admits they are load-bearing.
scenario: "User of the public-facing tests (which import `_do_labelling`) reads the incomplete Parameters block → passes wrong types or values for the two undocumented parameters and produces silently wrong labels."
contract: Add Parameters entries for `allow_single_cluster : int` and `cluster_selection_epsilon : float` describing their roles in the epsilon-based noise assignment.
instances: single-instance

### F8 — `_hdbscan_brute`, `_hdbscan_prims`, and `_brute_mst` misstate `min_samples` default (and `_brute_mst` misnames its first parameter)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:179 "min_samples : int, default=None" for `_hdbscan_brute` whose signature at line 160 has `min_samples=5`; line 290 same doc for `_hdbscan_prims` whose signature at line 272 has `min_samples=5`; line 95 same doc for `_brute_mst` whose signature at line 82 makes `min_samples` a required positional (no default). Additionally, `_brute_mst`'s Parameters block at line 91 documents the first argument as `mututal_reachability_graph` (also a typo) while the actual parameter is named `mutual_reachability` (line 82).
scenario: "Contributor relies on the stated `default=None` and calls the helper without `min_samples` → gets the wrong number of neighbours (`_hdbscan_brute`/`_hdbscan_prims` silently use 5) or a `TypeError: missing 1 required positional argument` (`_brute_mst`)."
contract: Fix every entry to state the real default: `min_samples : int, default=5` for `_hdbscan_brute` and `_hdbscan_prims`, `min_samples : int` (required) for `_brute_mst`, and rename `_brute_mst`'s documented parameter from `mututal_reachability_graph` to `mutual_reachability` to match the signature.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91, sklearn/cluster/_hdbscan/hdbscan.py:95, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F9 — `_condense_tree` docstring reports wrong shape for `hierarchy`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 "hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype". The function derives `n_samples = hierarchy.shape[0] + 1` at line 146, so the input's shape is actually `(n_samples - 1,)`. Every peer function that produces or consumes the hierarchy (`make_single_linkage` in _linkage.pyx:239, `labelling_at_cut` at _tree.pyx:377) documents the correct `(n_samples - 1,)` shape.
scenario: "Contributor writing new tests sizes a synthetic hierarchy array to `n_samples` rows per the docstring → `_condense_tree` computes `n_samples + 1` internally and downstream indexing goes off by one."
contract: State the correct shape: `hierarchy : ndarray of shape (n_samples - 1,), dtype=HIERARCHY_dtype`. Similarly correct the `condensed_tree` return-shape from `(n_samples,)` to describe that its size varies with the number of surviving edges.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F10 — RST math syntax broken by stray trailing colons and stray space in the HDBSCAN user guide
severity: medium
evidence: doc/modules/clustering.rst:1009-1010 — "removing any edges with value greater than :math:`\varepsilon`: from the original graph. Any points whose core distance is less than :math:`\varepsilon`: are at this staged marked as noise." The trailing `:` after the closing backtick of each `:math:` role renders as a literal ":" following the math span. Line 1011 also has "at this staged marked" (should be "at this stage"). Line 1025-1026 renders "the fully -connected mutual reachability graph" with an orphan space before the hyphen.
scenario: "Sphinx builds the User Guide with the current markup → renders awkward '`ε`:' punctuation and the sentence 'are at this staged marked as noise', which reads nonsensically to end users."
contract: Delete the stray colons after both `` :math:`\varepsilon` `` tokens, change "at this staged" to "at this stage", and join "fully-connected" without the intervening whitespace.
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011, doc/modules/clustering.rst:1025, doc/modules/clustering.rst:1026]

### F11 — Docstrings list wrong parameter defaults for helper functions
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:101 (`min_samples : int, default=None` for `_brute_mst` whose signature `def _brute_mst(mutual_reachability, min_samples)` at line 82 has no default), :179 (`min_samples : int, default=None` — but signature line 160 has `min_samples=5`), :183 (`alpha : float, default=1.0` — but signature line 161 has `alpha=None`), :290 (`min_samples : int, default=None` — but `_hdbscan_prims` signature line 272 has `min_samples=5`).
scenario: "A developer reads a docstring, believes the documented default matches the signature and omits the argument expecting `alpha=1.0` → at runtime line 241 executes `distance_matrix /= alpha` with `alpha=None`, raising `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`."
contract: Correct each parameter's default in the docstring to match the actual signature (`_brute_mst.min_samples`: no default, `_hdbscan_brute.min_samples`: 5, `_hdbscan_brute.alpha`: `None`, `_hdbscan_prims.min_samples`: 5).
instances: [sklearn/cluster/_hdbscan/hdbscan.py:101, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:183, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F12 — Example narrator claims scale-invariance demo but code never scales the fitted data
severity: medium
evidence: examples/cluster/plot_hdbscan.py:104-110 (narrative: "HDBSCAN is scale-invariant. ... One immediate advantage is that HDBSCAN is scale-invariant.") followed by lines 108-110 (`for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ..., parameters={"scale": scale})`). The `scale` loop variable is used only in the plot title — `hdb.fit(X)` and `plot(X, ...)` both use the unscaled `X`, so the three sub-plots are identical and do not demonstrate scale invariance.
scenario: "A user reads the example gallery to convince themselves HDBSCAN is scale-invariant → sees three identical panels labelled with different `scale` values but no evidence of actual rescaling; the narrative and code disagree and the intended pedagogical point is not conveyed."
contract: Fit and plot the rescaled data — `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` — so the code matches the "HDBSCAN is scale-invariant" narrative directly above it.
instances: single-instance

### F13 — Class docstring narration lists non-existent parameter name in error text
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:118-122 error message tells users to `Ensure your distance matrix has non-zero values for at least `min_sample`={min_samples} neighbors for each points` — the parameter is `min_samples` (with the trailing `s`), and grammar reads "for each points" instead of "for each point". [out-of-theme]
scenario: "user hits the ValueError, searches the estimator API for `min_sample` → nothing found, wastes time; the message also has a grammar error visible to end users"
contract: Rename to `min_samples` (the actual parameter) and fix "for each points" → "for each point".
instances: single-instance

### F14 — Example uses a `:ref:` target that does not exist (case mismatch)
severity: low
evidence: examples/cluster/plot_hdbscan.py:104 has `see :ref:`User Guide <HDBSCAN>`` — the only defined label is `.. _hdbscan:` at doc/modules/clustering.rst:971 (lowercase). ReST/Sphinx reference targets are case-sensitive; no `HDBSCAN` label exists in the docs tree.
scenario: "sphinx builds this example → the ref becomes an unresolved reference warning; the rendered example page has broken link text"
contract: Use `:ref:`User Guide <hdbscan>`` matching the declared label exactly.
instances: single-instance

### F15 — `dbscan_clustering` docstring has typo and dropped period
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:927 "particular cut_distance (or epsilon) DBSCAN* can be thought of as" — missing period between `(or epsilon)` and `DBSCAN*`, producing a run-on sentence; sklearn/cluster/_hdbscan/hdbscan.py:946 "Clusters smaller than this value with be called 'noise'" — `with be called` should be `will be called`; sklearn/cluster/_hdbscan/hdbscan.py:937 "and cluster smaller than `min_cluster_size`" — `cluster smaller` should be `clusters smaller` (subject/verb agreement).
scenario: "user reads the public method's docstring and encounters a run-on sentence and a "with be" grammar error → the method is a user-facing API entry point, so this is high-visibility"
contract: Add the missing period after `(or epsilon)`; `with be called` → `will be called`; `and cluster smaller than` → `and clusters smaller than`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:927, sklearn/cluster/_hdbscan/hdbscan.py:937, sklearn/cluster/_hdbscan/hdbscan.py:946]

### F16 — Broken narration in `_weighted_cluster_center` comment obscures the invariant
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 "Need to handle iteratively seen each cluster may have a different number of samples, hence we can't create a homogenous 3D array." The word `seen` should be `since` (or the sentence is otherwise malformed); "homogenous" is also nonstandard for "homogeneous".
scenario: "reader trying to understand why the loop can't be vectorized reads a nonsensical sentence and cannot recover the invariant (each cluster's mask has variable size) → the whole point of the comment is lost"
contract: `iteratively seen each cluster` → `iteratively since each cluster`; `homogenous` → `homogeneous`.
instances: single-instance

### F17 — `_do_labelling` docstring says "The determination of some points as noise is in large, single-cluster datasets is controlled by" — duplicated "is"
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:441-443 "The determination of some points as noise is in large, single-cluster datasets is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters." — grammatically broken (two auxiliary `is` verbs).
scenario: "reader trying to understand which parameters govern noise labeling parses the sentence with two `is` verbs → sentence is unparseable, so the intended mapping from `allow_single_cluster` / `cluster_selection_epsilon` to noise-labelling behavior is lost on the reader"
contract: Drop the first `is`: "The determination of some points as noise in large, single-cluster datasets is controlled by …".
instances: single-instance

### F18 — `whats_new/v1.3.rst` uses bare `:class:`DBSCAN`` where surrounding text uses fully-qualified names
severity: low
evidence: doc/whats_new/v1.3.rst:194 "Similarly to :class:`cluster.OPTICS`, it can be seen as a generalization of :class:`DBSCAN` by allowing…" — `DBSCAN` is the only unqualified `:class:` reference in the entry; both `cluster.OPTICS` (line 194) and `cluster.HDBSCAN` (line 192) use the `cluster.` prefix.
scenario: "sphinx may fail to resolve `:class:`DBSCAN`` unambiguously depending on intersphinx config → nitpicky warning at build time; the anchor text in the rendered changelog is inconsistent with the neighbors"
contract: Change `:class:`DBSCAN`` to `:class:`cluster.DBSCAN`` for consistency and to guarantee resolution.
instances: single-instance

### F19 — `_reachability.pyx` `mutual_reachability_graph` Returns section names the wrong (and misspelled) key
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:76-78 Returns block uses `mututal_reachability_graph:` (typo `mututal_`, and a bare colon rather than the numpydoc `name : type` format used elsewhere in the file).
scenario: "sphinx renders the Returns section without a properly formatted type/description separator → visually inconsistent with siblings, plus a visible misspelling"
contract: Fix to `mutual_reachability : {ndarray, sparse matrix} of shape (n_samples, n_samples)` — the actual function returns the modified input, whose canonical name is `distance_matrix` or `mutual_reachability`.
instances: single-instance

### F20 — Test file misnamed `test_reachibility.py` (misspelling)
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py — the module and file name are misspelled (`reachibility` → should be `reachability`). Same misspelling of the local variable `mutual_reachibility_distance` in sklearn/cluster/_hdbscan/_reachability.pyx:127, 144, 149, 182, 206, 209, 210.
scenario: "developer searches for `test_reachability` to locate the reachability tests → the ripgrep hit is empty because the file is misspelled `test_reachibility`, and the misspelled variable then surfaces in every git log and stack trace involving the Cython inner loop"
contract: Rename file to `test_reachability.py` and rename `mutual_reachibility_distance` → `mutual_reachability_distance` throughout `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:150, sklearn/cluster/_hdbscan/_reachability.pyx:155, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:188, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/_reachability.pyx:212]

### F21 — Example prose in `plot_hdbscan.py` has awkward `min_cluster_size` / `min_samples` grammar
severity: low
evidence: examples/cluster/plot_hdbscan.py:178 "Clusters smaller than the ones of this size will be left as noise." (redundant "the ones of"); examples/cluster/plot_hdbscan.py:181 "However values which too small will lead to false sub-clusters" (missing `are` — should be "values which are too small"); examples/cluster/plot_hdbscan.py:203 "`min_samples` better be tuned after finding a good value for `min_cluster_size`." (colloquial and grammatically broken).
scenario: "rendered example page uses ungrammatical English in explanatory paragraphs → poor first impression on new users being introduced to the estimator via `sphinx-gallery`"
contract: Rewrite each fragment: `smaller than the ones of this size` → `smaller than this size`; `values which too small` → `values which are too small`; `better be tuned after` → `should be tuned after`.
instances: [examples/cluster/plot_hdbscan.py:178, examples/cluster/plot_hdbscan.py:181, examples/cluster/plot_hdbscan.py:203]

### F22 — `remap_single_linkage_tree` docstring type of `non_finite` disagrees with the only caller
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:370-371 — "non_finite : ndarray / Boolean array of which entries in the raw data are non-finite". The sole caller at line 838 passes `non_finite=set(infinite_index + missing_index)` — a Python `set` of integer indices, and the function body uses `len(non_finite)` and `enumerate(non_finite)` for both `outlier_count` and per-outlier `outlier_tree` row values (line 389-395). A Boolean mask would produce meaningless `outlier` values (0/1) and wrong `outlier_count` semantics.
scenario: "Contributor writing a second call site follows the docstring and passes a Boolean mask → `outlier_count` becomes the mask length (n) rather than the number of non-finite entries, and every `outlier_tree` row gets a `left_node` of 0 or 1 instead of the raw sample index."
contract: Correct the docstring to describe `non_finite` as a set (or iterable) of integer indices into the raw data marking the non-finite samples.
instances: single-instance

### F23 — Systemic doc typos: "reahability", "collecteion", duplicated "tree tree", "simbling", "smaler", "with be called", "homogenous"
severity: low
evidence: Recurring typos in newly-added docstrings and comments:
- "reahability" for "reachability" — sklearn/cluster/_hdbscan/_linkage.pyx:81, 143, 234; sklearn/cluster/_hdbscan/hdbscan.py:108, 149
- "collecteion" for "collection" — sklearn/cluster/_hdbscan/_linkage.pyx:82, 144, 235; sklearn/cluster/_hdbscan/hdbscan.py:109, 150
- "single-linkage tree tree" (duplicated word) — sklearn/cluster/_hdbscan/_linkage.pyx:240; sklearn/cluster/_hdbscan/hdbscan.py:155, 226, 332, 361
- "simbling" for "sibling" — sklearn/cluster/_hdbscan/_tree.pyx:509; sklearn/cluster/tests/test_hdbscan.py:536
- "smaler" for "smaller" — sklearn/cluster/_hdbscan/_tree.pyx:139
- "with be called" for "will be called" — sklearn/cluster/_hdbscan/hdbscan.py:952
- "homogenous" (informal) — sklearn/cluster/_hdbscan/hdbscan.py:912
scenario: "Reviewer runs grep-based doc searches and API surface skimming (`git grep reachability`, `git grep sibling`) → misses these locations, and long-term the misspellings ossify into related identifiers and get copy-pasted."
contract: Correct each typo to its intended spelling ("reachability", "collection", "single-linkage tree (dendrogram)" with the duplication removed, "sibling", "smaller", "will be called", "homogeneous").
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:81, sklearn/cluster/_hdbscan/_linkage.pyx:82, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:143, sklearn/cluster/_hdbscan/_linkage.pyx:144, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/_linkage.pyx:235, sklearn/cluster/_hdbscan/_linkage.pyx:240, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:108, sklearn/cluster/_hdbscan/hdbscan.py:109, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:150, sklearn/cluster/_hdbscan/hdbscan.py:155, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:226, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:332, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/hdbscan.py:912, sklearn/cluster/_hdbscan/hdbscan.py:952, sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:139, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/_tree.pyx:509, sklearn/cluster/tests/test_hdbscan.py:530, sklearn/cluster/tests/test_hdbscan.py:536]
