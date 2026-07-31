### F1 — HDBSCAN `n_jobs` docstring contradicts its actual default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 documents `n_jobs : int, default=None` (`` `None` means 1 unless in a :obj:`joblib.parallel_backend` context ``), but `__init__` at sklearn/cluster/_hdbscan/hdbscan.py:658 sets `n_jobs=4`.
scenario: "user reads docstring, expects default n_jobs=None (single-thread unless under joblib context) → HDBSCAN() silently runs with 4 workers and behaves inconsistently with every other sklearn estimator; users cannot predict resource usage from the API reference"
contract: Documented `default=None` must match the constructor signature; make the docstring say `default=4` (or fix the constructor to `n_jobs=None` if the doc is authoritative). One or the other must move — they cannot disagree.
instances: single-instance

### F2 — User Guide references non-existent parameter `minimum_cluster_size`
severity: medium
evidence: doc/modules/clustering.rst:1057-1060 introduces the hyperparameter as `min_cluster_size` but then twice writes "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples`". The actual estimator parameter (sklearn/cluster/_hdbscan/hdbscan.py:429, 617, 649) is `min_cluster_size`; no `minimum_cluster_size` exists in the API.
scenario: "user copies `HDBSCAN(minimum_cluster_size=5)` from the User Guide → TypeError: unexpected keyword argument `minimum_cluster_size`, or (worse) the argument silently degrades to `**kwargs` behavior in downstream code"
contract: Use the exact parameter name `min_cluster_size` throughout the User Guide; every `minimum_cluster_size` occurrence in the "HDBSCAN can be smoothed…" paragraph must be renamed.
instances: [doc/modules/clustering.rst:1059, doc/modules/clustering.rst:1060]

### F3 — `_hdbscan_prims` docstring documents non-existent `copy` param and omits `algo`, `leaf_size`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature is `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric='euclidean', leaf_size=40, n_jobs=None, **metric_params)`, but the docstring (lines 285-322) documents `X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy`, `metric_params` — `copy` does not exist on this function, and required `algo` / `leaf_size` are undocumented.
scenario: "developer maintaining this file reads the docstring and assumes `_hdbscan_prims` accepts `copy=` (mirroring `_hdbscan_brute`), calls `_hdbscan_prims(..., copy=True)` from a new code path → TypeError; conversely, no docstring guidance exists for the required `algo` positional argument"
contract: Docstring parameter list must be the exact set of the function signature: drop the `copy` block, add `algo` and `leaf_size` sections mirroring the constructor's descriptions.
instances: single-instance

### F4 — Class docstring narration lists non-existent parameter name in error text
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:118-122 error message tells users to `Ensure your distance matrix has non-zero values for at least `min_sample`={min_samples} neighbors for each points` — the parameter is `min_samples` (with the trailing `s`), and grammar reads "for each points" instead of "for each point". [out-of-theme]
scenario: "user hits the ValueError, searches the estimator API for `min_sample` → nothing found, wastes time; the message also has a grammar error visible to end users"
contract: Rename to `min_samples` (the actual parameter) and fix "for each points" → "for each point".
instances: single-instance

### F5 — Stray colons make `:math:` renderings show "ε:" in User Guide prose
severity: low
evidence: doc/modules/clustering.rst:1009-1010 — "by removing any edges with value greater than :math:`\varepsilon`:\nfrom the original graph. Any points whose core distance is less than :math:`\varepsilon`:\nare at this staged marked as noise." The trailing `:` after each closing backtick is outside the math role and renders as a literal colon in the sentence.
scenario: "reader of the rendered docs opens the HDBSCAN section → sees `ε:` mid-sentence followed by `from the original graph`, grammatically nonsensical; the second occurrence combines with the `staged` typo to produce `less than ε: are at this staged marked as noise.`"
contract: Delete the stray `:` after each `:math:`\varepsilon``; fix `at this staged` → `at this stage`.
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011]

### F6 — Example uses a `:ref:` target that does not exist (case mismatch)
severity: low
evidence: examples/cluster/plot_hdbscan.py:104 has `see :ref:`User Guide <HDBSCAN>`` — the only defined label is `.. _hdbscan:` at doc/modules/clustering.rst:971 (lowercase). ReST/Sphinx reference targets are case-sensitive; no `HDBSCAN` label exists in the docs tree.
scenario: "sphinx builds this example → the ref becomes an unresolved reference warning; the rendered example page has broken link text"
contract: Use `:ref:`User Guide <hdbscan>`` matching the declared label exactly.
instances: single-instance

### F7 — `_hdbscan_brute` / `_hdbscan_prims` / `_brute_mst` docstrings claim `min_samples : int, default=None` while code defaults to 5 (or has no default)
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82 (`def _brute_mst(mutual_reachability, min_samples)` — no default) with docstring `min_samples : int, default=None` at line 95; sklearn/cluster/_hdbscan/hdbscan.py:160 (`min_samples=5`) with docstring `default=None` at line 179; sklearn/cluster/_hdbscan/hdbscan.py:272 (`min_samples=5`) with docstring `default=None` at line 290.
scenario: "future maintainer reads the docstring, assumes `_hdbscan_brute()` w/o arg → None then a sentinel path; actually gets 5 with no sentinel handling → silent behavior mismatch"
contract: Match each docstring's default to the actual signature: remove the `default=None` on `_brute_mst`, change to `default=5` for `_hdbscan_brute` and `_hdbscan_prims`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:95, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F8 — `_brute_mst` docstring parameter section names the wrong argument
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:91-93 documents "mututal_reachability_graph: {ndarray, sparse matrix} of shape (n_samples, n_samples) — Weighted adjacency matrix of the mutual reachability graph" but the actual parameter (line 82) is `mutual_reachability`. The docstring name contains a typo (`mututal`) *and* mismatches the signature.
scenario: "user passing kwargs `_brute_mst(mututal_reachability_graph=...)` per the docstring → TypeError; the ':' formatting is also wrong (colon-space is standard, not colon)"
contract: Rename the docstring key to `mutual_reachability` and match the numpydoc `name : type` style used elsewhere in the file.
instances: single-instance

### F9 — Repeated word "single-linkage tree tree" in dendrogram docstrings and User Guide
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:149 "The single-linkage tree tree (dendrogram)"; identical text at 220, 326, 361; sklearn/cluster/_hdbscan/_linkage.pyx:234 "The single-linkage tree tree (dendrogram)".
scenario: "docs consumer reads the return-type description across multiple functions, sees the doubled word every time → looks like copy-paste noise, weakens trust in the docs"
contract: Remove the duplicated "tree" — should read "The single-linkage tree (dendrogram)".
instances: [sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361, sklearn/cluster/_hdbscan/_linkage.pyx:234]

### F10 — Recurring typos "mutual-reahability" and "collecteion" propagated across Cython docstrings
severity: low
evidence: The pair `"The MST representation of the mutual-reahability graph. The MST is represented as a collecteion of edges."` appears verbatim at sklearn/cluster/_hdbscan/hdbscan.py:102-103 and 143-144, and at sklearn/cluster/_hdbscan/_linkage.pyx:75-76, 137-138, 228-229.
scenario: "docs generation surfaces the same two typos in five separate rendered API pages → visible sloppiness for a new user-facing estimator"
contract: `mutual-reahability` → `mutual-reachability`; `collecteion` → `collection`. Fix every occurrence.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229]

### F11 — Typo "simbling" in _do_labelling comment and its corresponding test comment
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:503 "The threshold should be calculated per-sample based on the largest lambda of any simbling node." and sklearn/cluster/tests/test_hdbscan.py:530 "The threshold should be calculated per-sample based on the largest lambda of any simbling node."
scenario: "future reader greps the codebase for `sibling` to trace the labeling invariant → hits neither occurrence because both are misspelled `simbling`, so the misspelling has already propagated from source to tests undetected"
contract: `simbling` → `sibling` in both files.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/tests/test_hdbscan.py:530]

### F12 — `dbscan_clustering` docstring has typo and dropped period
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:927 "particular cut_distance (or epsilon) DBSCAN* can be thought of as" — missing period between `(or epsilon)` and `DBSCAN*`, producing a run-on sentence; sklearn/cluster/_hdbscan/hdbscan.py:946 "Clusters smaller than this value with be called 'noise'" — `with be called` should be `will be called`; sklearn/cluster/_hdbscan/hdbscan.py:937 "and cluster smaller than `min_cluster_size`" — `cluster smaller` should be `clusters smaller` (subject/verb agreement).
scenario: "user reads the public method's docstring and encounters a run-on sentence and a "with be" grammar error → the method is a user-facing API entry point, so this is high-visibility"
contract: Add the missing period after `(or epsilon)`; `with be called` → `will be called`; `and cluster smaller than` → `and clusters smaller than`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:927, sklearn/cluster/_hdbscan/hdbscan.py:937, sklearn/cluster/_hdbscan/hdbscan.py:946]

### F13 — Broken narration in `_weighted_cluster_center` comment obscures the invariant
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:905-906 "Need to handle iteratively seen each cluster may have a different number of samples, hence we can't create a homogenous 3D array." The word `seen` should be `since` (or the sentence is otherwise malformed); "homogenous" is also nonstandard for "homogeneous".
scenario: "reader trying to understand why the loop can't be vectorized reads a nonsensical sentence and cannot recover the invariant (each cluster's mask has variable size) → the whole point of the comment is lost"
contract: `iteratively seen each cluster` → `iteratively since each cluster`; `homogenous` → `homogeneous`.
instances: single-instance

### F14 — `_do_labelling` docstring says "The determination of some points as noise is in large, single-cluster datasets is controlled by" — duplicated "is"
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:441-443 "The determination of some points as noise is in large, single-cluster datasets is controlled by the `allow_single_cluster` and `cluster_selection_epsilon` parameters." — grammatically broken (two auxiliary `is` verbs).
scenario: "reader trying to understand which parameters govern noise labeling parses the sentence with two `is` verbs → sentence is unparseable, so the intended mapping from `allow_single_cluster` / `cluster_selection_epsilon` to noise-labelling behavior is lost on the reader"
contract: Drop the first `is`: "The determination of some points as noise in large, single-cluster datasets is controlled by …".
instances: single-instance

### F15 — `_condense_tree` docstring typo "smaler"
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:133 "Clusters smaler than this are pruned from the tree."
scenario: "sphinx renders this `cpdef` function's docstring into the public API reference → the rendered page ships with a visible misspelling `smaler` for a newly-added user-facing estimator"
contract: `smaler` → `smaller`.
instances: single-instance

### F16 — `whats_new/v1.3.rst` uses bare `:class:`DBSCAN`` where surrounding text uses fully-qualified names
severity: low
evidence: doc/whats_new/v1.3.rst:194 "Similarly to :class:`cluster.OPTICS`, it can be seen as a generalization of :class:`DBSCAN` by allowing…" — `DBSCAN` is the only unqualified `:class:` reference in the entry; both `cluster.OPTICS` (line 194) and `cluster.HDBSCAN` (line 192) use the `cluster.` prefix.
scenario: "sphinx may fail to resolve `:class:`DBSCAN`` unambiguously depending on intersphinx config → nitpicky warning at build time; the anchor text in the rendered changelog is inconsistent with the neighbors"
contract: Change `:class:`DBSCAN`` to `:class:`cluster.DBSCAN`` for consistency and to guarantee resolution.
instances: single-instance

### F17 — Wrong array-shape annotations in `_tree.pyx` docstrings
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 documents `hierarchy : ndarray of shape (n_samples,)` but code at line 146 derives `n_samples = hierarchy.shape[0] + 1`, i.e. `hierarchy` is `(n_samples - 1,)`. Same wrong shape appears at line 138 (condensed_tree), line 371 (linkage in `labelling_at_cut`), lines 447 and 656 (condensed_tree in `_do_labelling` and `_get_clusters`). `condensed_tree` in particular has a variable length that is neither `n_samples` nor `n_samples - 1`.
scenario: "user calling `_condense_tree(hierarchy)` per the docstring passes an `(n_samples,)` array → shape mismatch when the function computes `n_samples + 1`; downstream indexing goes out of bounds"
contract: Change `(n_samples,)` to `(n_samples - 1,)` for `hierarchy`/`linkage` (single-linkage trees) and to the variable-length description used elsewhere for `condensed_tree` (e.g., "ndarray, dtype=CONDENSED_dtype").
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F18 — `_reachability.pyx` `mutual_reachability_graph` Returns section names the wrong (and misspelled) key
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:76-78 Returns block uses `mututal_reachability_graph:` (typo `mututal_`, and a bare colon rather than the numpydoc `name : type` format used elsewhere in the file).
scenario: "sphinx renders the Returns section without a properly formatted type/description separator → visually inconsistent with siblings, plus a visible misspelling"
contract: Fix to `mutual_reachability : {ndarray, sparse matrix} of shape (n_samples, n_samples)` — the actual function returns the modified input, whose canonical name is `distance_matrix` or `mutual_reachability`.
instances: single-instance

### F19 — Broken hyphen wrap in User Guide "fully -connected"
severity: low
evidence: doc/modules/clustering.rst:1025-1026 "HDBSCAN first extracts a minimum spanning tree (MST) from the fully\n-connected mutual reachability graph" — a stray `\n-connected` produces `fully -connected` in the rendered HTML (space before hyphen).
scenario: "rendered docs show `fully -connected` mid-sentence with visible spurious hyphen → visual defect"
contract: Reflow so the hyphenated compound stays together: `fully-connected` on one line, or reword as `fully connected` (no hyphen).
instances: single-instance

### F20 — Test file misnamed `test_reachibility.py` (misspelling)
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py — the module and file name are misspelled (`reachibility` → should be `reachability`). Same misspelling of the local variable `mutual_reachibility_distance` in sklearn/cluster/_hdbscan/_reachability.pyx:127, 144, 149, 182, 206, 209, 210.
scenario: "developer searches for `test_reachability` to locate the reachability tests → the ripgrep hit is empty because the file is misspelled `test_reachibility`, and the misspelled variable then surfaces in every git log and stack trace involving the Cython inner loop"
contract: Rename file to `test_reachability.py` and rename `mutual_reachibility_distance` → `mutual_reachability_distance` throughout `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F21 — Example prose in `plot_hdbscan.py` has awkward `min_cluster_size` / `min_samples` grammar
severity: low
evidence: examples/cluster/plot_hdbscan.py:178 "Clusters smaller than the ones of this size will be left as noise." (redundant "the ones of"); examples/cluster/plot_hdbscan.py:181 "However values which too small will lead to false sub-clusters" (missing `are` — should be "values which are too small"); examples/cluster/plot_hdbscan.py:203 "`min_samples` better be tuned after finding a good value for `min_cluster_size`." (colloquial and grammatically broken).
scenario: "rendered example page uses ungrammatical English in explanatory paragraphs → poor first impression on new users being introduced to the estimator via `sphinx-gallery`"
contract: Rewrite each fragment: `smaller than the ones of this size` → `smaller than this size`; `values which too small` → `values which are too small`; `better be tuned after` → `should be tuned after`.
instances: [examples/cluster/plot_hdbscan.py:178, examples/cluster/plot_hdbscan.py:181, examples/cluster/plot_hdbscan.py:203]
