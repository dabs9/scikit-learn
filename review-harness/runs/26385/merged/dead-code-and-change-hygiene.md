### F1 — `test_hdbscan_precomputed_non_brute` uses stale `prims_kdtree`/`prims_balltree` algorithm names
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`. The parameter constraint (`sklearn/cluster/_hdbscan/hdbscan.py:629-644`) restricts `algorithm` to `{"auto", "brute", "kdtree", "balltree"}`, so `_validate_params` rejects the `prims_*tree` strings unconditionally. The test therefore passes on a stale error path — `InvalidParameterError` for an unknown algorithm — not on the "precomputed + tree-based" rejection its name/docstring claim to cover.
scenario: "regression removes the real precomputed-vs-tree check in `fit` (lines 785-791) → this test still passes because the `prims_*tree` names always fail param-validation → the precomputed-with-tree path silently regresses undetected"
contract: Change the algorithm strings to the currently-valid `"kdtree"`/`"balltree"` so the test actually exercises the "Sparse data matrices only support algorithm `brute`" precomputed check the docstring describes.
instances: single-instance

### F2 — `_weighted_cluster_center` miscounts `n_clusters` by excluding only {-1, -2}, not -3 [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The class docstring at lines 561-562/572-573 states `n_clusters` for `centroids_`/`medoids_` "only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded." When labels contain -3 (missing), it is counted as a cluster: `range(n_clusters)` then iterates one index too far, producing a mask `self.labels_ == n_clusters` that matches nothing, so the extra row in `centroids_`/`medoids_` is left uninitialized from `np.empty(...)` at lines 901/903.
scenario: "fit non-precomputed data containing `np.nan` rows with `store_centers='centroid'` → `centroids_` shape reports one more cluster than actually exists and its last row contains uninitialized memory"
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` so the -3/missing label is excluded, matching the documented contract.
instances: single-instance

### F3 — "Scale Invariance" HDBSCAN example ignores its own scale factor
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`; `scale` is only used in the plot's parameter label — `X` is neither multiplied by `scale` on `fit` nor on `plot`, unlike the DBSCAN reference block just above which does `dbs.fit(X * scale)` and `plot(X * scale, ...)`.
scenario: "user runs the plot_hdbscan example → the three subplots labelled scale=1/0.5/3 are pixel-identical, so the narrative claim that 'HDBSCAN is scale-invariant' is not actually demonstrated"
contract: Fit and plot on the scaled data — `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, hdb.probabilities_, ...)` — matching the DBSCAN block that motivates the comparison.
instances: single-instance

### F4 — `TreeUnionFind.is_component` is written but never read
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:329-355 — `cdef cnp.uint8_t[::1] is_component` is declared, initialized to all-ones in `__init__` (line 337), and cleared to `False` inside `find` (line 355), but no reader (in `_tree.pyx` or anywhere else in the tree) queries `is_component`. `Grep is_component` in `/tmp/rh-wt-26385/sklearn` returns only these three write sites.
scenario: "each `find` call → extra memory-view store to `is_component` that no consumer reads → wasted allocation and write bandwidth in every union-find operation on the condensed tree"
contract: Remove the `is_component` attribute (declaration, allocation in `__init__`, and the `self.is_component[x] = False` write in `find`) entirely.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:330, sklearn/cluster/_hdbscan/_tree.pyx:337, sklearn/cluster/_hdbscan/_tree.pyx:355]

### F5 — `births` allocated twice in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears on line 252, and the very next executable statement on line 254 is the identical `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`. The first assignment is unconditionally overwritten before the array is ever read.
scenario: "every `_compute_stability` invocation → an extra full-array `np.full(...)` allocation whose result is immediately discarded"
contract: Delete the first `births = np.full(...)` line at line 252.
instances: single-instance

### F6 — Preallocated `mask` in `_weighted_cluster_center` is dead
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)`. Inside the loop at line 908 it is reassigned unconditionally with `mask = self.labels_ == idx`, so the empty allocation is never observed.
scenario: "each `fit` with `store_centers` set → one extra unused boolean array allocation of size `n_samples`"
contract: Remove the `mask = np.empty(...)` line entirely.
instances: single-instance

### F7 — `_hdbscan_prims` docstring documents `copy` and `precomputed` behavior it does not implement
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-318 — the docstring includes a `copy : bool, default=False` section (lines 313-318) that says "Currently, it only applies when `metric="precomputed"`", but the signature at lines 269-278 has no `copy` parameter and the function body handles only feature arrays (never `metric="precomputed"`). The docstring is a stale copy of `_hdbscan_brute`'s docstring.
scenario: "user reads `_hdbscan_prims` docstring → believes `copy=` and precomputed-metric semantics apply to Prims path → they don't"
contract: Delete the `copy : bool ...` block and the misleading "If `metric='precomputed'` then `X` must be a symmetric array of distances" paragraph from `_hdbscan_prims`'s docstring.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:280-283, sklearn/cluster/_hdbscan/hdbscan.py:313-318]

### F8 — `_get_clusters` docstring lists a `stabilities` return value that is never returned
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:686-696 — the Returns block documents `labels`, `probabilities`, and `stabilities : ndarray (n_clusters,)`, but the actual return statement at line 803 is `return (labels, probs)`; no `stabilities` array is computed or returned.
scenario: "caller reads `_get_clusters` docstring → expects a 3-tuple with cluster stabilities → gets a 2-tuple and their code breaks with a ValueError on unpack"
contract: Remove the `stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster.` entry from the Returns section.
instances: single-instance

### F9 — Stale outlier-encoding comment in `HDBSCAN.fit`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." Immediately below, lines 842-843 assign `new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` (= -2) and `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` (= -3), matching the class docstring at lines 534-537. The comment is leftover from the pre-`_OUTLIER_ENCODING` labeling scheme.
scenario: "future maintainer trusts the comment while touching this remap block → introduces off-by-one label shifts or breaks the -3/missing convention the public API advertises"
contract: Update the comment to "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3." (matching `_OUTLIER_ENCODING`).
instances: single-instance

### F10 — Unrelated whitespace-only fixes bundled in `clustering.rst`
severity: low
evidence: doc/modules/clustering.rst:23-25, doc/modules/clustering.rst:34-35, doc/modules/clustering.rst:42-43 — hunk diff shows trailing-whitespace edits in unrelated Mean Shift paragraphs (the "hill climbing" paragraph, the "In general, the equation for :math:`m`…" paragraph, and "In our implementation, :math:`K(x)`…"); these paragraphs are not part of the HDBSCAN feature and are not called out in the PR description.
scenario: "reviewer diffing this PR for HDBSCAN scope → sees unrelated whitespace churn intermixed with feature docs, inflating the diff and coupling unrelated concerns"
contract: Keep whitespace-only fixes to unrelated Mean Shift documentation out of this PR; revert those hunks or split them into a separate cleanup commit.
instances: [doc/modules/clustering.rst:23, doc/modules/clustering.rst:24, doc/modules/clustering.rst:34, doc/modules/clustering.rst:42, doc/modules/clustering.rst:43]

### F11 — Misspelled new test filename `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename spells "reachibility"; the module under test is `_reachability.py` and the docstrings/tests spell it "reachability".
scenario: "developer greps for `test_reachability` → misses this file; future rename creates git history churn"
contract: Rename the new test file to `test_reachability.py` to match the module it tests.
instances: single-instance

### F12 — Misspelled internal identifier `mutual_reachibility_distance` in new Cython code
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:127,144,149,182,206,209,210 — the local/temp variable is spelled `mutual_reachibility_distance`; every surrounding function, docstring, and public API uses the correct spelling `reachability`.
scenario: "future maintainer greps for `mutual_reachability_distance` inside the module → misses these occurrences due to the typo"
contract: Rename the local to `mutual_reachability_distance` (correct spelling) throughout `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F13 — Documented `min_samples` default mismatches signature in `_brute_mst`/`_hdbscan_brute`/`_hdbscan_prims` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:100-103, 185-187, 290-292 — three docstrings state `min_samples : int, default=None` while the corresponding signatures define `min_samples=5` (e.g. `_hdbscan_brute` at line 158, `_hdbscan_prims` at line 272) or `min_samples` as a required kwarg with no default (`_brute_mst` at line 82).
scenario: "reader consulting docs for these helpers → believes `min_samples` defaults to `None`; passing `None` would then be forwarded to `NearestNeighbors(n_neighbors=None)` or used in an f-string error message"
contract: Fix the docstrings to state `default=5` where the signature has `min_samples=5` and remove the `default=None` claim where `min_samples` has no signature default.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:100, sklearn/cluster/_hdbscan/hdbscan.py:185, sklearn/cluster/_hdbscan/hdbscan.py:290]
