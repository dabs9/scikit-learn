### F1 — `TreeUnionFind.is_component` is written but never read
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:329-355 — `cdef cnp.uint8_t[::1] is_component` is declared, initialized to all-ones in `__init__` (line 337), and cleared to `False` inside `find` (line 355), but no reader (in `_tree.pyx` or anywhere else in the tree) queries `is_component`. `Grep is_component` in `/tmp/rh-wt-26385/sklearn` returns only these three write sites.
scenario: "each `find` call → extra memory-view store to `is_component` that no consumer reads → wasted allocation and write bandwidth in every union-find operation on the condensed tree"
contract: Remove the `is_component` attribute (declaration, allocation in `__init__`, and the `self.is_component[x] = False` write in `find`) entirely.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:330, sklearn/cluster/_hdbscan/_tree.pyx:337, sklearn/cluster/_hdbscan/_tree.pyx:355]

### F2 — `births` allocated twice in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears on line 252, and the very next executable statement on line 254 is the identical `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`. The first assignment is unconditionally overwritten before the array is ever read.
scenario: "every `_compute_stability` invocation → an extra full-array `np.full(...)` allocation whose result is immediately discarded"
contract: Delete the first `births = np.full(...)` line at line 252.
instances: single-instance

### F3 — Preallocated `mask` in `_weighted_cluster_center` is dead
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)`. Inside the loop at line 908 it is reassigned unconditionally with `mask = self.labels_ == idx`, so the empty allocation is never observed.
scenario: "each `fit` with `store_centers` set → one extra unused boolean array allocation of size `n_samples`"
contract: Remove the `mask = np.empty(...)` line entirely.
instances: single-instance

### F4 — `_hdbscan_prims` docstring documents `copy` and `precomputed` behavior it does not implement
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:279-318 — the docstring includes a `copy : bool, default=False` section (lines 313-318) that says "Currently, it only applies when `metric="precomputed"`", but the signature at lines 269-278 has no `copy` parameter and the function body handles only feature arrays (never `metric="precomputed"`). The docstring is a stale copy of `_hdbscan_brute`'s docstring.
scenario: "user reads `_hdbscan_prims` docstring → believes `copy=` and precomputed-metric semantics apply to Prims path → they don't"
contract: Delete the `copy : bool ...` block and the misleading "If `metric='precomputed'` then `X` must be a symmetric array of distances" paragraph from `_hdbscan_prims`'s docstring.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:280-283, sklearn/cluster/_hdbscan/hdbscan.py:313-318]

### F5 — `_get_clusters` docstring lists a `stabilities` return value that is never returned
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:686-696 — the Returns block documents `labels`, `probabilities`, and `stabilities : ndarray (n_clusters,)`, but the actual return statement at line 803 is `return (labels, probs)`; no `stabilities` array is computed or returned.
scenario: "caller reads `_get_clusters` docstring → expects a 3-tuple with cluster stabilities → gets a 2-tuple and their code breaks with a ValueError on unpack"
contract: Remove the `stabilities : ndarray (n_clusters,) — The cluster coherence strengths of each cluster.` entry from the Returns section.
instances: single-instance

### F6 — `test_hdbscan_precomputed_non_brute` uses stale `prims_kdtree`/`prims_balltree` algorithm names
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`. The parameter constraint (`sklearn/cluster/_hdbscan/hdbscan.py:629-644`) restricts `algorithm` to `{"auto", "brute", "kdtree", "balltree"}`, so `_validate_params` rejects the `prims_*tree` strings unconditionally. The test therefore passes on a stale error path — `InvalidParameterError` for an unknown algorithm — not on the "precomputed + tree-based" rejection its name/docstring claim to cover.
scenario: "regression removes the real precomputed-vs-tree check in `fit` (lines 785-791) → this test still passes because the `prims_*tree` names always fail param-validation → the precomputed-with-tree path silently regresses undetected"
contract: Change the algorithm strings to the currently-valid `"kdtree"`/`"balltree"` so the test actually exercises the "Sparse data matrices only support algorithm `brute`" precomputed check the docstring describes.
instances: single-instance

### F7 — Stale outlier-encoding comment in `HDBSCAN.fit`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." Immediately below, lines 842-843 assign `new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` (= -2) and `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` (= -3), matching the class docstring at lines 534-537. The comment is leftover from the pre-`_OUTLIER_ENCODING` labeling scheme.
scenario: "future maintainer trusts the comment while touching this remap block → introduces off-by-one label shifts or breaks the -3/missing convention the public API advertises"
contract: Update the comment to "Samples with np.inf are mapped to -2 and those with np.nan are mapped to -3." (matching `_OUTLIER_ENCODING`).
instances: single-instance

### F8 — `_weighted_cluster_center` miscounts `n_clusters` by excluding only {-1, -2}, not -3 [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The class docstring at lines 561-562/572-573 states `n_clusters` for `centroids_`/`medoids_` "only counts non-outlier clusters. That is to say, the `-1, -2, -3` labels for the outlier clusters are excluded." When labels contain -3 (missing), it is counted as a cluster: `range(n_clusters)` then iterates one index too far, producing a mask `self.labels_ == n_clusters` that matches nothing, so the extra row in `centroids_`/`medoids_` is left uninitialized from `np.empty(...)` at lines 901/903.
scenario: "fit non-precomputed data containing `np.nan` rows with `store_centers='centroid'` → `centroids_` shape reports one more cluster than actually exists and its last row contains uninitialized memory"
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` so the -3/missing label is excluded, matching the documented contract.
instances: single-instance
