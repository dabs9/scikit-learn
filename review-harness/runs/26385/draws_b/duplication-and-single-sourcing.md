### F1 — Duplicated `births = np.full(...)` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is written on line 252 and re-executed verbatim on line 254 (with only a blank line between). The intermediate value is never read.
scenario: "every call to `_compute_stability` → allocates and fills the `births` array twice, wasting O(largest_child) work and adding cognitive noise for future readers"
contract: Delete the second `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` on line 254 (keep exactly one allocation).
instances: single-instance

### F2 — `_weighted_cluster_center` hard-codes `{-1, -2}` and drifts from `_OUTLIER_ENCODING`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. The single source of truth `_OUTLIER_ENCODING` at hdbscan.py:65-79 defines `"infinite"` → label `-2` and `"missing"` → label `-3`. `test_hdbscan.py:37` correctly derives its outlier set from `_OUTLIER_ENCODING` (`OUTLIER_SET = {-1} | {out["label"] for _, out in _OUTLIER_ENCODING.items()}`); this helper does not. Label `-3` is therefore treated as a real cluster.
scenario: "fit `X` that contains rows with `np.nan` (label `-3`) with `store_centers` set → `n_clusters` is inflated by 1; the loop iterates `range(n_clusters)` up to an index no sample actually has, and the empty-mask branch computes `np.average` over an empty slice (RuntimeWarning / NaN row appended to `self.centroids_`/`self.medoids_`)"
contract: Compute the outlier set from `_OUTLIER_ENCODING` at the single source, e.g. `n_clusters = len(set(self.labels_) - {-1} - {v["label"] for v in _OUTLIER_ENCODING.values()})`.
instances: single-instance

### F3 — `fit` comment states wrong outlier labels, drifting from `_OUTLIER_ENCODING`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The authoritative dict `_OUTLIER_ENCODING` (hdbscan.py:65-79) and the code two lines below (`new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` = -2; `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` = -3) map `np.inf` → `-2` and `np.nan` → `-3`.
scenario: "reader trusts the comment while debugging non-finite handling → mis-diagnoses observed `-2`/`-3` labels as bugs, or wires new callers against the wrong contract"
contract: Rewrite the comment to match `_OUTLIER_ENCODING` (`np.inf` → `-2`, `np.nan` → `-3`), or delete it and refer to `_OUTLIER_ENCODING`.
instances: single-instance

### F4 — `_hdbscan_brute` signature `alpha=None` contradicts docstring default and is unsafe on the in-place `distance_matrix /= alpha` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None` in the signature, while the docstring at line 183-184 states `alpha : float, default=1.0`, and line 241 unconditionally executes `distance_matrix /= alpha`. Every current caller passes `self.alpha` (a validated `float`), so the default is dead — but the drift between signature, docstring, and the in-place division means any direct caller relying on the documented default hits `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`.
scenario: "user or downstream code calls `_hdbscan_brute(X)` relying on the documented `alpha=1.0` default → immediate `TypeError` at `distance_matrix /= alpha`"
contract: Change the signature to `alpha=1.0` so it matches the docstring and the invariant assumed at line 241.
instances: single-instance
