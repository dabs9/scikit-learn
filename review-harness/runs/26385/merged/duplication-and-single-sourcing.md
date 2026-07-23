### F1 — `_weighted_cluster_center` hardcodes `{-1, -2}` and omits the `-3` (missing) outlier label from `_OUTLIER_ENCODING`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` hardcodes only two of the three outlier labels, while the class-level `_OUTLIER_ENCODING` (lines 65-79) defines both `"infinite": -2` and `"missing": -3`, and the docstring at lines 562 and 573 explicitly states "the `-1, -2, -3` labels for the outlier clusters are excluded".
scenario: "User fits `HDBSCAN(store_centers='centroid')` on non-precomputed data containing `np.nan` samples (which receive label `-3` at line 843) → `set(labels_) - {-1, -2}` retains `-3`, inflating `n_clusters` by one; the extra loop iteration `for idx in range(n_clusters)` at line 907 selects `mask = self.labels_ == n_clusters-1`, which matches no real samples, so `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError: Weights sum to zero`."
contract: Compute `n_clusters` by subtracting all encoded outlier labels sourced from `_OUTLIER_ENCODING` (plus `-1` for noise), e.g. `set(self.labels_) - ({-1} | {v["label"] for v in _OUTLIER_ENCODING.values()})`, so any future addition to `_OUTLIER_ENCODING` propagates automatically.
instances: single-instance

### F2 — `_hdbscan_brute` signature `alpha=None` contradicts docstring default and is unsafe on the in-place `distance_matrix /= alpha` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None` in the signature, while the docstring at line 183-184 states `alpha : float, default=1.0`, and line 241 unconditionally executes `distance_matrix /= alpha`. Every current caller passes `self.alpha` (a validated `float`), so the default is dead — but the drift between signature, docstring, and the in-place division means any direct caller relying on the documented default hits `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`.
scenario: "user or downstream code calls `_hdbscan_brute(X)` relying on the documented `alpha=1.0` default → immediate `TypeError` at `distance_matrix /= alpha`"
contract: Change the signature to `alpha=1.0` so it matches the docstring and the invariant assumed at line 241.
instances: single-instance

### F3 — Duplicate `births` allocation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 and :254 — two consecutive identical statements `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` with no intervening code (line 253 is blank).
scenario: "Reader/refactorer of `_compute_stability` → wastes an allocation and signals uncertainty about which assignment is authoritative; a future edit to one line without the other would silently produce inconsistent state."
contract: Delete the second identical assignment at line 254 so `births` is allocated exactly once.
instances: single-instance

### F4 — `_hdbscan_prims` docstring copy-pasted from `_hdbscan_brute` documents nonexistent `copy` parameter [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents a `copy : bool, default=False` parameter for `_hdbscan_prims`, but the function signature at lines 269-278 (`X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params`) does not accept `copy`; the docstring is a verbatim copy of the corresponding `_hdbscan_brute` block at lines 207-212. The signature parameters `algo` and `leaf_size` are also undocumented for the same reason.
scenario: "User reads the internal docstring and expects to be able to pass `copy=True` to `_hdbscan_prims` (or via `HDBSCAN(algorithm='kdtree', copy=True)`) → the parameter is silently ignored for prims paths, since `_hdbscan_prims` never receives `copy`."
contract: Remove the `copy` parameter description from `_hdbscan_prims`'s docstring and add descriptions for the `algo` and `leaf_size` parameters that actually exist in the signature.
instances: single-instance

### F5 — `PyArray_SHAPE` extern declaration duplicated between `_tree.pxd` and `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`, and sklearn/cluster/_hdbscan/_linkage.pyx:44-45 declares an identical block; `_linkage.pyx:40` already `cimport`s `HIERARCHY_t` from `_tree`, so the extern in the pxd is reachable via cimport rather than re-declaration.
scenario: "Signature of `PyArray_SHAPE` (or the numpy header path) is updated in one file but not the other → build breakage or, worse, ABI drift between the two translation units."
contract: Keep the `PyArray_SHAPE` extern declaration in `_tree.pxd` only, and have `_linkage.pyx` obtain it via `from ...cluster._hdbscan._tree cimport HIERARCHY_t, PyArray_SHAPE` (deleting lines 44-45).
instances: [sklearn/cluster/_hdbscan/_tree.pxd:48-49, sklearn/cluster/_hdbscan/_linkage.pyx:44-45]

### F6 — `fit` comment states wrong outlier labels, drifting from `_OUTLIER_ENCODING`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The authoritative dict `_OUTLIER_ENCODING` (hdbscan.py:65-79) and the code two lines below (`new_labels[infinite_index] = _OUTLIER_ENCODING["infinite"]["label"]` = -2; `new_labels[missing_index] = _OUTLIER_ENCODING["missing"]["label"]` = -3) map `np.inf` → `-2` and `np.nan` → `-3`.
scenario: "reader trusts the comment while debugging non-finite handling → mis-diagnoses observed `-2`/`-3` labels as bugs, or wires new callers against the wrong contract"
contract: Rewrite the comment to match `_OUTLIER_ENCODING` (`np.inf` → `-2`, `np.nan` → `-3`), or delete it and refer to `_OUTLIER_ENCODING`.
instances: single-instance
