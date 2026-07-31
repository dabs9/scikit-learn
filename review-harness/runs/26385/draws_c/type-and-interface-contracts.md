### F1 — `_weighted_cluster_center` receives reduced X but full-length labels when non-finite samples exist
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-910 — `X = X[finite_index]` reduces X to (n_finite, n_features); later `self.labels_ = new_labels` is expanded back to (n_raw,); then `self._weighted_cluster_center(X)` is called with the reduced X while `mask = self.labels_ == idx` produces a mask of length n_raw and `data = X[mask]` boolean-indexes X (length n_finite) with that longer mask.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_or_inf)` → `IndexError: boolean index did not match indexed array along dimension 0` from `X[mask]` inside `_weighted_cluster_center`"
contract: `_weighted_cluster_center` must be given data whose leading axis matches `self.labels_`; pass `self._raw_data` (and skip the outlier rows via the mask) rather than the finite-only `X`.
instances: single-instance

### F2 — `n_clusters` excludes only `{-1, -2}`, so label `-3` (missing) is counted as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`, but `_OUTLIER_ENCODING["missing"]["label"] == -3` (lines 74) and hdbscan.py:843 writes `-3` into `self.labels_` for missing samples.
scenario: "Fit with `store_centers` set and data containing `np.nan` → the -3 outlier label is treated as a real cluster id; `centroids_`/`medoids_` are allocated an extra row, and `range(n_clusters)` iterates one iteration too far, producing an empty-slice `np.average` warning/NaN row"
contract: enumerate the outlier labels from `_OUTLIER_ENCODING` (i.e., subtract `{-1} | {out["label"] for out in _OUTLIER_ENCODING.values()}`) so all outlier encodings are excluded.
instances: single-instance

### F3 — `_hdbscan_brute` signature default `alpha=None` contradicts its own contract and would TypeError
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166 declares `alpha=None`, the docstring at line 183 declares `alpha : float, default=1.0`, and line 241 executes `distance_matrix /= alpha`.
scenario: "Any caller relying on the documented default (`_hdbscan_brute(X)` with no `alpha`) → `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'` at line 241"
contract: signature default must match the documented default — set `alpha=1.0` in the `_hdbscan_brute` definition.
instances: single-instance

### F4 — `remap_single_linkage_tree` docstring says `non_finite` is a boolean ndarray but the caller passes a set of integer indices
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-365 docstring: `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`; line 838 caller passes `non_finite=set(infinite_index + missing_index)` (a set of integer row indices); the body at lines 383-389 uses `len(non_finite)` to size `outlier_tree` and treats each iterated element as a node id (`outlier_tree[i] = (outlier, ...)` where `left_node` is `intp`).
scenario: "A future caller follows the documented signature and passes a boolean ndarray of shape (n_raw,) → `outlier_tree` is sized as n_raw instead of the outlier count, and every `left_node` becomes 0/1 booleans cast to intp, silently corrupting the appended outlier subtree"
contract: rewrite the docstring to state that `non_finite` is an iterable of integer indices of non-finite rows (matching the actual call site).
instances: single-instance

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist in the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature has no `copy` argument; sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False` in the Parameters section.
scenario: "A user reading the docstring calls `_hdbscan_prims(..., copy=True)` → `TypeError: _hdbscan_prims() got an unexpected keyword argument 'copy'` because `copy` is not a real parameter"
contract: remove the `copy` entry from the `_hdbscan_prims` Parameters section (it belongs only in `_hdbscan_brute`).
instances: single-instance

### F6 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented default `None` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 `n_jobs=4` in `__init__` signature, while the docstring at lines 486-490 states `n_jobs : int, default=None` and describes `None` semantics.
scenario: "Users relying on the documented default expect `n_jobs=None` (serial unless in a `joblib.parallel_backend`) → the estimator silently runs with `n_jobs=4`, changing parallelism behavior and downstream reproducibility"
contract: set `n_jobs=None` in the `__init__` signature to match the docstring and the sklearn `n_jobs` convention.
instances: single-instance
