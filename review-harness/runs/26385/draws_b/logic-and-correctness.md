### F1 — `_weighted_cluster_center` uses truncated `X` against raw-shape `labels_`, producing shape mismatch when data has non-finite rows

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-909 — line 733 reassigns `X = X[finite_index]` (finite subset only); line 844 reassigns `self.labels_ = new_labels` sized to `self._raw_data.shape[0]`; line 855 then calls `self._weighted_cluster_center(X)` with the truncated `X`; inside, line 908 does `mask = self.labels_ == idx` (raw shape) and line 909 does `data = X[mask]` where X has fewer rows than the mask.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X_with_nan_or_inf)` → `IndexError: boolean index did not match indexed array along dimension 0` in `_weighted_cluster_center`, killing fit"
contract: When non-finite rows are present, `_weighted_cluster_center` must be called with the raw-shape feature array (`self._raw_data`) so the mask derived from `self.labels_` aligns with the indexed rows.
instances: single-instance

### F2 — `_weighted_cluster_center` miscounts `n_clusters` by ignoring the `-3` (missing) outlier label

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. `_OUTLIER_ENCODING["missing"]["label"]` is `-3` (line 74), but `-3` is not excluded from the set. When missing rows exist, `-3` inflates `n_clusters` by 1; the subsequent `for idx in range(n_clusters)` iterates over a non-existent cluster label (e.g. `idx == real_n_clusters`), producing empty `data`/`strength`, which then propagates through `np.average(empty, weights=empty, axis=0)` (raises `ZeroDivisionError`) or `np.argmin(empty)` (raises `ValueError`), and a bogus centroid/medoid row is otherwise stored.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X_with_nan)` → `centroids_` allocated with one row too many, iteration `idx == n_real_clusters` produces an empty slice, causing exception or garbage row in `centroids_`/`medoids_`"
contract: `n_clusters` must exclude every outlier label defined in `_OUTLIER_ENCODING`, i.e. `n_clusters = len(set(self.labels_) - {-1} - {v["label"] for v in _OUTLIER_ENCODING.values()})`.
instances: single-instance

### F3 — `n_jobs=4` default contradicts the estimator's documented default and sklearn convention

severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — signature has `n_jobs=4`. Docstring (lines 486-490) says `n_jobs : int, default=None` and `` `None` means 1 unless in a :obj:`joblib.parallel_backend` context. `-1` means using all processors.``. The default in code will parallelize across 4 workers regardless of the user's joblib context or system.
scenario: "user constructs `HDBSCAN()` expecting single-threaded execution as documented → estimator silently spawns 4 parallel jobs, contradicting the documented contract and departing from the sklearn-wide `n_jobs=None` convention"
contract: The default value of `n_jobs` in the `__init__` signature must be `None`, matching the docstring and sklearn convention.
instances: single-instance

### F4 — `remap_single_linkage_tree` docstring comment says inf→-1 and nan→-2 but code maps inf→-2 and nan→-3

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads `# Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2.` But `_OUTLIER_ENCODING["infinite"]["label"] == -2` (line 67) and `_OUTLIER_ENCODING["missing"]["label"] == -3` (line 74). The correct mapping (inf→-2, nan→-3) is applied at lines 842-843.
scenario: "future maintainer reads the inline comment and reasons/refactors based on the wrong label mapping → introduces a defect matching the incorrect comment"
contract: The inline comment must state that np.inf-bearing samples are mapped to -2 and np.nan-bearing samples are mapped to -3, matching `_OUTLIER_ENCODING`.
instances: single-instance

### F5 — `_hdbscan_brute` signature default `alpha=None` will crash `distance_matrix /= alpha` if the internal contract ever relaxes

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161,241 — signature is `alpha=None`, but line 241 unconditionally executes `distance_matrix /= alpha` (would raise `TypeError: unsupported operand type(s) for /=: '...' and 'NoneType'`). Docstring at line 183 says `alpha : float, default=1.0`. It only works today because every call site passes a validated float via `HDBSCAN.fit`, but the signature is a latent trap.
scenario: "someone imports `_hdbscan_brute` directly (it is not prefixed underscore in any way that blocks reuse across tests, and `test_hdbscan.py` already imports internals) and relies on defaults → immediate crash on `distance_matrix /= None`"
contract: Change signature to `alpha=1.0` (matching docstring) so the default is a usable value.
instances: single-instance

### F6 — `remap_single_linkage_tree` builds `outlier_tree` from an unordered `set`, making outlier ordering non-deterministic

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838,388-391 — call site passes `non_finite=set(infinite_index + missing_index)`; the function then does `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1); last_cluster_id += 1; last_cluster_size += 1`. Because `non_finite` is a Python `set`, iteration order is implementation-defined; the specific `left_node` values in the appended outlier tree entries (and their `cluster_size` labeling) will differ across runs / process starts. `self._single_linkage_tree_` is a public-ish artifact used by `dbscan_clustering`; end-user semantic labels are unaffected because they are re-overwritten from `self.labels_`, but the persisted tree itself is unstable across identical fits.
scenario: "user fits `HDBSCAN` twice on the same non-finite input and compares `_single_linkage_tree_` → the two trees have identical structure but different outlier row ordering"
contract: Pass `non_finite` as an ordered sequence with duplicates removed while preserving first-seen order (e.g. `list(dict.fromkeys(infinite_index + missing_index))`) so `remap_single_linkage_tree`'s output is deterministic.
instances: single-instance
