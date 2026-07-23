### F1 — store_centers path passes finite-subset X against raw-length labels_

severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:830-855 — inside `fit`, when input contains non-finite rows, line 733 reassigns `X = X[finite_index]` (finite_count rows), lines 840-844 replace `self.labels_` with `new_labels` of length `self._raw_data.shape[0]`, then line 855 invokes `self._weighted_cluster_center(X)` with the still-finite `X`; inside `_weighted_cluster_center` (line 908) `mask = self.labels_ == idx` has raw length, but `data = X[mask]` (line 909) indexes an array with finite_count rows.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data with any `np.nan` or `np.inf` row → numpy raises `IndexError: boolean index did not match indexed array` because mask length ≠ X row count"
contract: `_weighted_cluster_center` must be invoked with data whose row count matches `self.labels_`; pass `self._raw_data` (or realign `self.labels_` and `X` to the same length before calling) so the mask/data shapes agree.
instances: single-instance

### F2 — n_clusters computation treats `-3` (missing) label as a real cluster

severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; `_OUTLIER_ENCODING["missing"]["label"]` is `-3` (hdbscan.py:74), so `set(labels) - {-1, -2}` retains `-3`; the subsequent loop iterates `range(n_clusters)` and computes `X[self.labels_ == idx]` for an idx that never labels a real cluster, producing an empty slice and `np.average(empty, weights=empty)` → `ZeroDivisionError`.
scenario: "missing-data path fixed (F1 resolved) and user requests `store_centers` on data containing `np.nan` → centroid loop overshoots by one, hits empty cluster, raises `ZeroDivisionError` inside `np.average`"
contract: exclude every negative outlier code, not just `-1, -2`; write `n_clusters = len(set(self.labels_) - {out['label'] for out in _OUTLIER_ENCODING.values()} - {-1})` (or equivalent) so the count reflects only non-outlier clusters.
instances: single-instance

### F3 — `HDBSCAN.n_jobs` default of `4` contradicts documented default `None`

severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — class docstring says `n_jobs : int, default=None` with `None means 1 unless in a joblib.parallel_backend context`; sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__(..., n_jobs=4, ...)`.
scenario: "user reads doc / API contract expecting joblib-backend integration by default → construction silently forces four workers, ignoring `joblib.parallel_backend`, contradicting the class contract advertised to callers and to `sklearn.set_config`"
contract: the signature default must match the documented default; set `n_jobs=None` in `__init__` (and rely on the documented `None → 1` semantics) so the API contract holds.
instances: single-instance

### F4 — `_hdbscan_brute` signature default `alpha=None` violates the operation's required-numeric contract

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — signature is `alpha=None`; sklearn/cluster/_hdbscan/hdbscan.py:183-184 — docstring documents `alpha : float, default=1.0`; sklearn/cluster/_hdbscan/hdbscan.py:241 — body executes `distance_matrix /= alpha`, which raises `TypeError: unsupported operand type(s)` when `alpha is None`.
scenario: "any direct call `_hdbscan_brute(X)` (default args) → `TypeError` at `distance_matrix /= alpha`; the signature default advertises an invalid state that the code cannot honor"
contract: match the documented default in the signature: declare `alpha=1.0` in `_hdbscan_brute` so the default is a valid numeric divisor.
instances: single-instance

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter that its signature does not accept [out-of-theme]

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature is `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric='euclidean', leaf_size=40, n_jobs=None, **metric_params)`; sklearn/cluster/_hdbscan/hdbscan.py:313-318 — Parameters section lists `copy : bool, default=False` with a full description; no `copy` argument exists in the signature.
scenario: "downstream reader relies on the docstring, calls `_hdbscan_prims(..., copy=True)` → `copy` is silently swallowed into `**metric_params` and passed to `DistanceMetric.get_metric(metric, **metric_params)`, producing an obscure metric error instead of the documented behavior"
contract: drop the `copy` block from `_hdbscan_prims`'s docstring so the documented parameter list matches the function's real signature.
instances: single-instance

### F6 — `remap_single_linkage_tree` docstring types `non_finite` as a boolean ndarray, callers pass a set of integer indices

severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:362-366 — docstring: `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`; sklearn/cluster/_hdbscan/hdbscan.py:369, 383, 388 — body uses `outlier_count = len(non_finite)`, `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)`, and `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, …)`, requiring `non_finite` to yield integer indices, not booleans; sklearn/cluster/_hdbscan/hdbscan.py:838 — the only caller passes `set(infinite_index + missing_index)`, an iterable of integer indices.
scenario: "future caller reads the docstring and passes an actual boolean ndarray → `enumerate` yields `(0, True/False)` and the True/False values are written into `outlier_tree['left_node']` as `1`/`0`, silently corrupting the returned tree instead of raising"
contract: correct the parameter documentation to `non_finite : set of int — the integer row indices in the raw data that are non-finite` so the declared type matches how the function is used.
instances: single-instance
