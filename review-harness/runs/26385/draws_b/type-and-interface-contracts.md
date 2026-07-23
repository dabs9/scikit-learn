### F1 — `_weighted_cluster_center` breaks its interface contract with `fit` on non-finite input
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:854-921 — `fit` reaches `self._weighted_cluster_center(X)` at line 855 after `X` has been rebound to the *finite-only* subset (line 733: `X = X[finite_index]`), while `self.labels_` has been reassigned to a `new_labels` array of size `self._raw_data.shape[0]` (line 840: `new_labels = np.empty(self._raw_data.shape[0], dtype=np.int32)`; line 844: `self.labels_ = new_labels`). Inside the helper, line 908 forms `mask = self.labels_ == idx` (length `n_raw`) and immediately indexes `X` (length `n_finite`): `data = X[mask]`. Additionally line 895 uses `n_clusters = len(set(self.labels_) - {-1, -2})`, omitting the `-3` label declared in `_OUTLIER_ENCODING["missing"]["label"]` (line 74) and documented in the class attribute docstring at lines 534-537 ("Samples with missing data are given the label -3").
scenario: "User fits `HDBSCAN(store_centers=...)` on dense feature data containing `np.nan` → boolean-mask length mismatch raises `IndexError` at `X[mask]`; if the mismatch is fixed, `-3` outliers are still counted as a cluster so an over-sized `range(n_clusters)` reaches an empty group and `np.average(empty, weights=empty, axis=0)` raises `ZeroDivisionError`"
contract: Compute centers on the same `n_raw` frame the labels live in — pass the *raw* `X` (or reindex `self.labels_[finite_index]`) and exclude the full outlier set `{-1, -2, -3}` when counting non-noise clusters.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:855, sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/_hdbscan/hdbscan.py:908-909]

### F2 — `n_jobs` constructor default contradicts its documented value
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — the class docstring documents `n_jobs : int, default=None` ("`None` means 1 unless in a joblib.parallel_backend context"). sklearn/cluster/_hdbscan/hdbscan.py:658 — the `__init__` signature declares `n_jobs=4`, and no other layer reconciles the two.
scenario: "User relies on the documented `default=None` (single-threaded / joblib-context aware) → `HDBSCAN()` silently runs `pairwise_distances`/`NearestNeighbors` with `n_jobs=4`, launching four worker processes/threads inside code the user believed was single-threaded (e.g. within a `joblib.parallel_backend` block, causing over-subscription)"
contract: The `__init__` default must match the documented default — set `n_jobs=None` in the signature.
instances: single-instance

### F3 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist in its signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature is `_hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)`. sklearn/cluster/_hdbscan/hdbscan.py:313-318 — docstring block documents `copy : bool, default=False` with a full description; any positional/keyword caller trusting the docstring would either fail (unknown kwarg swallowed by `**metric_params`) or corrupt the metric_params dict.
scenario: "Downstream code (or a future maintainer) reads the docstring and calls `_hdbscan_prims(X, ..., copy=True)` → `copy` is silently packed into `**metric_params` and forwarded to `NearestNeighbors(metric_params=...)`, producing a confusing distance-metric error rather than a copy control"
contract: Remove the `copy` block from `_hdbscan_prims`'s docstring; it is inherited from `_hdbscan_brute` but has no support here.
instances: single-instance

### F4 — `_hdbscan_brute` declares `alpha=None` while its docstring promises `default=1.0`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166 — signature is `_hdbscan_brute(X, min_samples=5, alpha=None, metric="euclidean", n_jobs=None, copy=False, **metric_params)`. sklearn/cluster/_hdbscan/hdbscan.py:183-184 — docstring: `alpha : float, default=1.0`. sklearn/cluster/_hdbscan/hdbscan.py:241 — `distance_matrix /= alpha` — invoking with the documented default triggers `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`.
scenario: "A downstream caller (or unit test) invokes `_hdbscan_brute(X)` relying on the documented `default=1.0` → immediate `TypeError` at the `distance_matrix /= alpha` line"
contract: Set `alpha=1.0` in the signature to match the docstring and the sibling `_hdbscan_prims` (line 273).
instances: single-instance

### F5 — `remap_single_linkage_tree` type annotation for `non_finite` contradicts what the caller passes
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:365-366 — docstring: `non_finite : ndarray - Boolean array of which entries in the raw data are non-finite`. sklearn/cluster/_hdbscan/hdbscan.py:836-839 — caller invokes with `non_finite=set(infinite_index + missing_index)` (a Python `set` of *integer indices*). The function's body at lines 383-391 treats `non_finite` as an iterable of integer positions (`for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)`), so the docstring's "Boolean array" contract is not what the function accepts.
scenario: "A new caller reads the docstring and passes a boolean mask → each iteration `outlier_tree[i] = (outlier, ...)` writes a `bool` into an `intp` structured field; either raises or silently produces a mis-encoded tree with only 0/1 outlier ids"
contract: Update the docstring to describe `non_finite` as an iterable of integer indices (matching the actual usage), and type-check accordingly at the boundary.
instances: single-instance
