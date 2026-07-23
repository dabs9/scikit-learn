Now I have enough context to write findings.

### F1 — `_weighted_cluster_center` receives reduced X but full-length labels_ on non-finite input
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-909 — line 733 replaces `X = X[finite_index]` so it has shape `(n_finite, n_features)`; line 844 assigns `self.labels_ = new_labels` with shape `(self._raw_data.shape[0],) = (n_raw,)`; line 855 calls `self._weighted_cluster_center(X)` with the reduced X; inside, `mask = self.labels_ == idx` (line 908) has length `n_raw`, and `data = X[mask]` (line 909) indexes an array of length `n_finite` with a mask of length `n_raw`.
scenario: "Fit HDBSCAN on a feature array containing `np.nan` and/or `np.inf` with `store_centers` set to `centroid`/`medoid`/`both` → `X[mask]` triggers `IndexError: boolean index did not match indexed array along dimension 0` (or silently produces wrong slices when sizes coincidentally match) → `fit` raises for otherwise valid input."
contract: In the non-precomputed, `not all_finite` branch, call `self._weighted_cluster_center(self._raw_data)` (or build centers from the finite slice using the finite-only labels/probabilities before they are expanded), so the sample axis matches `self.labels_`.
instances: single-instance

### F2 — `_weighted_cluster_center` excludes only `{-1, -2}` from the cluster count, ignoring the `-3` (missing) outlier label
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895,907-920 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the class docstring at lines 561-573 and 869-870 documents `-1, -2, -3` as the outlier labels for centroid/medoid tallies.
scenario: "Fit with non-finite data such that any sample gets label `-3` (missing) and `store_centers` set → `n_clusters` is overcounted by one → the final `for idx in range(n_clusters)` iteration sees a mask that matches no rows → `np.average` raises `ZeroDivisionError: Weights sum to zero, can't be normalized` (or `medoids_[idx]` is populated from empty data)."
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` to match the documented set of outlier labels.
instances: single-instance

### F3 — `n_jobs` default in `HDBSCAN.__init__` is `4` but class docstring documents `default=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490,658 — docstring: "n_jobs : int, default=None" with "`None` means 1 unless in a :obj:`joblib.parallel_backend` context"; signature: `n_jobs=4`.
scenario: "User instantiates `HDBSCAN()` expecting the documented single-thread default → silently gets 4 worker threads (visible in performance, oversubscription in outer parallel contexts, non-reproducible timings)."
contract: Set `n_jobs=None` in `__init__` to align with the documented default and the sklearn convention.
instances: single-instance

### F4 — Comment describes the wrong outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." Actual code at lines 842-843 maps `infinite_index` to `_OUTLIER_ENCODING["infinite"]["label"]` (`-2`) and `missing_index` to `_OUTLIER_ENCODING["missing"]["label"]` (`-3`).
scenario: "Reader relies on the comment when reasoning about label semantics → forms a wrong mental model of the encoding used elsewhere in the file (dbscan_clustering, remapping, tests)."
contract: Update the comment to state "Samples with `np.inf` are mapped to `-2` and those with `np.nan` are mapped to `-3`."
instances: single-instance

### F5 — `_hdbscan_brute` uses `alpha=None` default while dividing by `alpha`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161,183-184,241 — signature `alpha=None`; docstring says "alpha : float, default=1.0"; line 241 `distance_matrix /= alpha` raises `TypeError` if `alpha is None`.
scenario: "Anyone calling `_hdbscan_brute` (private helper, but importable) with the documented/signature default → `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'` on line 241."
contract: Change the signature to `alpha=1.0` to match the docstring and avoid a divide-by-None on the documented default.
instances: single-instance

### F6 — `remap_single_linkage_tree` iterates a `set`, producing non-deterministic tree contents
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:838,388-391 — caller passes `non_finite=set(infinite_index + missing_index)`; `remap_single_linkage_tree` does `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)`. Iteration order of a Python `set` is not stable across processes (PYTHONHASHSEED), so the appended outlier rows of `self._single_linkage_tree_` vary between otherwise identical runs.
scenario: "Two runs on the same data with non-finite samples → identical `labels_`/`probabilities_` but different `self._single_linkage_tree_` contents → any downstream user (persistence, hashing, plotting, custom consumers of `_single_linkage_tree_`) sees non-reproducible internal state."
contract: Pass `non_finite` as a deterministically-ordered sequence (e.g. `np.unique(np.concatenate([infinite_index, missing_index]))`) and iterate that ordered structure in `remap_single_linkage_tree`.
instances: single-instance
