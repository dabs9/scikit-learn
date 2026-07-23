### F1 — `n_jobs` default lies: docstring says `None`, code sets `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the `None` semantics ("`None` means 1 unless in a :obj:`joblib.parallel_backend` context"), but sklearn/cluster/_hdbscan/hdbscan.py:658 declares `n_jobs=4` in `HDBSCAN.__init__`.
scenario: "User reads the class docstring (or introspects the signature help) → expects default = 1 core (per `None`/joblib semantics) → instantiates `HDBSCAN()` and unexpectedly gets 4 parallel workers, contradicting both the documented default and sklearn's project-wide `n_jobs=None` convention."
contract: Change `__init__` default to `n_jobs=None` so signature, docstring and library-wide convention agree.
instances: single-instance

### F2 — `remap_single_linkage_tree` docstring type is a lie (and set iteration is non-deterministic)
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 declares `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`, but the sole caller at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` — a `set` of integer indices — and the body iterates with `enumerate(non_finite)` (line 388) so the value written into `outlier_tree[i][0]` is an integer index, not a boolean.
scenario: "Someone maintaining or extending this helper reads the docstring, believes `non_finite` is a boolean mask and either passes one directly (writing `True`/`False` into `left_node`) or refactors the loop under that false assumption → silent index corruption in the reconstructed tree; separately, because a `set` is iterated, the order in which outliers appear in `outlier_tree` is non-deterministic across runs."
contract: Update the docstring to state `non_finite : sequence of int  Indices of non-finite rows in the raw data`, and change the caller (line 838) to pass a deterministically ordered container (e.g. `sorted(set(infinite_index + missing_index))`).
instances: single-instance

### F3 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 contains a full `copy : bool, default=False ...` block, but the function signature at lines 269-278 has no `copy` argument, and `_hdbscan_prims` is called at line 820 (via `mst_func(**kwargs)`) with `kwargs` that never contain `copy` (only the `_hdbscan_brute` branches inject it — lines 795, 808).
scenario: "A user reads the Prim's docstring, believes they can influence in-place behavior for KDTree/BallTree paths → passes `copy=True` and gets `TypeError: unexpected keyword argument`, or trusts that some in-place safeguard applies to the Prim's path when none does."
contract: Delete the `copy` parameter block from `_hdbscan_prims`'s docstring.
instances: single-instance

### F4 — `_hdbscan_brute` signature default `alpha=None` contradicts docstring default `1.0`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166 declares `def _hdbscan_brute(X, min_samples=5, alpha=None, ...)`, while sklearn/cluster/_hdbscan/hdbscan.py:183-184 documents `alpha : float, default=1.0`. Line 241 unconditionally executes `distance_matrix /= alpha`.
scenario: "A caller invokes `_hdbscan_brute(X)` relying on the documented default → `distance_matrix /= None` raises `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`; similarly `min_samples: int, default=None` on line 179 lies (signature default is `5`)."
contract: Change the signature default to `alpha=1.0` and update line 179 to say `default=5`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:161, sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F5 — `_weighted_cluster_center` treats `-3` (missing) as a cluster, contradicting its own class docstring [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:894-895 reads `# Number of non-noise clusters` and computes `n_clusters = len(set(self.labels_) - {-1, -2})` — the outlier label `-3` (defined at sklearn/cluster/_hdbscan/hdbscan.py:74 in `_OUTLIER_ENCODING["missing"]`) is NOT excluded. The class-level `centroids_`/`medoids_` docstring at lines 561-562 & 572-573 explicitly says "the `-1, -2, -3` labels for the outlier clusters are excluded."
scenario: "Fit a non-precomputed matrix containing `np.nan` rows with `store_centers='centroid'` → `-3` is present in `self.labels_` → `n_clusters` is over-counted by one → the loop `for idx in range(n_clusters)` at line 907 walks past the last real cluster; when `idx == n_clusters - 1` corresponds to an integer label that has no samples, `mask = self.labels_ == idx` is all-False, `data = X[mask]` is empty, and `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError`, breaking the promised outputs."
contract: Compute `n_clusters = len(set(self.labels_) - {-1, -2, -3})` so the comment, the class docstring and the loop bound all agree.
instances: single-instance

### F6 — `_compute_stability`: `births` is allocated twice back-to-back
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 contains
```
largest_child = max(largest_child, smallest_cluster)
births = np.full(largest_child + 1, np.nan, dtype=np.float64)

births = np.full(largest_child + 1, np.nan, dtype=np.float64)
```
The first assignment is immediately shadowed by the second; both allocate the identical NaN-filled array.
scenario: "Reader spots the duplicated line and wastes time hunting for a subtle reason (initial values that differ, side effects on a memoryview, etc.) that does not exist → a maintainer patches one of the two lines and leaves the other, producing genuinely confusing dead code."
contract: Delete the redundant duplicate at line 252 (retain only the assignment at line 254).
instances: single-instance
