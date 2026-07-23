Acknowledged. `dbscan_clustering` accepts caller-supplied `cut_distance` and `min_cluster_size` with no parameter validation (no `_validate_params`, no explicit range check). `min_cluster_size` gets passed to Cython `labelling_at_cut(... cnp.intp_t min_cluster_size)`. If the caller passes a negative or non-integer value, Cython will attempt coercion. Passing negative `cut_distance` may not error but produce unexpected labels. This is minor input-validation weakness.

Looking overall at security-and-boundaries: this is a numerical library added mostly from an existing implementation, no network, secrets, or PII handling. The `metric` parameter accepting callables is by design.

Let me finalize the findings.

### F1 — `_weighted_cluster_center` fails to exclude the missing-data (-3) outlier label from n_clusters
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` subtracts only `-1` and `-2` from the label set, ignoring the `-3` label defined by `_OUTLIER_ENCODING["missing"]["label"]` (sklearn/cluster/_hdbscan/hdbscan.py:73-78); the class docstring at line 561-573 states "the `-1, -2, -3` labels for the outlier clusters are excluded".
scenario: "User passes a feature array containing `np.nan` entries with `store_centers='centroid'` (or 'medoid'/'both') → n_clusters is inflated by 1, `self.centroids_`/`self.medoids_` are allocated with an extra row, and the final iteration `for idx in range(n_clusters)` computes `mask = self.labels_ == idx` for a `idx` that never appears as a real cluster label, producing an empty slice fed into `np.average(...)` (raising ZeroDivisionError / warning about weights summing to zero) or a garbage medoid, and the resulting attribute contains an uninitialized last row."
contract: Exclude every negative outlier label declared in `_OUTLIER_ENCODING` (i.e. `{-1} | {v["label"] for v in _OUTLIER_ENCODING.values()}`) when computing `n_clusters`, matching the promise in the class docstring.
instances: single-instance

### F2 — `dbscan_clustering` accepts caller-supplied parameters without validation [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-962 — `dbscan_clustering(self, cut_distance, min_cluster_size=5)` performs no `_validate_params` / range check and passes the values straight into the Cython routine `labelling_at_cut(const HIERARCHY_t[::1] linkage, cnp.float64_t cut, cnp.intp_t min_cluster_size)` at sklearn/cluster/_hdbscan/_tree.pyx:359-362.
scenario: "Caller invokes `est.dbscan_clustering(cut_distance=-1.0, min_cluster_size=-5)` → Cython coerces the negative `min_cluster_size` into an intp, `cluster_size[cluster] < min_cluster_size` (line 419) never triggers, producing silently-invalid labels while the estimator API contract of validated, positive integer parameters is broken."
contract: Validate `cut_distance` and `min_cluster_size` against numeric-interval constraints (e.g. `cut_distance>=0`, `min_cluster_size>=1`) before dispatching to the Cython routine, matching the fit-time parameter validation performed by `_validate_params`.
instances: single-instance
