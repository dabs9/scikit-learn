### F1 — Test asserts on parameter-validation error, not on the guard it names
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282-284 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` then `with pytest.raises(ValueError): hdb.fit(X)`. HDBSCAN's `_parameter_constraints["algorithm"]` (sklearn/cluster/_hdbscan/hdbscan.py:629-638) only permits `{"auto","brute","kdtree","balltree"}`. `"prims_kdtree"`/`"prims_balltree"` do not exist in this codebase (grep for `prims_kdtree|prims_balltree` in `sklearn/` returns nothing).
scenario: "run test_hdbscan_precomputed_non_brute → ValueError is raised by `_validate_params()` on the illegal algorithm string, before any precomputed/non-brute logic is executed → the test claims to prove `precomputed + tree` fails but actually proves only that an invalid algorithm string fails"
contract: Rewrite the test to use `algorithm="kdtree"` / `"balltree"` (the real algorithm names) so the ValueError comes from the precomputed/tree incompatibility path the test docstring describes.
instances: single-instance

### F2 — `_weighted_cluster_center` treats `-3` (missing) as a real cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`. `_OUTLIER_ENCODING["missing"]["label"] == -3` (line 80). When `metric != "precomputed"` and `X` contains rows with `np.nan`, `self.labels_` receives `-3` for those rows (line 843). `-3` is not subtracted from the set, so it inflates `n_clusters`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` with any NaN row in X → `n_clusters` includes the phantom `-3` cluster → the loop iterates `range(n_clusters)` and for the extra idx computes `data = X[self.labels_ == idx]` which is empty → `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError: Weights sum to zero, can't be normalized`"
contract: Exclude every outlier label — `n_clusters = len(set(self.labels_) - {out['label'] for out in _OUTLIER_ENCODING.values()} - {-1})` — or iterate `sorted(set(self.labels_) - <outliers>)` directly, so noise/missing/infinite rows never enter the centroid loop.
instances: single-instance

### F3 — Stale comment describes wrong outlier labels
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The very next lines use `_OUTLIER_ENCODING["infinite"]["label"]` = `-2` and `_OUTLIER_ENCODING["missing"]["label"]` = `-3` (see the encoding definitions at lines 72-84 and the class docstring at lines 540-543 which correctly document `-2` / `-3`).
scenario: "a maintainer reads the comment to reason about the label semantics → mis-conflates infinite with -1 (which is generic noise) and missing with -2 (which is infinite) → wrong changes to downstream code"
contract: Update the comment to state "np.inf → -2 and np.nan → -3", matching the class docstring and `_OUTLIER_ENCODING`.
instances: single-instance

### F4 — Duplicate `births` initialization in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-256 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appears twice in a row with an intervening blank line and nothing between them.
scenario: "the first `births` allocation is discarded and immediately replaced by the identical second allocation → no functional consequence, only a wasted allocation"
contract: Delete the redundant duplicate.
instances: single-instance

### F5 — Test computes `clean_idx` via array addition, not concatenation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212-221 — `missing_labels_idx = np.flatnonzero(...)` (line 212, returns ndarray `[2, 5]`), `infinite_labels_idx = np.flatnonzero(...)` (line 215, ndarray `[0]`), then line 218 `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx + infinite_labels_idx` is elementwise numpy addition, not concatenation — `[2, 5] + [0]` broadcasts to `[2, 5]`, so the sample at index `0` (an infinite outlier) is left in `clean_idx`. The subsequent `assert_array_equal(clean_labels, labels[clean_idx])` passes only because HDBSCAN still assigns the infinite sample `-2` when kept, matching the original labels at that index.
scenario: "test intends to compare a clean-refit against outlier-marked labels → actually keeps outlier row 0 in the 'clean' subset → the assertion succeeds by accident and any change that made the two labelings diverge at row 0 would silently be missed"
contract: Concatenate as lists — `clean_idx = list(set(range(200)) - set(missing_labels_idx.tolist() + infinite_labels_idx.tolist()))` (or `np.concatenate([...]).tolist()`) — so `clean_idx` truly excludes every outlier.
instances: single-instance

### F6 — Plan claim "cnp.*_t → *_t from _typedefs" not applied to _tree.pyx
severity: low
evidence: The PR description lists as a novel change: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`". `_linkage.pyx` and `_reachability.pyx` import scalar typedefs from `...utils._typedefs cimport ...` and use bare `intp_t`, `float64_t`, etc. However `sklearn/cluster/_hdbscan/_tree.pyx` still uses `cnp.intp_t`, `cnp.float64_t`, `cnp.uint8_t` etc. throughout — 90 total occurrences (see grep). Only the `_tree.pxd` header imports the new typedefs (`_tree.pxd:36`); the `.pyx` body itself was not converted.
scenario: "plan-of-record says every Cython file uses the new typedef import → _tree.pyx is inconsistent → next author touching this file has to guess which convention applies and may reintroduce mismatches"
contract: Convert `_tree.pyx` to use `intp_t`/`float64_t`/`uint8_t` from `...utils._typedefs cimport ...`, matching the sibling `_linkage.pyx`/`_reachability.pyx`.
instances: single-instance

### F7 — `n_jobs=4` default violates sklearn convention
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `def __init__(..., n_jobs=4, ...)`. The class docstring at lines 486-489 states "`None` means 1 unless in a :obj:`joblib.parallel_backend` context. `-1` means using all processors." This is the standard sklearn contract, which is only correct when the default is `None`; the actual default `4` hard-codes 4 workers regardless of the joblib context.
scenario: "user constructs `HDBSCAN()` inside a `joblib.parallel_backend('threading', n_jobs=1)` block expecting the doc-specified 'None means 1' behavior → 4 workers are still spawned → oversubscription and violation of the surrounding backend contract"
contract: Set the default to `n_jobs=None` in `__init__` to match the documented contract and the rest of sklearn.
instances: single-instance

### F8 — Docstring uses capitalized "KDTree"/"BallTree" as if they were the parameter values
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:472-478 — "algorithm : {"auto", "brute", "kdtree", "balltree"}, default="auto" ... Both `"KDTree"` and `"BallTree"` algorithms use the :class:`~sklearn.neighbors.NearestNeighbors` estimator." The accepted parameter values are lowercase `"kdtree"`/`"balltree"` (matches `_parameter_constraints` and `StrOptions`).
scenario: "user copy-pastes `algorithm=\"KDTree\"` from the description prose → `_validate_params()` raises `InvalidParameterError` because the value does not match the lowercase StrOptions set"
contract: Refer to the algorithm strings only in their true lowercase form throughout the docstring — `\"kdtree\"` / `\"balltree\"`.
instances: single-instance

### F9 — `_hdbscan_brute` signature defaults contradict its docstring
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166 — signature `def _hdbscan_brute(X, min_samples=5, alpha=None, ...)`; docstring lines 178-190 say `min_samples : int, default=None` and `alpha : float, default=1.0`. If called at the default `alpha=None`, line 241 `distance_matrix /= alpha` raises `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`.
scenario: "downstream contributor writes a new caller relying on documented defaults → passes no alpha → runtime crash inside the private helper"
contract: Change the signature to `alpha=1.0` and `min_samples=None` (or update the docstring so it matches the actual defaults) — keep signature and docstring in lockstep.
instances: single-instance

### F10 — `max_distance` is forwarded to `pairwise_distances` as an unknown kwarg
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:238-249 — `distance_matrix = pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` at 238, then `max_distance = metric_params.get("max_distance", 0.0)` at 249. `max_distance` is a HDBSCAN-specific control (documented in `mutual_reachability_graph`), but it is not popped from `metric_params`, so it is passed into `pairwise_distances`, which forwards unknown kwargs to the underlying metric callable.
scenario: "user passes `metric_params={'max_distance': 1.0}` for sparse-precomputed handling with a metric callable that rejects unknown kwargs (e.g. a `functools.partial` with strict signature) → pairwise_distances raises `TypeError: got an unexpected keyword argument 'max_distance'` before the sparse path is even reached"
contract: Pop `max_distance` out of `metric_params` before forwarding to `pairwise_distances` (e.g. `max_distance = metric_params.pop("max_distance", 0.0)` and keep the copy of `metric_params` separate if the caller depends on the original dict).
instances: single-instance
