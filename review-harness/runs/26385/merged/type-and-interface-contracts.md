### F1 — `n_jobs` documented default is `None` but constructor default is `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the text "`None` means 1 unless in a :obj:`joblib.parallel_backend` context", while sklearn/cluster/_hdbscan/hdbscan.py:658 declares the parameter as `n_jobs=4` in `HDBSCAN.__init__`.
scenario: "User reads docs → constructs `HDBSCAN()` expecting single-threaded / joblib-parallel-backend-controlled execution → estimator spawns 4 worker jobs unconditionally"
contract: Change the constructor default to `n_jobs=None` to match the documented, parameter-constraint-permitted (`[Integral, None]`) contract and the sibling `_hdbscan_brute` / `_hdbscan_prims` helpers (which both default `n_jobs=None`).
instances: single-instance

### F2 — `_weighted_cluster_center` receives reduced X but full-length labels when non-finite samples exist
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,855,908-910 — `X = X[finite_index]` reduces X to (n_finite, n_features); later `self.labels_ = new_labels` is expanded back to (n_raw,); then `self._weighted_cluster_center(X)` is called with the reduced X while `mask = self.labels_ == idx` produces a mask of length n_raw and `data = X[mask]` boolean-indexes X (length n_finite) with that longer mask.
scenario: "`HDBSCAN(store_centers='centroid').fit(X_with_nan_or_inf)` → `IndexError: boolean index did not match indexed array along dimension 0` from `X[mask]` inside `_weighted_cluster_center`"
contract: `_weighted_cluster_center` must be given data whose leading axis matches `self.labels_`; pass `self._raw_data` (and skip the outlier rows via the mask) rather than the finite-only `X`.
instances: single-instance

### F3 — `remap_single_linkage_tree(non_finite=...)` docstring says `ndarray` (boolean array) but caller passes a `set`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-365 documents `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`; the sole call site at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` (a `set` of raw integer indices). The body at hdbscan.py:369, 383 relies on `len(non_finite)` (count of unique indices) and hdbscan.py:388 iterates `enumerate(non_finite)` treating each element as a raw sample index — behavior that would silently give the wrong `outlier_tree` if a caller followed the docstring and passed a boolean mask.
scenario: "External caller (or future maintainer) reads the docstring, passes a boolean ndarray of length n_samples → `len(non_finite)` returns n_samples instead of the outlier count, `outlier` in the loop iterates the booleans (0/1) instead of raw indices, producing a corrupt hierarchy"
contract: Update the docstring to state `non_finite : set of int  A set of raw-data row indices that are non-finite (union of infinite and missing indices)`, and rename the parameter or type-annotate accordingly so the declared type matches what the function actually consumes.
instances: single-instance

### F4 — Length-1 ndarray coerced to scalar / used in `if` context in `_do_labelling` and `traverse_upwards`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 assigns `parent_lambda = lambda_array[child_array == n]` (an ndarray, not cdef-typed) and then `_tree.pyx:505` writes `if parent_lambda >= threshold:` — a length-1 boolean-array truth test. sklearn/cluster/_hdbscan/_tree.pyx:582 declares `cdef cnp.intp_t root, parent` and `_tree.pyx:586` assigns `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` (an ndarray) into the scalar; `_tree.pyx:593` likewise assigns `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']` (an ndarray) into `cdef cnp.float64_t parent_eps`. The comment at `_tree.pyx:496-497` acknowledges the pattern relies on the boolean mask matching exactly one row.
scenario: "NumPy ≥ 1.25 deprecates length-1-ndarray-to-scalar coercion → these coercions raise `DeprecationWarning` today and will raise `TypeError` in a future NumPy, breaking `_do_labelling`/`traverse_upwards` at runtime; if the invariant ever fails (e.g. duplicate child edge) the length-N array coercion raises `ValueError` with no diagnostic hinting at the real cause"
contract: Extract the scalar explicitly at each site — replace `lambda_array[child_array == n]` / `cluster_tree[...][field]` with `...[0]` (or `.item()`) so the RHS is a genuine scalar, mirroring the existing correct pattern at `_tree.pyx:621` (`eps = 1 / distances[leaf_nodes][0]`).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:498, sklearn/cluster/_hdbscan/_tree.pyx:505, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593]

### F5 — `UnionFind` cdef signatures disagree between `.pxd` and `.pyx` (`noexcept` present in one, absent in the other)
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; the corresponding definitions at sklearn/cluster/_hierarchical_fast.pyx:331 (`cdef void union(self, intp_t m, intp_t n):`) and sklearn/cluster/_hierarchical_fast.pyx:339 (`cdef intp_t fast_find(self, intp_t n):`) omit the `noexcept` specifier.
scenario: "Under Cython 3, a `cdef` function returning a value with no exception specifier defaults to propagating Python exceptions (equivalent to `except *`); the `.pxd` promises callers no exception propagation, but the definition does not opt into that guarantee → either a Cython signature-mismatch/deprecation warning at build time, or a silent semantic divergence between what the header advertises and what the body implements"
contract: Add `noexcept` to both definitions in `_hierarchical_fast.pyx` so the definition matches the header exactly (`cdef void union(self, intp_t m, intp_t n) noexcept:` and `cdef intp_t fast_find(self, intp_t n) noexcept:`).
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F6 — `labels_` docstring promises `-2` for infinite samples but precomputed path never assigns it
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:534-537 states "Samples with infinite elements (+/- np.inf) are given the label -2"; but the `-2` remapping at sklearn/cluster/_hdbscan/hdbscan.py:830-844 is gated on `self.metric != "precomputed"`. For `metric="precomputed"` (dense) the code at sklearn/cluster/_hdbscan/hdbscan.py:741-751 sets `force_all_finite=False`, keeps `np.inf` in the matrix, and never runs the outlier remap — so a row full of `np.inf` distances flows through `mst_from_mutual_reachability` untouched (warning only, line 256-265) and receives ordinary tree-derived labels (typically -1 or a cluster id).
scenario: "User passes a precomputed distance matrix containing `np.inf` entries → `labels_` for those samples is -1 (or a cluster label), never -2, silently violating the documented Attributes contract."
contract: Apply the `_OUTLIER_ENCODING` remap for infinite-distance samples on the precomputed path so `labels_` uniformly assigns `-2` to infinite-distance rows regardless of `metric`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:534, sklearn/cluster/_hdbscan/hdbscan.py:830, sklearn/cluster/_hdbscan/hdbscan.py:955]

### F7 — `test_hdbscan_precomputed_non_brute` uses non-existent algorithm names and passes for the wrong reason
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 sets `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"` / `"prims_balltree"`), but the `_parameter_constraints` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only accept `{"auto", "brute", "kdtree", "balltree"}`. `_validate_params()` therefore raises `InvalidParameterError` (a `ValueError` subclass) *before* the precomputed-vs-tree check at sklearn/cluster/_hdbscan/hdbscan.py:772-783 ever runs. The test only asserts `pytest.raises(ValueError)` with no `match=`, so it succeeds without ever exercising the code path its docstring claims to cover.
scenario: "Someone regresses the `metric=='precomputed'` + `algorithm=='kdtree'` guard → this test still passes because it fails at parameter validation instead, and the missing coverage lets the real regression ship."
contract: Change the test to `algorithm="kdtree"` / `algorithm="balltree"` and match the actual error message so it verifies the documented behaviour.
instances: single-instance

### F8 — `n_clusters` excludes only `{-1, -2}`, so label `-3` (missing) is counted as a cluster
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`, but `_OUTLIER_ENCODING["missing"]["label"] == -3` (lines 74) and hdbscan.py:843 writes `-3` into `self.labels_` for missing samples.
scenario: "Fit with `store_centers` set and data containing `np.nan` → the -3 outlier label is treated as a real cluster id; `centroids_`/`medoids_` are allocated an extra row, and `range(n_clusters)` iterates one iteration too far, producing an empty-slice `np.average` warning/NaN row"
contract: enumerate the outlier labels from `_OUTLIER_ENCODING` (i.e., subtract `{-1} | {out["label"] for out in _OUTLIER_ENCODING.values()}`) so all outlier encodings are excluded.
instances: single-instance

### F9 — `_hdbscan_brute` signature `alpha=None` contradicts its `default=1.0` docstring and would crash when invoked with the default
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`; the docstring at sklearn/cluster/_hdbscan/hdbscan.py:183-184 says `alpha : float, default=1.0`; sklearn/cluster/_hdbscan/hdbscan.py:241 does `distance_matrix /= alpha`, which raises `TypeError: unsupported operand type(s) for /=: ... 'NoneType'` if `alpha` really is `None`.
scenario: "Any refactor that calls `_hdbscan_brute(X, ..., **kwargs)` without an explicit `alpha` (relying on the signature-advertised default) → immediate `TypeError` at `/= alpha`."
contract: Change the signature default to `alpha=1.0` to match the docstring and match the sibling `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.
instances: single-instance

### F10 — `_hdbscan_prims` docstring documents a `copy` parameter absent from the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature is `def _hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)` — no `copy`. Yet the docstring at sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False` with a full description. Also `leaf_size` (in the signature) is not documented in the docstring.
scenario: "Caller reading the docstring passes `copy=True` → it silently lands in `**metric_params` and is forwarded to the distance metric, producing a `TypeError` from the distance backend at fit time rather than being ignored as the docstring implies."
contract: Remove the `copy` paragraph from the `_hdbscan_prims` docstring and add a `leaf_size` entry so signature and docstring match exactly.
instances: single-instance

### F11 — `_condense_tree` uses an inconsistent, redundant `<cnp.intp_t>` cast on `right_count` only
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:177 assigns `left_count = hierarchy[left - n_samples].cluster_size` with no cast; sklearn/cluster/_hdbscan/_tree.pyx:182 assigns `right_count = <cnp.intp_t> hierarchy[right - n_samples].cluster_size`. The `HIERARCHY_t.cluster_size` field is already declared `intp_t` in sklearn/cluster/_hdbscan/_tree.pxd:38, and `right_count` is declared `cnp.intp_t` at sklearn/cluster/_hdbscan/_tree.pyx:155 — the cast is a no-op and diverges from the parallel left-side assignment for no reason.
scenario: "A future reviewer sees the asymmetric cast and copies the pattern elsewhere thinking it's meaningful → the misleading precedent propagates and later masks a real widening/truncation when the struct layout changes."
contract: Drop the `<cnp.intp_t>` cast on line 182 so both branches read the field with identical typing.
instances: single-instance

### F12 — `_tree.pyx` retains `cnp.*_t` typing throughout, contradicting the PR's stated typedef migration
severity: low
evidence: The PR description lists as change #1: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`". `_linkage.pyx` (line 42) and `_reachability.pyx` (line 40) both `cimport` `intp_t, float64_t, int64_t, uint8_t` from `...utils._typedefs`, and use the unqualified names. But `_tree.pyx` still imports numpy as `cnp` at sklearn/cluster/_hdbscan/_tree.pyx:33 and uses `cnp.intp_t / cnp.float64_t / cnp.uint8_t` in 90 spots (grep count) — including cdef declarations at sklearn/cluster/_hdbscan/_tree.pyx:39-40, 66-67, 82-91, 144-156, 240-249, 302-305, 329-337, 388-394, 467-474, 519-525, 692-700. The paired `_tree.pxd` already imports `intp_t, float64_t, uint8_t` from `_typedefs`, so `_tree.pyx` is the only member of `_hdbscan/` still on the `cnp.*_t` typing regime.
scenario: "A later reader checks whether `_typedefs` migration is complete for `_hdbscan` → sees the plan says yes, sees `_tree.pyx` still uses `cnp.*_t`, has to spend time reconciling; future numpy header decoupling work has to re-open a file the plan claimed was already done."
contract: Complete the migration in `_tree.pyx` by replacing all `cnp.intp_t / cnp.float64_t / cnp.uint8_t` occurrences with the unqualified `intp_t / float64_t / uint8_t` imported from `...utils._typedefs`, matching `_linkage.pyx` and `_reachability.pyx`.
instances: single-instance
