Acknowledged. I have enough info to compile findings now.

### F1 — `remap_single_linkage_tree` receives a set of indices where docstring declares a boolean array
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 documents `non_finite : ndarray / Boolean array of which entries in the raw data are non-finite`, but the sole caller at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` — a `set` of integer indices. Inside the function line 383 does `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)` and line 388 `for i, outlier in enumerate(non_finite): outlier_tree[i] = (outlier, ...)`. If the argument were actually a boolean ndarray as documented, `len(non_finite)` would equal n_samples (not the outlier count) and `enumerate(non_finite)` would iterate `True/False` values, silently producing wrong outlier rows.
scenario: "Any code path that trusts the docstring and passes a boolean mask (the documented contract) → silently wrong tree construction, then arbitrary label assignment"
contract: Change the parameter documentation to `non_finite : set of int / A set of raw indices corresponding to non-finite samples` so the actual runtime contract matches what the sole caller provides.
instances: single-instance

### F2 — Docstring/signature drift on `HDBSCAN.n_jobs` default (`None` documented, `4` in signature)
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 documents `n_jobs : int, default=None / … ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context`, but the actual `__init__` at sklearn/cluster/_hdbscan/hdbscan.py:658 uses `n_jobs=4`. Every instance of `HDBSCAN()` therefore silently allocates 4 workers when the user believes they got the documented single-threaded default.
scenario: "User instantiates `HDBSCAN()` in a resource-constrained/multi-tenant environment expecting the documented single-threaded default → 4 worker processes are spawned per fit, contradicting the parameter contract"
contract: The `__init__` default MUST be `n_jobs=None` to match the documented default and the ``None means 1`` semantic promised in the parameter description.
instances: single-instance

### F3 — `_hdbscan_brute` `alpha=None` default violates its numeric contract
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`, sklearn/cluster/_hdbscan/hdbscan.py:183 documents `alpha : float, default=1.0`, and sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally executes `distance_matrix /= alpha`. Calling `_hdbscan_brute(X)` without `alpha` therefore raises `TypeError: unsupported operand type(s) for /=: … 'NoneType'` — the default value is not a legal value for the parameter's own contract.
scenario: "Any external / test caller invokes `_hdbscan_brute(X, metric='precomputed')` accepting documented defaults → immediate TypeError inside `distance_matrix /= alpha`"
contract: The default MUST be `alpha=1.0`, matching both the docstring and the corresponding parameter in `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.
instances: single-instance

### F4 — `_do_labelling` scalar-vs-array contract: `parent_lambda` compared as a scalar
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 assigns `parent_lambda = lambda_array[child_array == n]` — an untyped Python object holding a 1-D `float64` ndarray. Line 505 does `if parent_lambda >= threshold:`, which relies on the array having exactly one element (otherwise NumPy raises `ValueError: The truth value of an array with more than one element is ambiguous`). The comment at line 496 asserts this without any runtime type/shape guard, and `parent_lambda` is not `cdef`-typed as a scalar.
scenario: "A malformed condensed tree (test fixture, buggy caller, or duplicate child-row) in which more than one row has `child_array == n` → `if parent_lambda >= threshold` raises an ambiguous-truth-value ValueError inside cluster labelling"
contract: Extract a scalar explicitly (e.g. `cdef cnp.float64_t parent_lambda = lambda_array[child_array == n][0]`) and rely on the invariant, not on an implicit 1-element-array coercion.
instances: single-instance

### F5 — Type mismatch of `allow_single_cluster` between callers/definitions
severity: low
evidence: `_get_clusters` at sklearn/cluster/_hdbscan/_tree.pyx:646 declares `cnp.uint8_t allow_single_cluster=False`, but `_do_labelling` at sklearn/cluster/_hdbscan/_tree.pyx:435 declares `cnp.intp_t allow_single_cluster`, and `tree_to_labels` at sklearn/cluster/_hdbscan/_tree.pyx:60 declares `bint allow_single_cluster=False`. Three different Cython types are used for the same logical boolean flag across the same-file call chain (`tree_to_labels → _get_clusters → _do_labelling`).
scenario: "A future signed/unsigned overflow or refactor that widens the value (e.g. passing `-1` for a tri-state) → silent misinterpretation because each function sees a different underlying integer width/sign"
contract: Use `bint` uniformly for `allow_single_cluster` in `tree_to_labels`, `_get_clusters`, and `_do_labelling`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:60, sklearn/cluster/_hdbscan/_tree.pyx:435, sklearn/cluster/_hdbscan/_tree.pyx:646]

### F6 — `HIERARCHY_dtype` / `CONDENSED_dtype` array shapes documented as `(n_samples,)` instead of `(n_samples - 1,)`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 documents `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype`, sklearn/cluster/_hdbscan/_tree.pyx:138 documents `condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype`, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656 repeat the same claim. However, the actual construction in sklearn/cluster/_hdbscan/_linkage.pyx:252 uses `np.zeros(n_samples - 1, ...)` and callers such as sklearn/cluster/_hdbscan/hdbscan.py:219 correctly document shape `(n_samples - 1,)`. The Cython-side shape contract is off-by-one and disagrees with the Python-side contract.
scenario: "Consumer of `_condense_tree`/`labelling_at_cut` sizes buffers based on the documented `(n_samples,)` shape → off-by-one buffer / reader mismatch"
contract: Align the Cython docstrings to `(n_samples - 1,)` to match both the implementation and the Python-side docstring.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:371, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]

### F7 — `test_hdbscan_precomputed_non_brute` asserts on the wrong error path via an invalid `algorithm` string [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 sets `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"`, `"prims_balltree"`), but the `algorithm` StrOptions at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only permits `{"auto", "brute", "kdtree", "balltree"}`. `_validate_params()` (called at hdbscan.py:697) therefore raises `InvalidParameterError` (a `ValueError` subclass) before the intended `"precomputed + tree"` check at lines 772-783 is ever reached. The test passes for a reason unrelated to what its docstring at lines 279-281 claims to verify.
scenario: "A future refactor removes the precomputed-vs-tree ValueError branches at hdbscan.py:772-783 → this test still passes silently, giving false confidence in the coverage of that contract"
contract: The test MUST use a valid algorithm from the StrOptions set (`"kdtree"` or `"balltree"`) so that the intended precomputed-vs-tree branch is exercised.
instances: single-instance

### F8 — `test_dbscan_clustering_outlier_data` uses element-wise `+` on `np.ndarray` indices where set-union/concatenation is intended [out-of-theme]
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:212 computes `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`, where `missing_labels_idx` and `infinite_labels_idx` are `np.ndarray`s returned by `np.flatnonzero` at lines 206 and 209. `missing_labels_idx + infinite_labels_idx` performs element-wise addition (with broadcasting from shape `(1,)` to `(2,)`), producing `[2, 5]` for the specific fixture, not the intended concatenation `[2, 5, 0]`. The test only passes because the broadcast happens to leave the "missing" indices unchanged; had `infinite_labels_idx` contained a nonzero value, `clean_idx` would silently omit a different row than intended.
scenario: "Any future fixture change that puts a nonzero index into `infinite_labels_idx` → `clean_idx` is wrong, `clean_model` fits the wrong subset, and assertion at line 215 may compare different points"
contract: Concatenate the two arrays explicitly, e.g. `set(np.concatenate([missing_labels_idx, infinite_labels_idx]).tolist())`.
instances: single-instance

### F9 — `_get_clusters` `stabilities` return contract broken (docstring promises 3-tuple, code returns 2-tuple)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:681-690 documents three return values (`labels`, `probabilities`, `stabilities`), but the actual `return` at sklearn/cluster/_hdbscan/_tree.pyx:797 is `return (labels, probs)` — a 2-tuple. `tree_to_labels` at sklearn/cluster/_hdbscan/_tree.pyx:70 unpacks only two values, so the caller conforms to the implementation, but the docstring's third return (`stabilities`) is a broken contract for any external consumer of `_get_clusters`.
scenario: "External code (e.g. a follow-up feature or user script that calls `_get_clusters` directly) unpacks the documented 3-tuple → `ValueError: not enough values to unpack`"
contract: Remove the `stabilities` entry from the `_get_clusters` Returns block so the documented contract matches the actual 2-tuple return.
instances: single-instance

### F10 — `HDBSCAN` `algorithm` StrOptions omit the historical `"kd_tree"` / `"ball_tree"` names still emitted by the routing code
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:629-638 validates `algorithm` against `{"auto", "brute", "kdtree", "balltree"}`, but the routing code at sklearn/cluster/_hdbscan/hdbscan.py:798, :802, :812, :817 forwards `algo="kd_tree"` / `algo="ball_tree"` (underscored) into `NearestNeighbors(algorithm=algo)`. The user-facing API contract accepts one spelling while the internal contract to `NearestNeighbors.algorithm` requires another, and there is no cast/mapping documented; a reader/user cannot round-trip the accepted string.
scenario: "A user sets `algorithm='kd_tree'` matching the well-known `NearestNeighbors` spelling → InvalidParameterError, despite this being the exact string internally forwarded to NearestNeighbors"
contract: The parameter-constraints StrOptions and the internal dispatch strings MUST use a single, documented spelling; if `kdtree`/`balltree` is the chosen public spelling, translate exactly once at the boundary with a comment naming the sklearn API contract, and mention this translation in the docstring.
instances: single-instance
