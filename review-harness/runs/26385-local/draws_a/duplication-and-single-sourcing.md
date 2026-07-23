### F1 — `_tree.pyx` never migrated to `_typedefs.pxd` types, contradicting the plan
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:33-40 — `cimport numpy as cnp` with no `from ...utils._typedefs cimport ...`; body uses `cnp.float64_t INFTY`, `cnp.intp_t NOISE`, and 90 total `cnp.*_t` occurrences (`cnp.intp_t`, `cnp.float64_t`, `cnp.uint8_t`); yet the sibling files `_tree.pxd:30`, `_linkage.pyx:42`, and `_reachability.pyx:40` all cimport typedefs from `...utils._typedefs`.
scenario: "Plan claims `cnp.*_t` was replaced with `*_t` from `_typedefs.pxd` → `_tree.pyx` retains the old convention while the rest of the module has been migrated, giving two co-existing typing conventions in the same package and defeating the single-sourcing goal."
contract: Cimport `intp_t`, `float64_t`, `uint8_t` from `sklearn.utils._typedefs` in `_tree.pyx` and replace every `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` occurrence with the imported typedefs, matching the sibling `.pyx`/`.pxd` files in the same package.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:61, sklearn/cluster/_hdbscan/_tree.pyx:66-67, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90-91, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:145-155, sklearn/cluster/_hdbscan/_tree.pyx:240-249, sklearn/cluster/_hdbscan/_tree.pyx:280-290, sklearn/cluster/_hdbscan/_tree.pyx:299-329, sklearn/cluster/_hdbscan/_tree.pyx:359-475, sklearn/cluster/_hdbscan/_tree.pyx:513-559, sklearn/cluster/_hdbscan/_tree.pyx:606-698]

### F2 — Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`, but the same extern is already declared in `sklearn/cluster/_hdbscan/_tree.pxd:48-49` and `_linkage.pyx` already `cimport`s from `_tree` (line 40).
scenario: "One package-local header (`_tree.pxd`) already provides the `PyArray_SHAPE` extern; `_linkage.pyx` re-declares it → two sources of truth for the same C signature. If the extern is ever adjusted (e.g., signature or header), the copies drift silently."
contract: Delete the extern block in `_linkage.pyx` and rely on the declaration in `_tree.pxd` (already cimported).
instances: single-instance

### F3 — `HIERARCHY_t`/`CONDENSED_t` structs and their numpy dtypes are separately hand-declared in two files
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pxd:34-46 defines `ctypedef packed struct HIERARCHY_t { intp_t left_node; intp_t right_node; float64_t value; intp_t cluster_size; }` and `CONDENSED_t { intp_t parent; intp_t child; float64_t value; intp_t cluster_size; }`. The corresponding numpy dtypes are hand-written in `sklearn/cluster/_hdbscan/_tree.pyx:42-54` (`HIERARCHY_dtype = np.dtype([("left_node", np.intp), ("right_node", np.intp), ("value", np.float64), ("cluster_size", np.intp)])`, `CONDENSED_dtype = ...`).
scenario: "Field name/order/dtype of the packed struct and the numpy dtype must stay byte-identical or memory-view casts corrupt data → any future edit to one side (e.g., rename `value`→`lambda_val` or change `intp` widths) that misses the other side silently produces wrong clusters."
contract: Emit the ctypedef from the numpy dtype through a `.tp` Tempita template so the packed struct and the numpy dtype are generated from a single field list, and remove the redundant hand-maintained copy.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:34-38, sklearn/cluster/_hdbscan/_tree.pxd:42-46, sklearn/cluster/_hdbscan/_tree.pyx:42-47, sklearn/cluster/_hdbscan/_tree.pyx:49-54]

### F4 — Hardcoded outlier labels bypass `_OUTLIER_ENCODING`, driving values apart
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` hardcodes the noise (`-1`) and infinite-outlier (`-2`) labels; the missing-outlier label (`-3`) is silently omitted from the exclusion set. `_OUTLIER_ENCODING` is defined at lines 65-79 exactly to be the single source of truth for these labels, and is used elsewhere (lines 842-843, 850-851, 964-969). sklearn/cluster/_hdbscan/_tree.pyx:541 — `if cluster_num == -1: continue` hardcodes the noise label instead of the module-level `NOISE` constant defined on line 40 (`cdef cnp.intp_t NOISE = -1`) and used at lines 414/420/492.
scenario: "Outlier labels are re-encoded inline at `_weighted_cluster_center` and `_tree.pyx` `get_probabilities` → if `_OUTLIER_ENCODING`/`NOISE` are ever renumbered (e.g., adding a fourth outlier category or shifting `-3` handling), the hardcoded literals silently disagree; for the current tree, `_weighted_cluster_center` already excludes `-1,-2` but not `-3`, causing `medoids_`/`centroids_` to size incorrectly when missing-data rows exist and produce empty rows or an IndexError at line 908 (`mask = self.labels_ == idx`) where `idx` would iterate up to a count that includes `-3` as a cluster."
contract: Reference `_OUTLIER_ENCODING` (or a derived `_OUTLIER_LABELS = {v['label'] for v in _OUTLIER_ENCODING.values()} | {-1}`) instead of the literal set `{-1, -2}` in `_weighted_cluster_center`, and use the `NOISE` constant instead of `-1` in `_tree.pyx:541`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/_hdbscan/_tree.pyx:541]

### F5 — Duplicate `births` allocation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is immediately overwritten by the identical statement on line 254 `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`. The first allocation is dead work (its buffer is orphaned before any read).
scenario: "Cut-and-paste left two copies of the same allocation → wastes an allocation per call and is a maintenance trap: a future edit that changes one but not the other yields inconsistent init state."
contract: Delete the redundant `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252, keeping only the line 254 allocation.
instances: single-instance

### F6 — `INFTY`/`np.infty`/`np.inf` used inconsistently across sibling files
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:39 defines `cdef cnp.float64_t INFTY = np.inf` used at line 174; sklearn/cluster/_hdbscan/_linkage.pyx:93 and 159 use `np.infty` directly (`np.full(n_samples, fill_value=np.infty, ...)`); sklearn/cluster/_hdbscan/hdbscan.py:389 uses `np.inf` (`outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`). Three names for the same infinity value across three files in the same package.
scenario: "`np.infty` was deprecated in NumPy 1.20 and removed in NumPy 2.0 → the `_linkage.pyx` sites will break with newer NumPy while sibling files that use `np.inf`/`INFTY` continue to work; the drift is invisible until CI hits a supported NumPy version that removed the alias."
contract: Standardise on one spelling for the infinity sentinel in this package. Use `np.inf` in Python-visible code (Cython allocations) and `INFTY`/`from libc.math cimport INFINITY` in Cython nogil contexts; remove all `np.infty` usages.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:93, sklearn/cluster/_hdbscan/_linkage.pyx:159]

### F7 — Precomputed symmetry check re-implements `sklearn.utils.validation.check_symmetric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:222-234 — inside `_hdbscan_brute`, precomputed symmetry is validated by hand: shape check + `_allclose_dense_sparse(X, X.T)` with a bespoke error message. sklearn/utils/validation.py:1309 provides `check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False)` which is the module-wide helper used by other precomputed-matrix estimators.
scenario: "The bespoke check disagrees with the shared helper on tolerance (`_allclose_dense_sparse` defaults vs `check_symmetric`'s `tol=1e-10`) and on error wording → users of precomputed matrices see different behaviour from HDBSCAN than from every other precomputed-matrix estimator, and future tightening of `check_symmetric` won't propagate here."
contract: Replace the manual shape and `_allclose_dense_sparse(X, X.T)` block in `_hdbscan_brute` with `check_symmetric(X, raise_exception=True)`.
instances: single-instance

### F8 — Inline comment in `fit()` restates outlier labels incorrectly, driven by the same hardcoding pattern
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The actual encoding (lines 65-79 and lines 842-843 immediately below) maps `np.inf` → `-2` and `np.nan` → `-3`. The comment is a stale duplicate of the truth held in `_OUTLIER_ENCODING`.
scenario: "Because outlier labels are documented in prose next to code that reads them from `_OUTLIER_ENCODING`, prose and dict drift → reader assumes noise label `-1` covers infinite samples and misinterprets `labels_`."
contract: Replace prose-hardcoded label values in this comment with references to `_OUTLIER_ENCODING["infinite"]["label"]` and `_OUTLIER_ENCODING["missing"]["label"]`.
instances: single-instance

### F9 — `test_hdbscan_precomputed_non_brute` passes an invalid algorithm name and no longer tests the documented behavior [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` with `tree in {"kd","ball"}` produces `algorithm="prims_kdtree"` / `"prims_balltree"`, but the estimator's parameter constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-638 restricts `algorithm` to `{"auto","brute","kdtree","balltree"}`. `fit` raises `InvalidParameterError` from `_validate_params()` (line 697) before any precomputed+non-brute logic is reached.
scenario: "Test claims to verify 'HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm', but it triggers a parameter-validation error instead → the actual precomputed+kdtree/balltree code path is untested; a regression that silently accepts precomputed data with `algorithm='kdtree'` would pass CI."
contract: Change the test to pass a valid tree-based algorithm name (`f"{tree}tree"`, i.e. `"kdtree"`/`"balltree"`) so the ValueError from the precomputed+non-brute branch is what is asserted, and match against that branch's actual error message.
instances: single-instance
