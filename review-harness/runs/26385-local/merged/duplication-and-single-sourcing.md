I note the system reminders about malware analysis. The files I read are review-harness draft findings documents (markdown files describing code review findings) — not code and not malware. I am performing my assigned union-merge role and will not augment any code. I am only merging finding documents.

Now merging the three draws. Identifying same-defect groups:

- **Outlier labels `{-1,-2}` hardcoded**: A/F4 (also includes _tree.pyx:541 NOISE hardcoding), B/F2, C/F2. Draw A is most specific (covers two sites). Keep A/F4 verbatim, union instances.
- **Duplicate `births` allocation**: A/F5, B/F6, C/F3. All same. Keep A/F5 (comparable specificity), union instances (all single-instance).
- **Duplicate `PyArray_SHAPE` extern**: A/F2, B/F1, C/F5. B/F1 slightly more explicit about cimport transitive path; A/F2 is concise. All similar specificity — pick A/F2 (verbatim), union instances.
- **`_tree.pyx` `cnp.*_t` typing not migrated**: A/F1, C/F6. C/F6 has more detailed instances list (90 lines enumerated). Keep C/F6 verbatim (most specific), union instances with A/F1.
- **Struct + numpy dtype dual definitions**: A/F3 (HIERARCHY_t/CONDENSED_t), C/F4 (adds MST_edge_dtype/MST_edge_t). C/F4 is more specific (covers 3 pairs including MST). Keep C/F4 verbatim, union instances.
- **`.shape[0]` retained**: B/F7 (hdbscan.py sites only), C/F7 (broader — includes _tree.pyx, _reachability.pyx, hdbscan.py). C/F7 most specific. Keep C/F7 verbatim, union instances.
- **Outlier comment inf/nan mapping wrong**: A/F8, B/F5. Same site (hdbscan.py:832-833). A/F8 slightly more specific about being driven by hardcoding pattern. Keep A/F8, union instances (both single-instance).
- **Test uses invalid algorithm string**: A/F9 [out-of-theme], B/F8 [out-of-theme]. Same defect. A/F9 more specific. Keep A/F9, union.

Distinct findings:
- A/F6 (INFTY/np.infty/np.inf inconsistency) — unique
- A/F7 (check_symmetric duplication) — unique
- B/F3 (n_jobs default drift) — unique
- B/F4 (min_cluster_size defaults drift 10 vs 5) — unique
- C/F1 (_hdbscan_prims docstring `copy` param nonexistent) — unique
- C/F8 (NOISE sentinel not single-sourced) — related to outlier hardcoding but distinct site/scope, keep separate
- C/F9 (docstring typos duplicated) — unique
- C/F10 (two test locations split coverage) — unique
- C/F11 [out-of-theme] (alpha=None vs docstring default=1.0) — unique

Ordering by severity: high → medium → low.

- High: B/F2 (outlier labels `{-1,-2}`) — but A/F4 covers same defect at medium. Since we keep the MOST SPECIFIC draw's block verbatim, and A/F4 is more specific (covers 2 sites), we use A/F4's severity=medium.

Wait — re-reading rule 2: "keep the MOST SPECIFIC draw's block VERBATIM — title, severity, evidence, scenario, contract wording unchanged". So we use A/F4's severity (medium) since A/F4 is most specific.

Final ordering:
- Medium: A/F4 (outlier `{-1,-2}` hardcoded), C/F6 (cnp.*_t typing), C/F4 (dual struct/dtype), B/F3 (n_jobs drift), A/F9 [out-of-theme], B/F8-duplicated-into-A/F9, C/F11 [out-of-theme]
- Low: everything else

### F1 — Hardcoded outlier labels bypass `_OUTLIER_ENCODING`, driving values apart
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})` hardcodes the noise (`-1`) and infinite-outlier (`-2`) labels; the missing-outlier label (`-3`) is silently omitted from the exclusion set. `_OUTLIER_ENCODING` is defined at lines 65-79 exactly to be the single source of truth for these labels, and is used elsewhere (lines 842-843, 850-851, 964-969). sklearn/cluster/_hdbscan/_tree.pyx:541 — `if cluster_num == -1: continue` hardcodes the noise label instead of the module-level `NOISE` constant defined on line 40 (`cdef cnp.intp_t NOISE = -1`) and used at lines 414/420/492.
scenario: "Outlier labels are re-encoded inline at `_weighted_cluster_center` and `_tree.pyx` `get_probabilities` → if `_OUTLIER_ENCODING`/`NOISE` are ever renumbered (e.g., adding a fourth outlier category or shifting `-3` handling), the hardcoded literals silently disagree; for the current tree, `_weighted_cluster_center` already excludes `-1,-2` but not `-3`, causing `medoids_`/`centroids_` to size incorrectly when missing-data rows exist and produce empty rows or an IndexError at line 908 (`mask = self.labels_ == idx`) where `idx` would iterate up to a count that includes `-3` as a cluster."
contract: Reference `_OUTLIER_ENCODING` (or a derived `_OUTLIER_LABELS = {v['label'] for v in _OUTLIER_ENCODING.values()} | {-1}`) instead of the literal set `{-1, -2}` in `_weighted_cluster_center`, and use the `NOISE` constant instead of `-1` in `_tree.pyx:541`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/_hdbscan/_tree.pyx:541]

### F2 — `cnp.*_t` typing pervasive in `_tree.pyx` contradicts plan bullet #1 and diverges from `_tree.pxd`
severity: medium
evidence: The PR plan explicitly lists "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`" as a novel change. `sklearn/cluster/_hdbscan/_tree.pxd:36` already imports and uses `intp_t, float64_t, uint8_t` from `...utils._typedefs`. However `sklearn/cluster/_hdbscan/_tree.pyx` still declares 90 occurrences of `cnp.intp_t`/`cnp.float64_t`/`cnp.uint8_t` (verified via grep) — e.g. lines 39-40 (`cdef cnp.float64_t INFTY = np.inf` / `cdef cnp.intp_t NOISE = -1`), 58-73 (function signature), 145-155, 240-249, 281-290, 305-311, 335-343, 355-368, 388-400, 429-441, 471-483, 513-522, 555-565, 573-580, 605-621, 641-654, 693-706. Sister files `_linkage.pyx` and `_reachability.pyx` were converted (they cimport from `_typedefs` at lines 47 and 46 respectively), but `_tree.pyx` was not converted alongside them — the file uses two type systems in parallel (its own `cnp.*_t` locals interoperating with `_tree.pxd`-declared `intp_t`/`float64_t` struct fields).
scenario: "Plan requires single-sourcing typedefs on `_typedefs.pxd`; leaving `cnp.*_t` in `_tree.pyx` → a future rename in `_typedefs.pxd` (e.g. widening `intp_t`) silently produces mismatched types across the pxd/pyx boundary in this one file, and any subsequent developer trying to grep-audit uses of the shared typedefs will miss `_tree.pyx` entirely."
contract: Replace every `cnp.intp_t` / `cnp.float64_t` / `cnp.uint8_t` in `_tree.pyx` with the corresponding `intp_t` / `float64_t` / `uint8_t` cimported from `...utils._typedefs`, matching what `_tree.pxd`, `_linkage.pyx`, and `_reachability.pyx` do.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:61, sklearn/cluster/_hdbscan/_tree.pyx:66, sklearn/cluster/_hdbscan/_tree.pyx:66-67, sklearn/cluster/_hdbscan/_tree.pyx:67, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:90-91, sklearn/cluster/_hdbscan/_tree.pyx:91, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:145-155, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:147, sklearn/cluster/_hdbscan/_tree.pyx:150, sklearn/cluster/_hdbscan/_tree.pyx:151, sklearn/cluster/_hdbscan/_tree.pyx:153, sklearn/cluster/_hdbscan/_tree.pyx:154, sklearn/cluster/_hdbscan/_tree.pyx:155, sklearn/cluster/_hdbscan/_tree.pyx:182, sklearn/cluster/_hdbscan/_tree.pyx:240, sklearn/cluster/_hdbscan/_tree.pyx:240-249, sklearn/cluster/_hdbscan/_tree.pyx:241, sklearn/cluster/_hdbscan/_tree.pyx:243, sklearn/cluster/_hdbscan/_tree.pyx:244, sklearn/cluster/_hdbscan/_tree.pyx:246, sklearn/cluster/_hdbscan/_tree.pyx:247, sklearn/cluster/_hdbscan/_tree.pyx:248, sklearn/cluster/_hdbscan/_tree.pyx:249, sklearn/cluster/_hdbscan/_tree.pyx:280-290, sklearn/cluster/_hdbscan/_tree.pyx:281, sklearn/cluster/_hdbscan/_tree.pyx:289, sklearn/cluster/_hdbscan/_tree.pyx:290, sklearn/cluster/_hdbscan/_tree.pyx:299, sklearn/cluster/_hdbscan/_tree.pyx:299-329, sklearn/cluster/_hdbscan/_tree.pyx:305, sklearn/cluster/_hdbscan/_tree.pyx:306, sklearn/cluster/_hdbscan/_tree.pyx:307, sklearn/cluster/_hdbscan/_tree.pyx:311, sklearn/cluster/_hdbscan/_tree.pyx:335, sklearn/cluster/_hdbscan/_tree.pyx:336, sklearn/cluster/_hdbscan/_tree.pyx:340, sklearn/cluster/_hdbscan/_tree.pyx:343, sklearn/cluster/_hdbscan/_tree.pyx:355, sklearn/cluster/_hdbscan/_tree.pyx:358, sklearn/cluster/_hdbscan/_tree.pyx:359-475, sklearn/cluster/_hdbscan/_tree.pyx:360, sklearn/cluster/_hdbscan/_tree.pyx:361, sklearn/cluster/_hdbscan/_tree.pyx:362, sklearn/cluster/_hdbscan/_tree.pyx:363, sklearn/cluster/_hdbscan/_tree.pyx:388, sklearn/cluster/_hdbscan/_tree.pyx:389, sklearn/cluster/_hdbscan/_tree.pyx:395, sklearn/cluster/_hdbscan/_tree.pyx:396, sklearn/cluster/_hdbscan/_tree.pyx:400, sklearn/cluster/_hdbscan/_tree.pyx:429, sklearn/cluster/_hdbscan/_tree.pyx:432, sklearn/cluster/_hdbscan/_tree.pyx:434, sklearn/cluster/_hdbscan/_tree.pyx:435, sklearn/cluster/_hdbscan/_tree.pyx:436, sklearn/cluster/_hdbscan/_tree.pyx:437, sklearn/cluster/_hdbscan/_tree.pyx:441, sklearn/cluster/_hdbscan/_tree.pyx:471, sklearn/cluster/_hdbscan/_tree.pyx:472, sklearn/cluster/_hdbscan/_tree.pyx:473, sklearn/cluster/_hdbscan/_tree.pyx:483, sklearn/cluster/_hdbscan/_tree.pyx:513, sklearn/cluster/_hdbscan/_tree.pyx:513-559, sklearn/cluster/_hdbscan/_tree.pyx:514, sklearn/cluster/_hdbscan/_tree.pyx:517, sklearn/cluster/_hdbscan/_tree.pyx:518, sklearn/cluster/_hdbscan/_tree.pyx:519, sklearn/cluster/_hdbscan/_tree.pyx:520, sklearn/cluster/_hdbscan/_tree.pyx:522, sklearn/cluster/_hdbscan/_tree.pyx:535, sklearn/cluster/_hdbscan/_tree.pyx:555, sklearn/cluster/_hdbscan/_tree.pyx:557, sklearn/cluster/_hdbscan/_tree.pyx:565, sklearn/cluster/_hdbscan/_tree.pyx:569, sklearn/cluster/_hdbscan/_tree.pyx:573, sklearn/cluster/_hdbscan/_tree.pyx:576, sklearn/cluster/_hdbscan/_tree.pyx:582, sklearn/cluster/_hdbscan/_tree.pyx:605, sklearn/cluster/_hdbscan/_tree.pyx:606, sklearn/cluster/_hdbscan/_tree.pyx:606-698, sklearn/cluster/_hdbscan/_tree.pyx:610, sklearn/cluster/_hdbscan/_tree.pyx:641, sklearn/cluster/_hdbscan/_tree.pyx:642, sklearn/cluster/_hdbscan/_tree.pyx:646, sklearn/cluster/_hdbscan/_tree.pyx:647, sklearn/cluster/_hdbscan/_tree.pyx:693, sklearn/cluster/_hdbscan/_tree.pyx:694, sklearn/cluster/_hdbscan/_tree.pyx:695, sklearn/cluster/_hdbscan/_tree.pyx:696, sklearn/cluster/_hdbscan/_tree.pyx:697, sklearn/cluster/_hdbscan/_tree.pyx:698, sklearn/cluster/_hdbscan/_tree.pyx:700]

### F3 — Parallel `MST_edge_dtype` / `MST_edge_t` and `HIERARCHY_dtype` / `HIERARCHY_t` / `CONDENSED_dtype` / `CONDENSED_t` dual definitions
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:48-58 defines `MST_edge_dtype` (numpy structured dtype with `np.int64, np.int64, np.float64`) and separately `MST_edge_t` (Cython packed struct with `int64_t, int64_t, float64_t`). Same pattern for `HIERARCHY_dtype` (sklearn/cluster/_hdbscan/_tree.pyx:42-47, fields `intp, intp, float64, intp`) vs `HIERARCHY_t` (sklearn/cluster/_hdbscan/_tree.pxd:34-38, fields `intp_t, intp_t, float64_t, intp_t`), and `CONDENSED_dtype` (sklearn/cluster/_hdbscan/_tree.pyx:49-54) vs `CONDENSED_t` (sklearn/cluster/_hdbscan/_tree.pxd:42-46). Each pair encodes the exact same record layout in two independent locations that must be manually kept in sync — nothing in the code checks (e.g. via `sizeof` asserts) that the Cython struct matches the numpy dtype.
scenario: "A future edit widens `left_node` from `intp` to `int64` in the numpy `HIERARCHY_dtype` but forgets to update the `HIERARCHY_t` struct in `_tree.pxd` (or vice versa) → Cython memory-view accesses into the array read/write the wrong bytes for every element, producing silently corrupted trees rather than a type error."
contract: Move both the numpy dtype and the Cython struct for each pair to a single canonical `.pxd` with a runtime `sizeof(MST_edge_t) == MST_edge_dtype.itemsize` assertion at module import so the two representations cannot drift.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:48, sklearn/cluster/_hdbscan/_linkage.pyx:56, sklearn/cluster/_hdbscan/_tree.pyx:42, sklearn/cluster/_hdbscan/_tree.pyx:42-47, sklearn/cluster/_hdbscan/_tree.pyx:49, sklearn/cluster/_hdbscan/_tree.pyx:49-54, sklearn/cluster/_hdbscan/_tree.pxd:34, sklearn/cluster/_hdbscan/_tree.pxd:34-38, sklearn/cluster/_hdbscan/_tree.pxd:42, sklearn/cluster/_hdbscan/_tree.pxd:42-46]

### F4 — `n_jobs` default drifts between `__init__` and its docstring
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 — docstring `n_jobs : int, default=None`; sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature uses `n_jobs=4`. The `_brute_mst`/`_hdbscan_brute`/`_hdbscan_prims` helper docstrings (lines 197, 303) also state `n_jobs : int, default=None`.
scenario: "user reads docstring, expects default single-thread joblib behavior, but constructor silently uses 4 workers → unexpected parallelism, memory usage, and non-reproducible resource consumption in constrained environments"
contract: Pin the constructor default to `n_jobs=None` to match the documented single-sourced default across the file and other sklearn estimators.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:486, sklearn/cluster/_hdbscan/hdbscan.py:658]

### F5 — `test_hdbscan_precomputed_non_brute` passes an invalid algorithm name and no longer tests the documented behavior [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` with `tree in {"kd","ball"}` produces `algorithm="prims_kdtree"` / `"prims_balltree"`, but the estimator's parameter constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-638 restricts `algorithm` to `{"auto","brute","kdtree","balltree"}`. `fit` raises `InvalidParameterError` from `_validate_params()` (line 697) before any precomputed+non-brute logic is reached.
scenario: "Test claims to verify 'HDBSCAN correctly raises an error when passing precomputed data while requesting a tree-based algorithm', but it triggers a parameter-validation error instead → the actual precomputed+kdtree/balltree code path is untested; a regression that silently accepts precomputed data with `algorithm='kdtree'` would pass CI."
contract: Change the test to pass a valid tree-based algorithm name (`f"{tree}tree"`, i.e. `"kdtree"`/`"balltree"`) so the ValueError from the precomputed+non-brute branch is what is asserted, and match against that branch's actual error message.
instances: single-instance

### F6 — `alpha=None` default in `_hdbscan_brute` signature drifts from its docstring `default=1.0` [out-of-theme]
severity: medium
evidence: `sklearn/cluster/_hdbscan/hdbscan.py:161` declares `alpha=None` in the `_hdbscan_brute` signature, but the docstring at line 183 states `alpha : float, default=1.0`. When the caller path in `fit` (line 767) passes `self.alpha` (which is the constructor default `1.0`), `_hdbscan_brute` unconditionally runs `distance_matrix /= alpha` at line 241. If any code path ever invokes `_hdbscan_brute` relying on the documented default (as `_hdbscan_prims` in fact does, using `alpha=1.0`), the None default raises `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'`.
scenario: "Any refactor that calls `_hdbscan_brute(X)` without explicitly passing `alpha=1.0` (relying on the documented default) → immediate `TypeError` on line 241 `distance_matrix /= alpha` because `alpha` is `None`."
contract: Change `_hdbscan_brute(X, min_samples=5, alpha=None, ...)` to `alpha=1.0` so the signature matches its own docstring and its sibling `_hdbscan_prims`.
instances: single-instance

### F7 — Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`, but the same extern is already declared in `sklearn/cluster/_hdbscan/_tree.pxd:48-49` and `_linkage.pyx` already `cimport`s from `_tree` (line 40).
scenario: "One package-local header (`_tree.pxd`) already provides the `PyArray_SHAPE` extern; `_linkage.pyx` re-declares it → two sources of truth for the same C signature. If the extern is ever adjusted (e.g., signature or header), the copies drift silently."
contract: Delete the extern block in `_linkage.pyx` and rely on the declaration in `_tree.pxd` (already cimported).
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44, sklearn/cluster/_hdbscan/_linkage.pyx:44-45, sklearn/cluster/_hdbscan/_tree.pxd:48-49]

### F8 — Duplicate `births` allocation in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is immediately overwritten by the identical statement on line 254 `births = np.full(largest_child + 1, np.nan, dtype=np.float64)`. The first allocation is dead work (its buffer is orphaned before any read).
scenario: "Cut-and-paste left two copies of the same allocation → wastes an allocation per call and is a maintenance trap: a future edit that changes one but not the other yields inconsistent init state."
contract: Delete the redundant `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252, keeping only the line 254 allocation.
instances: single-instance

### F9 — `INFTY`/`np.infty`/`np.inf` used inconsistently across sibling files
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:39 defines `cdef cnp.float64_t INFTY = np.inf` used at line 174; sklearn/cluster/_hdbscan/_linkage.pyx:93 and 159 use `np.infty` directly (`np.full(n_samples, fill_value=np.infty, ...)`); sklearn/cluster/_hdbscan/hdbscan.py:389 uses `np.inf` (`outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`). Three names for the same infinity value across three files in the same package.
scenario: "`np.infty` was deprecated in NumPy 1.20 and removed in NumPy 2.0 → the `_linkage.pyx` sites will break with newer NumPy while sibling files that use `np.inf`/`INFTY` continue to work; the drift is invisible until CI hits a supported NumPy version that removed the alias."
contract: Standardise on one spelling for the infinity sentinel in this package. Use `np.inf` in Python-visible code (Cython allocations) and `INFTY`/`from libc.math cimport INFINITY` in Cython nogil contexts; remove all `np.infty` usages.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:93, sklearn/cluster/_hdbscan/_linkage.pyx:159]

### F10 — Precomputed symmetry check re-implements `sklearn.utils.validation.check_symmetric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:222-234 — inside `_hdbscan_brute`, precomputed symmetry is validated by hand: shape check + `_allclose_dense_sparse(X, X.T)` with a bespoke error message. sklearn/utils/validation.py:1309 provides `check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False)` which is the module-wide helper used by other precomputed-matrix estimators.
scenario: "The bespoke check disagrees with the shared helper on tolerance (`_allclose_dense_sparse` defaults vs `check_symmetric`'s `tol=1e-10`) and on error wording → users of precomputed matrices see different behaviour from HDBSCAN than from every other precomputed-matrix estimator, and future tightening of `check_symmetric` won't propagate here."
contract: Replace the manual shape and `_allclose_dense_sparse(X, X.T)` block in `_hdbscan_brute` with `check_symmetric(X, raise_exception=True)`.
instances: single-instance

### F11 — Inline comment in `fit()` restates outlier labels incorrectly, driven by the same hardcoding pattern
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:832-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The actual encoding (lines 65-79 and lines 842-843 immediately below) maps `np.inf` → `-2` and `np.nan` → `-3`. The comment is a stale duplicate of the truth held in `_OUTLIER_ENCODING`.
scenario: "Because outlier labels are documented in prose next to code that reads them from `_OUTLIER_ENCODING`, prose and dict drift → reader assumes noise label `-1` covers infinite samples and misinterprets `labels_`."
contract: Replace prose-hardcoded label values in this comment with references to `_OUTLIER_ENCODING["infinite"]["label"]` and `_OUTLIER_ENCODING["missing"]["label"]`.
instances: single-instance

### F12 — Cython entry-point defaults for `min_cluster_size` (10) drift from the `HDBSCAN` estimator default (5)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:58 (`tree_to_labels(..., cnp.intp_t min_cluster_size=10, ...)`) and sklearn/cluster/_hdbscan/_tree.pyx:119 (`_condense_tree(..., cnp.intp_t min_cluster_size=10)`) both default to 10; sklearn/cluster/_hdbscan/hdbscan.py:649 has `min_cluster_size=5` for the `HDBSCAN.__init__` and sklearn/cluster/_hdbscan/hdbscan.py:923 has `dbscan_clustering(..., min_cluster_size=5)`.
scenario: "downstream user (or test) calls `_condense_tree(tree)` directly relying on the same 5 default → gets a silently different pruning threshold than the public estimator produces"
contract: Align the Cython defaults to 5 (matching the public estimator) or drop the defaults entirely so callers must pass an explicit value from the single source (`HDBSCAN.min_cluster_size`).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:58, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/hdbscan.py:649, sklearn/cluster/_hdbscan/hdbscan.py:923]

### F13 — `.shape[0]` pattern retained in `_tree.pyx` and `_reachability.pyx` contradicts plan bullet #2
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:90, :145, :146, :396, :531, :563 still use `.shape[0]` on ndarray/memoryview objects (e.g. `hierarchy.shape[0] + 1`, `2 * hierarchy.shape[0]`, `2 * linkage.shape[0]`, `labels.shape[0]`, `children.shape[0]`); sklearn/cluster/_hdbscan/_reachability.pyx:103 and :132 similarly use `distance_matrix.shape[0]`; and sklearn/cluster/_hdbscan/hdbscan.py:758, :764, :767, :840, :846, :852, :896, :902, :907, :909 still use `X.shape[0]` for ndarray. The PR plan explicitly lists "Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects" as a novel change of this PR, but the replacement was applied inconsistently — sister file `_linkage.pyx` similarly retains raw `.shape[0]` in cdef.
scenario: "Reader trusts the plan and uses `git grep '\\.shape\\[0\\]'` to confirm no ndarray uses remain → discovers ~15 residual instances, does not know whether these are memoryviews (allowed) or ndarrays (expected to be `len(*)`), must audit each by hand → the plan's claim is falsified."
contract: Audit every `.shape[0]` in the added HDBSCAN files and rewrite the ndarray cases to `len(*)` so the code matches the plan claim.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:396, sklearn/cluster/_hdbscan/_tree.pyx:531, sklearn/cluster/_hdbscan/_tree.pyx:563, sklearn/cluster/_hdbscan/_reachability.pyx:103, sklearn/cluster/_hdbscan/_reachability.pyx:132, sklearn/cluster/_hdbscan/hdbscan.py:385, sklearn/cluster/_hdbscan/hdbscan.py:387, sklearn/cluster/_hdbscan/hdbscan.py:752, sklearn/cluster/_hdbscan/hdbscan.py:758, sklearn/cluster/_hdbscan/hdbscan.py:761, sklearn/cluster/_hdbscan/hdbscan.py:764, sklearn/cluster/_hdbscan/hdbscan.py:767, sklearn/cluster/_hdbscan/hdbscan.py:840, sklearn/cluster/_hdbscan/hdbscan.py:846, sklearn/cluster/_hdbscan/hdbscan.py:852, sklearn/cluster/_hdbscan/hdbscan.py:896, sklearn/cluster/_hdbscan/hdbscan.py:902, sklearn/cluster/_hdbscan/hdbscan.py:907, sklearn/cluster/_hdbscan/hdbscan.py:909]

### F14 — `_hdbscan_prims` docstring copy-pasted from `_hdbscan_brute` documents a nonexistent `copy` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — `_hdbscan_prims` docstring contains a `copy : bool, default=False` block copied verbatim from `_hdbscan_brute`, but the function signature at lines 269-278 does not accept a `copy` parameter (and unlike `_hdbscan_brute` it never uses `copy` semantics since it always calls `np.asarray(X, order="C")` unconditionally).
scenario: "Reader consults `_hdbscan_prims` docstring → assumes `copy=True` will protect their input data from in-place modification, but the argument silently vanishes into `**metric_params` and cannot influence behavior."
contract: Remove the `copy` docstring block from `_hdbscan_prims` (or delete the `_hdbscan_prims` docstring entirely and inherit from a shared helper) so it exactly matches its signature.
instances: single-instance

### F15 — `NOISE = -1` sentinel not single-sourced across HDBSCAN module
severity: low
evidence: `sklearn/cluster/_hdbscan/_tree.pyx:40` defines the sentinel `cdef cnp.intp_t NOISE = -1` and uses it at lines 420, 426, 492. However the Python-side `hdbscan.py` (line 63 comment refers to `-1 noise label`, line 895 uses hardcoded `{-1, -2}`, and `dbscan_clustering` docstring at line 955 refers to `-1`), tests (test_hdbscan.py line 43 `{-1}`, line 377 `int(-1 in hdb.labels_)`), and the example (examples/cluster/plot_hdbscan.py lines 41, 50, 53, 55) all hard-code the `-1` literal instead of referencing a single-sourced constant. The Cython constant is not exposed on the Python side and `_OUTLIER_ENCODING` intentionally starts at `-2` (see the comment at hdbscan.py:63 "extensions to the -1 noise label"), leaving the fundamental noise sentinel with no single source of truth across the Python and Cython halves of the module.
scenario: "A follow-up PR decides to shift the noise label from `-1` to another sentinel to disambiguate it from user labels → editor must find every hard-coded `-1` across `.py`, `.pyx`, tests, and the example, whereas `NOISE` gets renamed once; misses drift silently past unit tests that themselves hard-code `-1`."
contract: Extend `_OUTLIER_ENCODING` in `hdbscan.py` with a `"noise"` entry whose `label` is `-1`, and reference it from `_weighted_cluster_center`, `dbscan_clustering`, the tests, and the example so the noise sentinel has exactly one source of truth across the module.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:414, sklearn/cluster/_hdbscan/hdbscan.py:895, sklearn/cluster/tests/test_hdbscan.py:43, sklearn/cluster/tests/test_hdbscan.py:377, examples/cluster/plot_hdbscan.py:41, examples/cluster/plot_hdbscan.py:50, examples/cluster/plot_hdbscan.py:53, examples/cluster/plot_hdbscan.py:55]

### F16 — Copy-pasted docstring typos ("mutual-reahability", "collecteion", "single-linkage tree tree") duplicated across HDBSCAN files
severity: low
evidence: The misspellings "mutual-reahability" (missing 'c') and "collecteion" appear 5 times as a matched pair — sklearn/cluster/_hdbscan/_linkage.pyx:75-76, :137-138, :228-229 and sklearn/cluster/_hdbscan/hdbscan.py:102-103, :143-144. The redundant "single-linkage tree tree" (repeated word) appears 5 times — _linkage.pyx:234, hdbscan.py:149, :220, :326, :361. Each of these is one docstring template copy-pasted across multiple functions rather than being extracted or corrected in one place.
scenario: "Reader searches the API docs for 'mutual-reachability' → finds no hits, must guess the misspelling; fixing one instance leaves the others visible in generated Sphinx output."
contract: Fix all five instances of each phrase in a single edit; ideally extract the common `Returns`/`Parameters` blocks into a shared docstring template so future edits happen exactly once.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:228, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/hdbscan.py:144, sklearn/cluster/_hdbscan/hdbscan.py:149, sklearn/cluster/_hdbscan/hdbscan.py:220, sklearn/cluster/_hdbscan/hdbscan.py:326, sklearn/cluster/_hdbscan/hdbscan.py:361]

### F17 — Two separate HDBSCAN test locations split coverage without a shared helper
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 and sklearn/cluster/tests/test_hdbscan.py:1 — the PR adds tests in two locations: `sklearn/cluster/tests/test_hdbscan.py` (the main sklearn cluster tests directory) and `sklearn/cluster/_hdbscan/tests/test_reachibility.py` (a private submodule tests directory that only tests one Cython file). The two files share utility import patterns (`sklearn.utils._testing`) and both need to construct outlier fixtures, but each file re-creates its own random data setup rather than sharing a fixture module. This duplicates the "location where a reader looks for HDBSCAN tests" and permits fixture drift.
scenario: "Contributor changes the outlier semantics and updates test_hdbscan.py, missing test_reachibility.py → inconsistent test coverage across the two test directories, and the private-submodule tests may not be discovered by contributors who only look under `sklearn/cluster/tests/`."
contract: Consolidate all HDBSCAN tests under `sklearn/cluster/tests/test_hdbscan*.py` (splitting by aspect if desired, e.g. `test_hdbscan_reachability.py`) so the location convention matches every other sklearn cluster algorithm.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/tests/test_hdbscan.py:1]
