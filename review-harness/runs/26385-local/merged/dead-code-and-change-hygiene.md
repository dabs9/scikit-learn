I note the system reminders regarding malware analysis. These files are review-harness findings documents about scikit-learn code (an established open-source ML library), not malware. My task is a UNION merge of findings blocks — no code improvement or augmentation is being performed. Proceeding with the merge.

Now producing the union merge. Grouping by same defect (root cause + site):

- Duplicate `births` allocation: A/F1, B/F1, C/F1 → same defect. Most specific = B/F1 (explicit mention of nothing between them and describes redundancy clearly); A/F1 and C/F1 also detailed. A/F1 pinpoints line 252 vs 254 clearly. I'll keep A/F1 verbatim as most specific (identifies exact line to remove).
- Unused `uint8_t` cimport: B/F8, C/F2 → same. Keep B/F8 (more detail on hygiene pass mention).
- Unrelated whitespace edits to Mean Shift docs: A/F4, B/F10, C/F4 → same. A/F4 has a specific instances list; keep A/F4.
- `plot_hdbscan.py` scale-invariance: B/F2, C/F3 → same. B/F2 is most specific (high severity, DBSCAN comparison lines cited).
- `test_hdbscan_precomputed_non_brute`: B/F3, C/F5 → same. B/F3 has more specific line citation and severity.
- `HDBSCAN.__init__` default `n_jobs=4`: B/F11, C/F6 → same. B/F11 more specific with references to DBSCAN/OPTICS convention.
- Misspelled `test_reachibility.py`: A/F5, B/F9 → same. A/F5 more specific (mentions propagation into `_reachability.pyx:132`, `:187`).

Unique findings:
- A/F2 (redundant cast asymmetric)
- A/F3 (duplicate PyArray_SHAPE)
- A/F6 (pxd noexcept mismatch)
- B/F4 (test_dbscan_clustering_outlier_data numpy +)
- B/F5 (dead mst_func = None)
- B/F6 (_hdbscan_prims docs copy)
- B/F7 (_hdbscan_brute alpha=None)

Severity ordering: high, medium, low.

High: B/F2, B/F3, B/F4
Medium: B/F7, B/F11, C/F5 (already merged into B/F3? — C/F5 is medium; but B/F3 is high, so use high). B/F5 (test_hdbscan_precomputed_non_brute) is high per B. C/F3 (plot scale) is medium; merged into B/F2 high.
So mediums: B/F7 (alpha=None), B/F11 (n_jobs=4)
Low: everything else.

### F1 — `plot_hdbscan.py` scale-invariance demo never scales the data
severity: high
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ...)` uses `scale` only in the plot title; `X` is neither multiplied by `scale` on `fit` nor on `plot`, unlike the DBSCAN loop above (line 93-95) which does `dbs.fit(X * scale)` and `plot(X * scale, ...)`.
scenario: "User runs the published example → all three panes show the identical HDBSCAN result on the same unscaled data, so the section titled 'Scale Invariance' visually 'proves' scale invariance by never varying scale; the demonstration is meaningless and misleading"
contract: Use `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)` inside the loop, mirroring the DBSCAN loop that immediately precedes it.
instances: single-instance

### F2 — `test_hdbscan_precomputed_non_brute` exercises invalid algorithm names, not the intended path [out-of-theme]
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` constructs `"prims_kdtree"` / `"prims_balltree"`, but the estimator's `_parameter_constraints["algorithm"]` in sklearn/cluster/_hdbscan/hdbscan.py:635-644 is `StrOptions({"auto", "brute", "kdtree", "balltree"})`. The test asserts `pytest.raises(ValueError)`, so it passes — but on the parameter-validation error, never reaching the precomputed+tree combination the test docstring claims to guard.
scenario: "Someone regresses the precomputed+kdtree / precomputed+balltree guard in HDBSCAN.fit → this test still passes because any invalid string will raise ValueError during `_validate_params`; the regression ships"
contract: Replace the algorithm name with a valid value (`"kdtree"` / `"balltree"`) so the test drives the actual code path the docstring claims to cover, and match the specific error message from the fit-time guard.
instances: single-instance

### F3 — `test_dbscan_clustering_outlier_data` uses numpy `+` where set-union of indices is intended [out-of-theme]
severity: high
evidence: sklearn/cluster/tests/test_hdbscan.py:212 — `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx` is a `np.ndarray` (e.g. `[2, 5]` from `np.flatnonzero`) and `infinite_labels_idx` is an ndarray (e.g. `[0]`). `+` on two ndarrays broadcasts elementwise (giving `[2, 5]` here), it does not concatenate, so index `0` is silently dropped from the exclusion set.
scenario: "The test runs with these fixed arrays → index 0 (the sample set to `[np.inf, 1]`) remains in `clean_idx`, so `X_outlier[clean_idx]` still contains a non-finite row and `clean_model.fit` follows a different code path than intended; the assertion may still pass by coincidence but the test does not verify what it claims"
contract: Concatenate the indices with `np.concatenate([missing_labels_idx, infinite_labels_idx])` (or convert each to a list before `+`), then build the set of exclusions.
instances: single-instance

### F4 — `_hdbscan_brute` default `alpha=None` triggers `TypeError` on `/=`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`; sklearn/cluster/_hdbscan/hdbscan.py:241 unconditionally does `distance_matrix /= alpha`. Called from `HDBSCAN.fit` at line 767 which always passes `alpha=self.alpha` (validated `> 0`), so today the default is unreachable, but the default value is a live foot-gun for any future caller.
scenario: "New caller invokes `_hdbscan_brute(X)` relying on the documented default `alpha=1.0` → `TypeError: unsupported operand type(s) for /=: 'ndarray' and 'NoneType'` (dense) or scipy `TypeError` (sparse)"
contract: Change the default to `alpha=1.0` (matching both the docstring at line 183 and the `_hdbscan_prims` sibling default).
instances: single-instance

### F5 — `HDBSCAN.__init__` default `n_jobs=4` contradicts documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4` in the signature; sklearn/cluster/_hdbscan/hdbscan.py:492-496 docstring: "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context." A default of `4` is neither `None` nor matches any other sklearn clusterer (DBSCAN, OPTICS use `n_jobs=None`).
scenario: "User instantiates `HDBSCAN()` expecting single-threaded behavior per the docs → gets 4-thread parallelism unpredictably; also inconsistent with every other sklearn clusterer's convention"
contract: Set `n_jobs=None` in the `__init__` signature to match the documented default and sklearn-wide convention.
instances: single-instance

### F6 — Duplicate `births` allocation is dead code
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed on line 252 with the assigned array never read; the identical statement on line 254 immediately overwrites `births` with a fresh array.
scenario: "Reader/maintainer inspects `_compute_stability` → sees a duplicated allocation and must reason whether the first line has a side effect (it does not)"
contract: Remove the assignment at line 252 so `births` is allocated exactly once, on the line preceding the population loop.
instances: single-instance

### F7 — Redundant `<cnp.intp_t>` cast asymmetric with sibling branch
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:181-182 — `right_count = <cnp.intp_t> hierarchy[right - n_samples].cluster_size` uses a redundant cast; the symmetric branch at line 177 `left_count = hierarchy[left - n_samples].cluster_size` correctly does not, and `cluster_size` is already declared `intp_t` in `_tree.pxd:38`. The cast is superseded/leftover code from an earlier iteration.
scenario: "Reader compares the left/right branches → asymmetry falsely suggests the two counts have different types or that the cast matters"
contract: Drop the `<cnp.intp_t>` cast on line 182 so the two branches are symmetric.
instances: single-instance

### F8 — Duplicate `PyArray_SHAPE` extern declaration in `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:44-45 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` locally, but the same extern is already declared in `sklearn/cluster/_hdbscan/_tree.pxd:48-49` which `_linkage.pyx` already cimports from (see line 40 `from ...cluster._hdbscan._tree cimport HIERARCHY_t`). The local re-declaration is superseded/leftover code.
scenario: "Refactor to change the PyArray_SHAPE signature → must be updated in two places rather than one; risk of divergent declarations"
contract: Remove the local `cdef extern` block in `_linkage.pyx` and rely on the declaration exported by `_tree.pxd`.
instances: single-instance

### F9 — Unrelated trailing-whitespace cleanup bundled with HDBSCAN PR
severity: low
evidence: doc/modules/clustering.rst hunks at lines 392-425 (see hunks/doc_modules_clustering.rst.diff:22-45) — three edits that strip trailing spaces from Mean Shift prose ("hill climbing", "density estimation", "small enough and is") unrelated to HDBSCAN. The PR description enumerates three novel changes (typedef swap, `shape[0]→len`, unused-var trim); trailing-whitespace fixes to Mean Shift docs are not sanctioned by the plan.
scenario: "Blame/history archaeology for Mean Shift docs → unrelated commit surfaces as owning these lines; PR review scope inflated"
contract: Revert the whitespace-only edits to the Mean Shift documentation sections; keep only the HDBSCAN additions.
instances: [doc/modules/clustering.rst:399-400, doc/modules/clustering.rst:422, doc/modules/clustering.rst:429]

### F10 — Test file name misspelled ("reachibility") [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py (new file per DIFF_MANIFEST.md:26) — filename misspells "reachability" as "reachibility". Note the module under test is `_reachability.pyx` (spelled correctly), and the docstring inside the file references "mutual reachability graph". The misspelling also appears inside `_reachability.pyx:132` (`mutual_reachibility_distance`) and `_reachability.pyx:187`, but the file-name itself is the change-hygiene defect committed by this PR.
scenario: "Developer searches for `test_reachability` → does not find the test file; the misspelling propagates via cargo-cult"
contract: Rename the file to `test_reachability.py` before further work builds on the misspelling.
instances: single-instance

### F11 — `.pxd` declares `noexcept` but `.pyx` definitions omit it
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:14-15 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, but the corresponding definitions in sklearn/cluster/_hierarchical_fast.pyx:331 (`cdef void union(self, intp_t m, intp_t n):`) and sklearn/cluster/_hierarchical_fast.pyx:339 (`cdef intp_t fast_find(self, intp_t n):`) omit `noexcept`. The `noexcept` in the pxd is either a lost-semantics attempt to declare no-exception behavior that the pyx does not honor, or dead annotation that will trigger Cython deprecation warnings when signatures mismatch.
scenario: "Cython compiles with strict signature matching → mismatched exception-spec between `.pxd` and `.pyx` triggers warning/error, or exception propagation semantics differ from what the `.pxd` advertises to external cimporters"
contract: Add `noexcept` to both method definitions in `_hierarchical_fast.pyx` so the pyx matches the pxd declaration exactly.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F12 — Dead `mst_func = None` initialization in `HDBSCAN.fit`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:763 — `mst_func = None`. Every branch below (algorithm in {"kdtree","balltree","brute"} explicit, and every branch of the `auto` else) unconditionally assigns `mst_func`; the initial `None` is never observed.
scenario: "Reader assumes the None default is meaningful → wastes time verifying every path reassigns; or a future edit adds a branch that fails to assign `mst_func`, and the AttributeError becomes a confusing 'NoneType is not callable' at line 820 instead of a clear NameError"
contract: Remove the `mst_func = None` line and let missing-branch assignments raise NameError, which is louder and more debuggable.
instances: single-instance

### F13 — `_hdbscan_prims` documents a `copy` parameter it does not accept
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature has no `copy`; docstring at sklearn/cluster/_hdbscan/hdbscan.py:313-318 describes `copy : bool, default=False`. There is no `copy` in the signature or body of `_hdbscan_prims`.
scenario: "Contributor reads the docstring and passes `copy=True` → TypeError from `_hdbscan_prims`; or a maintainer copies the docstring elsewhere and propagates the ghost parameter"
contract: Delete the `copy` paragraph from `_hdbscan_prims`'s docstring; it belongs only on `_hdbscan_brute`.
instances: single-instance

### F14 — Unused `uint8_t` cimport in `_tree.pxd`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:30 — `from ...utils._typedefs cimport intp_t, float64_t, uint8_t`; grep shows only `intp_t` and `float64_t` used in the file. No struct field or declaration uses `uint8_t`.
scenario: "Cython lint / pre-commit that the PR description explicitly credits for 'trimmed unused variables' should have flagged this dead cimport → the PR claims that hygiene pass; this slipped through"
contract: Remove `uint8_t` from the cimport list in `_tree.pxd`.
instances: single-instance
