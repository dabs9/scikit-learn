### F1 — HDBSCAN `n_jobs` init default disagrees with documented default
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — the `__init__` signature is `n_jobs=4`, but the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None. Number of jobs to run in parallel to calculate distances. `None` means 1 unless in a :obj:`joblib.parallel_backend` context.`
scenario: "User instantiates `HDBSCAN()` following the documented API contract expecting `n_jobs=None` (single-thread by default and honoring `joblib.parallel_backend`) → estimator silently pins `self.n_jobs = 4` and dispatches parallel work regardless of the surrounding `joblib.parallel_backend`, breaking the documented `None`-means-1 semantics and yielding surprising CPU / joblib-backend behavior in constrained environments"
contract: The `__init__` default must be `n_jobs=None` to match the documented contract and sklearn convention; the concrete parallelism level is a runtime concern to be selected via `joblib.parallel_backend` or `_openmp_effective_n_threads`.
instances: single-instance

### F2 — `UnionFind` pxd declares `noexcept` but pyx methods do not
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` — no `noexcept` on either implementation.
scenario: "A build on any Cython version that enforces exception-clause coherence between pxd declaration and pyx definition (e.g. Cython 3.x, which scikit-learn will eventually adopt and which the neighboring `_k_means_common.{pxd,pyx}` files already track) → signature mismatch error / warning during `configure_extension_modules()` in setup.py:212-216, blocking or destabilizing the extension build for `sklearn.cluster._hdbscan._linkage` (which cimports `UnionFind`)"
contract: The pyx signatures for `UnionFind.union` and `UnionFind.fast_find` must include the same `noexcept` clause declared in the pxd; add ` noexcept` after each `cdef` method declaration in `_hierarchical_fast.pyx`.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F3 — HDBSCAN Cython modules call NumPy C API without invoking `cnp.import_array()`
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:45 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` and uses it at sklearn/cluster/_hdbscan/_linkage.pyx:87. sklearn/cluster/_hdbscan/_tree.pyx (via `_tree.pxd:48-49`) uses `PyArray_SHAPE` at lines 311, 484, 535, 571, 741. Neither `.pyx` file calls `cnp.import_array()`. The sibling `sklearn/cluster/_hdbscan/_reachability.pyx:41` correctly calls `cnp.import_array()`, matching every other `.pyx` in the repo that imports NumPy's C API (see `_dbscan_inner.pyx:8`, `_gradient_boosting.pyx:10`, `_tree.pyx:31`, and 16 further examples).
scenario: "The module extension `sklearn.cluster._hdbscan._linkage` (or `_tree`) is imported before any other module that has initialized the NumPy C API in the same process → future maintenance introducing a bona-fide NumPy C API call (e.g., `PyArray_DATA`, `PyArray_NewFromDescr`) into the same module will segfault at runtime because `import_array()` was never invoked; consistency with sibling extensions is required to make future edits safe"
contract: Add `cnp.import_array()` at module scope in both `sklearn/cluster/_hdbscan/_linkage.pyx` and `sklearn/cluster/_hdbscan/_tree.pyx`, matching the pattern used in every other sklearn Cython module that touches the NumPy C API.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:34, sklearn/cluster/_hdbscan/_tree.pyx:33]

### F4 — Test wires an unregistered `algorithm` string, so it never exercises the guard it claims to [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`. The `algorithm` parameter constraint at sklearn/cluster/_hdbscan/hdbscan.py:629-637 accepts only `{"auto","brute","kdtree","balltree"}` — the strings `"prims_kdtree"` / `"prims_balltree"` are not registered.
scenario: "Test is intended to validate the 'precomputed + non-brute → ValueError' logic in `HDBSCAN.fit` (sklearn/cluster/_hdbscan/hdbscan.py:785-791) → in practice the estimator raises `InvalidParameterError` (a `ValueError` subclass) from `self._validate_params()` before that logic is reached, so the test passes without ever exercising the target check; any future regression removing the precomputed-vs-tree guard would go undetected"
contract: Change the two parametrized algorithm strings to the registered values (`"kdtree"` and `"balltree"`) so the test hits the intended `fit`-time guard.
instances: single-instance

### F5 — HDBSCAN test module filename typo prevents targeted `--pyargs` selection
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — the file is named `test_reachibility.py` (misspelling of "reachability"); the module it tests is `sklearn/cluster/_hdbscan/_reachability.pyx:1` (spelled correctly).
scenario: "developer runs `pytest --pyargs sklearn.cluster._hdbscan.tests.test_reachability` (matching the source module name) → collection error because the file is actually `test_reachibility`; targeted CI/test invocations that use the correct spelling silently miss these tests"
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py` so the test module name matches the module under test.
instances: single-instance

### F6 — Scale-invariance example fits HDBSCAN on unscaled `X`, contradicting the demonstration [out-of-theme]
severity: low
evidence: examples/cluster/plot_hdbscan.py:106-110 — the loop advertised as showing HDBSCAN's scale invariance calls `hdb.fit(X)` with the un-scaled data and then plots `X` (not `X * scale`); the parallel DBSCAN example immediately above (lines 87-89) correctly does `dbs.fit(X * scale)` and plots `X * scale`.
scenario: "Reader compares the two loops to see HDBSCAN handling scaled data → sees identical plots but only because the same unscaled `X` was fed in, not because HDBSCAN is scale-invariant — the intended pedagogical point is unsupported by the code"
contract: Fit and plot on the scaled data in the HDBSCAN loop, mirroring the DBSCAN loop: replace `hdb.fit(X)` / `plot(X, ...)` with `hdb.fit(X * scale)` / `plot(X * scale, ...)`.
instances: single-instance
