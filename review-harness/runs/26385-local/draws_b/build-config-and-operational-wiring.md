### F1 — pxd declares `noexcept` on `UnionFind` methods that pyx implementation lacks
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, but sklearn/cluster/_hierarchical_fast.pyx:331 implements `cdef void union(self, intp_t m, intp_t n):` and line 339 implements `cdef intp_t fast_find(self, intp_t n):` (no `noexcept`). The pyx also drops the member declarations (`cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size`) which the pxd now owns — that part is fine, but the exception-specification mismatch on both methods is not.
scenario: "Cython 3.x (or a future scikit-learn upgrade past 0.29.x) compiling `_hierarchical_fast.pyx` → error 'signature does not match previous declaration' because the pxd asserts `noexcept` while the pyx implementation does not, breaking the build."
contract: The `noexcept` qualifier in `_hierarchical_fast.pxd` must be repeated verbatim on the `union` and `fast_find` implementations in `_hierarchical_fast.pyx` (matching the pattern used consistently elsewhere in scikit-learn, e.g. `_k_means_common.pxd`/`.pyx`).
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F2 — `HDBSCAN.__init__` default `n_jobs=4` contradicts documented `default=None` and sklearn-wide n_jobs convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 sets `n_jobs=4` in `__init__`, while the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None` with the standard "`None` means 1 unless in a `joblib.parallel_backend` context" wording. All other sklearn estimators default n_jobs to `None`; hard-coding `4` here spawns 4 pairwise-distance workers whether the user asked for parallelism or not.
scenario: "User instantiates `HDBSCAN()` without arguments → 4 threads are used for pairwise distances regardless of the ambient `joblib.parallel_backend`, breaking the documented default, silently overriding any global thread setting, and disagreeing with `repr(HDBSCAN())` promised behavior."
contract: The `__init__` default for `n_jobs` must be `None`, matching the docstring and the rest of scikit-learn.
instances: single-instance

### F3 — Test file `test_reachibility.py` misspelled, does not match the module it tests
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 exists (misspelling of "reachability"), yet the module it exercises is sklearn/cluster/_hdbscan/_reachability.pyx (correctly spelled). All test functions inside use the correct spelling (`test_mutual_reachability_graph_*`). Pytest still collects the file, but the packaging/wiring intent — a test module named after its subject — is broken.
scenario: "Developer greps for the tests of `_reachability` by module name → grep for `test_reachability.py` returns nothing and the test file is easy to miss."
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py` so the test module name matches its subject module.
instances: single-instance

### F4 — Internal Cython variable `mutual_reachibility_distance` misspelled in `_reachability.pyx` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:133 declares `floating mutual_reachibility_distance`, and it is reused at lines 144, 149, 182, 206, 209, 210. The surrounding docstrings and public API name spell it correctly as "reachability" (e.g. line 47 "mutual reachability graph"). This is a straight misspelling propagated through both the dense and sparse implementations.
scenario: "Grep-based navigation for `reachability_distance` in this module → does not match the loop-local variable, hindering search and code review."
contract: Rename every occurrence of `mutual_reachibility_distance` to `mutual_reachability_distance` in `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:133, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F5 — `HDBSCAN.algorithm` option strings diverge from sklearn-wide `kd_tree`/`ball_tree` convention
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:616-645 registers `algorithm` allowed values as `{"auto", "brute", "kdtree", "balltree"}` (no underscore). The rest of scikit-learn (e.g. sklearn/neighbors/_base.py:392 `StrOptions({"auto", "ball_tree", "kd_tree", "brute"})`) uses `"kd_tree"`/`"ball_tree"`. Internally HDBSCAN then translates `"kdtree"`→`"kd_tree"` (line 798) and `"balltree"`→`"ball_tree"` (line 802) before passing to `NearestNeighbors`, confirming the underscore form is the sklearn convention it is silently mapping to.
scenario: "User consulting sklearn documentation for `algorithm='ball_tree'` translates that call to `HDBSCAN(algorithm='ball_tree')` → InvalidParameterError because the parameter validator rejects underscored variants unique to the rest of the package."
contract: Accept the same string form (`"kd_tree"`, `"ball_tree"`) that every other sklearn estimator uses for the tree-based algorithm selectors.
instances: single-instance

### F6 — Docstring example loop uses `scale` variable but calls `hdb.fit(X)` unscaled [out-of-theme]
severity: low
evidence: examples/cluster/plot_hdbscan.py:112-116 iterates `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ..., parameters={"scale": scale})`. Compared to the immediately preceding DBSCAN loop at lines 93-95 which correctly does `dbs.fit(X * scale); plot(X * scale, ...)`, the HDBSCAN "scale invariance" demonstration never scales `X`, so it cannot actually demonstrate the property it claims to and simply plots the same clustering three times with different subtitle labels.
scenario: "Reader runs the rendered example intending to see HDBSCAN's scale invariance → all three subplots are identical because the data is never scaled, undermining the tutorial's argument."
contract: Fit and plot on `X * scale` instead of `X` inside the HDBSCAN scale-invariance loop, matching the DBSCAN loop above it.
instances: single-instance
