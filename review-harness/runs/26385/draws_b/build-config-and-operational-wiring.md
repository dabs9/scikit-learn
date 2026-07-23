### F1 — `n_jobs` default in `__init__` does not match documented default
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the standard "`None` means 1 unless in a :obj:`joblib.parallel_backend` context" contract, but sklearn/cluster/_hdbscan/hdbscan.py:658 sets the constructor default to `n_jobs=4`, and the `_parameter_constraints` at sklearn/cluster/_hdbscan/hdbscan.py:640 permits `[Integral, None]`.
scenario: "user instantiates `HDBSCAN()` expecting the documented single-thread default → estimator silently spawns 4 parallel workers on every pairwise-distance call, ignoring any `joblib.parallel_backend` context the caller set up"
contract: Set the `__init__` default to `n_jobs=None` so the runtime behavior matches the documented contract (and matches DBSCAN/OPTICS/AffinityPropagation defaults).
instances: single-instance

### F2 — `_hierarchical_fast` pxd declares `noexcept` but the pyx definitions omit it
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, but sklearn/cluster/_hierarchical_fast.pyx:331 and sklearn/cluster/_hierarchical_fast.pyx:339 define these methods as `cdef void union(self, intp_t m, intp_t n):` and `cdef intp_t fast_find(self, intp_t n):` — no `noexcept`.
scenario: "cluster module compiled under Cython 3.x (which enforces .pxd/.pyx signature parity for exception specifiers) → compilation fails with a signature-mismatch error, blocking the build on the next-supported Cython"
contract: Add `noexcept` to both method definitions in `_hierarchical_fast.pyx` so the implementation signature matches the newly extracted `.pxd` declaration.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F3 — HDBSCAN test module filename typo prevents targeted `--pyargs` selection
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — the file is named `test_reachibility.py` (misspelling of "reachability"); the module it tests is `sklearn/cluster/_hdbscan/_reachability.pyx:1` (spelled correctly).
scenario: "developer runs `pytest --pyargs sklearn.cluster._hdbscan.tests.test_reachability` (matching the source module name) → collection error because the file is actually `test_reachibility`; targeted CI/test invocations that use the correct spelling silently miss these tests"
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py` so the test module name matches the module under test.
instances: single-instance
