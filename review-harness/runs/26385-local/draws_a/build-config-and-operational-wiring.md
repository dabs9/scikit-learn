Now let me verify hdbscan tests directory `__init__.py` and `test_reachibility.py` typo. Since pytest discovers by `test_*.py`, this file will still run. Typo is cosmetic/naming.

I have enough evidence for findings.

### F1 — `.pxd` declares `noexcept` but `.pyx` implementation omits it, causing Cython 3 signature mismatch
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`, while sklearn/cluster/_hierarchical_fast.pyx:331 and :339 define them as `cdef void union(self, intp_t m, intp_t n):` and `cdef intp_t fast_find(self, intp_t n):` with no `noexcept` qualifier.
scenario: "Building with Cython 3.x → 'Function signature does not match previous declaration' warning (or hard error if `-Werror` / cython-lint-strict is enforced), and the two functions carry different exception-checking semantics depending on which declaration Cython chose."
contract: The `.pyx` method signatures for `UnionFind.union` and `UnionFind.fast_find` must be updated to include the `noexcept` qualifier so they match the newly added `.pxd` declarations exactly.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F2 — `HDBSCAN.__init__` default `n_jobs=4` contradicts its own documented `default=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with "`None` means 1 unless in a :obj:`joblib.parallel_backend` context", but the constructor at sklearn/cluster/_hdbscan/hdbscan.py:658 uses `n_jobs=4,` as the actual default.
scenario: "User instantiates `HDBSCAN()` on a shared/CI machine expecting single-threaded behavior per the docs → four parallel worker threads spawn against the pairwise distance computation, oversubscribing cores and breaking reproducibility relative to the documented contract."
contract: The `__init__` default for `n_jobs` must be `None` to match the documented `default=None` semantics used across the rest of the sklearn API (DBSCAN, OPTICS, etc.).
instances: single-instance

### F3 — HDBSCAN test module misnamed `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py exists; directory listing confirms no `test_reachability.py`. The underlying module under test is `sklearn/cluster/_hdbscan/_reachability.py[x]` (correctly spelled) — see the import at sklearn/cluster/_hdbscan/tests/test_reachibility.py:9 `from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph`.
scenario: "Developer greps or navigates via IDE for `test_reachability` and finds nothing → the test file for the reachability graph is invisible to expected discovery patterns, and the misspelling propagates to future PRs that reference it."
contract: Rename the file to `test_reachability.py` to match the module it tests.
instances: single-instance

### F4 — `HDBSCAN` docstring `Examples` block will fail doctest when run without a network / with a differently-shipped `load_digits`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:604-613 shows a doctest that hard-codes exact cluster labels (`array([ 2,  6, -1, ..., -1, -1, -1])`) from `HDBSCAN(min_cluster_size=20).fit(load_digits(...))`, but the default constructor also sets `n_jobs=4` (F2) which triggers parallel pairwise-distance computation whose outputs are not deterministic across BLAS thread counts. [out-of-theme]
scenario: "sklearn docs/doctest CI runs the example under a different BLAS thread configuration → labels differ slightly and the doctest fails."
contract: Pass `n_jobs=1` explicitly in the doctest invocation so the pairwise computation is single-threaded and the hard-coded labels remain deterministic across environments.
instances: single-instance
