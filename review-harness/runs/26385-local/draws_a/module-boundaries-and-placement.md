### F1 — `_hdbscan` subpackage cimports from parent package's private Cython (leaks internal UnionFind cross-package)
severity: medium
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39 — `from ...cluster._hierarchical_fast cimport UnionFind`; sklearn/cluster/_hierarchical_fast.pxd:3-9 — newly extracted `.pxd` exposes `UnionFind` (with `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` and cdef methods `union`/`fast_find`) at the module boundary specifically to satisfy the new subpackage's cimport
scenario: "The `_hdbscan` subpackage needs a union-find data structure → author extracts a `.pxd` for `_hierarchical_fast` exposing UnionFind's private state (attributes and cdef methods) publicly to the whole tree, promoting an implementation-detail class of the `_agglomerative`/`_hierarchical` code path into a de facto library-wide Cython utility"
contract: Move the shared `UnionFind` cdef class into a neutral utility module (e.g. `sklearn/utils/_union_find.pxd`) and have both `_hierarchical_fast.pyx` and `_hdbscan/_linkage.pyx` cimport from it, so the sibling subpackage does not depend on internals of another cluster module
instances: [sklearn/cluster/_hierarchical_fast.pxd:3-9, sklearn/cluster/_hdbscan/_linkage.pyx:39]

### F2 — Tests split across two locations for the same estimator
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 exists (a whole new `tests/` folder inside the private subpackage) while sklearn/cluster/tests/test_hdbscan.py:1 holds all other HDBSCAN tests; every other cluster estimator keeps ALL its tests under `sklearn/cluster/tests/` (see `test_dbscan.py`, `test_hierarchical.py`, `test_optics.py`, etc.) and no other cluster estimator has an internal `tests/` folder
scenario: "Contributor updates HDBSCAN reachability logic → searches `sklearn/cluster/tests/test_hdbscan.py` for related tests → does not find the reachability tests, which live in the nested `_hdbscan/tests/test_reachibility.py`, so related tests are easily missed / duplicated"
contract: Consolidate all HDBSCAN tests under `sklearn/cluster/tests/` (e.g. merge `test_reachibility.py` content into a `test_reachability.py` there) following the convention used by every other estimator in `sklearn.cluster`
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/tests/__init__.py:1, sklearn/cluster/tests/test_hdbscan.py:1]

### F3 — Public-looking helper `remap_single_linkage_tree` at module scope in `hdbscan.py`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 — `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):` is defined without the leading underscore that every other helper in this file uses (`_brute_mst` L82, `_process_mst` L135, `_hdbscan_brute` L158, `_hdbscan_prims` L269, `_get_finite_row_indices` L396, `_OUTLIER_ENCODING` L65); only used internally at L834
scenario: "The unmarked helper is discoverable via `sklearn.cluster._hdbscan.hdbscan.remap_single_linkage_tree` and looks like a public API → later refactoring assumes external consumers may exist and hesitates to change/remove it, locking in an internal helper's signature"
contract: Rename to `_remap_single_linkage_tree` to conform to the module's own convention for private helpers
instances: single-instance

### F4 — `_linkage.pyx` reaches `_tree` symbols via absolute cross-package path instead of local sibling import
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:40-41 — `from ...cluster._hdbscan._tree cimport HIERARCHY_t` and `from ...cluster._hdbscan._tree import HIERARCHY_dtype`; the same file imports `_typedefs` relatively but reroutes its own sibling `_tree` up three package levels and back down
scenario: "Package rename or relocation of `_hdbscan` → the absolute path `...cluster._hdbscan._tree` breaks, while a local `from ._tree cimport ...` would remain valid; also obscures the sibling relationship at read time"
contract: Import the sibling module locally: `from ._tree cimport HIERARCHY_t` and `from ._tree import HIERARCHY_dtype`, matching how `hdbscan.py` (L57-58) imports the same module
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F5 — `.pxd` declares `noexcept` for UnionFind methods but `.pyx` implementation omits it [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 — `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331 — `cdef void union(self, intp_t m, intp_t n):` (no `noexcept`); sklearn/cluster/_hierarchical_fast.pyx:339 — `cdef intp_t fast_find(self, intp_t n):` (no `noexcept`)
scenario: "Cython 3+ enforces matching exception-handling declarations between `.pxd` and `.pyx` → build fails or emits a signature-mismatch warning; even where it compiles, callers that cimport UnionFind will assume `noexcept` semantics that the implementation does not guarantee, changing exception propagation for downstream users of the newly-exposed class"
contract: Add `noexcept` to both method definitions in `_hierarchical_fast.pyx` so the implementation matches the `.pxd` declaration
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F6 — Test file name typo `test_reachibility.py` misspells "reachability"
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename `test_reachibility.py` while the module under test is `_reachability.py` (also see docstring at line 12: "mutual reachability graph")
scenario: "Developer runs `pytest -k reachability` looking for the tests of `_reachability.py` → the misspelt file is not matched by the intuitive query and its tests are skipped/undiscovered"
contract: Rename the file to `test_reachability.py` to match the module it tests
instances: single-instance
