### F1 — `UnionFind` internal fields exposed to cross-module callers via new `.pxd`
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:1-9 — the new pxd declares `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` (all storage fields, not just methods). `sklearn/cluster/_hdbscan/_linkage.pyx:266` then reaches into `U.size[current_node_cluster] + U.size[next_node_cluster]` from a sibling subpackage, treating the union-find's backing memoryview as public API.
scenario: "any future refactor of UnionFind's storage layout (e.g. changing `size` to lazy computation, renaming, or replacing with a scalar counter) → silent breakage of `_hdbscan/_linkage.pyx`'s `make_single_linkage` which depends on the field's shape and semantics"
contract: Extend `UnionFind` with a `cdef intp_t get_size(self, intp_t cluster) noexcept` accessor (and any other required accessors) declared in the pxd, and remove `next_label`/`parent`/`size` from the pxd so they stay module-private; `_hdbscan/_linkage.pyx` must not read backing arrays of a foreign class.
instances: single-instance

### F2 — Module-internal helper `remap_single_linkage_tree` exported as public API
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 — `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):` is defined without an underscore prefix, unlike every other helper in the same file (`_brute_mst`, `_process_mst`, `_hdbscan_brute`, `_hdbscan_prims`, `_get_finite_row_indices`, `_weighted_cluster_center`, `_OUTLIER_ENCODING`). Its only call site is `hdbscan.py:834` inside `HDBSCAN.fit`; no external module or test imports it.
scenario: "downstream users / IDE autocomplete pick up `sklearn.cluster._hdbscan.hdbscan.remap_single_linkage_tree` as a public helper → the maintainers cannot change its signature or delete it without a public-API deprecation cycle"
contract: Rename to `_remap_single_linkage_tree` to match the module's underscore-prefix convention for internal helpers.
instances: single-instance

### F3 — Self-referential absolute (cimport|import) paths from inside `_hdbscan`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:38-42 — `from ...metrics._dist_metrics cimport DistanceMetric`, `from ...cluster._hierarchical_fast cimport UnionFind`, `from ...cluster._hdbscan._tree cimport HIERARCHY_t`, `from ...cluster._hdbscan._tree import HIERARCHY_dtype`. The last two walk three levels up only to walk back down into the same `_hdbscan` subpackage, when `from ._tree cimport HIERARCHY_t` / `from ._tree import HIERARCHY_dtype` and `from .._hierarchical_fast cimport UnionFind` describe the boundary directly. The peer `.py` module uses relative imports for the same targets (`hdbscan.py:50-58`: `from ._reachability`, `from ._linkage`, `from ._tree`).
scenario: "the `_hdbscan` subpackage is moved / renamed → these absolute-path imports break, whereas relative-path imports would follow the move; also the direction of the dependency edge is obscured in code review"
contract: Rewrite the four imports at `_linkage.pyx:38-42` to relative form: `from ..._metrics._dist_metrics cimport ...` → `from ...metrics._dist_metrics cimport ...` is fine; but the two self-referential ones must be `from ._tree cimport HIERARCHY_t` and `from ._tree import HIERARCHY_dtype`, and the UnionFind import must be `from .._hierarchical_fast cimport UnionFind`.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:39, sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F4 — Test file misspells the module under test
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1-15 — the file is named `test_reachibility.py` (missing an 'a') while the module it exercises is `sklearn/cluster/_hdbscan/_reachability.py[x]` (spelled correctly), and the file's own imports are `from sklearn.cluster._hdbscan._reachability import mutual_reachability_graph`.
scenario: "engineer greps for `test_reachability` to locate reachability tests → nothing matches; the test file drifts out of sight; collection tooling that maps tests-to-modules by name misroutes"
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py` (matching the module).
instances: single-instance
