### F1 — HDBSCAN reaches into private `_dist_metrics` module for `DistanceMetric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:46 — `from ...metrics._dist_metrics import DistanceMetric`, while `sklearn/metrics/__init__.py:40` publicly re-exports `DistanceMetric` and every other `sklearn.cluster` Python file uses the public path (e.g. `sklearn/cluster/_agglomerative.py:21` — `from ..metrics import DistanceMetric`; the class docstrings themselves advertise `from sklearn.metrics import DistanceMetric`).
scenario: "downstream refactor renames/relocates `_dist_metrics` → HDBSCAN's import breaks even though the public `sklearn.metrics.DistanceMetric` surface is unchanged"
contract: Import `DistanceMetric` in `hdbscan.py` via the public entry point (`from ...metrics import DistanceMetric`), matching the convention used by every other Python file in `sklearn.cluster`.
instances: single-instance

### F2 — `_linkage.pyx` imports its own sibling via a top-level round-trip path
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39-41 — `from ...cluster._hierarchical_fast cimport UnionFind`, `from ...cluster._hdbscan._tree cimport HIERARCHY_t`, `from ...cluster._hdbscan._tree import HIERARCHY_dtype`. The last two go up three levels to `sklearn` and back down two into the file's own package. The peer file `sklearn/cluster/_hdbscan/hdbscan.py:58` uses the correct relative form `from ._tree import HIERARCHY_dtype`, and elsewhere in `sklearn.cluster` sibling Cython files use `from ._sibling cimport ...` (e.g. `sklearn/cluster/_k_means_lloyd.pyx:22-24`).
scenario: "rename or move of the `_hdbscan` subpackage → self-referencing `...cluster._hdbscan.*` imports break even though the sibling relationships are unchanged; also masks that `_tree` is a sibling of `_linkage`, not an external dependency"
contract: Rewrite the three sibling-facing lines in `_linkage.pyx` as `from ._tree cimport HIERARCHY_t`, `from ._tree import HIERARCHY_dtype`, and `from .._hierarchical_fast cimport UnionFind`, matching the sibling-import convention used elsewhere in `sklearn.cluster`.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:39, sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F3 — Cross-submodule reach into `UnionFind` internal state via newly-published `.pxd`
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:1-15 (new file) publicizes `UnionFind`'s previously private data layout: `next_label`, `parent`, `size`, plus `union`/`fast_find` methods. These fields were declared inline inside `sklearn/cluster/_hierarchical_fast.pyx:322-329` before this PR (see the diff at `hunks/sklearn_cluster__hierarchical_fast.pyx.diff:5-12` which removes them). `sklearn/cluster/_hdbscan/_linkage.pyx:266` directly indexes into the internal state — `U.size[current_node_cluster] + U.size[next_node_cluster]` — coupling the HDBSCAN linkage code to a struct-layout detail of a module whose name (`_hierarchical_fast`) advertises agglomerative clustering.
scenario: "future change to `UnionFind`'s size representation (e.g., lazy computation, weighted variant, dtype change) → silently breaks `_hdbscan/_linkage.pyx` because a supposedly private field is now a load-bearing part of the cross-module interface"
contract: Either add a narrow accessor method (`cdef intp_t size_of(self, intp_t idx) noexcept`) to the `.pxd` and have `_linkage.pyx` call it instead of reading `U.size[...]` directly, or relocate `UnionFind` to a neutral shared module (e.g. `sklearn/cluster/_union_find.{pxd,pyx}`) — leave `_hierarchical_fast.pxd` scoped to agglomerative-specific surface.
instances: [sklearn/cluster/_hierarchical_fast.pxd:3-8, sklearn/cluster/_hdbscan/_linkage.pyx:266]

### F4 — `remap_single_linkage_tree` lacks underscore prefix despite being an internal helper
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 — `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):`. It is called only from `hdbscan.py:834` inside `HDBSCAN.fit`, and is not re-exported anywhere (`sklearn/cluster/__init__.py:26` re-exports only `HDBSCAN` from this file). Every other private helper in the same file uses the underscore convention: `_brute_mst`, `_process_mst`, `_hdbscan_brute`, `_hdbscan_prims`, `_get_finite_row_indices`.
scenario: "future contributor treats `remap_single_linkage_tree` as public because of its unadorned name → downstream code depends on it → the module's private-vs-public boundary is quietly widened without review"
contract: Rename to `_remap_single_linkage_tree` and update the call site at `hdbscan.py:834` accordingly.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:351, sklearn/cluster/_hdbscan/hdbscan.py:834]

### F5 — Test module for `_reachability.pyx` is misnamed `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename contains "reachibility"; the target module `sklearn/cluster/_hdbscan/_reachability.pyx:1` and its function `mutual_reachability_graph` (imported at `sklearn/cluster/_hdbscan/tests/test_reachibility.py:15`) spell it "reachability".
scenario: "developer searches for `test_reachability` (correct spelling) via file finder / grep-by-basename → finds no matches and assumes the module is untested; discoverability of the test file relies on knowing about the typo"
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py`.
instances: single-instance

### F6 — `PyArray_SHAPE` numpy internal declared in `_tree.pxd` public surface instead of `_tree.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. No file outside `_tree.pyx` cimports this symbol from `_tree.pxd` (verified by grep — only uses are inside `_tree.pyx`), and the peer file that needs the same extern re-declares its own copy locally (`sklearn/cluster/_hdbscan/_linkage.pyx:44-45`), confirming the .pxd declaration is not being reused externally.
scenario: "numpy header layout changes or the sklearn build stops assuming numpy internals → the `.pxd` public surface breaks external cimports that never needed the symbol; also invites future cimports of a numpy-C-API primitive as if it were part of the module's contract"
contract: Move the `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` block from `_tree.pxd` into `_tree.pyx` so it remains a private compilation-unit helper; the `_linkage.pyx` local copy is fine to keep for its own use.
instances: single-instance
