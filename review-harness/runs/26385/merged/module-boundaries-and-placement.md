### F1 — HDBSCAN default `n_jobs=4` contradicts its own docstring and every other sklearn estimator [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature has `n_jobs=4`, while sklearn/cluster/_hdbscan/hdbscan.py:486-491 documents `n_jobs : int, default=None` with the standard "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context" wording. Every other clusterer's default is `n_jobs=None` (sklearn/cluster/_dbscan.py:36,330; sklearn/cluster/_optics.py:274; sklearn/cluster/_spectral.py:633; sklearn/cluster/_mean_shift.py:41,132,427), and the private helpers in this same file (sklearn/cluster/_hdbscan/hdbscan.py:163, 276) also default to `None`.
scenario: "user runs `HDBSCAN().fit(X)` inside a joblib parallel context or a CPU-capped environment → the estimator silently pins 4 workers regardless of the surrounding backend / `os.sched_getaffinity`, violating the documented 'one job unless in `parallel_backend`' contract and breaking joblib nesting."
contract: change the constructor default to `n_jobs=None` to match both the docstring and the sklearn-wide convention.
instances: single-instance

### F2 — UnionFind's private cdef state is leaked through .pxd to enable direct cross-module field access
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:1-9 — a new `.pxd` was created solely to publish `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` (previously declared inline in `_hierarchical_fast.pyx` and therefore invisible to other Cython modules), and sklearn/cluster/_hdbscan/_linkage.pyx:265 reaches directly into that state with `U.size[current_node_cluster] + U.size[next_node_cluster]` after `cimport UnionFind` at sklearn/cluster/_hdbscan/_linkage.pyx:39.
scenario: "any future refactor of `UnionFind`'s storage (e.g. packing `parent`/`size` into a single struct, moving off memoryviews, or making `size` lazy) → silently breaks `_hdbscan/_linkage.pyx::make_single_linkage`, because the sibling submodule now depends on the concrete memoryview layout rather than a documented interface."
contract: keep the `.pxd` limited to the two methods (`union`, `fast_find`) and have `union` return the newly-merged cluster's size (or add a `cdef intp_t get_size(intp_t label) noexcept`); remove the `next_label`/`parent`/`size` declarations from the `.pxd` so they stay private to `_hierarchical_fast.pyx`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:4-6, sklearn/cluster/_hdbscan/_linkage.pyx:265, sklearn/cluster/_hierarchical_fast.pxd:3-8, sklearn/cluster/_hdbscan/_linkage.pyx:266]

### F3 — `_linkage.pyx` imports its own sibling via a top-level round-trip path
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:39-41 — `from ...cluster._hierarchical_fast cimport UnionFind`, `from ...cluster._hdbscan._tree cimport HIERARCHY_t`, `from ...cluster._hdbscan._tree import HIERARCHY_dtype`. The last two go up three levels to `sklearn` and back down two into the file's own package. The peer file `sklearn/cluster/_hdbscan/hdbscan.py:58` uses the correct relative form `from ._tree import HIERARCHY_dtype`, and elsewhere in `sklearn.cluster` sibling Cython files use `from ._sibling cimport ...` (e.g. `sklearn/cluster/_k_means_lloyd.pyx:22-24`).
scenario: "rename or move of the `_hdbscan` subpackage → self-referencing `...cluster._hdbscan.*` imports break even though the sibling relationships are unchanged; also masks that `_tree` is a sibling of `_linkage`, not an external dependency"
contract: Rewrite the three sibling-facing lines in `_linkage.pyx` as `from ._tree cimport HIERARCHY_t`, `from ._tree import HIERARCHY_dtype`, and `from .._hierarchical_fast cimport UnionFind`, matching the sibling-import convention used elsewhere in `sklearn.cluster`.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:39, sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F4 — HIERARCHY-format tree remapping lives in the estimator file, not with the dtype it manipulates
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-393 — `remap_single_linkage_tree` reads/writes `tree[i]["left_node"]`, `tree[i]["right_node"]`, `tree[i]["cluster_size"]` and allocates `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)`, i.e. pure structured-array manipulation of the tree format whose dtype and Cython struct are owned by sklearn/cluster/_hdbscan/_tree.pyx:42-46 (`HIERARCHY_dtype`) and sklearn/cluster/_hdbscan/_tree.pxd:34-38 (`HIERARCHY_t`). The dtype is imported back into the estimator at sklearn/cluster/_hdbscan/hdbscan.py:58.
scenario: "HIERARCHY layout gains or renames a field in `_tree.pyx`/`_tree.pxd` → maintainer updates the tree module and its callers (`_linkage.pyx`, `_condense_tree`, `tree_to_labels`) but overlooks `remap_single_linkage_tree` because it is out-of-module; result: silent producer/consumer divergence on the tree format."
contract: move `remap_single_linkage_tree` into `sklearn/cluster/_hdbscan/_tree.pyx` (next to `HIERARCHY_dtype` and the other tree-format functions) and call it from `hdbscan.py`; keep `hdbscan.py` free of raw HIERARCHY structured-array indexing.
instances: single-instance

### F5 — Split test placement: one estimator's tests live in two directories
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1-64 — reachability tests live inside a nested `_hdbscan/tests/` package; sklearn/cluster/tests/test_hdbscan.py:1-533 — the estimator's main tests live in the sibling `cluster/tests/` directory. Every other clusterer (DBSCAN, OPTICS, Birch, KMeans, Bicluster, MeanShift, hierarchical, spectral, AffinityPropagation) has all of its tests under `sklearn/cluster/tests/` — this PR is the only clusterer that splits its tests across two locations.
scenario: "contributor adding a test for another `_hdbscan/*.pyx` file (e.g. `_linkage`, `_tree`) → has no canonical location to write it; over time the split leads to duplicate coverage or forgotten tests in the less-visible nested directory."
contract: relocate `sklearn/cluster/_hdbscan/tests/test_reachibility.py` to `sklearn/cluster/tests/test_reachability.py` and delete the `sklearn/cluster/_hdbscan/tests/` package so that all HDBSCAN tests live next to every other clusterer's tests.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/tests/__init__.py:1]

### F6 — HDBSCAN reaches into private `_dist_metrics` module for `DistanceMetric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:46 — `from ...metrics._dist_metrics import DistanceMetric`, while `sklearn/metrics/__init__.py:40` publicly re-exports `DistanceMetric` and every other `sklearn.cluster` Python file uses the public path (e.g. `sklearn/cluster/_agglomerative.py:21` — `from ..metrics import DistanceMetric`; the class docstrings themselves advertise `from sklearn.metrics import DistanceMetric`).
scenario: "downstream refactor renames/relocates `_dist_metrics` → HDBSCAN's import breaks even though the public `sklearn.metrics.DistanceMetric` surface is unchanged"
contract: Import `DistanceMetric` in `hdbscan.py` via the public entry point (`from ...metrics import DistanceMetric`), matching the convention used by every other Python file in `sklearn.cluster`.
instances: single-instance

### F7 — `remap_single_linkage_tree` lacks underscore prefix despite being an internal helper
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 — `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):`. It is called only from `hdbscan.py:834` inside `HDBSCAN.fit`, and is not re-exported anywhere (`sklearn/cluster/__init__.py:26` re-exports only `HDBSCAN` from this file). Every other private helper in the same file uses the underscore convention: `_brute_mst`, `_process_mst`, `_hdbscan_brute`, `_hdbscan_prims`, `_get_finite_row_indices`.
scenario: "future contributor treats `remap_single_linkage_tree` as public because of its unadorned name → downstream code depends on it → the module's private-vs-public boundary is quietly widened without review"
contract: Rename to `_remap_single_linkage_tree` and update the call site at `hdbscan.py:834` accordingly.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:351, sklearn/cluster/_hdbscan/hdbscan.py:834]

### F8 — Test module for `_reachability.pyx` is misnamed `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename contains "reachibility"; the target module `sklearn/cluster/_hdbscan/_reachability.pyx:1` and its function `mutual_reachability_graph` (imported at `sklearn/cluster/_hdbscan/tests/test_reachibility.py:15`) spell it "reachability".
scenario: "developer searches for `test_reachability` (correct spelling) via file finder / grep-by-basename → finds no matches and assumes the module is untested; discoverability of the test file relies on knowing about the typo"
contract: Rename the file to `sklearn/cluster/_hdbscan/tests/test_reachability.py`.
instances: single-instance

### F9 — `PyArray_SHAPE` numpy internal declared in `_tree.pxd` public surface instead of `_tree.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 — `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. No file outside `_tree.pyx` cimports this symbol from `_tree.pxd` (verified by grep — only uses are inside `_tree.pyx`), and the peer file that needs the same extern re-declares its own copy locally (`sklearn/cluster/_hdbscan/_linkage.pyx:44-45`), confirming the .pxd declaration is not being reused externally.
scenario: "numpy header layout changes or the sklearn build stops assuming numpy internals → the `.pxd` public surface breaks external cimports that never needed the symbol; also invites future cimports of a numpy-C-API primitive as if it were part of the module's contract"
contract: Move the `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)` block from `_tree.pxd` into `_tree.pyx` so it remains a private compilation-unit helper; the `_linkage.pyx` local copy is fine to keep for its own use.
instances: single-instance
