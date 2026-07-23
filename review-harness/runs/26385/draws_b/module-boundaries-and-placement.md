### F1 — HDBSCAN tests split across two directories with different conventions
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1-64 — a full second test directory (`sklearn/cluster/_hdbscan/tests/` with its own `__init__.py`) holds `test_reachibility.py`, while every other HDBSCAN test lives at `sklearn/cluster/tests/test_hdbscan.py:1-533`. No other cluster submodule in the repo (see `sklearn/cluster/` layout — only `sklearn/cluster/tests/` exists) creates a nested `tests/` folder under a private subpackage.
scenario: "contributors adding HDBSCAN tests → randomly pick one of two directories, silently creating duplicate coverage or losing tests" 
contract: Consolidate all HDBSCAN tests under `sklearn/cluster/tests/` (rename the reachability file to `test_reachability.py` at the same time to fix the misspelling) and delete `sklearn/cluster/_hdbscan/tests/` entirely — matching the convention used for `_kmeans`, `_agglomerative`, `_optics`, etc.
instances: [sklearn/cluster/_hdbscan/tests/__init__.py:1, sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/tests/test_hdbscan.py:1]

### F2 — `UnionFind` private attributes leaked across module boundary via new `.pxd`
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:1-16 — the new declaration file promotes what were previously `cdef intp_t next_label` / `cdef intp_t[:] parent` / `cdef intp_t[:] size` (private class body attributes at `_hierarchical_fast.pyx:324-326`, now deleted per the diff) into cross-module-visible fields. `_linkage.pyx:266` then reads `U.size[current_node_cluster] + U.size[next_node_cluster]` — a foreign module reaching directly into another class's internal state to compute cluster size, instead of asking the owner.
scenario: "future change to how `UnionFind` tracks component sizes (e.g. rank-based instead of size-tracking) → silently breaks `make_single_linkage` because a private field it was reading is gone/renamed"
contract: Keep `UnionFind`'s attributes private to `_hierarchical_fast.pyx` and expose a `cdef intp_t component_size(intp_t x) noexcept` (or equivalent) accessor. Have `_linkage.pyx` call the accessor rather than indexing `U.size` directly; the `.pxd` should declare only the union/find/size accessor methods, not the raw arrays.
instances: [sklearn/cluster/_hierarchical_fast.pxd:11-13, sklearn/cluster/_hdbscan/_linkage.pyx:266]

### F3 — `PyArray_SHAPE` cdef extern block duplicated instead of sharing the sibling `.pxd`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`; sklearn/cluster/_hdbscan/_linkage.pyx:44-45 re-declares the identical block instead of picking it up from the neighbouring `.pxd` (which it already cimports on line 40).
scenario: "future update of the extern signature (e.g. to `npy_intp` after NumPy 2.x cleanup) → edit propagates to one site, the other silently keeps the old signature and hides a build/type mismatch"
contract: Declare `PyArray_SHAPE` exactly once — in `_tree.pxd` (or a dedicated `_numpy_helpers.pxd`) — and remove the duplicate block from `_linkage.pyx`; the file already cimports from `_tree` and can share the declaration.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:44-45, sklearn/cluster/_hdbscan/_tree.pxd:48-49]

### F4 — Public-named helpers `remap_single_linkage_tree` / `_get_finite_row_indices` inconsistently marked
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 defines `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):` with no leading underscore, and sklearn/cluster/_hdbscan/hdbscan.py:396 defines `def _get_finite_row_indices(matrix):` with one. Both are module-level helpers used only within `hdbscan.py` (grep confirms zero external callers). The public-looking name of the first function suggests it is part of the module's API when it is not.
scenario: "downstream code / docs treats `remap_single_linkage_tree` as public API → future refactor renames or signature-changes it and silently breaks external users"
contract: Rename `remap_single_linkage_tree` to `_remap_single_linkage_tree` so both internal helpers follow the same underscore convention and neither leaks from the private `_hdbscan` submodule.
instances: single-instance

### F5 — `sklearn/cluster/_hdbscan/hdbscan.py` re-uses the package's own name for its main module
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:1 — the module is named `hdbscan.py` inside a package also called `_hdbscan`, forcing the awkward import path `from ._hdbscan.hdbscan import HDBSCAN` (sklearn/cluster/__init__.py:26) and `from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING` (sklearn/cluster/tests/test_hdbscan.py:18). Sibling submodules such as `_kmeans.py`, `_optics.py`, `_birch.py` place their estimator directly in the parent package instead of nesting a same-named file.
scenario: "reader/tooling encounters `_hdbscan.hdbscan` → confusion about which layer owns which behavior; typing `from sklearn.cluster._hdbscan import HDBSCAN` (a natural attempt) fails because the class lives one level deeper"
contract: Move the estimator's contents from `sklearn/cluster/_hdbscan/hdbscan.py` up to `sklearn/cluster/_hdbscan/__init__.py`, so `HDBSCAN` is reachable as `sklearn.cluster._hdbscan.HDBSCAN`; update the re-export in `sklearn/cluster/__init__.py` accordingly.
instances: single-instance
