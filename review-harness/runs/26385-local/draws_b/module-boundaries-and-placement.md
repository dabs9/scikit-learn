### F1 — HDBSCAN-specific `max_distance` smuggled through generic `metric_params` leaks into `pairwise_distances`, `NearestNeighbors`, and `DistanceMetric.get_metric`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:238-243 — `_hdbscan_brute` calls `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` BEFORE fetching `max_distance = metric_params.get("max_distance", 0.0)` at line 243 without popping the key. sklearn/cluster/_hdbscan/hdbscan.py:337 forwards the same dict verbatim as `NearestNeighbors(..., metric_params=metric_params, ...)`, and line 344 does `DistanceMetric.get_metric(metric, **metric_params)`. The error message at hdbscan.py:121 explicitly instructs users to "specify a `max_distance` in `metric_params`".
scenario: "User calls `HDBSCAN(metric='euclidean', metric_params={'max_distance': 1.0}).fit(X)` on a raw feature matrix → `pairwise_distances(..., max_distance=1.0)` raises `TypeError` because `max_distance` is not a valid pairwise-metric kwarg; alternatively `DistanceMetric.get_metric('euclidean', max_distance=1.0)` in the prims path rejects the kwarg."
contract: The HDBSCAN-specific `max_distance` option must be a distinct estimator parameter (or popped from a copied dict before forwarding) rather than smuggled through the neighbor/metric layer's generic `metric_params`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:238-243, sklearn/cluster/_hdbscan/hdbscan.py:337, sklearn/cluster/_hdbscan/hdbscan.py:344, sklearn/cluster/_hdbscan/hdbscan.py:770]

### F2 — Duplicate `cdef extern PyArray_SHAPE` declared in both `_tree.pxd` and `_linkage.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:48-49 declares `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`. `_linkage.pyx` already cimports from `_tree.pxd` (line 40 `from ...cluster._hdbscan._tree cimport HIERARCHY_t`) yet at sklearn/cluster/_hdbscan/_linkage.pyx:44-45 re-declares the identical `cdef extern from "numpy/arrayobject.h": intp_t * PyArray_SHAPE(cnp.PyArrayObject *)`.
scenario: "Future maintainer changes signature/name in one place → the other silently diverges; extern declared in a pxd should be the single source, private redeclaration in the pyx violates single-owner rule."
contract: The extern declaration must live in one location — the pxd owned by the numpy-shape utility — and every dependent pyx cimports it from there, never re-declaring locally.
instances: [sklearn/cluster/_hdbscan/_tree.pxd:48-49, sklearn/cluster/_hdbscan/_linkage.pyx:44-45]

### F3 — `CONDENSED_t` struct leaked into `_tree.pxd` despite having zero external consumers
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pxd:42-46 defines `ctypedef packed struct CONDENSED_t` in the public pxd. Grep shows every use of `CONDENSED_t` is inside `_tree.pyx` itself (17 hits, all in `_tree.pyx`); no other translation unit cimports it. Only `HIERARCHY_t` is legitimately consumed externally (by `_linkage.pyx:40`).
scenario: "Publishing an internal-only type in the pxd invites external cimports → freezes the on-disk layout of `CONDENSED_dtype` as a cross-module contract when it is in fact `_tree.pyx`-private."
contract: Types with no cross-module consumers must be defined inside the `.pyx` (or a private module-local `.pxi`), not in the shared `.pxd`.
instances: single-instance

### F4 — Module-level function `remap_single_linkage_tree` exposed without underscore prefix from the internal `_hdbscan.hdbscan` module, and placed in the estimator file rather than with the tree data it manipulates
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351 defines `def remap_single_linkage_tree(tree, internal_to_raw, non_finite):` at module top level with no underscore. It manipulates the `HIERARCHY_dtype` structured array (data owned by `_tree.pyx`) — indexing `tree[i]["left_node"]`, `tree[i]["right_node"]`, `tree[i]["cluster_size"]` — but lives in a Python file colocated with the estimator class rather than the module that owns the dtype. It is not re-exported through `sklearn.cluster.__init__` (see sklearn/cluster/__init__.py:26) yet its name suggests public API.
scenario: "Downstream code imports `sklearn.cluster._hdbscan.hdbscan.remap_single_linkage_tree` treating the un-prefixed name as public → subsequent refactor breaks it; meanwhile logic operating on `HIERARCHY_dtype` sits far from its type definition, forcing a Python-layer touch every time the tree layout changes."
contract: A helper that operates exclusively on the private `HIERARCHY_dtype` must (a) be underscore-prefixed to mark it internal and (b) live in `_tree.pyx` (or a private `_tree`-adjacent Python module) beside the dtype it mutates.
instances: single-instance

### F5 — Split test-file locations: `test_reachibility.py` under `_hdbscan/tests/` while sibling `test_hdbscan.py` sits under `sklearn/cluster/tests/`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 exists as the only file (besides empty `__init__.py`) in `sklearn/cluster/_hdbscan/tests/`, while sklearn/cluster/tests/test_hdbscan.py:1 contains the estimator tests. Every other clustering algorithm's tests (dbscan, birch, kmeans, optics, …) live only under `sklearn/cluster/tests/` — verified via `ls sklearn/cluster/tests/`. The `test_reachibility.py` filename is also misspelled ("reachibility" instead of "reachability").
scenario: "CI discovery configurations and contributors expect all cluster tests under `sklearn/cluster/tests/` → a solitary nested `tests/` package hides tests from convention-based tooling and encourages divergent import styles."
contract: All HDBSCAN unit tests must reside in a single directory (`sklearn/cluster/tests/`) to match the placement convention of every other cluster-module test.
instances: single-instance

### F6 — Empty `_hdbscan/__init__.py` forces all imports to reach across a two-level private path
severity: low
evidence: sklearn/cluster/_hdbscan/__init__.py is 0 bytes (Read reports "shorter than the provided offset (1). The file has 1 lines" and no content). Consumers must therefore write `from sklearn.cluster._hdbscan.hdbscan import HDBSCAN` (sklearn/cluster/__init__.py:26) and `from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING` (sklearn/cluster/tests/test_hdbscan.py:18), reaching into a nested private module rather than re-exporting `HDBSCAN` and any needed internals from the package's own `__init__`.
scenario: "A second Cython-backed cluster algorithm follows this pattern → deep private import paths spread throughout the tree and each internal consumer becomes coupled to the concrete filename `hdbscan.py`."
contract: The `_hdbscan/__init__.py` must re-export the estimator (`from .hdbscan import HDBSCAN`) so importers can depend on `sklearn.cluster._hdbscan` rather than `sklearn.cluster._hdbscan.hdbscan`.
instances: single-instance

### F7 — `UnionFind.union` / `UnionFind.fast_find` pxd declaration is `noexcept` but pyx implementation omits the keyword
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`. sklearn/cluster/_hierarchical_fast.pyx:331 defines `cdef void union(self, intp_t m, intp_t n):` (no `noexcept`) and sklearn/cluster/_hierarchical_fast.pyx:339 defines `cdef intp_t fast_find(self, intp_t n):` (no `noexcept`). The pxd was newly extracted in this PR (see hunks/sklearn_cluster__hierarchical_fast.pxd.diff), which lifts the field declarations out of the class body in the pyx (hunks/sklearn_cluster__hierarchical_fast.pyx.diff removes them) but silently strengthens the exception-handling contract with `noexcept` while the implementation still allows exceptions to propagate. [out-of-theme]
scenario: "A Python exception is raised inside `union`/`fast_find` at runtime (e.g. IndexError from `self.parent[m] = self.next_label` when sizes mismatch) → the `noexcept` declaration causes the exception to be swallowed with only an unraisable warning rather than propagated to the caller."
contract: The pxd declaration and the pyx implementation must carry the identical `noexcept` qualifier on both `union` and `fast_find`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:8, sklearn/cluster/_hierarchical_fast.pxd:9, sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]

### F8 — `MST_edge_t` struct is unpublished from any pxd, blocking Cython-level cross-module reuse
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:48-59 defines the `MST_edge_dtype` numpy dtype and the `MST_edge_t` packed struct at module scope with no corresponding pxd declaration (there is no `_linkage.pxd` in `sklearn/cluster/_hdbscan/`, verified via `ls sklearn/cluster/_hdbscan/`). `MST_edge_dtype` is imported at the Python level by sklearn/cluster/_hdbscan/hdbscan.py:55, but the Cython-visible `MST_edge_t` cannot be cimported by any other pyx. Meanwhile sklearn/cluster/_hdbscan/_linkage.pyx:190 inlines `mutual_reachability_distance = max(...)` inside `mst_from_data_matrix` while `_reachability.pyx` owns the mutual-reachability concept for the dense/sparse-precomputed paths.
scenario: "A future HDBSCAN Cython routine needs the MST edge struct → it must either duplicate the ctypedef or bounce through Python, because `MST_edge_t` has no pxd home."
contract: `MST_edge_t` must be published from `_linkage.pxd` (a new pxd colocated with `_linkage.pyx`) so any dependent pyx can cimport it.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:48-59]
