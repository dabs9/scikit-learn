### F1 — UnionFind's private cdef state is leaked through .pxd to enable direct cross-module field access
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:1-9 — a new `.pxd` was created solely to publish `cdef intp_t next_label`, `cdef intp_t[:] parent`, `cdef intp_t[:] size` (previously declared inline in `_hierarchical_fast.pyx` and therefore invisible to other Cython modules), and sklearn/cluster/_hdbscan/_linkage.pyx:265 reaches directly into that state with `U.size[current_node_cluster] + U.size[next_node_cluster]` after `cimport UnionFind` at sklearn/cluster/_hdbscan/_linkage.pyx:39.
scenario: "any future refactor of `UnionFind`'s storage (e.g. packing `parent`/`size` into a single struct, moving off memoryviews, or making `size` lazy) → silently breaks `_hdbscan/_linkage.pyx::make_single_linkage`, because the sibling submodule now depends on the concrete memoryview layout rather than a documented interface."
contract: keep the `.pxd` limited to the two methods (`union`, `fast_find`) and have `union` return the newly-merged cluster's size (or add a `cdef intp_t get_size(intp_t label) noexcept`); remove the `next_label`/`parent`/`size` declarations from the `.pxd` so they stay private to `_hierarchical_fast.pyx`.
instances: [sklearn/cluster/_hierarchical_fast.pxd:4-6, sklearn/cluster/_hdbscan/_linkage.pyx:265]

### F2 — Split test placement: one estimator's tests live in two directories
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1-64 — reachability tests live inside a nested `_hdbscan/tests/` package; sklearn/cluster/tests/test_hdbscan.py:1-533 — the estimator's main tests live in the sibling `cluster/tests/` directory. Every other clusterer (DBSCAN, OPTICS, Birch, KMeans, Bicluster, MeanShift, hierarchical, spectral, AffinityPropagation) has all of its tests under `sklearn/cluster/tests/` — this PR is the only clusterer that splits its tests across two locations.
scenario: "contributor adding a test for another `_hdbscan/*.pyx` file (e.g. `_linkage`, `_tree`) → has no canonical location to write it; over time the split leads to duplicate coverage or forgotten tests in the less-visible nested directory."
contract: relocate `sklearn/cluster/_hdbscan/tests/test_reachibility.py` to `sklearn/cluster/tests/test_reachability.py` and delete the `sklearn/cluster/_hdbscan/tests/` package so that all HDBSCAN tests live next to every other clusterer's tests.
instances: [sklearn/cluster/_hdbscan/tests/test_reachibility.py:1, sklearn/cluster/_hdbscan/tests/__init__.py:1]

### F3 — HIERARCHY-format tree remapping lives in the estimator file, not with the dtype it manipulates
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:351-393 — `remap_single_linkage_tree` reads/writes `tree[i]["left_node"]`, `tree[i]["right_node"]`, `tree[i]["cluster_size"]` and allocates `np.zeros(len(non_finite), dtype=HIERARCHY_dtype)`, i.e. pure structured-array manipulation of the tree format whose dtype and Cython struct are owned by sklearn/cluster/_hdbscan/_tree.pyx:42-46 (`HIERARCHY_dtype`) and sklearn/cluster/_hdbscan/_tree.pxd:34-38 (`HIERARCHY_t`). The dtype is imported back into the estimator at sklearn/cluster/_hdbscan/hdbscan.py:58.
scenario: "HIERARCHY layout gains or renames a field in `_tree.pyx`/`_tree.pxd` → maintainer updates the tree module and its callers (`_linkage.pyx`, `_condense_tree`, `tree_to_labels`) but overlooks `remap_single_linkage_tree` because it is out-of-module; result: silent producer/consumer divergence on the tree format."
contract: move `remap_single_linkage_tree` into `sklearn/cluster/_hdbscan/_tree.pyx` (next to `HIERARCHY_dtype` and the other tree-format functions) and call it from `hdbscan.py`; keep `hdbscan.py` free of raw HIERARCHY structured-array indexing.
instances: single-instance

### F4 — Sibling `_tree` module addressed via a fully-qualified upward path instead of a sibling-relative import
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:40-41 — `from ...cluster._hdbscan._tree cimport HIERARCHY_t` and `from ...cluster._hdbscan._tree import HIERARCHY_dtype` climb up three packages to `sklearn/` and descend back into the *same* subpackage `_hdbscan/` where `_linkage.pyx` already lives; the natural sibling form is `from ._tree cimport HIERARCHY_t`. The very next line, sklearn/cluster/_hdbscan/_linkage.pyx:42, uses that idiomatic form for `_typedefs`.
scenario: "`_hdbscan/` is renamed or moved as a package (a plausible follow-up given the `_hdbscan/tests/` split above) → the fully-qualified sibling import breaks even though the two files are still in the same directory, because it hard-codes the outer package layout."
contract: use `from ._tree cimport HIERARCHY_t` / `from ._tree import HIERARCHY_dtype` for sibling references and `from .._hierarchical_fast cimport UnionFind` for the parent-package reference.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:39, sklearn/cluster/_hdbscan/_linkage.pyx:40, sklearn/cluster/_hdbscan/_linkage.pyx:41]

### F5 — HDBSCAN default `n_jobs=4` contradicts its own docstring and every other sklearn estimator [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `__init__` signature has `n_jobs=4`, while sklearn/cluster/_hdbscan/hdbscan.py:486-491 documents `n_jobs : int, default=None` with the standard "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context" wording. Every other clusterer's default is `n_jobs=None` (sklearn/cluster/_dbscan.py:36,330; sklearn/cluster/_optics.py:274; sklearn/cluster/_spectral.py:633; sklearn/cluster/_mean_shift.py:41,132,427), and the private helpers in this same file (sklearn/cluster/_hdbscan/hdbscan.py:163, 276) also default to `None`.
scenario: "user runs `HDBSCAN().fit(X)` inside a joblib parallel context or a CPU-capped environment → the estimator silently pins 4 workers regardless of the surrounding backend / `os.sched_getaffinity`, violating the documented 'one job unless in `parallel_backend`' contract and breaking joblib nesting."
contract: change the constructor default to `n_jobs=None` to match both the docstring and the sklearn-wide convention.
instances: single-instance
