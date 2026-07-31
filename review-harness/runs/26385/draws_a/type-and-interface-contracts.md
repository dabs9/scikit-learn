Task tracking isn't relevant here — the deliverable is the findings document. Writing findings now.

### F1 — `n_jobs` documented default is `None` but constructor default is `4`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None` with the text "`None` means 1 unless in a :obj:`joblib.parallel_backend` context", while sklearn/cluster/_hdbscan/hdbscan.py:658 declares the parameter as `n_jobs=4` in `HDBSCAN.__init__`.
scenario: "User reads docs → constructs `HDBSCAN()` expecting single-threaded / joblib-parallel-backend-controlled execution → estimator spawns 4 worker jobs unconditionally"
contract: Change the constructor default to `n_jobs=None` to match the documented, parameter-constraint-permitted (`[Integral, None]`) contract and the sibling `_hdbscan_brute` / `_hdbscan_prims` helpers (which both default `n_jobs=None`).
instances: single-instance

### F2 — `remap_single_linkage_tree(non_finite=...)` docstring says `ndarray` (boolean array) but caller passes a `set`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:363-365 documents `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`; the sole call site at sklearn/cluster/_hdbscan/hdbscan.py:838 passes `non_finite=set(infinite_index + missing_index)` (a `set` of raw integer indices). The body at hdbscan.py:369, 383 relies on `len(non_finite)` (count of unique indices) and hdbscan.py:388 iterates `enumerate(non_finite)` treating each element as a raw sample index — behavior that would silently give the wrong `outlier_tree` if a caller followed the docstring and passed a boolean mask.
scenario: "External caller (or future maintainer) reads the docstring, passes a boolean ndarray of length n_samples → `len(non_finite)` returns n_samples instead of the outlier count, `outlier` in the loop iterates the booleans (0/1) instead of raw indices, producing a corrupt hierarchy"
contract: Update the docstring to state `non_finite : set of int  A set of raw-data row indices that are non-finite (union of infinite and missing indices)`, and rename the parameter or type-annotate accordingly so the declared type matches what the function actually consumes.
instances: single-instance

### F3 — Length-1 ndarray coerced to scalar / used in `if` context in `_do_labelling` and `traverse_upwards`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:498 assigns `parent_lambda = lambda_array[child_array == n]` (an ndarray, not cdef-typed) and then `_tree.pyx:505` writes `if parent_lambda >= threshold:` — a length-1 boolean-array truth test. sklearn/cluster/_hdbscan/_tree.pyx:582 declares `cdef cnp.intp_t root, parent` and `_tree.pyx:586` assigns `parent = cluster_tree[cluster_tree['child'] == leaf]['parent']` (an ndarray) into the scalar; `_tree.pyx:593` likewise assigns `parent_eps = 1 / cluster_tree[cluster_tree['child'] == parent]['value']` (an ndarray) into `cdef cnp.float64_t parent_eps`. The comment at `_tree.pyx:496-497` acknowledges the pattern relies on the boolean mask matching exactly one row.
scenario: "NumPy ≥ 1.25 deprecates length-1-ndarray-to-scalar coercion → these coercions raise `DeprecationWarning` today and will raise `TypeError` in a future NumPy, breaking `_do_labelling`/`traverse_upwards` at runtime; if the invariant ever fails (e.g. duplicate child edge) the length-N array coercion raises `ValueError` with no diagnostic hinting at the real cause"
contract: Extract the scalar explicitly at each site — replace `lambda_array[child_array == n]` / `cluster_tree[...][field]` with `...[0]` (or `.item()`) so the RHS is a genuine scalar, mirroring the existing correct pattern at `_tree.pyx:621` (`eps = 1 / distances[leaf_nodes][0]`).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:498, sklearn/cluster/_hdbscan/_tree.pyx:505, sklearn/cluster/_hdbscan/_tree.pyx:586, sklearn/cluster/_hdbscan/_tree.pyx:593]

### F4 — `UnionFind` cdef signatures disagree between `.pxd` and `.pyx` (`noexcept` present in one, absent in the other)
severity: medium
evidence: sklearn/cluster/_hierarchical_fast.pxd:8-9 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; the corresponding definitions at sklearn/cluster/_hierarchical_fast.pyx:331 (`cdef void union(self, intp_t m, intp_t n):`) and sklearn/cluster/_hierarchical_fast.pyx:339 (`cdef intp_t fast_find(self, intp_t n):`) omit the `noexcept` specifier.
scenario: "Under Cython 3, a `cdef` function returning a value with no exception specifier defaults to propagating Python exceptions (equivalent to `except *`); the `.pxd` promises callers no exception propagation, but the definition does not opt into that guarantee → either a Cython signature-mismatch/deprecation warning at build time, or a silent semantic divergence between what the header advertises and what the body implements"
contract: Add `noexcept` to both definitions in `_hierarchical_fast.pyx` so the definition matches the header exactly (`cdef void union(self, intp_t m, intp_t n) noexcept:` and `cdef intp_t fast_find(self, intp_t n) noexcept:`).
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
