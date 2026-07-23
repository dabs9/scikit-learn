### F1 — `n_jobs` docstring/default mismatch: doc claims `default=None`, actual is `4`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486-490 documents `n_jobs : int, default=None ... None means 1 unless in a joblib.parallel_backend context.` but sklearn/cluster/_hdbscan/hdbscan.py:658 declares `n_jobs=4` in `__init__`.
scenario: "User relies on the documented `default=None` (join joblib backend context) → gets a fixed 4-thread pool that ignores `joblib.parallel_backend`, breaking parallel context management and diverging from every other sklearn estimator's convention"
contract: Set the constructor default to `n_jobs=None` so it matches the documented contract and the standard sklearn convention (`None` = defer to joblib context / 1 thread).
instances: single-instance

### F2 — Missing `cnp.import_array()` in `_hdbscan/_tree.pyx` and `_hdbscan/_linkage.pyx` while sibling module wires it
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:41 calls `cnp.import_array()`. sklearn/cluster/_hdbscan/_linkage.pyx:34-45 does `cimport numpy as cnp`, declares `cnp.ndarray[...]` buffers, and casts through `<cnp.PyArrayObject*>` for `PyArray_SHAPE`, but never calls `cnp.import_array()`. sklearn/cluster/_hdbscan/_tree.pyx:33 does `cimport numpy as cnp` and uses `PyArray_SHAPE(<cnp.PyArrayObject*> ...)` at lines 311, 484, 535, 571, 741, again with no `cnp.import_array()`. Existing pattern in the tree (see sklearn/cluster/_dbscan_inner.pyx:8, plus every other new-code `.pyx` that uses `cnp.` types) is to call `cnp.import_array()` at module scope.
scenario: "A future numpy C-API call (e.g. `PyArray_SimpleNew`) added to these files → segfault at import/first call because the C-API state was never initialised, while `_reachability.pyx` continues to work"
contract: Add `cnp.import_array()` at module scope in `_hdbscan/_linkage.pyx` and `_hdbscan/_tree.pyx`, matching `_reachability.pyx` and every other numpy-cimporting `.pyx` in the tree.
instances: [sklearn/cluster/_hdbscan/_linkage.pyx:34, sklearn/cluster/_hdbscan/_tree.pyx:33]

### F3 — Scale-invariance example fits HDBSCAN on unscaled `X`, contradicting the demonstration [out-of-theme]
severity: low
evidence: examples/cluster/plot_hdbscan.py:106-110 — the loop advertised as showing HDBSCAN's scale invariance calls `hdb.fit(X)` with the un-scaled data and then plots `X` (not `X * scale`); the parallel DBSCAN example immediately above (lines 87-89) correctly does `dbs.fit(X * scale)` and plots `X * scale`.
scenario: "Reader compares the two loops to see HDBSCAN handling scaled data → sees identical plots but only because the same unscaled `X` was fed in, not because HDBSCAN is scale-invariant — the intended pedagogical point is unsupported by the code"
contract: Fit and plot on the scaled data in the HDBSCAN loop, mirroring the DBSCAN loop: replace `hdb.fit(X)` / `plot(X, ...)` with `hdb.fit(X * scale)` / `plot(X * scale, ...)`.
instances: single-instance
