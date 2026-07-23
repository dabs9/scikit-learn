### F1 — Duplicate `births` allocation immediately overwrites itself
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two consecutive identical statements `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` with only a blank line between them; the first allocation is discarded before being read.
scenario: "call _compute_stability → allocate an nan-filled array, overwrite the reference with a second identical allocation → the first array is immediately garbage"
contract: Remove the leftover line 252 assignment (and the trailing blank line) so `_compute_stability` allocates `births` exactly once.
instances: single-instance

### F2 — "Scale Invariance" HDBSCAN example ignores its own scale factor
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`; `scale` is only used in the plot's parameter label — `X` is neither multiplied by `scale` on `fit` nor on `plot`, unlike the DBSCAN reference block just above which does `dbs.fit(X * scale)` and `plot(X * scale, ...)`.
scenario: "user runs the plot_hdbscan example → the three subplots labelled scale=1/0.5/3 are pixel-identical, so the narrative claim that 'HDBSCAN is scale-invariant' is not actually demonstrated"
contract: Fit and plot on the scaled data — `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, hdb.probabilities_, ...)` — matching the DBSCAN block that motivates the comparison.
instances: single-instance

### F3 — `test_hdbscan_precomputed_non_brute` no longer exercises its documented branch
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:282 uses `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"` / `"prims_balltree"`), but sklearn/cluster/_hdbscan/hdbscan.py:635-643 restricts `algorithm` via `StrOptions({"auto","brute","kdtree","balltree"})`. The `ValueError` the test asserts on comes from `_validate_params()` rejecting the string, not from the "precomputed + tree-based" incompatibility the docstring on line 279-280 advertises.
scenario: "add a `prims_kdtree` accepted alias in the future → this test would silently keep passing without ever exercising the precomputed+tree guard it claims to cover"
contract: Change the algorithm strings to the currently valid `"kdtree"` and `"balltree"` so the test actually reaches the precomputed-vs-tree code path in `fit`.
instances: single-instance

### F4 — Dead `mask = np.empty(...)` allocation in `_weighted_cluster_center`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 pre-allocates `mask = np.empty((X.shape[0],), dtype=np.bool_)`; the very next use of `mask` at line 908 is `mask = self.labels_ == idx`, which rebinds `mask` to a freshly allocated boolean array without ever reading the pre-allocated one.
scenario: "call fit with store_centers set → an extra n_samples-sized boolean buffer is allocated on every fit and never used"
contract: Delete the pre-loop `mask = np.empty(...)` initialization; `mask` is created inside the loop.
instances: single-instance
