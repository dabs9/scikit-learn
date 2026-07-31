### F1 — Duplicate `births = np.full(...)` initialization in `_compute_stability`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is written on line 252 and re-executed verbatim on line 254 (only a blank line separates them). The first call's result is immediately discarded by the second identical call.
scenario: "`_compute_stability` is called on any HDBSCAN fit → the array-allocation call runs twice per fit, once useless — a copy-paste/rebase leftover that adds noise (and a small allocation cost) to a hot code path."
contract: Delete the redundant line 252 (or 254) so `births` is initialized exactly once before the population loop.
instances: single-instance

### F2 — Dead `mst_func = None` initialization in `HDBSCAN.fit`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:763 — `mst_func = None` is placed immediately before the `if self.algorithm != "auto": ... else: ...` block. `_parameter_constraints` (lines 636-644) restricts `algorithm` to `{"auto","brute","kdtree","balltree"}`, and every branch inside both arms assigns `mst_func` to `_hdbscan_brute` or `_hdbscan_prims`; the only exits without assignment are `raise ValueError` paths. The `None` sentinel can never reach `mst_func(**kwargs)` on line 820.
scenario: "Static readers see a `None` initialization suggesting `mst_func(**kwargs)` might be invoked on `None` → misleading future maintenance; the sentinel is otherwise unreachable so cannot be exercised."
contract: Remove the `mst_func = None` line; rely on the exhaustive assignment guaranteed by `_validate_params`.
instances: single-instance

### F3 — Stale algorithm strings in `test_hdbscan_precomputed_non_brute`
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282-290 — the test constructs `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` with `tree ∈ {"kd","ball"}`, producing `algorithm="prims_kdtree"` or `"prims_balltree"`. Neither value is in the `StrOptions({"auto","brute","kdtree","balltree"})` constraint declared on hdbscan.py:635-644, so `_validate_params` rejects the parameter before `fit` ever reaches the `if self.algorithm != "auto": ... "Sparse data matrices only support algorithm 'brute'."` or precomputed-plus-non-brute branches the test claims to exercise.
scenario: "Test parametrized as `test_hdbscan_precomputed_non_brute[kd|ball]` runs → `_validate_params` raises `InvalidParameterError` (a `ValueError` subclass), `pytest.raises(ValueError)` passes → the intended behavior ('precomputed + tree algorithm should error') is silently uncovered; a future regression that permitted `algorithm='kdtree'` with `metric='precomputed'` would go undetected by this test."
contract: Change the algorithm literal to the current valid tree name — `HDBSCAN(metric="precomputed", algorithm=f"{tree}tree")` — so the test actually reaches the precomputed-vs-tree-algorithm path it documents.
instances: single-instance

### F4 — `plot_hdbscan.py` "Scale Invariance" loop never rescales `X` [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:113-116 — `hdb = HDBSCAN(); for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`. The scale variable is threaded only into the plot label — `fit` receives the unscaled `X` and `plot` receives the unscaled `X` in every iteration, so all three subplots show identical clusterings against identical coordinates. Contrast with the DBSCAN loop above (lines 92-95) which correctly calls `dbs.fit(X * scale)` / `plot(X * scale, ...)`.
scenario: "Reader opens the rendered gallery example titled 'Scale Invariance' → three panels look identical with different `scale=` annotations → the demonstration of HDBSCAN's scale invariance is fake because no scaling ever occurs."
contract: Replace `hdb.fit(X)` and `plot(X, ...)` inside the loop with `hdb.fit(X * scale)` and `plot(X * scale, ...)`, matching the DBSCAN comparison block immediately above.
instances: single-instance

### F5 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented default `None` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4,` in the constructor signature, while the class docstring (lines 492-496) states "n_jobs : int, default=None ... `None` means 1 unless in a `joblib.parallel_backend` context." No parameter constraint pins this; the `_parameter_constraints` entry for `n_jobs` (line 646) accepts `[Integral, None]`.
scenario: "A user constructs `HDBSCAN()` expecting the documented single-threaded behavior → instead 4 worker threads are silently spawned inside `pairwise_distances` (through `kwargs['n_jobs']=self.n_jobs`) → surprising CPU usage and non-reproducible behavior against the sklearn convention that every estimator's default is `n_jobs=None`."
contract: Change the constructor signature to `n_jobs=None,` so the default matches both the docstring and sklearn's global convention.
instances: single-instance
