### F1 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented default `None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — the constructor signature line reads `n_jobs=4,` while the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None ... None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.` Peer estimators use `n_jobs=None` (sklearn/cluster/_dbscan.py:330, sklearn/cluster/_optics.py:274).
scenario: "A user reads the docstring and instantiates `HDBSCAN()` under `with joblib.parallel_backend('threading', n_jobs=1):` → the estimator ignores the context and silently spawns 4 workers via `pairwise_distances`/`NearestNeighbors`, hardcoding parallelism that contradicts the API contract and the pattern of every other clustering estimator."
contract: Set `n_jobs=None` in `HDBSCAN.__init__` so the wired default matches the docstring and the sklearn-wide `joblib.parallel_backend` contract.
instances: single-instance

### F2 — Test file name misspelled ("reachibility" → "reachability") diverges from module under test
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — the file tests the `mutual_reachability_graph` API exposed by `sklearn/cluster/_hdbscan/_reachability.pyx` (correctly spelled), but the test file itself is registered on disk as `test_reachibility.py` (missing the second `a`). The DIFF_MANIFEST also confirms `A sklearn/cluster/_hdbscan/tests/test_reachibility.py`.
scenario: "Maintainer runs `git grep test_reachability` or `pytest sklearn/**/test_reachability.py` to locate tests for the `_reachability` module → nothing is returned because the test file is misspelled `test_reachibility.py`; the tests are not discoverable through the canonical module name and the wiring between production module and its test file is broken by spelling."
contract: Rename `sklearn/cluster/_hdbscan/tests/test_reachibility.py` to `sklearn/cluster/_hdbscan/tests/test_reachability.py` so the test file name matches the module it tests (`_reachability.pyx`).
instances: single-instance

### F3 — `test_hdbscan_precomputed_non_brute` passes invalid algorithm names `prims_kdtree`/`prims_balltree`, so the ValueError comes from parameter validation instead of the guard the test claims to cover [out-of-theme]
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:282-290 — `@pytest.mark.parametrize("tree", ["kd", "ball"])` builds `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`, but the registered `StrOptions` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 accepts only `{"auto","brute","kdtree","balltree"}`. `_validate_params()` raises `InvalidParameterError` (a `ValueError`) before reaching the tree/metric compatibility guard at sklearn/cluster/_hdbscan/hdbscan.py:772-783 that the test's docstring says it is exercising.
scenario: "A future contributor changes the tree/metric compatibility check (or removes it) → this test still passes because it now trips on parameter validation, so the intended regression protection for the tree+precomputed guard is silently gone."
contract: Rename the parametrization values to the currently-registered `"kdtree"`/`"balltree"` algorithm names so the test actually exercises the metric-vs-tree guard it claims to cover.
instances: single-instance
