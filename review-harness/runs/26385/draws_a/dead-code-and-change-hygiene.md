### F1 — Dead test using non-existent algorithm names silently downgraded to parameter-validation error
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`; the valid `algorithm` values declared in `HDBSCAN._parameter_constraints` at sklearn/cluster/_hdbscan/hdbscan.py:635-644 are `{"auto", "brute", "kdtree", "balltree"}`, and `prims_*` is nowhere in the codebase.
scenario: "Test runs → `HDBSCAN.fit` raises ValueError from `_validate_params` because `prims_kdtree`/`prims_balltree` are not permitted algorithm strings → `pytest.raises(ValueError)` passes → the code path the docstring claims to exercise (rejection of precomputed metric on a tree-based algorithm) is never touched"
contract: Rename the parametrized values to the current algorithm names (`"kdtree"`, `"balltree"`), and match on the specific error message asserted by hdbscan.py:778-789 so the test actually verifies the intended contract.
instances: single-instance

### F2 — Hardcoded `n_jobs=4` default contradicts the documented `default=None` [out-of-theme]
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `HDBSCAN.__init__`, while the docstring at sklearn/cluster/_hdbscan/hdbscan.py:487-491 declares `n_jobs : int, default=None` with the standard sklearn semantics `"``None`` means 1 unless in a :obj:`joblib.parallel_backend` context"`.
scenario: "User instantiates `HDBSCAN()` inside a `joblib.parallel_backend` context expecting the documented `None` default → constructor silently sets `n_jobs=4`, oversubscribing the backend and contradicting rendered documentation; sklearn's estimator-convention lint would also flag this discrepancy"
contract: Change the default to `n_jobs=None` to match the docstring and the sklearn-wide `n_jobs` convention.
instances: single-instance

### F3 — Duplicate `births = np.full(...)` assignment (first is dead, immediately overwritten)
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two identical statements back-to-back:
```
    births = np.full(largest_child + 1, np.nan, dtype=np.float64)

    births = np.full(largest_child + 1, np.nan, dtype=np.float64)
```
No read of `births` occurs between the two writes, so the first allocation is unreachable/wasted.
scenario: "Every call to `_compute_stability` (one per `HDBSCAN.fit`) allocates and discards a `largest_child+1`-length float64 array before immediately overwriting it → dead code that directly contradicts the PR-description item 'Trimmed unused variables (thanks to Cython linting pre-commit)'"
contract: Delete the redundant first `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` at line 252.
instances: single-instance

### F4 — Scale-invariance HDBSCAN loop never applies the loop variable, rendering the demo dead
severity: low
evidence: examples/cluster/plot_hdbscan.py:112-116 —
```
hdb = HDBSCAN()
for idx, scale in enumerate((1, 0.5, 3)):
    hdb.fit(X)
    plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})
```
The `scale` variable is only used as a label; both `fit` and `plot` receive raw `X`. Contrast the immediately-preceding DBSCAN block (plot_hdbscan.py:92-95) that correctly uses `X * scale` throughout.
scenario: "Documentation build runs the example → generated 'Scale Invariance' section renders three identical HDBSCAN clusterings each labeled with a different `scale`, so the section does not demonstrate scale invariance — it demonstrates fitting the same X three times"
contract: Fit on and plot with `X * scale` in the loop body, matching the DBSCAN comparison block above.
instances: single-instance
