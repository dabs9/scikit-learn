### F1 — `HDBSCAN.__init__` `n_jobs=4` default contradicts docstring `default=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4` in the signature; sklearn/cluster/_hdbscan/hdbscan.py:486 documents `n_jobs : int, default=None`.
scenario: "User constructs `HDBSCAN()` expecting `n_jobs=None` (i.e. 1 thread) per the docs → estimator silently runs `pairwise_distances` on 4 threads and mutates joblib pool sizing behaviour that other libraries/tests may depend on."
contract: The class default in `__init__` must be `n_jobs=None` to match the documented default and align with the sklearn-wide convention.
instances: single-instance

### F2 — `labels_` docstring promises `-2` for infinite samples but precomputed path never assigns it
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:534-537 states "Samples with infinite elements (+/- np.inf) are given the label -2"; but the `-2` remapping at sklearn/cluster/_hdbscan/hdbscan.py:830-844 is gated on `self.metric != "precomputed"`. For `metric="precomputed"` (dense) the code at sklearn/cluster/_hdbscan/hdbscan.py:741-751 sets `force_all_finite=False`, keeps `np.inf` in the matrix, and never runs the outlier remap — so a row full of `np.inf` distances flows through `mst_from_mutual_reachability` untouched (warning only, line 256-265) and receives ordinary tree-derived labels (typically -1 or a cluster id).
scenario: "User passes a precomputed distance matrix containing `np.inf` entries → `labels_` for those samples is -1 (or a cluster label), never -2, silently violating the documented Attributes contract."
contract: Apply the `_OUTLIER_ENCODING` remap for infinite-distance samples on the precomputed path so `labels_` uniformly assigns `-2` to infinite-distance rows regardless of `metric`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:534, sklearn/cluster/_hdbscan/hdbscan.py:830, sklearn/cluster/_hdbscan/hdbscan.py:955]

### F3 — `test_hdbscan_precomputed_non_brute` uses non-existent algorithm names and passes for the wrong reason
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 sets `algorithm=f"prims_{tree}tree"` (i.e. `"prims_kdtree"` / `"prims_balltree"`), but the `_parameter_constraints` at sklearn/cluster/_hdbscan/hdbscan.py:629-638 only accept `{"auto", "brute", "kdtree", "balltree"}`. `_validate_params()` therefore raises `InvalidParameterError` (a `ValueError` subclass) *before* the precomputed-vs-tree check at sklearn/cluster/_hdbscan/hdbscan.py:772-783 ever runs. The test only asserts `pytest.raises(ValueError)` with no `match=`, so it succeeds without ever exercising the code path its docstring claims to cover.
scenario: "Someone regresses the `metric=='precomputed'` + `algorithm=='kdtree'` guard → this test still passes because it fails at parameter validation instead, and the missing coverage lets the real regression ship."
contract: Change the test to `algorithm="kdtree"` / `algorithm="balltree"` and match the actual error message so it verifies the documented behaviour.
instances: single-instance

### F4 — `_hdbscan_brute` signature `alpha=None` contradicts its `default=1.0` docstring and would crash when invoked with the default
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 declares `alpha=None`; the docstring at sklearn/cluster/_hdbscan/hdbscan.py:183-184 says `alpha : float, default=1.0`; sklearn/cluster/_hdbscan/hdbscan.py:241 does `distance_matrix /= alpha`, which raises `TypeError: unsupported operand type(s) for /=: ... 'NoneType'` if `alpha` really is `None`.
scenario: "Any refactor that calls `_hdbscan_brute(X, ..., **kwargs)` without an explicit `alpha` (relying on the signature-advertised default) → immediate `TypeError` at `/= alpha`."
contract: Change the signature default to `alpha=1.0` to match the docstring and match the sibling `_hdbscan_prims` at sklearn/cluster/_hdbscan/hdbscan.py:273.
instances: single-instance

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter absent from the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 signature is `def _hdbscan_prims(X, algo, min_samples=5, alpha=1.0, metric="euclidean", leaf_size=40, n_jobs=None, **metric_params)` — no `copy`. Yet the docstring at sklearn/cluster/_hdbscan/hdbscan.py:313-318 documents `copy : bool, default=False` with a full description. Also `leaf_size` (in the signature) is not documented in the docstring.
scenario: "Caller reading the docstring passes `copy=True` → it silently lands in `**metric_params` and is forwarded to the distance metric, producing a `TypeError` from the distance backend at fit time rather than being ignored as the docstring implies."
contract: Remove the `copy` paragraph from the `_hdbscan_prims` docstring and add a `leaf_size` entry so signature and docstring match exactly.
instances: single-instance

### F6 — `_condense_tree` uses an inconsistent, redundant `<cnp.intp_t>` cast on `right_count` only
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:177 assigns `left_count = hierarchy[left - n_samples].cluster_size` with no cast; sklearn/cluster/_hdbscan/_tree.pyx:182 assigns `right_count = <cnp.intp_t> hierarchy[right - n_samples].cluster_size`. The `HIERARCHY_t.cluster_size` field is already declared `intp_t` in sklearn/cluster/_hdbscan/_tree.pxd:38, and `right_count` is declared `cnp.intp_t` at sklearn/cluster/_hdbscan/_tree.pyx:155 — the cast is a no-op and diverges from the parallel left-side assignment for no reason.
scenario: "A future reviewer sees the asymmetric cast and copies the pattern elsewhere thinking it's meaningful → the misleading precedent propagates and later masks a real widening/truncation when the struct layout changes."
contract: Drop the `<cnp.intp_t>` cast on line 182 so both branches read the field with identical typing.
instances: single-instance

### F7 — `_tree.pyx` retains `cnp.*_t` typing throughout, contradicting the PR's stated typedef migration
severity: low
evidence: The PR description lists as change #1: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`". `_linkage.pyx` (line 42) and `_reachability.pyx` (line 40) both `cimport` `intp_t, float64_t, int64_t, uint8_t` from `...utils._typedefs`, and use the unqualified names. But `_tree.pyx` still imports numpy as `cnp` at sklearn/cluster/_hdbscan/_tree.pyx:33 and uses `cnp.intp_t / cnp.float64_t / cnp.uint8_t` in 90 spots (grep count) — including cdef declarations at sklearn/cluster/_hdbscan/_tree.pyx:39-40, 66-67, 82-91, 144-156, 240-249, 302-305, 329-337, 388-394, 467-474, 519-525, 692-700. The paired `_tree.pxd` already imports `intp_t, float64_t, uint8_t` from `_typedefs`, so `_tree.pyx` is the only member of `_hdbscan/` still on the `cnp.*_t` typing regime.
scenario: "A later reader checks whether `_typedefs` migration is complete for `_hdbscan` → sees the plan says yes, sees `_tree.pyx` still uses `cnp.*_t`, has to spend time reconciling; future numpy header decoupling work has to re-open a file the plan claimed was already done."
contract: Complete the migration in `_tree.pyx` by replacing all `cnp.intp_t / cnp.float64_t / cnp.uint8_t` occurrences with the unqualified `intp_t / float64_t / uint8_t` imported from `...utils._typedefs`, matching `_linkage.pyx` and `_reachability.pyx`.
instances: single-instance
