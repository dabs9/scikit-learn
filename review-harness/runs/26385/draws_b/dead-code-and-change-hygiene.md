### F1 — Dead duplicated `births` initialization in `_compute_stability`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:251-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice consecutively (lines 252 and 254) with identical arguments; the first result is discarded before use.
scenario: "run `_compute_stability` on any condensed tree → an extra np.full allocation is created and immediately overwritten, wasting time/memory and leaving a dead assignment in the source"
contract: Delete the duplicate line at `_tree.pyx:252` so `births` is allocated exactly once.
instances: single-instance

### F2 — Dead `mask` allocation before loop in `_weighted_cluster_center`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:896 — `mask = np.empty((X.shape[0],), dtype=np.bool_)` is immediately overwritten on every iteration (line 908 `mask = self.labels_ == idx`) and never read before that.
scenario: "call HDBSCAN(store_centers=...).fit → `_weighted_cluster_center` performs a wasted allocation whose result is unused"
contract: Remove the pre-loop `mask = np.empty(...)` line; the loop's `mask = self.labels_ == idx` is the sole live definition.
instances: single-instance

### F3 — Unrelated whitespace-only fixes bundled in `clustering.rst`
severity: low
evidence: doc/modules/clustering.rst:23-25, doc/modules/clustering.rst:34-35, doc/modules/clustering.rst:42-43 — hunk diff shows trailing-whitespace edits in unrelated Mean Shift paragraphs (the "hill climbing" paragraph, the "In general, the equation for :math:`m`…" paragraph, and "In our implementation, :math:`K(x)`…"); these paragraphs are not part of the HDBSCAN feature and are not called out in the PR description.
scenario: "reviewer diffing this PR for HDBSCAN scope → sees unrelated whitespace churn intermixed with feature docs, inflating the diff and coupling unrelated concerns"
contract: Keep whitespace-only fixes to unrelated Mean Shift documentation out of this PR; revert those hunks or split them into a separate cleanup commit.
instances: [doc/modules/clustering.rst:23, doc/modules/clustering.rst:24, doc/modules/clustering.rst:34, doc/modules/clustering.rst:42, doc/modules/clustering.rst:43]

### F4 — Dead `copy` parameter in `_hdbscan_prims` docstring
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — the `_hdbscan_prims` docstring documents a `copy : bool, default=False` parameter, but the function signature (lines 269-278) does not accept `copy`; nothing in the body references `copy` either.
scenario: "reader consulting `_hdbscan_prims` docs → believes a `copy` argument exists, but passing `copy=...` would be silently swallowed via `**metric_params` and forwarded to the distance metric"
contract: Remove the `copy` block from the `_hdbscan_prims` docstring — it describes a parameter that does not exist on this function.
instances: single-instance

### F5 — Misspelled new test filename `test_reachibility.py`
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — filename spells "reachibility"; the module under test is `_reachability.py` and the docstrings/tests spell it "reachability".
scenario: "developer greps for `test_reachability` → misses this file; future rename creates git history churn"
contract: Rename the new test file to `test_reachability.py` to match the module it tests.
instances: single-instance

### F6 — Misspelled internal identifier `mutual_reachibility_distance` in new Cython code
severity: low
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:127,144,149,182,206,209,210 — the local/temp variable is spelled `mutual_reachibility_distance`; every surrounding function, docstring, and public API uses the correct spelling `reachability`.
scenario: "future maintainer greps for `mutual_reachability_distance` inside the module → misses these occurrences due to the typo"
contract: Rename the local to `mutual_reachability_distance` (correct spelling) throughout `_reachability.pyx`.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210]

### F7 — Test uses non-existent algorithm strings, masking intended check [out-of-theme]
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282-284 — `HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` uses `"prims_kdtree"`/`"prims_balltree"` which are not in the `algorithm` StrOptions `{"auto","brute","kdtree","balltree"}` (see hdbscan.py:635-644); the raised `ValueError` therefore comes from parameter validation, not from the "precomputed + tree algorithm" branch the test purports to exercise. The intended semantic ("precomputed data with tree algorithm should error") is untested.
scenario: "future refactor removes the precomputed-vs-tree rejection logic → `test_hdbscan_precomputed_non_brute` still passes because parameter-name validation always trips first"
contract: Change the algorithm arguments to `"kdtree"` and `"balltree"` (matching the parametrize `tree` values `"kd"`/`"ball"`) so the test actually exercises the precomputed+tree rejection path.
instances: single-instance

### F8 — Documented `min_samples` default mismatches signature in `_brute_mst`/`_hdbscan_brute`/`_hdbscan_prims` [out-of-theme]
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:100-103, 185-187, 290-292 — three docstrings state `min_samples : int, default=None` while the corresponding signatures define `min_samples=5` (e.g. `_hdbscan_brute` at line 158, `_hdbscan_prims` at line 272) or `min_samples` as a required kwarg with no default (`_brute_mst` at line 82).
scenario: "reader consulting docs for these helpers → believes `min_samples` defaults to `None`; passing `None` would then be forwarded to `NearestNeighbors(n_neighbors=None)` or used in an f-string error message"
contract: Fix the docstrings to state `default=5` where the signature has `min_samples=5` and remove the `default=None` claim where `min_samples` has no signature default.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:100, sklearn/cluster/_hdbscan/hdbscan.py:185, sklearn/cluster/_hdbscan/hdbscan.py:290]
