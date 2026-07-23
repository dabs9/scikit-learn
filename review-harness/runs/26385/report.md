# PR 26385 — HDBSCAN Estimator Review

Range: `86541f2b3bc8a96264e265cb810cf79858544340..04c8b6954e3e4b8f8af086cea9d386f954b76bfe`

Counts by severity: **high: 5**, **medium: 11**, **low: 25** (41 units total).

---

## High severity

### [high] U1 — `_weighted_cluster_center` crashes on non-finite input when `store_centers` is set

`HDBSCAN.fit` reduces `X` to `X[finite_index]` but rebuilds `self.labels_` at raw shape; then hands the reduced `X` to `_weighted_cluster_center`, whose mask `self.labels_ == idx` is raw-length. Any `HDBSCAN(store_centers='centroid').fit(X_with_nan_or_inf)` immediately fails with `IndexError: boolean index did not match indexed array along dimension 0`. Fix by passing `self.labels_[finite_index]` alongside the reduced `X` so the mask and data lengths agree.

<details><summary>verbatim finding</summary>

```
### F1 — `_weighted_cluster_center` crashes when input has non-finite rows and `store_centers` is set
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:733,844,854-855,895,908-909 — after the non-finite branch, `X = X[finite_index]` (shape `finite_count × n_features`) and `self.labels_` is rebuilt to shape `self._raw_data.shape[0]` (raw shape). Then `if self.store_centers: self._weighted_cluster_center(X)` is invoked with the reduced `X` while `_weighted_cluster_center` computes `mask = self.labels_ == idx` (raw-length boolean) and applies `data = X[mask]`, which raises `IndexError: boolean index did not match indexed array along dimension 0`.
scenario: "user calls `HDBSCAN(store_centers='centroid').fit(X)` on data containing any np.inf/np.nan row → fit() raises IndexError instead of storing centroids"
contract: Pass the reduced `X` together with `self.labels_[finite_index]` into `_weighted_cluster_center` so mask length matches data length.
instances: single-instance
```
</details>

### [high] U2 — `_weighted_cluster_center` miscounts clusters when missing-label `-3` is present

`n_clusters = len(set(self.labels_) - {-1, -2})` at hdbscan.py:895 does not subtract `-3`, the missing-data label defined at hdbscan.py:80. When any NaN row is present, `-3` remains in the set, `n_clusters` is inflated by 1, and the last iteration's mask is empty — the medoid path calls `pairwise_distances` on an empty array and the centroid path returns NaN or errors. Subtract every outlier label in `_OUTLIER_ENCODING` in one place and mirror that set everywhere non-outlier labels are enumerated.

<details><summary>verbatim finding</summary>

```
### F2 — `_weighted_cluster_center` miscounts clusters when missing-data label `-3` is present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the outlier-encoding at hdbscan.py:80 defines the missing label as `-3`, and hdbscan.py:849 assigns `-3` to samples with NaN rows
scenario: "User calls `HDBSCAN(store_centers='centroid').fit(X)` on data with any NaN rows → `-3` remains in the label set, `n_clusters` is inflated by 1, the final iteration `idx = n_clusters-1` selects an empty mask, and `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError` / returns nan-filled row (medoid path calls `pairwise_distances` on an empty array)"
contract: exclude every outlier label — use `n_clusters = len(set(self.labels_) - {-1, -2, -3})` (or `- set(v['label'] for v in _OUTLIER_ENCODING.values()) - {-1}`); mirror this same set everywhere non-outlier labels are enumerated
instances: single-instance
```
</details>

### [high] U3 — `_hdbscan_prims` never applies `alpha` — silent divergence between brute and tree backends

In the brute path, `distance_matrix /= alpha` is applied before core distances are derived from it. In the prims path, `alpha` is threaded only into `mst_from_data_matrix` — `core_distances` are computed via `NearestNeighbors.kneighbors` with no alpha scaling. Non-default `alpha` therefore produces different labels between `algorithm='brute'` and `algorithm='kdtree'/'balltree'` on identical data. Divide `core_distances` by `alpha` in the prims path so the semantics match brute-force.

<details><summary>verbatim finding</summary>

```
### F3 — `_hdbscan_prims` never applies `alpha` — silent parameter drop for tree-based algorithms
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-348 — the function receives `alpha=1.0` and passes it through as `mst_from_data_matrix(X, core_distances, dist_metric, alpha)`; `alpha` is only honored inside `mst_from_data_matrix` (sklearn/cluster/_hdbscan/_linkage.pyx:193 `pair_distance /= alpha`). But `core_distances` are computed via `NearestNeighbors(...).kneighbors(X)` at lines 332-343 with no `alpha` scaling. In `_hdbscan_brute` (line 241) `distance_matrix /= alpha` divides the *entire* distance matrix before core distances are derived from it, so core distances are also scaled by 1/alpha. Result: with `algorithm="brute"` core distances participate scaled; with tree algorithms they participate unscaled. The two backends produce different clusterings for the same non-default `alpha`.
scenario: "user sets `HDBSCAN(alpha=0.5)` and switches between `algorithm='brute'` and `algorithm='kdtree'` on identical data → the two backends return different labels because prims-based code paths silently ignore alpha when computing core distances"
contract: Divide `core_distances` by `alpha` in the prims path before they enter `mst_from_data_matrix`, matching the brute-force semantics.
instances: single-instance
```
</details>

### [high] U4 — `_hdbscan_brute` unconditionally divides `distance_matrix /= alpha` even when `alpha is None`

`_hdbscan_brute(X, min_samples=5, alpha=None, …)` documents `alpha : float, default=1.0` but the signature actually defaults `alpha=None`; line 241 then executes `distance_matrix /= alpha` unconditionally. Any caller who invokes `_hdbscan_brute(X)` as documented gets a `TypeError` at that division. End-user code is shielded because the estimator always passes a validated float, but internal or test-code callers are broken. Fix by defaulting `alpha=1.0` in the signature (or guarding the division on `alpha is not None`).

<details><summary>verbatim finding</summary>

```
### F29 — `_hdbscan_brute` unconditionally divides `distance_matrix /= alpha` even when `alpha is None`
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166 — `_hdbscan_brute(X, min_samples=5, alpha=None, metric="euclidean", ...)` defaults `alpha=None`, and at line 241 unconditionally executes `distance_matrix /= alpha`. When invoked with the default signature (`alpha=None`), this raises `TypeError: unsupported operand type(s) for /=: 'numpy.ndarray' and 'NoneType'`. The class-level caller in `HDBSCAN.fit` at hdbscan.py:767 always passes `alpha=self.alpha`, which is validated by `_parameter_constraints` to be a positive float (line 628), so end-user calls never hit this — but any downstream or test caller that uses `_hdbscan_brute` with its documented default fails immediately. F16 flags the docstring/signature mismatch (`alpha : float, default=1.0` vs signature `alpha=None`); this is the runtime consequence.
scenario: "internal or test-code caller invokes `_hdbscan_brute(X)` with defaults as documented → `TypeError` immediately at line 241 because the default `alpha=None` is not a valid divisor"
contract: fix the `_hdbscan_brute` signature default to `alpha=1.0` (or guard the `/= alpha` with `if alpha is not None`), so the documented API matches runtime behavior; complements F16's docstring fix.
instances: single-instance
```
</details>

### [high] U5 — `dbscan_clustering` sample-order fragility with non-finite inputs

`labelling_at_cut` iterates `range(n_samples)` under the invariant that raw sample indices correspond exactly to the leading rows of the (remapped) single-linkage tree. That invariant is preserved by the current append-order in `remap_single_linkage_tree`, but any refactor that reorders outlier merges (for example, sorting them per F10's proposal) silently breaks the sample-index-to-tree-row mapping and yields wrong labels for finite rows. Add an explicit invariant assertion and a regression test that verifies label-to-sample correspondence on data with both `np.inf` and `np.nan`.

<details><summary>verbatim finding</summary>

```
### F33 — `HDBSCAN.dbscan_clustering` calls `labelling_at_cut` with the length-`n_samples` internal single-linkage tree but returns labels sized to `finite_count`, breaking `dbscan_clustering` after non-finite handling
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:923-970, 834-844 — after `fit`, when the input contained non-finite rows, `self._single_linkage_tree_` is *remapped* via `remap_single_linkage_tree` (line 834) and now has length `finite_count - 1 + len(non_finite)` (line 392 `np.concatenate([tree, outlier_tree])`). But `labelling_at_cut` (sklearn/cluster/_hdbscan/_tree.pyx:396-397) computes `root = 2 * linkage.shape[0]` and `n_samples = root // 2 + 1 = linkage.shape[0] + 1`. Since the remapped tree has one row per merge (including outlier merges), its length is `raw_n_samples - 1`, so `labelling_at_cut` returns an array of length `raw_n_samples` — good. However, at line 964-969 `dbscan_clustering` reads `infinite_index = self.labels_ == -2` and writes `labels[infinite_index] = -2`; because the outlier tree merges at `np.inf` distance, the outlier samples are already assigned distinct clusters by `labelling_at_cut` (via `TreeUnionFind` with cut < inf), so those clusters may satisfy `cluster_size >= min_cluster_size` and get valid labels. That labelling is then silently overwritten by the outlier constants, which is fine — but any user who checked `set(labels) - {-1,-2,-3}` before overwrite would have seen phantom labels. More importantly, `labelling_at_cut` uses `cluster_size < min_cluster_size` to mark clusters as noise, and the outlier merges each contribute one distinct cluster of size 1 to the union-find (since each outlier merges only into the last cluster at `np.inf`, they never become < cut), leading to N stale entries in `cluster_size` for every outlier — memory O(raw_n_samples) rather than O(finite_count). For huge datasets with a small non-finite fraction, this is fine; for datasets that are mostly non-finite, `cluster_size = np.zeros(cluster, dtype=np.intp)` at _tree.pyx:408 allocates `2 * (raw_n_samples - 1) + 1` intp — still OK. Real bug: `dbscan_clustering(cut_distance)` for `cut_distance >= np.inf` (or the special case handled by `labelling_at_cut` when `node.value < cut` and outlier merges have value `np.inf`) never merges outlier samples with any cluster because `np.inf < np.inf` is false, so each outlier becomes its own singleton — and `_do_labelling`/`labelling_at_cut` treats singletons as noise. The overwrite at line 968 then relabels these to `-2`/`-3`, which is correct, but the sample-order of `labels` from `labelling_at_cut` is over `range(n_samples)` — and `n_samples = linkage.shape[0] + 1 = raw_n_samples`, matching. So the sample ordering is correct only because `raw_n_samples = linkage.shape[0] + 1` after remapping; if `remap_single_linkage_tree` ever changes to prepend rather than append outlier merges (or dedupe them via F8's fix using `sorted(set(...))`), the sample-index-to-linkage-row invariant that `labelling_at_cut` relies on (samples 0..n_samples-1 correspond exactly to leaf positions) may break silently. There is no test that exercises `dbscan_clustering` on data with non-finite entries and verifies label positions.
scenario: "user calls `HDBSCAN().fit(X_with_nan).dbscan_clustering(0.5)` and expects labels[i] to correspond to X[i] → after F8's proposed fix (sorting outlier indices), the ordering of appended outlier merges no longer matches the raw sample-index sequence, so `labelling_at_cut`'s `range(n_samples)` iteration returns labels for the wrong sample slots and the subsequent `labels[infinite_index] = -2` overwrite fixes the outlier positions but not the finite positions"
contract: add an assertion in `remap_single_linkage_tree` that `tree.shape[0] + 1 == raw_n_samples` after append; add a test that runs `dbscan_clustering` on X-with-nan-and-inf and verifies label-to-sample correspondence for both finite and non-finite rows.
instances: single-instance
```
</details>

---

## Medium severity

### [medium] U6 — `HDBSCAN.__init__` default `n_jobs=4` contradicts docstring and sklearn convention

The constructor signature hard-codes `n_jobs=4` while the docstring claims `n_jobs : int, default=None`. `4` unconditionally spawns four workers even inside a `joblib.parallel_backend('threading', n_jobs=1)` context, contradicting both the documented behavior and sklearn's glossary. Change the default to `n_jobs=None`.

<details><summary>verbatim finding</summary>

```
### F4 — `HDBSCAN.__init__` default `n_jobs=4` contradicts docstring and sklearn convention
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in the `__init__` signature, while the docstring at :486 documents `n_jobs : int, default=None`. Sklearn's convention (and every other estimator listed in the whats_new entry) is `n_jobs=None`, meaning "1 unless in a `joblib.parallel_backend` context". A hard-coded `4` silently spawns 4 worker threads regardless of the user's `parallel_backend`.
scenario: "user creates `HDBSCAN()` inside `with joblib.parallel_backend('threading', n_jobs=1):` → sklearn spawns 4 threads anyway, contradicting the documented behavior and the context manager the user set"
contract: Change the default in `__init__` to `n_jobs=None` so behavior matches the docstring and sklearn glossary semantics.
instances: single-instance
```
</details>

### [medium] U7 — `pytest.raises(ValueError)` without `match=` shadows the guard under test

Both `test_hdbscan_precomputed_non_brute` (test_hdbscan.py:282) and `test_hdbscan_algorithms` (test_hdbscan.py:168-170) use `pytest.raises(ValueError)` without a `match=` clause. Parameter-validator failures (`InvalidParameterError` subclasses `ValueError`) satisfy the assertion, so the tests can pass even if the intended guard (precomputed-vs-tree, KDTree/BallTree metric compatibility) is removed. Anchor the assertion with `match=` and use the actual algorithm strings supported by `_parameter_constraints`.

<details><summary>verbatim finding</summary>

```
### F5 — `test_hdbscan_precomputed_non_brute` triggers `InvalidParameterError`, not the error under test
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")`. The valid `algorithm` values in `_parameter_constraints` (sklearn/cluster/_hdbscan/hdbscan.py:629-638) are `{"auto","brute","kdtree","balltree"}`, so `"prims_kdtree"`/`"prims_balltree"` fail parameter validation before any precomputed-vs-tree logic runs. The assertion `pytest.raises(ValueError)` still passes because `InvalidParameterError` is a subclass of `ValueError`, but the test never exercises the code path it names in its docstring ("correctly raises an error when passing precomputed data while requesting a tree-based algorithm").
scenario: "test claims to guard the precomputed-vs-tree combination but actually only guards the algorithm StrOptions validator → the intended incompatibility check becomes uncovered and any regression there ships unnoticed"
contract: Use the actual algorithm strings supported by this estimator (`"kdtree"`, `"balltree"`) and assert with `match=` on the "Sparse data matrices only support algorithm `brute`" style message so the test fails if that guard is removed. Also verify a corresponding guard exists for the precomputed dense case.
instances: single-instance
```

```
### F1 — rule of D:F5 recurs in `test_hdbscan_algorithms` — `pytest.raises(ValueError)` used with no `match=`, so validator-level errors can shadow the guard under test
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:168-170 — the branch `if metric not in ALGOS_TREES[algo].valid_metrics(): with pytest.raises(ValueError): hdb.fit(X)`. The intended guard is `hdbscan.py:772-783` (`f"{self.metric} is not a valid metric for a KDTree-based algorithm."`). But `_parameter_constraints["metric"]` at hdbscan.py:626 is `StrOptions(FAST_METRICS | {"precomputed"}), callable` — any metric in `_VALID_METRICS` that is not in `FAST_METRICS` (e.g. `"correlation"`, `"cosine"` when the current `FAST_METRICS` set doesn't include it, or `"jaccard"`/`"matching"`/etc.) raises `InvalidParameterError` (a subclass of `ValueError`) *before* the tree-vs-metric guard runs. Since `pytest.raises(ValueError)` has no `match=`, the outer test passes whether the tripped code path was the parameter validator or the intended tree-metric guard — mirrors the exact defect F5 flagged at test_hdbscan.py:282.
scenario: "regression removes the KDTree/BallTree metric guard in `HDBSCAN.fit` → `test_hdbscan_algorithms` still passes for many `metric` values because the parameter validator's `InvalidParameterError` (a `ValueError`) satisfies the raise-assertion instead"
contract: Add `match=r"is not a valid metric for a .*-based algorithm"` to the `pytest.raises(ValueError, ...)` at line 169 so the assertion actually pins the intended guard.
instances: single-instance
```
</details>

### [medium] U8 — Test computes `clean_idx` via array addition, not concatenation

`missing_labels_idx + infinite_labels_idx` on ndarrays does elementwise addition, not list concatenation. `[2,5] + [0]` broadcasts to `[2,5]`, so row 0 (an infinite outlier) survives in `clean_idx`. The assertion only passes by accident, because HDBSCAN still marks that row `-2` when kept. Convert with `.tolist()` or `np.concatenate([...]).tolist()`.

<details><summary>verbatim finding</summary>

```
### F6 — Test computes `clean_idx` via array addition, not concatenation
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:212-221 — `missing_labels_idx = np.flatnonzero(...)` (line 212, returns ndarray `[2, 5]`), `infinite_labels_idx = np.flatnonzero(...)` (line 215, ndarray `[0]`), then line 218 `clean_idx = list(set(range(200)) - set(missing_labels_idx + infinite_labels_idx))`. `missing_labels_idx + infinite_labels_idx` is elementwise numpy addition, not concatenation — `[2, 5] + [0]` broadcasts to `[2, 5]`, so the sample at index `0` (an infinite outlier) is left in `clean_idx`. The subsequent `assert_array_equal(clean_labels, labels[clean_idx])` passes only because HDBSCAN still assigns the infinite sample `-2` when kept, matching the original labels at that index.
scenario: "test intends to compare a clean-refit against outlier-marked labels → actually keeps outlier row 0 in the 'clean' subset → the assertion succeeds by accident and any change that made the two labelings diverge at row 0 would silently be missed"
contract: Concatenate as lists — `clean_idx = list(set(range(200)) - set(missing_labels_idx.tolist() + infinite_labels_idx.tolist()))` (or `np.concatenate([...]).tolist()`) — so `clean_idx` truly excludes every outlier.
instances: single-instance
```
</details>

### [medium] U9 — Plot example claims scale-invariance but never scales `X`

Inside `for scale in (1, 0.5, 3)`, the HDBSCAN block calls `hdb.fit(X)` and `plot(X, ...)` with unscaled `X`, while the parallel DBSCAN block scales via `X * scale`. The three panels are identical and demonstrate nothing about the property being taught. Change to `hdb.fit(X * scale)` / `plot(X * scale, ...)`.

<details><summary>verbatim finding</summary>

```
### F7 — Plot example claims scale-invariance but never scales `X`
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — inside `for idx, scale in enumerate((1, 0.5, 3))` the body calls `hdb.fit(X)` and `plot(X, hdb.labels_, ...)` with the unscaled `X`, while the surrounding narrative at examples/cluster/plot_hdbscan.py:104-105 says "HDBSCAN is scale-invariant" and the parallel DBSCAN block at examples/cluster/plot_hdbscan.py:92-95 uses `X * scale`. Every panel receives identical data, so the rendered figure demonstrates nothing about scale invariance.
scenario: "Reader looks at the auto-generated example expecting scaled inputs → sees three identical HDBSCAN plots labelled `scale=1`, `scale=0.5`, `scale=3`, which is trivially self-consistent and does not evidence the property being taught"
contract: Change the loop body to `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, ...)`, mirroring the DBSCAN block.
instances: single-instance
```
</details>

### [medium] U10 — `remap_single_linkage_tree` uses set iteration for outlier order — nondeterministic tree

The caller passes `non_finite=set(infinite_index + missing_index)`, whose iteration order is not guaranteed across runs. Downstream consumers of `self._single_linkage_tree_` (equality-based caches, serialization, `dbscan_clustering`) see edges appended in nondeterministic order on identical inputs. Pass `sorted(set(infinite_index + missing_index))` and iterate that ordering when appending outlier rows.

<details><summary>verbatim finding</summary>

```
### F8 — `remap_single_linkage_tree` writes `outlier + outlier_count` into left/right node columns that are meant to be raw sample indices, corrupting bookkeeping when both nan and inf are present
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:383-398 — `outlier_tree[i] = (outlier, last_cluster_id + 1, np.inf, last_cluster_size + 1)`; the `outlier` value used for the left node is the raw index of the non-finite point, but at line 838 the caller passes `non_finite=set(infinite_index + missing_index)`, which is a **set** whose iteration order is not deterministic. The subsequent `new_labels[infinite_index] = ...` and `new_labels[missing_index] = ...` at lines 842-843 are index-based writes and therefore correct, but any consumer that walks `self._single_linkage_tree_` (e.g. `dbscan_clustering` via `labelling_at_cut`) will see outlier merges appended in nondeterministic order, meaning `_single_linkage_tree_` — a public-through-`dbscan_clustering` attribute — is not reproducible across runs on the same data when non-finite entries are present.
scenario: "user calls `HDBSCAN().fit(X)` twice on identical non-finite data → `hdb._single_linkage_tree_` differs in edge order between runs; downstream `dbscan_clustering(cut_distance=...)` output is still correct but internal tree state is nondeterministic and breaks equality-based caching or serialization checks"
contract: Pass a deterministically ordered container (e.g. `sorted(set(infinite_index + missing_index))`) to `remap_single_linkage_tree`, and iterate that same ordering when constructing `outlier_tree`.
instances: single-instance
```
</details>

### [medium] U11 — Sparse feature path with non-finite entries mishandles `np.matrix` reductions

For a sparse feature matrix with non-finite entries, `reduced_X = X.sum(axis=1)` returns `np.matrix`, and `.nonzero()` returns 2-D indices whose downstream use is fragile. The current tests only pass because the shape happens to be workable. Convert with `np.asarray(reduced_X).ravel()` before `.nonzero()`, and add an explicit `np.inf` sparse-input test.

<details><summary>verbatim finding</summary>

```
### F9 — Sparse feature path with non-finite entries never runs `_get_finite_row_indices`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:699-733 — the non-finite handling branch is guarded by `if self.metric != "precomputed":` and calls `_get_finite_row_indices(X)`. When `X` is a sparse (csr/lil) feature matrix (i.e. non-precomputed sparse), `_get_finite_row_indices` at line 401-407 handles sparse via `matrix.tolil().data`. But the diff test `test_hdbscan_sparse` at sklearn/cluster/tests/test_hdbscan.py:302-306 sets `sparse_X_nan[0, 0] = np.nan` on a `csr_matrix`, and `_assert_all_finite(X.data if issparse(X) else X)` at line 710 correctly detects it. Then `finite_index = _get_finite_row_indices(X)` runs `matrix.tolil().data` where each entry is a Python list of stored values — however `_get_finite_row_indices` treats rows with all-zero stored values as "finite" even when the un-stored zeros are irrelevant, but real bug: `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` at line 732 uses `finite_index` which is a `np.ndarray` for the dense case and a `np.array([...])` from list comprehension in the sparse case, whose dtype defaults to `int64` on most platforms — OK. However `reduced_X = X.sum(axis=1)` on a sparse matrix at line 721 returns a `np.matrix` (not ndarray); `np.isnan(reduced_X).nonzero()[0]` on a `np.matrix` returns 2-D indices whose semantics are then mishandled: `list(np.isnan(reduced_X).nonzero()[0])` yields row indices when the matrix is 1-column but for the general sparse case the `sum(axis=1)` yields `(n_samples, 1)` matrix, so `.nonzero()` returns row/col pairs and `.nonzero()[0]` gives rows — but the resulting `missing_index`/`infinite_index` still get passed to `new_labels[missing_index] = ...` at line 843 with dtype int64 — this happens to work, but is fragile and inconsistent with the dense path.
scenario: "sparse feature matrix with np.nan in stored data flows through the non-finite branch → `reduced_X = X.sum(axis=1)` returns np.matrix and subsequent scalar ops rely on np.matrix quirks; test coverage passes only because n_features>1 makes the shape workable"
contract: Convert `reduced_X` to an ndarray via `np.asarray(reduced_X).ravel()` before running `.nonzero()`; add an explicit test for np.inf (not just np.nan) in a sparse feature matrix to lock in behavior.
instances: single-instance
```
</details>

### [medium] U12 — `_do_labelling` sizes `result` to `root_cluster`, not `n_samples`

The function assumes the invariant `root_cluster == n_samples` (implied by `_condense_tree`), then allocates `np.empty(root_cluster, ...)`. Any hand-crafted tree where parent ids don't start at `n_samples` — e.g. after external filtering — silently returns a mis-sized `result`. Take `n_samples` as an explicit parameter and add an invariant docstring/assert.

<details><summary>verbatim finding</summary>

```
### F10 — `_do_labelling` returns `result` sized to `root_cluster` while callers assume it has length `n_samples`
severity: medium
evidence: sklearn/cluster/_hdbscan/_tree.pyx:481 — `result = np.empty(root_cluster, dtype=np.intp)`, then `for n in range(root_cluster): ... result[n] = label`. `root_cluster = np.min(parent_array)`, which by construction equals `n_samples` (see `_condense_tree` at line 147 seeding `next_label = n_samples + 1` and relabel[root] = n_samples). So the assumption "root_cluster == n_samples" is invariant-dependent, undocumented, and if `_do_labelling` is ever called with a hand-crafted condensed tree whose parent ids don't start at n_samples (as the test `test_labelling_thresholding` does with `parent=5, n_samples=5` — matching the invariant) the array size will be wrong. In fact test_hdbscan.py:508-517 constructs a tree with parents=5 and n_samples=5 which fits the invariant only by coincidence.
scenario: "reviewer or downstream user calls `_do_labelling` with a valid condensed tree that has parent ids ≥ n_samples+1 (e.g. after external filtering) → `result` is too short and later `_get_clusters` / caller silently trims samples"
contract: Take `n_samples` as an explicit argument (or infer it from `child_array.max() + 1` where child < parents), and size `result = np.empty(n_samples, dtype=np.intp)`. Add a docstring assertion that `parent_array.min() == n_samples`.
instances: single-instance
```
</details>

### [medium] U13 — `_hdbscan_brute` silently mutates caller's precomputed `X` on the `copy=False` path

`distance_matrix = X.copy() if copy else X` correctly protects the caller when `copy=True`, but the subsequent in-place `distance_matrix /= alpha` mutates the caller's `X` when `copy=False`. Combined with the estimator's `copy=False` default and any non-default `alpha`, calling `HDBSCAN(metric='precomputed', alpha=0.5).fit(D)` twice on the same `D` yields progressively-shrunk distances. Replace the in-place division with `distance_matrix = distance_matrix / alpha` and add a round-trip regression test.

<details><summary>verbatim finding</summary>

```
### F30 — `_hdbscan_brute`'s `copy` parameter is silently ignored on the non-precomputed branch
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:158-166,222-241 — the function accepts `copy=False` (line 164) and documents at :207-212 that "if `copy=True` ... it only applies when `metric='precomputed'`". However, in the `else` (non-precomputed) branch at :237-241, `distance_matrix = pairwise_distances(...)` returns a *fresh* ndarray, then `distance_matrix /= alpha` mutates it in place — no user data is at risk. In the precomputed branch at :236, `distance_matrix = X.copy() if copy else X` correctly protects the caller's `X` — but the subsequent `distance_matrix /= alpha` at :241 mutates `X` in place when `copy=False`, silently corrupting the caller's precomputed matrix. The docstring promises "in-place modifications ... will overwrite `X`", but the estimator-level `copy=False` default combined with a non-default `alpha` makes this the common case: any repeated `HDBSCAN(alpha=0.5, metric='precomputed').fit(D)` on the same `D` fits progressively-shrinking distances.
scenario: "user calls `HDBSCAN(metric='precomputed', alpha=0.5).fit(D)` twice on the same `D` → second fit sees `D / 0.5 / 0.5 = 4·D_original`, producing different clusters each call; test_hdbscan_distance_matrix at test_hdbscan.py:78-83 only sets `copy=True` and never asserts round-trip immutability under `alpha != 1.0`."
contract: Replace the in-place `distance_matrix /= alpha` on the precomputed branch with an out-of-place `distance_matrix = distance_matrix / alpha`, so the caller's `X` is never mutated regardless of `copy`. Add a regression test that runs `HDBSCAN(metric='precomputed', alpha=0.5).fit(D)` twice on identical `D` and asserts identical outputs.
instances: single-instance
```
</details>

### [medium] U14 — `_get_finite_row_indices` sparse path treats empty rows as finite and inconsistent with dense

The sparse branch tests each row's stored values via `np.all(np.isfinite(row))`, which returns True for empty rows (vacuous truth), and doesn't share a canonical implementation with the dense reduction. A sparse row whose single stored value is non-finite may end up in both `finite_index` and `missing_index`, so it participates in the MST and is then relabeled `-3`. Unify both codepaths on `reduced_X = np.asarray(X.sum(axis=1)).ravel()` and add a dense-vs-sparse equivalence test.

<details><summary>verbatim finding</summary>

```
### F35 — `_get_finite_row_indices` returns row indices sorted only for dense input; sparse path uses list-comprehension order matching `matrix.tolil().data` (still sorted) but the two paths are not covered by an equivalence test, and empty stored-value rows are treated as finite despite `_assert_all_finite` having already flagged them
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:396-407 — dense path: `(row_indices,) = np.isfinite(matrix.sum(axis=1)).nonzero()` returns sorted indices. Sparse path: `row_indices = np.array([i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))])` — `matrix.tolil().data` has one Python list per row, but if a row has *no* stored values (empty list), `np.all(np.isfinite([]))` returns `True` (vacuous quantifier), so a completely empty row is treated as finite. For a CSR feature matrix where a row's non-zero stored values are all finite but a coordinate outside the sparsity pattern was explicitly `np.nan` (impossible for csr_matrix stored data, so mostly a corner case) or where `X.data` contains inf that `_assert_all_finite(X.data)` at hdbscan.py:710 already flagged — the sparse row-indexing must include that row's index in `missing_index`/`infinite_index`, but the finite-index computation on line 725/728 uses `reduced_X = X.sum(axis=1)` which sums *stored* values — an implicit 0 for unstored positions — so a nan/inf in `X.data` at row `r` propagates to `reduced_X[r]`. However, `list(np.isnan(reduced_X).nonzero()[0])` on a `np.matrix` (returned by sparse `.sum(axis=1)`) returns row indices, and those indices are then not filtered out of `finite_index`: `_get_finite_row_indices` may include row `r` in `finite_index` (because `matrix.tolil().data[r]` sees only the stored values that were flagged), yet also include it in `missing_index`, so `X = X[finite_index]` (line 733) keeps the row and later `new_labels[missing_index] = -3` overwrites it. Net effect: no crash, but the row is *both* used in clustering *and* re-labeled as -3, so its true cluster label is silently discarded — the row's true single-linkage assignment differs from `-3`.
scenario: "sparse CSR feature matrix with a single `np.nan` in `X.data` at row `r` → row `r` is included in `finite_index` (because sparse `_get_finite_row_indices` uses `matrix.tolil().data[r]` which shows only the stored values, one of which is nan — actually excluded — but scikit-learn's Sparse-with-nan handling has flip-flopped) → the row participates in Prim's MST but is then relabeled -3, so a sample influences its neighbors' cluster assignment but is itself marked as missing"
contract: unify the sparse and dense codepaths to use `reduced_X = np.asarray(X.sum(axis=1)).ravel()` (fixing F9's np.matrix issue) and derive `finite_index = np.isfinite(reduced_X).nonzero()[0]` for both dense and sparse; add an equivalence test that fits `HDBSCAN` on dense `X_with_nan` and on `csr_matrix(X_with_nan)` and asserts identical labels.
instances: single-instance
```
</details>

### [medium] U15 — `_hdbscan_prims` forwards `p=None` to `NearestNeighbors` and `DistanceMetric.get_metric`

`NearestNeighbors(..., p=None, metric_params=metric_params, ...)` currently passes `p=None` explicitly. For `metric='minkowski'` this is invalid (Minkowski requires a finite `p`) and would inconsistently interact with any `p` inside `metric_params`. Drop `p=None` and let `metric_params` carry the value.

<details><summary>verbatim finding</summary>

```
### F38 — `_hdbscan_prims` invokes `NearestNeighbors(...).fit(X)` then `.kneighbors(X, min_samples, return_distance=True)` where `min_samples` is used both as `n_neighbors` and as the second positional argument to `kneighbors`, causing double parameterization
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:332-343 — `nbrs = NearestNeighbors(n_neighbors=min_samples, ...).fit(X)` sets the estimator's default `n_neighbors=min_samples`. Then `neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True)` explicitly overrides with the same value — redundant but harmless. However, the `p=None` argument at line 339 is *not* valid for `NearestNeighbors`: `p` is only meaningful when `metric='minkowski'` and must be a positive real. Passing `p=None` triggers `_validate_params` failure in older sklearn (`InvalidParameterError: 'p' parameter of NearestNeighbors must be a float in the range [1.0, inf]`); the fact that this passes in current tests indicates the constraint accepts `None` — but `DistanceMetric.get_metric("minkowski", p=None)` at line 344 also fails because Minkowski requires a finite `p`. The `p=None` therefore only works because `metric != "minkowski"` in the test coverage.
scenario: "user sets `HDBSCAN(algorithm='kdtree', metric='minkowski', metric_params={'p': 3})` → `NearestNeighbors(..., p=None, metric_params={'p': 3})` propagates `p=None` and `metric_params={'p': 3}` to the underlying tree — depending on which wins, the resulting core distances use inconsistent `p` between `NearestNeighbors` and `DistanceMetric.get_metric`, silently corrupting the MST"
contract: remove `p=None` from `NearestNeighbors(...)` at line 339 and let `metric_params` carry `p` if needed; audit test coverage for `metric='minkowski'` with explicit `p`.
instances: single-instance
```
</details>

### [medium] U16 — Plan says `cnp.*_t` → `*_t` from `_typedefs` — 90 `cnp.*_t` usages remain in `_tree.pyx`

The plan-of-record lists this as a novel change and `_linkage.pyx` / `_reachability.pyx` have been converted, but `_tree.pyx` still uses `cnp.intp_t`, `cnp.float64_t`, `cnp.uint8_t` etc. throughout — 90 occurrences. Migration is not purely cosmetic: `cnp.PyArrayObject*` casts at multiple sites require `cimport numpy as cnp` to stay, and `cnp.import_array()` is missing (contrast `_reachability.pyx:41`), so a blind removal would break the build or segfault. Convert only the scalar dtypes, retain `cimport numpy as cnp`, and add `cnp.import_array()`.

<details><summary>verbatim finding</summary>

```
### F24 — Plan claim "cnp.*_t → *_t from _typedefs" not applied to _tree.pyx
severity: low
evidence: The PR description lists as a novel change: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`". `_linkage.pyx` and `_reachability.pyx` import scalar typedefs from `...utils._typedefs cimport ...` and use bare `intp_t`, `float64_t`, etc. However `sklearn/cluster/_hdbscan/_tree.pyx` still uses `cnp.intp_t`, `cnp.float64_t`, `cnp.uint8_t` etc. throughout — 90 total occurrences (see grep). Only the `_tree.pxd` header imports the new typedefs (`_tree.pxd:36`); the `.pyx` body itself was not converted.
scenario: "plan-of-record says every Cython file uses the new typedef import → _tree.pyx is inconsistent → next author touching this file has to guess which convention applies and may reintroduce mismatches"
contract: Convert `_tree.pyx` to use `intp_t`/`float64_t`/`uint8_t` from `...utils._typedefs cimport ...`, matching the sibling `_linkage.pyx`/`_reachability.pyx`.
instances: single-instance
```

```
### F31 — additional instances of F24: `_tree.pyx` also uses `cimport numpy as cnp` scalar names in signatures, blocking removal
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:33,40 — the file `cimport numpy as cnp` and defines `cdef cnp.intp_t NOISE = -1`; virtually every cdef/cpdef signature uses `cnp.intp_t`, `cnp.float64_t`, `cnp.uint8_t`, and `cnp.PyArrayObject`. F24 already flags 90 occurrences, but the migration is not just cosmetic — the module also imports `PyArray_SHAPE` via `cdef extern from "numpy/arrayobject.h"` in `_tree.pxd:48-49`, and the `.pyx` re-uses `cnp.PyArrayObject*` casts (lines 311, 484, 535, 571, 741). Any conversion to `intp_t`/`float64_t` from `_typedefs` still needs to preserve `cnp.PyArrayObject` (there is no `_typedefs` alias for it) and keep `cnp.import_array()` if any array creation is added — but `_tree.pyx` currently omits `cnp.import_array()` entirely (contrast `_reachability.pyx:41`). If the migration inadvertently removes `cimport numpy as cnp`, all `PyArray_SHAPE(<cnp.PyArrayObject*>...)` casts break.
scenario: "author converts _tree.pyx to `_typedefs` per F24's contract → removes `cimport numpy as cnp` on cleanup → build fails on `cnp.PyArrayObject`; or keeps `cimport numpy as cnp` but forgets `cnp.import_array()` → runtime segfault on first PyArray_SHAPE call"
contract: extend F24 to explicitly retain `cimport numpy as cnp` for `cnp.PyArrayObject*` casts, add `cnp.import_array()` at module scope in `_tree.pyx` (matching `_reachability.pyx`), and only replace scalar dtypes with `_typedefs` imports.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:33, sklearn/cluster/_hdbscan/_tree.pyx:39-40, sklearn/cluster/_hdbscan/_tree.pyx:311, sklearn/cluster/_hdbscan/_tree.pyx:484, sklearn/cluster/_hdbscan/_tree.pyx:535]
```

```
### F1 — Plan says `cnp.*_t` typing was replaced with `*_t` from `_typedefs.pxd`, but many `cnp.*_t` usages remain in the changed files
severity: medium
evidence:
- Plan of record `/work/review-harness/runs/26385/PLAN.md:15` lists as a novel change #1: "Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`".
- Actual tree still uses `cnp.*_t` heavily: `sklearn/cluster/_hdbscan/_tree.pyx` has 90 `cnp.(intp_t|float64_t|uint8_t|int64_t|int32_t)` occurrences (e.g. `_tree.pyx:39` `cdef cnp.float64_t INFTY = np.inf`, `_tree.pyx:40` `cdef cnp.intp_t NOISE = -1`, `_tree.pyx:83`, `:90`, `:117`, `:119`, ... through `:700`).
- Beyond ndarray dtype slots, standalone scalar typings such as `cnp.intp_t bfs_root`, `cnp.intp_t node`, `cnp.float64_t lambda_value` remain in `_tree.pyx` even though `_tree.pxd` (added in this PR at `sklearn/cluster/_hdbscan/_tree.pxd:36`) already `cimport`s `intp_t, float64_t, uint8_t` from `..._typedefs`. The sibling files `_linkage.pyx` and `_reachability.pyx` do use the unprefixed `intp_t`, `float64_t`, etc. — so the substitution was applied to those files but not to `_tree.pyx`.
scenario: "Reviewer trusts the plan's novel-changes inventory that `cnp.*_t` → `*_t` is done → skips re-reviewing typings in `_tree.pyx` → the incomplete substitution ships unnoticed for the largest changed Cython module."
contract: Plan claims to describe the deltas this PR actually introduces on top of the already-reviewed sub-PRs; readers rely on that inventory being accurate for their diff-review budget.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:39, sklearn/cluster/_hdbscan/_tree.pyx:40, sklearn/cluster/_hdbscan/_tree.pyx:83, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:117, sklearn/cluster/_hdbscan/_tree.pyx:119, sklearn/cluster/_hdbscan/_tree.pyx:700]
```
</details>

---

## Low severity

### [low] U17 — `_compute_stability` duplicated `births` allocation

Two back-to-back `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` calls; the first is immediately overwritten. Delete the duplicate.

<details><summary>verbatim finding</summary>

```
### F11 — `_compute_stability` duplicated `births` allocation
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` is executed twice back-to-back with no intervening use. The first allocation is dead code (immediately overwritten). Per the PR description item 3 ("Trimmed unused variables (thanks to Cython linting pre-commit)"), this is exactly the sort of leftover Cython linting should have flagged.
scenario: "`_compute_stability` is invoked during every HDBSCAN fit → the redundant `np.full` allocation runs on every call, wasting memory and signaling that the advertised Cython cleanup was incomplete"
contract: Delete the duplicate line at :252 (keep :254).
instances: single-instance
```
</details>

### [low] U18 — `remap_single_linkage_tree` docstring conflicts with actual call-site type

The docstring documents `non_finite` as a boolean ndarray; the call site passes a Python `set` of integer indices, which the body implicitly relies on. Rewrite the docstring and pass a deterministically ordered sequence (`sorted(...)`).

<details><summary>verbatim finding</summary>

```
### F12 — `remap_single_linkage_tree` docstring conflicts with actual call-site type
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 documents `non_finite : ndarray  Boolean array of which entries in the raw data are non-finite`, but the call site at :838 passes `non_finite=set(infinite_index + missing_index)` — a Python set of integer raw-data indices. The function's body relies on that set behaviour (`len(non_finite)` at :369 and `for i, outlier in enumerate(non_finite)` at :388 where `outlier` is used as a raw-data index at :389), which would not work with a boolean array. Iterating a `set` also gives non-deterministic order across Python invocations, so `self._single_linkage_tree_` for the non-finite path is nondeterministic between runs.
scenario: "reader relies on the docstring → passes a boolean mask and gets wrong results or an exception; separately, `_single_linkage_tree_` values differ across process invocations for otherwise identical inputs, complicating debugging and downstream serialization"
contract: Rewrite the docstring to state that `non_finite` is a collection of raw-data indices of non-finite rows, and pass a `sorted(...)` sequence (e.g. `sorted(set(infinite_index + missing_index))`) at :838 so ordering is deterministic.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:364, sklearn/cluster/_hdbscan/hdbscan.py:388, sklearn/cluster/_hdbscan/hdbscan.py:838]
```
</details>

### [low] U19 — Broken `:ref:` anchors — `<HDBSCAN>` and example anchor variants

`:ref:\`User Guide <HDBSCAN>\`` uses uppercase but the target label at `clustering.rst:51` is lowercase `hdbscan`. Sibling anchors around the class docstring and the gallery example likewise depend on the sphinx-gallery `.py`-suffix convention. Fix the casing and run a full docs build to confirm both anchors resolve.

<details><summary>verbatim finding</summary>

```
### F13 — Cross-reference `:ref:`User Guide <HDBSCAN>`` in example uses wrong-case anchor
severity: low
evidence: examples/cluster/plot_hdbscan.py:110 — `see :ref:`User Guide <HDBSCAN>``. The corresponding label defined in `doc/modules/clustering.rst:51` is `.. _hdbscan:` (lowercase). Sphinx label lookups are case-sensitive, so this cross-reference fails to resolve and the rendered example will emit a Sphinx warning and produce a broken link.
scenario: "docs build → Sphinx warns about unknown label `HDBSCAN`; user clicking the User Guide link in the rendered example hits a dead reference"
contract: Change the reference to `:ref:`User Guide <hdbscan>`` to match the declared label.
instances: single-instance
```

```
### F41 — additional instances of F13: broken `:ref:` anchors also in the User Guide topic block and in `plot_hdbscan.py` topic references
severity: low
evidence: examples/cluster/plot_hdbscan.py:110 (already flagged in F13) is one instance. Sibling issue: the class docstring `HDBSCAN` at sklearn/cluster/_hdbscan/hdbscan.py:419 correctly references `:ref:`User Guide <hdbscan>`` (lowercase), but the sibling example cross-reference at :422 `:ref:`plotting demo <sphx_glr_auto_examples_cluster_plot_hdbscan.py>`` depends on the example being generated with exactly that anchor. sphinx-gallery auto-generates `sphx_glr_auto_examples_cluster_plot_hdbscan.py` (with the trailing `.py`) — this convention has historically varied between sphinx-gallery versions (some emit without `.py`). If the site's sphinx-gallery config strips the `.py`, this ref breaks too. Not clearly wrong, but paired with F13 it suggests the example-related anchors were not verified in a docs build.
scenario: "docs build with a sphinx-gallery version that strips `.py` from generated labels → the class-level `See :ref:`plotting demo ...`` link breaks silently, in addition to F13's `<HDBSCAN>` casing issue"
contract: run a full docs build and confirm both `hdbscan.py:419` and `hdbscan.py:422` resolve to real anchors; if the trailing-`.py` variant is used, keep as-is; otherwise adjust.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:419, sklearn/cluster/_hdbscan/hdbscan.py:422, examples/cluster/plot_hdbscan.py:110]
```
</details>

### [low] U20 — `metric_params={'max_distance': …}` forwarded blindly to `pairwise_distances` / `NearestNeighbors`

Both `_hdbscan_brute` (line 238) and `_hdbscan_prims` (lines 337, 344) forward `**metric_params` to callees that do not accept `max_distance`. Setting the documented sparse-precomputed key on a non-precomputed metric raises `TypeError`. Pop `max_distance` before forwarding in both helpers.

<details><summary>verbatim finding</summary>

```
### F14 — `_hdbscan_brute` forwards `metric_params` to `pairwise_distances` including `max_distance`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:238-243 — the non-precomputed branch calls `pairwise_distances(X, metric=metric, n_jobs=n_jobs, **metric_params)` and only afterwards extracts `max_distance = metric_params.get("max_distance", 0.0)`. `pairwise_distances` does not accept a `max_distance` keyword, so any user who sets `metric_params={"max_distance": ...}` with a non-precomputed metric gets `TypeError: ... got an unexpected keyword argument 'max_distance'`. The docstring at sklearn/cluster/_hdbscan/hdbscan.py:459-460 advertises `metric_params : dict` as generic "Arguments passed to the distance metric" without excluding this key.
scenario: "User sets `metric_params={'max_distance': 5.0}` (documented for sparse precomputed) but with `metric='euclidean'` on a dense array → `pairwise_distances` rejects the kwarg and raises TypeError instead of quietly ignoring it"
contract: Pop `max_distance` out of `metric_params` before forwarding to `pairwise_distances`, and treat it as a top-level runtime option.
instances: single-instance
```

```
### F2 — rule of D:F14 recurs in `_hdbscan_prims` — `metric_params` including `max_distance` forwarded to `NearestNeighbors`/`DistanceMetric.get_metric`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:332-344 — `_hdbscan_prims` receives `**metric_params` and forwards them via `NearestNeighbors(..., metric_params=metric_params, ...)` and `DistanceMetric.get_metric(metric, **metric_params)`. Neither call site strips the `max_distance` key documented for sparse precomputed inputs (see F14's discussion of `_hdbscan_brute`). Setting `HDBSCAN(algorithm='kdtree', metric_params={'max_distance': 5.0})` yields `DistanceMetric.get_metric("euclidean", max_distance=5.0)` → `TypeError` from Cython dispatch. F14's contract ("pop `max_distance` out of `metric_params` before forwarding") therefore must be applied in both `_hdbscan_brute` and `_hdbscan_prims`.
scenario: "user copies the sparse-precomputed `metric_params={'max_distance': ...}` idiom into a tree-based HDBSCAN call → TypeError from DistanceMetric.get_metric or NearestNeighbors metric constructor"
contract: In `_hdbscan_prims`, pop `max_distance` (and any other reserved runtime keys) out of `metric_params` before it reaches `NearestNeighbors` or `DistanceMetric.get_metric`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:337, sklearn/cluster/_hdbscan/hdbscan.py:344]
```
</details>

### [low] U21 — `whats_new` reference to `DBSCAN` missing module prefix

`:class:\`DBSCAN\`` in the 1.3 whats-new omits `cluster.` — the sibling references use `cluster.OPTICS`/`cluster.HDBSCAN`. Under nitpicky mode Sphinx emits an unresolved-reference warning and the link is broken.

<details><summary>verbatim finding</summary>

```
### F15 — `whats_new` reference to `DBSCAN` missing module prefix
severity: low
evidence: doc/whats_new/v1.3.rst:194-196 — reads `Similarly to :class:`cluster.OPTICS`, it can be seen as a generalization of :class:`DBSCAN` by allowing for hierarchical instead of flat clustering`. The `:class:` role on `DBSCAN` omits the `cluster.` module prefix used everywhere else in the paragraph (and used for `cluster.HDBSCAN`, `cluster.OPTICS`), so Sphinx cannot resolve the cross-reference.
scenario: "Doc build for 1.3 whats-new runs with `nitpicky` mode → emits an unresolved-reference warning and renders `DBSCAN` as unlinked literal text instead of a link to the DBSCAN class"
contract: Change to `:class:`cluster.DBSCAN`` to match the sibling references.
instances: single-instance
```
</details>

### [low] U22 — Internal helper docstrings misdocument defaults and list a nonexistent `copy` parameter

`_hdbscan_brute` docstring says `min_samples=None, alpha=1.0` while the signature is `min_samples=5, alpha=None`. `_hdbscan_prims` docstring documents a `copy` parameter that does not exist in the signature. `_brute_mst`'s docstring lists a `default=None` for a required parameter. Sync each docstring to the actual signature.

<details><summary>verbatim finding</summary>

```
### F16 — `_hdbscan_brute` and `_hdbscan_prims` docstrings misdocument defaults and list an unused parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:163-172 (signature: `min_samples=5, alpha=None`) vs. docstring at 185/189 (`min_samples : int, default=None`, `alpha : float, default=1.0`); sklearn/cluster/_hdbscan/hdbscan.py:275-283 (signature: `min_samples=5`, no `copy` parameter) vs. docstring at 296 (`min_samples : int, default=None`) and 319-324 (documents `copy : bool, default=False` that does not exist in the signature).
scenario: "readers of the internal API get the wrong defaults from the docstrings → anyone extending or calling these internals passes the docstring's `None`/`copy=False` and gets silently different behavior; the phantom `copy` parameter also misleads anyone trying to add copy semantics to the tree algorithms."
contract: sync each `default=` and parameter list to the actual signatures — `min_samples : int, default=5`; `alpha : float, default=None` in `_hdbscan_brute` (or change the signature to `alpha=1.0` if the default was the intent); delete the `copy` docstring block from `_hdbscan_prims`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:185, sklearn/cluster/_hdbscan/hdbscan.py:189, sklearn/cluster/_hdbscan/hdbscan.py:296, sklearn/cluster/_hdbscan/hdbscan.py:319]
```

```
### F3 — rule of D:F16 recurs in `_brute_mst` — docstring documents a `min_samples` default that the signature does not provide
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:82,95 — signature is `def _brute_mst(mutual_reachability, min_samples):` (no default), but the docstring states `min_samples : int, default=None`. Same defect pattern as F16 for `_hdbscan_brute`/`_hdbscan_prims`: the documented default contradicts the runtime signature, misleading anyone who calls the helper with defaults.
scenario: "internal caller invokes `_brute_mst(matrix)` as the docstring suggests → `TypeError: _brute_mst() missing 1 required positional argument: 'min_samples'`; the documented `default=None` is a phantom"
contract: Edit the docstring at `hdbscan.py:95` to remove the `default=None` clause so the parameter is documented as `min_samples : int` matching the signature; sync with F16's fix so all three helpers' documented defaults are consistent.
instances: single-instance
```
</details>

### [low] U23 — User-guide typos and stray colons in the new HDBSCAN section

`doc/modules/clustering.rst` around the new HDBSCAN section renders trailing `:` characters literally after two `:math:` roles, spells "at this staged marked" (should be "stage"), and breaks "fully-connected" across lines. Fix the punctuation and grammar.

<details><summary>verbatim finding</summary>

```
### F17 — User-guide typos and stray colons in the new HDBSCAN section
severity: low
evidence: doc/modules/clustering.rst:1009-1011 — "removing any edges with value greater than :math:`\varepsilon`:\nfrom the original graph. Any points whose core distance is less than :math:`\varepsilon`:\nare at this staged marked as noise." Both math roles carry a trailing colon that renders literally, and "at this staged marked" should be "at this stage marked". doc/modules/clustering.rst:1025-1026 — "the fully\n-connected mutual reachability graph" renders with a stray hyphen/space split.
scenario: "Sphinx build renders the trailing `:` literally after each formula and prints 'at this staged marked' → published user guide has broken punctuation and a grammar error immediately after the algorithm's central formulas, which readers arriving at the new section see first."
contract: remove the trailing `:` after each `:math:` role, fix "staged" → "stage", and rejoin "fully-connected" onto one line.
instances: [doc/modules/clustering.rst:1009, doc/modules/clustering.rst:1010, doc/modules/clustering.rst:1011, doc/modules/clustering.rst:1025]
```
</details>

### [low] U24 — `_hierarchical_fast.pxd` declares `noexcept` on methods whose `.pyx` implementations omit it

Under Cython 3.x this triggers a signature-mismatch warning; without `noexcept` in the pyx, the compiler emits exception-propagation stubs at every call site, negating the pxd benefit. Add `noexcept` to both `.pyx` methods.

<details><summary>verbatim finding</summary>

```
### F18 — `_hierarchical_fast.pxd` declares `noexcept` on methods whose `.pyx` implementations omit it
severity: low
evidence: sklearn/cluster/_hierarchical_fast.pxd:7-8 declares `cdef void union(self, intp_t m, intp_t n) noexcept` and `cdef intp_t fast_find(self, intp_t n) noexcept`; sklearn/cluster/_hierarchical_fast.pyx:331,339 defines them without the `noexcept` specifier. Under Cython 3.x this triggers a compile-time signature-mismatch warning; without `noexcept` the compiler must generate exception-propagation stubs on every call, defeating the point of the pxd declaration.
scenario: "Cython 3 compilation of `_hierarchical_fast.pyx` → emits `noexcept` mismatch warnings and inserts exception-check code around every call site, making `UnionFind.union` / `fast_find` slower and diverging from the header contract"
contract: Add `noexcept` to both `.pyx` method signatures so the definition matches the `.pxd` declaration.
instances: [sklearn/cluster/_hierarchical_fast.pyx:331, sklearn/cluster/_hierarchical_fast.pyx:339]
```
</details>

### [low] U25 — Deprecated `np.core.records` and `np.infty` usage

`hdbscan.py:128` uses the private `np.core.records.fromarrays`; `_linkage.pyx:99,165` use the deprecated `np.infty`. NumPy 2.x further tightens both. Switch to `np.rec.fromarrays` and `np.inf`, and add a test that exercises the deprecated `_brute_mst` code path (sparse precomputed with missing entries) so CI catches the removal.

<details><summary>verbatim finding</summary>

```
### F19 — Deprecated `np.core.records` and `np.infty` usage
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:128 `mst = np.core.records.fromarrays(...)` — `np.core` is a private submodule that NumPy 1.25+ warns is on deprecation track (public spelling is `np.rec.fromarrays`). sklearn/cluster/_hdbscan/_linkage.pyx:99,165 use `np.infty`, which was deprecated in NumPy 1.20 in favor of `np.inf`.
scenario: "NumPy 2.x removes `np.infty` and further tightens `np.core` access → HDBSCAN imports emit DeprecationWarning and eventually fail"
contract: Replace `np.core.records.fromarrays` with `np.rec.fromarrays` and `np.infty` with `np.inf`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:128, sklearn/cluster/_hdbscan/_linkage.pyx:99, sklearn/cluster/_hdbscan/_linkage.pyx:165]
```

```
### F28 — additional instances of F19: deprecated `np.infty` also used in `_tree.pyx`
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:39 — `cdef cnp.float64_t INFTY = np.inf` at module scope uses `np.inf` correctly, but the merged.md F19 only flagged `np.infty` in `_linkage.pyx:99,165` and `np.core.records` in `hdbscan.py:128`. A broader grep confirms `np.infty` and `np.core` do not appear elsewhere in the new HDBSCAN files, so this is a completeness note rather than a new site — F19's contract already covers everything. However, F19 missed that the class docstring examples at sklearn/cluster/_hdbscan/hdbscan.py:604-613 do not exercise the `np.core.records` path (they run on `load_digits`, a purely finite array), so no doctest guards the deprecation; the deprecation will only surface for non-finite inputs.
scenario: "when NumPy removes `np.core` / `np.infty`, the doctest in the class docstring still passes on finite data → CI stays green while every non-finite fit fails at import time inside `_brute_mst`"
contract: extend F19's fix to also add a targeted test that constructs a sparse precomputed matrix with a missing entry (exercising `_brute_mst` line 128) so that the deprecation is caught during regression testing, not only in production.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:128, sklearn/cluster/_hdbscan/_linkage.pyx:99, sklearn/cluster/_hdbscan/_linkage.pyx:165]
```
</details>

### [low] U26 — `bfs_from_hierarchy` and `bfs_from_cluster_tree` allocate Python lists in Cython hot loops

Both BFS helpers declare a `cdef list` queue and use Python-level list operations / `np.isin` / `.tolist()` on every iteration. They are called from `_condense_tree` and from `_get_clusters`/`epsilon_search`, so allocations dominate the condensation and cluster-selection hot paths. Replace each queue with a preallocated `intp_t[::1]` buffer with front/back indices and use a boolean membership array instead of `np.isin`.

<details><summary>verbatim finding</summary>

```
### F20 — `bfs_from_hierarchy` list-of-Python-ints in a hot loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:87-114 — `bfs_from_hierarchy` is declared `cdef list ...` and uses Python-level list comprehensions/`extend`. It is called once per non-tiny internal node from `_condense_tree` at lines 200, 207, 216, 225. For a 10⁴-sample dataset this dominates runtime and is why HDBSCAN condensation is materially slower than the reference implementation. The plan of record's item "Clean `_hdbscan/_tree.pyx`" was expected to reach Cython-idiomatic code; this function is still a Python routine wrapped in `cdef`.
scenario: "user runs `HDBSCAN().fit(X)` on a large dataset → hot-path condensation is bottlenecked on Python object allocation inside `bfs_from_hierarchy`, and sub-cluster BFS iterations grow O(cluster_count²) worst-case"
contract: Replace the Python `list` queue with a preallocated `intp_t[::1]` buffer and index-tracked front/back pointers, and remove per-node Python allocations.
instances: single-instance
```

```
### F4 — rule of D:F20 recurs in `bfs_from_cluster_tree` — Python-object BFS queue in a Cython hot loop
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:279-296 — `bfs_from_cluster_tree` is declared `cdef list` and its main loop performs `np.isin(parents, process_queue)` plus `.tolist()` on Python `list` objects, which allocates new ndarrays and Python lists at each iteration. It is invoked from `epsilon_search` (line 632) and `_get_clusters` (line 737), both in the cluster-selection hot path. Same defect pattern as F20 for `bfs_from_hierarchy`: the "Cython lint cleanup" advertised in the PR description missed this sibling routine.
scenario: "user runs `HDBSCAN(cluster_selection_method='leaf', cluster_selection_epsilon=…).fit(X)` or `HDBSCAN(cluster_selection_method='eom').fit(X)` → the flat-clustering step allocates a new `np.isin` boolean array and a fresh Python list on every BFS layer, dominating post-condensation runtime"
contract: Replace `process_queue` with a preallocated `intp_t[::1]` buffer and front/back indices, and use a boolean membership array indexed by `parent` instead of `np.isin` at each iteration — mirroring F20's fix.
instances: single-instance
```
</details>

### [low] U27 — Sparse-precomputed input lacks square/symmetric validation

Shape and symmetry checks live inside the brute-dense branch; sparse precomputed input skips both. Non-square or asymmetric sparse matrices fail obscurely inside `mutual_reachability_graph` instead of raising the documented error. Move validation to a shared entry point and add coverage.

<details><summary>verbatim finding</summary>

```
### F21 — Docstring says "distance matrix must be square" but validation checks only equal dims (redundant), missing "same value on transpose"-only checks for sparse
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:222-234 vs 734-740 — the brute path guards precomputed symmetry via `_allclose_dense_sparse(X, X.T)` and `X.shape[0] != X.shape[1]`; the sparse-precomputed branch at lines 734-740 skips both checks. A user passing a non-square or asymmetric sparse precomputed matrix will not receive the documented error; instead an obscure downstream failure in `mutual_reachability_graph` occurs.
scenario: "user calls `HDBSCAN(metric='precomputed').fit(sparse_matrix)` with a non-square sparse matrix → obscure downstream error instead of the documented ValueError"
contract: Move the shape/symmetry validation out of the brute branch (lines 222-234) into the shared entry point so it applies to sparse precomputed input as well; add tests covering sparse-precomputed shape and symmetry errors.
instances: single-instance
```
</details>

### [low] U28 — `test_hdbscan_min_cluster_size` skips its assertion when every point is noise

The loop `for min_cluster_size in range(2, len(X)):` runs 198 fits, but the sole assertion is guarded by `if len(true_labels) != 0`. Large `min_cluster_size` values that classify every sample as noise pass the test silently. Bound the range or assert an explicit "no clusters" condition.

<details><summary>verbatim finding</summary>

```
### F22 — `test_hdbscan_min_cluster_size` never actually asserts noise handling when `min_cluster_size` exceeds `n_samples/2`
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:260-269 — the loop `for min_cluster_size in range(2, len(X), 1):` iterates through 198 fit-predict runs, but the body only asserts `np.min(np.bincount(true_labels)) >= min_cluster_size` **inside** `if len(true_labels) != 0`. There is no assertion for the many cases where every point becomes noise and `true_labels` is empty; the loop silently accepts "no clusters" for large `min_cluster_size` values. Given that `_min_samples` defaults to `min_cluster_size` and the check at hdbscan.py:758 forbids `_min_samples > X.shape[0]`, the top of the range approaches that limit with no explicit expectation — a change that makes the classifier degrade silently (e.g. always emitting -1) would keep this test green.
scenario: "regression turns every HDBSCAN run into all-noise for large min_cluster_size → test still passes because the assertion is skipped when `true_labels` is empty"
contract: Assert an explicit condition for the "no clusters" case (e.g. bound the min_cluster_size range so at least one cluster is guaranteed, or assert that `len(true_labels) > 0` above some cutoff).
instances: single-instance
```
</details>

### [low] U29 — Stale comment describes wrong outlier labels

`hdbscan.py:831-833` says "np.inf → -1, np.nan → -2" while `_OUTLIER_ENCODING` defines them as `-2` and `-3`. Update the comment.

<details><summary>verbatim finding</summary>

```
### F23 — Stale comment describes wrong outlier labels
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — comment reads "Samples with np.inf are mapped to -1 and those with np.nan are mapped to -2." The very next lines use `_OUTLIER_ENCODING["infinite"]["label"]` = `-2` and `_OUTLIER_ENCODING["missing"]["label"]` = `-3` (see the encoding definitions at lines 72-84 and the class docstring at lines 540-543 which correctly document `-2` / `-3`).
scenario: "a maintainer reads the comment to reason about the label semantics → mis-conflates infinite with -1 (which is generic noise) and missing with -2 (which is infinite) → wrong changes to downstream code"
contract: Update the comment to state "np.inf → -2 and np.nan → -3", matching the class docstring and `_OUTLIER_ENCODING`.
instances: single-instance
```
</details>

### [low] U30 — Docstring uses capitalized `"KDTree"`/`"BallTree"` as parameter values

The `algorithm` StrOptions accepts lowercase `"kdtree"`/`"balltree"` only. The docstring prose uses the capitalized forms, tempting users into `algorithm="KDTree"` which raises `InvalidParameterError`. Use lowercase in the docstring.

<details><summary>verbatim finding</summary>

```
### F25 — Docstring uses capitalized "KDTree"/"BallTree" as if they were the parameter values
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:472-478 — "algorithm : {"auto", "brute", "kdtree", "balltree"}, default="auto" ... Both `"KDTree"` and `"BallTree"` algorithms use the :class:`~sklearn.neighbors.NearestNeighbors` estimator." The accepted parameter values are lowercase `"kdtree"`/`"balltree"` (matches `_parameter_constraints` and `StrOptions`).
scenario: "user copy-pastes `algorithm=\"KDTree\"` from the description prose → `_validate_params()` raises `InvalidParameterError` because the value does not match the lowercase StrOptions set"
contract: Refer to the algorithm strings only in their true lowercase form throughout the docstring — `\"kdtree\"` / `\"balltree\"`.
instances: single-instance
```
</details>

### [low] U31 — User-guide references undefined parameter name `minimum_cluster_size`

The clustering.rst text names `minimum_cluster_size` while the actual parameter is `min_cluster_size`. Copying the recommendation verbatim yields `TypeError`. Rename in both occurrences.

<details><summary>verbatim finding</summary>

```
### F26 — User-guide references undefined parameter name `minimum_cluster_size`
severity: low
evidence: doc/modules/clustering.rst:138-141 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples` to couple the parameters"; the actual estimator parameter is `min_cluster_size` (hdbscan.py:649, 435)
scenario: "User copies the recommendation verbatim → `HDBSCAN(minimum_cluster_size=min_samples)` raises `TypeError: unexpected keyword argument` (or the parameter is silently ignored, depending on estimator plumbing)"
contract: rename both occurrences to the true parameter, `min_cluster_size`
instances: [doc/modules/clustering.rst:139, doc/modules/clustering.rst:140]
```
</details>

### [low] U32 — `internal_to_raw` built as a dict for what is a plain index lookup

`internal_to_raw = {x: y for x, y in enumerate(finite_index)}` builds an O(n) hashmap for what could be ndarray indexing. Pass `finite_index` directly and index it.

<details><summary>verbatim finding</summary>

```
### F27 — `internal_to_raw` built as a dict for what is a plain index lookup
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:732 — `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` where `finite_index` is already an ndarray of positional integers; the callee at hdbscan.py:375/379 only ever does `internal_to_raw[left]` / `[right]` where `left`/`right` are `< finite_count`
scenario: "`HDBSCAN.fit` runs on data with any non-finite rows → the dict comprehension allocates one Python-int mapping per finite sample (O(n) hashmap entries) purely to reproduce ndarray indexing, dominating the outlier-handling path for large `n_samples` with a small non-finite fraction"
contract: pass the `finite_index` ndarray directly and use ndarray indexing in `remap_single_linkage_tree` (`internal_to_raw[left]` becomes exactly the same expression); update the docstring accordingly
instances: single-instance
```
</details>

### [low] U33 — Typos throughout the new HDBSCAN files — `simbling`, `smaler`, `homogenous`, `reahability`, `collecteion`, `reachibility`, `mututal`

New HDBSCAN prose and identifiers ship with multiple misspellings: `simbling`, `smaler`, `homogenous`, `reahability` (5×), `collecteion` (~9×), plus `homogenous` in the gallery example and the pervasive `reachibility` (missing `a`) as an identifier and test filename in `_reachability.pyx` / `test_reachibility.py`. Fix each in prose; for identifiers, rename the local variable, the test file, and any imports/CI references.

<details><summary>verbatim finding</summary>

```
### F32 — additional instances of F17: user-guide typos beyond stray colons — "simbling", "smaler", "homogenous", "reahability", "collecteion"
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:503 "largest lambda of any simbling node" (also duplicated at :530 in code comment); sklearn/cluster/_hdbscan/_tree.pyx:133 "Clusters smaler than this are pruned"; sklearn/cluster/_hdbscan/_tree.pyx:906 "can't create a homogenous 3D array" (hdbscan.py:906); sklearn/cluster/_hdbscan/_linkage.pyx:75,102,137,229,234 "mutual-reahability graph" (repeated 5× — should be "reachability") and "collecteion of edges" (also 4× in the same docstrings); sklearn/cluster/_hdbscan/hdbscan.py:102-103 mirrors the same typos in `_brute_mst`. These are all newly-added strings from this PR (not preexisting) and are visible in the Sphinx-rendered API docs for `mst_from_mutual_reachability`, `mst_from_data_matrix`, and `make_single_linkage`.
scenario: "Sphinx builds the private Cython API documentation → the new HDBSCAN module ships with 'reahability', 'simbling', 'collecteion', 'smaler', 'homogenous' visible in the rendered API cross-reference and in every source-view rendering, embarrassing typography for a MajorFeature launch"
contract: fix "simbling" → "sibling"; "smaler" → "smaller"; "homogenous" → "homogeneous"; "reahability" → "reachability" (5 occurrences in `_linkage.pyx` docstrings + mirrors in `hdbscan.py`); "collecteion" → "collection" (~9 occurrences across the new files).
instances: [sklearn/cluster/_hdbscan/_tree.pyx:133, sklearn/cluster/_hdbscan/_tree.pyx:503, sklearn/cluster/_hdbscan/_tree.pyx:530, sklearn/cluster/_hdbscan/hdbscan.py:102, sklearn/cluster/_hdbscan/hdbscan.py:103, sklearn/cluster/_hdbscan/hdbscan.py:143, sklearn/cluster/_hdbscan/_linkage.pyx:75, sklearn/cluster/_hdbscan/_linkage.pyx:76, sklearn/cluster/_hdbscan/_linkage.pyx:137, sklearn/cluster/_hdbscan/_linkage.pyx:138, sklearn/cluster/_hdbscan/_linkage.pyx:229, sklearn/cluster/_hdbscan/_linkage.pyx:234]
```

```
### F5 — rule of D:F32 recurs — "homogenous" typo in the plot example
severity: low
evidence: examples/cluster/plot_hdbscan.py:116 — "Traditional DBSCAN assumes that any potential clusters are homogenous in density." F32's evidence flagged the identical typo in `hdbscan.py:906` but its `instances` list did not include the example file. `homogenous` is not standard English (the correct term is `homogeneous`, as used throughout the rest of the scikit-learn tree — see `doc/modules/clustering.rst:978`).
scenario: "Sphinx renders the auto-generated HDBSCAN gallery example → the reader-facing narrative displays 'homogenous', an obvious misspelling directly beneath the algorithm's headline feature comparison with DBSCAN"
contract: Change `homogenous` to `homogeneous` at `examples/cluster/plot_hdbscan.py:116`, alongside F32's fix at `hdbscan.py:906`.
instances: single-instance
```

```
### F6 — rule of D:F32 recurs — "reachibility" typo pervasively in `_reachability.pyx` code + test filename
severity: low
evidence: F32 flagged `reahability` (missing `c`) in docstrings of `_linkage.pyx` and `hdbscan.py`, but a distinct misspelling — `reachibility` (missing `a`) — is used consistently in `_reachability.pyx` **as an identifier**: `mutual_reachibility_distance` at sklearn/cluster/_hdbscan/_reachability.pyx:127, 144, 149, 182, 206, 209, 210, and the test module filename itself is `sklearn/cluster/_hdbscan/tests/test_reachibility.py`. Because these are identifiers/paths (not free-text prose), fixing them requires renames rather than a docstring edit — a scope F32 did not address.
scenario: "reviewers grep for `mutual_reachability_distance` (the correct spelling used in prose everywhere else) → get zero hits in the actual computation code, wasting review cycles; the misspelled test module name will render as `test_reachibility` in any test-report or CI summary, cementing the typo in perpetuity"
contract: Rename the local variable `mutual_reachibility_distance` → `mutual_reachability_distance` at every occurrence in `sklearn/cluster/_hdbscan/_reachability.pyx`; rename the test file to `test_reachability.py` and update any imports/CI configs that reference it.
instances: [sklearn/cluster/_hdbscan/_reachability.pyx:127, sklearn/cluster/_hdbscan/_reachability.pyx:144, sklearn/cluster/_hdbscan/_reachability.pyx:149, sklearn/cluster/_hdbscan/_reachability.pyx:182, sklearn/cluster/_hdbscan/_reachability.pyx:206, sklearn/cluster/_hdbscan/_reachability.pyx:209, sklearn/cluster/_hdbscan/_reachability.pyx:210, sklearn/cluster/_hdbscan/tests/test_reachibility.py]
```
</details>

### [low] U34 — `_brute_mst` docstring parameter name mismatch (`mututal_reachability_graph`)

Docstring names the first parameter `mututal_reachability_graph` (two-character typo + wrong name); signature is `mutual_reachability`. numpydoc lint flags this. Fix both errors.

<details><summary>verbatim finding</summary>

```
### F34 — Docstring at `_brute_mst` misnames the parameter (`mututal_reachability_graph` instead of `mutual_reachability`)
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:91-93 — the docstring lists `mututal_reachability_graph: {ndarray, sparse matrix} of shape (n_samples, n_samples)` but the function signature at :82 is `def _brute_mst(mutual_reachability, min_samples):`. Beyond the two-character typo (`mututal`), the parameter is named `mutual_reachability` — `mututal_reachability_graph` matches no argument. numpydoc validators (used by scikit-learn CI) flag this as an unknown-parameter docstring entry.
scenario: "numpydoc lint runs → warns 'Parameters {mutual_reachability} not documented' and 'Unknown parameter mututal_reachability_graph in docstring'; the internal API doc rendered by Sphinx shows a nonexistent parameter"
contract: rename the docstring entry to `mutual_reachability` matching the signature; fix the typo `mututal` → `mutual`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:91]
```
</details>

### [low] U35 — `fit_predict` on non-finite data is not exercised by any test

None of the 30 tests calls `HDBSCAN(...).fit_predict(X_outlier)`; every non-finite test goes through the two-call `.fit(...).labels_` form. A regression that broke `fit_predict`'s handling of outlier rows would slip through. Add a parametrized test covering `fit_predict(X_with_nan_and_inf)`.

<details><summary>verbatim finding</summary>

```
### F36 — `HDBSCAN` estimator does not implement `predict`/`fit_predict` for out-of-sample data yet documents `fit_predict` — semantically fine, but tests never cover `fit_predict` on non-finite data
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:858-877 — `fit_predict` calls `self.fit(X); return self.labels_`, so it exposes the same non-finite handling as `fit`. However, sklearn/cluster/tests/test_hdbscan.py has 30 test functions and only 7 exercise `fit_predict` (test_hdbscan_distance_matrix line 79, test_hdbscan_sparse_distance_matrix line 116, test_hdbscan_feature_array line 126, test_hdbscan_algorithms line 143, test_dbscan_clustering line 186, test_hdbscan_no_clusters line 249, test_hdbscan_min_cluster_size line 260, test_hdbscan_callable_metric line 271, test_hdbscan_allow_single_cluster_with_epsilon lines 342/357). None of them pass non-finite data, so the code paths in `fit_predict` that route through the non-finite branch of `fit` are covered only via `test_outlier_data` (which calls `.fit(X_outlier)`, not `.fit_predict(X_outlier)`) and `test_dbscan_clustering_outlier_data` (which calls `.fit`, then `.dbscan_clustering`). Result: `fit_predict` on non-finite data is unlockable by the test suite; a regression that broke the reassignment logic (e.g. F1, F2, F33) would still ship if `fit_predict` were the affected entry point.
scenario: "user's primary API is `HDBSCAN().fit_predict(X_outlier)` → regression sneaks past the test suite because tests target the two-call `.fit(X).labels_` form only"
contract: add a parametrized test that runs `HDBSCAN(store_centers='centroid').fit_predict(X_with_nan_and_inf)` and asserts the returned labels match the reference in `test_outlier_data`; ensures F1, F2, F33 all remain caught if entry point is `fit_predict`.
instances: single-instance
```
</details>

### [low] U36 — `HDBSCAN` exposes no `n_iter_` / `n_clusters_` attribute unlike sibling cluster estimators

Design gap rather than a bug. Consider exposing `n_clusters_` (derived from `set(self.labels_) - OUTLIER_SET`) to match `AgglomerativeClustering.n_clusters_`.

<details><summary>verbatim finding</summary>

```
### F37 — `HDBSCAN` never sets `n_iter_` or any convergence attribute, unlike other cluster estimators listed in the same whats-new block
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:410-614 attribute docstring — lists `labels_`, `probabilities_`, `n_features_in_`, `feature_names_in_`, `centroids_`, `medoids_`. Other clustering estimators (`KMeans.n_iter_`, `OPTICS`) expose convergence/iteration counts; `_condense_tree` and `_get_clusters` do have a natural per-condensation iteration count. The absence is not a bug per se, but the class docstring at :575-580 lists `See Also: DBSCAN, OPTICS, Birch` — all of which expose additional attributes not present on HDBSCAN.
scenario: "user familiar with `KMeans` or `Birch` inspects `hdb.n_iter_` for convergence diagnostics → AttributeError; sklearn's cluster-estimator API doc pages will show HDBSCAN as an exception without a n_iter_ or similar diagnostic"
contract: N/A — this is a design gap, not a bug. If desired, expose `n_clusters_` (already trivially derivable as `len(set(self.labels_) - OUTLIER_SET)`) as an attribute, matching `AgglomerativeClustering.n_clusters_`.
instances: single-instance
```
</details>

### [low] U37 — `_hdbscan_prims` discards `kneighbors` indices, only to recompute all pairwise distances

`kneighbors(X, min_samples, return_distance=True)` returns neighbor indices but the caller drops them via `_`; `mst_from_data_matrix` then recomputes every pair. On large-n high-dim data this doubles the distance work. Either cache the k-nearest distances or switch to a boruvka-style k-NN MST.

<details><summary>verbatim finding</summary>

```
### F39 — `_hdbscan_prims` performs `NearestNeighbors.kneighbors` with `return_distance=True` and discards the neighbor indices, only to recompute pairwise distances inside `mst_from_data_matrix`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:342-347 — `neighbors_distances, _ = nbrs.kneighbors(X, min_samples, return_distance=True)`; the underscore discards the neighbor indices. Then `mst_from_data_matrix(X, core_distances, dist_metric, alpha)` at `_linkage.pyx:174-194` iterates over every pair `(i, j)` and recomputes `pair_distance = dist_metric.dist(&raw_data[i,0], &raw_data[j,0], num_features)` — the pairwise distances originally computed to derive core_distances are never reused. For high-dimensional data this doubles the distance work: kNN once (O(n log n) via the tree) plus O(n²) pairwise recomputation. The `NearestNeighbors` tree could be re-queried for a wider radius to derive most of the MST edges via a k-nearest-neighbor MST algorithm (as HDBSCAN* originally does), reducing the O(n²) work.
scenario: "user runs `HDBSCAN(algorithm='kdtree').fit(X)` on 10⁴ samples in 100 dimensions → the tree-based backend still incurs O(n²) distance recomputation inside Prim's, negating the tree's benefit for large n"
contract: use the neighbor indices returned by `kneighbors` (currently discarded via `_`) to prime `mst_from_data_matrix` with a boruvka-style k-nearest-neighbor MST, or at minimum cache the `neighbors_distances` matrix in `mst_from_data_matrix` so already-computed distances are reused.
instances: single-instance
```
</details>

### [low] U38 — Multiple condensed-tree / linkage docstrings claim `shape (n_samples,)` for variable-length edgelists

Several `_tree.pyx` docstrings say `shape (n_samples,)` for arrays whose actual length is `(n_samples-1,)` (linkage / hierarchy input) or `(n_edges,)` (condensed tree). Correct each docstring shape to match the true dimensions.

<details><summary>verbatim finding</summary>

```
### F40 — `_condense_tree` docstring says `condensed_tree` has shape `(n_samples,)` but the returned length is `n_samples - 1` or larger
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:137-141 — docstring says "condensed_tree : ndarray of shape (n_samples,), dtype=CONDENSED_dtype". Actual return at line 232 is `np.array(result_list, dtype=CONDENSED_dtype)` whose length is `2 * (# large-vs-large internal splits) + (# collapse-BFS single-child appends)`, generally on the order of `n_samples - 1` but with no strict equality. Similar problem in `_do_labelling` docstring at :447 ("condensed_tree : ndarray of shape (n_samples,)"), `_get_clusters` at :656 ("condensed_tree : ndarray of shape (n_samples,)"), and `get_probabilities` (implied). The docstrings copy-paste the wrong shape spec across all condensed-tree consumers.
scenario: "numpydoc / Sphinx renders the internal Cython API → users reading the docstrings expect `condensed_tree.shape == (n_samples,)` and index accordingly, then get IndexError because true shape is `(n_edges,)` where `n_edges` ranges roughly `[n_samples - 1, 2·n_samples]`"
contract: correct all four docstrings to state "shape (n_edges,)" where `n_edges` depends on the condensation; the CONDENSED_dtype array is a *variable-length edgelist*, not a fixed-length per-sample array.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:138, sklearn/cluster/_hdbscan/_tree.pyx:447, sklearn/cluster/_hdbscan/_tree.pyx:656]
```

```
### F7 — rule of D:F40 recurs — additional docstrings claim `shape (n_samples,)` for variable-length edge-list / linkage arrays
severity: low
evidence: F40 flagged `_tree.pyx:138, 447, 656`. Two more docstrings in the same file assert the same wrong shape for arrays that are actually of length `n_samples - 1`:
  - sklearn/cluster/_hdbscan/_tree.pyx:129 — `_condense_tree`'s input `hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` (actual shape is `(n_samples - 1,)` — see `_linkage.pyx:252` `single_linkage = np.zeros(n_samples - 1, ...)`).
  - sklearn/cluster/_hdbscan/_tree.pyx:371 — `labelling_at_cut`'s input `linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype` (again `(n_samples - 1,)`; `root = 2 * linkage.shape[0]` on line 396 depends on this and would break if the shape were truly `(n_samples,)`).
Same defect pattern as F40: shape docstrings copied wholesale across all consumers without matching the actual array size.
scenario: "reviewer or downstream user reads `_condense_tree`/`labelling_at_cut` docstrings → expects `hierarchy`/`linkage` to have length `n_samples` and slices/indexes accordingly, silently corrupting the linkage tree or triggering `IndexError` on the last row"
contract: Correct the two docstrings to state `shape (n_samples - 1,), dtype=HIERARCHY_dtype`, mirroring the correct shape declared in `_linkage.pyx:74, 136, 227, 233`.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:371]
```
</details>

### [low] U39 — Plan says `.shape[0]` → `len(*)` for ndarray objects; 15+ residual `.shape[0]` sites remain on ndarray operands

The novel-changes list claims this replacement is done, but ~15 residual `.shape[0]` calls remain on clearly-ndarray targets across `hdbscan.py`, `_tree.pyx`, and `_linkage.pyx`. Reviewers who trust the plan's inventory will skip these hunks.

<details><summary>verbatim finding</summary>

```
### F2 — Plan says `*.shape[0]` was replaced with `len(*)` for ndarray objects, but many `.shape[0]` calls on ndarray objects remain
severity: low
evidence:
- Plan `/work/review-harness/runs/26385/PLAN.md:16` novel change #2: "Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects".
- Grep on the changed tree finds 20 residual `.shape[0]` occurrences across 4 changed files; several are clearly on ndarray-typed objects:
  - Python side: `sklearn/cluster/_hdbscan/hdbscan.py:223 if X.shape[0] != X.shape[1]:`; `:385 tree[tree.shape[0] - 1]…`; `:387 last_cluster_size = tree[tree.shape[0] - 1]["cluster_size"]`; `:752 if X.shape[0] == 1:`; `:758 if self._min_samples > X.shape[0]:`; `:840 np.empty(self._raw_data.shape[0], …)`; `:846 np.zeros(self._raw_data.shape[0], …)`; `:896 mask = np.empty((X.shape[0],), …)`.
  - Cython on `cnp.ndarray[…]`-declared objects: `_tree.pyx:90 hierarchy.shape[0] + 1`, `:145 2 * hierarchy.shape[0]`, `:146 hierarchy.shape[0] + 1`, `:396 2 * linkage.shape[0]`, `:531 labels.shape[0]`, `:563 children.shape[0] == 0`.
  - `_linkage.pyx:153 n_samples = raw_data.shape[0]` (raw_data is passed in as an ndarray).
scenario: "Reviewer trusts the plan's assertion that `.shape[0]` → `len(*)` is complete for ndarrays → does not re-check `.shape[0]` sites in the diff → the incomplete substitution ships with 15+ residual ndarray-operand call sites."
contract: See F1.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:223, sklearn/cluster/_hdbscan/hdbscan.py:385, sklearn/cluster/_hdbscan/hdbscan.py:387, sklearn/cluster/_hdbscan/hdbscan.py:752, sklearn/cluster/_hdbscan/hdbscan.py:758, sklearn/cluster/_hdbscan/hdbscan.py:840, sklearn/cluster/_hdbscan/hdbscan.py:846, sklearn/cluster/_hdbscan/hdbscan.py:896, sklearn/cluster/_hdbscan/_tree.pyx:90, sklearn/cluster/_hdbscan/_tree.pyx:145, sklearn/cluster/_hdbscan/_tree.pyx:146, sklearn/cluster/_hdbscan/_tree.pyx:396, sklearn/cluster/_hdbscan/_tree.pyx:531, sklearn/cluster/_hdbscan/_tree.pyx:563, sklearn/cluster/_hdbscan/_linkage.pyx:15]
```
</details>

### [low] U40 — Unlisted refactor: new `_hierarchical_fast.pxd` + edit to `_hierarchical_fast.pyx`

The PR adds `sklearn/cluster/_hierarchical_fast.pxd` and edits `_hierarchical_fast.pyx` — promoting `UnionFind`'s cdef fields into a header — but the plan-of-record's novel-changes list does not sanction this refactor. Motivation exists (`_linkage.pyx` needs to `cimport UnionFind`) but reviewers who follow the plan's inventory will skip these hunks.

<details><summary>verbatim finding</summary>

```
### F3 — `_hierarchical_fast.pxd` (new) and edit to `_hierarchical_fast.pyx` are not sanctioned by any plan requirement
severity: low
evidence:
- Manifest `/work/review-harness/runs/26385/DIFF_MANIFEST.md:27-28`:
  - `A sklearn/cluster/_hierarchical_fast.pxd +9/-0`
  - `M sklearn/cluster/_hierarchical_fast.pyx +0/-5`
- The plan lists as novel changes only (1) `cnp.*_t` → `*_t`, (2) `.shape[0]` → `len`, (3) trim unused variables (`PLAN.md:14-17`). Promoting `UnionFind`'s cdef fields from the pyx to a new pxd is a public-API refactor of an unrelated module, not any of those three.
- Motivation is visible in the tree — `sklearn/cluster/_hdbscan/_linkage.pyx:39 from ...cluster._hierarchical_fast cimport UnionFind` requires `UnionFind` to have a pxd — but nothing in the plan of record sanctions this cross-module refactor.
scenario: "Reviewer uses the plan's novel-changes list as their diff-review inventory → never opens the `_hierarchical_fast.pxd`/`.pyx` hunks → an unlisted public-API refactor of an already-shipped module lands unreviewed."
contract: Plan is expected to enumerate every novel edit beyond the pre-reviewed sub-PRs; the `_hierarchical_fast.*` change is novel and unlisted.
instances: [sklearn/cluster/_hierarchical_fast.pxd:1, sklearn/cluster/_hierarchical_fast.pyx:5]
```
</details>

### [low] U41 — Unrelated whitespace-only churn in the Mean Shift section of `clustering.rst`

Trailing-whitespace strips on Mean Shift prose are bundled into the HDBSCAN estimator PR without any sanctioning plan item. Either split the edits out or note them explicitly in the PR description.

<details><summary>verbatim finding</summary>

```
### F4 — Unrelated whitespace-only edits in `doc/modules/clustering.rst`'s Mean Shift section are not sanctioned by any plan requirement
severity: low
evidence:
- `hunks/doc_modules_clustering.rst.diff:19-45` shows trailing-whitespace strips on Mean Shift prose (lines 399, 419, 422, 428) inside the same PR that adds the HDBSCAN section (which is sanctioned by the overall "Add HDBSCAN" goal).
- Plan `PLAN.md:4` scope is "Add `HDBSCAN` as a new estimator" plus the three novel-change items at `PLAN.md:14-17`; drive-by whitespace fixes to Mean Shift documentation are outside both.
scenario: "Reviewer treats the plan as authoritative inventory of doc edits → skips the Mean Shift hunks assuming they are HDBSCAN-scope → unrelated whitespace churn on unrelated docs is bundled into the estimator PR without notice."
contract: See F3.
instances: [doc/modules/clustering.rst:399, doc/modules/clustering.rst:419, doc/modules/clustering.rst:422, doc/modules/clustering.rst:428]
```
</details>
