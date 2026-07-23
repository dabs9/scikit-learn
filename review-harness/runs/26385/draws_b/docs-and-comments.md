### F1 — Wrong outlier label mapping in comment above remap
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:830-833 — "Remap indices to align with original data in the case of / non-finite entries. Samples with np.inf are mapped to -1 and / those with np.nan are mapped to -2."
scenario: "A maintainer reads this comment while debugging outlier handling → they trust the -1/-2 mapping and build wrong invariants; the actual encoding is np.inf → -2 and np.nan → -3 (see `_OUTLIER_ENCODING` at hdbscan.py:65-79 and the immediately-following assignments at hdbscan.py:842-843 which use `_OUTLIER_ENCODING['infinite']['label']` (=-2) and `['missing']['label']` (=-3))."
contract: Rewrite the comment to state that non-finite samples are mapped to -2 (`np.inf`) and -3 (`np.nan`), matching `_OUTLIER_ENCODING`.
instances: single-instance

### F2 — User Guide inverts the noise condition on core distance
severity: medium
evidence: doc/modules/clustering.rst:1009-1012 — "removing any edges with value greater than :math:`\varepsilon`: / from the original graph. Any points whose core distance is less than :math:`\varepsilon`: / are at this staged marked as noise."
scenario: "A user reads the HDBSCAN User Guide to understand DBSCAN* equivalence → they conclude that points with small core distance are noise, which is the reverse of the truth (a point is noise in DBSCAN* iff its core distance exceeds ε, i.e. it has fewer than `min_samples` neighbors within ε — this is exactly the condition documented at _reachability.pyx:64-72 and the DBSCAN* reference [CM2013])."
contract: Change "less than :math:`\varepsilon`" to "greater than :math:`\varepsilon`" so the noise criterion matches DBSCAN*.
instances: single-instance

### F3 — User Guide uses non-existent parameter name `minimum_cluster_size`
severity: medium
evidence: doc/modules/clustering.rst:1057-1061 — "HDBSCAN can be smoothed with an additional hyperparameter `min_cluster_size` / which specifies that during the hierarchical clustering, components with fewer / than `minimum_cluster_size` many samples are considered noise. In practice, one / can set `minimum_cluster_size = min_samples` to couple the parameters".
scenario: "A user copies the guidance `minimum_cluster_size = min_samples` into their code → `HDBSCAN(minimum_cluster_size=...)` raises TypeError because the constructor accepts only `min_cluster_size` (see hdbscan.py:649)."
contract: Replace both occurrences of `minimum_cluster_size` with `min_cluster_size` to match the actual public parameter.
instances: [doc/modules/clustering.rst:1059, doc/modules/clustering.rst:1060]

### F4 — Public `n_jobs` docstring claims `default=None` while __init__ defaults to 4
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:486 — "n_jobs : int, default=None" in the class docstring; hdbscan.py:658 — the signature is `n_jobs=4`.
scenario: "A user reads the class docstring, expects the sklearn-standard default (`None`, honoring the parallel_backend context) → in practice their fits always spawn 4 workers because the actual default is 4, contradicting the documented and sklearn-conventional behavior."
contract: Update the docstring to `default=4` so it matches the actual signature.
instances: single-instance

### F5 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — "copy : bool, default=False / If `copy=True` then any time an in-place modifications would be made / …" inside the `_hdbscan_prims` docstring; the function signature at hdbscan.py:269-278 has no `copy` parameter.
scenario: "A future maintainer reads the docstring and assumes `_hdbscan_prims` honors `copy` → they route the `HDBSCAN(copy=True)` case through the prims path expecting the guarantee, but `copy` is silently ignored because it never reaches this branch (see hdbscan.py:793-803 where `copy` is added to kwargs only for the brute branch)."
contract: Delete the `copy` block from `_hdbscan_prims`'s docstring; it describes behavior the function does not implement.
instances: single-instance

### F6 — `_sparse_mutual_reachability_graph` docstring lists a phantom parameter and omits all real ones
severity: medium
evidence: sklearn/cluster/_hdbscan/_reachability.pyx:152-179 — the signature exposes `data, indices, indptr, n_samples, further_neighbor_idx, max_distance`, but the docstring's Parameters block describes only `distance_matrix`, `further_neighbor_idx`, and `max_distance`.
scenario: "A maintainer reads the docstring to understand the calling contract → they see a `distance_matrix` argument that doesn't exist and no explanation of the raw CSR triple (`data`, `indices`, `indptr`, `n_samples`), which is the actual interface used by `mutual_reachability_graph` at _reachability.pyx:93-100."
contract: Rewrite the Parameters section to document `data`, `indices`, `indptr`, `n_samples`, `further_neighbor_idx`, `max_distance` — matching the signature — and drop the fictitious `distance_matrix`.
instances: single-instance

### F7 — `_hdbscan_brute` / `_hdbscan_prims` docstrings claim `min_samples` default is `None`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:179 and hdbscan.py:290 — both docstrings state "min_samples : int, default=None"; the signatures at hdbscan.py:160 and hdbscan.py:272 both use `min_samples=5`.
scenario: "A maintainer wiring a new caller relies on the stated `None` default → they omit the argument, expecting some sentinel that means 'auto', but the real default is the literal 5, silently producing different behavior."
contract: Change both docstrings to `min_samples : int, default=5`.
instances: [sklearn/cluster/_hdbscan/hdbscan.py:179, sklearn/cluster/_hdbscan/hdbscan.py:290]

### F8 — `_hdbscan_prims` docstring omits `algo` and `leaf_size`
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:269-278 — signature has `algo`, `leaf_size=40`; the docstring at hdbscan.py:279-326 documents `X`, `min_samples`, `alpha`, `metric`, `n_jobs`, `copy` (which doesn't exist — see F5), and `metric_params` but never mentions `algo` or `leaf_size`.
scenario: "A maintainer reads the docstring to understand how to invoke `_hdbscan_prims` → they cannot tell how `algo` selects kd_tree vs. ball_tree or what `leaf_size` controls, forcing them to inspect the call site to reverse-engineer semantics."
contract: Add Parameter entries for `algo` (values `{'kd_tree', 'ball_tree'}` passed to `NearestNeighbors`) and `leaf_size` (mirrors the class parameter), matching the entries in the `HDBSCAN` class docstring.
instances: single-instance

### F9 — `_brute_mst` docstring parameter name is misspelled and doesn't match the signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:91-93 — the docstring's parameter is written as "mututal_reachability_graph: {ndarray, sparse matrix} …" (with typo); the signature at hdbscan.py:82 uses `mutual_reachability`.
scenario: "A tooling pass (e.g. numpydoc validation) tries to associate the Parameters block with the signature → it fails to find `mututal_reachability_graph` and warns; readers looking for `mutual_reachability` in the doc block see a different name."
contract: Rename the docstring entry to `mutual_reachability` (correct spelling) to match the actual parameter.
instances: single-instance

### F10 — `mst_from_data_matrix` docstring omits `alpha`
severity: low
evidence: sklearn/cluster/_hdbscan/_linkage.pyx:111-138 — signature includes `float64_t alpha=1.0`, but the Parameters block documents only `raw_data`, `core_distances`, `dist_metric` and never `alpha`.
scenario: "A caller wanting to know what `alpha` does in the prims path finds no documentation → they must read the body to discover that it scales `pair_distance /= alpha` at _linkage.pyx:187, wasting time and risking misuse."
contract: Add an `alpha : float, default=1.0` entry to the Parameters block, mirroring the description used in `HDBSCAN`'s class docstring.
instances: single-instance

### F11 — Shape annotations in `_tree.pyx` docstrings are off by one
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:129 declares "hierarchy : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype"; sklearn/cluster/_hdbscan/_tree.pyx:371 declares "linkage : ndarray of shape (n_samples,), dtype=HIERARCHY_dtype"; the code itself asserts `n_samples = hierarchy.shape[0] + 1` (_tree.pyx:90, 146), so the actual shape is `(n_samples - 1,)`.
scenario: "A caller sizes an output buffer or asserts a shape based on the docstring `(n_samples,)` → they are off by one; the value elsewhere in this same PR (e.g. hdbscan.py:219, 325) correctly states `(n_samples - 1,)`, exposing the divergence."
contract: Change every `(n_samples,)` annotation for the single-linkage hierarchy in `_tree.pyx` docstrings to `(n_samples - 1,)`, matching the invariant enforced by the code and the `hdbscan.py` docstrings.
instances: [sklearn/cluster/_hdbscan/_tree.pyx:129, sklearn/cluster/_hdbscan/_tree.pyx:371]

### F12 — Stray trailing colons and typo in the HDBSCAN User Guide paragraph
severity: low
evidence: doc/modules/clustering.rst:1009-1011 — "…value greater than :math:`\varepsilon`: / from the original graph. Any points whose core distance is less than :math:`\varepsilon`: / are at this staged marked as noise." — two stray colons after the `:math:` roles and "at this staged" for "at this stage".
scenario: "The User Guide is rendered to HTML → the published page shows visible stray colons after each ε and prints 'at this staged', signalling low prose quality in an official sklearn document."
contract: Remove the extra colon after both `:math:`\varepsilon`` occurrences and fix "staged" to "stage".
instances: single-instance

### F13 — `test_hdbscan_precomputed_non_brute` docstring/test name mis-describes what it checks
severity: low
evidence: sklearn/cluster/tests/test_hdbscan.py:276-284 — the docstring says "correctly raises an error when passing precomputed data while requesting a tree-based algorithm", but the test parameterizes `algorithm=f"prims_{tree}tree"` (values `prims_kdtree`, `prims_balltree`) which are not valid values of the `algorithm` parameter (`StrOptions({"auto","brute","kdtree","balltree"})` at hdbscan.py:629-638), so the `ValueError` raised is from parameter validation — not from the "precomputed + tree" combination the docstring claims to exercise.
scenario: "A maintainer relying on this test to detect a regression in the precomputed-vs-tree guard (e.g. hdbscan.py:786-791) → the test still passes even if the guard is removed, because param validation raises first."
contract: Use the actually-supported names (`kdtree`, `balltree`) in the parametrization so the intended precomputed-vs-tree path is exercised, matching the docstring.
instances: single-instance

### F14 — Scale-invariance example plots identical data three times [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — "fig, axes = plt.subplots(3, 1, figsize=(10, 12)) / hdb = HDBSCAN() / for idx, scale in enumerate((1, 0.5, 3)): / hdb.fit(X) / plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={'scale': scale})"
scenario: "A user reads the surrounding narrative — 'HDBSCAN is scale-invariant. …One immediate advantage is that HDBSCAN is scale-invariant' — and expects the three subplots to demonstrate identical clusterings across three distinct scales → in fact all three subplots fit and plot the same un-scaled `X`, so the identical plots are a tautology (same input, same output) rather than a demonstration of scale invariance, while the axis title still displays scale=0.5, scale=3 as if the data had been rescaled."
contract: Change the loop body to `hdb.fit(X * scale)` and `plot(X * scale, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={'scale': scale})`, so the demonstration actually varies the input by `scale`, mirroring the DBSCAN block immediately above (plot_hdbscan.py:85-89).
instances: single-instance
