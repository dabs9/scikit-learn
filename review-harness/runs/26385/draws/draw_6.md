I have enough evidence. Producing the findings:

### F1 — `n_jobs=4` default contradicts documented `default=None`
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in `__init__`, while line 486-490 docstring says `n_jobs : int, default=None` and describes `None means 1 unless in a joblib.parallel_backend context`
scenario: "User relies on the documented default (None → 1 worker unless in joblib context) → HDBSCAN silently spawns 4 processes for pairwise_distances, causing unexpected CPU/memory pressure or nested-parallelism conflicts with outer joblib backends"
contract: change the default to `n_jobs=None` in `__init__` to match the docstring and the sklearn convention shared by every other n_jobs-bearing estimator
instances: single-instance

### F2 — `_weighted_cluster_center` miscounts clusters when missing-data label `-3` is present
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:895 — `n_clusters = len(set(self.labels_) - {-1, -2})`; the outlier-encoding at hdbscan.py:80 defines the missing label as `-3`, and hdbscan.py:849 assigns `-3` to samples with NaN rows
scenario: "User calls `HDBSCAN(store_centers='centroid').fit(X)` on data with any NaN rows → `-3` remains in the label set, `n_clusters` is inflated by 1, the final iteration `idx = n_clusters-1` selects an empty mask, and `np.average(data, weights=strength, axis=0)` raises `ZeroDivisionError` / returns nan-filled row (medoid path calls `pairwise_distances` on an empty array)"
contract: exclude every outlier label — use `n_clusters = len(set(self.labels_) - {-1, -2, -3})` (or `- set(v['label'] for v in _OUTLIER_ENCODING.values()) - {-1}`); mirror this same set everywhere non-outlier labels are enumerated
instances: single-instance

### F3 — `test_hdbscan_precomputed_non_brute` tests parameter-validation error, not the intended dispatch check
severity: medium
evidence: sklearn/cluster/tests/test_hdbscan.py:282 — `hdb = HDBSCAN(metric="precomputed", algorithm=f"prims_{tree}tree")` where the accepted algorithm values (hdbscan.py:636-644) are only `{"auto","brute","kdtree","balltree"}`; `prims_kdtree` / `prims_balltree` fail `_validate_params()` before any precomputed-vs-tree dispatch runs
scenario: "The test always raises `InvalidParameterError` from `_validate_params`, so it passes even if the real precomputed→tree-algo guard is removed → regression of the intended behaviour is undetectable via this test"
contract: use the actually-valid algorithm strings — `HDBSCAN(metric="precomputed", algorithm="kdtree")` / `"balltree"` — and match against the specific message emitted by the true dispatch check
instances: single-instance

### F4 — Scale-invariance example ignores the scale factor
severity: medium
evidence: examples/cluster/plot_hdbscan.py:107-110 — `hdb = HDBSCAN(); for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, ..., parameters={"scale": scale})` — the loop refits on `X` (never `X * scale`) and plots `X` again, so all three subplots are identical
scenario: "User reads the 'Scale Invariance' section of the user-facing example → sees three identical panels labeled with different `scale` values; the demo silently fails to illustrate the claim that HDBSCAN is scale-invariant"
contract: fit and plot on the scaled data, i.e. `hdb.fit(X * scale); plot(X * scale, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`, matching the DBSCAN reference block at plot_hdbscan.py:93-95
instances: single-instance

### F5 — `fit` inline comment misstates outlier label mapping
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:831-833 — `# Remap indices to align with original data in the case of / # non-finite entries. Samples with np.inf are mapped to -1 and / # those with np.nan are mapped to -2.` — the actual mapping (hdbscan.py:72-84, and the class docstring at hdbscan.py:539-542) is `np.inf → -2`, `np.nan → -3`
scenario: "A future maintainer trusts the comment and 'fixes' the code to align with it, or copies -1/-2 constants elsewhere → introduces regression that reclassifies infinite samples as ordinary noise and missing samples as infinite"
contract: update the comment to the true encoding — inf-samples are mapped to `-2` and nan-samples to `-3`
instances: single-instance

### F6 — Sphinx cross-reference target `<HDBSCAN>` does not match the label `_hdbscan`
severity: low
evidence: examples/cluster/plot_hdbscan.py:110 — `see :ref:\`User Guide <HDBSCAN>\``; the label defined in doc/modules/clustering.rst:51 is `.. _hdbscan:` (lowercase). Sphinx `:ref:` targets are case-sensitive.
scenario: "Doc build resolves the ref against no matching label → Sphinx emits a warning and (depending on `nitpicky`) the rendered example page has a broken cross-reference to the User Guide"
contract: use `:ref:\`User Guide <hdbscan>\`` to match the declared label
instances: single-instance

### F7 — `remap_single_linkage_tree` docstring mislabels the `non_finite` parameter
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:364-365 — `non_finite : ndarray / Boolean array of which entries in the raw data are non-finite`; the caller at hdbscan.py:838 actually passes `non_finite=set(infinite_index + missing_index)` (a Python set of integer indices), and the function body at hdbscan.py:388 uses `for i, outlier in enumerate(non_finite)` with `outlier` used as an index value (hdbscan.py:389)
scenario: "A caller reads the docstring and prepares a boolean mask as documented → gets nonsensical outlier node indices in the appended `outlier_tree` (or crashes on `enumerate(bool_array)` producing 0/1 index values)"
contract: rewrite the parameter doc to describe what is really consumed — an iterable of raw-index integers identifying the non-finite samples
instances: single-instance

### F8 — User-guide references undefined parameter name `minimum_cluster_size`
severity: low
evidence: doc/modules/clustering.rst:138-141 — "components with fewer than `minimum_cluster_size` many samples are considered noise. In practice, one can set `minimum_cluster_size = min_samples` to couple the parameters"; the actual estimator parameter is `min_cluster_size` (hdbscan.py:649, 435)
scenario: "User copies the recommendation verbatim → `HDBSCAN(minimum_cluster_size=min_samples)` raises `TypeError: unexpected keyword argument` (or the parameter is silently ignored, depending on estimator plumbing)"
contract: rename both occurrences to the true parameter, `min_cluster_size`
instances: [doc/modules/clustering.rst:139, doc/modules/clustering.rst:140]

### F9 — Broken RST markup in the HDBSCAN user guide (stray trailing colons, grammar)
severity: low
evidence: doc/modules/clustering.rst:89-91 — `by removing any edges with value greater than :math:\`\varepsilon\`: / from the original graph. Any points whose core distance is less than :math:\`\varepsilon\`: / are at this staged marked as noise.` — the trailing colons after each `:math:` role are stray text (they render literally), and "at this staged marked" is ungrammatical
scenario: "Sphinx builds the HDBSCAN user-guide section as written → rendered docs show a literal `:` after each ε and an ungrammatical sentence, degrading the introductory explanation for every reader of the new estimator's user guide"
contract: drop the stray colons and replace `at this staged marked as noise` with `at this stage marked as noise`
instances: [doc/modules/clustering.rst:89, doc/modules/clustering.rst:90, doc/modules/clustering.rst:91]

### F10 — `_hdbscan_brute` default `alpha=None` is a latent divide error
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — `alpha=None,` while the docstring at hdbscan.py:183 states `alpha : float, default=1.0`, and hdbscan.py:241 unconditionally executes `distance_matrix /= alpha`
scenario: "A future caller invokes `_hdbscan_brute` relying on the documented default → `TypeError: unsupported operand type(s) for /=` on the in-place division; today it happens to be safe only because `HDBSCAN.fit` always passes `alpha=self.alpha`"
contract: use the same default as `_hdbscan_prims` — `alpha=1.0` — so the signature matches its docstring and the function is safely callable in isolation
instances: single-instance

### F11 — `internal_to_raw` built as a dict for what is a plain index lookup
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:732 — `internal_to_raw = {x: y for x, y in enumerate(finite_index)}` where `finite_index` is already an ndarray of positional integers; the callee at hdbscan.py:375/379 only ever does `internal_to_raw[left]` / `[right]` where `left`/`right` are `< finite_count`
scenario: "`HDBSCAN.fit` runs on data with any non-finite rows → the dict comprehension allocates one Python-int mapping per finite sample (O(n) hashmap entries) purely to reproduce ndarray indexing, dominating the outlier-handling path for large `n_samples` with a small non-finite fraction"
contract: pass the `finite_index` ndarray directly and use ndarray indexing in `remap_single_linkage_tree` (`internal_to_raw[left]` becomes exactly the same expression); update the docstring accordingly
instances: single-instance
