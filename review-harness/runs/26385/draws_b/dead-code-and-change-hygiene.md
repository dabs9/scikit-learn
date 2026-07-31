### F1 — Duplicate `births = np.full(...)` allocation; first assignment is immediately overwritten dead code
severity: low
evidence: sklearn/cluster/_hdbscan/_tree.pyx:252-254 — two identical statements `births = np.full(largest_child + 1, np.nan, dtype=np.float64)` appear back-to-back with only a blank line between them; nothing reads `births` between them
scenario: "any call to _compute_stability → the first np.full allocation runs, is never read, and is immediately shadowed by an identical allocation on line 254; wasted allocation + reader confusion"
contract: Delete the duplicated line at 252 (or 254), leaving a single allocation of `births`.
instances: single-instance

### F2 — `HDBSCAN.__init__` default `n_jobs=4` contradicts the documented default of `None` and looks like a debug leftover
severity: medium
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 — `n_jobs=4,` in the constructor signature; the docstring at sklearn/cluster/_hdbscan/hdbscan.py:486-490 states `n_jobs : int, default=None` with "``None`` means 1 unless in a :obj:`joblib.parallel_backend` context"
scenario: "user constructs `HDBSCAN()` per the documented API → n_jobs is silently 4 not None, so pairwise-distance computation spawns 4 worker threads/processes on every fit even inside a parallel_backend context expecting a single-worker default"
contract: Change the default to `n_jobs=None` to match the docstring and the sklearn convention.
instances: single-instance

### F3 — `_hdbscan_prims` docstring documents a `copy` parameter that does not exist in the function signature
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:313-318 — the Parameters section of `_hdbscan_prims` describes `copy : bool, default=False ...`, but the signature at lines 269-278 has no `copy` argument (only X, algo, min_samples, alpha, metric, leaf_size, n_jobs, **metric_params)
scenario: "reader / documentation build inspects `_hdbscan_prims` → sees a documented `copy` parameter that cannot be passed and has no effect; likely copy-paste residue from `_hdbscan_brute` which does accept `copy`"
contract: Remove the `copy` parameter block from the `_hdbscan_prims` docstring.
instances: single-instance

### F4 — `_hdbscan_brute` default `alpha=None` is dead-and-broken code (docstring says `default=1.0`, and the only path that would ever hit `None` would crash on `distance_matrix /= alpha`)
severity: low
evidence: sklearn/cluster/_hdbscan/hdbscan.py:161 — signature `alpha=None,`; docstring at lines 183-184 declares `alpha : float, default=1.0`; the body unconditionally executes `distance_matrix /= alpha` on line 241
scenario: "any direct caller of `_hdbscan_brute` relying on the documented default → `distance_matrix /= None` raises TypeError; sole in-tree caller (`HDBSCAN.fit`) always passes `self.alpha` (default 1.0) so the None default is never actually exercised — it is dead code that also contradicts documented behavior"
contract: Change the signature default to `alpha=1.0` to match the docstring and to prevent latent breakage of the internal API.
instances: single-instance

### F5 — plot_hdbscan.py "scale invariance" demo never scales `X`, so the loop is a superseded / non-functional copy of the DBSCAN demo above [out-of-theme]
severity: medium
evidence: examples/cluster/plot_hdbscan.py:106-110 — `for idx, scale in enumerate((1, 0.5, 3)): hdb.fit(X); plot(X, hdb.labels_, hdb.probabilities_, ax=axes[idx], parameters={"scale": scale})`; compare with the DBSCAN loop at lines 92-95 which correctly does `dbs.fit(X * scale)` / `plot(X * scale, ...)`
scenario: "rendered gallery example → three subplots titled scale=1, scale=0.5, scale=3 all show identical clusterings because `X` is never multiplied by `scale`; the narrative claim ('One immediate advantage is that HDBSCAN is scale-invariant') is not demonstrated by the code, misleading readers"
contract: Replace `hdb.fit(X)` / `plot(X, ...)` with `hdb.fit(X * scale)` / `plot(X * scale, ...)` to mirror the DBSCAN demo and actually exercise scale invariance.
instances: single-instance
