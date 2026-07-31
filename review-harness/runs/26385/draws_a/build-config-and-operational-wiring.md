Based on my analysis, I've walked the changed files through the build-config-and-operational-wiring lens. Setup.py properly registers all 3 Cython extensions under `cluster._hdbscan`, `find_packages` discovers the subpackage via its `__init__.py`, `package_data` includes `*.pxd`, the class is registered in `cluster/__init__.py`, `doc/modules/classes.rst`, whats-new, and the estimator-checks override. Most in-theme wiring is clean.

The single concrete finding I verified is out-of-theme but a clear user-visible defect.

### F1 — HDBSCAN default `n_jobs=4` contradicts its documented default of `None` [out-of-theme]
severity: high
evidence: sklearn/cluster/_hdbscan/hdbscan.py:658 has `n_jobs=4` in `HDBSCAN.__init__`, while the class docstring at sklearn/cluster/_hdbscan/hdbscan.py:492-496 documents `n_jobs : int, default=None` with the standard sklearn semantics (`None` means 1 unless in a `joblib.parallel_backend` context, `-1` means all processors).
scenario: "User constructs `HDBSCAN()` on a machine with 1 or 2 CPUs → `pairwise_distances` is dispatched with `n_jobs=4`, oversubscribing joblib and ignoring `joblib.parallel_backend`; the constructor is also inconsistent with every other sklearn cluster estimator and produces a `repr` that surprises users who trust the documented default."
contract: Set the `HDBSCAN.__init__` default for `n_jobs` to `None` to match the documented semantics and the rest of scikit-learn.
instances: single-instance
