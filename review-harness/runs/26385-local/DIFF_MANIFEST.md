# DIFF MANIFEST — 86541f2b3bc8..04c8b6954e3e
Total: 19 files changed, +3297/-10
Hunk diffs: one file per changed path in the hunks directory (path with '/' replaced by '_', suffix .diff).

## (root)
M     setup.py  +5/-0

## doc
M     doc/modules/classes.rst  +1/-0
M     doc/modules/clustering.rst  +118/-5
M     doc/whats_new/v1.3.rst  +13/-0

## examples
M     examples/cluster/plot_cluster_comparison.py  +9/-0
A     examples/cluster/plot_hdbscan.py  +239/-0

## sklearn
M     sklearn/cluster/__init__.py  +2/-0
A     sklearn/cluster/_hdbscan/__init__.py  +0/-0
A     sklearn/cluster/_hdbscan/_linkage.pyx  +270/-0
A     sklearn/cluster/_hdbscan/_reachability.pyx  +212/-0
A     sklearn/cluster/_hdbscan/_tree.pxd  +49/-0
A     sklearn/cluster/_hdbscan/_tree.pyx  +797/-0
A     sklearn/cluster/_hdbscan/hdbscan.py  +973/-0
A     sklearn/cluster/_hdbscan/tests/__init__.py  +0/-0
A     sklearn/cluster/_hdbscan/tests/test_reachibility.py  +64/-0
A     sklearn/cluster/_hierarchical_fast.pxd  +9/-0
M     sklearn/cluster/_hierarchical_fast.pyx  +0/-5
A     sklearn/cluster/tests/test_hdbscan.py  +533/-0
M     sklearn/utils/estimator_checks.py  +3/-0
