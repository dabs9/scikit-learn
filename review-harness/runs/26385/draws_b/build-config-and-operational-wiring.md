### F1 — Test file name misspelled ("reachibility" → "reachability") diverges from module under test
severity: low
evidence: sklearn/cluster/_hdbscan/tests/test_reachibility.py:1 — the file tests the `mutual_reachability_graph` API exposed by `sklearn/cluster/_hdbscan/_reachability.pyx` (correctly spelled), but the test file itself is registered on disk as `test_reachibility.py` (missing the second `a`). The DIFF_MANIFEST also confirms `A sklearn/cluster/_hdbscan/tests/test_reachibility.py`.
scenario: "Maintainer runs `git grep test_reachability` or `pytest sklearn/**/test_reachability.py` to locate tests for the `_reachability` module → nothing is returned because the test file is misspelled `test_reachibility.py`; the tests are not discoverable through the canonical module name and the wiring between production module and its test file is broken by spelling."
contract: Rename `sklearn/cluster/_hdbscan/tests/test_reachibility.py` to `sklearn/cluster/_hdbscan/tests/test_reachability.py` so the test file name matches the module it tests (`_reachability.pyx`).
instances: single-instance
