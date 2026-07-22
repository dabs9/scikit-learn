# Plan of record — PR scikit-learn/scikit-learn#26385

## PR title
ENH Add `HDBSCAN` as a new estimator in `sklearn.cluster`

## PR description (top comment)

#### Reference Issues/PRs
Towards https://github.com/scikit-learn/scikit-learn/issues/24686

#### What does this implement/fix? Explain your changes.
Each change has been separately reviewed (see https://github.com/scikit-learn/scikit-learn/issues/24686 for details).

Edit: due to git shenanigans, some changes needed to be made within this PR. The novel changes included w/ this PR are:
1. Replaced `cnp.*_t` typing with `*_t` from `_typedefs.pxd`
2. Replaced `*.shape[0]` pattern with `len(*)` for `ndarray` objects
3. Trimmed unused variables (thanks to Cython linting pre-commit)

#### Any other comments?
cc: @thomasjpfan @jjerphan @glemaitre 

---

# Linked issue: scikit-learn/scikit-learn#24686 — Path to HDBSCAN Inclusion

## Introduction
The HDBSCAN estimator implementation from [`scikit-learn-contrib/hdbscan`](https://github.com/scikit-learn-contrib/hdbscan) has been adopted, modified and refactored to conform to scikit-learn API and is now merged into the [`hdbscan`](https://github.com/scikit-learn/scikit-learn/tree/hdbscan) feature branch. There are still several changes to be made both before and after merging `hdbscan-->main`, and the goal of this issue is to serve as a tracker for the remaining changes, as well as to host meta discussion regarding the estimator as a whole as needed.

In particular, I would encourage discussion regarding:
1. What other tasks may be relevant/necessary for `HDBSCAN` overall.
2. What tasks should be promoted from follow-up work to mandatory work _before_ merging into `main`.

## To do for merger into `main`
Mandatory work before consideration for final merger

- [x] #24857
- [x] https://github.com/scikit-learn/scikit-learn/pull/24701
- [x] Clean `_hdbscan/_tree.pyx` 
    - [x] https://github.com/scikit-learn/scikit-learn/pull/25768
    - [x] https://github.com/scikit-learn/scikit-learn/pull/25826
    - [x] https://github.com/scikit-learn/scikit-learn/pull/25827
    - [x] `HDBSCAN` `_tree.pyx` overhaul
        - [x] https://github.com/scikit-learn/scikit-learn/pull/26011
        - [x] https://github.com/scikit-learn/scikit-learn/pull/26096
        - [x] https://github.com/scikit-learn/scikit-learn/pull/26101
- [x] https://github.com/scikit-learn/scikit-learn/pull/24698
- [x] https://github.com/scikit-learn/scikit-learn/pull/25538
- [x] #25134 

## Follow-up after merger into `main`
Discussion regarding follow-up tasks has been moved to #26801

CC: @thomasjpfan @jjerphan @glemaitre 
