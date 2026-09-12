"""Anchored consensus measured against the exact answer.

At n small enough for the exact path to run, both are computable, so the
approximation is measured rather than asserted. The numbers this produces
back the supplementary table described in the design document.
"""

import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from carve import CARVE

N = 5000
GRIDS = [(KMeans, {"n_clusters": [2, 3, 4], "n_init": [10]})]

# The design document names m in {500, 1000, 2000, 5000}. At N = 5000 the
# fourth value is the exact path by definition, since resolve_anchors returns
# None once m reaches n; that identity is already covered by
# tests/test_consensus.py::TestConsensusAnchorBlock. The three values below
# are the ones that actually exercise the approximation.


def _data(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [8, 0], [4, 7]], dtype=float)
    y = rng.integers(0, 3, N)
    X = centers[y] + rng.normal(0, 1.2, (N, 2))
    return X, y


def _fit(X, *, anchors):
    # The anchored fits raise anchor_threshold above N so the anchored path
    # runs at all at this n. The exact fit does not need that and does not
    # get it: it runs under the default threshold, which is exactly N, so it
    # is a genuine boundary fit and covers the configuration the published
    # Levine case study sits on. resolve_anchors returns None for either
    # threshold at this n, so the numbers below are unaffected.
    kwargs = (
        {}
        if anchors is None
        else {"anchor_threshold": 10_000, "consensus_anchors": anchors}
    )
    carve = CARVE(
        estimator_param_grids=GRIDS, n_resamples=25, random_state=0, **kwargs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        carve.fit(X)
    return carve


@pytest.fixture(scope="module")
def exact():
    X, y = _data()
    return X, y, _fit(X, anchors=None)


def test_exact_fixture_sits_on_the_default_threshold_boundary(exact):
    # N is exactly the default anchor_threshold, so the exact fixture is a
    # real fit at the boundary rather than one held open by a raised
    # threshold. Everything the table below reports is measured against it.
    _, _, exact_carve = exact
    assert exact_carve.anchor_threshold == N
    assert exact_carve.consensus_anchors_ is None
    assert exact_carve.consensus_matrices_[0].shape == (N, N)


@pytest.mark.parametrize("m", [500, 1000, 2000])
def test_anchored_tracks_exact(exact, m):
    X, y, exact_carve = exact
    anchored = _fit(X, anchors=m)

    print(f"m={m}: anchor count = {anchored.consensus_anchors_.size}")

    assert anchored.consensus_anchors_.size == m

    # Selected k agrees.
    anchored_k = anchored.get_k(measure="stability", rule="1se")
    exact_k = exact_carve.get_k(measure="stability", rule="1se")
    print(f"m={m}: selected k (anchored) = {anchored_k}, selected k (exact) = {exact_k}")
    assert anchored_k == exact_k

    # Per-sample stability scores correlate with the exact ones.
    corr = np.corrcoef(
        anchored.stability_gini_scores_[0], exact_carve.stability_gini_scores_[0]
    )[0, 1]
    print(f"m={m}: gini correlation = {corr:.3f}")
    assert corr > 0.8, f"gini correlation {corr:.3f} at m={m}"

    # Same estimator family and 1/m variance argument as gini, so the same
    # bound applies for the same reason.
    ce_corr = np.corrcoef(
        anchored.stability_ce_scores_[0], exact_carve.stability_ce_scores_[0]
    )[0, 1]
    print(f"m={m}: ce correlation = {ce_corr:.3f}")
    assert ce_corr > 0.8, f"ce correlation {ce_corr:.3f} at m={m}"

    # PAC, reported without an acceptance threshold: under anchoring PAC is
    # computed over the m-by-m anchor block rather than over all pairs, so
    # it is a legitimately different quantity from the exact PAC, and
    # neither the plan nor the spec establishes how close the two should
    # be.
    exact_pac = exact_carve.estimator_results_.loc[
        exact_carve.estimator_results_["config_id"] == 0, "consensus_pac_stability"
    ].item()
    anchored_pac = anchored.estimator_results_.loc[
        anchored.estimator_results_["config_id"] == 0, "consensus_pac_stability"
    ].item()
    print(
        f"m={m}: PAC (exact) = {exact_pac:.3f}, PAC (anchored) = {anchored_pac:.3f}, "
        f"abs diff = {abs(exact_pac - anchored_pac):.3f}"
    )

    # Labels agree with each other and both recover the planted structure.
    both = adjusted_rand_score(
        exact_carve.get_labels(k=3), anchored.get_labels(k=3)
    )
    print(f"m={m}: label ARI (anchored vs exact) = {both:.3f}")
    assert both > 0.9, f"label ARI {both:.3f} at m={m}"
    planted = adjusted_rand_score(y, anchored.get_labels(k=3))
    print(f"m={m}: label ARI (anchored vs planted truth) = {planted:.3f}")
    assert planted > 0.9


def test_accuracy_improves_with_more_anchors(exact):
    X, _, exact_carve = exact
    reference = exact_carve.stability_gini_scores_[0]

    correlations = []
    for m in (500, 2000):
        anchored = _fit(X, anchors=m)
        corr = np.corrcoef(anchored.stability_gini_scores_[0], reference)[0, 1]
        print(f"m={m}: gini correlation = {corr:.3f}")
        correlations.append(corr)

    # Variance of the row-mean estimator falls as 1/m, so more anchors must
    # track the exact answer better. A flat result would mean the anchor
    # count is not actually being honored.
    assert correlations[1] > correlations[0]
