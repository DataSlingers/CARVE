"""Anchored consensus measured against the exact answer.

At n small enough for the exact path to run, both are computable, so the
approximation is measured rather than asserted. The assertions are the
contract; the correlations are not printed.
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
        # warnings.filterwarnings matches from the start of the message, and
        # the message here leads with the reason, not "anchored consensus"
        # (which only appears mid-sentence) -- anchor the pattern on the
        # actual leading text this test always triggers.
        warnings.filterwarnings(
            "ignore",
            message=r"consensus_anchors=\d+ opts this run in regardless of anchor_threshold",
            category=RuntimeWarning,
        )
        carve.fit(X)
    return carve


ANCHOR_COUNTS = (500, 1000, 2000)


@pytest.fixture(scope="module")
def exact():
    X, y = _data()
    return X, y, _fit(X, anchors=None)


@pytest.fixture(scope="module")
def anchored_fits(exact):
    """One anchored fit per anchor count, on the exact fixture's data."""
    X, _, _ = exact
    return {m: _fit(X, anchors=m) for m in ANCHOR_COUNTS}


def test_exact_fixture_sits_on_the_default_threshold_boundary(exact):
    # N is exactly the default anchor_threshold, so the exact fixture is a
    # real fit at the boundary rather than one held open by a raised
    # threshold. Everything below is measured against it.
    _, _, exact_carve = exact
    assert exact_carve.anchor_threshold == N
    assert exact_carve.consensus_anchors_ is None
    assert exact_carve.consensus_matrices_[0].shape == (N, N)


@pytest.mark.parametrize("m", ANCHOR_COUNTS)
def test_anchored_tracks_exact(exact, anchored_fits, m):
    X, y, exact_carve = exact
    anchored = anchored_fits[m]

    assert anchored.consensus_anchors_.size == m

    # Selected k agrees.
    assert anchored.get_k(measure="stability", rule="1se") == exact_carve.get_k(
        measure="stability", rule="1se"
    )

    # Per-sample stability scores correlate with the exact ones. Same
    # estimator family and 1/m variance argument for gini and ce, so the
    # same bound applies to both.
    for attr in ("stability_gini_scores_", "stability_ce_scores_"):
        corr = np.corrcoef(getattr(anchored, attr)[0], getattr(exact_carve, attr)[0])[
            0, 1
        ]
        assert corr > 0.8, f"{attr} correlation {corr:.3f} at m={m}"

    # PAC is not compared: under anchoring it is computed over the m-by-m
    # anchor block rather than over all pairs, so it is a different
    # quantity from the exact PAC.

    # Labels agree with each other and both recover the planted structure.
    both = adjusted_rand_score(exact_carve.get_labels(k=3), anchored.get_labels(k=3))
    assert both > 0.9, f"label ARI {both:.3f} at m={m}"
    assert adjusted_rand_score(y, anchored.get_labels(k=3)) > 0.9


def test_accuracy_improves_with_more_anchors(exact, anchored_fits):
    _, _, exact_carve = exact
    reference = exact_carve.stability_gini_scores_[0]
    correlations = [
        np.corrcoef(anchored_fits[m].stability_gini_scores_[0], reference)[0, 1]
        for m in (500, 2000)
    ]
    # Variance of the row-mean estimator falls as 1/m, so more anchors must
    # track the exact answer better. A flat result would mean the anchor
    # count is not actually being honored.
    assert correlations[1] > correlations[0]
