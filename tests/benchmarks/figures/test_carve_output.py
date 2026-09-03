"""Tests for the two CARVE-output figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from benchmarks.figures import figure_carve_output_klein, figure_carve_output_levine
from benchmarks.figures._carve_output import _config_id_at_k
from benchmarks.figures._case_study import CompositeInputs


class _StubCarve:
    """Minimal stand-in for a fitted CARVE, so figure tests need no fit.

    Columns follow the canonical estimator_results_ schema -- ari_stability
    and ari_generalizability plus their _se variants -- since carve_lines
    reads exactly those names. An earlier alias ("stability" without the
    "ari_" prefix) was tried elsewhere in this plan and was a latent
    KeyError.

    config_id is a join key, deliberately not 0..n-1, so a positional
    lookup would pick the wrong matrix and the test would catch it.
    """

    def __init__(self, ks, n):
        rng = np.random.default_rng(0)
        self.estimator_results_ = pd.DataFrame(
            {
                "config_id": [100 + i for i in range(len(ks))],
                "n_clusters": list(ks),
                "ari_stability": np.linspace(0.5, 0.9, len(ks)),
                "ari_generalizability": np.linspace(0.4, 0.85, len(ks)),
                "ari_stability_se": np.full(len(ks), 0.02),
                "ari_generalizability_se": np.full(len(ks), 0.03),
            }
        )
        base = rng.random((n, n))
        self.consensus_matrices_ = {
            100 + i: (base + base.T) / 2 for i in range(len(ks))
        }
        self.stability_gini_scores_ = {100 + i: rng.random(n) for i in range(len(ks))}
        self._ks = list(ks)

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return self._ks[len(self._ks) // 2]


class _ScrambledStubCarve:
    """A stub built to defeat any positional lookup, silently.

    estimator_results_ carries a pandas index that matches neither row
    position nor config_id, and its rows are not sorted by n_clusters or
    by config_id either. consensus_matrices_ and stability_gini_scores_
    are filled with distinct, recognizable constants keyed by config_id
    only. If carve_output_figure ever indexed those dicts by row position
    or by the DataFrame's index label instead of the config_id column
    value, it would silently draw a different (but still valid-looking)
    matrix -- no exception, just the wrong figure. Asserting the drawn
    array's value against the one true matching config_id's constant is
    what catches that: a positional bug changes the asserted value, it
    does not merely fail to raise.
    """

    def __init__(self):
        # Row order (by position) is k=5, k=3, k=4 -- not sorted by k.
        # config_id order (by position) is 102, 100, 101 -- not sorted by
        # config_id either, and not equal to position (0, 1, 2).
        # The pandas index label order is 30, 10, 20 -- distinct from both
        # position and config_id.
        self.estimator_results_ = pd.DataFrame(
            {
                "config_id": [102, 100, 101],
                "n_clusters": [5, 3, 4],
                "ari_stability": [0.7, 0.5, 0.6],
                "ari_generalizability": [0.65, 0.45, 0.55],
                "ari_stability_se": [0.02, 0.02, 0.02],
                "ari_generalizability_se": [0.03, 0.03, 0.03],
            },
            index=[30, 10, 20],
        )
        n = 12
        self.consensus_matrices_ = {
            100: np.full((n, n), 0.25),
            101: np.full((n, n), 0.55),
            102: np.full((n, n), 0.85),
        }
        self.stability_gini_scores_ = {
            100: np.full(n, 0.10),
            101: np.full(n, 0.50),
            102: np.full(n, 0.90),
        }

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        # The configuration at k=4 is config_id 101 -- neither position 0
        # (config_id 102) nor the "middle by position" guess (config_id
        # 100) that a naive positional lookup would produce.
        return 4


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    n = 60
    return CompositeInputs(
        X=rng.normal(size=(n, 4)),
        y=rng.choice(["a", "b"], size=n),
        Z=rng.normal(size=(n, 2)),
        carve=_StubCarve([3, 4, 5], n),
        carve_labels=rng.integers(0, 3, size=n),
        comparison_labels=rng.integers(0, 2, size=n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [4],
                "score": [0.6],
                "ari": [0.5],
            }
        ),
        best_df=pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [4],
                "score": [0.6],
                "ari": [0.5],
            }
        ),
    )


@pytest.fixture
def scrambled_inputs():
    rng = np.random.default_rng(1)
    n = 12
    return CompositeInputs(
        X=rng.normal(size=(n, 4)),
        y=rng.choice(["a", "b"], size=n),
        Z=rng.normal(size=(n, 2)),
        carve=_ScrambledStubCarve(),
        carve_labels=rng.integers(0, 3, size=n),
        comparison_labels=rng.integers(0, 2, size=n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [4],
                "score": [0.6],
                "ari": [0.5],
            }
        ),
        best_df=pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [4],
                "score": [0.6],
                "ari": [0.5],
            }
        ),
    )


class TestCarveOutputFigures:
    def test_klein_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_carve_output_klein(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "CARVE_output_klein.png").exists()
        plt.close(fig)

    def test_klein_save_false_writes_nothing(self, inputs, tmp_path):
        fig = figure_carve_output_klein(inputs, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_levine_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_carve_output_levine(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "CARVE_output_levine.png").exists()
        plt.close(fig)

    def test_levine_save_false_writes_nothing(self, inputs, tmp_path):
        fig = figure_carve_output_levine(inputs, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_draws_three_lettered_panels(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        letters = {t.get_text() for ax in fig.get_axes() for t in ax.texts}
        assert {"A", "B", "C"} <= letters
        plt.close(fig)

    def test_consensus_matrix_is_rendered_as_an_image(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        assert any(ax.images for ax in fig.get_axes())
        plt.close(fig)

    def test_consensus_matrix_is_looked_up_by_join_key_not_position(
        self, scrambled_inputs
    ):
        """config_id, row position, and the pandas index all disagree here.

        get_k() selects k=4, which lives at config_id=101 (position 1,
        index label 20). Its consensus matrix is filled with 0.55; the
        matrices at the other two config_ids are 0.25 and 0.85. A
        positional or index-label lookup would silently draw one of those
        instead, so asserting the exact value is what makes this test
        meaningful -- a stub whose index equals its position could not
        distinguish a correct lookup from a positional one.
        """
        fig = figure_carve_output_klein(scrambled_inputs, save=False)
        image_ax = next(ax for ax in fig.get_axes() if ax.images)
        drawn = np.asarray(image_ax.images[0].get_array())
        np.testing.assert_allclose(drawn, np.full((12, 12), 0.55))
        plt.close(fig)

    def test_gini_annotation_is_looked_up_by_join_key_not_position(
        self, scrambled_inputs
    ):
        """Same join-key guard, for the stability_gini_scores_ lookup.

        config_id=101's Gini scores are all 0.50, so the annotated median
        must read 0.500; the other two config_ids would give 0.100 or
        0.900 under a positional bug.
        """
        fig = figure_carve_output_klein(scrambled_inputs, save=False)
        xlabels = " ".join(ax.get_xlabel() for ax in fig.get_axes())
        assert "0.500" in xlabels
        plt.close(fig)

    def test_levine_uses_the_smaller_marker_size(self, inputs):
        # ax.collections is not scatter-only: carve_lines' fill_between SE
        # bands on panel A are collections too (FillBetweenPolyCollection),
        # and their get_sizes() is an empty array rather than a marker
        # size. Picking "the first axes with any collection" -- as the
        # brief's own version of this test did -- lands on panel A instead
        # of panel C's scatter and raises IndexError on get_sizes()[0].
        # Filtering to PathCollection (what ax.scatter returns) is what
        # actually finds the marker-size collection.
        def marker_size(fig):
            for ax in fig.get_axes():
                for collection in ax.collections:
                    if isinstance(collection, PathCollection):
                        return collection.get_sizes()[0]
            raise AssertionError("no scatter (PathCollection) found in figure")

        klein = figure_carve_output_klein(inputs, save=False)
        levine = figure_carve_output_levine(inputs, save=False)
        assert marker_size(levine) < marker_size(klein)
        plt.close(klein)
        plt.close(levine)


class _ResultsOnlyCarve:
    """Bare enough for _config_id_at_k: it only reads estimator_results_."""

    def __init__(self, estimator_results):
        self.estimator_results_ = estimator_results


class TestConfigIdAtK:
    """_config_id_at_k's two failure paths: no match, and an ambiguous one.

    Both raise rather than guess. The no-match path already existed but was
    untested; the ambiguous-match path is new. Neither Klein nor Levine can
    trigger the ambiguous path today -- both sweep n_clusters directly, so
    n_clusters is unique per row -- but nothing enforced that assumption
    before this guard, and a resolution-based sweep (Leiden/Louvain) can
    observe the same empirical cluster count at two different resolutions.
    Silently returning match.iloc[0] in that case would be exactly the
    wrong-config-without-an-exception failure this whole module exists to
    prevent, just one step further upstream of the config_id join-key tests
    above.
    """

    def test_raises_when_no_row_matches_k(self):
        results = pd.DataFrame(
            {"config_id": [100, 101], "n_clusters": [3, 5]}
        )
        carve = _ResultsOnlyCarve(results)
        with pytest.raises(ValueError, match="No configuration at k=4"):
            _config_id_at_k(carve, 4)

    def test_raises_when_more_than_one_row_matches_k(self):
        # Two config_ids both observed k=4 -- as a resolution-based sweep
        # could produce for two different resolutions. There is no correct
        # single answer here without the canonical sweep_rank selection, so
        # this must raise rather than pick match.iloc[0] arbitrarily.
        results = pd.DataFrame(
            {"config_id": [100, 101, 102], "n_clusters": [3, 4, 4]}
        )
        carve = _ResultsOnlyCarve(results)
        with pytest.raises(ValueError, match="2 configurations share k=4"):
            _config_id_at_k(carve, 4)
