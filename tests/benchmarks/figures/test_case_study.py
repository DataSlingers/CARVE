"""Tests for the shared case-study composite and its two instantiations."""

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from benchmarks.figures import (
    figure_klein_results,
    figure_levine_results,
    prepare_composite,
)
from benchmarks.figures import (
    _klein_results,
    _levine_results,
)
from benchmarks.figures._case_study import CompositeInputs


class _StubCarve:
    """Minimal stand-in for a fitted CARVE, so figure tests need no fit.

    Columns follow the canonical estimator_results_ schema -- ari_stability
    and ari_generalizability plus their _se variants -- since carve_lines
    reads exactly those names. An earlier alias ("stability" without the
    "ari_" prefix) was tried elsewhere in this plan and was a latent
    KeyError.

    get_labels is only needed by prepare_composite's tests -- the figure
    tests build CompositeInputs directly and never call it, so it stays
    unset (None) unless a test supplies labels explicitly. Each call to
    get_labels records the keyword arguments it received on
    self.get_labels_calls, so a test can assert prepare_composite forwarded
    measure/rule/not_two under their own names rather than, say, transposed.
    get_k is not similarly instrumented: prepare_composite never calls it,
    and carve_lines (which does call it) uses its own hardcoded measures and
    rule, unrelated to what a caller passes to prepare_composite.
    """

    def __init__(self, ks, labels=None):
        self.estimator_results_ = pd.DataFrame(
            {
                "n_clusters": list(ks),
                "ari_stability": np.linspace(0.5, 0.9, len(ks)),
                "ari_generalizability": np.linspace(0.4, 0.85, len(ks)),
                "ari_stability_se": np.full(len(ks), 0.02),
                "ari_generalizability_se": np.full(len(ks), 0.03),
            }
        )
        self._ks = list(ks)
        self._labels = labels
        self.get_labels_calls = []

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return self._ks[len(self._ks) // 2]

    def get_labels(self, *, measure="stability", rule="1se", not_two=False):
        self.get_labels_calls.append(
            {"measure": measure, "rule": rule, "not_two": not_two}
        )
        if self._labels is None:
            raise AssertionError(
                "get_labels called on a _StubCarve built without labels"
            )
        return self._labels


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5))
    Z = rng.normal(size=(n, 2))
    y = pd.Series(rng.choice(["a", "b", "c"], size=n))
    carve_labels = rng.integers(0, 3, size=n)
    comparison_labels = rng.integers(0, 2, size=n)
    curves = pd.DataFrame(
        {
            "metric": ["silhouette"] * 3 + ["gap"] * 3,
            "model": ["KMeans"] * 6,
            "k": [3, 4, 5] * 2,
            "score": [0.4, 0.6, 0.5, 0.2, 0.3, 0.35],
            "ari": [0.5] * 6,
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["KMeans", "KMeans"],
            "k": [4, 5],
            "score": [0.6, 0.35],
            "ari": [0.5, 0.5],
        }
    )
    return CompositeInputs(
        X=X,
        y=y.to_numpy(),
        Z=Z,
        carve=_StubCarve([3, 4, 5]),
        carve_labels=carve_labels,
        comparison_labels=comparison_labels,
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=curves,
        best_df=best,
    )


class TestKleinFigure:
    def test_returns_a_figure_with_six_drawn_panels(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        lettered = {t.get_text() for ax in fig.get_axes() for t in ax.texts}
        assert {"A", "B", "C", "D", "E", "F"} <= lettered
        plt.close(fig)

    def test_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_klein_results(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "klein_results.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, inputs, tmp_path):
        fig = figure_klein_results(inputs, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_builds_exactly_two_panel_legends(self, inputs):
        """The original built legend D twice and discarded the first."""
        fig = figure_klein_results(inputs, save=False)
        legend_axes = [ax for ax in fig.get_axes() if ax.get_legend() is not None]
        assert len(legend_axes) == 2
        plt.close(fig)

    def test_bottom_panel_is_the_alluvial(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        # The alluvial turns its axes off and carries three column titles.
        alluvial_axes = [ax for ax in fig.get_axes() if not ax.axison and ax.patches]
        assert alluvial_axes
        plt.close(fig)


class TestLevineFigure:
    def test_saves_under_the_manuscript_filename(self, inputs, tmp_path):
        fig = figure_levine_results(inputs, save=True, out_dir=tmp_path)
        assert (tmp_path / "levine_results.png").exists()
        plt.close(fig)

    def test_save_false_writes_nothing(self, inputs, tmp_path):
        fig = figure_levine_results(inputs, save=False, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_bottom_panel_is_the_ari_lollipop(self, inputs):
        fig = figure_levine_results(inputs, save=False)
        labels = [ax.get_xlabel() for ax in fig.get_axes()]
        assert any("ARI" in label for label in labels)
        plt.close(fig)

    def test_uses_the_smaller_marker_size(self, inputs):
        klein = figure_klein_results(inputs, save=False)
        levine = figure_levine_results(inputs, save=False)
        klein_sizes = klein.get_axes()[0].collections[0].get_sizes()
        levine_sizes = levine.get_axes()[0].collections[0].get_sizes()
        assert levine_sizes[0] < klein_sizes[0]
        plt.close(klein)
        plt.close(levine)


class TestSharedComposite:
    def test_both_figures_come_from_one_builder(self):
        for module in (_klein_results, _levine_results):
            assert "composite_figure" in inspect.getsource(module)

    def test_neither_module_lays_out_its_own_gridspec(self):
        for module in (_klein_results, _levine_results):
            assert "add_gridspec" not in inspect.getsource(module)


class TestPrepareComposite:
    def test_raises_when_best_df_has_no_row_for_the_comparison_metric(self):
        rng = np.random.default_rng(0)
        n = 40
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _StubCarve([2, 3], labels=rng.integers(0, 2, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["gap"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["gap"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        with pytest.raises(ValueError, match="No row for 'silhouette'"):
            prepare_composite(
                X,
                y,
                carve,
                curves_df=curves,
                best_df=best,
                comparison_metric="silhouette",
            )

    def test_uses_the_supplied_embedding_instead_of_fitting_pca(self):
        rng = np.random.default_rng(1)
        n = 30
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _StubCarve([2, 3], labels=rng.integers(0, 2, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        # Distinctive values a PCA of X would essentially never reproduce, so
        # equality here can only hold if the override was actually used.
        embedding = np.arange(n * 2, dtype=float).reshape(n, 2)

        result = prepare_composite(
            X, y, carve, curves_df=curves, best_df=best, embedding=embedding
        )

        np.testing.assert_array_equal(result.Z, embedding)

    def test_fits_pca_and_the_comparison_estimator_when_no_embedding_is_supplied(
        self,
    ):
        rng = np.random.default_rng(2)
        n = 50
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b", "c"], size=n)
        carve = _StubCarve([2, 3, 4], labels=rng.integers(0, 3, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )

        result = prepare_composite(
            X, y, carve, curves_df=curves, best_df=best, random_state=7
        )

        expected_Z = PCA(n_components=2, random_state=7).fit_transform(X)
        np.testing.assert_allclose(result.Z, expected_Z)
        assert result.comparison_k == 3
        assert result.comparison_labels.shape == (n,)
        assert len(set(result.comparison_labels.tolist())) == 3

    def test_forwards_measure_rule_and_not_two_to_get_labels_by_name(self):
        rng = np.random.default_rng(3)
        n = 20
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _StubCarve([2, 3], labels=rng.integers(0, 2, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        # Distinguishable from each other and from the defaults ("stability",
        # "1se", False), so a transposed or dropped forward is detectable --
        # measure and rule cannot be confused for one another here.
        prepare_composite(
            X,
            y,
            carve,
            curves_df=curves,
            best_df=best,
            measure="generalizability",
            rule="quantile",
            not_two=True,
        )

        assert carve.get_labels_calls == [
            {"measure": "generalizability", "rule": "quantile", "not_two": True}
        ]
