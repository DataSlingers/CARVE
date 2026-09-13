"""Tests for the shared case-study composite and its two instantiations."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from benchmarks.figures import (
    _klein_results,
    _levine_results,
    figure_klein_results,
    figure_levine_results,
    prepare_composite,
)
from benchmarks.figures._case_study import (
    CompositeInputs,
    _estimator_spec_from_model_label,
    composite_color_maps,
)
from tests.benchmarks._helpers import StubCarve


def _two_method_results(ks):
    """estimator_results_ for two method_ids swept over the same ks.

    Mirrors a real case study that sweeps two estimators (Klein sweeps Ward
    agglomerative clustering and spectral clustering), the shape that
    exposed carve_lines' original interleaving bug: a single line drawn
    from every row, not just the selected configuration's own sweep.
    Columns follow the canonical estimator_results_ schema -- ari_stability
    and ari_generalizability plus their _se variants, plus method_id/
    method_label -- since carve_lines reads exactly those names. An earlier
    alias ("stability" without the "ari_" prefix) was tried elsewhere in
    this plan and was a latent KeyError.
    """
    ks = list(ks)
    return pd.DataFrame(
        {
            "n_clusters": ks + ks,
            "method_id": ["m0"] * len(ks) + ["m1"] * len(ks),
            "method_label": ["KMeans"] * len(ks)
            + ["AgglomerativeClustering, linkage=ward"] * len(ks),
            "ari_stability": list(np.linspace(0.5, 0.9, len(ks)))
            + list(np.linspace(0.3, 0.6, len(ks))),
            "ari_generalizability": list(np.linspace(0.4, 0.85, len(ks)))
            + list(np.linspace(0.35, 0.7, len(ks))),
            "ari_stability_se": [0.02] * (2 * len(ks)),
            "ari_generalizability_se": [0.03] * (2 * len(ks)),
        }
    )


def _stub(ks, labels=None):
    """A StubCarve resolving to "m0" (KMeans), varying k with not_two.

    Enough to let a test tell whether not_two was actually forwarded,
    without needing to fit real selection logic.
    """
    ks = list(ks)

    def select(measure, not_two):
        candidates = ks[1:] if not_two else ks
        return "m0", candidates[len(candidates) // 2]

    return StubCarve(_two_method_results(ks), select=select, labels=labels)


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
            "model": ["KMeans (n_init=10)"] * 6,
            "k": [3, 4, 5] * 2,
            "score": [0.4, 0.6, 0.5, 0.2, 0.3, 0.35],
            "ari": [0.5] * 6,
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["KMeans (n_init=10)"] * 2,
            "k": [4, 5],
            "score": [0.6, 0.35],
            "ari": [0.5, 0.5],
        }
    )
    return CompositeInputs(
        X=X,
        y=y.to_numpy(),
        Z=Z,
        carve=_stub([3, 4, 5]),
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

    def _panel_d(self, fig):
        return next(ax for ax in fig.get_axes() if ax.get_ylabel() == "ARI")

    def test_panel_d_lines_use_one_configurations_k_values_not_both(self, inputs):
        # inputs.carve (a StubCarve) carries two method_ids swept over the
        # same k values -- the shape that exposed carve_lines' original
        # interleaving bug. Panel D must show one estimator's own sweep per
        # measure, not both method_ids' rows concatenated.
        fig = figure_klein_results(inputs, save=False)
        panel_d = self._panel_d(fig)
        data_lines = [ln for ln in panel_d.lines if ln.get_marker() == "o"]
        assert len(data_lines) == 2  # one per measure
        n_ks = int(
            (inputs.carve.estimator_results_["method_id"] == "m0").sum()
        )
        for line in data_lines:
            assert len(line.get_xdata()) == n_ks
        plt.close(fig)

    def test_panel_d_selected_k_marker_moves_with_not_two(self, inputs):
        # composite_figure's own carve_lines call must forward
        # CompositeInputs.not_two, not silently default it to False.
        from dataclasses import replace

        default_fig = figure_klein_results(inputs, save=False)
        not_two_inputs = replace(inputs, not_two=True)
        not_two_fig = figure_klein_results(not_two_inputs, save=False)

        def _selected_ks(fig):
            panel_d = self._panel_d(fig)
            return sorted(
                float(ln.get_xdata()[0])
                for ln in panel_d.lines
                if ln.get_marker() != "o"
            )

        assert _selected_ks(default_fig) != _selected_ks(not_two_fig)
        plt.close(default_fig)
        plt.close(not_two_fig)


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


class TestEstimatorSpecFromModelLabel:
    def test_maps_the_label_the_sweep_renders(self):
        assert _estimator_spec_from_model_label("KMeans (n_init=10)").name == "kmeans"

    def test_rejects_a_bare_class_name(self):
        # cvi_sweep never renders one: every registered estimator has fixed
        # defaults, so its label always carries them. Accepting a prefix
        # here is what let ward and single linkage collapse into one spec.
        with pytest.raises(ValueError, match="Cannot map model label"):
            _estimator_spec_from_model_label("AgglomerativeClustering")

    def test_maps_a_class_name_with_fixed_parameters(self):
        assert (
            _estimator_spec_from_model_label(
                "AgglomerativeClustering (linkage=ward)"
            ).name
            == "agglomerative"
        )

    def test_maps_spectral(self):
        assert (
            _estimator_spec_from_model_label(
                "SpectralClustering (affinity=self_tuning)"
            ).name
            == "spectral"
        )

    def test_distinguishes_single_from_ward_linkage(self):
        # A class-name prefix match sent both linkages to the ward spec, so a
        # single-linkage CVI winner would have been refit with Ward in panel C.
        assert (
            _estimator_spec_from_model_label("AgglomerativeClustering (linkage=single)").name
            == "agglomerative_single"
        )

    def test_raises_on_an_unknown_label(self):
        with pytest.raises(ValueError, match="Cannot map model label"):
            _estimator_spec_from_model_label("SomeOtherEstimator")


class TestPrepareComposite:
    def test_raises_when_best_df_has_no_row_for_the_comparison_metric(self):
        rng = np.random.default_rng(0)
        n = 40
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _stub([2, 3], labels=rng.integers(0, 2, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["gap"],
                "model": ["KMeans (n_init=10)"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["gap"],
                "model": ["KMeans (n_init=10)"],
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

    def test_labels_come_back_aligned_to_the_reported_labels(self):
        """Cluster i must mean "the cluster matching reported label i".

        Two things have to line up, and the data here separates them. The
        stub's clustering is a perfect but *permuted* copy of the reported
        labels, so alignment has to happen at all; and the reported labels
        appear in the order "b" then "a", so their first-occurrence order is
        the reverse of their sorted order. Only sorted order is right --
        aligned_color_maps hands out the palette in sorted category order --
        so aligning against first occurrence would give every cluster the
        wrong label's colour while still looking "aligned".
        """
        n = 40
        y = np.array(["b"] * (n // 2) + ["a"] * (n // 2))
        permuted = np.array([0] * (n // 2) + [1] * (n // 2))
        rng = np.random.default_rng(3)
        X = rng.normal(size=(n, 5)) + permuted[:, None]
        carve = _stub([2, 3], labels=permuted)
        frame = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
                "k": [2],
                "score": [0.5],
                "ari": [0.5],
            }
        )

        inputs = prepare_composite(
            X, y, carve, curves_df=frame, best_df=frame, comparison_metric="silhouette"
        )

        categorical = pd.Categorical(y)
        assert np.array_equal(inputs.carve_labels, categorical.codes)
        # Unaligned, the labels would be the permutation the stub returned.
        assert not np.array_equal(inputs.carve_labels, permuted)

        # The invariant the scatter panels rest on, end to end: the colour
        # panel B gives a cluster is the colour panel A gives the reported
        # label that cluster was matched to.
        true_cmap, carve_cmap, _ = composite_color_maps(inputs)
        for code, category in enumerate(categorical.categories):
            assert carve_cmap[code] == true_cmap[category]

    def test_uses_the_supplied_embedding_instead_of_fitting_pca(self):
        rng = np.random.default_rng(1)
        n = 30
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _stub([2, 3], labels=rng.integers(0, 2, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
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
        carve = _stub([2, 3, 4], labels=rng.integers(0, 3, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
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

    def test_uses_the_winning_models_estimator_not_a_hardcoded_kmeans(
        self, monkeypatch
    ):
        """Klein's grid sweeps Ward agglomerative and spectral, never
        KMeans, so a comparison estimator hardcoded to KMeans is not even in
        the sweep best_df describes. best_df's model column here names
        AgglomerativeClustering (Ward) as silhouette's winner; build_estimator
        must be called with that estimator's spec, not a hardcoded
        EstimatorSpec(name="kmeans").
        """
        from benchmarks import _estimators

        calls = []
        real_build_estimator = _estimators.build_estimator

        def spy_build_estimator(spec, n_clusters, random_state):
            calls.append(spec)
            return real_build_estimator(spec, n_clusters, random_state)

        monkeypatch.setattr(_estimators, "build_estimator", spy_build_estimator)

        rng = np.random.default_rng(11)
        n = 40
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _stub([2, 3], labels=rng.integers(0, 2, size=n))
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["AgglomerativeClustering (linkage=ward)"],
                "k": [2],
                "score": [0.9],
                "ari": [0.9],
            }
        )
        curves = best.copy()

        prepare_composite(
            X, y, carve, curves_df=curves, best_df=best, comparison_metric="silhouette"
        )

        assert len(calls) == 1
        assert calls[0].name == "agglomerative"

    def test_forwards_measure_rule_and_not_two_to_get_labels_by_name(self):
        rng = np.random.default_rng(3)
        n = 20
        X = rng.normal(size=(n, 5))
        y = rng.choice(["a", "b"], size=n)
        carve = _stub([2, 3], labels=rng.integers(0, 2, size=n))
        curves = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
                "k": [3],
                "score": [0.5],
                "ari": [0.5],
            }
        )
        best = pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans (n_init=10)"],
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
