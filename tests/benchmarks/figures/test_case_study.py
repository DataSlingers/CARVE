"""Tests for the shared case-study composite and its two instantiations."""

import dataclasses
import inspect

import matplotlib.colors as mcolors
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
from benchmarks._estimators import param_grids
from benchmarks._types import EstimatorSpec
from benchmarks.figures._case_study import (
    CompositeInputs,
    _align_to_reference,
    _estimator_spec_from_model_label,
    carve_cluster_numbers,
    carve_labels_aligned,
    carve_output_cmap,
    composite_color_maps,
)
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import adjusted_rand_score

from benchmarks._theme import PRINT_FONT_SIZES, PRINT_MAX_HEIGHT_IN, PRINT_WIDTH_IN
from carve import CARVE
from tests.benchmarks._helpers import StubCarve, resolution_results

RESOLUTIONS = [0.01, 0.043, 0.41, 2.3]


LEIDEN_LABEL = "LeidenClustering, n_neighbors=30, objective_function=modularity, weighting=jaccard"
LEIDEN_MODEL = (
    "LeidenClustering (n_neighbors=30, objective_function=modularity, weighting=jaccard)"
)


@pytest.fixture
def resolution_inputs():
    """Composite inputs for a resolution run, the shape the Levine study has:
    one Leiden setting, labeled as CARVE and the CVI sweep label it."""
    rng = np.random.default_rng(1)
    n = 120
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    curves = pd.DataFrame(
        {
            "metric": np.repeat(metrics, 4),
            "model": [LEIDEN_MODEL] * 12,
            "resolution": RESOLUTIONS * 3,
            "k": [4, 6, 8, 20] * 3,
            "score": [0.30, 0.37, 0.28, 0.13, 0.40, 0.45, 0.20, 0.10, 9.0, 7.0, 5.0, 3.0],
            "ari": [0.60, 0.70, 0.97, 0.55] * 3,
        }
    )
    best = curves.iloc[[1, 5, 8]].reset_index(drop=True)
    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=rng.choice(["a", "b", "c"], size=n),
        Z=rng.normal(size=(n, 2)),
        carve=StubCarve(
            resolution_results(RESOLUTIONS, LEIDEN_LABEL),
            select=lambda measure, not_two: ("m0", 0.41),
            sweep_param="resolution",
        ),
        carve_labels=rng.integers(0, 8, size=n),
        comparison_labels=rng.integers(0, 6, size=n),
        comparison_name="Silhouette",
        comparison_k=6,
        curves_df=curves,
        best_df=best,
        comparison_resolution=0.043,
    )


class _Groups(ClusterMixin, BaseEstimator):
    """Below resolution 0.1 two groups; from 0.1 n_groups, shuffled by random_state."""

    def __init__(self, resolution=1.0, n_groups=3, random_state=None):
        self.resolution = resolution
        self.n_groups = n_groups
        self.random_state = random_state

    def fit_predict(self, X, y=None):
        n = len(X)
        if self.resolution < 0.1:
            return np.arange(n) % 2
        return np.random.default_rng(self.random_state).permutation(np.arange(n) % self.n_groups)


def _two_method_results(ks):
    """estimator_results_ for two method_ids swept over the same ks.

    As in a real case study that sweeps two estimators (Klein: Ward and
    spectral). Columns follow the canonical estimator_results_ schema
    (ari_stability and ari_generalizability, their _se variants, method_id
    and method_label), which carve_lines reads.
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


def _lettered(fig):
    """Map each panel letter to the axes panel_letter() drew it on."""
    return {
        text.get_text(): ax
        for ax in fig.get_axes()
        for text in ax.texts
        if text.get_text() in list("ABCDEF")
    }


def _row(ax):
    return ax.get_subplotspec().rowspan.start


class TestKleinFigure:
    def test_curves_come_first_and_the_scatters_below_them(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        panels = _lettered(fig)
        assert panels["A"].get_title() == "CARVE ARI over k"
        assert panels["B"].get_title() == "CVIs over k"
        assert panels["C"].get_title() == "Reported Labels"
        assert panels["D"].get_title() == "CARVE clustering"
        assert panels["E"].get_title().startswith("CVI (")
        assert _row(panels["A"]) == _row(panels["B"])
        assert _row(panels["C"]) == _row(panels["D"]) == _row(panels["E"])
        assert _row(panels["A"]) < _row(panels["C"]) < _row(panels["F"])
        plt.close(fig)

    def test_the_curve_legends_sit_between_the_curves_and_the_scatters(
        self, inputs
    ):
        fig = figure_klein_results(inputs, save=False)
        panels = _lettered(fig)
        legend_rows = {
            _row(ax)
            for ax in fig.get_axes()
            if ax.get_legend() is not None and not ax.axison
        }
        assert legend_rows == {_row(panels["A"]) + 1}
        assert _row(panels["A"]) + 1 < _row(panels["C"])
        plt.close(fig)

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

    def test_saves_the_plos_figure_file_as_fig5(self, inputs, tmp_path):
        from PIL import Image

        fig = figure_klein_results(inputs, save=True, out_dir=tmp_path)
        plt.close(fig)
        with Image.open(tmp_path / "Fig5.tif") as image:
            assert image.mode == "RGB"
            assert image.info["compression"] == "tiff_lzw"

    def test_is_laid_out_at_the_print_sizes(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        assert fig.get_figwidth() == PRINT_WIDTH_IN
        assert fig.get_figheight() <= PRINT_MAX_HEIGHT_IN
        for ax in fig.get_axes():
            assert ax.title.get_fontsize() <= PRINT_FONT_SIZES["title"]
            for text in ax.texts:
                assert text.get_fontsize() <= PRINT_FONT_SIZES["panel_letter"]
        plt.close(fig)

    def test_legend_strip_names_the_estimators_briefly(self, inputs):
        fig = figure_klein_results(inputs, save=False)
        texts = [
            text.get_text()
            for ax in fig.get_axes()
            if ax.get_legend() is not None
            for text in ax.get_legend().get_texts()
        ]
        assert texts
        assert not any("Clustering" in text for text in texts)
        plt.close(fig)

    def test_builds_exactly_two_panel_legends(self, inputs):
        """Legend D is built once."""
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
        # same k values; panel D shows one estimator's sweep per measure.
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
        # composite_figure's carve_lines call forwards
        # CompositeInputs.not_two.
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

    def test_an_si_figure_writes_no_plos_file(self, inputs, tmp_path):
        fig = figure_levine_results(inputs, save=True, out_dir=tmp_path)
        plt.close(fig)
        assert not list(tmp_path.glob("*.tif"))

    def test_bottom_panel_is_the_ari_lollipop(self, inputs):
        fig = figure_levine_results(inputs, save=False)
        labels = [ax.get_xlabel() for ax in fig.get_axes()]
        assert any("ARI" in label for label in labels)
        plt.close(fig)

    def test_keeps_the_scatters_first(self, inputs):
        fig = figure_levine_results(inputs, save=False)
        panels = _lettered(fig)
        assert panels["A"].get_title() == "Reported Labels"
        assert panels["D"].get_title() == "CARVE ARI over k"
        assert _row(panels["A"]) < _row(panels["D"])
        plt.close(fig)

    def test_a_resolution_run_titles_its_curves_over_resolution(self, resolution_inputs):
        fig = figure_levine_results(resolution_inputs, save=False)
        panels = _lettered(fig)
        assert panels["D"].get_title() == "CARVE ARI over resolution"
        assert panels["E"].get_title() == "CVIs over resolution"
        plt.close(fig)

    def test_a_resolution_run_draws_both_curve_panels_on_a_log_axis(
        self, resolution_inputs
    ):
        # The grid is log-spaced from 0.01 to 2.3; on a linear axis its low
        # end, where the CVIs select, would crowd into the first tick.
        fig = figure_levine_results(resolution_inputs, save=False)
        panels = _lettered(fig)
        assert panels["D"].get_xscale() == "log"
        assert panels["E"].get_xscale() == "log"
        plt.close(fig)

    def test_panel_c_names_the_comparison_resolution_and_its_k(self, resolution_inputs):
        fig = figure_levine_results(resolution_inputs, save=False)
        title = _lettered(fig)["C"].get_title()
        assert "0.043" in title
        assert "k=6" in title
        plt.close(fig)

    def test_legends_name_the_leiden_setting_briefly(self, resolution_inputs):
        # CARVE's full configuration string ran both legends into each other.
        fig = figure_levine_results(resolution_inputs, save=False)
        texts = [
            text.get_text()
            for ax in fig.get_axes()
            if ax.get_legend() is not None
            for text in ax.get_legend().get_texts()
        ]
        assert "Silhouette — Leiden, 30 neighbors, Jaccard" in texts
        assert not any("objective_function" in text for text in texts)
        plt.close(fig)

    def test_cvi_legend_stays_clear_of_the_resolution_axis_label(
        self, resolution_inputs
    ):
        # The resolution ticks stand upright and take most of the legend
        # strip.
        fig = figure_levine_results(resolution_inputs, save=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        panels = _lettered(fig)
        legends = [ax.get_legend() for ax in fig.get_axes() if ax.get_legend()]
        for letter in ("D", "E"):
            label = panels[letter].xaxis.label.get_window_extent(renderer)
            for legend in legends:
                assert not label.overlaps(legend.get_window_extent(renderer))
        plt.close(fig)

    def test_curve_legends_start_right_under_the_resolution_labels(
        self, resolution_inputs
    ):
        fig = figure_levine_results(resolution_inputs, save=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        panels = _lettered(fig)
        legends = [
            ax.get_legend().get_window_extent(renderer)
            for ax in fig.get_axes()
            if ax.get_legend() is not None and not ax.axison
        ]
        f_top = panels["F"].get_tightbbox(renderer).y1
        for letter in ("D", "E"):
            curves = panels[letter].get_tightbbox(renderer)
            (legend,) = [
                box for box in legends if curves.x0 < (box.x0 + box.x1) / 2 < curves.x1
            ]
            assert 0 <= (curves.y0 - legend.y1) / fig.dpi < 0.2
        # The strip ends a short way below the taller legend, clear of F.
        lowest = min(box.y0 for box in legends)
        assert 0 < (lowest - f_top) / fig.dpi < 0.45
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
        # defaults, so its label always carries them.
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
        assert (
            _estimator_spec_from_model_label("AgglomerativeClustering (linkage=single)").name
            == "agglomerative_single"
        )

    def test_distinguishes_a_hundred_restarts_from_ten(self):
        assert (
            _estimator_spec_from_model_label("KMeans (n_init=100)").name
            == "kmeans_n_init_100"
        )

    def test_raises_on_an_unknown_label(self):
        with pytest.raises(ValueError, match="Cannot map model label"):
            _estimator_spec_from_model_label("SomeOtherEstimator")


class TestPrepareCompositeOverResolution:
    """The CVI winner of a resolution sweep is refit from the grid it came from:
    a Leiden on a non-default graph has no registered spec to rebuild it from."""

    GRIDS = [(_Groups, {"resolution": [0.05, 0.2], "n_groups": [4]})]

    def _best(self, model="_Groups (n_groups=4)"):
        return pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": [model],
                "resolution": [0.2],
                "k": [4],
                "score": [0.5],
                "ari": [0.5],
            }
        )

    def _carve(self, n):
        return StubCarve(
            resolution_results([0.05, 0.2], "_Groups"),
            select=lambda measure, not_two: ("m0", 0.2),
            labels=np.arange(n) % 2,
            sweep_param="resolution",
        )

    def test_refits_the_winner_at_its_resolution_with_the_grid_s_parameters(self):
        n = 40
        X = np.random.default_rng(0).normal(size=(n, 3))
        y = np.repeat(["a", "b"], n // 2)
        result = prepare_composite(
            X, y, self._carve(n), curves_df=self._best(), best_df=self._best(),
            model_grids=self.GRIDS, embedding=X[:, :2], random_state=5,
        )
        expected = np.random.default_rng(5).permutation(np.arange(n) % 4)
        assert adjusted_rand_score(expected, result.comparison_labels) == 1.0
        assert result.comparison_k == 4
        assert result.comparison_resolution == 0.2

    def test_a_resolution_winner_needs_the_swept_grids(self):
        n = 40
        X = np.zeros((n, 3))
        with pytest.raises(ValueError, match="model_grids"):
            prepare_composite(
                X, np.repeat(["a", "b"], n // 2), self._carve(n),
                curves_df=self._best(), best_df=self._best(), embedding=X[:, :2],
            )

    def test_refuses_a_winner_that_no_swept_grid_renders(self):
        n = 40
        X = np.zeros((n, 3))
        with pytest.raises(ValueError, match="_Groups \\(n_groups=7\\)"):
            prepare_composite(
                X, np.repeat(["a", "b"], n // 2), self._carve(n),
                curves_df=self._best(), best_df=self._best("_Groups (n_groups=7)"),
                model_grids=self.GRIDS, embedding=X[:, :2],
            )


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

        The stub's clustering is a permuted copy of the reported labels, and
        the reported labels appear as "b" then "a", so first-occurrence order
        is the reverse of sorted order. Alignment follows sorted order, the
        order aligned_color_maps hands out the palette in.
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

        # The color panel B gives a cluster is the color panel A gives the
        # reported label that cluster was matched to.
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
        KMeans. best_df names AgglomerativeClustering (Ward) as silhouette's
        winner, so build_estimator is called with that estimator's spec.
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
        # Values distinct from each other and from the defaults
        # ("stability", "1se", False).
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


class TestAriTableExtraRows:
    def test_the_carve_row_follows_a_generalizability_selection(self, inputs):
        from benchmarks.figures._case_study import ari_table

        table = ari_table(dataclasses.replace(inputs, measure="generalizability"))
        assert table.loc[table["method"] == "CARVE", "metric"].item() == "ari_generalizability_1se"

    def test_the_carve_row_keeps_stability_green_by_default(self, inputs):
        from benchmarks.figures._case_study import ari_table

        table = ari_table(inputs)
        assert table.loc[table["method"] == "CARVE", "metric"].item() == "ari_stability_1se"

    def test_the_carve_row_names_the_quantile_rule_as_the_theme_does(self, inputs):
        from benchmarks._theme import METRIC_COLORS
        from benchmarks.figures._case_study import ari_table

        table = ari_table(dataclasses.replace(inputs, measure="generalizability", rule="quantile"))
        metric = table.loc[table["method"] == "CARVE", "metric"].item()
        assert metric == "ari_generalizability_quant"
        assert metric in METRIC_COLORS

    def test_default_is_unchanged(self, inputs):
        from benchmarks.figures._case_study import ari_table

        assert "M3C" not in ari_table(inputs)["method"].tolist()

    def test_extra_rows_are_appended_after_the_cvi_rows(self, inputs):
        from benchmarks.figures._case_study import ari_table

        extra = [
            {"method": "M3C", "metric": "m3c_rcsi", "ari": 0.41, "k": 2},
            {"method": "M3C (k=4)", "metric": "m3c_rcsi", "ari": 0.55, "k": 4},
        ]
        table = ari_table(inputs, extra_rows=extra)
        assert table["method"].tolist()[-2:] == ["M3C", "M3C (k=4)"]
        assert table["ari"].tolist()[-2:] == [0.41, 0.55]

    def test_extra_rows_keep_the_frame_s_columns(self, inputs):
        from benchmarks.figures._case_study import ari_table

        base = ari_table(inputs)
        table = ari_table(
            inputs,
            extra_rows=[
                {
                    "method": "M3C",
                    "metric": "m3c_rcsi",
                    "ari": 0.41,
                    "k": 2,
                    "stray": "x",
                }
            ],
        )
        assert list(table.columns) == list(base.columns)
        assert "stray" not in table.columns

    def test_rejects_a_row_missing_a_column(self, inputs):
        from benchmarks.figures._case_study import ari_table

        with pytest.raises(ValueError, match="ari"):
            ari_table(inputs, extra_rows=[{"method": "M3C", "k": 2}])


def _numbering_inputs(*, X, y, carve, carve_labels):
    """CompositeInputs for the numbering tests, which read no curves."""
    return CompositeInputs(
        X=np.asarray(X),
        y=np.asarray(y),
        Z=np.zeros((len(y), 2)),
        carve=carve,
        carve_labels=np.asarray(carve_labels),
        comparison_labels=np.asarray(carve_labels),
        comparison_name="Silhouette",
        comparison_k=int(len(np.unique(carve_labels))),
        curves_df=pd.DataFrame(),
        best_df=pd.DataFrame(),
    )


@pytest.fixture
def coarser_fit():
    """A small real fit whose selected k differs from the reported label count.

    Three blobs carry four reported labels, so inside carve_labels_aligned
    CARVE keeps its own ids, as it does on Levine (8 clusters against 14
    populations), while the aligned ids follow the reported codes.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=6.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=(0.0, 6.0, 0.0), scale=0.4, size=(40, 3)),
    ])
    y = np.array(["d"] * 40 + ["c"] * 40 + ["b"] * 35 + ["a"] * 5)
    carve = CARVE(
        estimator_param_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
        n_resamples=3,
        random_state=0,
    )
    carve.fit(X, reference_labels=y)
    aligned = _align_to_reference(carve.get_labels(measure="stability", rule="1se"), y)
    return _numbering_inputs(X=X, y=y, carve=carve, carve_labels=aligned)


def _own_labels(inputs):
    with carve_labels_aligned(inputs.carve, inputs.y):
        return np.asarray(inputs.carve.get_labels(measure="stability", rule="1se"))


class TestCarveClusterNumbers:
    def test_each_cluster_gets_the_number_carve_s_figure_prints(self, coarser_fit):
        own = _own_labels(coarser_fit)
        aligned = coarser_fit.carve_labels
        # Only meaningful while the two labelings differ.
        assert set(np.unique(aligned).tolist()) != set(np.unique(own).tolist())
        numbers = carve_cluster_numbers(coarser_fit)
        assert sorted(numbers) == sorted(np.unique(aligned).tolist())
        for aligned_id, number in numbers.items():
            assert set((own[aligned == aligned_id] + 1).tolist()) == {number}

    def test_a_different_partition_raises(self):
        results = pd.DataFrame(
            {"method_id": ["m0"], "method_label": ["KMeans"], "n_clusters": [2]}
        )
        carve = StubCarve(
            results,
            select=lambda measure, not_two: ("m0", 2),
            labels=np.array([0, 1, 1, 1]),
        )
        inputs = _numbering_inputs(
            X=np.zeros((4, 2)),
            y=np.array(["a", "a", "b", "b"]),
            carve=carve,
            carve_labels=np.array([0, 0, 1, 1]),
        )
        with pytest.raises(ValueError, match="same partition"):
            carve_cluster_numbers(inputs)


class TestCarveOutputCmap:
    def test_entry_j_is_the_composite_color_of_carve_s_own_cluster_j(
        self, coarser_fit
    ):
        own = _own_labels(coarser_fit)
        _, carve_cmap, _ = composite_color_maps(coarser_fit)
        cmap = carve_output_cmap(coarser_fit)
        own_ids = np.unique(own)
        assert cmap.N == own_ids.size
        for j, own_id in enumerate(own_ids):
            aligned_id = int(coarser_fit.carve_labels[own == own_id][0])
            assert mcolors.to_hex(cmap(j)) == mcolors.to_hex(carve_cmap[aligned_id])
