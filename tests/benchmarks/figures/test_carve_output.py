"""Tests for the two CARVE-output figures (Fig 3 / S4 Fig).

The composed module (``_carve_output.py``) never resolves config_id itself
-- it only calls the five plotting methods CARVE already ships on a fitted
object and lets them resolve their own selection. That is exactly why these
tests fit a tiny, real CARVE instead of hand-building a stub: a stub would
have to reimplement _select_row/get_labels/get_k correctly to be trustworthy,
which is the same duplication this restructure exists to remove. A real fit
is fast enough here (a few seconds, module-scoped so it runs once) and
guarantees every plotting call below exercises CARVE's actual selection and
labeling code, not a guess at it.

The fixture's data-generating seed (2, not the more common 0) is chosen
deliberately: at this seed, stability's 1-SE rule selects k=3 while
generalizability's 1-SE rule selects k=4. That divergence is what lets the
tests below tell panel A (stability) apart from panel B (generalizability)
by more than a coincidentally-matching selected k, and is the most direct
guard against the restructure's most likely regression: swapping the two
measures between those panels.
"""

import dataclasses

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from benchmarks._estimators import resolution_grids
from benchmarks._theme import ESTIMATOR_COLORS, cluster_colors
from benchmarks.figures._heca_carve import MARKER_LINEWIDTH, MARKER_SIZE
from benchmarks._types import EstimatorSpec
from benchmarks.figures import (
    figure_carve_output_heca,
    figure_carve_output_klein,
    figure_carve_output_levine,
)
from benchmarks.figures._case_study import (
    CompositeInputs,
    _align_to_reference,
    carve_labels_aligned,
    composite_color_maps,
    reference_codes,
)
from carve import CARVE


def _panel_by_letter(fig):
    """Map each panel letter to its axes via the text panel_letter() drew.

    fig.get_axes() also returns the consensus matrix's divider-appended band
    and colorbar axes, interleaved in creation order, so a fixed positional
    index into that list does not reliably pick out "panel B". The letter
    text panel_letter() writes directly onto the axes we composed is a
    stable handle regardless of how many extra axes a given panel's method
    appended.
    """
    panels = {}
    for ax in fig.get_axes():
        for text in ax.texts:
            letter = text.get_text()
            if letter in "ABCDEF":
                panels[letter] = ax
    return panels


def _line_matching(ax, y_values, x_values):
    """Whether some Line2D on ax carries exactly this (x, y) series."""
    y_values = np.asarray(y_values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        x = np.asarray(line.get_xdata(), dtype=float)
        if y.shape == y_values.shape and np.allclose(y, y_values):
            if x.shape == x_values.shape and np.allclose(x, x_values):
                return True
    return False


@pytest.fixture(scope="module")
def fitted_carve():
    """A tiny, real, already-fitted CARVE (module-scoped so it fits once).

    Sweeps two estimators (KMeans and Ward agglomerative clustering), the
    shape a real case study uses -- Klein sweeps Ward agglomerative
    clustering and spectral clustering. At this fixture's seed, KMeans
    (method_id "m0") remains the winning configuration for both stability
    and generalizability with Agglomerative added, so the k=3/k=4
    divergence the rest of this module's tests and docstring rely on is
    unchanged; verified against carve._select_row directly while writing
    this fixture. estimator_results_ having two method_ids is what
    exercises CARVE's own plot_metric_over_n_clusters grouping by
    method_id, rather than a fixture shape where "one line" and "every
    row" happen to be the same thing.
    """
    rng = np.random.RandomState(2)
    n_per = 15
    X = np.vstack(
        [
            rng.randn(n_per, 4) + [4, 0, 0, 0],
            rng.randn(n_per, 4) + [0, 4, 0, 0],
            rng.randn(n_per, 4) + [0, 0, 4, 0],
        ]
    )
    carve = CARVE(
        n_clusters=np.array([2, 3, 4]),
        n_resamples=4,
        subsample_ratio=0.8,
        estimator_param_grids=[
            (KMeans, {"n_clusters": [2, 3, 4]}),
            (AgglomerativeClustering, {"n_clusters": [2, 3, 4], "linkage": ["ward"]}),
        ],
        normalization_options=[],
        dim_reduction_options=[],
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    carve.fit(X)
    return carve, X


@pytest.fixture
def inputs(fitted_carve):
    carve, X = fitted_carve
    rng = np.random.RandomState(3)
    n = X.shape[0]
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    carve_labels = np.asarray(carve.get_labels(measure="stability", rule="1se"))
    y = np.array(["a"] * 15 + ["b"] * 15 + ["c"] * 15)
    return CompositeInputs(
        X=X,
        y=y,
        Z=Z,
        carve=carve,
        carve_labels=carve_labels,
        comparison_labels=rng.randint(0, 2, size=n),
        comparison_name="Silhouette",
        comparison_k=3,
        curves_df=pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
                "score": [0.6],
                "ari": [0.5],
            }
        ),
        best_df=pd.DataFrame(
            {
                "metric": ["silhouette"],
                "model": ["KMeans"],
                "k": [3],
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

    def test_draws_six_lettered_panels(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        assert set(_panel_by_letter(fig)) == set("ABCDEF")
        plt.close(fig)

    def test_panels_read_left_to_right_then_top_to_bottom_in_three_rows(
        self, inputs
    ):
        fig = figure_carve_output_klein(inputs, save=False)
        panels = _panel_by_letter(fig)
        positions = {}
        for letter, ax in panels.items():
            # A colorbar (panels C and F) moves its parent into a sub-gridspec
            # of the parent's own cell, so the figure-level cell is the
            # topmost spec.
            spec = ax.get_subplotspec().get_topmost_subplotspec()
            assert spec.get_gridspec().get_geometry() == (3, 2)
            positions[letter] = (spec.rowspan.start, spec.colspan.start)
        assert positions == {
            "A": (0, 0),
            "B": (0, 1),
            "C": (1, 0),
            "D": (1, 1),
            "E": (2, 0),
            "F": (2, 1),
        }
        plt.close(fig)

    @pytest.mark.parametrize("letter", ["A", "B"])
    def test_estimator_lines_are_tab10_blue_and_red(self, inputs, letter):
        fig = figure_carve_output_klein(inputs, save=False)
        panel = _panel_by_letter(fig)[letter]
        curve_lines = [ln for ln in panel.get_lines() if ln.get_marker() == "o"]
        drawn = {mcolors.to_hex(line.get_color()) for line in curve_lines}
        assert drawn == {color.lower() for color in ESTIMATOR_COLORS}
        plt.close(fig)

    @pytest.mark.parametrize(
        ("measure", "letter", "other"),
        [("stability", "A", "B"), ("generalizability", "B", "A")],
    )
    def test_the_estimator_legend_sits_on_the_curve_of_the_measure_shown(
        self, inputs, measure, letter, other
    ):
        # Its dashed-line entry names the selection panels C to F show; at
        # this seed the two measures select different k.
        inputs = dataclasses.replace(
            inputs,
            measure=measure,
            carve_labels=np.asarray(inputs.carve.get_labels(measure=measure, rule="1se")),
        )
        selected_k = int(inputs.carve.get_k(measure=measure, rule="1se"))
        fig = figure_carve_output_klein(inputs, save=False)
        panels = _panel_by_letter(fig)
        assert panels[other].get_legend() is None
        legend = panels[letter].get_legend()
        assert legend is not None
        texts = [text.get_text() for text in legend.get_texts()]
        assert f"Selected k (1-SE rule): {selected_k}" in texts
        fig.canvas.draw()
        box = legend.get_window_extent().transformed(
            panels[letter].transAxes.inverted()
        )
        assert box.x0 < 0.1 and box.y0 < 0.1
        assert box.x1 < 1.0 and box.y1 < 1.0
        plt.close(fig)

    def test_panel_a_is_stability_and_panel_b_is_generalizability(self, inputs):
        """The restructure's most likely regression: swapping A and B.

        The fixture is built so stability and generalizability genuinely
        differ in value at every k (not just in which k each one selects),
        so a test that only checked panel titles or the selected-k line
        could still pass with the two measures swapped. Matching the full
        drawn curve against each measure's actual column values cannot.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        panels = _panel_by_letter(fig)
        # The fixture now sweeps two estimators (method_id "m0"/KMeans and
        # "m1"/Agglomerative); restrict to "m0", the one both measures
        # select at this seed, so the expected curve is one estimator's own
        # three-point sweep and not the two concatenated.
        all_results = inputs.carve.estimator_results_
        results = all_results.loc[all_results["method_id"] == "m0"].sort_values(
            "n_clusters"
        )
        x = results["n_clusters"].to_numpy(dtype=float)
        stability_y = results["ari_stability"].to_numpy(dtype=float)
        generalizability_y = results["ari_generalizability"].to_numpy(dtype=float)

        # Sanity check on the fixture itself: if these coincided, the
        # assertions below could pass by accident.
        assert not np.allclose(stability_y, generalizability_y)

        assert _line_matching(panels["A"], stability_y, x)
        assert not _line_matching(panels["A"], generalizability_y, x)
        assert _line_matching(panels["B"], generalizability_y, x)
        assert not _line_matching(panels["B"], stability_y, x)
        plt.close(fig)

    def test_panel_a_draws_a_separate_line_per_estimator_not_one_combined_line(
        self, inputs
    ):
        """The fixture sweeps two estimators (KMeans and Agglomerative
        ward), the shape a real case study uses. CARVE's own
        plot_metric_over_n_clusters groups by method_id already; this pins
        that panel A shows two three-point lines, one per estimator, rather
        than one six-point line formed by concatenating both estimators'
        rows -- the interleaving defect fixed elsewhere in this plan for
        the benchmarks package's own carve_lines primitive.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        panel_a = _panel_by_letter(fig)["A"]
        curve_lines = [ln for ln in panel_a.get_lines() if ln.get_marker() == "o"]
        assert len(curve_lines) == 2
        for line in curve_lines:
            x = np.sort(line.get_xdata())
            assert len(x) == 3
            assert np.all(np.diff(x) > 0)
        plt.close(fig)

    def test_panel_c_consensus_matrix_is_stabilitys_selected_configuration(
        self, inputs
    ):
        """Panel C draws stability's consensus matrix, not generalizability's.

        Checking only the title text was not enough here: this module's
        title is built from a separate carve.get_k(measure="stability", ...)
        call rather than from whatever plot_consensus_matrix actually drew,
        so a title-only check kept passing even when plot_consensus_matrix
        itself was mutated to measure="generalizability" -- caught while
        mutation-testing this test, not by inspection. Comparing the drawn
        pixel array against a matrix independently rendered with
        measure="stability" (and confirming it disagrees with
        measure="generalizability", which is a different matrix at this
        fixture's seed) verifies the data actually plotted, not just a label
        that happens to be computed the same way today.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        ax_c = _panel_by_letter(fig)["C"]
        assert len(ax_c.images) == 1
        drawn = np.asarray(ax_c.images[0].get_array())

        carve = inputs.carve
        stability_k = int(carve.get_k(measure="stability", rule="1se"))
        generalizability_k = int(carve.get_k(measure="generalizability", rule="1se"))
        assert stability_k != generalizability_k
        assert ax_c.get_title() == f"Consensus matrix ($k={stability_k}$)"

        # The references are drawn under the same label alignment the
        # figure draws under (see carve_labels_aligned): the matrix is
        # ordered by cluster, so an unaligned reference differs from the
        # drawn panel by a permutation of samples and nothing here would be
        # comparable.
        with carve_labels_aligned(carve, inputs.y):
            reference_fig, reference_ax = plt.subplots()
            carve.plot_consensus_matrix(
                measure="stability", rule="1se", ax=reference_ax
            )
            stability_matrix = np.asarray(reference_ax.images[0].get_array())
            plt.close(reference_fig)

            other_fig, other_ax = plt.subplots()
            carve.plot_consensus_matrix(
                measure="generalizability", rule="1se", ax=other_ax
            )
            generalizability_matrix = np.asarray(other_ax.images[0].get_array())
            plt.close(other_fig)

        assert not np.array_equal(stability_matrix, generalizability_matrix)
        np.testing.assert_array_equal(drawn, stability_matrix)
        assert not np.array_equal(drawn, generalizability_matrix)
        plt.close(fig)

    def test_panel_d_violin_has_one_body_per_cluster(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        ax_d = _panel_by_letter(fig)["D"]
        bodies = [c for c in ax_d.collections if isinstance(c, PolyCollection)]
        n_clusters = len(np.unique(inputs.carve_labels))
        assert len(bodies) == n_clusters
        plt.close(fig)

    def test_panel_d_reports_gini_stability_not_a_different_source(self, inputs):
        """plot_cluster_violin's ``source`` selects gini/ce/accuracy scores,
        all of which produce a violin body per cluster -- the previous test
        cannot tell them apart, so a ``source="accuracy"`` regression would
        slip through it silently. The caption is specifically about
        stability, so this checks the ylabel plot_cluster_violin derives
        from ``source`` and the actual violin body shapes against a
        reference gini render (and against an accuracy render, which is a
        different shape at this fixture's seed).
        """
        fig = figure_carve_output_klein(inputs, save=False)
        ax_d = _panel_by_letter(fig)["D"]
        assert ax_d.get_ylabel() == "Cluster Stability (Gini)"

        def body_extents(ax):
            return [
                (
                    round(float(path.vertices[:, 1].min()), 6),
                    round(float(path.vertices[:, 1].max()), 6),
                )
                for c in ax.collections
                if isinstance(c, PolyCollection)
                for path in c.get_paths()
            ]

        carve = inputs.carve
        # Drawn under the figure's own label alignment, so the violins come
        # out in the same cluster order the panel uses.
        with carve_labels_aligned(carve, inputs.y):
            gini_fig, gini_ax = plt.subplots()
            carve.plot_cluster_violin(
                source="gini", measure="stability", rule="1se", ax=gini_ax
            )
            gini_extents = body_extents(gini_ax)
            plt.close(gini_fig)

            accuracy_fig, accuracy_ax = plt.subplots()
            carve.plot_cluster_violin(
                source="accuracy", measure="stability", rule="1se", ax=accuracy_ax
            )
            accuracy_extents = body_extents(accuracy_ax)
            plt.close(accuracy_fig)

        assert gini_extents != accuracy_extents
        assert body_extents(ax_d) == gini_extents
        plt.close(fig)

    def test_panels_e_and_f_use_different_encodings(self, inputs):
        """E encodes score via size (one collection); F via shape (one per
        cluster, fixed size) -- the caption's actual distinction between
        them. A test only checking "both have a scatter" cannot tell them
        apart; this checks the encoding each caption describes.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        panels = _panel_by_letter(fig)
        n_clusters = len(np.unique(inputs.carve_labels))

        e_collections = [
            c for c in panels["E"].collections if isinstance(c, PathCollection)
        ]
        assert len(e_collections) == 1
        e_sizes = e_collections[0].get_sizes()
        assert len(set(np.round(e_sizes, 3))) > 1  # score -> size, so it varies

        f_collections = [
            c for c in panels["F"].collections if isinstance(c, PathCollection)
        ]
        assert len(f_collections) == n_clusters  # one scatter call per cluster
        f_sizes = np.concatenate([c.get_sizes() for c in f_collections])
        assert len(set(np.round(f_sizes, 3))) == 1  # fixed marker_size, not score
        plt.close(fig)

    def test_panel_e_reports_gini_stability_not_a_different_source(self, inputs):
        """Mirrors the panel D source guard for plot_cluster_scatter.

        ``source`` selects which per-sample score panel E encodes as marker
        size; a ``source="accuracy"`` regression would still draw one scatter
        with varying sizes (the previous test's only check). E's sizes must
        match a gini rendering and differ from an accuracy one.
        """
        fig = figure_carve_output_levine(inputs, save=False)
        drawn = np.sort(_panel_by_letter(fig)["E"].collections[0].get_sizes())
        plt.close(fig)

        def sizes(source):
            scratch, ax = plt.subplots()
            with carve_labels_aligned(inputs.carve, inputs.y):
                inputs.carve.plot_cluster_scatter(
                    source=source,
                    measure="stability",
                    rule="1se",
                    X=inputs.X,
                    embedding=inputs.Z,
                    ax=ax,
                    size_range=(6.0, 24.0),  # Levine's marker_size of 8
                    legend=False,
                    show=False,
                )
            drawn_sizes = np.sort(ax.collections[0].get_sizes())
            plt.close(scratch)
            return drawn_sizes

        assert not np.allclose(sizes("accuracy"), sizes("gini"))
        np.testing.assert_allclose(drawn, sizes("gini"))

    def test_klein_panels_e_and_f_carry_no_legend(self, inputs):
        fig = figure_carve_output_klein(inputs, save=False)
        panels = _panel_by_letter(fig)
        assert panels["E"].get_legend() is None
        assert panels["F"].get_legend() is None
        plt.close(fig)

    @pytest.mark.parametrize(
        ("draw", "ratio"),
        [(figure_carve_output_klein, 1.2), (figure_carve_output_levine, 1.3)],
    )
    def test_scatter_row_is_taller_and_rows_a_to_d_keep_their_size(
        self, inputs, draw, ratio
    ):
        fig = draw(inputs, save=False)
        gs = (
            _panel_by_letter(fig)["E"]
            .get_subplotspec()
            .get_topmost_subplotspec()
            .get_gridspec()
        )
        assert tuple(gs.get_height_ratios()) == (1.0, 1.0, ratio)
        # The extra height goes to row E-F alone; rows A-B and C-D keep 14/3
        # inches each.
        assert fig.get_size_inches()[1] == pytest.approx(14.0 / 3 * (2 + ratio))
        plt.close(fig)

    def test_levine_panels_e_and_f_carry_no_legend(self, inputs):
        fig = figure_carve_output_levine(inputs, save=False)
        panels = _panel_by_letter(fig)
        assert panels["E"].get_legend() is None
        assert panels["F"].get_legend() is None
        plt.close(fig)

    @pytest.mark.parametrize(
        ("draw", "width"),
        [(figure_carve_output_klein, 0.2), (figure_carve_output_levine, 0.05)],
    )
    def test_diagnostic_outlines_thin_for_levine_only(self, inputs, draw, width):
        # Over 5,000 cells, 0.2 pt black outlines hid the unstable cells' color.
        fig = draw(inputs, save=False)
        collections = _panel_by_letter(fig)["F"].collections
        widths = np.concatenate([points.get_linewidths() for points in collections])
        np.testing.assert_allclose(widths, width)
        plt.close(fig)

    def test_panel_f_reports_gini_stability_not_a_different_source(self, inputs):
        """Mirrors the source guard above for plot_diagnostic_scatter.

        Its colorbar (a separate axes, not one of the six lettered panels)
        carries the score-source name as its label.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        panels = _panel_by_letter(fig)
        xlabels = {
            ax.get_xlabel() for ax in fig.get_axes() if ax not in panels.values()
        }
        assert "Gini Stability" in xlabels
        plt.close(fig)

    def test_palette_colors_come_from_the_theme_not_accent(self, inputs):
        """The point of the palette work: assert it, not just assume it.

        The expected colors are cluster_colors(n) -- the palette's first n
        entries -- which is exactly what cluster_cmap(n) hands CARVE's own
        plotting methods here, so a violin's color, the consensus band's,
        and the composite figure's are one list rather than three.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        ax_d = _panel_by_letter(fig)["D"]
        bodies = [c for c in ax_d.collections if isinstance(c, PolyCollection)]
        n_clusters = len(bodies)

        drawn = {tuple(np.round(b.get_facecolor()[0][:3], 3)) for b in bodies}
        theme_colors = {
            tuple(np.round(mcolors.to_rgba(c)[:3], 3))
            for c in cluster_colors(n_clusters)
        }
        accent_colors = {
            tuple(np.round(plt.get_cmap("Accent", n_clusters)(i)[:3], 3))
            for i in range(n_clusters)
        }
        assert drawn <= theme_colors
        assert not (drawn & accent_colors)
        plt.close(fig)

    def test_klein_and_levine_axis_labels_match_their_embeddings(self, inputs):
        klein_fig = figure_carve_output_klein(inputs, save=False)
        levine_fig = figure_carve_output_levine(inputs, save=False)
        klein_e = _panel_by_letter(klein_fig)["E"]
        levine_e = _panel_by_letter(levine_fig)["E"]

        assert (klein_e.get_xlabel(), klein_e.get_ylabel()) == ("PC1", "PC2")
        assert (levine_e.get_xlabel(), levine_e.get_ylabel()) == (
            "t-SNE 1",
            "t-SNE 2",
        )
        plt.close(klein_fig)
        plt.close(levine_fig)

    def test_levine_uses_smaller_diagnostic_markers_than_klein(self, inputs):
        klein_fig = figure_carve_output_klein(inputs, save=False)
        levine_fig = figure_carve_output_levine(inputs, save=False)
        klein_f = _panel_by_letter(klein_fig)["F"]
        levine_f = _panel_by_letter(levine_fig)["F"]

        klein_sizes = np.concatenate(
            [
                c.get_sizes()
                for c in klein_f.collections
                if isinstance(c, PathCollection)
            ]
        )
        levine_sizes = np.concatenate(
            [
                c.get_sizes()
                for c in levine_f.collections
                if isinstance(c, PathCollection)
            ]
        )
        np.testing.assert_allclose(klein_sizes, 20.0)
        np.testing.assert_allclose(levine_sizes, 8.0)
        plt.close(klein_fig)
        plt.close(levine_fig)


class TestClusterIdsAgreeWithTheComposite:
    """Fig 3 and Fig 5 must call the same cluster "cluster 1".

    CARVE's plot_* methods take no labels argument -- each resolves its own
    through get_labels -- so the only hook is get_labels' own relabeling
    onto reference_labels. These pin that it is actually used, and that the
    fitted object is handed back unchanged.
    """

    def test_the_figure_draws_while_labels_are_aligned(self, inputs, monkeypatch):
        carve = inputs.carve
        seen = []
        original = carve.plot_cluster_violin

        def spy(*args, **kwargs):
            seen.append(np.asarray(carve.reference_labels).copy())
            return original(*args, **kwargs)

        monkeypatch.setattr(carve, "plot_cluster_violin", spy)
        fig = figure_carve_output_klein(inputs, save=False)
        plt.close(fig)

        assert seen, "the figure never called plot_cluster_violin"
        assert np.array_equal(seen[0], reference_codes(inputs.y))

    def test_aligned_labels_match_what_the_composite_computed(self, inputs):
        # Inside the context, the labels CARVE's own methods resolve are the
        # ones _align_to_reference gives the composite -- same relabeling,
        # so the two figures agree on both color and cluster number.
        carve = inputs.carve
        saved = carve.reference_labels
        selection = {
            "measure": inputs.measure,
            "rule": inputs.rule,
            "not_two": inputs.not_two,
        }
        try:
            raw = np.asarray(carve.get_labels(**selection))
            with carve_labels_aligned(carve, inputs.y):
                drawn = np.asarray(carve.get_labels(**selection))
        finally:
            carve.reference_labels = saved

        assert np.array_equal(drawn, _align_to_reference(raw, inputs.y))

    def test_the_fitted_object_is_left_as_it_was_found(self, inputs):
        # The carve object is shared with the composite figure and is often
        # a loaded cache the caller draws from more than once.
        carve = inputs.carve
        before = carve.reference_labels
        fig = figure_carve_output_klein(inputs, save=False)
        plt.close(fig)
        assert carve.reference_labels is before


@pytest.fixture
def coarser_inputs(inputs):
    """Four reported labels over the fixture's three blobs, as on Levine.

    With fewer clusters than reported labels, CARVE keeps its own ids inside
    carve_labels_aligned, while the composite's ids follow the reported
    codes. The labels are named in reverse so the two sets of ids differ.
    """
    y = np.array(["d"] * 15 + ["c"] * 15 + ["b"] * 10 + ["a"] * 5)
    raw = inputs.carve.get_labels(measure="stability", rule="1se")
    return dataclasses.replace(inputs, y=y, carve_labels=_align_to_reference(raw, y))


class TestClusterColorsMatchTheComposite:
    def test_e_draws_each_cell_in_its_composite_color(self, coarser_inputs):
        assert set(np.unique(coarser_inputs.carve_labels).tolist()) != {0, 1, 2}
        _, carve_cmap, _ = composite_color_maps(coarser_inputs)
        fig = figure_carve_output_levine(coarser_inputs, save=False)
        points = _panel_by_letter(fig)["E"].collections[0]
        # E draws the cells sorted by score; find each one by its position.
        cell_at = {tuple(xy): i for i, xy in enumerate(coarser_inputs.Z.tolist())}
        cells = [cell_at[tuple(xy)] for xy in points.get_offsets().tolist()]
        expected = [
            mcolors.to_rgb(carve_cmap[int(coarser_inputs.carve_labels[i])])
            for i in cells
        ]
        np.testing.assert_allclose(points.get_facecolors()[:, :3], expected)
        plt.close(fig)


# The Levine study sweeps Leiden resolution, so its S Fig draws the generic
# six panels from a resolution run: three blobs, three resolutions, a
# Jaccard-weighted graph as in the study.
LEVINE_RESOLUTIONS = (0.05, 0.2, 1.0)


@pytest.fixture(scope="module")
def leiden_inputs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    y = rng.integers(0, 3, 240)
    X = centers[y] + rng.normal(0, 0.6, (240, 2))
    spec = EstimatorSpec(
        name="leiden", params=(("n_neighbors", 10), ("weighting", "jaccard"))
    )
    carve = CARVE(
        estimator_param_grids=resolution_grids(spec, LEVINE_RESOLUTIONS),
        n_resamples=3,
        random_state=0,
    ).fit(X)
    curves = pd.DataFrame(
        {"metric": ["silhouette"], "model": ["Leiden"], "resolution": [0.05],
         "k": [3], "score": [0.6], "ari": [0.9]}
    )
    return CompositeInputs(
        X=X,
        y=np.array(["a", "b", "c"])[y],
        Z=X,
        carve=carve,
        carve_labels=np.asarray(carve.get_labels(measure="stability", rule="1se")),
        comparison_labels=y,
        comparison_name="Silhouette",
        comparison_k=3,
        curves_df=curves,
        best_df=curves,
        comparison_resolution=0.05,
    )


class TestCarveOutputOverResolution:
    def test_titles_its_curves_over_resolution(self, leiden_inputs):
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        panels = _panel_by_letter(fig)
        assert panels["A"].get_title() == "Stability ARI over resolution"
        assert panels["B"].get_title() == "Generalizability ARI over resolution"
        plt.close(fig)

    def test_draws_its_curves_on_a_log_axis_with_one_tick_per_resolution(
        self, leiden_inputs
    ):
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        panels = _panel_by_letter(fig)
        for letter in ("A", "B"):
            assert panels[letter].get_xscale() == "log"
            np.testing.assert_allclose(panels[letter].get_xticks(), LEVINE_RESOLUTIONS)
        plt.close(fig)

    def test_no_text_carries_the_full_leiden_configuration(self, leiden_inputs):
        # B's legend and D's note ran past the figure edge with it.
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        texts = [text.get_text() for text in fig.findobj(mtext.Text)]
        assert not any("objective_function" in text for text in texts)
        assert any("Leiden, 10 neighbors, Jaccard" in text for text in texts)
        plt.close(fig)

    def test_curve_panels_keep_a_third_of_the_figure_width(self, leiden_inputs):
        # tight_layout sizes each column for E's and F's notes; a one-line
        # "Leiden, ..., (resolution = ..., k ≈ ...)" halved A and B and piled
        # up their resolution labels. The k-based figure gives them a third.
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        panels = _panel_by_letter(fig)
        for letter in ("A", "B"):
            assert panels[letter].get_position().width > 0.28
        plt.close(fig)

    def test_columns_meet_without_a_legend_gap(self, leiden_inputs):
        # E's cluster legend once widened the left column's margin, leaving
        # a fifth of the figure blank between A and B.
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        panels = _panel_by_letter(fig)
        gap = panels["B"].get_position().x0 - panels["A"].get_position().x1
        assert gap < 0.12
        plt.close(fig)

    def test_consensus_letter_lines_up_with_a_and_the_matrix_stays_centered(
        self, leiden_inputs
    ):
        # The square matrix sits centered in its wider cell; its letter, placed
        # relative to the matrix, fell right of A's and E's.
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        fig.draw_without_rendering()
        panels = _panel_by_letter(fig)
        a, c = panels["A"], panels["C"]
        to_figure = fig.transFigure.inverted()

        def letter_right_edge(ax, letter):
            text = next(t for t in ax.texts if t.get_text() == letter)
            return text.get_window_extent().transformed(to_figure).x1

        matrix, cell = c.get_position(), c.get_position(original=True)
        assert (matrix.x0 + matrix.x1) / 2 == pytest.approx((cell.x0 + cell.x1) / 2, abs=1e-3)
        assert matrix.x0 > a.get_position().x0 + 0.01
        assert letter_right_edge(c, "C") == pytest.approx(
            letter_right_edge(a, "A"), abs=1e-3
        )
        plt.close(fig)

    def test_legend_names_the_resolution_panels_c_to_f_show(self, leiden_inputs):
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        resolution = leiden_inputs.carve.get_sweep_value(measure="stability", rule="1se")
        legend = _panel_by_letter(fig)["A"].get_legend()
        texts = [text.get_text() for text in legend.get_texts()]
        assert f"Selected resolution (1-SE rule): {resolution:g}" in texts
        plt.close(fig)

    def test_consensus_panel_names_the_selected_partition_s_k(self, leiden_inputs):
        fig = figure_carve_output_levine(leiden_inputs, save=False)
        k = len(np.unique(leiden_inputs.carve_labels))
        assert f"$k={k}$" in _panel_by_letter(fig)["C"].get_title()
        plt.close(fig)


# hECA's overview draws a pinned resolution rather than the rule's pick, on
# an anchored fit, with the two scatters on a subsample. The fixture is
# shaped the same way at toy size: two Leiden settings over three
# resolutions, 60 anchors. At this seed both criteria's 1-SE rule select
# resolution 0.5 (3 clusters) while the pin, 1.0, resolves to 5 clusters,
# so every test below can tell the pinned partition from the selected one.
HECA_RESOLUTIONS = (0.5, 1.0, 2.0)
HECA_PIN = 1.0
HECA_ANCHORS = 60


@pytest.fixture(scope="module")
def anchored_leiden():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = centers[rng.integers(0, 3, 240)] + rng.normal(0, 0.6, (240, 2))
    grids = []
    for n_neighbors in (10, 20):
        spec = EstimatorSpec(name="leiden", params=(("n_neighbors", n_neighbors),))
        grids += resolution_grids(spec, HECA_RESOLUTIONS)
    with pytest.warns(RuntimeWarning, match="anchored consensus"):
        carve = CARVE(
            estimator_param_grids=grids,
            n_resamples=3,
            random_state=0,
            consensus_anchors=HECA_ANCHORS,
        ).fit(X)
    labels = carve.get_labels(sweep_value=HECA_PIN)
    # One row from every cluster first, so the subsample is valid whatever
    # the cluster sizes; the rest at random.
    first = [int(np.flatnonzero(labels == lab)[0]) for lab in np.unique(labels)]
    rest = rng.choice(np.setdiff1d(np.arange(len(X)), first), 96, replace=False)
    rows = np.sort(np.concatenate([first, rest]))
    embedding = rng.normal(size=(len(rows), 2))
    return carve, rows, embedding


def _draw_heca(anchored_leiden, **kwargs):
    carve, rows, embedding = anchored_leiden
    kwargs.setdefault("save", False)
    return figure_carve_output_heca(
        carve,
        sweep_value=HECA_PIN,
        resolutions=HECA_RESOLUTIONS,
        rows=rows,
        embedding=embedding,
        **kwargs,
    )


def _vertical_line_positions(ax):
    """x of every vertical line on ax (two points sharing one x)."""
    positions = set()
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        if x.size == 2 and x[0] == x[1]:
            positions.add(float(x[0]))
    return positions


class TestCarveOutputHeca:
    def test_fixture_pin_differs_from_the_rule_selection(self, anchored_leiden):
        carve, _, _ = anchored_leiden
        for measure in ("stability", "generalizability"):
            assert float(carve.get_sweep_value(measure=measure)) != HECA_PIN

    def test_saves_under_its_own_filename(self, anchored_leiden, tmp_path):
        fig = _draw_heca(anchored_leiden, save=True, out_dir=tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["CARVE_output_heca.png"]
        plt.close(fig)

    def test_save_false_writes_nothing(self, anchored_leiden, tmp_path):
        fig = _draw_heca(anchored_leiden, out_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
        plt.close(fig)

    def test_draws_six_lettered_panels(self, anchored_leiden):
        fig = _draw_heca(anchored_leiden)
        assert set(_panel_by_letter(fig)) == set("ABCDEF")
        plt.close(fig)

    @pytest.mark.parametrize(
        ("letter", "measure"), [("A", "stability"), ("B", "generalizability")]
    )
    def test_curves_mark_the_rule_selection_and_the_shown_resolution(
        self, anchored_leiden, letter, measure
    ):
        carve, _, _ = anchored_leiden
        fig = _draw_heca(anchored_leiden)
        ax = _panel_by_letter(fig)[letter]
        assert ax.get_xscale() == "log"
        selected = float(carve.get_sweep_value(measure=measure))
        assert _vertical_line_positions(ax) == {selected, HECA_PIN}
        plt.close(fig)

    def test_panel_b_legend_names_each_setting_by_neighbor_count(
        self, anchored_leiden
    ):
        fig = _draw_heca(anchored_leiden)
        legend = _panel_by_letter(fig)["B"].get_legend()
        texts = {text.get_text() for text in legend.get_texts()}
        assert {"Leiden, 10 neighbors", "Leiden, 20 neighbors"} <= texts
        assert f"Shown in C to F: {HECA_PIN:g}" in texts
        plt.close(fig)

    def test_no_text_carries_the_estimator_repr(self, anchored_leiden):
        """CARVE's configuration note and full estimator names overflow the
        figure at hECA's panel width; neither is drawn anywhere."""
        fig = _draw_heca(anchored_leiden)
        texts = [t.get_text() for t in fig.findobj(mtext.Text)]
        assert not any("LeidenClustering" in text for text in texts)
        plt.close(fig)

    def test_consensus_panel_is_the_pinned_anchor_block(self, anchored_leiden):
        """The drawn matrix is the pinned configuration's, not the rule's.

        The references are CARVE's own renderings, which fill the pairs a
        short fit never co-sampled. Cluster order permutes rows and columns,
        so the comparison is on the sorted values, which it leaves unchanged.
        """
        carve, _, _ = anchored_leiden

        def rendered(**pin):
            ref_fig, ref_ax = plt.subplots()
            carve.plot_consensus_matrix(ax=ref_ax, **pin)
            values = np.sort(np.asarray(ref_ax.images[0].get_array()).ravel())
            plt.close(ref_fig)
            return values

        pinned = rendered(sweep_value=HECA_PIN)
        selected = rendered()
        assert not np.allclose(pinned, selected)

        fig = _draw_heca(anchored_leiden)
        drawn = np.asarray(_panel_by_letter(fig)["C"].images[0].get_array())
        assert drawn.shape == (HECA_ANCHORS, HECA_ANCHORS)
        assert np.allclose(np.sort(drawn.ravel()), pinned)
        plt.close(fig)

    def test_violins_cover_the_pinned_clusters_without_dots(self, anchored_leiden):
        carve, _, _ = anchored_leiden
        fig = _draw_heca(anchored_leiden)
        ax_d = _panel_by_letter(fig)["D"]
        bodies = [c for c in ax_d.collections if isinstance(c, PolyCollection)]
        n_clusters = np.unique(carve.get_labels(sweep_value=HECA_PIN)).size
        assert len(bodies) == n_clusters
        assert not any(isinstance(c, PathCollection) for c in ax_d.collections)
        plt.close(fig)

    def test_panel_e_draws_the_subsample_in_its_pinned_cluster_colors(
        self, anchored_leiden
    ):
        """Every drawn point is a subsample row, colored by that row's
        cluster at the pin. A figure that drew the rule's labels, or the
        labels of the wrong rows, gives at least one point the wrong color.
        """
        carve, rows, embedding = anchored_leiden
        labels = carve.get_labels(sweep_value=HECA_PIN)[rows]
        uniq = np.unique(labels)
        palette = cluster_colors(uniq.size)
        expected = {
            tuple(np.round(point, 9)): mcolors.to_rgb(
                palette[int(np.searchsorted(uniq, lab))]
            )
            for point, lab in zip(embedding, labels, strict=True)
        }
        fig = _draw_heca(anchored_leiden)
        ax_e = _panel_by_letter(fig)["E"]
        (points,) = [c for c in ax_e.collections if isinstance(c, PathCollection)]
        offsets = np.asarray(points.get_offsets())
        colors = np.asarray(points.get_facecolors())[:, :3]
        assert len(offsets) == len(rows)
        for point, color in zip(offsets, colors, strict=True):
            assert np.allclose(color, expected[tuple(np.round(point, 9))])
        plt.close(fig)

    def test_panel_f_groups_the_subsample_by_pinned_cluster(self, anchored_leiden):
        carve, rows, embedding = anchored_leiden
        labels = carve.get_labels(sweep_value=HECA_PIN)[rows]
        by_point = {
            tuple(np.round(p, 9)): lab
            for p, lab in zip(embedding, labels, strict=True)
        }
        fig = _draw_heca(anchored_leiden)
        ax_f = _panel_by_letter(fig)["F"]
        groups = [
            np.asarray(c.get_offsets())
            for c in ax_f.collections
            if isinstance(c, PathCollection) and len(c.get_offsets())
        ]
        assert sum(len(g) for g in groups) == len(rows)
        assert len(groups) == np.unique(labels).size
        for group in groups:
            assert len({by_point[tuple(np.round(p, 9))] for p in group}) == 1
        plt.close(fig)

    def test_panels_e_and_f_carry_no_cluster_legend(self, anchored_leiden):
        fig = _draw_heca(anchored_leiden)
        panels = _panel_by_letter(fig)
        assert panels["E"].get_legend() is None
        assert panels["F"].get_legend() is None
        # F keeps its score colorbar.
        assert any(ax.get_label() == "<colorbar>" for ax in fig.get_axes())
        plt.close(fig)

    @pytest.mark.parametrize("letter", ["E", "F"])
    def test_scatter_outlines_take_the_hECA_linewidth(self, anchored_leiden, letter):
        fig = _draw_heca(anchored_leiden)
        ax = _panel_by_letter(fig)[letter]
        points = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert points
        for collection in points:
            assert np.allclose(collection.get_linewidths(), MARKER_LINEWIDTH)
        plt.close(fig)

    def test_scatter_markers_take_the_hECA_size(self, anchored_leiden):
        fig = _draw_heca(anchored_leiden)
        panels = _panel_by_letter(fig)
        (points_e,) = [
            c for c in panels["E"].collections if isinstance(c, PathCollection)
        ]
        # A cell the toy fit never co-sampled has a NaN score and a NaN size.
        sizes_e = points_e.get_sizes()
        assert np.nanmin(sizes_e) == pytest.approx(MARKER_SIZE * 0.75)
        assert np.nanmax(sizes_e) == pytest.approx(MARKER_SIZE * 3.0)
        for collection in panels["F"].collections:
            if isinstance(collection, PathCollection) and len(collection.get_offsets()):
                assert np.allclose(collection.get_sizes(), MARKER_SIZE)
        plt.close(fig)

    def test_scatter_row_is_taller_and_rows_a_to_d_keep_their_size(
        self, anchored_leiden
    ):
        fig = _draw_heca(anchored_leiden)
        gs = (
            _panel_by_letter(fig)["E"]
            .get_subplotspec()
            .get_topmost_subplotspec()
            .get_gridspec()
        )
        assert tuple(gs.get_height_ratios()) == (1.0, 1.0, 1.4)
        # Each of rows A-B and C-D stays 14/3 inches; row E-F gets 1.4 of one.
        assert fig.get_size_inches()[1] == pytest.approx(14.0 / 3 * 3.4)
        plt.close(fig)

    def test_a_subsample_missing_a_cluster_is_refused(self, anchored_leiden):
        """CARVE's scatters color by position among the labels they are
        given, so a missing cluster would shift every later cluster's color
        away from panels C and D."""
        carve, rows, embedding = anchored_leiden
        labels = carve.get_labels(sweep_value=HECA_PIN)
        keep = labels[rows] != labels[rows][0]
        with pytest.raises(ValueError, match="every cluster"):
            figure_carve_output_heca(
                carve,
                sweep_value=HECA_PIN,
                resolutions=HECA_RESOLUTIONS,
                rows=rows[keep],
                embedding=embedding[keep],
                save=False,
            )

