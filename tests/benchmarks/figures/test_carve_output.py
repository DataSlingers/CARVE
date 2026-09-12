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
tests below tell panel A (stability) apart from panel C (generalizability)
by more than a coincidentally-matching selected k, and is the most direct
guard against the restructure's most likely regression: swapping the two
measures between those panels.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection, PolyCollection
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from benchmarks._theme import cluster_colors
from benchmarks.figures import figure_carve_output_klein, figure_carve_output_levine
from benchmarks.figures._case_study import (
    CompositeInputs,
    _align_to_reference,
    carve_labels_aligned,
    reference_codes,
)
from carve import CARVE


def _panel_by_letter(fig):
    """Map each panel letter to its axes via the text panel_letter() drew.

    fig.get_axes() also returns the consensus matrix's divider-appended band
    and colorbar axes, interleaved in creation order, so a fixed positional
    index into that list does not reliably pick out "panel C". The letter
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

    def test_panel_a_is_stability_and_panel_c_is_generalizability(self, inputs):
        """The restructure's most likely regression: swapping A and C.

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
        assert _line_matching(panels["C"], generalizability_y, x)
        assert not _line_matching(panels["C"], stability_y, x)
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

    def test_panel_b_consensus_matrix_is_stabilitys_selected_configuration(
        self, inputs
    ):
        """Panel B draws stability's consensus matrix, not generalizability's.

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
        ax_b = _panel_by_letter(fig)["B"]
        assert len(ax_b.images) == 1
        drawn = np.asarray(ax_b.images[0].get_array())

        carve = inputs.carve
        stability_k = int(carve.get_k(measure="stability", rule="1se"))
        generalizability_k = int(carve.get_k(measure="generalizability", rule="1se"))
        assert stability_k != generalizability_k
        assert ax_b.get_title() == f"Consensus matrix ($k={stability_k}$)"

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

        ``source`` selects which per-sample score panel E encodes; a
        ``source="accuracy"`` regression would still draw one scatter with
        varying point sizes (the previous test's only check), so it needs
        its own guard. plot_cluster_scatter's auto-generated legend title
        names the score source directly.
        """
        fig = figure_carve_output_klein(inputs, save=False)
        ax_e = _panel_by_letter(fig)["E"]
        legend = ax_e.get_legend()
        assert legend is not None
        assert "Gini Stability" in legend.get_title().get_text()
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
