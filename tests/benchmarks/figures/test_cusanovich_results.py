"""Tests for the Cusanovich case-study figure."""

from dataclasses import replace

import matplotlib.colors as mcolors
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from sklearn.manifold import TSNE
from sklearn.preprocessing import FunctionTransformer

from benchmarks._cusanovich_compare import CusanovichInputs
from benchmarks.figures import figure_cusanovich_results
from benchmarks._theme import PRINT_FONT_SIZES, PRINT_MAX_HEIGHT_IN, PRINT_WIDTH_IN
from benchmarks.figures._cusanovich_results import _PRINT_MARKER_SCALE, MARKER_SIZE
from carve._pipeline import PipelineSpec, PipelineStep
from tests.benchmarks._helpers import StubCarve, pipeline_results, resolution_results

RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
IDENTITY = PipelineStep(cls=FunctionTransformer, params={}, name="identity")
SPECS = {
    spec.label: spec
    for spec in (
        PipelineSpec(normalization=IDENTITY, dim_reduction=step)
        for step in (
            IDENTITY,
            PipelineStep(cls=TSNE, params={"perplexity": 30}, name="TSNE"),
            PipelineStep(cls=object, params={"n_neighbors": 15}, name="UMAP"),
        )
    )
}


@pytest.fixture
def inputs():
    # Six source clusters of 20 cells; CARVE's four clusters are listed in an
    # order different from their ids. The stub selects resolution 0.6, where
    # resolution_results observes 12 clusters.
    rng = np.random.default_rng(0)
    n = 120
    carve = StubCarve(
        resolution_results(RESOLUTIONS, "LeidenClustering"),
        select=lambda measure, not_two: ("m0", 0.6),
        sweep_param="resolution",
    )
    carve.preprocessing_results_ = pipeline_results(tuple(SPECS), RESOLUTIONS)
    carve.preprocessing_pipelines_ = SPECS
    table = carve.preprocessing_results_
    best = table.loc[
        (table["sweep_value"] == 0.6) & (table["pipeline"] == "identity | identity")
    ].iloc[0]
    return CusanovichInputs(
        X=rng.normal(size=(n, 5)),
        y=np.repeat(["1", "2", "3", "4", "5", "6"], n // 6),
        carve=carve,
        carve_labels=np.repeat([3, 1, 0, 2], n // 4),
        map_coordinates=rng.normal(size=(n, 2)),
        map_name="t-SNE, perplexity 100",
        map_axis_labels=("t-SNE 1", "t-SNE 2"),
        best_pipeline_row=best,
        operating_point=(0.8, 16.0),
    )


def _panels(fig):
    """Map each panel letter to its axes via the text panel_letter drew."""
    return {
        text.get_text(): ax
        for ax in fig.get_axes()
        for text in ax.texts
        if text.get_text() in {"A", "B", "C", "D"}
    }


def _scatter_collections(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_draws_four_lettered_panels_and_writes_nothing_when_save_is_false(
    inputs, tmp_path
):
    fig = figure_cusanovich_results(inputs, save=False, out_dir=tmp_path)
    assert set(_panels(fig)) == set("ABCD")
    assert list(tmp_path.iterdir()) == []


def test_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_cusanovich_results(inputs, save=True, out_dir=tmp_path)
    assert (tmp_path / "cusanovich_results.png").is_file()


def test_writes_the_plos_figure_file_as_fig6(inputs, tmp_path):
    from PIL import Image

    figure_cusanovich_results(inputs, save=True, out_dir=tmp_path)
    with Image.open(tmp_path / "Fig6.tif") as image:
        assert image.mode == "RGB"
        assert image.info["compression"] == "tiff_lzw"


def test_is_laid_out_at_the_print_sizes(inputs):
    fig = figure_cusanovich_results(inputs, save=False)
    assert fig.get_figwidth() == PRINT_WIDTH_IN
    assert fig.get_figheight() <= PRINT_MAX_HEIGHT_IN
    for ax in fig.get_axes():
        assert ax.title.get_fontsize() <= PRINT_FONT_SIZES["title"]
        assert ax.xaxis.label.get_fontsize() <= PRINT_FONT_SIZES["axis_label"]
        for text in ax.texts:
            assert text.get_fontsize() <= PRINT_FONT_SIZES["panel_letter"]


def test_titles_name_the_selection_and_the_source(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    assert panels["A"].get_title() == (
        "CARVE: Leiden, resolution 0.6, 12 clusters\non t-SNE, perplexity 100"
    )
    assert panels["B"].get_title() == (
        "Cusanovich et al.: 6 clusters\non t-SNE, perplexity 100"
    )
    assert panels["C"].get_title() == (
        "Stability & Generalizability over resolution,\npooled over pipelines"
    )
    assert panels["D"].get_title() == (
        "Stability & Generalizability\nacross preprocessing pipelines"
    )


def test_a_and_b_both_draw_the_map(inputs):
    # A shows CARVE's partition on the map B shows the source's on, whichever
    # pipeline CARVE rates best.
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    for letter in ("A", "B"):
        ax = panels[letter]
        points, names = inputs.map_coordinates, inputs.map_axis_labels
        drawn = np.concatenate([c.get_offsets() for c in _scatter_collections(ax)])
        assert sorted(map(tuple, drawn)) == sorted(map(tuple, points))
        assert set(names) <= {text.get_text() for text in ax.texts}
        sizes = np.concatenate([c.get_sizes() for c in _scatter_collections(ax)])
        # MARKER_SIZE is given for the screen layout and scaled with the area
        # of the print-width figure.
        np.testing.assert_allclose(sizes, MARKER_SIZE * _PRINT_MARKER_SCALE)


def test_a_carve_cluster_takes_the_color_of_the_source_cluster_it_matches(inputs):
    # carve_labels are relabeled onto y's sorted codes, so CARVE cluster 0
    # and source cluster "1" share a palette entry, as do 3 and "4".
    panels = _panels(figure_cusanovich_results(inputs, save=False))

    def color_of(ax, label):
        (collection,) = [c for c in _scatter_collections(ax) if c.get_label() == label]
        return mcolors.to_hex(collection.get_facecolor()[0])

    assert color_of(panels["A"], "0") == color_of(panels["B"], "1")
    assert color_of(panels["A"], "3") == color_of(panels["B"], "4")


def test_carves_matches_take_the_palettes_distinct_colors_first(inputs):
    # Twelve source clusters; CARVE's two match codes 0 and 10. In palette
    # order code 10 would be the eleventh entry, a dark indigo; matched
    # first, it takes the second, tab10 orange, in both panels.
    n = 120
    y = np.repeat([f"{i:02d}" for i in range(12)], n // 12)
    fig = figure_cusanovich_results(
        replace(inputs, y=y, carve_labels=np.repeat([0, 10], n // 2)), save=False
    )
    panels = _panels(fig)

    def color_of(ax, label):
        (collection,) = [c for c in _scatter_collections(ax) if c.get_label() == label]
        return mcolors.to_hex(collection.get_facecolor()[0])

    assert color_of(panels["A"], "10") == "#ff7f0e"
    assert color_of(panels["B"], "10") == "#ff7f0e"


def test_c_marks_the_operating_point_and_draws_no_horizontal_line(inputs):
    ax = _panels(figure_cusanovich_results(inputs, save=False))["C"]
    assert ax.get_xlabel() == "Resolution"

    at_point = inputs.carve.estimator_results_.set_index("sweep_value").loc[0.8]
    markers = [line for line in ax.lines if line.get_marker() == "D"]
    assert sorted((line.get_xdata()[0], line.get_ydata()[0]) for line in markers) == (
        sorted(
            [(0.8, at_point["ari_stability"]), (0.8, at_point["ari_generalizability"])]
        )
    )
    horizontal = [
        line
        for line in ax.lines
        if len(line.get_ydata()) > 1 and len(set(line.get_ydata())) == 1
    ]
    assert horizontal == []

    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "nearest the source's 6 clusters (16 observed)" in legend
    assert not any("published" in text for text in legend)


def test_c_counts_the_observed_clusters_on_a_secondary_axis(inputs):
    fig = figure_cusanovich_results(inputs, save=False)
    fig.canvas.draw()
    (top,) = _panels(fig)["C"].child_axes
    assert [label.get_text() for label in top.get_xticklabels()] == [
        "4",
        "8",
        "12",
        "16",
        "20",
    ]


def test_d_draws_one_line_pair_per_pipeline_sharing_cs_x_axis(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    labels = [container.get_label() for container in panels["D"].containers]
    assert sorted(labels) == sorted([*SPECS, *SPECS])
    assert panels["D"].get_shared_x_axes().joined(panels["D"], panels["C"])


def test_c_and_d_draw_resolution_on_a_log_axis_ticked_at_the_grid(inputs):
    fig = figure_cusanovich_results(inputs, save=False)
    fig.canvas.draw()
    panels = _panels(fig)
    assert panels["C"].get_xscale() == "log"
    assert panels["D"].get_xscale() == "log"
    for letter in "CD":
        labels = [label.get_text() for label in panels[letter].get_xticklabels()]
        assert labels == ["0.2", "0.4", "0.6", "0.8", "1"]


def test_every_selection_uses_the_inputs_rule_and_not_two(inputs):
    figure_cusanovich_results(replace(inputs, rule="max", not_two=True), save=False)
    calls = inputs.carve.selection_calls
    assert calls
    assert all(call["rule"] == "max" and call["not_two"] for call in calls)


def test_a_long_resolution_grid_labels_every_other_value():
    from benchmarks.figures._cusanovich_results import resolution_tick_labels

    grid = np.geomspace(0.02, 3.0, 15)
    labels = resolution_tick_labels(grid)
    assert len(labels) == 15
    assert labels[0::2] == [f"{value:g}" for value in grid[0::2]]
    assert set(labels[1::2]) == {""}
    assert resolution_tick_labels(grid[:8]) == [f"{value:g}" for value in grid[:8]]


def test_c_and_d_titles_and_letters_share_one_line(inputs):
    # C's and D's titles sit on one line.
    fig = figure_cusanovich_results(inputs, save=False)
    fig.draw_without_rendering()
    panels = _panels(fig)
    to_figure = fig.transFigure.inverted()

    def extent(artist):
        return artist.get_window_extent().transformed(to_figure)

    c, d = panels["C"], panels["D"]
    assert extent(c.title).y0 == pytest.approx(extent(d.title).y0, abs=1e-3)
    letters = {
        letter: next(t for t in ax.texts if t.get_text() == letter)
        for letter, ax in (("C", c), ("D", d))
    }
    assert extent(letters["C"]).y1 == pytest.approx(extent(letters["D"]).y1, abs=1e-3)
    # The cluster axis label sits left-aligned under C's title.
    (top,) = c.child_axes
    label = top.xaxis.label
    assert label.get_horizontalalignment() == "left"
    assert extent(label).y1 < extent(c.title).y0


def test_c_names_the_one_estimator_in_a_not_in_every_legend_entry(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    texts = [t.get_text() for t in panels["C"].get_legend().get_texts()]
    carve = [text for text in texts if text.startswith("CARVE ")]
    assert carve == ["CARVE Stability (1SE)", "CARVE Generalizability (1SE)"]
