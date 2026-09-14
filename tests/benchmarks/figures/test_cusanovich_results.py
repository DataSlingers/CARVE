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
from benchmarks.figures._cusanovich_results import MARKER_SIZE, SOURCE_TSNE_LABELS
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
    # order different from their ids, so a color map keyed on first
    # appearance would disagree with the aligned one. The stub selects
    # resolution 0.6, where resolution_results observes 12 clusters.
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
        embedding_A=rng.normal(size=(n, 2)),
        embedding_A_labels=("LSI 1", "LSI 2"),
        source_tsne=rng.normal(size=(n, 2)),
        best_pipeline_row=best,
        operating_point=(0.8, 16.0),
        published_generalizability=(0.55, 0.02),
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


def test_titles_name_the_selection_and_the_source(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    title = panels["A"].get_title()
    assert "CARVE: Leiden, resolution 0.6, 12 clusters" in title
    assert "consensus over LSI, t-SNE, UMAP" in title
    assert "shown on LSI 1/2" in title
    assert panels["B"].get_title() == "Cusanovich et al.: Louvain on t-SNE, 6 clusters"


def test_a_draws_the_best_pipelines_embedding_and_b_the_source_tsne(inputs):
    panels = _panels(figure_cusanovich_results(inputs, save=False))
    for letter, points, names in (
        ("A", inputs.embedding_A, inputs.embedding_A_labels),
        ("B", inputs.source_tsne, SOURCE_TSNE_LABELS),
    ):
        ax = panels[letter]
        drawn = np.concatenate([c.get_offsets() for c in _scatter_collections(ax)])
        assert sorted(map(tuple, drawn)) == sorted(map(tuple, points))
        assert set(names) <= {text.get_text() for text in ax.texts}
        sizes = np.concatenate([c.get_sizes() for c in _scatter_collections(ax)])
        np.testing.assert_allclose(sizes, MARKER_SIZE)


def test_a_carve_cluster_takes_the_color_of_the_source_cluster_it_matches(inputs):
    # carve_labels are relabeled onto y's sorted codes, so CARVE cluster 0
    # and source cluster "1" share a palette entry, as do 3 and "4".
    panels = _panels(figure_cusanovich_results(inputs, save=False))

    def color_of(ax, label):
        (collection,) = [c for c in _scatter_collections(ax) if c.get_label() == label]
        return mcolors.to_hex(collection.get_facecolor()[0])

    assert color_of(panels["A"], "0") == color_of(panels["B"], "1")
    assert color_of(panels["A"], "3") == color_of(panels["B"], "4")


def test_c_marks_the_operating_point_and_the_published_partition(inputs):
    ax = _panels(figure_cusanovich_results(inputs, save=False))["C"]
    assert ax.get_xlabel() == "Resolution"

    at_point = inputs.carve.estimator_results_.set_index("sweep_value").loc[0.8]
    markers = [line for line in ax.lines if line.get_marker() == "D"]
    assert sorted((line.get_xdata()[0], line.get_ydata()[0]) for line in markers) == (
        sorted(
            [(0.8, at_point["ari_stability"]), (0.8, at_point["ari_generalizability"])]
        )
    )
    horizontal = [line for line in ax.lines if list(line.get_ydata()) == [0.55, 0.55]]
    assert len(horizontal) == 1

    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "source operating point, 16 clusters on t-SNE" in legend
    assert "published 6 clusters, RF probe" in legend


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


def test_every_selection_uses_the_inputs_rule_and_not_two(inputs):
    figure_cusanovich_results(replace(inputs, rule="max", not_two=True), save=False)
    calls = inputs.carve.selection_calls
    assert calls
    assert all(call["rule"] == "max" and call["not_two"] for call in calls)
