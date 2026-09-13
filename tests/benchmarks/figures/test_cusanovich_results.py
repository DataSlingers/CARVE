import numpy as np
import pandas as pd
import pytest

from benchmarks.figures import figure_cusanovich_results
from benchmarks.figures._cusanovich_results import AXIS_LABELS, MARKER_SIZE
from tests.benchmarks._helpers import StubCarve, simple_results


@pytest.fixture
def inputs():
    from benchmarks.figures._case_study import CompositeInputs

    rng = np.random.default_rng(0)
    n = 90
    y = np.repeat(["Lung", "Liver", "Spleen"], n // 3)
    Z = rng.normal(size=(n, 2))
    curves = pd.DataFrame(
        {
            "metric": np.repeat(["silhouette", "gap"], 6),
            "model": "KMeans",
            "k": list(range(4, 10)) * 2,
            "score": rng.random(12),
            "ari": rng.random(12),
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["KMeans", "KMeans"],
            "k": [4, 5],
            "score": [0.4, 0.3],
            "ari": [0.5, 0.4],
        }
    )

    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=Z,
        carve=StubCarve(
            simple_results([4, 5, 6], "KMeans"), select=lambda m, nt: ("m0", 5)
        ),
        carve_labels=rng.integers(0, 3, n),
        comparison_labels=rng.integers(0, 3, n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=curves,
        best_df=best,
    )


def test_axis_labels_name_the_source_tsne():
    # Panels A to C draw the source publication's own t-SNE coordinates,
    # which the loader carries through as meta["source_tsne"] and the
    # notebook passes to prepare_composite; the overview scatter that opens
    # the notebook draws the same embedding. Labeling them LSI or PCA would
    # misreport what is drawn.
    assert AXIS_LABELS == ("t-SNE 1", "t-SNE 2")


def test_marker_size_matches_the_levine_scale():
    # Thousands of cells, like Levine, not Klein's 1,358.
    assert MARKER_SIZE == 8.0


def test_figure_has_six_panels_and_does_not_write_when_save_is_false(inputs, tmp_path):
    fig = figure_cusanovich_results(inputs, save=False, out_dir=tmp_path)
    drawn = [ax for ax in fig.axes if ax.has_data() or ax.get_title()]
    assert len(drawn) >= 6
    assert list(tmp_path.iterdir()) == []


def test_figure_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_cusanovich_results(inputs, save=True, out_dir=tmp_path)
    assert (tmp_path / "cusanovich_results.png").is_file()


def test_bottom_panel_is_the_ari_lollipop_not_an_alluvial(inputs):
    # The notebook's section heading and summary both describe panel F as
    # the ARI comparison against the reported tissue labels; the alluvial
    # the module first shipped did not report that number anywhere.
    fig = figure_cusanovich_results(inputs, save=False)
    labels = [ax.get_xlabel() for ax in fig.get_axes()]
    assert any("ARI" in label for label in labels)
    titles = [ax.get_title() for ax in fig.get_axes()]
    assert not any("Reported Tissue" in title for title in titles)
