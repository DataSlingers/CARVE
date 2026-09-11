"""figure_reference_scatter: the case-study overview of the reference labels."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.colors import to_rgba

from benchmarks._panels import aligned_color_maps
from benchmarks.figures import figure_reference_scatter

N = 300
AXIS_LABELS = ("t-SNE 1", "t-SNE 2")


@pytest.fixture
def embedding():
    rng = np.random.default_rng(0)
    # Deliberately not in sorted order, so a legend or color map that keyed
    # on first occurrence would disagree with the composite's sorted one.
    y = np.repeat(["Spleen", "Liver", "Lung"], N // 3)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    Z = centers[np.searchsorted(["Liver", "Lung", "Spleen"], y)] + rng.normal(
        size=(N, 2)
    )
    return Z, y


def _scatter_ax(fig):
    return fig.axes[0]


def _drawn_points(ax):
    """(x, y, label) for every drawn marker, across all collections."""
    return {
        (float(x), float(y), collection.get_label())
        for collection in ax.collections
        for x, y in collection.get_offsets()
    }


def test_draws_one_collection_per_label_with_hidden_axes(embedding, tmp_path):
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, save=False, out_dir=tmp_path
    )
    ax = _scatter_ax(fig)
    assert len(ax.collections) == 3
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
    assert not any(spine.get_visible() for spine in ax.spines.values())


def test_title_is_set_when_given(embedding, tmp_path):
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, title="Reported Tissue",
        save=False, out_dir=tmp_path,
    )
    assert _scatter_ax(fig).get_title() == "Reported Tissue"


def test_marker_size_reaches_the_markers(embedding, tmp_path):
    # The notebooks pass their composite's own marker size so the overview
    # and the composite's panel A read at the same density.
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, marker_size=3.0, save=False, out_dir=tmp_path
    )
    for collection in _scatter_ax(fig).collections:
        assert collection.get_sizes()[0] == 3.0


def test_default_colors_are_the_composite_reported_label_colors(embedding, tmp_path):
    # The composite's panel A colors reported labels through
    # aligned_color_maps in sorted category order. The overview must use the
    # same map by default, so a tissue keeps its color from the first figure
    # in the notebook through to the composite.
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, save=False, out_dir=tmp_path
    )
    (expected,) = aligned_color_maps(y)
    for collection in _scatter_ax(fig).collections:
        drawn = collection.get_facecolor()[0][:3]
        assert np.allclose(drawn, to_rgba(expected[collection.get_label()])[:3])


def test_explicit_color_map_is_honored(embedding, tmp_path):
    Z, y = embedding
    color_map = {"Liver": "#112233", "Lung": "#445566", "Spleen": "#778899"}
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, color_map=color_map,
        save=False, out_dir=tmp_path,
    )
    for collection in _scatter_ax(fig).collections:
        drawn = collection.get_facecolor()[0][:3]
        assert np.allclose(drawn, to_rgba(color_map[collection.get_label()])[:3])


def test_axis_arrows_name_the_embedding_axes(embedding, tmp_path):
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=("UMAP 1", "UMAP 2"), save=False, out_dir=tmp_path
    )
    texts = {text.get_text() for text in _scatter_ax(fig).texts}
    assert {"UMAP 1", "UMAP 2"} <= texts


def test_legend_lists_each_label_once_in_sorted_order(embedding, tmp_path):
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, save=False, out_dir=tmp_path
    )
    assert len(fig.legends) == 1
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert labels == ["Liver", "Lung", "Spleen"]


def test_legend_order_follows_the_palette_order_for_integer_labels(tmp_path):
    # aligned_color_maps assigns the palette in sorted order of the raw
    # label values. The legend must walk the same order, so sorting the
    # string form ("1", "10", "2") would put the palette out of sequence.
    rng = np.random.default_rng(0)
    y = np.repeat([10, 2, 1], 20)
    Z = rng.normal(size=(60, 2))
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, save=False, out_dir=tmp_path
    )
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert labels == ["1", "2", "10"]


def test_max_points_draws_a_seeded_subsample_and_keeps_rows_together(
    embedding, tmp_path
):
    # One index for Z and y together: a drawn marker must be a real
    # (position, label) row of the input, not a position from one cell
    # paired with the label of another.
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, max_points=50, random_state=1,
        save=False, out_dir=tmp_path,
    )
    again = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, max_points=50, random_state=1,
        save=False, out_dir=tmp_path,
    )
    drawn = _drawn_points(_scatter_ax(fig))
    assert len(drawn) == 50
    assert drawn == _drawn_points(_scatter_ax(again))

    rows = {(float(a), float(b), str(label)) for (a, b), label in zip(Z, y)}
    assert drawn <= rows


def test_max_points_at_or_above_n_draws_every_point(embedding, tmp_path):
    Z, y = embedding
    fig = figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, max_points=N, save=False, out_dir=tmp_path
    )
    assert len(_drawn_points(_scatter_ax(fig))) == N


def test_writes_under_the_requested_name(embedding, tmp_path):
    Z, y = embedding
    figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, save=True, out_dir=tmp_path,
        save_name="cusanovich_reference_scatter.png",
    )
    assert (tmp_path / "cusanovich_reference_scatter.png").is_file()


def test_does_not_write_when_save_is_false(embedding, tmp_path):
    Z, y = embedding
    figure_reference_scatter(
        Z, y, axis_labels=AXIS_LABELS, save=False, out_dir=tmp_path
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_Z",
    [
        pytest.param(np.zeros((N - 1, 2)), id="row-count-differs-from-y"),
        pytest.param(np.zeros((N, 3)), id="not-two-dimensional"),
    ],
)
def test_misaligned_embedding_is_rejected(embedding, tmp_path, bad_Z):
    _, y = embedding
    with pytest.raises(ValueError, match="n_cells, 2"):
        figure_reference_scatter(
            bad_Z, y, axis_labels=AXIS_LABELS, save=False, out_dir=tmp_path
        )
