"""Tests for Fig 1."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection

from benchmarks._theme import PRINT_FONT_SIZES, PRINT_MAX_HEIGHT_IN, PRINT_WIDTH_IN
from benchmarks.figures import DatasetRow, figure_clustering_problem

RNG = np.random.default_rng(0)


def _row(name: str, ks: tuple[int, ...], marker_size: float) -> DatasetRow:
    embedding = RNG.normal(size=(60, 2))
    partitions = {k: RNG.integers(0, k, size=60) for k in ks}
    return DatasetRow(
        name=name, embedding=embedding, partitions=partitions, marker_size=marker_size
    )


@pytest.fixture
def rows():
    return [_row("Levine", (6, 10, 14, 18), 12.0), _row("Klein", (2, 3, 4, 5), 30.0)]


def _scatters(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def test_draws_one_row_per_data_set_and_one_panel_per_k(rows):
    fig = figure_clustering_problem(rows, save=False)
    axes = np.asarray(fig.axes).reshape(2, 4)
    for row, panels in zip(rows, axes):
        assert [ax.get_title() for ax in panels] == [f"k={k}" for k in row.partitions]
        assert panels[0].get_ylabel() == row.name
    plt.close(fig)


def test_each_panel_colors_its_own_partition(rows):
    fig = figure_clustering_problem(rows, save=False)
    axes = np.asarray(fig.axes).reshape(2, 4)
    for row, panels in zip(rows, axes):
        for ax, (k, labels) in zip(panels, row.partitions.items()):
            drawn = {int(c.get_label()) for c in _scatters(ax)}
            assert drawn == set(np.unique(labels).tolist())
    plt.close(fig)


def test_is_laid_out_at_the_print_sizes(rows):
    fig = figure_clustering_problem(rows, save=False)
    assert fig.get_figwidth() == PRINT_WIDTH_IN
    assert fig.get_figheight() <= PRINT_MAX_HEIGHT_IN
    for ax in fig.axes:
        assert ax.title.get_fontsize() == PRINT_FONT_SIZES["title"]
        assert ax.yaxis.label.get_fontsize() <= PRINT_FONT_SIZES["title"]
    letters = [t for t in fig.texts if t.get_text() in {"A", "B"}]
    assert {t.get_text() for t in letters} == {"A", "B"}
    assert {t.get_fontsize() for t in letters} == {PRINT_FONT_SIZES["panel_letter"]}
    plt.close(fig)


def test_saves_the_png_and_the_plos_figure_file_as_fig1(rows, tmp_path):
    from PIL import Image

    fig = figure_clustering_problem(rows, save=True, out_dir=tmp_path)
    plt.close(fig)
    assert (tmp_path / "clustering_problem.png").is_file()
    with Image.open(tmp_path / "Fig1.tif") as image:
        assert image.mode == "RGB"
        assert image.info["compression"] == "tiff_lzw"


def test_save_false_writes_nothing(rows, tmp_path):
    fig = figure_clustering_problem(rows, save=False, out_dir=tmp_path)
    plt.close(fig)
    assert list(tmp_path.iterdir()) == []


def test_rejects_rows_with_different_numbers_of_panels(rows):
    rows[1] = _row("Klein", (2, 3, 4), 30.0)
    with pytest.raises(ValueError, match="same number"):
        figure_clustering_problem(rows, save=False)
