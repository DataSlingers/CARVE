"""Tests for the two schematic guidance figures."""

import numpy as np
import pytest

from benchmarks._theme import PRINT_MAX_HEIGHT_IN, PRINT_WIDTH_IN
from benchmarks.figures import _guidance, figure_guidance_clusters, figure_guidance_global

FIGURES = (
    (figure_guidance_global, "guidance_global"),
    (figure_guidance_clusters, "guidance_clusters"),
)


@pytest.mark.parametrize(("draw", "name"), FIGURES)
class TestGuidanceFigures:
    def test_fits_the_plos_page(self, draw, name):
        width, height = draw(save=False).get_size_inches()
        assert width <= PRINT_WIDTH_IN
        assert height <= PRINT_MAX_HEIGHT_IN

    def test_saves_png_and_pdf(self, draw, name, tmp_path):
        draw(save=True, out_dir=tmp_path)
        assert (tmp_path / f"{name}.png").exists()
        assert (tmp_path / f"{name}.pdf").exists()

    def test_is_deterministic(self, draw, name):
        first, second = draw(save=False), draw(save=False)
        first.canvas.draw()
        second.canvas.draw()
        a = np.asarray(first.canvas.buffer_rgba())
        b = np.asarray(second.canvas.buffer_rgba())
        assert np.array_equal(a, b)


class TestQuadrantsEncodeTheMetaphor:
    """Tight darts mean stable; darts centered on the target mean generalizable."""

    @pytest.mark.parametrize("quadrant", _guidance.QUADRANTS)
    def test_dart_pattern_matches_the_row_and_column(self, quadrant):
        row, col, offset, spread = quadrant[:4]
        stable, generalizable = row == 0, col == 0
        assert (spread < 0.1) == stable
        assert (np.hypot(*offset) == 0.0) == generalizable

    def test_every_quadrant_appears_once(self):
        assert sorted((q[0], q[1]) for q in _guidance.QUADRANTS) == [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]


BOARDS = [
    board
    for outcome in _guidance.candidate_outcomes(np.random.default_rng(0))
    for board in outcome["boards"]
]


@pytest.mark.parametrize("board", BOARDS)
def test_numbered_targets_repeat_their_scenario(board):
    """A target in section B carries the number of the quadrant whose darts it shows."""
    offset, spread, scenario = board
    row, col = next(q[:2] for q in _guidance.QUADRANTS if 2 * q[0] + q[1] + 1 == scenario)
    assert (spread < 0.1) == (row == 0)
    assert (np.hypot(*offset) == 0.0) == (col == 0)


def test_global_curves_use_the_criterion_colors():
    fig = figure_guidance_global(save=False)
    expected = {_guidance.STABILITY.lower(), _guidance.GENERALIZABILITY.lower()}
    colors = {
        line.get_color().lower()
        for ax in fig.get_axes()
        for line in ax.get_lines()
        if line.get_marker() == "o"
    }
    assert colors == expected
