"""Tests for the hECA CARVE figure."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from benchmarks._estimators import resolution_grids
from benchmarks._types import EstimatorSpec
from benchmarks.figures import figure_heca_carve
from carve import CARVE


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
    X = centers[rng.integers(0, 3, 240)] + rng.normal(0, 0.6, (240, 2))
    grids = []
    for k in (10, 20):
        spec = EstimatorSpec(name="leiden", params=(("n_neighbors", k),))
        grids += resolution_grids(spec, (0.5, 1.0))
    return CARVE(estimator_param_grids=grids, n_resamples=3, random_state=0).fit(X)


def test_draws_three_panels_on_a_log_resolution_axis(fitted, tmp_path):
    fig = figure_heca_carve(fitted, resolutions=(0.5, 1.0), out_dir=tmp_path)
    assert isinstance(fig, Figure)
    panels = fig.axes[:3]
    assert len(panels) == 3
    assert all(ax.get_xscale() == "log" for ax in panels)
    assert (tmp_path / "heca_carve.png").is_file()
