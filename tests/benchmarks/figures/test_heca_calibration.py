"""Tests for the hECA calibration figure."""

import pandas as pd
import pytest
from matplotlib.figure import Figure

from benchmarks.figures import figure_heca_calibration


def _scan(settings=("leiden_15", "leiden_50")):
    rows = []
    for offset, setting in enumerate(settings):
        for resolution, n in ((0.01, 3), (0.1, 8), (1.0, 60), (10.0, 300)):
            rows.append(
                {
                    "setting": setting,
                    "resolution": resolution,
                    "n_clusters": n + offset,
                    "ari_organ": 0.5,
                    "ari_cell_type": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_draws_counts_and_agreement_and_saves(tmp_path):
    fig = figure_heca_calibration(
        _scan(), grid=(0.1, 1.0), lower_target=5, upper_target=108, out_dir=tmp_path
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    assert (tmp_path / "heca_calibration.png").is_file()


def test_an_empty_scan_raises():
    with pytest.raises(ValueError, match="empty"):
        figure_heca_calibration(
            _scan().iloc[:0], grid=(0.1, 1.0), lower_target=5, upper_target=108,
            save=False,
        )


def test_more_settings_than_colors_raises():
    with pytest.raises(ValueError, match="settings"):
        figure_heca_calibration(
            _scan(("a", "b", "c")), grid=(0.1, 1.0), lower_target=5,
            upper_target=108, save=False,
        )
