"""Tests for the hECA runtime figure."""

import pandas as pd
import pytest
from matplotlib.figure import Figure

from benchmarks.figures import figure_heca_runtime


def _components():
    rows = []
    for setting in ("leiden_15", "leiden_50"):
        for name, hours in (
            ("neighbor search", 30.0),
            ("graph assembly", 5.0),
            ("Leiden", 8.0),
        ):
            rows.append({"component": name, "setting": setting, "core_hours": hours})
    rows += [
        {"component": "forest fit", "setting": "all", "core_hours": 20.0},
        {"component": "forest predict", "setting": "all", "core_hours": 1.0},
        {"component": "CARVE overhead", "setting": "all", "core_hours": 4.0},
    ]
    table = pd.DataFrame(rows)
    table["share"] = table["core_hours"] / table["core_hours"].sum()
    return table


def _memory():
    return pd.DataFrame(
        {
            "elapsed_s": [0.0, 3600.0, 7200.0],
            "own_rss_bytes": [2e9, 3e9, 3e9],
            "children_rss_bytes": [0.0, 50e9, 60e9],
            "n_children": [0, 24, 24],
        }
    )


def test_draws_three_panels_and_saves(tmp_path):
    fig = figure_heca_runtime(_components(), _memory(), out_dir=tmp_path)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    assert (tmp_path / "heca_runtime.png").is_file()


def test_empty_components_raise():
    with pytest.raises(ValueError, match="empty"):
        figure_heca_runtime(_components().iloc[:0], _memory(), save=False)
