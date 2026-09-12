import pandas as pd
import pytest

from benchmarks.figures import figure_study_scaling


@pytest.fixture
def sweep():
    return pd.DataFrame(
        {
            "n": [1000, 5000, 25_000],
            "n_configs": [26, 26, 26],
            "wall_clock_s": [12.0, 61.0, 320.0],
            "peak_rss_bytes": [int(1e9), int(3e9), int(9e9)],
            "selected_k": [5, 5, 6],
            "ari": [0.7, 0.72, 0.71],
        }
    )


def test_draws_three_panels(sweep, tmp_path):
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    drawn = [ax for ax in fig.axes if ax.has_data()]
    assert len(drawn) == 3


def test_panels_are_labeled_for_their_quantities(sweep, tmp_path):
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    labels = " ".join(ax.get_ylabel() for ax in fig.axes).lower()
    assert "second" in labels
    assert "gb" in labels or "memory" in labels
    # Not "k" in labels: "peak" (the memory panel's own label) contains a
    # "k", so that check could never fail.
    assert "selected k" in labels


def test_memory_is_plotted_in_gigabytes_not_bytes(sweep, tmp_path):
    # Plotting raw bytes would put 9e9 on the axis and make the panel
    # unreadable, which no shape assertion would catch.
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    memory_ax = [ax for ax in fig.axes if "gb" in ax.get_ylabel().lower()][0]
    ydata = memory_ax.lines[0].get_ydata()
    assert max(ydata) == pytest.approx(9.0, rel=0.01)


def test_x_axis_is_the_sample_count(sweep, tmp_path):
    fig = figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    for ax in fig.axes:
        if ax.has_data():
            assert list(ax.lines[0].get_xdata()) == [1000, 5000, 25_000]


def test_writes_under_the_requested_name(sweep, tmp_path):
    figure_study_scaling(sweep, save=True, out_dir=tmp_path)
    assert (tmp_path / "heca_scaling.png").is_file()


def test_does_not_write_when_save_is_false(sweep, tmp_path):
    figure_study_scaling(sweep, save=False, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_empty_sweep_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one row"):
        figure_study_scaling(pd.DataFrame(columns=["n"]), save=False, out_dir=tmp_path)
