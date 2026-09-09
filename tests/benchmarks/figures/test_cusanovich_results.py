import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from benchmarks.figures import figure_cusanovich_results
from benchmarks.figures._cusanovich_results import AXIS_LABELS, MARKER_SIZE


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

    class _Carve:
        # composite_figure's panel D draws through _panels.carve_lines,
        # which reads estimator_results_/_select_row/get_k -- not a
        # plot_metric_over_n_clusters method. Mirrors the _StubCarve in
        # tests/benchmarks/figures/test_case_study.py.
        def __init__(self):
            ks = [4, 5, 6]
            self.estimator_results_ = pd.DataFrame(
                {
                    "n_clusters": ks,
                    "method_id": ["m0"] * len(ks),
                    "method_label": ["KMeans"] * len(ks),
                    "ari_stability": [0.1, 0.2, 0.3],
                    "ari_generalizability": [0.15, 0.25, 0.35],
                }
            )

        def _select_row(self, *, measure, rule="1se", not_two=False):
            row = self.estimator_results_.iloc[1]
            return row, 0, int(row["n_clusters"]), False

        def get_k(self, *, measure="stability", rule="1se", not_two=False):
            return 5

    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=Z,
        carve=_Carve(),
        carve_labels=rng.integers(0, 3, n),
        comparison_labels=rng.integers(0, 3, n),
        comparison_name="Silhouette",
        comparison_k=4,
        curves_df=curves,
        best_df=best,
    )


def test_axis_labels_name_the_lsi_components():
    # The embedding is LSI, not PCA; mislabeling it would misreport the
    # preprocessing the manuscript describes.
    assert AXIS_LABELS == ("LSI 1", "LSI 2")


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
