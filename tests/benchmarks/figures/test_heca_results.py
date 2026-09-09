import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from benchmarks.figures import figure_heca_results
from benchmarks.figures._heca_results import AXIS_LABELS, subsample_inputs


@pytest.fixture
def inputs():
    from benchmarks.figures._case_study import CompositeInputs

    rng = np.random.default_rng(0)
    n = 600
    y = rng.choice(["Lung", "Brain", "Kidney"], n)
    curves = pd.DataFrame(
        {
            "metric": np.repeat(["silhouette", "gap"], 5),
            "model": "MiniBatchKMeans",
            "k": list(range(3, 8)) * 2,
            "score": rng.random(10),
            "ari": rng.random(10),
        }
    )
    best = pd.DataFrame(
        {
            "metric": ["silhouette", "gap"],
            "model": ["MiniBatchKMeans"] * 2,
            "k": [3, 5],
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
            ks = [3, 4, 5]
            self.estimator_results_ = pd.DataFrame(
                {
                    "n_clusters": ks,
                    "method_id": ["m0"] * len(ks),
                    "method_label": ["MiniBatchKMeans"] * len(ks),
                    "ari_stability": [0.1, 0.2, 0.3],
                    "ari_generalizability": [0.15, 0.25, 0.35],
                }
            )

        def _select_row(self, *, measure, rule="1se", not_two=False):
            row = self.estimator_results_.iloc[1]
            return row, 0, int(row["n_clusters"]), False

        def get_k(self, *, measure="stability", rule="1se", not_two=False):
            return 4

    return CompositeInputs(
        X=rng.normal(size=(n, 5)),
        y=y,
        Z=rng.normal(size=(n, 2)),
        carve=_Carve(),
        carve_labels=rng.integers(0, 3, n),
        comparison_labels=rng.integers(0, 3, n),
        comparison_name="Silhouette",
        comparison_k=3,
        curves_df=curves,
        best_df=best,
    )


class TestSubsampleInputs:
    def test_reduces_every_per_cell_array_together(self, inputs):
        out = subsample_inputs(inputs, size=100)
        assert out.Z.shape[0] == 100
        assert out.y.shape[0] == 100
        assert out.carve_labels.shape[0] == 100
        assert out.comparison_labels.shape[0] == 100
        assert out.X.shape[0] == 100

    def test_rows_stay_aligned_across_arrays(self, inputs):
        # Independently subsampling each array would silently decouple a
        # cell's label from its position, which no shape assertion catches.
        #
        # y and carve_labels alone are too coarse a fingerprint to catch
        # that: with only 3 tissues and 3 cluster ids, all 9 combinations
        # already occur among the fixture's 600 rows, so a decoupled (y,
        # carve_labels) pair still lands on some real row's combination by
        # chance -- confirmed by mutating subsample_inputs to draw each
        # array's index independently: that broken version still passed a
        # membership check on (y, carve_labels) alone. X's continuous,
        # effectively-unique rows are folded into the fingerprint so a
        # decoupled row cannot coincidentally match an unrelated one.
        out = subsample_inputs(inputs, size=50, random_state=1)
        again = subsample_inputs(inputs, size=50, random_state=1)
        assert np.array_equal(out.y, again.y)
        assert np.array_equal(out.carve_labels, again.carve_labels)

        def _fingerprint(y, label, x_row):
            return (str(y), int(label), tuple(np.round(x_row, 8)))

        rows = {
            _fingerprint(a, b, x)
            for a, b, x in zip(inputs.y, inputs.carve_labels, inputs.X)
        }
        for a, b, x in zip(out.y, out.carve_labels, out.X):
            assert _fingerprint(a, b, x) in rows

    def test_a_size_at_or_above_n_is_a_passthrough(self, inputs):
        out = subsample_inputs(inputs, size=10_000)
        assert out.y.shape[0] == inputs.y.shape[0]

    def test_non_per_cell_fields_are_preserved(self, inputs):
        out = subsample_inputs(inputs, size=100)
        assert out.comparison_k == inputs.comparison_k
        assert out.comparison_name == inputs.comparison_name
        assert out.curves_df.equals(inputs.curves_df)


def test_axis_labels_name_the_pca_components():
    # hECA follows the source's log-normalization and PCA chain, not the
    # TF-IDF and LSI chain the Cusanovich study uses.
    assert AXIS_LABELS == ("PC1", "PC2")


def test_figure_writes_under_its_manuscript_name(inputs, tmp_path):
    figure_heca_results(inputs, scatter_subsample=200, save=True, out_dir=tmp_path)
    assert (tmp_path / "heca_results.png").is_file()


def test_figure_does_not_write_when_save_is_false(inputs, tmp_path):
    fig = figure_heca_results(inputs, scatter_subsample=200, save=False,
                              out_dir=tmp_path)
    assert len([ax for ax in fig.axes if ax.has_data() or ax.get_title()]) >= 6
    assert list(tmp_path.iterdir()) == []
