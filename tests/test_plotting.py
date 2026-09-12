"""Tests for carve._plotting module."""

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from sklearn.cluster import KMeans

from carve import CARVE
from carve._plotting import (
    _build_estimator_label,
    _prepare_cluster_score_groups,
    plot_cluster_boxplot,
    plot_cluster_scatter,
    plot_cluster_violin,
    plot_consensus_matrix,
    plot_metric_over_n_clusters,
)

from conftest import with_sweep_cols


@pytest.fixture()
def metric_results_df():
    """Minimal results DataFrame for metric-over-k plots.

    Includes a 'linkage' column so that the groupby in
    plot_metric_over_n_clusters has at least one group key.
    """
    return with_sweep_cols(
        pd.DataFrame(
            {
                "estimator": ["KMeans"] * 3,
                "n_clusters": [2, 3, 4],
                "linkage": ["ward"] * 3,
                "ari_stability": [0.9, 0.85, 0.7],
                "ari_stability_se": [0.02, 0.03, 0.05],
                "ari_stability_upper": [0.92, 0.88, 0.75],
                "ari_stability_lower": [0.88, 0.82, 0.65],
                "ari_generalizability": [0.85, 0.80, 0.65],
                "ari_generalizability_se": [0.03, 0.04, 0.06],
                "ari_generalizability_upper": [0.88, 0.84, 0.71],
                "ari_generalizability_lower": [0.82, 0.76, 0.59],
            }
        ),
        param="n_clusters",
        method_label="KMeans, linkage=ward",
        observed=[2.0, 3.0, 4.0],
    )


def _two_method_df():
    """Two methods x three sweep values, one with an all-NaN param column.

    Mirrors a custom grid mixing KMeans (no ``affinity``) with spectral
    clustering, which is where the old substring-heuristic grouping used
    to fragment curves.
    """
    df = pd.DataFrame(
        {
            "estimator": ["KMeans"] * 3 + ["SpectralClustering"] * 3,
            "n_clusters": [2, 3, 4, 2, 3, 4],
            "affinity": [np.nan] * 3 + ["self_tuning"] * 3,
            "ari_stability": [0.9, 0.85, 0.7, 0.88, 0.82, 0.68],
            "ari_stability_se": [0.02, 0.03, 0.05, 0.02, 0.03, 0.05],
            "ari_stability_upper": [0.92, 0.88, 0.75, 0.90, 0.85, 0.73],
            "ari_stability_lower": [0.88, 0.82, 0.65, 0.86, 0.79, 0.63],
            "ari_generalizability": [0.85, 0.80, 0.65, 0.83, 0.78, 0.63],
            "ari_generalizability_se": [0.03, 0.04, 0.06, 0.03, 0.04, 0.06],
            "ari_generalizability_upper": [0.88, 0.84, 0.71, 0.86, 0.82, 0.69],
            "ari_generalizability_lower": [0.82, 0.76, 0.59, 0.80, 0.74, 0.57],
        }
    )
    df = with_sweep_cols(
        df,
        param="n_clusters",
        method_label="",
        observed=[2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
    )
    # Two curves: the runner assigns one method_id per (estimator, params
    # except the swept one).
    df["method_id"] = ["m0"] * 3 + ["m1"] * 3
    df["method_label"] = ["KMeans"] * 3 + [
        "SpectralClustering, affinity=self_tuning"
    ] * 3
    return df


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# -----------------------------------------------------------------------
# _build_estimator_label
# -----------------------------------------------------------------------


class TestBuildEstimatorLabel:
    """The label is now produced by the runner and carried on the row.

    The formatting rules themselves live in ``_sweep.format_method_label``
    and are tested in ``test_sweep.py``.
    """

    def test_returns_method_label_verbatim(self):
        row = pd.Series(
            {
                "estimator": "KMeans",
                "method_label": "KMeans, linkage=ward",
                "n_clusters": 3,
                "ari_stability": 0.9,
            }
        )
        assert _build_estimator_label(row) == "KMeans, linkage=ward"

    def test_ignores_other_columns(self):
        """Metric and sweep columns never leak into the label."""
        row = pd.Series(
            {
                "estimator": "KMeans",
                "method_label": "KMeans",
                "ari_stability": 0.9,
                "sweep_value": 3,
                "gamma": 0.123456,
            }
        )
        label = _build_estimator_label(row)
        assert label == "KMeans"
        assert "ari_stability" not in label
        assert "gamma" not in label

    def test_tight_layout_wraps_on_commas(self):
        row = pd.Series({"method_label": "AgglomerativeClustering, linkage=ward"})
        label = _build_estimator_label(row, tight_layout=True)
        assert label == "AgglomerativeClustering\nlinkage=ward"

    def test_tight_layout_single_part_unchanged(self):
        row = pd.Series({"method_label": "KMeans"})
        assert _build_estimator_label(row, tight_layout=True) == "KMeans"


# -----------------------------------------------------------------------
# _prepare_cluster_score_groups
# -----------------------------------------------------------------------


class TestPrepareClusterScoreGroups:
    def test_basic(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        groups, order = _prepare_cluster_score_groups(scores, labels)
        assert len(groups) == 2
        assert order == [1, 2]
        np.testing.assert_array_equal(groups[0], [0.9, 0.8])
        np.testing.assert_array_equal(groups[1], [0.7, 0.6])

    def test_custom_order(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        groups, order = _prepare_cluster_score_groups(
            scores,
            labels,
            order=[1, 0],
        )
        assert order == [2, 1]
        np.testing.assert_array_equal(groups[0], [0.7, 0.6])
        np.testing.assert_array_equal(groups[1], [0.9, 0.8])

    def test_nan_filtered(self):
        scores = np.array([0.9, np.nan, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        groups, order = _prepare_cluster_score_groups(scores, labels)
        assert len(groups) == 2
        assert len(groups[0]) == 1  # NaN removed

    def test_all_nan_raises(self):
        scores = np.array([np.nan, np.nan])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="No finite scores"):
            _prepare_cluster_score_groups(scores, labels)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="matching length"):
            _prepare_cluster_score_groups(np.array([0.5]), np.array([0, 1]))

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1D array"):
            _prepare_cluster_score_groups(
                np.array([[0.5]]),
                np.array([0]),
            )


# -----------------------------------------------------------------------
# plot_metric_over_n_clusters
# -----------------------------------------------------------------------


class TestPlotMetricOverNClusters:
    def test_basic(self, metric_results_df):
        ax = plot_metric_over_n_clusters(metric_results_df, measure="stability")
        assert ax is not None

    def test_returns_axes(self, metric_results_df):
        ax = plot_metric_over_n_clusters(metric_results_df, measure="stability")
        assert isinstance(ax, plt.Axes)

    def test_save(self, metric_results_df, tmp_path):
        path = tmp_path / "metric_plot.png"
        result = plot_metric_over_n_clusters(
            metric_results_df,
            measure="stability",
            save=str(path),
        )
        assert result is None
        assert path.exists()

    def test_invalid_measure(self, metric_results_df):
        with pytest.raises(ValueError, match="not found"):
            plot_metric_over_n_clusters(metric_results_df, measure="nonexistent")

    def test_empty_df(self):
        df = pd.DataFrame()
        with pytest.raises(RuntimeError, match="empty"):
            plot_metric_over_n_clusters(df, measure="stability")

    def test_custom_figsize(self, metric_results_df):
        ax = plot_metric_over_n_clusters(
            metric_results_df,
            measure="stability",
            figsize=(12, 8),
        )
        fig = ax.get_figure()
        assert fig.get_size_inches()[0] == pytest.approx(12)

    def test_with_existing_ax(self, metric_results_df):
        fig, ax = plt.subplots()
        returned = plot_metric_over_n_clusters(
            metric_results_df,
            measure="stability",
            ax=ax,
        )
        assert returned is ax

    def test_no_legend(self, metric_results_df):
        ax = plot_metric_over_n_clusters(
            metric_results_df,
            measure="stability",
            legend=False,
        )
        assert ax is not None

    def test_mixed_estimators_single_nan_group_col(self):
        """Regression: 2 estimators, one with an all-NaN param column.

        Grouping is on ``method_id``, so a column that is NaN for one
        estimator can no longer fragment or drop a curve.
        """
        df = _two_method_df()
        ax = plot_metric_over_n_clusters(df, measure="stability")
        lines = [
            c
            for c in ax.get_children()
            if hasattr(c, "get_linestyle")
            and hasattr(c, "get_xydata")
            and len(c.get_xydata()) > 1
        ]
        assert len(lines) >= 2

    def test_one_line_per_method_not_per_row(self):
        """Two methods x three sweep values draws two curves, not six."""
        df = _two_method_df()
        ax = plot_metric_over_n_clusters(df, measure="stability")
        # Errorbar caps share the data lines' point count, so identify the
        # curves themselves by their solid linestyle.
        curves = [
            line
            for line in ax.get_lines()
            if line.get_linestyle() == "-" and len(line.get_xydata()) == 3
        ]
        assert len(curves) == 2

    def test_legend_labels_are_method_labels(self):
        df = _two_method_df()
        ax = plot_metric_over_n_clusters(df, measure="stability")
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert "KMeans" in texts
        assert "SpectralClustering, affinity=self_tuning" in texts

    def test_xlabel_follows_sweep_param(self, metric_results_df):
        ax = plot_metric_over_n_clusters(metric_results_df, measure="stability")
        assert ax.get_xlabel() == "Number of Clusters (k)"

    def test_xlabel_resolution_mode(self, resolution_results_df):
        ax = plot_metric_over_n_clusters(resolution_results_df, measure="stability")
        assert ax.get_xlabel() == "Resolution"

    def test_xticks_are_sweep_values(self, resolution_results_df):
        ax = plot_metric_over_n_clusters(resolution_results_df, measure="stability")
        np.testing.assert_allclose(ax.get_xticks(), [0.25, 0.5, 1.0, 2.0])

    def test_custom_labels(self, metric_results_df):
        ax = plot_metric_over_n_clusters(
            metric_results_df,
            measure="stability",
            title="My Title",
            xlabel="k",
            ylabel="Score",
        )
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "k"
        assert ax.get_ylabel() == "Score"


# -----------------------------------------------------------------------
# plot_consensus_matrix
# -----------------------------------------------------------------------


class TestPlotConsensusMatrix:
    def test_basic(self):
        M = np.zeros((6, 6))
        M[:3, :3] = 1.0
        M[3:, 3:] = 1.0
        labels = np.array([0, 0, 0, 1, 1, 1])
        ax = plot_consensus_matrix(M, labels)
        assert ax is not None

    def test_save(self, tmp_path):
        M = np.eye(4)
        labels = np.array([0, 0, 1, 1])
        path = tmp_path / "consensus.png"
        result = plot_consensus_matrix(M, labels, save=str(path))
        assert result is None
        assert path.exists()

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            plot_consensus_matrix(np.zeros((3, 4)), np.array([0, 0, 0]))

    def test_mismatched_labels_raises(self):
        with pytest.raises(ValueError, match="matching"):
            plot_consensus_matrix(np.eye(3), np.array([0, 1]))

    def test_labels_not_1d_raises(self):
        with pytest.raises(ValueError, match="1D"):
            plot_consensus_matrix(np.eye(3), np.array([[0, 1, 2]]))

    def test_nan_handling(self):
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        labels = np.array([0, 1])
        ax = plot_consensus_matrix(M, labels)
        assert ax is not None


# -----------------------------------------------------------------------
# plot_cluster_boxplot
# -----------------------------------------------------------------------


class TestPlotClusterBoxplot:
    def test_basic(self):
        scores = np.array([0.9, 0.85, 0.7, 0.65, 0.5, 0.45])
        labels = np.array([0, 0, 0, 1, 1, 1])
        ax = plot_cluster_boxplot(scores, labels)
        assert ax is not None

    def test_save(self, tmp_path):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        path = tmp_path / "boxplot.png"
        result = plot_cluster_boxplot(scores, labels, save=str(path))
        assert result is None
        assert path.exists()

    def test_custom_order(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        ax = plot_cluster_boxplot(scores, labels, order=[1, 0])
        assert ax is not None


# -----------------------------------------------------------------------
# plot_cluster_violin
# -----------------------------------------------------------------------


class TestPlotClusterViolin:
    def test_basic(self):
        scores = np.array([0.9, 0.85, 0.7, 0.65, 0.5, 0.45])
        labels = np.array([0, 0, 0, 1, 1, 1])
        ax = plot_cluster_violin(scores, labels)
        assert ax is not None

    def test_save(self, tmp_path):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        path = tmp_path / "violin.png"
        result = plot_cluster_violin(scores, labels, save=str(path))
        assert result is None
        assert path.exists()

    def test_no_stripplot(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        labels = np.array([0, 0, 1, 1])
        ax = plot_cluster_violin(scores, labels, stripplot=False)
        assert ax is not None

    def test_inner_quartile(self):
        scores = np.array([0.9, 0.85, 0.7, 0.65, 0.5, 0.45])
        labels = np.array([0, 0, 0, 1, 1, 1])
        ax = plot_cluster_violin(scores, labels, inner="quartile")
        assert ax is not None

    def test_inner_none(self):
        scores = np.array([0.9, 0.85, 0.7, 0.65, 0.5, 0.45])
        labels = np.array([0, 0, 0, 1, 1, 1])
        ax = plot_cluster_violin(scores, labels, inner="none")
        assert ax is not None

    def test_violin_bodies_clipped_to_ylim(self):
        """Violin polygon vertices must not exceed ylim boundaries."""
        from matplotlib.collections import PolyCollection

        rng = np.random.RandomState(42)
        scores = np.concatenate(
            [
                rng.uniform(0.90, 0.99, size=50),
                rng.uniform(0.85, 0.95, size=50),
            ]
        )
        labels = np.array([0] * 50 + [1] * 50)
        ax = plot_cluster_violin(scores, labels, ylim=(0.0, 1.0))

        bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(bodies) > 0, "Expected violin body PolyCollections"

        for body in bodies:
            for path in body.get_paths():
                y_vals = path.vertices[:, 1]
                assert y_vals.min() >= 0.0 - 1e-9
                assert y_vals.max() <= 1.0 + 1e-9

    def test_violin_no_clip_when_ylim_none(self):
        """When ylim is None, violin bodies should not be clipped."""
        scores = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 0.60, 0.61, 0.62])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        ax = plot_cluster_violin(scores, labels, ylim=None)
        assert ax is not None


# -----------------------------------------------------------------------
# plot_cluster_scatter
# -----------------------------------------------------------------------


class TestPlotClusterScatter:
    def test_basic(self):
        X = np.random.RandomState(0).randn(20, 3)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores)
        assert ax is not None

    def test_save(self, tmp_path):
        X = np.random.RandomState(0).randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        path = tmp_path / "scatter.png"
        result = plot_cluster_scatter(X, labels, scores, save=str(path))
        assert result is None
        assert path.exists()

    def test_with_embedding(self):
        X = np.random.RandomState(0).randn(20, 5)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        embedding = np.random.RandomState(0).randn(20, 2)
        ax = plot_cluster_scatter(X, labels, scores, embedding=embedding)
        assert ax is not None

    def test_1d_data(self):
        X = np.random.RandomState(0).randn(20, 1)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores)
        assert ax is not None

    def test_2d_data_no_pca(self):
        X = np.random.RandomState(0).randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores)
        assert ax is not None

    def test_high_dim_uses_pca(self):
        X = np.random.RandomState(0).randn(20, 10)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores)
        assert ax is not None

    def test_alpha_range(self):
        X = np.random.RandomState(0).randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(
            X,
            labels,
            scores,
            alpha_range=(0.3, 0.9),
        )
        assert ax is not None

    def test_box_annotation_preserves_right_margin_legend(self):
        X = np.random.RandomState(0).randn(30, 4)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
        scores = np.random.RandomState(0).rand(30)

        ax = plot_cluster_scatter(
            X,
            labels,
            scores,
            legend=True,
            legend_loc="right margin",
            annotation="Model\n1-SE rule",
            annotation_style="box",
        )

        main_legend = ax.get_legend()
        assert main_legend is not None
        assert "Cluster" in main_legend.get_title().get_text()

        annotation_legends = [
            artist for artist in ax.artists if isinstance(artist, Legend)
        ]
        assert len(annotation_legends) == 1
        annotation_texts = [t.get_text() for t in annotation_legends[0].get_texts()]
        assert any("1-SE rule" in text for text in annotation_texts)

    def test_box_annotation_save_with_right_margin_legend(self, tmp_path):
        X = np.random.RandomState(0).randn(24, 3)
        labels = np.array([0] * 12 + [1] * 12)
        scores = np.random.RandomState(0).rand(24)
        path = tmp_path / "scatter_box_annotation.png"

        result = plot_cluster_scatter(
            X,
            labels,
            scores,
            legend=True,
            legend_loc="right margin",
            annotation="Selected model",
            annotation_style="box",
            save=str(path),
        )

        assert result is None
        assert path.exists()

    def test_mismatched_dims_raises(self):
        X = np.random.RandomState(0).randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(10)  # wrong size
        with pytest.raises(ValueError, match="matching"):
            plot_cluster_scatter(X, labels, scores)

    def test_no_finite_scores_raises(self):
        X = np.random.RandomState(0).randn(5, 2)
        labels = np.array([0, 0, 1, 1, 1])
        scores = np.full(5, np.nan)
        with pytest.raises(ValueError, match="No finite scores"):
            plot_cluster_scatter(X, labels, scores)


def test_plot_consensus_matrix_under_anchoring():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(6, 1, (30, 4))])
    c = CARVE(
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [10]})],
        n_resamples=6,
        random_state=0,
        anchor_threshold=30,
    )
    with pytest.warns(RuntimeWarning, match="anchored consensus"):
        c.fit(X)

    ax = c.plot_consensus_matrix(k=2)
    assert ax is not None
    # The rendered image is the anchor block, not the full matrix.
    assert c.consensus_matrices_[0].shape == (30, 30)


def test_sample_level_plots_work_under_anchoring():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(6, 1, (30, 4))])
    c = CARVE(
        estimator_param_grids=[(KMeans, {"n_clusters": [2, 3], "n_init": [10]})],
        n_resamples=6,
        random_state=0,
        anchor_threshold=30,
    )
    with pytest.warns(RuntimeWarning, match="anchored consensus"):
        c.fit(X)

    # The property the whole feature rests on: consensus matrices shrink to
    # the anchor block, but every per-sample array stays length n.
    assert c.stability_gini_scores_.shape[1] == X.shape[0]
    assert c.stability_ce_scores_.shape[1] == X.shape[0]
    assert c.generalizability_scores_[0].shape == (X.shape[0],)

    # Each of these reads a per-sample score array and calls get_labels;
    # a length mismatch between the two would raise here.
    assert c.plot_cluster_boxplot(k=2) is not None
    assert c.plot_cluster_violin(k=2) is not None
    assert c.plot_cluster_scatter(X=X, k=2) is not None
    assert c.plot_diagnostic_scatter(X=X, k=2) is not None
