"""Tests for carve._plotting module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from carve import CARVE
import carve._plotting as carve_plotting
from carve._plotting import (
    _build_estimator_label,
    _prepare_cluster_score_groups,
    plot_cluster_boxplot,
    plot_cluster_scatter,
    plot_cluster_violin,
    plot_consensus_matrix,
    plot_diagnostic_scatter,
    plot_metric_by_pipeline,
    plot_metric_over_n_clusters,
)
from tests._helpers import with_sweep_cols


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
        assert ax.get_legend() is None
        assert (
            plot_metric_over_n_clusters(
                metric_results_df, measure="stability"
            ).get_legend()
            is not None
        )

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

    def test_marker_label_and_default_ylabel(self, resolution_results_df):
        # Pins the text _draw_metric_lines built itself before its callers
        # passed y_col, ylabel and selection_label in.
        ax = plot_metric_over_n_clusters(
            resolution_results_df, measure="stability", rule="max"
        )
        (dashed,) = _dashed(ax)
        assert dashed.get_label() == "Selected resolution (Max rule): 0.25"
        assert ax.get_ylabel() == "ARI Stability"


# -----------------------------------------------------------------------
# plot_metric_by_pipeline
# -----------------------------------------------------------------------


def _pipeline_results_df():
    """A preprocessing_results_ table over two configurations.

    m0 has two pipelines whose values peak at k=3. m1 has a third pipeline,
    a decoy peaking at k=4 above anything in m0, so a plot or a selection
    that reads the whole table instead of m0's rows is visibly wrong.
    """
    curves = {
        ("m0", "identity | identity"): [0.70, 0.90, 0.60],
        ("m0", "identity | PCA(n_components=2)"): [0.65, 0.80, 0.50],
        ("m1", "StandardScaler | identity"): [0.20, 0.30, 0.99],
    }
    rows = []
    for (method_id, pipeline), values in curves.items():
        normalization, dim_reduction = pipeline.split(" | ")
        for rank, (k, value) in enumerate(zip((2, 3, 4), values)):
            rows.append(
                {
                    "method_id": method_id,
                    "method_label": f"KMeans {method_id}",
                    "pipeline": pipeline,
                    "normalization": normalization,
                    "dim_reduction": dim_reduction,
                    "n_clusters": k,
                    "n_resamples": 5,
                    "ari_stability": value,
                    "ari_stability_se": 0.01,
                    "ari_generalizability": value - 0.1,
                    "ari_generalizability_se": 0.02,
                    "n_clusters_observed": float(k),
                    "sweep_param": "n_clusters",
                    "sweep_value": k,
                    "sweep_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def _estimator_results_df():
    """The estimator_results_ table behind _pipeline_results_df, pooled.

    It disagrees with the per-pipeline rows on purpose. Under "1se" the best
    row, m1 at k=2 with SE 0.10, admits m0 at k=4, so CARVE selects (m0, 4);
    the rule over m0's rows alone, pooled or per pipeline, stops at k=3. For
    m1 the rule over its own pooled rows stops at k=2, where its per-pipeline
    rows peak at k=4.
    """
    curves = {
        "m0": ([0.80, 0.90, 0.86], [0.01, 0.01, 0.01]),
        "m1": ([0.95, 0.40, 0.30], [0.10, 0.01, 0.01]),
    }
    rows = []
    for method_id, (values, ses) in curves.items():
        for rank, (k, value, se) in enumerate(zip((2, 3, 4), values, ses)):
            rows.append(
                {
                    "method_id": method_id,
                    "method_label": f"KMeans {method_id}",
                    "n_clusters": k,
                    "ari_stability": value,
                    "ari_stability_se": se,
                    "ari_generalizability": value - 0.1,
                    "ari_generalizability_se": se,
                    "n_clusters_observed": float(k),
                    "sweep_param": "n_clusters",
                    "sweep_value": k,
                    "sweep_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def _curves(ax):
    """Each errorbar's data line, keyed by its legend label."""
    return {container.get_label(): container[0] for container in ax.containers}


def _dashed(ax):
    """The dashed selection lines on ax."""
    return [line for line in ax.get_lines() if line.get_linestyle() == "--"]


class TestPlotMetricByPipeline:
    def test_one_line_per_pipeline_of_the_configuration(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(), estimator_df=_estimator_results_df(), method_id="m0"
        )
        assert set(_curves(ax)) == {
            "identity | identity",
            "identity | PCA(n_components=2)",
        }

    def test_lines_carry_the_rows_values(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(),
            estimator_df=_estimator_results_df(),
            method_id="m0",
            measure="generalizability",
        )
        curves = _curves(ax)
        np.testing.assert_allclose(
            curves["identity | identity"].get_ydata(), [0.60, 0.80, 0.50]
        )
        np.testing.assert_allclose(
            curves["identity | PCA(n_components=2)"].get_ydata(), [0.55, 0.70, 0.40]
        )
        np.testing.assert_allclose(curves["identity | identity"].get_xdata(), [2, 3, 4])

    def test_marks_the_sweep_value_carve_selected(self):
        # Under "1se" CARVE selects (m0, k=4) from the pooled table. The rule
        # over m0's per-pipeline rows, or over m0's pooled rows alone, stops
        # at k=3.
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(),
            estimator_df=_estimator_results_df(),
            method_id="m0",
            rule="1se",
        )
        (dashed,) = _dashed(ax)
        assert list(dashed.get_xdata()) == [4.0, 4.0]
        assert dashed.get_label() == "Selected k (1-SE rule): 4"

    def test_unselected_configuration_marks_the_rule_over_its_pooled_rows(self):
        # CARVE selects m0 under "1se". For m1 the rule runs over m1's pooled
        # rows and stops at k=2; m1's per-pipeline rows peak at k=4.
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(),
            estimator_df=_estimator_results_df(),
            method_id="m1",
            rule="1se",
        )
        (dashed,) = _dashed(ax)
        assert list(dashed.get_xdata()) == [2.0, 2.0]

    def test_legend_and_axis_labels(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(), estimator_df=_estimator_results_df(), method_id="m0"
        )
        assert ax.get_legend().get_title().get_text() == "Pipelines"
        assert ax.get_xlabel() == "Number of Clusters (k)"
        assert ax.get_ylabel() == "ARI Stability"

    def test_resolution_axis(self):
        def as_resolution(df):
            df = df.rename(columns={"n_clusters": "resolution"})
            df["sweep_param"] = "resolution"
            df["sweep_value"] = df["sweep_value"] / 4
            return df

        ax = plot_metric_by_pipeline(
            as_resolution(_pipeline_results_df()),
            estimator_df=as_resolution(_estimator_results_df()),
            method_id="m0",
        )
        assert ax.get_xlabel() == "Resolution"
        np.testing.assert_allclose(ax.get_xticks(), [0.5, 0.75, 1.0])
        (dashed,) = _dashed(ax)
        assert list(dashed.get_xdata()) == [1.0, 1.0]

    def test_colors_come_from_the_palette(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(),
            estimator_df=_estimator_results_df(),
            method_id="m0",
            palette="viridis",
        )
        drawn = [tuple(c[0].get_color()) for c in ax.containers]
        expected = [tuple(rgba) for rgba in plt.get_cmap("viridis")(np.linspace(0, 1, 2))]
        assert drawn == expected

    def test_kwargs_reach_the_errorbar(self):
        ax = plot_metric_by_pipeline(
            _pipeline_results_df(),
            estimator_df=_estimator_results_df(),
            method_id="m0",
            linestyle=":",
        )
        assert [c[0].get_linestyle() for c in ax.containers] == [":", ":"]

    def test_save_writes_the_file_and_returns_none(self, tmp_path):
        path = tmp_path / "pipelines.png"
        result = plot_metric_by_pipeline(
            _pipeline_results_df(),
            estimator_df=_estimator_results_df(),
            method_id="m0",
            save=path,
        )
        assert result is None
        assert path.exists()

    def test_none_table_names_the_cause(self):
        with pytest.raises(RuntimeError, match="the fit was not randomized"):
            plot_metric_by_pipeline(
                None, estimator_df=_estimator_results_df(), method_id="m0"
            )

    def test_empty_table(self):
        with pytest.raises(RuntimeError, match="empty"):
            plot_metric_by_pipeline(
                _pipeline_results_df().iloc[0:0],
                estimator_df=_estimator_results_df(),
                method_id="m0",
            )

    def test_unknown_method_id(self):
        with pytest.raises(ValueError, match=r"method_id 'm9' not found.*\['m0', 'm1'\]"):
            plot_metric_by_pipeline(
                _pipeline_results_df(), estimator_df=_estimator_results_df(), method_id="m9"
            )

    def test_measure_the_table_does_not_carry(self):
        with pytest.raises(
            ValueError,
            match=r"must be 'stability' or 'generalizability'.*got 'pac'",
        ):
            plot_metric_by_pipeline(
                _pipeline_results_df(),
                estimator_df=_estimator_results_df(),
                method_id="m0",
                measure="pac",
            )

    def test_unsupported_measure_names_the_two_ari_criteria(self):
        with pytest.raises(
            ValueError,
            match=(
                r"The per-pipeline table carries only the ARI criteria, so "
                r"measure must be 'stability' or 'generalizability' "
                r"\(or an alias of either\); got 'average'\."
            ),
        ):
            plot_metric_by_pipeline(
                _pipeline_results_df(),
                estimator_df=_estimator_results_df(),
                method_id="m0",
                measure="average",
            )


class TestSharedMetricDrawing:
    def test_both_metric_plots_draw_through_one_helper(
        self, monkeypatch, metric_results_df
    ):
        calls = []

        def spy(df, **kwargs):
            calls.append(
                (kwargs["group_col"], kwargs["legend_title"], kwargs["y_col"], len(df))
            )
            return "drawn"

        monkeypatch.setattr(carve_plotting, "_draw_metric_lines", spy)
        assert plot_metric_over_n_clusters(metric_results_df) == "drawn"
        assert (
            plot_metric_by_pipeline(
                _pipeline_results_df(),
                estimator_df=_estimator_results_df(),
                method_id="m0",
            )
            == "drawn"
        )
        assert calls == [
            ("method_id", "Estimators", "ari_stability", 3),
            ("pipeline", "Pipelines", "ari_stability", 6),
        ]


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
        # Never co-sampled pairs are drawn at 0.5, the value get_labels also
        # substitutes; the diagonal is forced to 1.
        M = np.array([[1.0, np.nan], [np.nan, 1.0]])
        labels = np.array([0, 1])
        ax = plot_consensus_matrix(M, labels)
        np.testing.assert_array_equal(
            np.asarray(ax.images[0].get_array()), [[1.0, 0.5], [0.5, 1.0]]
        )

    def test_with_existing_ax(self):
        fig, ax = plt.subplots()
        M = np.eye(3)
        labels = np.array([0, 1, 1])
        assert plot_consensus_matrix(M, labels, ax=ax) is ax
        assert len(ax.images) == 1


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
        # Tick labels are one-based cluster numbers, in the requested order.
        assert [t.get_text() for t in ax.get_xticklabels()] == ["2", "1"]


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
        """ylim=None with fit_ylim=False leaves the KDE bodies unclipped.

        matplotlib evaluates the KDE on [min, max] of each group, so the top
        vertex of the unclipped bodies is the data maximum; a ylim inside the
        data range clips it.
        """
        scores = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 0.60, 0.61, 0.62])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])

        def top(ax):
            bodies = [c for c in ax.collections if isinstance(c, PolyCollection)]
            return max(p.vertices[:, 1].max() for b in bodies for p in b.get_paths())

        clipped = plot_cluster_violin(scores, labels, ylim=(0.0, 0.9), fit_ylim=False)
        assert top(clipped) == pytest.approx(0.9)
        free = plot_cluster_violin(scores, labels, ylim=None, fit_ylim=False)
        assert top(free) == pytest.approx(0.99)


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
        ax = plot_cluster_scatter(X, labels, scores, sort_order=False)
        expected = PCA(n_components=2, random_state=0).fit_transform(X)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), expected)

    def test_alpha_range(self):
        X = np.random.RandomState(0).randn(20, 2)
        labels = np.array([0] * 10 + [1] * 10)
        scores = np.random.RandomState(0).rand(20)
        ax = plot_cluster_scatter(X, labels, scores, alpha_range=(0.3, 0.9))
        alphas = ax.collections[0].get_facecolors()[:, 3]
        assert alphas.min() == pytest.approx(0.3)
        assert alphas.max() == pytest.approx(0.9)

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


# -----------------------------------------------------------------------
# plot_diagnostic_scatter
# -----------------------------------------------------------------------


class TestPlotDiagnosticScatter:
    def _data(self, n=30, p=4, k=3):
        rng = np.random.RandomState(0)
        return rng.randn(n, p), np.repeat(np.arange(k), n // k), rng.rand(n)

    def test_one_collection_per_cluster(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores)
        assert len(ax.collections) == 3
        assert sorted(c.get_offsets().shape[0] for c in ax.collections) == [10, 10, 10]

    def test_markers_override_in_label_order(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores, markers=["o", "s", "^"])
        handles = ax.get_legend().legend_handles
        assert [h.get_marker() for h in handles] == ["o", "s", "^"]

    def test_more_clusters_than_markers_warns_and_cycles(self):
        X, labels, scores = self._data()
        with pytest.warns(UserWarning, match="Markers will cycle"):
            ax = plot_diagnostic_scatter(X, labels, scores, markers=["o", "s"])
        handles = ax.get_legend().legend_handles
        assert [h.get_marker() for h in handles] == ["o", "s", "o"]

    def test_colorbar_label_defaults_to_scores_name(self):
        X, labels, scores = self._data()
        fig, ax = plt.subplots()
        plot_diagnostic_scatter(X, labels, scores, ax=ax, scores_name="Foo")
        cbar_axes = [a for a in fig.axes if a is not ax]
        assert len(cbar_axes) == 1
        assert cbar_axes[0].get_xlabel() == "Foo"

    def test_colorbar_label_override(self):
        X, labels, scores = self._data()
        fig, ax = plt.subplots()
        plot_diagnostic_scatter(
            X, labels, scores, ax=ax, scores_name="Foo", colorbar_label="Bar"
        )
        assert [a for a in fig.axes if a is not ax][0].get_xlabel() == "Bar"

    def test_no_colorbar_adds_no_axes(self):
        X, labels, scores = self._data()
        fig, ax = plt.subplots()
        plot_diagnostic_scatter(X, labels, scores, ax=ax, colorbar=False)
        assert fig.axes == [ax]

    def test_sort_order_draws_high_scores_first(self):
        # alpha encodes the score: alpha_range=(alpha_high, alpha_low), so a
        # high score is transparent. Drawing high scores first means the
        # alphas within a cluster's collection are non-decreasing.
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores, sort_order=True)
        for c in ax.collections:
            assert np.all(np.diff(c.get_facecolors()[:, 3]) >= -1e-12)
        unsorted = plot_diagnostic_scatter(X, labels, scores, sort_order=False)
        assert any(
            np.any(np.diff(c.get_facecolors()[:, 3]) < 0) for c in unsorted.collections
        )

    def test_alpha_range_bounds_the_facecolors(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(X, labels, scores, alpha_range=(0.4, 0.9))
        alphas = np.vstack([c.get_facecolors() for c in ax.collections])[:, 3]
        assert alphas.min() == pytest.approx(0.4)
        assert alphas.max() == pytest.approx(0.9)

    def test_nan_scores_are_drawn_gray(self):
        X, labels, scores = self._data()
        scores = scores.copy()
        scores[0] = np.nan
        ax = plot_diagnostic_scatter(X, labels, scores)
        facecolors = np.vstack([c.get_facecolors() for c in ax.collections])
        assert np.any(np.all(np.isclose(facecolors, (0.5, 0.5, 0.5, 0.2)), axis=1))

    def test_legend_annotation_prefixes_the_title(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(
            X, labels, scores, annotation="note", annotation_style="legend"
        )
        assert ax.get_legend().get_title().get_text() == "note\nCluster"

    def test_box_annotation_keeps_the_cluster_legend(self):
        X, labels, scores = self._data()
        ax = plot_diagnostic_scatter(
            X, labels, scores, annotation="note", annotation_style="box"
        )
        legends = [c for c in ax.get_children() if isinstance(c, Legend)]
        assert any(
            [t.get_text() for t in legend.get_texts()] == ["note"] for legend in legends
        )
        assert ax.get_legend().get_title().get_text() == "Cluster"

    def test_save(self, tmp_path):
        X, labels, scores = self._data()
        path = tmp_path / "diagnostic.png"
        assert plot_diagnostic_scatter(X, labels, scores, save=path) is None
        assert path.exists()

    def test_no_finite_scores_raises(self):
        X, labels, _ = self._data()
        with pytest.raises(ValueError, match="No finite scores"):
            plot_diagnostic_scatter(X, labels, np.full(30, np.nan))

    def test_mismatched_lengths_raise(self):
        X, labels, scores = self._data()
        with pytest.raises(ValueError, match="matching n_samples"):
            plot_diagnostic_scatter(X, labels, scores[:-1])


@pytest.fixture(scope="module")
def anchored():
    """Two blobs fitted under anchoring (threshold 30 of 60 samples)."""
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
    return X, c


def test_plot_consensus_matrix_under_anchoring(anchored):
    _, c = anchored
    ax = c.plot_consensus_matrix(k=2)
    # The rendered image is the anchor block, not the full 60-by-60 matrix.
    assert np.asarray(ax.images[0].get_array()).shape == (30, 30)


def test_sample_level_plots_work_under_anchoring(anchored):
    X, c = anchored
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
