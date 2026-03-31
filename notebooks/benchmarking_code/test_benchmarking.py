"""Tests for benchmarking utility modules.

Run from the ``code/`` directory:
    python -m pytest notebooks/benchmarking_code/test_benchmarking.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the benchmarking_code directory is on sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── benchmarking_config ───────────────────────────────────────────────────────

from benchmarking_config import (  # noqa: E402
    SCALING_RANGES,
    GRANULARITY,
    make_scaling_x_values,
    CARVE_METRICS_ALL,
    CARVE_METRICS_STABILITY,
    CARVE_METRICS_GENERALIZABILITY,
    EXTERNAL_METRICS,
    METRIC_DISPLAY_NAMES,
    METRIC_COLOR,
    METRIC_LABEL,
    PLOT_METRICS,
)


class TestScalingConfig:
    def test_scaling_ranges_keys(self):
        assert set(SCALING_RANGES) == {"n_total", "p", "embed_dim"}

    def test_scaling_ranges_min_less_than_max(self):
        for name, r in SCALING_RANGES.items():
            assert r["min"] < r["max"], f"{name}: min >= max"

    def test_make_scaling_x_values_length(self):
        for axis in SCALING_RANGES:
            vals = make_scaling_x_values(axis, granularity=10)
            assert len(vals) == 10

    def test_make_scaling_x_values_endpoints(self):
        vals = make_scaling_x_values("n_total", 10)
        assert vals[0] == SCALING_RANGES["n_total"]["min"]
        assert vals[-1] == SCALING_RANGES["n_total"]["max"]

    def test_make_scaling_x_values_linear_spacing(self):
        vals = make_scaling_x_values("p", 5)
        # Differences between consecutive values should be approximately equal
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        assert max(diffs) - min(diffs) <= 2  # allow rounding tolerance

    def test_make_scaling_x_values_all_ints(self):
        for axis in SCALING_RANGES:
            vals = make_scaling_x_values(axis)
            assert all(isinstance(v, int) for v in vals)

    def test_make_scaling_x_values_invalid_axis(self):
        with pytest.raises(ValueError, match="axis_name must be one of"):
            make_scaling_x_values("invalid_axis")

    def test_make_scaling_x_values_default_granularity(self):
        vals = make_scaling_x_values("n_total")
        assert len(vals) == GRANULARITY


class TestMetricConstants:
    def test_carve_metrics_all_is_superset(self):
        all_set = set(CARVE_METRICS_ALL)
        assert set(CARVE_METRICS_STABILITY).issubset(all_set)
        assert set(CARVE_METRICS_GENERALIZABILITY).issubset(all_set)

    def test_no_overlap_stability_generalizability(self):
        overlap = set(CARVE_METRICS_STABILITY) & set(CARVE_METRICS_GENERALIZABILITY)
        assert len(overlap) == 0

    def test_external_metrics_tuple(self):
        assert isinstance(EXTERNAL_METRICS, tuple)
        assert len(EXTERNAL_METRICS) == 4

    def test_display_names_cover_externals(self):
        for m in EXTERNAL_METRICS:
            assert m in METRIC_DISPLAY_NAMES

    def test_plot_metrics_have_colors(self):
        for m in PLOT_METRICS:
            assert m in METRIC_COLOR, f"missing color for {m}"
            assert m in METRIC_LABEL, f"missing label for {m}"


# ── benchmarking_utils ────────────────────────────────────────────────────────

from benchmarking_utils import (  # noqa: E402
    _wilson_ci,
    _pick_first,
    _summary_stats,
    get_rule,
    get_measure,
    align_labels,
)


class TestWilsonCI:
    def test_zero_n_returns_nan(self):
        lo, hi = _wilson_ci(0, 0)
        assert np.isnan(lo) and np.isnan(hi)

    def test_all_successes(self):
        lo, hi = _wilson_ci(100, 100)
        assert lo > 0.9
        assert hi <= 1.0

    def test_half_successes(self):
        lo, hi = _wilson_ci(50, 100)
        assert lo < 0.5 < hi

    def test_bounds_in_unit_interval(self):
        lo, hi = _wilson_ci(30, 100)
        assert 0.0 <= lo <= hi <= 1.0


class TestPickFirst:
    def test_list_input(self):
        assert _pick_first([1, 2, 3]) == 1

    def test_tuple_input(self):
        assert _pick_first((42, 99)) == 42

    def test_scalar_input(self):
        assert _pick_first(7) == 7

    def test_string_input(self):
        assert _pick_first("hello") == "hello"


class TestGetRuleAndMeasure:
    @pytest.mark.parametrize(
        "metric,expected_rule",
        [
            ("ari_stability_1se", "1se"),
            ("ari_generalizability_quant", "quantile"),
            ("ari_stability", "max"),
            ("consensus_pac_stability", "max"),
        ],
    )
    def test_get_rule(self, metric, expected_rule):
        assert get_rule(metric) == expected_rule

    @pytest.mark.parametrize(
        "metric,expected_measure",
        [
            ("ari_stability_1se", "ari_stability"),
            ("ari_generalizability_quant", "ari_generalizability"),
            ("ari_stability", "ari_stability"),
            ("accuracy_generalizability", "accuracy_generalizability"),
        ],
    )
    def test_get_measure(self, metric, expected_measure):
        assert get_measure(metric) == expected_measure


class TestAlignLabels:
    def test_matching_labels_unchanged(self):
        true = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 1, 1, 2, 2])
        aligned = align_labels(true, pred)
        np.testing.assert_array_equal(aligned, true)

    def test_permuted_labels_realigned(self):
        true = np.array([0, 0, 1, 1, 2, 2])
        # Permutation: 0->2, 1->0, 2->1
        pred = np.array([2, 2, 0, 0, 1, 1])
        aligned = align_labels(true, pred)
        np.testing.assert_array_equal(aligned, true)


class TestSummaryStats:
    def test_basic_output(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _summary_stats(s)
        assert "mean" in result
        assert "sd" in result
        assert "median" in result
        assert abs(result["mean"] - 3.0) < 1e-10

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        result = _summary_stats(s)
        assert np.isnan(result["mean"])


# ── benchmarking_simulation_helpers ───────────────────────────────────────────

from benchmarking_simulation_helpers import _interpolate_settings  # noqa: E402


class TestInterpolateSettings:
    @pytest.fixture
    def anchors(self):
        return {
            "a": {"x": [1.0], "y": [10.0]},
            "b": {"x": [2.0], "y": [20.0]},
            "c": {"x": [3.0], "y": [30.0]},
        }

    def test_returns_first_anchor_at_stage_0(self, anchors):
        result = _interpolate_settings(anchors, ("a", "b", "c"), 0, 10)
        assert result == {"x": [1.0], "y": [10.0]}

    def test_returns_middle_anchor_at_midpoint(self, anchors):
        result = _interpolate_settings(anchors, ("a", "b", "c"), 5, 10)
        assert result == {"x": [2.0], "y": [20.0]}

    def test_returns_last_anchor_at_final_stage(self, anchors):
        result = _interpolate_settings(anchors, ("a", "b", "c"), 9, 10)
        assert result == {"x": [3.0], "y": [30.0]}

    def test_interpolation_between_first_and_middle(self, anchors):
        # Stage 2 out of 10: midpoint is 5, so frac = 2/5 = 0.4
        result = _interpolate_settings(anchors, ("a", "b", "c"), 2, 10)
        expected_x = (1 - 0.4) * 1.0 + 0.4 * 2.0  # 1.4
        assert abs(result["x"][0] - expected_x) < 1e-10

    def test_interpolation_between_middle_and_last(self, anchors):
        # Stage 7 out of 10: midpoint is 5, frac = (7-5)/(9-5) = 0.5
        result = _interpolate_settings(anchors, ("a", "b", "c"), 7, 10)
        expected_x = (1 - 0.5) * 2.0 + 0.5 * 3.0  # 2.5
        assert abs(result["x"][0] - expected_x) < 1e-10

    def test_invalid_total_stages_raises(self, anchors):
        with pytest.raises(ValueError, match="total_stages must be >= 3"):
            _interpolate_settings(anchors, ("a", "b", "c"), 0, 2)

    def test_out_of_range_stage_raises(self, anchors):
        with pytest.raises(ValueError, match="stage must be in"):
            _interpolate_settings(anchors, ("a", "b", "c"), 10, 10)


# ── benchmarking_metrics ─────────────────────────────────────────────────────

from benchmarking_metrics import calculate_metric  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402


class TestCalculateMetric:
    @pytest.fixture
    def well_separated_data(self):
        """3 well-separated clusters in 2D."""
        rng = np.random.default_rng(42)
        X = np.vstack(
            [
                rng.normal(loc=[0, 0], scale=0.3, size=(30, 2)),
                rng.normal(loc=[5, 0], scale=0.3, size=(30, 2)),
                rng.normal(loc=[0, 5], scale=0.3, size=(30, 2)),
            ]
        )
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        return X, labels

    def test_silhouette_positive(self, well_separated_data):
        X, labels = well_separated_data
        val = calculate_metric(X, labels, "silhouette", estimator_cls=KMeans)
        assert val > 0.5

    def test_gap_positive(self, well_separated_data):
        X, labels = well_separated_data
        val = calculate_metric(
            X, labels, "gap", estimator_cls=KMeans, random_state=42
        )
        assert val > 0

    def test_davies_bouldin_in_unit(self, well_separated_data):
        X, labels = well_separated_data
        val = calculate_metric(X, labels, "davies_bouldin", estimator_cls=KMeans)
        assert 0 < val <= 1.0

    def test_calinski_harabasz_positive(self, well_separated_data):
        X, labels = well_separated_data
        val = calculate_metric(X, labels, "calinski_harabasz", estimator_cls=KMeans)
        assert val > 0

    def test_unknown_metric_raises(self, well_separated_data):
        X, labels = well_separated_data
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_metric(X, labels, "nonexistent", estimator_cls=KMeans)


# ── benchmarking_summary_functions ────────────────────────────────────────────

from benchmarking_summary_functions import _coerce_bool  # noqa: E402


class TestCoerceBool:
    def test_bool_passthrough(self):
        s = pd.Series([True, False, True])
        result = _coerce_bool(s)
        assert result.dtype == bool
        assert list(result) == [True, False, True]

    def test_int_coercion(self):
        s = pd.Series([1, 0, 1, 0])
        result = _coerce_bool(s)
        assert list(result) == [True, False, True, False]

    def test_string_coercion(self):
        s = pd.Series(["True", "false", "1", "0", "yes", "no"])
        result = _coerce_bool(s)
        assert list(result) == [True, False, True, False, True, False]


# ── case_study_plotting: ARI comparison helpers ─────────────────────────────

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt  # noqa: E402

from case_study_plotting import (  # noqa: E402
    build_baseline_best_labels,
    extract_ari_comparison,
    plot_ari_comparison_lollipop,
    plot_ari_comparison_bar,
    plot_ari_comparison_dotplot,
)
from sklearn.cluster import AgglomerativeClustering  # noqa: E402


# --- fixtures ---


@pytest.fixture
def well_separated_2d():
    """3 well-separated clusters in 2D with ground-truth labels."""
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=[0, 0], scale=0.3, size=(30, 2)),
            rng.normal(loc=[5, 0], scale=0.3, size=(30, 2)),
            rng.normal(loc=[0, 5], scale=0.3, size=(30, 2)),
        ]
    )
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    return X, y


@pytest.fixture
def simple_model_grids():
    return [
        (KMeans, {"n_clusters": [2, 3, 4]}),
    ]


@pytest.fixture
def simple_best_df():
    return pd.DataFrame(
        [
            {
                "metric": "silhouette",
                "best_model": "KMeans",
                "best_k": 3,
                "best_score": 0.85,
                "best_ari": 0.90,
            },
            {
                "metric": "gap",
                "best_model": "KMeans",
                "best_k": 4,
                "best_score": 0.60,
                "best_ari": 0.70,
            },
        ]
    )


@pytest.fixture
def sample_ari_df():
    return pd.DataFrame(
        [
            {"method": "Silhouette", "model": "KMeans", "k": 3, "ari": 0.90, "source": "baseline"},
            {"method": "Gap Statistic", "model": "KMeans", "k": 4, "ari": 0.70, "source": "baseline"},
            {"method": "CARVE Stability (1se)", "model": "CARVE consensus", "k": 3, "ari": 0.95, "source": "carve"},
            {"method": "CARVE Generalizability (1se)", "model": "CARVE consensus", "k": 3, "ari": 0.92, "source": "carve"},
        ]
    )


# --- mock CARVE object for testing extract_ari_comparison ---


class _MockCARVE:
    """Minimal mock that mimics carve_obj.get_k() and get_labels()."""

    def __init__(self, X, y, k=3):
        self._k = k
        # just return KMeans labels as "consensus" for testing
        self._labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(X)

    def get_k(self, *, measure="stability", rule="1se", not_two=False):
        return self._k

    def get_labels(self, *, measure="stability", rule="1se", k=None, not_two=False, mode="default", estimator=None):
        return self._labels


# --- TestBuildBaselineBestLabels ---


class TestBuildBaselineBestLabels:
    def test_returns_correct_shape(self, well_separated_2d, simple_best_df, simple_model_grids):
        X, y = well_separated_2d
        labels, model_name, k = build_baseline_best_labels(
            X, simple_best_df, simple_model_grids, metric="silhouette", random_state=42
        )
        assert labels.shape == (X.shape[0],)

    def test_returns_matching_k(self, well_separated_2d, simple_best_df, simple_model_grids):
        X, y = well_separated_2d
        labels, model_name, k = build_baseline_best_labels(
            X, simple_best_df, simple_model_grids, metric="silhouette", random_state=42
        )
        assert k == 3
        assert model_name == "KMeans"

    def test_labels_are_valid_ids(self, well_separated_2d, simple_best_df, simple_model_grids):
        X, y = well_separated_2d
        labels, _, k = build_baseline_best_labels(
            X, simple_best_df, simple_model_grids, metric="gap", random_state=42
        )
        assert all(0 <= lab < k for lab in labels)

    def test_unknown_metric_raises(self, well_separated_2d, simple_best_df, simple_model_grids):
        X, y = well_separated_2d
        with pytest.raises(KeyError, match="nonexistent"):
            build_baseline_best_labels(
                X, simple_best_df, simple_model_grids, metric="nonexistent"
            )


# --- TestExtractAriComparison ---


class TestExtractAriComparison:
    def test_returns_expected_columns(self, well_separated_2d, simple_best_df):
        X, y = well_separated_2d
        mock_carve = _MockCARVE(X, y, k=3)
        df = extract_ari_comparison(y, simple_best_df, mock_carve, X)
        assert set(df.columns) == {"method", "model", "k", "ari", "source"}

    def test_baseline_aris_match_best_df(self, well_separated_2d, simple_best_df):
        X, y = well_separated_2d
        mock_carve = _MockCARVE(X, y, k=3)
        df = extract_ari_comparison(y, simple_best_df, mock_carve, X)
        baselines = df[df["source"] == "baseline"]
        assert len(baselines) == 2
        assert baselines.iloc[0]["ari"] == pytest.approx(0.90)
        assert baselines.iloc[1]["ari"] == pytest.approx(0.70)

    def test_carve_aris_computed_from_consensus_labels(self, well_separated_2d, simple_best_df):
        X, y = well_separated_2d
        mock_carve = _MockCARVE(X, y, k=3)
        df = extract_ari_comparison(y, simple_best_df, mock_carve, X)
        carve_rows = df[df["source"] == "carve"]
        assert len(carve_rows) == 2  # stability + generalizability
        for _, row in carve_rows.iterrows():
            assert 0.0 <= row["ari"] <= 1.0
            assert np.isfinite(row["ari"])

    def test_empty_carve_measures_returns_baselines_only(self, well_separated_2d, simple_best_df):
        X, y = well_separated_2d
        mock_carve = _MockCARVE(X, y, k=3)
        df = extract_ari_comparison(y, simple_best_df, mock_carve, X, carve_measures=[])
        assert len(df) == 2
        assert all(df["source"] == "baseline")

    def test_empty_best_df_returns_carve_only(self, well_separated_2d):
        X, y = well_separated_2d
        empty_best = pd.DataFrame(columns=["metric", "best_model", "best_k", "best_score", "best_ari"])
        mock_carve = _MockCARVE(X, y, k=3)
        df = extract_ari_comparison(y, empty_best, mock_carve, X)
        assert len(df) == 2
        assert all(df["source"] == "carve")


# --- TestPlotAriComparisonLollipop ---


class TestPlotAriComparisonLollipop:
    def test_returns_figure(self, sample_ari_df):
        fig = plot_ari_comparison_lollipop(sample_ari_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_number_of_stems(self, sample_ari_df):
        fig = plot_ari_comparison_lollipop(sample_ari_df)
        ax = fig.axes[0]
        # The scatter call creates a PathCollection with exactly N offsets
        from matplotlib.collections import PathCollection
        path_colls = [c for c in ax.collections if isinstance(c, PathCollection)]
        n_dots = sum(len(c.get_offsets()) for c in path_colls)
        assert n_dots == len(sample_ari_df)
        plt.close(fig)

    def test_annotate_k_adds_text(self, sample_ari_df):
        fig = plot_ari_comparison_lollipop(sample_ari_df, annotate_k=True)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("k=" in t for t in texts)
        plt.close(fig)

    def test_xlim_bounded(self, sample_ari_df):
        fig = plot_ari_comparison_lollipop(sample_ari_df)
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        assert xlim[0] == 0
        assert xlim[1] >= 1.0
        plt.close(fig)


# --- TestPlotAriComparisonBar ---


class TestPlotAriComparisonBar:
    def test_returns_figure(self, sample_ari_df):
        fig = plot_ari_comparison_bar(sample_ari_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_number_of_bars(self, sample_ari_df):
        fig = plot_ari_comparison_bar(sample_ari_df)
        ax = fig.axes[0]
        bars = ax.patches
        assert len(bars) == len(sample_ari_df)
        plt.close(fig)

    def test_annotate_k_adds_text(self, sample_ari_df):
        fig = plot_ari_comparison_bar(sample_ari_df, annotate_k=True)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("k=" in t for t in texts)
        plt.close(fig)


# --- TestPlotAriComparisonDotplot ---


class TestPlotAriComparisonDotplot:
    def test_returns_figure(self, sample_ari_df):
        fig = plot_ari_comparison_dotplot(sample_ari_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_number_of_dots(self, sample_ari_df):
        fig = plot_ari_comparison_dotplot(sample_ari_df)
        ax = fig.axes[0]
        scatter_collections = [c for c in ax.collections if hasattr(c, "get_offsets")]
        n_dots = sum(len(c.get_offsets()) for c in scatter_collections)
        assert n_dots == len(sample_ari_df)
        plt.close(fig)

    def test_ari_values_annotated(self, sample_ari_df):
        fig = plot_ari_comparison_dotplot(sample_ari_df, annotate_k=True)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Should have ARI values like "0.900  (k=3)"
        assert any("k=" in t for t in texts)
        assert any("0.9" in t for t in texts)
        plt.close(fig)
