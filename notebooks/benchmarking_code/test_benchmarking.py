"""Tests for benchmarking utility modules.

Run from the ``code/`` directory:
    python -m pytest notebooks/benchmarking_code/test_benchmarking.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt

# Ensure the benchmarking_code directory is on sys.path
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import case_study_plotting as csp  # noqa: E402


# ── benchmarking_config ───────────────────────────────────────────────────────

from benchmarking_config import (  # noqa: E402
    SCALING_RANGES,
    GRANULARITY,
    make_scaling_x_values,
    CARVE_METRICS_ALL,
    CARVE_METRICS_STABILITY,
    CARVE_METRICS_GENERALIZABILITY,
    NON_CARVE_METRICS,
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
        assert isinstance(NON_CARVE_METRICS, tuple)
        assert len(NON_CARVE_METRICS) == 4

    def test_display_names_cover_externals(self):
        for m in NON_CARVE_METRICS:
            assert m in METRIC_DISPLAY_NAMES

    def test_plot_metrics_have_colors(self):
        for m in PLOT_METRICS:
            assert m in METRIC_COLOR, f"missing color for {m}"
            assert m in METRIC_LABEL, f"missing label for {m}"


# ── case_study_plotting ──────────────────────────────────────────────────────


class TestCaseStudyColorPalettes:
    def test_get_color_mapping_uses_discrete_palette(self):
        colors = csp._get_color_mapping(3, palette_name="tab20")
        expected = list(plt.get_cmap("tab20").colors)[:3]
        np.testing.assert_allclose(np.asarray(colors), np.asarray(expected))

    def test_get_color_mapping_falls_back_to_glasbey_on_overflow(self, monkeypatch):
        calls = {}

        def fake_create_palette(*, palette_size):
            calls["palette_size"] = palette_size
            return [f"fallback-{i}" for i in range(palette_size)]

        monkeypatch.setattr(csp.glasbey, "create_palette", fake_create_palette)

        colors = csp._get_color_mapping(25, palette_name="Set1")

        assert calls["palette_size"] == 25
        assert colors == [f"fallback-{i}" for i in range(25)]

    def test_plot_dim_red_uses_palette_helper_for_categorical_labels(self, monkeypatch):
        calls = {}

        def fake_get_color_mapping(k, palette_name=None):
            calls["args"] = (k, palette_name)
            return ["#111111", "#222222"][:k]

        monkeypatch.setattr(csp, "_get_color_mapping", fake_get_color_mapping)

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        y = np.array(["a", "b"])
        fig, ax = plt.subplots()
        csp.plot_dim_red(
            X,
            y=y,
            palette="tab20",
            ax=ax,
            show=False,
            show_legend=False,
            hide_axes=False,
        )

        assert calls["args"] == (2, "tab20")


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

from benchmarking_simulation_helpers import (  # noqa: E402
    DIFFICULTY_LABELS,
    STAGE_LABELS,
    parse_difficulty_and_simulate,
    parse_range_and_simulate,
)


class TestAnchorConstants:
    def test_difficulty_labels(self):
        assert DIFFICULTY_LABELS == ("easy", "medium", "hard")

    def test_stage_labels(self):
        assert STAGE_LABELS == ("start", "middle", "end")


class TestParseDifficultyAndSimulate:
    @pytest.fixture
    def anchors(self):
        return {
            "easy": {"cluster_scale": [4.0] * 5, "cluster_size_dirichlet_alpha": 0.9},
            "medium": {"cluster_scale": [4.5] * 5, "cluster_size_dirichlet_alpha": 0.5},
            "hard": {"cluster_scale": [4.6] * 5, "cluster_size_dirichlet_alpha": 0.1},
        }

    def test_invalid_difficulty_raises(self, anchors):
        with pytest.raises(ValueError, match="difficulty must be one of"):
            parse_difficulty_and_simulate(
                anchor_settings=anchors,
                other_settings={"n_total": 100, "p": 5},
                difficulty="impossible",
                true_cluster_count=5,
                seed=0,
            )


class TestParseRangeAndSimulate:
    @pytest.fixture
    def anchors(self):
        return {
            "start": {"cluster_scale": [3.0] * 5},
            "middle": {"cluster_scale": [4.0] * 5},
            "end": {"cluster_scale": [5.0] * 5},
        }

    def test_invalid_axis_name_raises(self, anchors):
        with pytest.raises(ValueError, match="axis_name must be"):
            parse_range_and_simulate(
                anchor_settings=anchors,
                other_settings={},
                stage_label="start",
                true_cluster_count=5,
                axis_name="bogus",
                axis_value=100,
            )

    def test_invalid_stage_label_raises(self, anchors):
        with pytest.raises(ValueError, match="stage_label must be"):
            parse_range_and_simulate(
                anchor_settings=anchors,
                other_settings={},
                stage_label="impossible",
                true_cluster_count=5,
                axis_name="n_total",
                axis_value=100,
            )


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
        val = calculate_metric(X, labels, "gap", estimator_cls=KMeans, random_state=42)
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
