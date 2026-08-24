"""Tests for the sweep-axis abstraction (_sweep.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import HDBSCAN, KMeans

from carve._sweep import (
    IDENTITY_COLS,
    SWEEP_META_COLS,
    SWEEP_REGISTRY,
    MethodIds,
    coerce_sweep_values,
    config_id_of,
    format_method_label,
    grid_sweep_values,
    infer_sweep_param,
    observed_k,
    observed_k_series,
    resolve_sweep,
    sweep_axis_label,
    sweep_exclude_cols,
    sweep_param_name,
    validate_grids,
)


class TestValidateGrids:
    def _spec(self, param, values, **kw):
        return resolve_sweep(sweep=param, sweep_values=values, **kw)

    def test_fixed_registry_param_ok(self):
        spec = self._spec("min_samples", [3, 5, 10], finer_is_larger=False)
        grids = [
            (
                HDBSCAN,
                {
                    "min_samples": [3, 5, 10],
                    "min_cluster_size": [20],
                    "cluster_selection_method": ["eom"],
                },
            ),
        ]
        vals = validate_grids(grids, spec)
        np.testing.assert_array_equal(vals, [3, 5, 10])

    def test_multi_value_registry_param_clashes(self):
        spec = self._spec("min_samples", [3, 5, 10], finer_is_larger=False)
        grids = [
            (
                HDBSCAN,
                {
                    "min_samples": [3, 5, 10],
                    "min_cluster_size": [20, 50, 100],
                },
            ),
        ]
        with pytest.raises(ValueError, match="mixes sweep parameters"):
            validate_grids(grids, spec)

    def test_checks_all_grids(self):
        spec = self._spec("min_samples", [3, 5, 10], finer_is_larger=False)
        grids = [
            (
                HDBSCAN,
                {
                    "min_samples": [3, 5, 10],
                    "min_cluster_size": [20],
                },
            ),
            (
                HDBSCAN,
                {
                    "min_samples": [3, 5, 10],
                    "min_cluster_size": [20, 50, 100],
                },
            ),
        ]
        with pytest.raises(ValueError, match="mixes sweep parameters"):
            validate_grids(grids, spec)

    def test_missing_sweep_param_raises(self):
        spec = self._spec("min_samples", [3, 5, 10], finer_is_larger=False)
        grids = [
            (HDBSCAN, {"min_cluster_size": [20]}),
        ]
        with pytest.raises(ValueError, match="does not contain"):
            validate_grids(grids, spec)

    def test_mismatched_values_raises(self):
        spec = self._spec("n_clusters", [2, 3, 4])
        grids = [
            (KMeans, {"n_clusters": [2, 3, 4]}),
            (KMeans, {"n_clusters": [2, 3, 5]}),
        ]
        with pytest.raises(ValueError, match="same.*values"):
            validate_grids(grids, spec)


class TestInferSweepParam:
    def test_ignores_fixed_registry_param(self):
        grids = [
            (
                HDBSCAN,
                {
                    "min_samples": [3, 5, 10],
                    "min_cluster_size": [20],
                },
            ),
        ]
        assert infer_sweep_param(grids) is None

    def test_finds_multi_value_registry_param(self):
        grids = [
            (
                HDBSCAN,
                {
                    "min_cluster_size": [20, 50, 100],
                    "cluster_selection_method": ["eom"],
                },
            ),
        ]
        assert infer_sweep_param(grids) == "min_cluster_size"

    def test_multiple_swept_registry_params_raises(self):
        grids = [
            (
                KMeans,
                {
                    "n_clusters": [2, 3, 4],
                    "resolution": [0.5, 1.0],
                },
            ),
        ]
        with pytest.raises(ValueError, match="more than one parameter"):
            infer_sweep_param(grids)

    def test_no_registry_params_returns_none(self):
        grids = [
            (
                HDBSCAN,
                {
                    "min_samples": [3, 5, 10],
                    "cluster_selection_method": ["eom"],
                },
            ),
        ]
        assert infer_sweep_param(grids) is None


class TestResolveSweep:
    def test_defaults_to_k_mode(self):
        spec = resolve_sweep(n_clusters=np.array([2, 3, 4]))
        assert spec.param == "n_clusters"
        assert spec.is_k_mode
        assert spec.fixes_k
        assert spec.finer_is_larger
        assert spec.label == "Number of Clusters (k)"
        np.testing.assert_array_equal(spec.values, [2, 3, 4])

    def test_resolution_switches_mode(self):
        spec = resolve_sweep(
            n_clusters=np.array([2, 3]), resolution=np.array([0.5, 1.0])
        )
        assert spec.param == "resolution"
        assert not spec.is_k_mode
        assert not spec.fixes_k
        assert spec.finer_is_larger
        assert spec.label == "Resolution"
        np.testing.assert_allclose(spec.values, [0.5, 1.0])

    def test_min_cluster_size_is_inverted(self):
        spec = resolve_sweep(sweep="min_cluster_size", sweep_values=[5, 10, 25])
        assert not spec.finer_is_larger
        assert not spec.fixes_k
        assert spec.label == "Minimum Cluster Size"

    def test_values_are_sorted(self):
        spec = resolve_sweep(resolution=[2.0, 0.5, 1.0])
        np.testing.assert_allclose(spec.values, [0.5, 1.0, 2.0])

    def test_scalar_resolution_accepted(self):
        spec = resolve_sweep(resolution=0.8)
        np.testing.assert_allclose(spec.values, [0.8])

    def test_resolution_with_conflicting_sweep_raises(self):
        with pytest.raises(ValueError, match="resolution= was given but sweep="):
            resolve_sweep(resolution=[0.5], sweep="min_cluster_size")

    def test_resolution_with_matching_sweep_ok(self):
        spec = resolve_sweep(resolution=[0.5, 1.0], sweep="resolution")
        assert spec.param == "resolution"

    def test_no_values_raises(self):
        with pytest.raises(ValueError, match="No values supplied for sweep parameter"):
            resolve_sweep(sweep="min_cluster_size")

    def test_unknown_param_needs_finer_is_larger(self):
        with pytest.raises(ValueError, match="Pass finer_is_larger="):
            resolve_sweep(sweep="min_samples", sweep_values=[3, 5])

    def test_unknown_param_gets_titled_label(self):
        spec = resolve_sweep(
            sweep="min_samples", sweep_values=[3, 5], finer_is_larger=False
        )
        assert spec.param == "min_samples"
        assert spec.label == "Min Samples"
        assert not spec.finer_is_larger
        assert not spec.fixes_k

    def test_explicit_finer_is_larger_overrides_registry(self):
        spec = resolve_sweep(
            sweep="min_cluster_size", sweep_values=[5, 10], finer_is_larger=True
        )
        assert spec.finer_is_larger


class TestSweepSpecRanks:
    def test_ranks_ascending_for_n_clusters(self):
        spec = resolve_sweep(n_clusters=np.array([2, 3, 4, 5]))
        np.testing.assert_array_equal(spec.ranks(), [0, 1, 2, 3])

    def test_ranks_ascending_for_resolution(self):
        spec = resolve_sweep(resolution=[0.25, 0.5, 1.0, 2.0])
        np.testing.assert_array_equal(spec.ranks(), [0, 1, 2, 3])

    def test_ranks_descending_for_min_cluster_size(self):
        spec = resolve_sweep(sweep="min_cluster_size", sweep_values=[5, 10, 25])
        # Larger min_cluster_size yields FEWER clusters, so the largest
        # value is the coarsest and gets rank 0.
        np.testing.assert_array_equal(spec.ranks(), [2, 1, 0])

    def test_rank_of_round_trips(self):
        spec = resolve_sweep(resolution=[0.25, 0.5, 1.0, 2.0])
        assert [spec.rank_of(v) for v in spec.values] == [0, 1, 2, 3]

    def test_rank_of_inverted_axis(self):
        spec = resolve_sweep(sweep="min_cluster_size", sweep_values=[5, 10, 25])
        assert spec.rank_of(25) == 0
        assert spec.rank_of(5) == 2

    def test_rank_of_unswept_value_raises(self):
        spec = resolve_sweep(resolution=[0.5, 1.0])
        with pytest.raises(
            ValueError, match="is not among the swept resolution values"
        ):
            spec.rank_of(3.0)

    def test_is_k_mode(self):
        assert resolve_sweep(n_clusters=np.array([2, 3])).is_k_mode
        assert not resolve_sweep(resolution=[0.5]).is_k_mode


class TestCoerceSweepValues:
    def test_n_clusters_delegates(self):
        np.testing.assert_array_equal(coerce_sweep_values(4, "n_clusters"), [2, 3, 4])

    def test_resolution_sorted_floats(self):
        out = coerce_sweep_values([2.0, 0.5], "resolution")
        np.testing.assert_allclose(out, [0.5, 2.0])
        assert np.issubdtype(out.dtype, np.floating)

    def test_resolution_rejects_non_positive(self):
        with pytest.raises(ValueError, match="All resolution values must be > 0"):
            coerce_sweep_values([0.5, 0.0], "resolution")

        with pytest.raises(ValueError, match="All resolution values must be > 0"):
            coerce_sweep_values([-1.0], "resolution")

    def test_resolution_rejects_2d(self):
        with pytest.raises(ValueError, match="must be a scalar or a 1D array"):
            coerce_sweep_values(np.ones((2, 2)), "resolution")

    def test_min_cluster_size_rejects_floats(self):
        with pytest.raises(TypeError, match="min_cluster_size values must be integers"):
            coerce_sweep_values([5.5, 10.0], "min_cluster_size")

    def test_min_cluster_size_rejects_below_two(self):
        with pytest.raises(
            ValueError, match="All min_cluster_size values must be >= 2"
        ):
            coerce_sweep_values([1, 5], "min_cluster_size")

    def test_min_cluster_size_sorted_ints(self):
        out = coerce_sweep_values([25, 5, 10], "min_cluster_size")
        np.testing.assert_array_equal(out, [5, 10, 25])
        assert np.issubdtype(out.dtype, np.integer)


class TestGridSweepValues:
    def test_returns_first_grid_carrying_param(self):
        grids = [
            (KMeans, {"linkage": ["ward"]}),
            (KMeans, {"n_clusters": [2, 3, 4]}),
        ]
        assert grid_sweep_values(grids, "n_clusters") == [2, 3, 4]

    def test_returns_none_when_absent(self):
        grids = [(KMeans, {"linkage": ["ward"]})]
        assert grid_sweep_values(grids, "n_clusters") is None


class TestResultsTableHelpers:
    def test_sweep_param_name(self, resolution_results_df):
        assert sweep_param_name(resolution_results_df) == "resolution"

    def test_sweep_axis_label_known(self, resolution_results_df, results_df):
        assert sweep_axis_label(resolution_results_df) == "Resolution"
        assert sweep_axis_label(results_df) == "Number of Clusters (k)"

    def test_sweep_axis_label_unknown_param(self):
        df = pd.DataFrame({"sweep_param": ["min_samples"]})
        assert sweep_axis_label(df) == "Min Samples"

    def test_sweep_exclude_cols_covers_identity_and_meta(self, results_df):
        cols = sweep_exclude_cols(results_df)
        assert IDENTITY_COLS <= cols
        assert SWEEP_META_COLS <= cols
        # The swept hyperparameter's own column is excluded too.
        assert "n_clusters" in cols

    def test_sweep_exclude_cols_includes_sweep_param(self, resolution_results_df):
        assert "resolution" in sweep_exclude_cols(resolution_results_df)

    def test_observed_k_series_rounds(self):
        df = pd.DataFrame({"n_clusters_observed": [2.4, 3.5, 8.6]})
        np.testing.assert_allclose(observed_k_series(df), [2.0, 4.0, 9.0])

    def test_observed_k_returns_int(self, resolution_results_df):
        row = resolution_results_df.iloc[2]
        assert observed_k(row) == 5
        assert isinstance(observed_k(row), int)

    def test_registry_contents(self):
        assert set(SWEEP_REGISTRY) == {
            "n_clusters",
            "resolution",
            "min_cluster_size",
        }
        assert SWEEP_REGISTRY["n_clusters"][1] is True  # fixes_k
        assert SWEEP_REGISTRY["min_cluster_size"][0] is False  # finer_is_larger


class TestFormatMethodLabel:
    """Guards the exact strings that end up in figure legends."""

    def test_estimator_only(self):
        assert format_method_label("KMeans", {"n_clusters": 3}, "n_clusters") == (
            "KMeans"
        )

    def test_includes_non_swept_params_sorted(self):
        label = format_method_label(
            "AgglomerativeClustering",
            {"n_clusters": 3, "linkage": "ward"},
            "n_clusters",
        )
        assert label == "AgglomerativeClustering, linkage=ward"

    def test_params_sorted_alphabetically(self):
        label = format_method_label("Est", {"z": 1, "a": 2, "k": 3}, "k")
        assert label == "Est, a=2, z=1"

    def test_int_formatting(self):
        assert format_method_label("E", {"n_neighbors": 15}, "k") == (
            "E, n_neighbors=15"
        )

    def test_integral_float_drops_decimal(self):
        assert format_method_label("E", {"gamma": 2.0}, "k") == "E, gamma=2"

    def test_non_integral_float_uses_3g(self):
        assert format_method_label("E", {"gamma": 0.123456}, "k") == ("E, gamma=0.123")

    def test_bool_formatting(self):
        assert format_method_label("E", {"scale": True}, "k") == "E, scale=True"

    def test_string_formatting(self):
        assert format_method_label("E", {"affinity": "rbf"}, "k") == ("E, affinity=rbf")

    def test_numpy_scalars(self):
        label = format_method_label(
            "E", {"n_neighbors": np.int64(15), "gamma": np.float64(0.5)}, "k"
        )
        assert label == "E, gamma=0.5, n_neighbors=15"


class TestMethodIds:
    def test_same_id_across_sweep_values(self):
        ids = MethodIds("n_clusters")
        first, label = ids.assign("KMeans", {"n_clusters": 2})
        second, _ = ids.assign("KMeans", {"n_clusters": 3})
        assert first == second == "m0"
        assert label == "KMeans"

    def test_different_id_when_other_param_differs(self):
        ids = MethodIds("n_clusters")
        ward, ward_label = ids.assign(
            "AgglomerativeClustering", {"n_clusters": 2, "linkage": "ward"}
        )
        average, average_label = ids.assign(
            "AgglomerativeClustering", {"n_clusters": 2, "linkage": "average"}
        )
        assert ward != average
        assert ward_label != average_label

    def test_different_id_per_estimator(self):
        ids = MethodIds("n_clusters")
        assert (
            ids.assign("KMeans", {"n_clusters": 2})[0]
            != ids.assign("AgglomerativeClustering", {"n_clusters": 2})[0]
        )

    def test_first_seen_order(self):
        ids = MethodIds("n_clusters")
        seen = [
            ids.assign("KMeans", {"n_clusters": 2})[0],
            ids.assign("KMeans", {"n_clusters": 3})[0],
            ids.assign("Spectral", {"n_clusters": 2})[0],
            ids.assign("Spectral", {"n_clusters": 3})[0],
            ids.assign("Ward", {"n_clusters": 2})[0],
        ]
        assert seen == ["m0", "m0", "m1", "m1", "m2"]

    def test_repeat_calls_are_stable(self):
        ids = MethodIds("resolution")
        params = {"resolution": 0.5, "n_neighbors": 15}
        first = ids.assign("LeidenClustering", params)
        assert ids.assign("LeidenClustering", params) == first
        assert ids.assign("LeidenClustering", params) == first

    def test_tolerates_unhashable_param_value(self):
        """User grids may hold lists; the key uses repr(), not the value."""
        ids = MethodIds("n_clusters")
        a, _ = ids.assign("E", {"n_clusters": 2, "gammas": [0.1, 0.2]})
        b, _ = ids.assign("E", {"n_clusters": 3, "gammas": [0.1, 0.2]})
        c, _ = ids.assign("E", {"n_clusters": 2, "gammas": [0.3]})
        assert a == b == "m0"
        assert c == "m1"


class TestConfigIdOf:
    def test_reads_the_column(self, resolution_results_df):
        row = resolution_results_df.iloc[2]
        assert config_id_of(row) == 2
        assert isinstance(config_id_of(row), int)

    def test_survives_reordering(self, resolution_results_df):
        """config_id travels with the row, unlike the index label."""
        shuffled = resolution_results_df.sort_values("ari_stability").reset_index(
            drop=True
        )
        row = shuffled[shuffled["sweep_value"] == 1.0].iloc[0]
        assert config_id_of(row) == 2
