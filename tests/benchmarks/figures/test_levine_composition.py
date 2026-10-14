"""Tests for the Levine composition figure and its tables."""

import numpy as np
import pandas as pd
import pytest

from benchmarks._estimators import param_grids
from benchmarks._types import EstimatorSpec
from benchmarks.datasets._levine import _POPULATION_NAMES
from benchmarks.figures._case_study import (
    CompositeInputs,
    _align_to_reference,
    carve_labels_aligned,
)
from benchmarks.figures._levine_composition import (
    POPULATION_ORDER,
    carve_cluster_numbers,
    cluster_composition,
    composition_tables,
    marker_medians,
    marker_order,
    marker_table,
)
from carve import CARVE
from tests.benchmarks._helpers import StubCarve

# The 5,000-cell subsample's population sizes, in POPULATION_ORDER (the
# notebook's population table).
SIZES = [1265, 965, 187, 108, 25, 294, 793, 16, 1013, 59, 58, 44, 158, 15]

# Levine_32dim's 32 type markers, as load_levine32 returns them in
# meta["markers"] (read from HDCytoData 1.28.0 on 2026-10-14).
MARKERS = [
    "CD45RA", "CD133", "CD19", "CD22", "CD11b", "CD4", "CD8", "CD34",
    "Flt3", "CD20", "CXCR4", "CD235ab", "CD45", "CD123", "CD321", "CD14",
    "CD33", "CD47", "CD11c", "CD7", "CD15", "CD16", "CD44", "CD38",
    "CD13", "CD3", "CD61", "CD117", "CD49d", "HLA-DR", "CD64", "CD41",
]


def _inputs(*, X, y, carve, carve_labels, comparison_labels):
    """CompositeInputs with the fields these tests never read left minimal."""
    best = pd.DataFrame(
        {"metric": ["silhouette"], "model": ["KMeans (n_init=100)"], "k": [7]}
    )
    return CompositeInputs(
        X=np.asarray(X),
        y=np.asarray(y),
        Z=np.zeros((len(y), 2)),
        carve=carve,
        carve_labels=np.asarray(carve_labels),
        comparison_labels=np.asarray(comparison_labels),
        comparison_name="Silhouette",
        comparison_k=int(len(np.unique(comparison_labels))),
        curves_df=best,
        best_df=best,
    )


def _stub(own_labels):
    """A StubCarve whose get_labels returns own_labels, selecting KMeans."""
    k = int(len(np.unique(own_labels)))
    results = pd.DataFrame(
        {"method_id": ["m0"], "method_label": ["KMeans, n_init=100"], "n_clusters": [k]}
    )
    return StubCarve(
        results, select=lambda measure, not_two: ("m0", k), labels=np.asarray(own_labels)
    )


def _levine_like_inputs():
    """Inputs at Levine's sizes: 14 populations, 10 CARVE and 7 CVI clusters.

    CARVE's own ids are a permutation of the aligned ones, as on Levine,
    where k=10 differs from the 14 populations. The CVI partition pools CD4 T,
    CD8 T and CD16- NK cells into one cluster. A plain function, so a script
    can draw the figure from it too.
    """
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(14), SIZES)
    y = np.array(POPULATION_ORDER, dtype=object)[codes]
    X = rng.normal(size=(len(y), 32)) + 3.0 * (codes[:, None] % 5 == np.arange(32) % 5)
    carve_of = np.array([0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9])
    comparison_of = np.array([0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6])
    own = (carve_of[codes] * 3) % 10
    return _inputs(
        X=X,
        y=y,
        carve=_stub(own),
        carve_labels=_align_to_reference(own, y),
        comparison_labels=_align_to_reference(comparison_of[codes], y),
    )


@pytest.fixture
def levine_like():
    return _levine_like_inputs()


@pytest.fixture
def fitted():
    """A small real fit whose selected k differs from the reported label count.

    Three blobs carry four reported labels, so inside carve_labels_aligned
    CARVE keeps its own ids, as it does on Levine (k=10 against 14
    populations), while the aligned ids follow the reported codes.
    """
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=6.0, scale=0.4, size=(40, 3)),
        rng.normal(loc=(0.0, 6.0, 0.0), scale=0.4, size=(40, 3)),
    ])
    y = np.array(["d"] * 40 + ["c"] * 40 + ["b"] * 35 + ["a"] * 5)
    carve = CARVE(
        estimator_param_grids=param_grids(EstimatorSpec(name="kmeans"), (2, 3)),
        n_resamples=3,
        random_state=0,
    )
    carve.fit(X, reference_labels=y)
    aligned = _align_to_reference(carve.get_labels(measure="stability", rule="1se"), y)
    return _inputs(X=X, y=y, carve=carve, carve_labels=aligned, comparison_labels=aligned)


class TestPopulationOrder:
    def test_lists_every_gated_population_the_loader_names_once(self):
        assert len(POPULATION_ORDER) == 14
        assert set(POPULATION_ORDER) == set(_POPULATION_NAMES.values())


class TestClusterComposition:
    def test_a_merged_cluster_has_one_count_per_population_it_holds(self):
        y = ["CD4_T_cells"] * 5 + ["CD8_T_cells"] * 3 + ["Monocytes"] * 4
        counts = cluster_composition([0] * 8 + [1] * 4, y, order=POPULATION_ORDER)
        assert counts.loc[0, "CD4_T_cells"] == 5
        assert counts.loc[0, "CD8_T_cells"] == 3
        assert (counts.loc[0] > 0).sum() == 2
        assert counts.loc[1, "Monocytes"] == 4
        assert (counts.loc[1] > 0).sum() == 1

    def test_every_cell_is_counted_once_in_its_own_cluster(self):
        rng = np.random.default_rng(0)
        y = rng.choice(np.array(POPULATION_ORDER), size=300)
        labels = rng.integers(0, 6, size=300)
        counts = cluster_composition(labels, y, order=POPULATION_ORDER)
        ids, sizes = np.unique(labels, return_counts=True)
        assert counts.to_numpy().sum() == 300
        assert counts.sum(axis=1).loc[ids].tolist() == sizes.tolist()
        fractions = counts.div(counts.sum(axis=1), axis=0)
        np.testing.assert_allclose(fractions.sum(axis=1), 1.0)

    def test_columns_are_every_population_in_lineage_order(self):
        counts = cluster_composition([0, 0], ["Monocytes", "pDCs"], order=POPULATION_ORDER)
        assert list(counts.columns) == list(POPULATION_ORDER)
        assert counts.loc[0, "CD4_T_cells"] == 0

    def test_rows_follow_the_populations_their_cells_come_from(self):
        # Cluster 0 holds monocytes (lineage position 8), cluster 1 CD4 T
        # cells (position 0), so cluster 1 comes first.
        y = ["Monocytes"] * 4 + ["CD4_T_cells"] * 4
        counts = cluster_composition([0] * 4 + [1] * 4, y, order=POPULATION_ORDER)
        assert list(counts.index) == [1, 0]

    def test_a_population_the_order_does_not_list_raises(self):
        with pytest.raises(ValueError, match="unassigned"):
            cluster_composition(
                [0, 1], ["CD4_T_cells", "unassigned"], order=POPULATION_ORDER
            )


class TestCarveClusterNumbers:
    def test_each_cluster_gets_the_number_s8_fig_prints(self, fitted):
        with carve_labels_aligned(fitted.carve, fitted.y):
            own = np.asarray(fitted.carve.get_labels(measure="stability", rule="1se"))
        aligned = fitted.carve_labels
        # Only meaningful while the two labelings differ.
        assert set(np.unique(aligned).tolist()) != set(np.unique(own).tolist())
        numbers = carve_cluster_numbers(fitted)
        assert sorted(numbers) == sorted(np.unique(aligned).tolist())
        for aligned_id, number in numbers.items():
            assert set((own[aligned == aligned_id] + 1).tolist()) == {number}

    def test_a_different_partition_raises(self):
        y = np.array(["a", "a", "b", "b"])
        aligned = np.array([0, 0, 1, 1])
        inputs = _inputs(
            X=np.zeros((4, 2)),
            y=y,
            carve=_stub(np.array([0, 1, 1, 1])),
            carve_labels=aligned,
            comparison_labels=aligned,
        )
        with pytest.raises(ValueError, match="same partition"):
            carve_cluster_numbers(inputs)


class TestCompositionTables:
    def test_carve_rows_carry_the_numbers_s8_fig_prints(self, levine_like):
        carve, _ = composition_tables(levine_like)
        counts = cluster_composition(
            levine_like.carve_labels, levine_like.y, order=POPULATION_ORDER
        )
        own = levine_like.carve.get_labels()
        expected = []
        for cluster in counts.index:
            (own_id,) = np.unique(own[levine_like.carve_labels == cluster])
            expected.append(f"C{own_id + 1}")
        assert list(carve.index) == expected
        np.testing.assert_array_equal(carve.to_numpy(), counts.to_numpy())

    def test_comparison_rows_are_numbered_in_display_order(self, levine_like):
        _, comparison = composition_tables(levine_like)
        assert list(comparison.index) == [f"C{i}" for i in range(1, 8)]

    def test_the_comparison_pools_the_three_lymphocyte_populations(self, levine_like):
        _, comparison = composition_tables(levine_like)
        pooled = comparison.loc[
            :, ["CD4_T_cells", "CD8_T_cells", "CD16-_NK_cells"]
        ].gt(0).all(axis=1)
        assert pooled.sum() == 1


class TestMarkerMedians:
    def test_a_marker_high_in_one_group_is_one_there_and_zero_elsewhere(self):
        groups = np.repeat(["a", "b", "c"], 30)
        rng = np.random.default_rng(0)
        X = np.column_stack([np.where(groups == "a", 10.0, 0.0), rng.normal(size=90)])
        medians = marker_medians(
            X, groups, markers=["high_in_a", "noise"], group_order=["a", "b", "c"]
        )
        assert medians.loc["a", "high_in_a"] == 1.0
        assert medians.loc["b", "high_in_a"] == 0.0
        assert medians.loc["c", "high_in_a"] == 0.0

    def test_a_positive_affine_map_per_marker_changes_nothing(self):
        # The loader's RobustScaler is such a map of the arcsinh values, so
        # this is what makes scaling X equal to scaling the arcsinh input.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 4))
        groups = rng.integers(0, 3, size=200)
        mapped = X * np.array([0.5, 2.0, 10.0, 3.0]) + np.array([-1.0, 5.0, 0.0, 100.0])
        kwargs = {"markers": ["m0", "m1", "m2", "m3"], "group_order": [0, 1, 2]}
        pd.testing.assert_frame_equal(
            marker_medians(mapped, groups, **kwargs),
            marker_medians(X, groups, **kwargs),
            check_exact=False,
            atol=1e-12,
        )

    def test_a_constant_marker_is_zero_without_a_warning(self):
        X = np.column_stack([np.full(20, 3.0), np.arange(20.0)])
        medians = marker_medians(
            X, np.repeat([0, 1], 10), markers=["flat", "ramp"], group_order=[0, 1]
        )
        assert (medians["flat"] == 0.0).all()

    def test_rows_follow_group_order_and_columns_the_marker_names(self):
        X = np.arange(12.0).reshape(6, 2)
        medians = marker_medians(
            X, np.array([2, 2, 0, 0, 1, 1]), markers=["m0", "m1"], group_order=[1, 2, 0]
        )
        assert list(medians.index) == [1, 2, 0]
        assert list(medians.columns) == ["m0", "m1"]

    def test_group_order_must_list_exactly_the_groups_present(self):
        with pytest.raises(ValueError, match="group_order"):
            marker_medians(
                np.zeros((4, 1)), np.array([0, 0, 1, 1]), markers=["m"], group_order=[0]
            )

    def test_one_name_per_column(self):
        with pytest.raises(ValueError, match="2 columns but 1 markers"):
            marker_medians(
                np.zeros((4, 2)), np.array([0, 0, 1, 1]), markers=["m"], group_order=[0, 1]
            )


class TestMarkerOrder:
    def test_markers_with_the_same_profile_sit_next_to_each_other(self):
        rng = np.random.default_rng(2)
        values = rng.uniform(size=(14, 8))
        values[:, 5] = values[:, 0]
        values[:, 3] = values[:, 1]
        medians = pd.DataFrame(values, columns=[f"m{i}" for i in range(8)])
        order = marker_order(medians)
        assert sorted(order) == sorted(medians.columns)
        assert abs(order.index("m0") - order.index("m5")) == 1
        assert abs(order.index("m1") - order.index("m3")) == 1

    def test_the_order_depends_only_on_the_profiles(self):
        rng = np.random.default_rng(3)
        medians = pd.DataFrame(
            rng.uniform(size=(14, 32)), columns=[f"m{i}" for i in range(32)]
        )
        first = marker_order(medians)
        assert marker_order(medians) == first
        assert marker_order(medians.iloc[rng.permutation(14)]) == first


class TestMarkerTable:
    def test_blocks_are_stacked_with_rows_named_as_drawn(self, levine_like):
        table = marker_table(levine_like, MARKERS)
        carve, comparison = composition_tables(levine_like)
        blocks = list(table.index.get_level_values("block").unique())
        assert blocks == ["Gated", "CARVE", "Silhouette"]
        assert list(table.loc["Gated"].index) == [
            p.replace("_", " ") for p in POPULATION_ORDER
        ]
        assert list(table.loc["CARVE"].index) == list(carve.index)
        assert list(table.loc["Silhouette"].index) == list(comparison.index)

    def test_markers_are_in_the_gated_blocks_clustered_order(self, levine_like):
        table = marker_table(levine_like, MARKERS)
        gated = marker_medians(
            levine_like.X,
            levine_like.y,
            markers=MARKERS,
            group_order=list(POPULATION_ORDER),
        )
        assert list(table.columns) == marker_order(gated)
        assert list(table.columns) != MARKERS
